#include "llama-moe-experts.h"

#include "llama-impl.h"

#include "gguf.h"
#include "llama.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <unordered_set>
#include <vector>
namespace {

ggml_type parse_type(std::string name) {
    std::transform(name.begin(), name.end(), name.begin(), [](unsigned char value) {
        return (char) std::tolower(value);
    });

    for (int value = 0; value < GGML_TYPE_COUNT; ++value) {
        const auto type = (ggml_type) value;
        const char * type_name = ggml_type_name(type);
        if (!type_name) {
            continue;
        }
        std::string candidate(type_name);
        std::transform(candidate.begin(), candidate.end(), candidate.begin(), [](unsigned char item) {
            return (char) std::tolower(item);
        });
        if (candidate == name) {
            return type;
        }
    }

    throw std::runtime_error(format("unknown MoE experts quant type '%s'", name.c_str()));
}

bool routed_family(const std::string & family) {
    return family == "ffn_gate_exps" ||
           family == "ffn_up_exps" ||
           family == "ffn_down_exps" ||
           family == "ffn_gate_up_exps";
}

}

llama_moe_experts::llama_moe_experts(const char * input_path) {
    namespace fs = std::filesystem;

    if (!input_path || input_path[0] == '\0') {
        throw std::invalid_argument("MoE experts path is empty");
    }

    const fs::path input(input_path);
    const fs::path manifest_path = fs::is_directory(input) ? input / "manifest.json" : input;
    const fs::path manifest_dir = manifest_path.parent_path();

    std::ifstream stream(manifest_path);
    if (!stream.is_open()) {
        throw std::runtime_error(format("failed to open MoE experts manifest '%s'", manifest_path.string().c_str()));
    }

    nlohmann::json manifest;
    stream >> manifest;

    if (manifest.value("schema_version", 0) != 1) {
        throw std::runtime_error("unsupported MoE experts schema version");
    }
    // 兼容旧 manifest：早期写的是 `sidecar_kind`。数据格式改名不该让已生成的
    // 13 GiB 产物作废，所以这里两个 key 都收。
    const std::string kind = manifest.contains("experts_kind")
        ? manifest.value("experts_kind", std::string())
        : manifest.value("sidecar_kind", std::string());
    if (kind != "omni_moe_gguf" && kind != "flashmoe_gguf") {
        throw std::runtime_error(format("unsupported MoE experts kind '%s'", kind.c_str()));
    }
    const std::string layout = manifest.value("layout", std::string());
    if (layout != "layer_major_expert" && layout != "expert_major") {
        throw std::runtime_error(format("MoE experts layout '%s' is not expert-major", layout.c_str()));
    }

    const auto & model = manifest.at("model");
    n_expert = model.at("expert_count").get<int32_t>();
    n_expert_used = model.at("expert_used_count").get<int32_t>();
    if (n_expert <= 0 || n_expert_used <= 0 || n_expert_used > n_expert) {
        throw std::runtime_error("invalid expert counts in MoE experts manifest");
    }

    std::unordered_set<int32_t> layers;
    for (const auto & item : manifest.at("entries")) {
        const std::string family = item.at("tensor_family").get<std::string>();
        if (!routed_family(family)) {
            continue;
        }

        llama_moe_experts_entry entry;
        entry.layer = item.at("layer").get<int32_t>();
        entry.tensor_name = item.at("tensor_name").get<std::string>();
        entry.tensor_family = family;
        entry.path = (manifest_dir / item.at("repacked_file").get<std::string>()).string();
        entry.type = parse_type(item.at("quant_type").get<std::string>());
        entry.offset = item.at("repacked_offset").get<size_t>();
        entry.tensor_bytes = item.at("exact_byte_length").get<size_t>();
        entry.bytes_per_expert = item.at("bytes_per_expert").get<size_t>();
        entry.expert_stride = item.at("expert_stride").get<size_t>();
        entry.expert_count = item.value(
            "expert_count",
            entry.bytes_per_expert == 0 ? 0 : (int32_t) (entry.tensor_bytes / entry.bytes_per_expert));

        if (entry.layer < 0 || entry.bytes_per_expert == 0 || entry.expert_count != n_expert) {
            throw std::runtime_error(format("invalid MoE experts entry '%s'", entry.tensor_name.c_str()));
        }
        if (entry.offset > std::numeric_limits<size_t>::max() - entry.bytes_per_expert) {
            throw std::runtime_error(format("MoE experts entry '%s' overflows family extent", entry.tensor_name.c_str()));
        }
        const size_t family_end = entry.offset + entry.bytes_per_expert;
        if (entry.expert_stride < family_end) {
            throw std::runtime_error(format("invalid expert stride for MoE experts entry '%s'", entry.tensor_name.c_str()));
        }
        if ((size_t) (entry.expert_count - 1) >
            (std::numeric_limits<size_t>::max() - family_end) / entry.expert_stride) {
            throw std::runtime_error(format("MoE experts entry '%s' overflows file extent", entry.tensor_name.c_str()));
        }

        const size_t required = (size_t) (entry.expert_count - 1) * entry.expert_stride +
                                family_end;
        std::error_code error;
        const size_t file_size = (size_t) fs::file_size(entry.path, error);
        if (error || file_size < required) {
            throw std::runtime_error(format(
                "MoE experts file '%s' is too small for tensor '%s'",
                entry.path.c_str(), entry.tensor_name.c_str()));
        }

        bytes_per_slot += entry.bytes_per_expert;
        layers.insert(entry.layer);
        if (!entries.emplace(entry.tensor_name, std::move(entry)).second) {
            throw std::runtime_error("duplicate tensor in MoE experts manifest");
        }
    }

    if (entries.empty()) {
        throw std::runtime_error("MoE experts manifest has no routed expert tensors");
    }
    n_layer = (int32_t) layers.size();
}

// ---------------------------------------------------------------------------------------------
// in-model 模式：专家区在主 gguf 里，由 les.moe.descriptor 描述
// ---------------------------------------------------------------------------------------------

namespace {

constexpr int32_t LES_MOE_VERSION = 1;
constexpr const char * LES_MOE_LAYOUT = "expert_major_packed";
// 每段（gate/up/down）的对齐。K 量化的 block 是 32 元素，所以字节数天然是 32 的倍数；
// 引擎侧的 slot 读取（一条 preadv 读一个专家）依赖这个前提。
constexpr size_t LES_FAMILY_ALIGN = 32;

std::string layer_tensor_name(int32_t layer, const std::string & family) {
    return format("blk.%d.%s.weight", layer, family.c_str());
}

} // namespace

std::unique_ptr<llama_moe_experts> llama_moe_experts::from_descriptor(
        const std::string & descriptor_json, const std::string & model_path) {
    nlohmann::json descriptor;
    try {
        descriptor = nlohmann::json::parse(descriptor_json);
    } catch (const std::exception & e) {
        throw std::runtime_error(format("failed to parse les.moe.descriptor: %s", e.what()));
    }

    if (descriptor.value("les_version", 0) != LES_MOE_VERSION) {
        throw std::runtime_error(format(
            "unsupported les.moe.descriptor version %d", descriptor.value("les_version", 0)));
    }
    if (descriptor.value("layout", std::string()) != LES_MOE_LAYOUT) {
        throw std::runtime_error(format(
            "les.moe.descriptor layout '%s' is not '%s'",
            descriptor.value("layout", std::string()).c_str(), LES_MOE_LAYOUT));
    }
    if (!descriptor.contains("layers") || !descriptor.at("layers").is_array() ||
            descriptor.at("layers").empty()) {
        throw std::runtime_error("les.moe.descriptor has no layers");
    }

    auto out = std::unique_ptr<llama_moe_experts>(new llama_moe_experts());

    // 一个全局值做交叉校验，用来给 slot 记账提供单一几何；每层的声明是权威来源。
    int32_t n_expert = -1;
    int32_t n_expert_used = -1;
    std::unordered_set<int32_t> layers;

    std::vector<const nlohmann::json *> layer_items;
    for (const auto & item : descriptor.at("layers")) {
        layer_items.push_back(&item);
    }
    std::sort(layer_items.begin(), layer_items.end(),
              [](const nlohmann::json * a, const nlohmann::json * b) {
                  return a->at("layer").get<int32_t>() < b->at("layer").get<int32_t>();
              });

    for (const nlohmann::json * item : layer_items) {
        const int32_t layer = item->at("layer").get<int32_t>();
        const int32_t layer_experts = item->at("n_expert").get<int32_t>();
        const int32_t layer_used = item->value("expert_used", item->at("n_expert").get<int32_t>());

        if (layer < 0 || layer_experts <= 0 || layer_used <= 0 || layer_used > layer_experts) {
            throw std::runtime_error(format(
                "les.moe.descriptor: invalid expert counts for layer %d", layer));
        }
        if (!layers.insert(layer).second) {
            throw std::runtime_error(format(
                "les.moe.descriptor: duplicate layer %d", layer));
        }
        if (n_expert < 0) {
            n_expert = layer_experts;
            n_expert_used = layer_used;
        } else if (layer_experts != n_expert) {
            // 这个构建要求所有层用同一个专家数：计算图的 expert 轴、slot 记账和预读
            // 都是按单值分配的。层与层不一致时明确拒绝，而不是静默地算出错的结果。
            throw std::runtime_error(format(
                "les.moe.descriptor: layer %d has %d experts but layer 0 has %d; "
                "this build requires a uniform expert count across layers",
                layer, layer_experts, n_expert));
        }

        if (!item->contains("groups") || !item->at("groups").is_array() ||
                item->at("groups").empty()) {
            throw std::runtime_error(format("les.moe.descriptor: layer %d has no groups", layer));
        }
        if (item->at("groups").size() > 1) {
            // 一个 group 对应一族固定的 slot 张量几何。这个构建只支持每层一个 group。
            throw std::runtime_error(format(
                "les.moe.descriptor: layer %d has %zu groups; this build supports one group per layer",
                layer, item->at("groups").size()));
        }

        size_t layer_slot_bytes = 0;
        int32_t covered = 0;
        for (const auto & group : item->at("groups")) {
            const int32_t begin = group.at("expert_begin").get<int32_t>();
            const int32_t end   = group.at("expert_end").get<int32_t>();
            const size_t  unit  = group.at("unit_bytes").get<size_t>();
            const std::string packed = group.at("tensor").get<std::string>();

            if (begin != covered || end <= begin || end > layer_experts) {
                throw std::runtime_error(format(
                    "les.moe.descriptor: layer %d group [%d,%d) does not cover experts "
                    "contiguously (covered up to %d of %d)",
                    layer, begin, end, covered, layer_experts));
            }
            if (unit % LES_FAMILY_ALIGN) {
                throw std::runtime_error(format(
                    "les.moe.descriptor: layer %d unit_bytes %zu is not a multiple of %zu",
                    layer, unit, LES_FAMILY_ALIGN));
            }

            size_t sum = 0;
            for (const auto & family : group.at("families")) {
                const std::string name = family.at("name").get<std::string>();
                const size_t family_offset = family.at("offset").get<size_t>();
                const size_t family_bytes  = family.at("bytes").get<size_t>();

                if (family_offset % LES_FAMILY_ALIGN || family_bytes % LES_FAMILY_ALIGN) {
                    throw std::runtime_error(format(
                        "les.moe.descriptor: layer %d family '%s' is not %zu-byte aligned",
                        layer, name.c_str(), LES_FAMILY_ALIGN));
                }
                if (family_offset + family_bytes > unit) {
                    throw std::runtime_error(format(
                        "les.moe.descriptor: layer %d family '%s' runs past unit_bytes "
                        "(%zu + %zu > %zu)", layer, name.c_str(), family_offset, family_bytes, unit));
                }

                llama_moe_experts_entry entry;
                entry.layer          = layer;
                entry.tensor_family  = name;
                entry.tensor_name    = layer_tensor_name(layer, name);
                entry.packed_tensor  = packed;
                entry.family_offset  = family_offset;
                entry.path           = model_path;
                entry.type           = parse_type(family.at("type").get<std::string>());
                entry.offset         = family_offset; // bind_packed() 之后由载体偏移重定基
                entry.bytes_per_expert = family_bytes;
                entry.expert_stride  = unit;
                entry.expert_count   = layer_experts;
                entry.tensor_bytes   = family_bytes * (size_t) layer_experts;

                if (!out->entries.emplace(entry.tensor_name, std::move(entry)).second) {
                    throw std::runtime_error(format(
                        "les.moe.descriptor: duplicate tensor '%s'",
                        layer_tensor_name(layer, name).c_str()));
                }
                sum += family_bytes;
            }
            if (sum != unit) {
                throw std::runtime_error(format(
                    "les.moe.descriptor: layer %d families sum to %zu but unit_bytes is %zu",
                    layer, sum, unit));
            }
            layer_slot_bytes += unit;
            covered = end;
        }
        if (covered != layer_experts) {
            throw std::runtime_error(format(
                "les.moe.descriptor: layer %d groups cover %d of %d experts",
                layer, covered, layer_experts));
        }

        if ((int32_t) out->bytes_per_layer.size() <= layer) {
            out->bytes_per_layer.resize((size_t) layer + 1, 0);
        }
        out->bytes_per_layer[(size_t) layer] = layer_slot_bytes;
    }

    out->n_expert      = n_expert;
    out->n_expert_used = n_expert_used;
    out->n_layer       = (int32_t) layers.size();
    for (size_t bytes : out->bytes_per_layer) {
        out->bytes_per_slot += bytes;
    }

    if (out->entries.empty()) {
        throw std::runtime_error("les.moe.descriptor declares no expert tensors");
    }
    return out;
}

void llama_moe_experts::bind_packed(const std::string & packed_tensor, size_t tensor_offs,
                                    const std::string & path) {
    // 载体张量在文件里的位置一旦解析出来，就可以给引用它的每条 entry 定绝对偏移。
    // 同一层三个 family 共用一个载体，所以 base 相同、只有 family_offset 不同 ——
    // `finalize_pack` 要求的「同文件、同 stride、族在 stride 内」由此自动成立。
    int bound = 0;
    for (auto & kv : entries) {
        auto & entry = kv.second;
        if (entry.packed_tensor != packed_tensor) {
            continue;
        }
        entry.offset = tensor_offs + entry.family_offset;
        entry.path   = path;
        bound++;
    }
    if (bound == 0) {
        throw std::runtime_error(format(
            "les.moe.descriptor references packed tensor '%s' but no entry uses it",
            packed_tensor.c_str()));
    }
}

std::vector<std::string> llama_moe_experts::packed_tensor_names() const {    std::vector<std::string> names;
    names.reserve(entries.size());
    for (const auto & kv : entries) {
        if (kv.second.packed_tensor.empty()) {
            continue;
        }
        if (std::find(names.begin(), names.end(), kv.second.packed_tensor) == names.end()) {
            names.push_back(kv.second.packed_tensor);
        }
    }
    std::sort(names.begin(), names.end());
    return names;
}

const llama_moe_experts_entry * llama_moe_experts::find(const char * tensor_name) const {    const auto found = entries.find(tensor_name);
    return found == entries.end() ? nullptr : &found->second;
}

int32_t llama_moe_experts::expert_count() const {
    return n_expert;
}

int32_t llama_moe_experts::expert_used_count() const {
    return n_expert_used;
}

int32_t llama_moe_experts::layer_count() const {
    return n_layer;
}

size_t llama_moe_experts::bytes_per_slot_all_layers() const {
    return bytes_per_slot;
}

const std::vector<size_t> & llama_moe_experts::bytes_per_slot_per_layer() const {
    return bytes_per_layer;
}

// ---------------------------------------------------------------------------------------------
// 不加载模型的探测：只读 gguf header 里的 descriptor
// ---------------------------------------------------------------------------------------------

bool llama_les_probe(const char * gguf_path, llama_les_info * out) {
    if (!gguf_path || !out) {
        return false;
    }
    // 只读 metadata：no_alloc 让 gguf 不去碰张量数据，所以这个调用对 14 GiB 的文件
    // 也只是一次 header 读。
    gguf_init_params params = {
        /*.no_alloc =*/ true,
        /*.ctx      =*/ nullptr,
    };
    gguf_context * ctx = gguf_init_from_file(gguf_path, params);
    if (!ctx) {
        return false;
    }

    bool ok = false;
    const int key_desc = gguf_find_key(ctx, "les.moe.descriptor");
    if (key_desc >= 0) {
        try {
            const auto descriptor = nlohmann::json::parse(
                    gguf_get_val_str(ctx, key_desc));

            size_t bytes_per_slot = 0;
            int32_t n_layer = 0;
            int32_t n_expert = 0;
            int32_t expert_used = 0;

            for (const auto & layer : descriptor.at("layers")) {
                for (const auto & group : layer.at("groups")) {
                    bytes_per_slot += group.at("unit_bytes").get<size_t>();
                }
                if (n_layer == 0) {
                    n_expert    = layer.at("n_expert").get<int32_t>();
                    expert_used = layer.value("expert_used", n_expert);
                }
                n_layer++;
            }

            if (n_layer > 0 && bytes_per_slot > 0) {
                out->n_layer        = n_layer;
                out->n_expert       = n_expert;
                out->expert_used    = expert_used;
                out->bytes_per_slot = bytes_per_slot;
                ok = true;
            }
        } catch (const std::exception &) {
            ok = false;
        }
    }

    gguf_free(ctx);
    return ok;
}
