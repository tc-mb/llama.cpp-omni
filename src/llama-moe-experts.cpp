#include "llama-moe-experts.h"

#include "llama-impl.h"

#include "gguf.h"
#include "llama.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
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

// ---------------------------------------------------------------------------------------------
// Source: the layout declared by the model file itself
// ---------------------------------------------------------------------------------------------

constexpr const char * LES_LAYOUT_KIND   = "expert_major_packed";
// 一个 family 在一个专家内的对齐。K 量化的 block 是 32 元素，所以字节数天然是 32 的
// 倍数；引擎侧的 slot 读取（一条 preadv 读一个专家）依赖这个前提。
constexpr size_t      LES_TENSOR_ALIGN   = 32;

std::string layer_tensor_name(int32_t layer, const std::string & tensor_family) {
    return format("blk.%d.%s.weight", layer, tensor_family.c_str());
}

} // namespace

std::unique_ptr<llama_moe_experts> llama_moe_experts::from_layout(
        const std::string & layout_json, const std::string & path_model) {
    nlohmann::json layout;
    try {
        layout = nlohmann::json::parse(layout_json);
    } catch (const std::exception & e) {
        throw std::runtime_error(format("failed to parse the expert layout: %s", e.what()));
    }

    if (layout.value("layout", std::string()) != LES_LAYOUT_KIND) {
        throw std::runtime_error(format(
            "expert layout '%s' is not '%s'",
            layout.value("layout", std::string()).c_str(), LES_LAYOUT_KIND));
    }
    if (!layout.contains("layers") || !layout.at("layers").is_array() || layout.at("layers").empty()) {
        throw std::runtime_error("expert layout declares no layers");
    }

    auto out = std::unique_ptr<llama_moe_experts>(new llama_moe_experts());

    // 每一层的声明都是权威；这里只保留一个全局副本，用来给 slot 记账提供单一几何。
    int32_t n_expert      = -1;
    int32_t n_expert_used = -1;
    std::unordered_set<int32_t> layers;

    std::vector<const nlohmann::json *> layer_items;
    layer_items.reserve(layout.at("layers").size());
    for (const auto & item : layout.at("layers")) {
        layer_items.push_back(&item);
    }
    std::sort(layer_items.begin(), layer_items.end(),
              [](const nlohmann::json * a, const nlohmann::json * b) {
                  return a->at("layer").get<int32_t>() < b->at("layer").get<int32_t>();
              });

    for (const nlohmann::json * item : layer_items) {
        const int32_t layer       = item->at("layer").get<int32_t>();
        const int32_t layer_experts = item->at("n_expert").get<int32_t>();
        const int32_t layer_used    = item->value("n_expert_used", layer_experts);

        if (layer < 0 || layer_experts <= 0 || layer_used <= 0 || layer_used > layer_experts) {
            throw std::runtime_error(format(
                "expert layout: invalid expert counts for layer %d", layer));
        }
        if (!layers.insert(layer).second) {
            throw std::runtime_error(format("expert layout: duplicate layer %d", layer));
        }
        if (n_expert < 0) {
            n_expert      = layer_experts;
            n_expert_used = layer_used;
        } else if (layer_experts != n_expert) {
            // 这个构建要求所有层用同一个专家数：计算图的 expert 轴、slot 记账和预读
            // 都是按单值分配的。不一致时明确拒绝，而不是静默地算出错的结果。
            throw std::runtime_error(format(
                "expert layout: layer %d has %d experts but layer 0 has %d; "
                "this build requires a uniform expert count across layers",
                layer, layer_experts, n_expert));
        }

        if (!item->contains("groups") || !item->at("groups").is_array() ||
                item->at("groups").empty()) {
            throw std::runtime_error(format("expert layout: layer %d has no groups", layer));
        }
        if (item->at("groups").size() > 1) {
            // 一个 group 对应一族固定的 slot 张量几何。这个构建只支持每层一个 group。
            throw std::runtime_error(format(
                "expert layout: layer %d has %zu groups; this build supports one group per layer",
                layer, item->at("groups").size()));
        }

        size_t  layer_slot_bytes = 0;
        int32_t covered          = 0;
        for (const auto & group : item->at("groups")) {
            const int32_t     expert_begin = group.at("expert_begin").get<int32_t>();
            const int32_t     group_count  = group.at("expert_count").get<int32_t>();
            const size_t      expert_stride = group.at("expert_stride").get<size_t>();
            const std::string tensor_name   = group.at("tensor_name").get<std::string>();

            if (expert_begin != covered || group_count <= 0 ||
                    expert_begin + group_count > layer_experts) {
                throw std::runtime_error(format(
                    "expert layout: layer %d group [%d,+%d) does not cover experts contiguously "
                    "(covered up to %d of %d)",
                    layer, expert_begin, group_count, covered, layer_experts));
            }
            if (expert_stride % LES_TENSOR_ALIGN) {
                throw std::runtime_error(format(
                    "expert layout: layer %d expert_stride %zu is not a multiple of %zu",
                    layer, expert_stride, LES_TENSOR_ALIGN));
            }

            size_t sum = 0;
            for (const auto & tensor : group.at("tensors")) {
                const std::string tensor_family    = tensor.at("tensor_family").get<std::string>();
                const size_t      offset_in_expert = tensor.at("offset_in_expert").get<size_t>();
                const size_t      bytes_per_expert = tensor.at("bytes_per_expert").get<size_t>();

                if (offset_in_expert % LES_TENSOR_ALIGN || bytes_per_expert % LES_TENSOR_ALIGN) {
                    throw std::runtime_error(format(
                        "expert layout: layer %d tensor '%s' is not %zu-byte aligned",
                        layer, tensor_family.c_str(), LES_TENSOR_ALIGN));
                }
                if (offset_in_expert + bytes_per_expert > expert_stride) {
                    throw std::runtime_error(format(
                        "expert layout: layer %d tensor '%s' runs past expert_stride "
                        "(%zu + %zu > %zu)",
                        layer, tensor_family.c_str(), offset_in_expert, bytes_per_expert,
                        expert_stride));
                }

                llama_moe_experts_entry entry;
                entry.layer               = layer;
                entry.tensor_family       = tensor_family;
                entry.tensor_name         = layer_tensor_name(layer, tensor_family);
                entry.packed_tensor_name  = tensor_name;
                entry.offset_in_expert    = offset_in_expert;
                entry.path                = path_model;
                entry.type                = parse_type(tensor.at("type").get<std::string>());
                entry.offset              = offset_in_expert; // bind_region() 之后重定基
                entry.bytes_per_expert    = bytes_per_expert;
                entry.expert_stride       = expert_stride;
                entry.expert_count        = layer_experts;
                entry.tensor_bytes        = bytes_per_expert * (size_t) layer_experts;

                if (!out->entries.emplace(entry.tensor_name, std::move(entry)).second) {
                    throw std::runtime_error(format(
                        "expert layout: duplicate tensor '%s'",
                        layer_tensor_name(layer, tensor_family).c_str()));
                }
                sum += bytes_per_expert;
            }
            if (sum != expert_stride) {
                throw std::runtime_error(format(
                    "expert layout: layer %d tensors sum to %zu but expert_stride is %zu",
                    layer, sum, expert_stride));
            }
            layer_slot_bytes += expert_stride;
            covered = expert_begin + group_count;
        }
        if (covered != layer_experts) {
            throw std::runtime_error(format(
                "expert layout: layer %d groups cover %d of %d experts",
                layer, covered, layer_experts));
        }

        if ((int32_t) out->bytes_per_slot_layer.size() <= layer) {
            out->bytes_per_slot_layer.resize((size_t) layer + 1, 0);
        }
        out->bytes_per_slot_layer[(size_t) layer] = layer_slot_bytes;
    }

    out->n_expert      = n_expert;
    out->n_expert_used = n_expert_used;
    out->n_layer       = (int32_t) layers.size();
    for (size_t bytes : out->bytes_per_slot_layer) {
        out->bytes_per_slot += bytes;
    }

    if (out->entries.empty()) {
        throw std::runtime_error("expert layout declares no expert tensors");
    }
    return out;
}

void llama_moe_experts::bind_region(const std::string & packed_tensor_name,
                                    size_t tensor_offset,
                                    const std::string & path) {
    // 载体张量在文件里的位置一旦解析出来，就可以给引用它的每条 entry 定绝对偏移。
    // 同一层三个 family 共用一个载体，所以 base 相同、只有 offset_in_expert 不同 ——
    // `finalize_pack` 要求的「同文件、同 stride、族在 stride 内」由此自动成立。
    int bound = 0;
    for (auto & kv : entries) {
        auto & entry = kv.second;
        if (entry.packed_tensor_name != packed_tensor_name) {
            continue;
        }
        entry.offset = tensor_offset + entry.offset_in_expert;
        entry.path   = path;
        bound++;
    }
    if (bound == 0) {
        throw std::runtime_error(format(
            "expert layout references tensor '%s' but no entry uses it",
            packed_tensor_name.c_str()));
    }
}

std::vector<std::string> llama_moe_experts::region_names() const {
    std::vector<std::string> names;
    names.reserve(entries.size());
    for (const auto & kv : entries) {
        const auto & name = kv.second.packed_tensor_name;
        if (name.empty()) {
            continue;
        }
        if (std::find(names.begin(), names.end(), name) == names.end()) {
            names.push_back(name);
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
    return bytes_per_slot_layer;
}

// ---------------------------------------------------------------------------------------------
// Reading the geometry without loading the model
// ---------------------------------------------------------------------------------------------

bool llama_moe_experts_geometry_from_file(const char * path_model,
                                          struct llama_moe_experts_geometry * out) {
    if (!path_model || !out) {
        return false;
    }
    // 只要 metadata：no_alloc 让 gguf 不去碰张量数据，所以对十几 GiB 的文件也只是
    // 一次 header 读。
    gguf_init_params params = {
        /*.no_alloc =*/ true,
        /*.ctx      =*/ nullptr,
    };
    gguf_context * ctx = gguf_init_from_file(path_model, params);
    if (!ctx) {
        return false;
    }

    bool ok = false;
    const int key_layout = gguf_find_key(ctx, "les.moe.layout");
    if (key_layout >= 0) {
        try {
            const auto layout = nlohmann::json::parse(gguf_get_val_str(ctx, key_layout));

            size_t  bytes_per_slot = 0;
            int32_t n_layer        = 0;
            int32_t n_expert       = 0;
            int32_t n_expert_used  = 0;

            for (const auto & layer : layout.at("layers")) {
                for (const auto & group : layer.at("groups")) {
                    bytes_per_slot += group.at("expert_stride").get<size_t>();
                }
                if (n_layer == 0) {
                    n_expert      = layer.at("n_expert").get<int32_t>();
                    n_expert_used = layer.value("n_expert_used", n_expert);
                }
                n_layer++;
            }

            if (n_layer > 0 && bytes_per_slot > 0) {
                out->n_layer        = n_layer;
                out->n_expert       = n_expert;
                out->n_expert_used  = n_expert_used;
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
