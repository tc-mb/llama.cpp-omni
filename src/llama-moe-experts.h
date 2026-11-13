#pragma once

#include "ggml.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

struct llama_moe_experts_entry {
    int32_t layer = -1;
    std::string tensor_name;
    std::string tensor_family;

    /// in-model 模式下承载这一层专家字节的张量名（`blk.N.moe_exps_packed.weight`）。
    /// manifest 模式下为空。
    std::string packed_tensor;
    /// 该 family 在**一个专家内**的偏移。绝对偏移 = packed 张量偏移 + 这个值。
    size_t family_offset = 0;

    std::string path;
    ggml_type type = GGML_TYPE_COUNT;
    size_t offset = 0;
    size_t tensor_bytes = 0;
    size_t bytes_per_expert = 0;
    size_t expert_stride = 0;
    int32_t expert_count = 0;
};

class llama_moe_experts {
public:
    /// 从 `experts/` 目录（或它的 manifest.json）构造：旧格式，一个张量一族一段文件。
    explicit llama_moe_experts(const char * input_path);

    /// 从 gguf 里的 `les.moe.descriptor` 构造：专家区内联在主模型文件里，
    /// 每层一个 I8 载体张量，多族字节按专家交错。
    ///
    /// 这里只解析「描述」，偏移还是未知的（要等 loader 把载体张量的位置解析出来）；
    /// 构造完成后必须对每个载体张量调一次 bind_packed()。
    static std::unique_ptr<llama_moe_experts> from_descriptor(
            const std::string & descriptor_json, const std::string & model_path);

    /// 告诉这些 entry：载体张量 `packed_tensor` 的数据在 `tensor_offs`、位于 `path`。
    /// 逐条把 entry.offset 定成 `tensor_offs + family_offset`。
    void bind_packed(const std::string & packed_tensor, size_t tensor_offs, const std::string & path);

    /// 这个模型用到的全部载体张量名（去重、按名字排序）。调用方据此把它们从
    /// weights_map 里取出来。
    std::vector<std::string> packed_tensor_names() const;

    const llama_moe_experts_entry * find(const char * tensor_name) const;

    int32_t expert_count() const;
    int32_t expert_used_count() const;
    int32_t layer_count() const;
    size_t bytes_per_slot_all_layers() const;

    /// 每一层的 bytes_per_slot（一个 slot 装一个专家的全部 family）。下标是 layer id。
    const std::vector<size_t> & bytes_per_slot_per_layer() const;

private:
    llama_moe_experts() = default;

    int32_t n_expert = 0;
    int32_t n_expert_used = 0;
    int32_t n_layer = 0;
    size_t bytes_per_slot = 0;
    std::vector<size_t> bytes_per_layer;
    std::unordered_map<std::string, llama_moe_experts_entry> entries;
};
