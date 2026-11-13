#pragma once

#include "ggml.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

// One routed tensor family of one layer (e.g. `blk.7.ffn_gate_exps.weight`).
struct llama_moe_experts_entry {
    int32_t layer = -1;
    std::string tensor_name;
    std::string tensor_family;

    /// Name of the tensor that carries this layer's expert bytes
    /// (`blk.N.moe_exps_packed.weight`).
    std::string packed_tensor_name;
    /// Offset of this family inside one expert. The absolute file offset is
    /// `packed_tensor_offset + offset_in_expert`.
    size_t offset_in_expert = 0;

    std::string path;
    ggml_type type = GGML_TYPE_COUNT;
    size_t offset = 0;
    size_t tensor_bytes = 0;
    size_t bytes_per_expert = 0;
    size_t expert_stride = 0;
    int32_t expert_count = 0;
};

// Routed expert weights for a MoE model.
//
// The layout always comes from the model file's own `les.moe.layout` metadata; there is
// no second source to reconcile. The absolute offsets are unknown until the loader has
// resolved where the carrier tensors sit, so `bind_region()` has to be called for each.
class llama_moe_experts {
public:
    static std::unique_ptr<llama_moe_experts> from_layout(
            const std::string & layout_json, const std::string & path_model);

    /// Resolve the absolute offsets of every entry carried by `packed_tensor_name`,
    /// now that the loader knows where that tensor lives.
    void bind_region(const std::string & packed_tensor_name,
                     size_t tensor_offset,
                     const std::string & path);

    /// Carrier tensor names, deduplicated and sorted.
    std::vector<std::string> region_names() const;

    const llama_moe_experts_entry * find(const char * tensor_name) const;

    int32_t expert_count() const;
    int32_t expert_used_count() const;
    int32_t layer_count() const;

    /// Bytes of one expert slot, summed over all routed layers.
    size_t bytes_per_slot_all_layers() const;

    /// Bytes of one expert slot, indexed by layer id (0 for non-routed layers).
    const std::vector<size_t> & bytes_per_slot_per_layer() const;

private:
    llama_moe_experts() = default;

    int32_t n_expert = 0;
    int32_t n_expert_used = 0;
    int32_t n_layer = 0;
    size_t bytes_per_slot = 0;
    /// Bytes of one expert slot per layer id. Distinct from `bytes_per_slot`, which is
    /// the sum over layers; a caller budgeting a per-layer cache needs this one.
    std::vector<size_t> bytes_per_slot_layer;
    std::unordered_map<std::string, llama_moe_experts_entry> entries;
};