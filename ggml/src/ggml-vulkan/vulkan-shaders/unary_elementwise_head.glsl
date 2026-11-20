#include "generic_unary_head.glsl"

// These pipelines use param1/param2 as independent dense-layout flags.
uint unary_load_idx(uint i) {
    return get_aoffset() + (p.param1 != 0.0 ? i : src0_idx(i));
}

uint unary_store_idx(uint i) {
    return get_doffset() + (p.param2 != 0.0 ? i : dst_idx(i));
}
