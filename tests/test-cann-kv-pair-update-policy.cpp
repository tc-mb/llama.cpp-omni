#include "kv-pair-update-policy.h"

#include <cstdio>
#include <cstdlib>

namespace {

using fallback = ggml_cann_kv_pair_update_fallback;
using input = ggml_cann_kv_pair_update_policy_input;
using single_fallback = ggml_cann_set_rows_f32_f16_fallback;

input canonical(int64_t width) {
    input value;
    value.adjacent_set_rows = true;
    value.first_reason = single_fallback::NONE;
    value.second_reason = single_fallback::NONE;
    value.first_row_width = width;
    value.second_row_width = width;
    value.first_cache_rows = 4096;
    value.second_cache_rows = 4096;
    value.pair_buffers_disjoint = true;
    return value;
}

void expect(const input & value, fallback wanted, const char * label) {
    const fallback got = ggml_cann_kv_pair_update_validate_policy(value);
    if (got != wanted) {
        std::fprintf(stderr,
                     "%s: got fallback %u, wanted %u\n",
                     label,
                     static_cast<unsigned>(got),
                     static_cast<unsigned>(wanted));
        std::exit(1);
    }
}

}  // namespace

int main() {
    expect(canonical(768), fallback::NONE, "canonical-768");
    expect(canonical(1024), fallback::NONE, "canonical-1024");

    input value = canonical(768);
    value.adjacent_set_rows = false;
    expect(value, fallback::NOT_ADJACENT_SET_ROWS, "not-adjacent");

    value = canonical(768);
    value.first_reason = single_fallback::INDEX_COUNT;
    expect(value, fallback::FIRST_POLICY, "first-multi-index");

    value = canonical(768);
    value.second_reason = single_fallback::DST_LAYOUT;
    expect(value, fallback::SECOND_POLICY, "second-layout");

    value = canonical(768);
    value.second_row_width = 1024;
    expect(value, fallback::ROW_WIDTH, "width-mismatch");

    value = canonical(768);
    value.second_cache_rows = 2048;
    expect(value, fallback::CACHE_ROWS, "cache-mismatch");

    value = canonical(768);
    value.pair_buffers_disjoint = false;
    expect(value, fallback::PAIR_ALIAS, "cross-alias");

    if (ggml_cann_kv_pair_update_legacy_launches_per_pair != 4 ||
        ggml_cann_kv_pair_update_candidate_launches_per_pair != 1 ||
        ggml_cann_kv_pair_update_launches_removed_per_hit != 3) {
        std::fprintf(stderr, "unexpected launch accounting\n");
        return 1;
    }

    return 0;
}
