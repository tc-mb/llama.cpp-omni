#include "modulate-fusion-policy.h"

#include <cstdio>
#include <cstdlib>

namespace {

using fallback = ggml_cann_modulate_fusion_fallback;
using input = ggml_cann_modulate_fusion_policy_input;

input canonical() {
    input value;
    value.fusible_chain = true;
    value.exact_edges = true;
    value.all_f32 = true;
    value.x_shape = true;
    value.param_shape = true;
    value.supported_layout = true;
    value.buffers_disjoint = true;
    value.buffers_aligned = true;
    return value;
}

void expect(const input & value, fallback wanted, const char * label) {
    const fallback got = ggml_cann_modulate_fusion_validate_policy(value);
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
    expect(canonical(), fallback::NONE, "canonical");

    input value = canonical();
    value.fusible_chain = false;
    expect(value, fallback::NOT_FUSIBLE_CHAIN, "extra-consumer-or-op");

    value = canonical();
    value.exact_edges = false;
    expect(value, fallback::EDGE_IDENTITY, "operand-order");

    value = canonical();
    value.all_f32 = false;
    expect(value, fallback::DTYPE, "dtype");

    value = canonical();
    value.x_shape = false;
    expect(value, fallback::X_SHAPE, "x-shape");

    value = canonical();
    value.param_shape = false;
    expect(value, fallback::PARAM_SHAPE, "param-shape");

    value = canonical();
    value.supported_layout = false;
    expect(value, fallback::LAYOUT, "layout");

    value = canonical();
    value.buffers_disjoint = false;
    expect(value, fallback::ALIAS, "alias");

    value = canonical();
    value.buffers_aligned = false;
    expect(value, fallback::ALIGNMENT, "alignment");

    if (ggml_cann_modulate_fusion_legacy_launches_per_hit != 3 ||
        ggml_cann_modulate_fusion_candidate_launches_per_hit != 1 ||
        ggml_cann_modulate_fusion_launches_removed_per_hit != 2) {
        std::fprintf(stderr, "unexpected launch accounting\n");
        return 1;
    }

    return 0;
}
