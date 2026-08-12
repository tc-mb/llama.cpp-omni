#include "set-rows-f32-f16-policy.h"

#include <cstdio>
#include <cstdlib>

namespace {

using fallback = ggml_cann_set_rows_f32_f16_fallback;
using input = ggml_cann_set_rows_f32_f16_policy_input;

input canonical(int64_t row_width) {
    input value;
    value.src_is_f32 = true;
    value.dst_is_f16 = true;
    value.index_is_i64 = true;
    value.src_row_dense = true;
    value.dst_rows_dense = true;
    value.index_dense = true;
    value.buffers_disjoint = true;
    value.src_ne[0] = row_width;
    value.src_ne[1] = 1;
    value.src_ne[2] = 1;
    value.src_ne[3] = 1;
    value.dst_ne[0] = row_width;
    value.dst_ne[1] = 4096;
    value.dst_ne[2] = 1;
    value.dst_ne[3] = 1;
    value.index_ne[0] = 1;
    value.index_ne[1] = 1;
    value.index_ne[2] = 1;
    value.index_ne[3] = 1;
    value.src_address = 0x1000;
    value.dst_address = 0x2000;
    value.index_address = 0x3000;
    return value;
}

void expect(const input & value, fallback wanted, const char * label) {
    const fallback got = ggml_cann_set_rows_f32_f16_validate_policy(value);
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
    value.src_is_f32 = false;
    expect(value, fallback::SRC_DTYPE, "src-dtype");

    value = canonical(768);
    value.dst_is_f16 = false;
    expect(value, fallback::DST_DTYPE, "dst-dtype");

    value = canonical(768);
    value.index_is_i64 = false;
    expect(value, fallback::INDEX_DTYPE, "index-dtype");

    value = canonical(768);
    value.src_row_dense = false;
    expect(value, fallback::SRC_LAYOUT, "src-layout");

    value = canonical(768);
    value.dst_rows_dense = false;
    expect(value, fallback::DST_LAYOUT, "dst-layout");

    value = canonical(768);
    value.index_dense = false;
    expect(value, fallback::INDEX_LAYOUT, "index-layout");

    value = canonical(768);
    value.dst_ne[2] = 2;
    expect(value, fallback::BATCH, "batch");

    value = canonical(768);
    value.src_ne[1] = 2;
    value.index_ne[0] = 2;
    expect(value, fallback::INDEX_COUNT, "multi-index");

    value = canonical(768);
    value.src_ne[0] = 512;
    value.dst_ne[0] = 512;
    expect(value, fallback::ROW_WIDTH, "row-width");

    value = canonical(768);
    value.dst_ne[1] = 2048;
    expect(value, fallback::CACHE_ROWS, "cache-rows");

    value = canonical(768);
    value.buffers_disjoint = false;
    expect(value, fallback::ALIAS, "alias");

    value = canonical(768);
    value.src_address += 4;
    expect(value, fallback::ALIGNMENT, "src-alignment");

    value = canonical(768);
    value.dst_address += 2;
    expect(value, fallback::ALIGNMENT, "dst-alignment");

    value = canonical(768);
    value.index_address += 4;
    expect(value, fallback::ALIGNMENT, "index-alignment");

    value = canonical(768);
    value.src_address = 0;
    expect(value, fallback::ALIGNMENT, "null-address");

    return 0;
}
