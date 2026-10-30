// Include the implementation to exercise its file-local packing helper and
// actual input graph builder without adding a library interface for the test.
#include "../vision.cpp"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static void check(bool condition, const char * message) {
    if (!condition) {
        std::fprintf(stderr, "%s\n", message);
        std::exit(1);
    }
}

static void test_projection(ggml_backend_t backend, ggml_type type, int ps, int nx, int ny,
                            int outputs, bool with_bias) {
    ggml_init_params params = { 4 * 1024 * 1024, nullptr, true };
    ggml_context * ctx = ggml_init(params);
    check(ctx != nullptr, "context allocation failed");
    ggml_tensor * weights = ggml_new_tensor_4d(ctx, type, ps, ps, 3, outputs);
    ggml_tensor * bias = with_bias ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, outputs) : nullptr;
    check(vision_patch_mm_supported(nx, ny, ps, 1, weights), "valid shape rejected");
    check(!vision_patch_mm_supported(nx, ny, ps, 2, weights), "batched input must fall back");
    check(!vision_patch_mm_supported(nx + 1, ny, ps, 1, weights), "partial patch must fall back");
    check(!vision_patch_mm_supported(nx, ny + 1, ps, 1, weights), "partial height must fall back");
    check(!vision_patch_mm_supported(nx, ny, 0, 1, weights), "zero patch must fall back");
    check(!vision_patch_mm_supported(0, ny, ps, 1, weights), "empty image must fall back");

    ggml_tensor * wrong_channels = ggml_new_tensor_4d(ctx, type, ps, ps, 1, outputs);
    check(!vision_patch_mm_supported(nx, ny, ps, 1, wrong_channels), "non-RGB kernel must fall back");
    ggml_tensor * wrong_kernel = ggml_new_tensor_4d(ctx, type, ps + 1, ps, 3, outputs);
    check(!vision_patch_mm_supported(nx, ny, ps, 1, wrong_kernel), "kernel mismatch must fall back");

    check(!vision_patch_mm_supported(nx, ny, ps, 1, nullptr), "missing weights must fall back");
    ggml_tensor * noncontiguous = ggml_transpose(ctx, weights);
    check(!vision_patch_mm_supported(nx, ny, ps, 1, noncontiguous), "strided weights must fall back");
    ggml_tensor * wrong_type = ggml_new_tensor_4d(ctx, GGML_TYPE_BF16, ps, ps, 3, outputs);
    check(!vision_patch_mm_supported(nx, ny, ps, 1, wrong_type), "unsupported weights must fall back");

    vision_context_params vision_params = {};
    vision_ctx vision(vision_params);
    vision.model.hparams = {};
    vision.model.hparams.patch_size = ps;
    vision.model.hparams.n_embd = outputs;
    vision.model.hparams.n_head = 1;
    vision.model.patch_embeddings = weights;
    vision.model.patch_bias = bias;
    vision.buf_compute_meta.resize(4 * 1024 * 1024);
    vision_image_f32 image = { nx, ny, {} };
    vision_graph builder(&vision, image);
    check(builder.patch_mm, "opt-in packed projection not selected");

    builder.patch_mm = false;
    ggml_tensor * reference = builder.build_inp();
    ggml_tensor * input = ggml_get_tensor(builder.ctx0, "inp_raw");
    check(input != nullptr, "convolution graph input missing");
    ggml_set_name(input, "inp_conv");
    builder.patch_mm = true;
    ggml_tensor * actual = builder.build_inp();
    check(actual->ne[0] == outputs && actual->ne[1] == (nx / ps) * (ny / ps),
          "unexpected projection output layout");
    ggml_cgraph * graph = builder.gf;
    ggml_build_forward_expand(graph, reference);
    ggml_build_forward_expand(graph, actual);
    ggml_tensor * packed_input = ggml_graph_get_tensor(graph, "inp_raw");
    check(packed_input != nullptr, "packed graph input missing");
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    check(buffer != nullptr, "weight allocation failed");
    ggml_backend_buffer_t graph_buffer = ggml_backend_alloc_ctx_tensors(builder.ctx0, backend);
    check(graph_buffer != nullptr, "graph allocation failed");
    const int patches = (nx / ps) * (ny / ps);

    std::vector<float> rgb((size_t) nx * ny * 3);
    std::vector<float> chw(rgb.size());
    for (size_t i = 0; i < rgb.size(); ++i) {
        rgb[i] = (float) ((int) ((i * 17 + i / 13) % 101) - 50) / 61.0f;
    }
    for (int c = 0; c < 3; ++c) {
        for (int p = 0; p < nx * ny; ++p) {
            chw[(size_t) c * nx * ny + p] = rgb[3 * p + c];
        }
    }
    const std::vector<float> packed = vision_pack_patches(rgb.data(), nx, ny, ps);
    ggml_backend_tensor_set(input, chw.data(), 0, ggml_nbytes(input));
    ggml_backend_tensor_set(packed_input, packed.data(), 0, ggml_nbytes(packed_input));
    std::vector<float> w((size_t) ps * ps * 3 * outputs);
    for (size_t i = 0; i < w.size(); ++i) {
        w[i] = (float) ((int) ((i * 7 + i / 5) % 23) - 11) / 127.0f;
    }
    if (type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> half(w.size());
        for (size_t i = 0; i < w.size(); ++i) {
            half[i] = ggml_fp32_to_fp16(w[i]);
        }
        ggml_backend_tensor_set(weights, half.data(), 0, ggml_nbytes(weights));
    } else {
        ggml_backend_tensor_set(weights, w.data(), 0, ggml_nbytes(weights));
    }
    if (bias) {
        std::vector<float> b(outputs);
        for (int i = 0; i < outputs; ++i) {
            b[i] = (float) (i - outputs / 2) / 32.0f;
        }
        ggml_backend_tensor_set(bias, b.data(), 0, ggml_nbytes(bias));
    }
    check(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "graph compute failed");
    std::vector<float> expected((size_t) patches * outputs), result(expected.size());
    ggml_backend_tensor_get(reference, expected.data(), 0, ggml_nbytes(reference));
    ggml_backend_tensor_get(actual, result.data(), 0, ggml_nbytes(actual));
    float max_error = 0.0f;
    for (size_t i = 0; i < expected.size(); ++i) {
        check(std::isfinite(expected[i]) && std::isfinite(result[i]), "non-finite projection output");
        max_error = std::max(max_error, std::fabs(expected[i] - result[i]));
        check(std::fabs(expected[i] - result[i]) <= 1e-4f * (1.0f + std::fabs(expected[i])),
              "patch matmul differs from conv2d projection");
    }
    std::printf("%s patch=%d image=%dx%d channels=%d bias=%d max_abs_error=%g\n",
                ggml_type_name(type), ps, nx, ny, outputs, with_bias, max_error);
    ggml_backend_buffer_free(graph_buffer);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
}

int main(int argc, char ** argv) {
#ifdef _WIN32
    check(_putenv_s("OMNI_VPM_PATCHMM", "1") == 0, "could not enable packed projection");
#else
    check(setenv("OMNI_VPM_PATCHMM", "1", 1) == 0, "could not enable packed projection");
#endif
    ggml_backend_load_all();
    const char * name = argc > 1 ? argv[1] : "CPU";
    ggml_backend_t backend = ggml_backend_init_by_name(name, nullptr);
    check(backend != nullptr, "requested backend unavailable");
    if (ggml_backend_is_cpu(backend)) {
        ggml_backend_cpu_set_n_threads(backend, 2);
    }
    for (ggml_type type : { GGML_TYPE_F32, GGML_TYPE_F16 }) {
        for (bool bias : { false, true }) {
            test_projection(backend, type, 2, 6, 4, 5, bias);
            test_projection(backend, type, 14, 28, 42, 8, bias);
            test_projection(backend, type, 16, 32, 16, 16, bias);
        }
    }
    ggml_backend_free(backend);
    return 0;
}
