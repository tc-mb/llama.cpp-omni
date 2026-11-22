#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-cpp.h>
#include <ggml-cpu.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

// Context allocation alone never exercises gallocr's automatic source reuse.
static bool test_alias(ggml_backend_t backend, bool rms, int width, int alias, int threads) {
    const char * name = rms ? "RMS_NORM_BACK" : "SOFT_MAX_BACK";
    const float eps = width == 7 ? 1e-6f : 0.25f;
    const float scale = width == 7 ? 0.5f : 2.0f;
    const int rows = 6;
    ggml_init_params params = {16*ggml_tensor_overhead() + ggml_graph_overhead_custom(8, false), nullptr, true};
    ggml_context_ptr ctx(ggml_init(params));
    GGML_ASSERT(ctx);
    ggml_tensor * a = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, width, 3, 2, 1);
    ggml_tensor * b = ggml_dup_tensor(ctx.get(), a);
    ggml_set_input(a);
    ggml_set_input(b);
    // A protected graph output cannot be reused: select exactly one source, or neither.
    if (alias != 0) ggml_set_output(a);
    if (alias != 1) ggml_set_output(b);
    ggml_tensor * out = rms ? ggml_rms_norm_back(ctx.get(), a, b, eps)
                           : ggml_soft_max_ext_back(ctx.get(), a, b, scale, 0.0f);
    ggml_set_output(out);
    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 8, false);
    ggml_build_forward_expand(graph, out);
    ggml_gallocr_ptr alloc(ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend)));
    GGML_ASSERT(ggml_gallocr_alloc_graph(alloc.get(), graph));
    GGML_ASSERT((out->data == a->data) == (alias == 0));
    GGML_ASSERT((out->data == b->data) == (alias == 1));
    ggml_backend_cpu_set_n_threads(backend, threads);

    std::vector<float> da(width*rows), db(width*rows), actual(width*rows);
    std::vector<double> expected(width*rows);
    bool ok = true;
    for (int run = 0; run < 2; ++run) {
        for (int row = 0; row < rows; ++row) {
            double sum_b = 0;
            for (int col = 0; col < width; ++col) {
                const int i = row*width + col;
                da[i] = float((i + run*3) % 17 - 8)/8;
                db[i] = rms ? float((i*3 + run) % 19 - 9)/8 : float((i + run) % 11 + 1);
                sum_b += db[i];
            }
            if (!rms) {
                for (int col = 0; col < width; ++col) db[row*width + col] /= sum_b;
            }
            double dot = 0, norm = 0;
            for (int col = 0; col < width; ++col) {
                const int i = row*width + col;
                dot += double(da[i])*db[i];
                norm += double(db[i])*db[i];
            }
            const double inv_rms = 1/std::sqrt(norm/width + eps);
            for (int col = 0; col < width; ++col) {
                const int i = row*width + col;
                expected[i] = rms ? (da[i] - db[i]*dot/(norm + width*eps))*inv_rms
                                  : scale*db[i]*(da[i] - dot);
            }
        }
        // Aliased leaves are overwritten by the previous compute; upload fresh inputs each run.
        ggml_backend_tensor_set(a, da.data(), 0, da.size()*sizeof(float));
        ggml_backend_tensor_set(b, db.data(), 0, db.size()*sizeof(float));
        GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
        ggml_backend_tensor_get(out, actual.data(), 0, actual.size()*sizeof(float));
        for (size_t i = 0; i < actual.size(); ++i) {
            if (!std::isfinite(actual[i]) || std::abs(actual[i] - expected[i]) > 2e-5*std::max(1.0, std::abs(expected[i]))) {
                printf("%s width=%d alias=%d threads=%d run=%d index=%zu got=%g expected=%g\n",
                       name, width, alias, threads, run, i, actual[i], expected[i]);
                ok = false;
                break;
            }
        }
    }
    printf("%s width=%d alias=%d threads=%d: %s\n", name, width, alias, threads, ok ? "OK" : "FAIL");
    return ok;
}

static bool test_soft_max_back_range(ggml_backend_t backend) {
    ggml_init_params params = {16*ggml_tensor_overhead() + ggml_graph_overhead_custom(8, false), nullptr, true};
    ggml_context_ptr ctx(ggml_init(params));
    GGML_ASSERT(ctx);
    ggml_tensor * dy = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 3);
    ggml_tensor * y = ggml_dup_tensor(ctx.get(), dy);
    ggml_set_input(dy);
    ggml_set_input(y);
    ggml_set_output(dy);
    ggml_set_output(y);
    ggml_tensor * dx = ggml_soft_max_ext_back(ctx.get(), dy, y, 4.0f, 0.0f);
    ggml_set_output(dx);
    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 8, false);
    ggml_build_forward_expand(graph, dx);
    ggml_gallocr_ptr alloc(ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend)));
    GGML_ASSERT(ggml_gallocr_alloc_graph(alloc.get(), graph));
    ggml_backend_cpu_set_n_threads(backend, 1);

    // Multiplying scale before y overflows 4 * 1e38 to +/-inf and yields NaN for y[2] == 0.
    const float grads[3] = {1.0e38f, -1.0e38f, 1.0e38f};
    const float probs[3] = {0.5f, 0.5f, 0.0f};
    const float expected[3] = {2.0e38f, -2.0e38f, 0.0f};
    float actual[3] = {};
    ggml_backend_tensor_set(dy, grads, 0, sizeof(grads));
    ggml_backend_tensor_set(y, probs, 0, sizeof(probs));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    ggml_backend_tensor_get(dx, actual, 0, sizeof(actual));
    bool ok = true;
    for (int i = 0; i < 3; ++i) {
        if (!std::isfinite(actual[i]) || actual[i] != expected[i]) {
            printf("SOFT_MAX_BACK range index=%d got=%g expected=%g\n", i, actual[i], expected[i]);
            ok = false;
        }
    }
    printf("SOFT_MAX_BACK large-gradient range: %s\n", ok ? "OK" : "FAIL");
    return ok;
}

int main() {
    ggml_backend_ptr backend(ggml_backend_cpu_init());
    GGML_ASSERT(backend);
    int passed = 0, total = 0;
    for (bool rms : {true, false}) {
        for (int width : {7, 33}) {
            for (int alias : {-1, 0, 1}) {
                for (int threads : {1, 4}) {
                    passed += test_alias(backend.get(), rms, width, alias, threads);
                    ++total;
                }
            }
        }
    }
    passed += test_soft_max_back_range(backend.get());
    ++total;
    printf("%d/%d alias and range configurations passed\n", passed, total);
    return passed == total ? 0 : 1;
}
