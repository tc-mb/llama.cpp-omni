// test_t2w_dit_ane.cpp
//
// 独立单元测试：验证 coreml_t2w_dit C ABI 工作正常，
// 且 C++ 端 predict 输出与 Python 端 predict 输出 bit/数值一致。
//
// 工作流程：
//   1. 离线在 Python 侧生成一组确定性 dummy 输入（fix-seed），喂给 ANE 模型，
//      把 inputs/outputs 全部落盘到 raw fp32 .bin 文件。
//      （由 tools/convert/o45_tts/dump_test_vectors.py 生成）
//   2. 本测试读这些 .bin，调 t2w_dit_predict()，比对输出与 Python 端落盘的输出。
//
// 退出码: 0 成功 / 非 0 失败。
//
// 用法:
//   ./build/bin/llama-omni-test-t2w-dit-ane \
//       --mlpkg <path/to/coreml_minicpmo45_t2w_dit.mlpackage> \
//       --vec-dir <path/to/test_vectors>

#include "coreml/coreml_t2w_dit.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <fstream>
#include <string>
#include <vector>

static bool read_fp32(const std::string & path, std::vector<float> & out, size_t expected_n = 0) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        fprintf(stderr, "[test] failed to open: %s\n", path.c_str());
        return false;
    }
    f.seekg(0, std::ios::end);
    size_t bytes = (size_t) f.tellg();
    f.seekg(0, std::ios::beg);
    if (bytes % sizeof(float) != 0) {
        fprintf(stderr, "[test] %s: size %zu not multiple of 4\n", path.c_str(), bytes);
        return false;
    }
    size_t n = bytes / sizeof(float);
    if (expected_n > 0 && n != expected_n) {
        fprintf(stderr, "[test] %s: nelem=%zu expected %zu\n", path.c_str(), n, expected_n);
        return false;
    }
    out.resize(n);
    f.read(reinterpret_cast<char *>(out.data()), bytes);
    if (!f) {
        fprintf(stderr, "[test] read failed: %s\n", path.c_str());
        return false;
    }
    return true;
}

struct DiffStats {
    double max_d  = 0.0;
    double mean_d = 0.0;
    double rms    = 0.0;
    double rel    = 0.0;
};

static DiffStats compute_diff(const std::vector<float> & a, const std::vector<float> & b) {
    DiffStats st;
    if (a.size() != b.size() || a.empty()) return st;
    double max_d = 0.0, sum_abs = 0.0, sum_sq = 0.0;
    float a_max = a[0], a_min = a[0];
    for (size_t i = 0; i < a.size(); ++i) {
        const double d = std::fabs((double) a[i] - (double) b[i]);
        if (d > max_d) max_d = d;
        sum_abs += d;
        sum_sq += d * d;
        if (a[i] > a_max) a_max = a[i];
        if (a[i] < a_min) a_min = a[i];
    }
    st.max_d  = max_d;
    st.mean_d = sum_abs / (double) a.size();
    st.rms    = std::sqrt(sum_sq / (double) a.size());
    const double range = (double) a_max - (double) a_min;
    st.rel    = max_d / (range + 1e-9);
    return st;
}

int main(int argc, char ** argv) {
    std::string mlpkg;
    std::string vec_dir;
    int repeat = 5;
    double tol_max = -1.0;  // -1 表示自动按 compute_units 选

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--mlpkg" && i + 1 < argc) {
            mlpkg = argv[++i];
        } else if (a == "--vec-dir" && i + 1 < argc) {
            vec_dir = argv[++i];
        } else if (a == "--repeat" && i + 1 < argc) {
            repeat = std::atoi(argv[++i]);
        } else if (a == "--tol" && i + 1 < argc) {
            tol_max = std::atof(argv[++i]);
        } else if (a == "-h" || a == "--help") {
            printf("Usage: %s [--mlpkg <path>] [--vec-dir <dir>] [--repeat <n>] [--tol <max|Δ|>]\n", argv[0]);
            return 0;
        }
    }
    if (tol_max < 0) {
        const char * cu = getenv("OMNI_T2W_DIT_COMPUTE_UNITS");
        if (cu && std::string(cu) == "cpu") {
            tol_max = 1e-4;     // CPU 路径 bit-exact
        } else {
            tol_max = 0.5;      // ANE / GPU 路径 fp16 量化预期损失
        }
    }

    if (mlpkg.empty() || vec_dir.empty()) {
        fprintf(stderr, "Error: --mlpkg <path> and --vec-dir <dir> are required\n");
        fprintf(stderr, "Usage: %s --mlpkg <path> --vec-dir <dir> [--repeat <n>] [--tol <max|Δ|>]\n", argv[0]);
        return 1;
    }

    printf("=== test_t2w_dit_ane ===\n");
    printf("  mlpkg:    %s\n", mlpkg.c_str());
    printf("  vec_dir:  %s\n", vec_dir.c_str());
    printf("  repeat:   %d\n\n", repeat);

    // 1. load model
    printf("[1/4] load CoreML model ...\n");
    auto t0 = std::chrono::high_resolution_clock::now();
    t2w_dit_handle_t h = t2w_dit_load(mlpkg.c_str());
    if (!h) {
        fprintf(stderr, "load failed\n");
        return 1;
    }
    auto t_load = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - t0).count();
    printf("  loaded in %.1f ms\n", t_load);

    // 2. dims & buffer sizes
    t2w_dit_dims_t d;
    if (t2w_dit_get_dims(h, &d) != 0) {
        fprintf(stderr, "get_dims failed\n");
        t2w_dit_free(h);
        return 1;
    }
    printf("  dims: depth=%d batch=%d chunk=%d cache=%d heads=%d head_dim=%d "
           "mel=%d spk=%d cnn(c=%d t=%d)\n",
           d.depth, d.batch, d.chunk_size, d.max_cache_len, d.num_heads, d.head_dim,
           d.mel_channels, d.spk_dim, d.cnn_cache_channels, d.cnn_cache_t);

    size_t n_xmcf, n_t, n_spks, n_cnn, n_att, n_mask;
    t2w_dit_predict_buffer_sizes(h, &n_xmcf, &n_t, &n_spks, &n_cnn, &n_att, &n_mask);
    printf("  buffer sizes (fp32 elems):\n");
    printf("    x/mu/cond/feat: %zu\n", n_xmcf);
    printf("    t:              %zu\n", n_t);
    printf("    spks:           %zu\n", n_spks);
    printf("    cnn_cache:      %zu\n", n_cnn);
    printf("    att_cache:      %zu\n", n_att);
    printf("    attn_mask:      %zu\n", n_mask);

    // 3. load test vectors (from Python dumper)
    printf("\n[2/4] load test vectors from %s ...\n", vec_dir.c_str());
    std::vector<float> in_x, in_mu, in_t, in_spks, in_cond,
                       in_cnn, in_att, in_mask,
                       ref_feat, ref_cnn_out, ref_att_out;
    bool ok = true;
    ok &= read_fp32(vec_dir + "/in_x.bin",            in_x,        n_xmcf);
    ok &= read_fp32(vec_dir + "/in_mu.bin",           in_mu,       n_xmcf);
    ok &= read_fp32(vec_dir + "/in_t.bin",            in_t,        n_t);
    ok &= read_fp32(vec_dir + "/in_spks.bin",         in_spks,     n_spks);
    ok &= read_fp32(vec_dir + "/in_cond.bin",         in_cond,     n_xmcf);
    ok &= read_fp32(vec_dir + "/in_cnn_cache.bin",    in_cnn,      n_cnn);
    ok &= read_fp32(vec_dir + "/in_att_cache.bin",    in_att,      n_att);
    ok &= read_fp32(vec_dir + "/in_attn_mask.bin",    in_mask,     n_mask);
    ok &= read_fp32(vec_dir + "/ref_feat.bin",        ref_feat,    n_xmcf);
    ok &= read_fp32(vec_dir + "/ref_cnn_cache.bin",   ref_cnn_out, n_cnn);
    ok &= read_fp32(vec_dir + "/ref_att_cache.bin",   ref_att_out, n_att);
    if (!ok) {
        fprintf(stderr, "test vectors missing/wrong size; run dump_test_vectors.py first\n");
        t2w_dit_free(h);
        return 2;
    }
    printf("  all 11 vectors loaded\n");

    // 打印每个输入张量的 stat，跟 Python dump 端的 stat 对照（看字节是否一致）
    auto stat = [](const std::vector<float> & v) {
        if (v.empty()) return std::make_tuple(0.0, 0.0, 0.0, 0.0);
        double mn = v[0], mx = v[0], s = 0.0, ss = 0.0;
        for (float x : v) {
            if (x < mn) mn = x;
            if (x > mx) mx = x;
            s += x;
            ss += (double) x * x;
        }
        return std::make_tuple(mn, mx, s / (double) v.size(),
                               std::sqrt(ss / (double) v.size()));
    };
    auto pr = [&](const char * n, const std::vector<float> & v) {
        auto [mn, mx, mean, rms] = stat(v);
        printf("    %-22s n=%-9zu min=%-+10.4e max=%-+10.4e mean=%-+10.4e rms=%-10.4e\n",
               n, v.size(), mn, mx, mean, rms);
    };
    printf("  input stats (C++):\n");
    pr("in_x",         in_x);
    pr("in_mu",        in_mu);
    pr("in_t",         in_t);
    pr("in_spks",      in_spks);
    pr("in_cond",      in_cond);
    pr("in_cnn_cache", in_cnn);
    pr("in_att_cache", in_att);
    pr("in_attn_mask", in_mask);
    printf("  expected ref stats (loaded from .bin):\n");
    pr("ref_feat",     ref_feat);
    pr("ref_cnn_cache", ref_cnn_out);
    pr("ref_att_cache", ref_att_out);

    // 4. predict + verify
    printf("\n[3/4] predict (n=%d) ...\n", repeat);
    std::vector<float> out_feat(n_xmcf), out_cnn(n_cnn), out_att(n_att);
    double ms_first = 0.0, ms_steady_sum = 0.0;
    for (int i = 0; i < repeat; ++i) {
        auto ti = std::chrono::high_resolution_clock::now();
        int rc = t2w_dit_predict(h,
            in_x.data(), in_mu.data(), in_t.data(), in_spks.data(), in_cond.data(),
            in_cnn.data(), in_att.data(), in_mask.data(),
            out_feat.data(), out_cnn.data(), out_att.data());
        auto te = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(te - ti).count();
        if (rc != 0) {
            fprintf(stderr, "predict failed rc=%d at iter %d\n", rc, i);
            t2w_dit_free(h);
            return 3;
        }
        if (i == 0) ms_first = ms;
        else        ms_steady_sum += ms;
        printf("  iter %d  %.2f ms\n", i, ms);
    }
    if (repeat >= 2) {
        printf("  first call:    %.2f ms\n", ms_first);
        printf("  steady avg:    %.2f ms (n=%d)\n",
               ms_steady_sum / (double) (repeat - 1), repeat - 1);
    }

    // 5. compare to reference
    printf("\n[4/4] verify against reference ...\n");
    printf("  output stats (C++):\n");
    pr("out_feat",      out_feat);
    pr("out_cnn_cache", out_cnn);
    pr("out_att_cache", out_att);
    DiffStats df = compute_diff(ref_feat,    out_feat);
    DiffStats dc = compute_diff(ref_cnn_out, out_cnn);
    DiffStats da = compute_diff(ref_att_out, out_att);
    printf("  feat:           max|Δ|=%.4e  mean|Δ|=%.4e  rms=%.4e  rel=%.4f%%\n",
           df.max_d, df.mean_d, df.rms, df.rel * 100);
    printf("  cnn_cache_out:  max|Δ|=%.4e  mean|Δ|=%.4e  rms=%.4e  rel=%.4f%%\n",
           dc.max_d, dc.mean_d, dc.rms, dc.rel * 100);
    printf("  att_cache_out:  max|Δ|=%.4e  mean|Δ|=%.4e  rms=%.4e  rel=%.4f%%\n",
           da.max_d, da.mean_d, da.rms, da.rel * 100);

    // 阈值：CPU_ONLY 时 bit-exact (~1e-4)；ANE 模式预期 fp16 量化损失 ~0.1-0.5。
    bool pass = (df.max_d <= tol_max) && (dc.max_d <= tol_max) && (da.max_d <= tol_max);
    if (!pass) {
        printf("\n  >>> FAIL: 数值差距超过容差 %.4f\n", tol_max);
        t2w_dit_free(h);
        return 4;
    }
    printf("\n  >>> PASS (容差 max|Δ| ≤ %.4f)\n", tol_max);

    t2w_dit_free(h);
    return 0;
}
