// test_t2w_ane_end2end.cpp
//
// 完整端到端 ANE 路径在 C++ 端的真实运行 + mel 落盘：
//
//   - PyTorch 已离线在 /tmp/ane_t2w_work/runtime_state 准备：
//       * spk_proj.bin                  (1, 80) speaker 投影
//       * init_cnn_cache_step{0..4}.bin (32, 1024, 2)        每 timestep 各一份初始 CNN cache
//       * init_att_cache_step{0..4}.bin (32, 8, 600, 128)    每 timestep 各一份 padded att cache
//       * chunk_<i>_mu.bin              (1, 80, 56)           encoder 算好的 mu
//       * chunk_<i>_z.bin               (1, 80, 56)           初始噪声
//       * chunk_<i>_cond.bin            (1, 80, 56)           cond
//       * meta.txt                                             n_chunks / valid_cache_len_init / inference_cfg_rate
//
//   - C++ 端：
//       1. 加载 CoreML 模型（coreml_t2w_dit）
//       2. 对每个 chunk：cosine schedule + CFG batch=2 + Euler ODE × n_timesteps
//          每 step 调一次 t2w_dit_predict
//       3. 把每 chunk 的 mel 拼接，落盘 /tmp/ane_t2w_work/runtime_state/mel_cpp_ane.bin
//
//   - 后续 Python 端拿 mel_cpp_ane.bin 跑 vocoder 出 wav，做三方对比。
//
// 这个 test 验证 C++ ANE 完整 pipeline 在真实 prompt + 真实 mu 输入下的:
//   * 数值准确性（vs PyTorch 端 ANE 路径，应数值一致）
//   * 真实运行时延（包含 5 timesteps × N chunks 的 ANE 调用，扫除 dummy 输入测试中可能的 outlier）

#include "coreml/coreml_t2w_dit.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// IO helpers
// ---------------------------------------------------------------------------

static bool read_fp32(const std::string & path, std::vector<float> & out, size_t expect = 0) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        fprintf(stderr, "[end2end] open failed: %s\n", path.c_str());
        return false;
    }
    f.seekg(0, std::ios::end);
    size_t bytes = (size_t) f.tellg();
    f.seekg(0, std::ios::beg);
    if (bytes % sizeof(float) != 0) {
        fprintf(stderr, "[end2end] %s: %zu not 4-aligned\n", path.c_str(), bytes);
        return false;
    }
    size_t n = bytes / sizeof(float);
    if (expect > 0 && n != expect) {
        fprintf(stderr, "[end2end] %s: nelem=%zu expect=%zu\n", path.c_str(), n, expect);
        return false;
    }
    out.resize(n);
    f.read(reinterpret_cast<char *>(out.data()), bytes);
    return (bool) f;
}

static bool write_fp32(const std::string & path, const std::vector<float> & v) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    f.write(reinterpret_cast<const char *>(v.data()), v.size() * sizeof(float));
    return (bool) f;
}

static bool read_meta(const std::string & path, std::vector<std::string> & lines) {
    std::ifstream f(path);
    if (!f) return false;
    std::string l;
    while (std::getline(f, l)) lines.push_back(l);
    return true;
}

static int meta_int(const std::vector<std::string> & lines, const std::string & key, int fallback) {
    for (const auto & l : lines) {
        std::istringstream iss(l);
        std::string k; iss >> k;
        if (k == key) {
            int v; iss >> v;
            return v;
        }
    }
    return fallback;
}

static double meta_double(const std::vector<std::string> & lines, const std::string & key, double fallback) {
    for (const auto & l : lines) {
        std::istringstream iss(l);
        std::string k; iss >> k;
        if (k == key) {
            double v; iss >> v;
            return v;
        }
    }
    return fallback;
}

// ---------------------------------------------------------------------------
// Solver helpers
// ---------------------------------------------------------------------------

// cosine schedule: t_span[i] = 1 - cos(pi/2 * i/n)
static std::vector<float> cosine_t_span(int n_timesteps) {
    std::vector<float> ts((size_t) n_timesteps + 1);
    for (int i = 0; i <= n_timesteps; ++i) {
        const double u = (double) i / (double) n_timesteps;
        ts[(size_t) i] = (float) (1.0 - std::cos(u * 0.5 * M_PI));
    }
    return ts;
}

// 构造 K 维布局 = [chunk(T) | cache(Tc)] 的 additive mask:
//   前 T 全 0；接下来 valid_cache_len 个 0；其余 -1e4
static void build_attn_mask(int B, int T, int Tc, int valid_cache_len, std::vector<float> & out) {
    valid_cache_len = std::max(0, std::min(valid_cache_len, Tc));
    out.assign((size_t) B * (size_t) T * (size_t) (T + Tc), -1e4f);
    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < T; ++t) {
            float * row = out.data() + ((size_t) b * T + t) * (size_t) (T + Tc);
            for (int k = 0; k < T + valid_cache_len; ++k) {
                row[k] = 0.0f;  // valid
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 主流程
// ---------------------------------------------------------------------------

int main(int argc, char ** argv) {
    std::string state_dir;
    std::string mlpkg;
    std::string out_mel;
    std::string dump_io_dir;  // 若非空，把 chunk 0 step 1 的所有 IO 落盘，方便 cross-check

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--state" && i + 1 < argc) state_dir = argv[++i];
        else if (a == "--mlpkg" && i + 1 < argc) mlpkg = argv[++i];
        else if (a == "--out-mel" && i + 1 < argc) out_mel = argv[++i];
        else if (a == "--dump-io-dir" && i + 1 < argc) dump_io_dir = argv[++i];
        else if (a == "-h" || a == "--help") {
            printf("Usage: %s [--state <dir>] [--mlpkg <pkg>] [--out-mel <file>] "
                   "[--dump-io-dir <dir>]\n", argv[0]);
            return 0;
        }
    }

    if (mlpkg.empty() || state_dir.empty()) {
        fprintf(stderr, "Error: --mlpkg <path> and --state <dir> are required\n");
        fprintf(stderr, "Usage: %s --mlpkg <path> --state <dir> [--out-mel <file>] [--dump-io-dir <dir>]\n", argv[0]);
        return 1;
    }
    if (out_mel.empty()) {
        out_mel = state_dir + "/mel_cpp_ane.bin";
    }

    // 1. read meta
    std::vector<std::string> meta;
    if (!read_meta(state_dir + "/meta.txt", meta)) {
        fprintf(stderr, "[end2end] no meta.txt at %s\n", state_dir.c_str());
        return 1;
    }
    const int n_chunks       = meta_int(meta, "n_chunks", 0);
    const int n_timesteps    = meta_int(meta, "n_timesteps", 5);
    const int chunk_size     = meta_int(meta, "chunk_size", 56);
    const int max_cache_len  = meta_int(meta, "max_cache_len", 600);
    const int valid_cache_init = meta_int(meta, "valid_cache_len_init", 302);
    const int depth          = meta_int(meta, "depth", 16);
    const int B              = meta_int(meta, "batch", 2);
    const int num_heads      = meta_int(meta, "num_heads", 8);
    const int head_dim       = meta_int(meta, "head_dim", 64);
    const int mel_channels   = meta_int(meta, "mel_channels", 80);
    const float cfg_rate     = (float) meta_double(meta, "inference_cfg_rate", 0.7);

    printf("=== test_t2w_ane_end2end ===\n");
    printf("  state_dir:   %s\n", state_dir.c_str());
    printf("  mlpkg:       %s\n", mlpkg.c_str());
    printf("  n_chunks:    %d\n", n_chunks);
    printf("  n_timesteps: %d\n", n_timesteps);
    printf("  chunk_size:  %d  max_cache_len: %d  valid_init: %d\n",
           chunk_size, max_cache_len, valid_cache_init);
    printf("  cfg_rate:    %.3f\n", cfg_rate);

    // 2. load ANE
    printf("\n[1/4] load CoreML model ...\n");
    auto t0 = std::chrono::high_resolution_clock::now();
    t2w_dit_handle_t h = t2w_dit_load(mlpkg.c_str());
    if (!h) { fprintf(stderr, "load failed\n"); return 2; }
    double ms_load = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - t0).count();
    printf("  loaded in %.1f ms\n", ms_load);

    t2w_dit_dims_t dims;
    t2w_dit_get_dims(h, &dims);
    if (dims.depth != depth || dims.batch != B || dims.chunk_size != chunk_size
        || dims.max_cache_len != max_cache_len || dims.num_heads != num_heads
        || dims.head_dim != head_dim || dims.mel_channels != mel_channels) {
        fprintf(stderr, "[end2end] meta vs CoreML model dims mismatch\n");
        t2w_dit_free(h);
        return 3;
    }

    // 3. load static state: spk_proj + per-timestep init caches
    printf("\n[2/4] load static state ...\n");
    const size_t n_xmcf      = (size_t) B * mel_channels * chunk_size;
    const size_t n_spks      = (size_t) B * mel_channels;
    const size_t n_cnn       = (size_t) depth * B * 1024 * 2;
    const size_t n_att       = (size_t) depth * B * num_heads * max_cache_len * (2 * head_dim);
    const size_t n_mask      = (size_t) B * chunk_size * (chunk_size + max_cache_len);

    std::vector<float> spk_proj_b1;  // (1, 80)
    if (!read_fp32(state_dir + "/spk_proj.bin", spk_proj_b1, (size_t) mel_channels)) {
        t2w_dit_free(h);
        return 4;
    }

    // CFG batch=2: spks_b2 = [spk_proj; 0]
    std::vector<float> spks_b2(n_spks, 0.0f);
    std::memcpy(spks_b2.data(), spk_proj_b1.data(), n_spks / 2 * sizeof(float));

    // 每 timestep 各自一份 cnn / att cache
    std::vector<std::vector<float>> cnn_caches(n_timesteps), att_caches(n_timesteps);
    for (int s = 0; s < n_timesteps; ++s) {
        char buf[64];
        snprintf(buf, sizeof(buf), "/init_cnn_cache_step%d.bin", s);
        if (!read_fp32(state_dir + buf, cnn_caches[s], n_cnn)) { t2w_dit_free(h); return 5; }
        snprintf(buf, sizeof(buf), "/init_att_cache_step%d.bin", s);
        if (!read_fp32(state_dir + buf, att_caches[s], n_att)) { t2w_dit_free(h); return 5; }
    }
    printf("  per-timestep cache loaded (%d × cnn=%zu MB + att=%zu MB)\n",
           n_timesteps, n_cnn * 4 / (1<<20), n_att * 4 / (1<<20));

    // ---- 4. 主循环 ----
    printf("\n[3/4] inference %d chunks × %d timesteps ...\n", n_chunks, n_timesteps);
    const std::vector<float> t_span_v = cosine_t_span(n_timesteps);

    std::vector<float> mel_total;  // 拼接所有 chunk 的 mel：(80, T_total)，row-major

    int valid_cache_len = valid_cache_init;
    double total_ane_ms = 0.0;
    double total_chunk_ms = 0.0;

    // 预分配 working buffers
    std::vector<float> x_b1(n_xmcf / 2);
    std::vector<float> mu_b1(n_xmcf / 2);
    std::vector<float> cond_b1(n_xmcf / 2);
    std::vector<float> z_b1(n_xmcf / 2);

    std::vector<float> x_b2(n_xmcf);
    std::vector<float> mu_b2(n_xmcf);
    std::vector<float> cond_b2(n_xmcf);
    std::vector<float> t_b2((size_t) B);
    std::vector<float> mask_b2(n_mask);

    std::vector<float> feat_b2(n_xmcf);
    std::vector<float> cnn_out(n_cnn);
    std::vector<float> att_out(n_att);

    for (int ci = 0; ci < n_chunks; ++ci) {
        char buf[64];
        snprintf(buf, sizeof(buf), "/chunk_%03d_mu.bin", ci);
        if (!read_fp32(state_dir + buf, mu_b1, n_xmcf / 2)) { t2w_dit_free(h); return 6; }
        snprintf(buf, sizeof(buf), "/chunk_%03d_z.bin", ci);
        if (!read_fp32(state_dir + buf, z_b1, n_xmcf / 2)) { t2w_dit_free(h); return 6; }
        snprintf(buf, sizeof(buf), "/chunk_%03d_cond.bin", ci);
        if (!read_fp32(state_dir + buf, cond_b1, n_xmcf / 2)) { t2w_dit_free(h); return 6; }

        // 把 b=1 上的 mu/cond 复制成 b=2: [orig; 0] (uncond branch 用 0 替代)
        std::memset(mu_b2.data(),   0, mu_b2.size()   * sizeof(float));
        std::memset(cond_b2.data(), 0, cond_b2.size() * sizeof(float));
        std::memcpy(mu_b2.data(),   mu_b1.data(),   mu_b1.size()   * sizeof(float));
        std::memcpy(cond_b2.data(), cond_b1.data(), cond_b1.size() * sizeof(float));

        // x init = z
        std::copy(z_b1.begin(), z_b1.end(), x_b1.begin());

        // build attn_mask once（每个 step 共用，valid 长度在本 chunk 内不变）
        build_attn_mask(B, chunk_size, max_cache_len, valid_cache_len, mask_b2);

        const auto t_chunk0 = std::chrono::high_resolution_clock::now();
        float t_scalar = t_span_v[0];
        float dt = t_span_v[1] - t_span_v[0];

        for (int step = 1; step <= n_timesteps; ++step) {
            // x_b2 = [x; x]
            std::memcpy(x_b2.data(),                       x_b1.data(), x_b1.size() * sizeof(float));
            std::memcpy(x_b2.data() + x_b1.size(),         x_b1.data(), x_b1.size() * sizeof(float));
            // t_b2 = [t; t]
            t_b2[0] = t_scalar; t_b2[1] = t_scalar;

            // 可选：dump chunk 0 第 1 个 step 的所有 IO（用于 cross-check 数值）
            if (!dump_io_dir.empty() && ci == 0 && step == 1) {
                auto dump = [&](const std::string & name, const std::vector<float> & v) {
                    std::ofstream f(dump_io_dir + "/" + name, std::ios::binary);
                    f.write(reinterpret_cast<const char *>(v.data()), v.size() * sizeof(float));
                };
                dump("io_x.bin",            x_b2);
                dump("io_mu.bin",           mu_b2);
                dump("io_t.bin",            t_b2);
                dump("io_spks.bin",         spks_b2);
                dump("io_cond.bin",         cond_b2);
                dump("io_cnn_cache_in.bin", cnn_caches[step - 1]);
                dump("io_att_cache_in.bin", att_caches[step - 1]);
                dump("io_attn_mask.bin",    mask_b2);
                fprintf(stderr, "[dump] chunk0 step1 inputs saved to %s\n", dump_io_dir.c_str());
            }

            const auto t_pred0 = std::chrono::high_resolution_clock::now();
            int rc = t2w_dit_predict(h,
                x_b2.data(), mu_b2.data(), t_b2.data(), spks_b2.data(), cond_b2.data(),
                cnn_caches[step - 1].data(), att_caches[step - 1].data(), mask_b2.data(),
                feat_b2.data(), cnn_out.data(), att_out.data());
            if (rc != 0) {
                fprintf(stderr, "predict failed rc=%d at chunk %d step %d\n", rc, ci, step);
                t2w_dit_free(h);
                return 7;
            }
            const double ms_pred = std::chrono::duration<double, std::milli>(
                std::chrono::high_resolution_clock::now() - t_pred0).count();
            total_ane_ms += ms_pred;

            if (!dump_io_dir.empty() && ci == 0 && step == 1) {
                auto dump = [&](const std::string & name, const std::vector<float> & v) {
                    std::ofstream f(dump_io_dir + "/" + name, std::ios::binary);
                    f.write(reinterpret_cast<const char *>(v.data()), v.size() * sizeof(float));
                };
                dump("io_feat.bin",          feat_b2);
                dump("io_cnn_cache_out.bin", cnn_out);
                dump("io_att_cache_out.bin", att_out);
                fprintf(stderr, "[dump] chunk0 step1 outputs saved\n");
            }

            // CFG: dphi = (1+cfg) * cond_branch - cfg * uncond_branch
            // feat layout: [B0=cond_branch; B1=uncond_branch]，B0 占前一半
            const size_t half = n_xmcf / 2;
            const float k1 = 1.0f + cfg_rate;
            const float k2 = cfg_rate;
            for (size_t i = 0; i < half; ++i) {
                const float dphi = k1 * feat_b2[i] - k2 * feat_b2[half + i];
                x_b1[i] = x_b1[i] + dt * dphi;
            }

            t_scalar += dt;
            if (step < n_timesteps) {
                dt = t_span_v[(size_t) step + 1] - t_scalar;
            }

            // 更新该 timestep 的 cache（流转到下一个 chunk）
            std::swap(cnn_caches[step - 1], cnn_out);
            std::swap(att_caches[step - 1], att_out);
            // 注意 swap 后 cnn_out / att_out 持有上一帧的 buffer，下一次 predict
            // 会被覆盖；但 size 不变所以 OK
        }
        const double ms_chunk = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t_chunk0).count();
        total_chunk_ms += ms_chunk;

        // x_b1 即本 chunk 输出 mel，shape (1, 80, 56)。
        // 拼接到 mel_total（按 (80, T_total) 行优先），保持每行（channel）连续。
        // 输入 row-major (1, 80, 56)：c 慢、t 快。要拼成 (80, T_total)，每行 c 紧密放，按 t 拼接。
        const size_t T_old = mel_total.empty() ? 0 : mel_total.size() / mel_channels;
        const size_t T_new = T_old + chunk_size;
        std::vector<float> next(mel_channels * T_new);
        for (int c = 0; c < mel_channels; ++c) {
            // 拷贝旧的 c 行
            if (T_old > 0) {
                std::memcpy(next.data() + (size_t) c * T_new,
                            mel_total.data() + (size_t) c * T_old,
                            T_old * sizeof(float));
            }
            // 追加本 chunk 的 c 行（在 x_b1 中位置 = c*56）
            std::memcpy(next.data() + (size_t) c * T_new + T_old,
                        x_b1.data() + (size_t) c * chunk_size,
                        (size_t) chunk_size * sizeof(float));
        }
        mel_total.swap(next);

        printf("  chunk %d  valid_cache=%3d  chunk=%.1fms  ane_avg/step=%.2fms\n",
               ci, valid_cache_len, ms_chunk, ms_chunk / n_timesteps);

        valid_cache_len = std::min(valid_cache_len + chunk_size, max_cache_len);
    }

    printf("\n[4/4] save mel ...\n");
    if (!write_fp32(out_mel, mel_total)) {
        fprintf(stderr, "write failed: %s\n", out_mel.c_str());
        t2w_dit_free(h);
        return 8;
    }
    printf("  saved: %s   shape=(80, %zu)  bytes=%zu\n",
           out_mel.c_str(), mel_total.size() / mel_channels, mel_total.size() * 4);

    printf("\n=== Summary ===\n");
    printf("  total chunk wall: %.1fms (avg %.1fms / chunk)\n",
           total_chunk_ms, total_chunk_ms / n_chunks);
    printf("  total ANE predict: %.1fms (avg %.2fms / call, %d calls)\n",
           total_ane_ms, total_ane_ms / (double) (n_chunks * n_timesteps),
           n_chunks * n_timesteps);

    t2w_dit_free(h);
    return 0;
}
