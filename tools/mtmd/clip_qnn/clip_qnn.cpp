//
// clip_qnn.cpp
//
// QNN bridge implementation for MiniCPM-OCR 4.5 ViT (Hexagon NPU)
//
// Input shapes (OCR 1.0, linear layout):
//   pixel_values:  [1, 3, 14, 14336] (float16)
//   position_ids:  [1024] (int32)
//   merger_ds_idx_0..3: [256] (int32)
//   attn_bias:     [1024] (float16)
// Output shape:
//   output:        [256, 1024] (float16)
//

#include "clip_qnn.h"

#if defined(__ANDROID__) && defined(ENABLE_QNN)

#include "clip-impl.h"

#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <mutex>
#include <string>
#include <vector>

#include "QNN/QnnInterface.h"
#include "QNN/QnnTypes.h"
#include "QNN/System/QnnSystemInterface.h"

#define QNN_TAG "clip_qnn"

namespace {

constexpr int kNEmbd      = 1024;
constexpr int kPatchSize  = 14;
constexpr int kPatches    = 1024;
constexpr int kCanvasW    = kPatches * kPatchSize;
constexpr int kNTokens    = kPatches / 4;

struct Session {
    std::mutex mu;

    void * lib_backend = nullptr;
    void * lib_system  = nullptr;

    Qnn_BackendHandle_t backend = nullptr;
    Qnn_DeviceHandle_t  device  = nullptr;
    Qnn_ContextHandle_t context = nullptr;
    Qnn_GraphHandle_t   graph   = nullptr;
    Qnn_LogHandle_t     logger  = nullptr;

    std::vector<uint8_t> binary;
    std::string graph_name;

    std::vector<uint16_t> in_image;
    std::vector<int32_t>  in_pos;
    std::vector<int32_t>  in_ds[4];
    std::vector<uint16_t> in_bias;
    std::vector<uint16_t> out_buf;

    Qnn_Tensor_t in_tensors[7];
    Qnn_Tensor_t out_tensor;
};

Session g_s;

const QNN_INTERFACE_VER_TYPE * g_qnn = nullptr;

// ---------------------------------------------------------------- f16 helpers

float f16_to_f32(uint16_t h) {
    const uint32_t sign = (h >> 15) & 1u;
    const uint32_t exp  = (h >> 10) & 0x1fu;
    const uint32_t mant = h & 0x3ffu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign << 31;
        } else {
            uint32_t e = 0;
            uint32_t m = mant;
            while ((m & 0x400u) == 0) {
                m <<= 1;
                ++e;
            }
            m &= 0x3ffu;
            bits = (sign << 31) | ((127 - 15 - static_cast<int32_t>(e) + 1) << 23) | (m << 13);
        }
    } else if (exp == 0x1f) {
        bits = (sign << 31) | 0x7f800000u | (mant << 13);
    } else {
        bits = (sign << 31) | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

uint16_t f32_to_f16(float f) {
    uint32_t x;
    memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    const int32_t exp = static_cast<int32_t>((x >> 23) & 0xffu) - 127;
    const uint32_t mant = x & 0x7fffffu;
    if (exp > 15) {
        return static_cast<uint16_t>(sign | 0x7c00u);
    }
    if (exp < -14) {
        if (exp < -24) {
            return static_cast<uint16_t>(sign);
        }
        const uint32_t shift = static_cast<uint32_t>(-14 - exp);
        const uint32_t m = (mant | 0x800000u) >> (13 + shift);
        return static_cast<uint16_t>(sign | m);
    }
    const uint32_t rounded = mant + 0x1000u;
    if (rounded & 0x800000u) {
        const int32_t e = exp + 1;
        if (e > 15) {
            return static_cast<uint16_t>(sign | 0x7c00u);
        }
        return static_cast<uint16_t>(sign | (static_cast<uint32_t>(e + 15) << 10));
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp + 15) << 10) | (rounded >> 13));
}

// ---------------------------------------------------------------- QNN helpers

void log_cb(const char * fmt, QnnLog_Level_t level, uint64_t, va_list args) {
    if (level > QNN_LOG_LEVEL_WARN) {
        return;
    }
    char buf[512];
    vsnprintf(buf, sizeof(buf), fmt, args);
    LOG_INF("[qnn] %s", buf);
}

size_t dtype_bytes(Qnn_DataType_t t) {
    switch (t) {
        case QNN_DATATYPE_FLOAT_32:
        case QNN_DATATYPE_INT_32:
        case QNN_DATATYPE_UINT_32:   return 4;
        case QNN_DATATYPE_FLOAT_16:
        case QNN_DATATYPE_BFLOAT_16: return 2;
        default:                     return 1;
    }
}

size_t tensor_numel(const Qnn_Tensor_t & t) {
    size_t n = 1;
    for (uint32_t i = 0; i < t.v2.rank; ++i) {
        n *= t.v2.dimensions[i];
    }
    return n;
}

bool read_file(const char * path, std::vector<uint8_t> & out) {
    FILE * f = fopen(path, "rb");
    if (!f) {
        LOG_ERR("%s: cannot open %s", QNN_TAG, path);
        return false;
    }
    fseek(f, 0, SEEK_END);
    const long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (size <= 0) {
        fclose(f);
        LOG_ERR("%s: %s is empty", QNN_TAG, path);
        return false;
    }
    out.resize(static_cast<size_t>(size));
    const size_t got = fread(out.data(), 1, out.size(), f);
    fclose(f);
    if (got != out.size()) {
        LOG_ERR("%s: short read %zu/%zu", QNN_TAG, got, out.size());
        return false;
    }
    return true;
}

const Qnn_Tensor_t * find_input(const QnnSystemContext_GraphInfoV1_t & gi, const char * name) {
    for (uint32_t i = 0; i < gi.numGraphInputs; ++i) {
        if (gi.graphInputs[i].v2.name && strcmp(gi.graphInputs[i].v2.name, name) == 0) {
            return &gi.graphInputs[i];
        }
    }
    return nullptr;
}

void bind_tensor(Qnn_Tensor_t & dst, const Qnn_Tensor_t & desc, void * data, size_t bytes) {
    dst = desc;
    dst.v2.type = QNN_TENSOR_TYPE_APP_WRITE;
    dst.v2.memType = QNN_TENSORMEMTYPE_RAW;
    dst.v2.clientBuf.data = data;
    dst.v2.clientBuf.dataSize = static_cast<uint32_t>(bytes);
}

}  // namespace

// ---------------------------------------------------------------- public API

const void* clip_qnn_loadModel(const char* model_path, const char* skel_dir) {
    std::lock_guard<std::mutex> lock(g_s.mu);

    if (g_s.graph) {
        return &g_s;
    }
    if (!model_path) {
        return nullptr;
    }

    // the DSP skel must match the host libraries, else the invocation symbol
    // resolves to an older build on the vendor partition and fails
    if (skel_dir) {
        setenv("ADSP_LIBRARY_PATH", skel_dir, 1);
        LOG_INF("%s: ADSP_LIBRARY_PATH=%s", QNN_TAG, skel_dir);
    }

    g_s.lib_backend = dlopen("libQnnHtp.so", RTLD_NOW | RTLD_GLOBAL);
    if (!g_s.lib_backend) {
        LOG_ERR("%s: dlopen libQnnHtp.so: %s", QNN_TAG, dlerror());
        return nullptr;
    }
    g_s.lib_system = dlopen("libQnnSystem.so", RTLD_NOW | RTLD_GLOBAL);
    if (!g_s.lib_system) {
        LOG_ERR("%s: dlopen libQnnSystem.so: %s", QNN_TAG, dlerror());
        return nullptr;
    }

    auto get_providers = reinterpret_cast<Qnn_ErrorHandle_t (*)(const QnnInterface_t ***, uint32_t *)>(
        dlsym(g_s.lib_backend, "QnnInterface_getProviders"));
    auto get_sys_providers = reinterpret_cast<Qnn_ErrorHandle_t (*)(const QnnSystemInterface_t ***, uint32_t *)>(
        dlsym(g_s.lib_system, "QnnSystemInterface_getProviders"));
    if (!get_providers || !get_sys_providers) {
        LOG_ERR("%s: dlsym QnnInterface_getProviders failed", QNN_TAG);
        return nullptr;
    }

    const QnnInterface_t ** providers = nullptr;
    uint32_t n_providers = 0;
    if (get_providers(&providers, &n_providers) != QNN_SUCCESS || n_providers == 0) {
        LOG_ERR("%s: QnnInterface_getProviders failed", QNN_TAG);
        return nullptr;
    }
    const QnnInterface_t * provider = nullptr;
    for (uint32_t i = 0; i < n_providers; ++i) {
        const auto & v = providers[i]->apiVersion.coreApiVersion;
        if (v.major == QNN_API_VERSION_MAJOR && v.minor >= QNN_API_VERSION_MINOR) {
            provider = providers[i];
        }
    }
    if (!provider) {
        LOG_ERR("%s: no QNN provider with API %d.%d", QNN_TAG, QNN_API_VERSION_MAJOR, QNN_API_VERSION_MINOR);
        return nullptr;
    }
    const QNN_INTERFACE_VER_TYPE & qnn = provider->QNN_INTERFACE_VER_NAME;
    g_qnn = &qnn;

    const QnnSystemInterface_t ** sys_providers = nullptr;
    uint32_t n_sys = 0;
    if (get_sys_providers(&sys_providers, &n_sys) != QNN_SUCCESS || n_sys == 0) {
        LOG_ERR("%s: QnnSystemInterface_getProviders failed", QNN_TAG);
        return nullptr;
    }
    const QNN_SYSTEM_INTERFACE_VER_TYPE & sys = sys_providers[0]->QNN_SYSTEM_INTERFACE_VER_NAME;

    if (qnn.logCreate(log_cb, QNN_LOG_LEVEL_WARN, &g_s.logger) != QNN_SUCCESS) {
        LOG_ERR("%s: logCreate failed", QNN_TAG);
        return nullptr;
    }
    if (qnn.backendCreate(g_s.logger, nullptr, &g_s.backend) != QNN_SUCCESS) {
        LOG_ERR("%s: backendCreate failed", QNN_TAG);
        return nullptr;
    }
    if (qnn.deviceCreate(g_s.logger, nullptr, &g_s.device) != QNN_SUCCESS) {
        LOG_ERR("%s: deviceCreate failed", QNN_TAG);
        return nullptr;
    }

    if (!read_file(model_path, g_s.binary)) {
        return nullptr;
    }
    LOG_INF("%s: context binary %s: %.1f MiB", QNN_TAG, model_path, g_s.binary.size() / 1048576.0);

    if (qnn.contextCreateFromBinary(g_s.backend, g_s.device, nullptr, g_s.binary.data(),
                                    g_s.binary.size(), &g_s.context, nullptr) != QNN_SUCCESS) {
        LOG_ERR("%s: contextCreateFromBinary failed", QNN_TAG);
        return nullptr;
    }

    QnnSystemContext_Handle_t sys_ctx = nullptr;
    const QnnSystemContext_BinaryInfo_t * info = nullptr;
    Qnn_ContextBinarySize_t info_size = 0;
    if (sys.systemContextCreate(&sys_ctx) != QNN_SUCCESS ||
        sys.systemContextGetBinaryInfo(sys_ctx, g_s.binary.data(), g_s.binary.size(),
                                       &info, &info_size) != QNN_SUCCESS) {
        LOG_ERR("%s: systemContextGetBinaryInfo failed", QNN_TAG);
        return nullptr;
    }

    const QnnSystemContext_GraphInfo_t * graphs = nullptr;
    uint32_t n_graphs = 0;
    if (info->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_3) {
        n_graphs = info->contextBinaryInfoV3.numGraphs;
        graphs   = info->contextBinaryInfoV3.graphs;
    } else if (info->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2) {
        n_graphs = info->contextBinaryInfoV2.numGraphs;
        graphs   = info->contextBinaryInfoV2.graphs;
    } else {
        n_graphs = info->contextBinaryInfoV1.numGraphs;
        graphs   = info->contextBinaryInfoV1.graphs;
    }
    if (n_graphs == 0) {
        LOG_ERR("%s: context binary contains no graph", QNN_TAG);
        return nullptr;
    }

    const QnnSystemContext_GraphInfoV1_t & gi = graphs[0].graphInfoV1;
    g_s.graph_name = gi.graphName ? gi.graphName : "";

    if (gi.numGraphInputs != 7 || gi.numGraphOutputs != 1) {
        LOG_ERR("%s: unexpected arity %u in / %u out, want 7 / 1",
                QNN_TAG, gi.numGraphInputs, gi.numGraphOutputs);
        return nullptr;
    }

    // fixed graph interface: image, position_ids, 4x merger_ds_idx, attn_bias
    const Qnn_Tensor_t * t_img  = find_input(gi, "image");
    const Qnn_Tensor_t * t_pos  = find_input(gi, "position_ids");
    const Qnn_Tensor_t * t_bias = find_input(gi, "attn_bias");
    if (!t_img || !t_pos || !t_bias) {
        LOG_ERR("%s: graph is missing image / position_ids / attn_bias", QNN_TAG);
        return nullptr;
    }

    const size_t n_image = static_cast<size_t>(3) * kCanvasW * kPatchSize;
    if (tensor_numel(*t_img) != n_image) {
        LOG_ERR("%s: image is %zu, want %zu", QNN_TAG, tensor_numel(*t_img), n_image);
        return nullptr;
    }

    g_s.in_image.assign(n_image, 0);
    g_s.in_pos.assign(kPatches, 0);
    g_s.in_bias.assign(kPatches, 0);
    for (int i = 0; i < 4; ++i) {
        g_s.in_ds[i].assign(kNTokens, 0);
    }
    g_s.out_buf.assign(static_cast<size_t>(kNTokens) * kNEmbd, 0);

    bind_tensor(g_s.in_tensors[0], *t_img, g_s.in_image.data(), g_s.in_image.size() * 2);
    bind_tensor(g_s.in_tensors[1], *t_pos, g_s.in_pos.data(), g_s.in_pos.size() * 4);
    bind_tensor(g_s.in_tensors[2], *t_bias, g_s.in_bias.data(), g_s.in_bias.size() * 2);
    for (int i = 0; i < 4; ++i) {
        const std::string name = "merger_ds_idx_" + std::to_string(i);
        const Qnn_Tensor_t * t_ds = find_input(gi, name.c_str());
        if (!t_ds) {
            LOG_ERR("%s: graph is missing %s", QNN_TAG, name.c_str());
            return nullptr;
        }
        bind_tensor(g_s.in_tensors[3 + i], *t_ds, g_s.in_ds[i].data(), g_s.in_ds[i].size() * 4);
    }

    const Qnn_Tensor_t & out = gi.graphOutputs[0];
    bind_tensor(g_s.out_tensor, out, g_s.out_buf.data(), g_s.out_buf.size() * 2);

    if (qnn.graphRetrieve(g_s.context, g_s.graph_name.c_str(), &g_s.graph) != QNN_SUCCESS) {
        LOG_ERR("%s: graphRetrieve('%s') failed", QNN_TAG, g_s.graph_name.c_str());
        return nullptr;
    }

    LOG_INF("%s: QNN vision ready (graph '%s', %d tokens)", QNN_TAG, g_s.graph_name.c_str(), kNTokens);
    return &g_s;
}

int clip_qnn_encode(const void * model, const float * buf, int nx, int ny, float * vec) {
    std::lock_guard<std::mutex> lock(g_s.mu);

    if (!model || !g_s.graph || !buf || !vec) {
        return -1;
    }

    const int pos_w = nx / kPatchSize;
    const int pos_h = ny / kPatchSize;
    if (pos_w <= 0 || pos_h <= 0 || pos_w * pos_h > kPatches) {
        LOG_ERR("%s: image %dx%d does not fit the %d-patch canvas", QNN_TAG, nx, ny, kPatches);
        return -1;
    }
    // clip_n_output_tokens() divides the patch count by n_merge^2, which only
    // agrees with the 2x2 gather when both sides are even. mtmd aligns crops to
    // 28 px (two patches), so this holds; refuse the odd case rather than let
    // the caller's embedding buffer disagree with what we write.
    if ((pos_w % 2) != 0 || (pos_h % 2) != 0) {
        LOG_ERR("%s: image %dx%d gives an odd patch grid (%dx%d)", QNN_TAG, nx, ny, pos_w, pos_h);
        return -1;
    }
    const int n_out = (pos_h / 2) * (pos_w / 2);
    if (n_out <= 0) {
        return -1;
    }

    // pack patches at native resolution, rest of the canvas stays zero
    {
        std::fill(g_s.in_image.begin(), g_s.in_image.end(), 0);
        const int canvas_w = kPatches * kPatchSize;
        const int n = canvas_w * kPatchSize;
        int patch_index = 0;
        for (int i = 0; i < pos_h && patch_index < kPatches; ++i) {
            for (int j = 0; j < pos_w && patch_index < kPatches; ++j) {
                for (int pi = 0; pi < kPatchSize; ++pi) {
                    for (int pj = 0; pj < kPatchSize; ++pj) {
                        const int src = ((i * kPatchSize + pi) * nx + (j * kPatchSize + pj)) * 3;
                        const int dst = canvas_w * pi + patch_index * kPatchSize + pj;
                        g_s.in_image[dst]           = f32_to_f16(buf[src]);
                        g_s.in_image[n + dst]       = f32_to_f16(buf[src + 1]);
                        g_s.in_image[2 * n + dst]   = f32_to_f16(buf[src + 2]);
                    }
                }
                ++patch_index;
            }
        }
    }

    // position_ids
    {
        int32_t bh[1024], bw[1024];
        for (int i = 0; i < pos_h; ++i) {
            bh[i] = static_cast<int32_t>(std::floor(70.0 * i / pos_h));
        }
        for (int i = 0; i < pos_w; ++i) {
            bw[i] = static_cast<int32_t>(std::floor(70.0 * i / pos_w));
        }
        int id = 0;
        for (int i = 0; i < pos_h; ++i) {
            for (int j = 0; j < pos_w; ++j) {
                g_s.in_pos[id++] = bh[i] * 70 + bw[j];
            }
        }
        for (; id < kPatches; ++id) {
            g_s.in_pos[id] = 0;
        }
    }

    // merger 2x2 gather
    {
        const int h2 = pos_h / 2;
        const int w2 = pos_w / 2;
        for (int o = 0; o < 4; ++o) {
            std::fill(g_s.in_ds[o].begin(), g_s.in_ds[o].end(), 0);
        }
        int k = 0;
        for (int i = 0; i < h2; ++i) {
            for (int j = 0; j < w2; ++j) {
                g_s.in_ds[0][k] = (2 * i)     * pos_w + (2 * j);
                g_s.in_ds[1][k] = (2 * i)     * pos_w + (2 * j + 1);
                g_s.in_ds[2][k] = (2 * i + 1) * pos_w + (2 * j);
                g_s.in_ds[3][k] = (2 * i + 1) * pos_w + (2 * j + 1);
                ++k;
            }
        }
    }

    // 0 for real patches, large negative for padding, otherwise the padding
    // still lands in every softmax denominator and shifts the real tokens
    {
        constexpr float kMasked = -3.0e4f;
        const int n_real = pos_w * pos_h;
        for (int i = 0; i < kPatches; ++i) {
            g_s.in_bias[i] = f32_to_f16(i < n_real ? 0.0f : kMasked);
        }
    }

    const Qnn_ErrorHandle_t rc = g_qnn->graphExecute(
        g_s.graph, g_s.in_tensors, 7, &g_s.out_tensor, 1, nullptr, nullptr);
    if (rc != QNN_SUCCESS) {
        LOG_ERR("%s: graphExecute failed: 0x%llx", QNN_TAG, static_cast<unsigned long long>(rc));
        return -1;
    }

    const size_t n_out_total = static_cast<size_t>(n_out) * kNEmbd;
    for (size_t i = 0; i < n_out_total; ++i) {
        vec[i] = f16_to_f32(g_s.out_buf[i]);
    }
    return 0;
}

void clip_qnn_closeModel(const void* model) {
    std::lock_guard<std::mutex> lock(g_s.mu);

    if (!model) {
        return;
    }

    g_s.graph   = nullptr;
    g_s.context = nullptr;
    g_s.device  = nullptr;
    g_s.backend = nullptr;
    g_qnn = nullptr;

    g_s.in_image.clear();
    g_s.in_pos.clear();
    for (auto & d : g_s.in_ds) {
        d.clear();
    }
    g_s.in_bias.clear();
    g_s.out_buf.clear();
    g_s.binary.clear();
    g_s.binary.shrink_to_fit();

    LOG_INF("%s: QNN vision released", QNN_TAG);
}

#else  // !(__ANDROID__ && ENABLE_QNN)

const void* clip_qnn_loadModel(const char*, const char*) {
    return nullptr;
}

int clip_qnn_encode(const void*, const float*, int, int, float*) {
    return -1;
}

void clip_qnn_closeModel(const void*) {}

#endif
