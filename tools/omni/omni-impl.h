#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "vision.h"
#include "audition.h"

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cinttypes>
#include <fstream>
#include <sstream>
#include <vector>
#include <set>
#include <string>
#include <memory>

#define KEY_FTYPE               "general.file_type"
#define KEY_NAME                "general.name"
#define KEY_DESCRIPTION         "general.description"
#define KEY_MODEL_TYPE          "general.model_type"
#define KEY_HAS_AUDIO_ENC       "clip.has_audio_encoder"
#define KEY_HAS_VISION_ENC      "clip.has_vision_encoder"
#define KEY_USE_GELU            "clip.use_gelu"
#define KEY_USE_SILU            "clip.use_silu"

#define KEY_N_EMBD              "clip.%s.embedding_length"
#define KEY_N_FF                "clip.%s.feed_forward_length"
#define KEY_N_BLOCK             "clip.%s.block_count"
#define KEY_PROJ_DIM            "clip.%s.projection_dim"
#define KEY_N_HEAD              "clip.%s.attention.head_count"
#define KEY_LAYER_NORM_EPS      "clip.%s.attention.layer_norm_epsilon"

// vision-specific
#define KEY_IMAGE_SIZE          "clip.vision.image_size"
#define KEY_PREPROC_IMAGE_SIZE  "clip.vision.preproc_image_size"
#define KEY_PATCH_SIZE          "clip.vision.patch_size"
#define KEY_IMAGE_MEAN          "clip.vision.image_mean"
#define KEY_IMAGE_STD           "clip.vision.image_std"
#define KEY_FEATURE_LAYER       "clip.vision.feature_layer"
#define KEY_PROJ_SCALE_FACTOR   "clip.vision.projector.scale_factor"
#define KEY_SPATIAL_MERGE_SIZE  "clip.vision.spatial_merge_size"

#define KEY_MM_PATCH_MERGE_TYPE   "clip.vision.mm_patch_merge_type"
#define KEY_IMAGE_GRID_PINPOINTS  "clip.vision.image_grid_pinpoints"
#define KEY_IMAGE_CROP_RESOLUTION "clip.vision.image_crop_resolution"
#define KEY_WIN_ATTN_PATTERN      "clip.vision.n_wa_pattern"
#define KEY_ATTN_WINDOW_SIZE      "clip.vision.window_size"
#define KEY_MINICPMV_VERSION      "clip.minicpmv_version"
#define KEY_MINICPMV_QUERY_NUM    "clip.minicpmv_query_num"

// audio-specific
#define KEY_A_NUM_MEL_BINS      "clip.audio.num_mel_bins"
#define KEY_A_PROJ_STACK_FACTOR "clip.audio.projector.stack_factor"

#define TN_POS_EMBD        "%s.position_embd.weight"
#define TN_CLASS_EMBD      "v.class_embd"
#define TN_PATCH_EMBD      "v.patch_embd.weight"  // not rename tensor with ".0" postfix for backwrad compat
#define TN_PATCH_EMBD_1    "v.patch_embd.weight.1"
#define TN_PATCH_BIAS      "v.patch_embd.bias"
#define TN_ATTN_K          "%s.blk.%d.attn_k.%s"
#define TN_ATTN_Q          "%s.blk.%d.attn_q.%s"
#define TN_ATTN_V          "%s.blk.%d.attn_v.%s"
#define TN_ATTN_OUTPUT     "%s.blk.%d.attn_out.%s"
#define TN_ATTN_K_NORM     "%s.blk.%d.attn_k_norm.%s"
#define TN_ATTN_Q_NORM     "%s.blk.%d.attn_q_norm.%s"
#define TN_FFN_DOWN        "%s.blk.%d.ffn_down.%s"
#define TN_FFN_GATE        "%s.blk.%d.ffn_gate.%s"
#define TN_FFN_UP          "%s.blk.%d.ffn_up.%s"
#define TN_FFN_GATE        "%s.blk.%d.ffn_gate.%s"
#define TN_LN_1            "%s.blk.%d.ln1.%s" // layer norm
#define TN_LN_2            "%s.blk.%d.ln2.%s" // layer norm
#define TN_LS_1            "%s.blk.%d.ls1.%s" // layer scale
#define TN_LS_2            "%s.blk.%d.ls2.%s" // layer scale
#define TN_LN_PRE          "%s.pre_ln.%s"
#define TN_LN_POST         "%s.post_ln.%s"
#define TN_LLAVA_PROJ      "mm.%d.%s"
#define TN_MVLM_PROJ_MLP   "mm.model.mlp.%d.%s"
#define TN_MVLM_PROJ_BLOCK "mm.model.mb_block.%d.block.%d.%s"
#define TN_MVLM_PROJ_PEG   "mm.model.peg.%d.%s"
#define TN_IMAGE_NEWLINE   "model.image_newline"
#define TN_MM_INP_NORM     "mm.input_norm.weight"
#define TN_MM_INP_NORM_B   "mm.input_norm.bias"
#define TN_MM_INP_PROJ     "mm.input_projection.weight" // gemma3
#define TN_MM_SOFT_EMB_N   "mm.soft_emb_norm.weight"    // gemma3
#define TN_MM_PROJECTOR    "mm.model.fc.weight"         // idefics3
#define TN_MM_PATCH_MERGER "mm.patch_merger.weight"     // mistral small 3.1
#define TN_TOK_IMG_BREAK   "v.token_embd.img_break"     // pixtral
#define TN_TOK_GLM_BOI     "adapter.boi"                // glm-edge (these embeddings are not in text model)
#define TN_TOK_GLM_EOI     "adapter.eoi"                // glm-edge (these embeddings are not in text model)

// mimicpmv
#define TN_MINICPMV_POS_EMBD_K "resampler.pos_embed_k"
#define TN_MINICPMV_QUERY      "resampler.query"
#define TN_MINICPMV_PROJ       "resampler.proj.weight"
#define TN_MINICPMV_KV_PROJ    "resampler.kv.weight"
#define TN_MINICPMV_ATTN       "resampler.attn.%s.%s"
#define TN_MINICPMV_LN         "resampler.ln_%s.%s"

#define TN_GLM_ADAPER_CONV      "adapter.conv.%s"
#define TN_GLM_ADAPTER_LINEAR   "adapter.linear.linear.%s"
#define TN_GLM_ADAPTER_NORM_1   "adapter.linear.norm1.%s"
#define TN_GLM_ADAPTER_D_H_2_4H "adapter.linear.dense_h_to_4h.%s"
#define TN_GLM_ADAPTER_GATE     "adapter.linear.gate.%s"
#define TN_GLM_ADAPTER_D_4H_2_H "adapter.linear.dense_4h_to_h.%s"

// align x to upper multiple of n
#define VISION_ALIGN(x, n) ((((x) + (n) - 1) / (n)) * (n))

enum omni_model_type {
    MiniCPM_o,
    OMNI_MODEL_TYPE_UNKNOWN,
};

static std::map<omni_model_type, std::string> OMNI_MODEL_TYPE_NAMES = {
    { MiniCPM_o, "MiniCPM-o" },
};

static omni_model_type omni_model_type_from_string(const std::string & str) {
    for (const auto & pair : OMNI_MODEL_TYPE_NAMES) {
        if (pair.second == str) {
            return pair.first;
        }
    }
    return OMNI_MODEL_TYPE_UNKNOWN;
}

//
// logging
//
static void omni_log_callback_default(enum ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    (void) user_data;
    fputs(text, stderr);
    fflush(stderr);
}

struct omni_logger_state {
    ggml_log_level verbosity_thold;
    ggml_log_callback log_callback;
    void * log_callback_user_data;
};

extern struct omni_logger_state g_logger_state;

static void omni_log_internal_v(enum ggml_log_level level, const char * format, va_list args) {
    if (format == NULL) {
        return;
    }
    va_list args_copy;
    va_copy(args_copy, args);
    char buffer[128];
    int len = vsnprintf(buffer, 128, format, args);
    if (len < 128) {
        g_logger_state.log_callback(level, buffer, g_logger_state.log_callback_user_data);
    } else {
        char * buffer2 = (char *) calloc(len + 1, sizeof(char));
        vsnprintf(buffer2, len + 1, format, args_copy);
        buffer2[len] = 0;
        g_logger_state.log_callback(level, buffer2, g_logger_state.log_callback_user_data);
        free(buffer2);
    }
    va_end(args_copy);
}

static void omni_log_internal(enum ggml_log_level level, const char * format, ...) {
    va_list args;
    va_start(args, format);
    omni_log_internal_v(level, format, args);
    va_end(args);
}

#define LOG_TMPL(level, ...) \
    do { \
        if ((level) >= g_logger_state.verbosity_thold) { \
            omni_log_internal((level), __VA_ARGS__); \
        } \
    } while (0)
#define LOG_INF(...) LOG_TMPL(GGML_LOG_LEVEL_INFO,  __VA_ARGS__)
#define LOG_WRN(...) LOG_TMPL(GGML_LOG_LEVEL_WARN,  __VA_ARGS__)
#define LOG_ERR(...) LOG_TMPL(GGML_LOG_LEVEL_ERROR, __VA_ARGS__)
#define LOG_DBG(...) LOG_TMPL(GGML_LOG_LEVEL_DEBUG, __VA_ARGS__)
#define LOG_CNT(...) LOG_TMPL(GGML_LOG_LEVEL_CONT,  __VA_ARGS__)


//
// common utils
//

static std::string string_format(const char * fmt, ...) {
    va_list ap;
    va_list ap2;
    va_start(ap, fmt);
    va_copy(ap2, ap);
    int size = vsnprintf(NULL, 0, fmt, ap);
    GGML_ASSERT(size >= 0 && size < INT_MAX); // NOLINT
    std::vector<char> buf(size + 1);
    int size2 = vsnprintf(buf.data(), size + 1, fmt, ap2);
    GGML_ASSERT(size2 == size);
    va_end(ap2);
    va_end(ap);
    return std::string(buf.data(), buf.size());
}




