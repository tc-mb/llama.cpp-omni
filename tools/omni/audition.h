#pragma once

#include "ggml.h"
#include "gguf.h"

#include <stddef.h>
#include <stdint.h>

#include <climits>
#include <cstdarg>
#include <cinttypes>
#include <string>
#include <map>
#include <sstream>
#include <vector>
#include <memory>

//
// tensor name constants
//

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

static void audition_log_callback_default(enum ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    (void) user_data;
    fputs(text, stderr);
    fflush(stderr);
}

//from mtmd-audio.h
#define WHISPER_ASSERT GGML_ASSERT

#define WHISPER_SAMPLE_RATE 16000
#define WHISPER_N_FFT       400
#define WHISPER_HOP_LENGTH  160
#define WHISPER_CHUNK_SIZE  30

#define COMMON_SAMPLE_RATE 16000

namespace whisper_preprocessor {

struct whisper_mel {
    int n_len;
    int n_len_org;
    int n_mel;

    std::vector<float> data;
};

struct whisper_filters {
    int32_t n_mel;
    int32_t n_fft;

    std::vector<float> data;
};

bool preprocess_audio(
        const float * samples,
        size_t n_samples,
        const whisper_filters & filters,
        std::vector<whisper_mel> & output);

} // namespace whisper_preprocessor

struct audition_ctx;

// uint8 audio
struct audition_audio_u8 {
    int nx;
    int ny;

    std::vector<uint8_t> buf;
};

// For audio, only one channel is used, buf.size() == nx*ny
//     nx will be n_frames and ny will be n_mel
struct audition_audio_f32 {
    int nx;
    int ny;

    std::vector<float> buf;
};

void audition_audio_u8_free(struct audition_audio_u8 * audio);
void audition_audio_f32_free(struct audition_audio_f32 * audio);
void audition_audio_f32_batch_free(struct audition_audio_f32_batch * batch);

// wrapper for audition_audio_f32
struct audition_audio_f32_deleter {
    void operator()(audition_audio_f32 * val) { audition_audio_f32_free(val); }
};
typedef std::unique_ptr<audition_audio_f32, audition_audio_f32_deleter> audition_audio_f32_ptr;

struct audition_audio_f32_batch {
    std::vector<audition_audio_f32_ptr> entries;
    bool is_audio = true;

    int grid_x = 0;
    int grid_y = 0;

    audition_audio_f32_batch clone() const {
        audition_audio_f32_batch new_batch{
            /* entries  */ {},
            /* is_audio */ is_audio,
            /* grid_x   */ grid_x,
            /* grid_y   */ grid_y,
        };
        new_batch.entries.reserve(entries.size());
        for (const auto & entry : entries) {
            new_batch.entries.emplace_back(new audition_audio_f32(*entry));
        }
        return new_batch;
    }
};

struct audition_context_params {
    bool use_gpu;
    enum ggml_log_level verbosity;
};

struct audition_ctx * audition_init(const char * fname, struct audition_context_params ctx_params);

void audition_free(struct audition_ctx * ctx);

int audition_n_output_tokens(const struct audition_ctx * ctx, struct audition_audio_f32 * audio);

// get mel filters data from the audio model
whisper_preprocessor::whisper_filters audition_get_mel_filters(const struct audition_ctx * ctx);

// this should be equal to the embedding dimension of the text model
int audition_n_mmproj_embd(const struct audition_ctx * ctx);

struct audition_audio_u8    * audition_audio_u8_init(void);
struct audition_audio_f32   * audition_audio_f32_init(void);
struct audition_audio_f32_batch * audition_audio_f32_batch_init(void);

bool audition_audio_preprocess  (struct audition_ctx * ctx, struct audition_audio_u8 * audio, struct audition_audio_f32 ** out_mel);
bool audition_audio_encode      (struct audition_ctx * ctx, int n_threads, struct audition_audio_f32 * audio, float * vec);
bool audition_audio_batch_encode(struct audition_ctx * ctx, int n_threads, const struct audition_audio_f32_batch * audios, float * vec);

void audition_whisper_init_kv_cache(struct audition_ctx * ctx, int n_state, int n_layer, int n_audio_ctx);

// Free KV cache
void audition_whisper_free_kv_cache(struct audition_ctx * ctx);

// Clear KV cache and reset iteration counter (call before processing new audio sequence)
void audition_whisper_clear_kv_cache(struct audition_ctx * ctx);
