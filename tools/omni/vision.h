#pragma once

#include "ggml.h"

#include <stddef.h>
#include <stdint.h>
#include <vector>
#include <memory>

struct vision_ctx;

struct vision_image_size {
    int width;
    int height;
};

// RGB uint8 image
struct vision_image_u8 {
    int nx;
    int ny;

    std::vector<uint8_t> buf;
};

// For images, buf.size() == nx*ny*3
//     Memory layout: RGBRGBRGB...
// For audio, only one channel is used, buf.size() == nx*ny
//     nx will be n_frames and ny will be n_mel
struct vision_image_f32 {
    int nx;
    int ny;

    std::vector<float> buf;
};

// forward declarations for deleters
void vision_image_u8_free(struct vision_image_u8  * img);
void vision_image_f32_free(struct vision_image_f32 * img);

struct vision_image_u8_deleter {
    void operator()(vision_image_u8 * val) { vision_image_u8_free(val); }
};
typedef std::unique_ptr<vision_image_u8, vision_image_u8_deleter> vision_image_u8_ptr;
struct vision_image_u8_batch {
    std::vector<vision_image_u8_ptr> entries;
};

struct vision_image_f32_deleter {
    void operator()(vision_image_f32 * val) { vision_image_f32_free(val); }
};
typedef std::unique_ptr<vision_image_f32, vision_image_f32_deleter> vision_image_f32_ptr;
struct vision_image_f32_batch {
    std::vector<vision_image_f32_ptr> entries;

    // for llava-uhd style models, we need to know the grid size
    // note: entries.size() == grid_x * grid_y + 1 (one overview image)
    int grid_x = 0;
    int grid_y = 0;

    vision_image_f32_batch clone() const {
        vision_image_f32_batch new_batch{
            /* entries  */ {},
            /* grid_x   */ grid_x,
            /* grid_y   */ grid_y,
        };
        new_batch.entries.reserve(entries.size());
        for (const auto & entry : entries) {
            new_batch.entries.emplace_back(new vision_image_f32(*entry));
        }
        return new_batch;
    }
};

struct vision_context_params {
    bool use_gpu;
    enum ggml_log_level verbosity;
    const char * coreml_model_path; // path to CoreML model (.mlmodelc) for ANE acceleration
};


//
// vision functions
//

// vision layers
struct vision_ctx * vision_init(const char * fname, struct vision_context_params ctx_params);
void vision_free(struct vision_ctx * ctx);

// vision processor
void vision_image_u8_free(struct vision_image_u8  * img);
void vision_image_f32_free(struct vision_image_f32 * img);
struct vision_image_u8        * vision_image_u8_init (void);
struct vision_image_f32       * vision_image_f32_init(void);
struct vision_image_f32_batch * vision_image_f32_batch_init(void);

// vision query
int vision_n_output_tokens(const struct vision_ctx * ctx);
int vision_n_mmproj_embd(const struct vision_ctx * ctx);

// vision preprocess
bool vision_image_preprocess(struct vision_ctx * ctx, const struct vision_image_u8 * img, struct vision_image_f32_batch * res_imgs );

// vision forward
bool vision_image_encode      (struct vision_ctx * ctx, int n_threads, struct vision_image_f32 * img, float * vec);
bool vision_image_batch_encode(struct vision_ctx * ctx, int n_threads, const struct vision_image_f32_batch * imgs, float * vec);

// CoreML / ANE support
void vision_set_coreml_model_path(struct vision_ctx * ctx, const char * coreml_model_path);
bool vision_coreml_warmup(struct vision_ctx * ctx);
bool vision_image_batch_encode_coreml(struct vision_ctx * ctx, const struct vision_image_f32_batch * imgs, float * vec);
