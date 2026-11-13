//
// clip_qnn.h
//
// C bridge header for QNN vision encoder (MiniCPM-OCR 4x ViT)
//

#pragma once

#include <stdint.h>

#if __cplusplus
extern "C" {
#endif

const void * clip_qnn_loadModel(const char * model_path, const char * skel_dir);
void clip_qnn_closeModel(const void * model);

// Encodes one preprocessed image. buf is nx*ny*3 floats, interleaved RGB,
// already normalised to [-1, 1]. Returns 0 on success.
int clip_qnn_encode(const void * model, const float * buf, int nx, int ny, float * vec);

#if __cplusplus
}
#endif
