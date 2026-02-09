//
// omni_coreml.h
//
// C bridge header for CoreML vision encoder (MiniCPM-o 4.5)
// Adapted from mtmd_coreml.h
//

#if __cplusplus
extern "C" {
#endif

const void* omni_coreml_loadModel(const char* model_path);
void omni_coreml_closeModel(const void* model);
void omni_coreml_predictWith(const void* model, float* pixel_values, int32_t* position_ids, float* pos_embed_2d, float* encoderOutput);

#if __cplusplus
}   // Extern C
#endif
