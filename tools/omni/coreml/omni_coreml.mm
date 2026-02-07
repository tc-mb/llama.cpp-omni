//
// omni_coreml.mm
//
// CoreML bridge implementation for MiniCPM-o 4.5 ViT+Resampler (ANE)
// Adapted from mtmd_coreml.mm
//
// Input shapes (O4.5):
//   pixel_values:  1 × 3 × 14 × 14336 (float32)
//   position_ids:  1 × 1024 (int32)
//   pos_embed_2d:  1024 × 1 × 4096 (float32)
// Output shape:
//   output:        1 × 64 × 4096 (float32)
//

#import <CoreML/CoreML.h>
#import <Accelerate/Accelerate.h>
#import "omni_coreml.h"
#import "coreml_minicpmo45_vit_all_f16.h"
#include <stdlib.h>

#if __cplusplus
extern "C" {
#endif

const void* omni_coreml_loadModel(const char* model_path) {
    if (!model_path) {
        NSLog(@"Error: model_path is null");
        return nullptr;
    }

    NSString *pathString = [NSString stringWithUTF8String:model_path];

    // Check if file exists
    NSFileManager *fileManager = [NSFileManager defaultManager];
    if (![fileManager fileExistsAtPath:pathString]) {
        NSLog(@"Error: CoreML model file does not exist at path: %@", pathString);
        return nullptr;
    }

    // Check if it's a directory (for .mlmodelc packages)
    BOOL isDirectory;
    if ([fileManager fileExistsAtPath:pathString isDirectory:&isDirectory]) {
        if (!isDirectory && ![pathString hasSuffix:@".mlmodelc"]) {
            NSLog(@"Warning: CoreML model path should typically be a .mlmodelc directory: %@", pathString);
        }
    }

    NSURL *modelURL = [NSURL fileURLWithPath:pathString];

    NSError *error = nil;
    const void* model = CFBridgingRetain([[coreml_minicpmo45_vit_all_f16 alloc] initWithContentsOfURL:modelURL error:&error]);

    if (error) {
        NSLog(@"Error loading CoreML model: %@", error.localizedDescription);
        return nullptr;
    }

    if (!model) {
        NSLog(@"Error: Failed to create CoreML model instance");
        return nullptr;
    }

    NSLog(@"Successfully loaded CoreML model (MiniCPM-o 4.5 ViT+Resampler) from: %@", pathString);
    return model;
}

void omni_coreml_predictWith(const void* model, float* pixel_values, int32_t* position_ids, float* pos_embed_2d, float* encoderOutput) {
    // pixel_values: shape [1,3,14,14336], float32
    MLMultiArray *pixelMA = [[MLMultiArray alloc] initWithDataPointer: pixel_values
                                                                 shape: @[@1, @3, @14, @14336]
                                                              dataType: MLMultiArrayDataTypeFloat32
                                                               strides: @[@(602112), @(200704), @(14336), @(1)]
                                                           deallocator: nil
                                                                 error: nil];

    // position_ids: shape [1,1024], int32
    MLMultiArray *posIdsMA = [[MLMultiArray alloc] initWithDataPointer: position_ids
                                                                  shape: @[@1, @1024]
                                                               dataType: MLMultiArrayDataTypeInt32
                                                                strides: @[@(1024), @(1)]
                                                            deallocator: nil
                                                                  error: nil];

    // pos_embed_2d: shape [1024,1,4096], float32
    MLMultiArray *posEmbedMA = [[MLMultiArray alloc] initWithDataPointer: pos_embed_2d
                                                                   shape: @[@1024, @1, @4096]
                                                                dataType: MLMultiArrayDataTypeFloat32
                                                                 strides: @[@(4096), @(4096), @(1)]
                                                             deallocator: nil
                                                                   error: nil];

    NSError *error = nil;
    coreml_minicpmo45_vit_all_f16Output *modelOutput = [(__bridge coreml_minicpmo45_vit_all_f16 *)model predictionFromPixel_values:pixelMA position_ids:posIdsMA pos_embed_2d:posEmbedMA error:&error];

    if (!modelOutput || error) {
        NSLog(@"CoreML prediction failed: %@", error.localizedDescription);
        return;
    }

    MLMultiArray *outMA = modelOutput.output;
    cblas_scopy((int)outMA.count, (float*)outMA.dataPointer, 1, encoderOutput, 1);
}

void omni_coreml_closeModel(const void* model) {
    CFRelease(model);
}

#if __cplusplus
} //Extern C
#endif
