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
#include <chrono>
#include <cstdio>

#if __cplusplus
extern "C" {
#endif

const void* omni_coreml_loadModel(const char* model_path) {
    if (!model_path) {
        NSLog(@"Error: model_path is null");
        return nullptr;
    }

    NSString *pathString = [NSString stringWithUTF8String:model_path];

    NSFileManager *fileManager = [NSFileManager defaultManager];
    if (![fileManager fileExistsAtPath:pathString]) {
        NSLog(@"Error: CoreML model file does not exist at path: %@", pathString);
        return nullptr;
    }

    BOOL isDirectory;
    if ([fileManager fileExistsAtPath:pathString isDirectory:&isDirectory]) {
        if (!isDirectory && ![pathString hasSuffix:@".mlmodelc"]) {
            NSLog(@"Warning: CoreML model path should typically be a .mlmodelc directory: %@", pathString);
        }
    }

    NSURL *modelURL = [NSURL fileURLWithPath:pathString];

    // 显式指定 compute units：
    // 默认 MLComputeUnitsAll 会让 CoreML 在 ANE/GPU/CPU 之间自动分图，
    // 子图可能仍然 fallback 到 GPU，从而和 LLM 抢 Metal 资源。
    // 我们的目标是让 VPM 完全跑在 ANE 上、彻底解耦 GPU。
    // OMNI_COREML_COMPUTE_UNITS 环境变量可覆盖：
    //   "ane"  -> CPUAndNeuralEngine (默认)
    //   "all"  -> All (ANE+GPU+CPU)
    //   "cpu_gpu" -> CPUAndGPU
    //   "cpu" -> CPUOnly
    MLModelConfiguration * config = [[MLModelConfiguration alloc] init];
    const char * cu_env = getenv("OMNI_COREML_COMPUTE_UNITS");
    NSString * cu_str = cu_env ? [NSString stringWithUTF8String:cu_env] : @"ane";
    if ([cu_str isEqualToString:@"all"]) {
        config.computeUnits = MLComputeUnitsAll;
    } else if ([cu_str isEqualToString:@"cpu_gpu"]) {
        config.computeUnits = MLComputeUnitsCPUAndGPU;
    } else if ([cu_str isEqualToString:@"cpu"]) {
        config.computeUnits = MLComputeUnitsCPUOnly;
    } else {
        // 默认 / "ane"：尽量上 ANE
        config.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
    }
    NSLog(@"CoreML compute units: %@", cu_str);

    NSError *error = nil;
    const void* model = CFBridgingRetain([[coreml_minicpmo45_vit_all_f16 alloc] initWithContentsOfURL:modelURL configuration:config error:&error]);

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
    static int s_prof_enabled = -1;
    if (s_prof_enabled < 0) {
        s_prof_enabled = (getenv("OMNI_COREML_PROF") != NULL) ? 1 : 0;
    }
    auto t0 = std::chrono::high_resolution_clock::now();

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

    auto t_pre = std::chrono::high_resolution_clock::now();

    NSError *error = nil;
    coreml_minicpmo45_vit_all_f16Output *modelOutput = [(__bridge coreml_minicpmo45_vit_all_f16 *)model predictionFromPixel_values:pixelMA position_ids:posIdsMA pos_embed_2d:posEmbedMA error:&error];

    auto t_pred = std::chrono::high_resolution_clock::now();

    if (!modelOutput || error) {
        NSLog(@"CoreML prediction failed: %@", error.localizedDescription);
        return;
    }

    MLMultiArray *outMA = modelOutput.output;
    cblas_scopy((int)outMA.count, (float*)outMA.dataPointer, 1, encoderOutput, 1);

    auto t1 = std::chrono::high_resolution_clock::now();

    if (s_prof_enabled) {
        double ms_wrap   = std::chrono::duration<double, std::milli>(t_pre  - t0).count();
        double ms_predict= std::chrono::duration<double, std::milli>(t_pred - t_pre).count();
        double ms_post   = std::chrono::duration<double, std::milli>(t1     - t_pred).count();
        double ms_total  = std::chrono::duration<double, std::milli>(t1     - t0).count();
        fprintf(stderr, "[prof] coreml VPM total=%.1fms (wrap=%.1f predict=%.1f post=%.1f)\n",
                ms_total, ms_wrap, ms_predict, ms_post);
    }
}

void omni_coreml_closeModel(const void* model) {
    CFRelease(model);
}

#if __cplusplus
} //Extern C
#endif
