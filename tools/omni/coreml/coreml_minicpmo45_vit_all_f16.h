//
// coreml_minicpmo45_vit_all_f16.h
//
// CoreML wrapper for MiniCPM-o 4.5 ViT+Resampler (ANE)
// Adapted from coreml_minicpmv40_vit_f16.h
//
// Input shapes:
//   pixel_values:  1 × 3 × 14 × 14336 (float32)
//   position_ids:  1 × 1024 (int32)
//   pos_embed_2d:  1024 × 1 × 4096 (float32)
// Output shape:
//   output:        1 × 64 × 4096 (float32)
//

#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
#include <stdint.h>
#include <os/log.h>

NS_ASSUME_NONNULL_BEGIN

/// Model Prediction Input Type
API_AVAILABLE(macos(12.0), ios(15.0), watchos(8.0), tvos(15.0)) __attribute__((visibility("hidden")))
@interface coreml_minicpmo45_vit_all_f16Input : NSObject<MLFeatureProvider>

/// pixel_values as 1 × 3 × 14 × 14336 4-dimensional array of floats
@property (readwrite, nonatomic, strong) MLMultiArray * pixel_values;

/// position_ids as 1 by 1024 matrix of 32-bit integers
@property (readwrite, nonatomic, strong) MLMultiArray * position_ids;

/// pos_embed_2d as 1024 × 1 × 4096 3-dimensional array of floats
@property (readwrite, nonatomic, strong) MLMultiArray * pos_embed_2d;
- (instancetype)init NS_UNAVAILABLE;
- (instancetype)initWithPixel_values:(MLMultiArray *)pixel_values position_ids:(MLMultiArray *)position_ids pos_embed_2d:(MLMultiArray *)pos_embed_2d NS_DESIGNATED_INITIALIZER;

@end

/// Model Prediction Output Type
API_AVAILABLE(macos(12.0), ios(15.0), watchos(8.0), tvos(15.0)) __attribute__((visibility("hidden")))
@interface coreml_minicpmo45_vit_all_f16Output : NSObject<MLFeatureProvider>

/// output as 1 × 64 × 4096 3-dimensional array of floats
@property (readwrite, nonatomic, strong) MLMultiArray * output;
- (instancetype)init NS_UNAVAILABLE;
- (instancetype)initWithOutput:(MLMultiArray *)output NS_DESIGNATED_INITIALIZER;

@end

/// Class for model loading and prediction
API_AVAILABLE(macos(12.0), ios(15.0), watchos(8.0), tvos(15.0)) __attribute__((visibility("hidden")))
@interface coreml_minicpmo45_vit_all_f16 : NSObject
@property (readonly, nonatomic, nullable) MLModel * model;

+ (nullable NSURL *)URLOfModelInThisBundle;

- (instancetype)initWithMLModel:(MLModel *)model NS_DESIGNATED_INITIALIZER;

- (nullable instancetype)init;

- (nullable instancetype)initWithConfiguration:(MLModelConfiguration *)configuration error:(NSError * _Nullable __autoreleasing * _Nullable)error;

- (nullable instancetype)initWithContentsOfURL:(NSURL *)modelURL error:(NSError * _Nullable __autoreleasing * _Nullable)error;

- (nullable instancetype)initWithContentsOfURL:(NSURL *)modelURL configuration:(MLModelConfiguration *)configuration error:(NSError * _Nullable __autoreleasing * _Nullable)error;

+ (void)loadWithConfiguration:(MLModelConfiguration *)configuration completionHandler:(void (^)(coreml_minicpmo45_vit_all_f16 * _Nullable model, NSError * _Nullable error))handler;

+ (void)loadContentsOfURL:(NSURL *)modelURL configuration:(MLModelConfiguration *)configuration completionHandler:(void (^)(coreml_minicpmo45_vit_all_f16 * _Nullable model, NSError * _Nullable error))handler;

- (nullable coreml_minicpmo45_vit_all_f16Output *)predictionFromFeatures:(coreml_minicpmo45_vit_all_f16Input *)input error:(NSError * _Nullable __autoreleasing * _Nullable)error;

- (nullable coreml_minicpmo45_vit_all_f16Output *)predictionFromFeatures:(coreml_minicpmo45_vit_all_f16Input *)input options:(MLPredictionOptions *)options error:(NSError * _Nullable __autoreleasing * _Nullable)error;

- (nullable coreml_minicpmo45_vit_all_f16Output *)predictionFromPixel_values:(MLMultiArray *)pixel_values position_ids:(MLMultiArray *)position_ids pos_embed_2d:(MLMultiArray *)pos_embed_2d error:(NSError * _Nullable __autoreleasing * _Nullable)error;

- (nullable NSArray<coreml_minicpmo45_vit_all_f16Output *> *)predictionsFromInputs:(NSArray<coreml_minicpmo45_vit_all_f16Input*> *)inputArray options:(MLPredictionOptions *)options error:(NSError * _Nullable __autoreleasing * _Nullable)error;
@end

NS_ASSUME_NONNULL_END
