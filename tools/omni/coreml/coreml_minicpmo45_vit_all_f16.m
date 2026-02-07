//
// coreml_minicpmo45_vit_all_f16.m
//
// CoreML wrapper for MiniCPM-o 4.5 ViT+Resampler (ANE)
// Adapted from coreml_minicpmv40_vit_f16.m
//

#if !__has_feature(objc_arc)
#error This file must be compiled with automatic reference counting enabled (-fobjc-arc)
#endif

#import "coreml_minicpmo45_vit_all_f16.h"

@implementation coreml_minicpmo45_vit_all_f16Input

- (instancetype)initWithPixel_values:(MLMultiArray *)pixel_values position_ids:(MLMultiArray *)position_ids pos_embed_2d:(MLMultiArray *)pos_embed_2d {
    self = [super init];
    if (self) {
        _pixel_values = pixel_values;
        _position_ids = position_ids;
        _pos_embed_2d = pos_embed_2d;
    }
    return self;
}

- (NSSet<NSString *> *)featureNames {
    return [NSSet setWithArray:@[@"pixel_values", @"position_ids", @"pos_embed_2d"]];
}

- (nullable MLFeatureValue *)featureValueForName:(NSString *)featureName {
    if ([featureName isEqualToString:@"pixel_values"]) {
        return [MLFeatureValue featureValueWithMultiArray:self.pixel_values];
    }
    if ([featureName isEqualToString:@"position_ids"]) {
        return [MLFeatureValue featureValueWithMultiArray:self.position_ids];
    }
    if ([featureName isEqualToString:@"pos_embed_2d"]) {
        return [MLFeatureValue featureValueWithMultiArray:self.pos_embed_2d];
    }
    return nil;
}

@end

@implementation coreml_minicpmo45_vit_all_f16Output

- (instancetype)initWithOutput:(MLMultiArray *)output {
    self = [super init];
    if (self) {
        _output = output;
    }
    return self;
}

- (NSSet<NSString *> *)featureNames {
    return [NSSet setWithArray:@[@"output"]];
}

- (nullable MLFeatureValue *)featureValueForName:(NSString *)featureName {
    if ([featureName isEqualToString:@"output"]) {
        return [MLFeatureValue featureValueWithMultiArray:self.output];
    }
    return nil;
}

@end

@implementation coreml_minicpmo45_vit_all_f16

+ (nullable NSURL *)URLOfModelInThisBundle {
    NSString *assetPath = [[NSBundle bundleForClass:[self class]] pathForResource:@"coreml_minicpmo45_vit_all_f16" ofType:@"mlmodelc"];
    if (nil == assetPath) { os_log_error(OS_LOG_DEFAULT, "Could not load coreml_minicpmo45_vit_all_f16.mlmodelc in the bundle resource"); return nil; }
    return [NSURL fileURLWithPath:assetPath];
}

- (instancetype)initWithMLModel:(MLModel *)model {
    if (model == nil) {
        return nil;
    }
    self = [super init];
    if (self != nil) {
        _model = model;
    }
    return self;
}

- (nullable instancetype)init {
    return [self initWithContentsOfURL:(NSURL * _Nonnull)self.class.URLOfModelInThisBundle error:nil];
}

- (nullable instancetype)initWithConfiguration:(MLModelConfiguration *)configuration error:(NSError * _Nullable __autoreleasing * _Nullable)error {
    return [self initWithContentsOfURL:(NSURL * _Nonnull)self.class.URLOfModelInThisBundle configuration:configuration error:error];
}

- (nullable instancetype)initWithContentsOfURL:(NSURL *)modelURL error:(NSError * _Nullable __autoreleasing * _Nullable)error {
    MLModel *model = [MLModel modelWithContentsOfURL:modelURL error:error];
    if (model == nil) { return nil; }
    return [self initWithMLModel:model];
}

- (nullable instancetype)initWithContentsOfURL:(NSURL *)modelURL configuration:(MLModelConfiguration *)configuration error:(NSError * _Nullable __autoreleasing * _Nullable)error {
    MLModel *model = [MLModel modelWithContentsOfURL:modelURL configuration:configuration error:error];
    if (model == nil) { return nil; }
    return [self initWithMLModel:model];
}

+ (void)loadWithConfiguration:(MLModelConfiguration *)configuration completionHandler:(void (^)(coreml_minicpmo45_vit_all_f16 * _Nullable model, NSError * _Nullable error))handler {
    [self loadContentsOfURL:(NSURL * _Nonnull)[self URLOfModelInThisBundle]
              configuration:configuration
          completionHandler:handler];
}

+ (void)loadContentsOfURL:(NSURL *)modelURL configuration:(MLModelConfiguration *)configuration completionHandler:(void (^)(coreml_minicpmo45_vit_all_f16 * _Nullable model, NSError * _Nullable error))handler {
    [MLModel loadContentsOfURL:modelURL
                 configuration:configuration
             completionHandler:^(MLModel *model, NSError *error) {
        if (model != nil) {
            coreml_minicpmo45_vit_all_f16 *typedModel = [[coreml_minicpmo45_vit_all_f16 alloc] initWithMLModel:model];
            handler(typedModel, nil);
        } else {
            handler(nil, error);
        }
    }];
}

- (nullable coreml_minicpmo45_vit_all_f16Output *)predictionFromFeatures:(coreml_minicpmo45_vit_all_f16Input *)input error:(NSError * _Nullable __autoreleasing * _Nullable)error {
    return [self predictionFromFeatures:input options:[[MLPredictionOptions alloc] init] error:error];
}

- (nullable coreml_minicpmo45_vit_all_f16Output *)predictionFromFeatures:(coreml_minicpmo45_vit_all_f16Input *)input options:(MLPredictionOptions *)options error:(NSError * _Nullable __autoreleasing * _Nullable)error {
    id<MLFeatureProvider> outFeatures = [self.model predictionFromFeatures:input options:options error:error];
    if (!outFeatures) { return nil; }
    return [[coreml_minicpmo45_vit_all_f16Output alloc] initWithOutput:(MLMultiArray *)[outFeatures featureValueForName:@"output"].multiArrayValue];
}

- (nullable coreml_minicpmo45_vit_all_f16Output *)predictionFromPixel_values:(MLMultiArray *)pixel_values position_ids:(MLMultiArray *)position_ids pos_embed_2d:(MLMultiArray *)pos_embed_2d error:(NSError * _Nullable __autoreleasing * _Nullable)error {
    coreml_minicpmo45_vit_all_f16Input *input_ = [[coreml_minicpmo45_vit_all_f16Input alloc] initWithPixel_values:pixel_values position_ids:position_ids pos_embed_2d:pos_embed_2d];
    return [self predictionFromFeatures:input_ error:error];
}

- (nullable NSArray<coreml_minicpmo45_vit_all_f16Output *> *)predictionsFromInputs:(NSArray<coreml_minicpmo45_vit_all_f16Input*> *)inputArray options:(MLPredictionOptions *)options error:(NSError * _Nullable __autoreleasing * _Nullable)error {
    id<MLBatchProvider> inBatch = [[MLArrayBatchProvider alloc] initWithFeatureProviderArray:inputArray];
    id<MLBatchProvider> outBatch = [self.model predictionsFromBatch:inBatch options:options error:error];
    if (!outBatch) { return nil; }
    NSMutableArray<coreml_minicpmo45_vit_all_f16Output*> *results = [NSMutableArray arrayWithCapacity:(NSUInteger)outBatch.count];
    for (NSInteger i = 0; i < outBatch.count; i++) {
        id<MLFeatureProvider> resultProvider = [outBatch featuresAtIndex:i];
        coreml_minicpmo45_vit_all_f16Output * result = [[coreml_minicpmo45_vit_all_f16Output alloc] initWithOutput:(MLMultiArray *)[resultProvider featureValueForName:@"output"].multiArrayValue];
        [results addObject:result];
    }
    return results;
}

@end
