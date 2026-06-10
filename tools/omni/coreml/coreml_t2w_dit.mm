// coreml_t2w_dit.mm
//
// Token2Mel DiT estimator → CoreML 桥接实现。
//
// 不依赖 Xcode 自动生成的 typed wrapper class（model 输入太多、有 stateful
// streaming 形状），用通用 MLModel + MLDictionaryFeatureProvider。

#import <CoreML/CoreML.h>
#import <Accelerate/Accelerate.h>
#import "coreml_t2w_dit.h"
#include <chrono>
#include <cstdio>
#include <cstring>

#if __cplusplus
extern "C" {
#endif

// ----------------------------------------------------------------------------
// 内部 handle 结构
// ----------------------------------------------------------------------------

typedef struct t2w_dit_ctx {
    MLModel * __strong         model;
    t2w_dit_dims_t             dims;
    bool                       prof_enabled;
    int64_t                    n_calls;
    double                     ms_total;
    double                     ms_predict;
} t2w_dit_ctx_t;

// 默认 dims（与 ane_export_dit.py 写死的常量一致；model metadata 也存了一份，
// 后面 t2w_dit_load 会从 metadata 覆盖以校验一致性）。
static void fill_default_dims(t2w_dit_dims_t * d) {
    d->depth              = 16;
    d->batch              = 2;
    d->chunk_size         = 56;
    d->max_cache_len      = 600;
    d->cnn_cache_channels = 1024;
    d->cnn_cache_t        = 2;
    d->num_heads          = 8;
    d->head_dim           = 64;
    d->mel_channels       = 80;
    d->spk_dim            = 80;
}

static int parse_int_md(NSDictionary * md, NSString * key, int fallback) {
    id v = md[key];
    if ([v isKindOfClass:[NSString class]]) {
        return [(NSString *) v intValue];
    }
    return fallback;
}

// ----------------------------------------------------------------------------
// load
// ----------------------------------------------------------------------------

t2w_dit_handle_t t2w_dit_load(const char * model_path) {
    @autoreleasepool {
        if (!model_path) {
            NSLog(@"[t2w_dit] error: model_path is null");
            return nullptr;
        }
        NSString * pathStr = [NSString stringWithUTF8String:model_path];
        NSFileManager * fm = [NSFileManager defaultManager];
        if (![fm fileExistsAtPath:pathStr]) {
            NSLog(@"[t2w_dit] error: file not found: %@", pathStr);
            return nullptr;
        }
        NSURL * url = [NSURL fileURLWithPath:pathStr];
        NSError * err = nil;

        // C++ 端 CoreML 不像 Python 那样自动编译 .mlpackage。
        // 这里：若传入是 .mlpackage（或非 .mlmodelc），先用 MLModel:compileModelAtURL
        // 编译为 .mlmodelc 缓存到工作目录，避免每次启动重复编译。
        if (![pathStr hasSuffix:@".mlmodelc"]) {
            // 缓存路径：在 mlpackage 同目录下，<name>.mlmodelc
            NSString * stem = [pathStr.lastPathComponent stringByDeletingPathExtension];
            NSString * parent = [pathStr stringByDeletingLastPathComponent];
            NSString * cachedPath = [parent stringByAppendingPathComponent:
                                        [stem stringByAppendingString:@".mlmodelc"]];
            NSURL * cachedURL = [NSURL fileURLWithPath:cachedPath];

            // 没有 cache 才重编译（编译要 5-30s，是大头）
            if (![fm fileExistsAtPath:cachedPath]) {
                NSLog(@"[t2w_dit] compiling CoreML model -> %@ ...", cachedPath);
                NSURL * tmpURL = [MLModel compileModelAtURL:url error:&err];
                if (err || !tmpURL) {
                    NSLog(@"[t2w_dit] error: compileModelAtURL failed: %@",
                          err.localizedDescription);
                    return nullptr;
                }
                // compileModelAtURL 把结果放到 NSTemporaryDirectory，需要复制到我们要的路径
                if ([fm fileExistsAtPath:cachedPath]) {
                    [fm removeItemAtURL:cachedURL error:nil];
                }
                BOOL moved = [fm moveItemAtURL:tmpURL toURL:cachedURL error:&err];
                if (!moved) {
                    // moveItemAtURL 跨卷可能失败 → 退回 copyItemAtURL
                    err = nil;
                    BOOL copied = [fm copyItemAtURL:tmpURL toURL:cachedURL error:&err];
                    if (!copied) {
                        NSLog(@"[t2w_dit] error: cache install failed: %@",
                              err.localizedDescription);
                        // fallback: 直接用临时路径加载（不缓存）
                        cachedURL = tmpURL;
                    }
                }
                NSLog(@"[t2w_dit] compiled & cached: %@", cachedURL.path);
            } else {
                NSLog(@"[t2w_dit] using cached mlmodelc: %@", cachedPath);
            }
            url = cachedURL;
        }

        // compute units 选择
        MLModelConfiguration * cfg = [[MLModelConfiguration alloc] init];
        const char * cu_env = getenv("OMNI_T2W_DIT_COMPUTE_UNITS");
        NSString * cu_str = cu_env ? [NSString stringWithUTF8String:cu_env] : @"ane";
        if ([cu_str isEqualToString:@"all"]) {
            cfg.computeUnits = MLComputeUnitsAll;
        } else if ([cu_str isEqualToString:@"cpu_gpu"]) {
            cfg.computeUnits = MLComputeUnitsCPUAndGPU;
        } else if ([cu_str isEqualToString:@"cpu"]) {
            cfg.computeUnits = MLComputeUnitsCPUOnly;
        } else {
            cfg.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
        }

        // 加载编译后的 .mlmodelc
        MLModel * model = [MLModel modelWithContentsOfURL:url configuration:cfg error:&err];
        if (!model || err) {
            NSLog(@"[t2w_dit] error: failed to load model: %@", err.localizedDescription);
            return nullptr;
        }

        t2w_dit_ctx_t * ctx = (t2w_dit_ctx_t *) calloc(1, sizeof(t2w_dit_ctx_t));
        if (!ctx) {
            NSLog(@"[t2w_dit] error: calloc failed");
            return nullptr;
        }
        ctx->model = model;
        fill_default_dims(&ctx->dims);

        // 用 CoreML 模型自定义 metadata 校验/覆盖 dims。
        NSDictionary * userMD = model.modelDescription.metadata[MLModelCreatorDefinedKey];
        if ([userMD isKindOfClass:[NSDictionary class]]) {
            ctx->dims.depth         = parse_int_md(userMD, @"depth",         ctx->dims.depth);
            ctx->dims.chunk_size    = parse_int_md(userMD, @"chunk_size",    ctx->dims.chunk_size);
            ctx->dims.max_cache_len = parse_int_md(userMD, @"max_cache_len", ctx->dims.max_cache_len);
            ctx->dims.num_heads     = parse_int_md(userMD, @"num_heads",     ctx->dims.num_heads);
            ctx->dims.head_dim      = parse_int_md(userMD, @"head_dim",      ctx->dims.head_dim);
            ctx->dims.mel_channels  = parse_int_md(userMD, @"out_channels",  ctx->dims.mel_channels);
            // spk 已经被 spk_embed_affine_layer 投影到 out_channels=80
            ctx->dims.spk_dim       = ctx->dims.mel_channels;
        }

        ctx->prof_enabled = (getenv("OMNI_T2W_DIT_PROF") != nullptr);
        ctx->n_calls   = 0;
        ctx->ms_total  = 0.0;
        ctx->ms_predict = 0.0;

        NSLog(@"[t2w_dit] loaded CoreML model: %@", pathStr);
        NSLog(@"[t2w_dit]   compute_units = %@", cu_str);
        NSLog(@"[t2w_dit]   depth=%d batch=%d chunk=%d cache=%d heads=%d head_dim=%d",
              ctx->dims.depth, ctx->dims.batch, ctx->dims.chunk_size, ctx->dims.max_cache_len,
              ctx->dims.num_heads, ctx->dims.head_dim);
        return (t2w_dit_handle_t) ctx;
    }
}

// ----------------------------------------------------------------------------
// dims / sizes
// ----------------------------------------------------------------------------

int t2w_dit_get_dims(t2w_dit_handle_t h, t2w_dit_dims_t * out) {
    if (!h || !out) return -1;
    t2w_dit_ctx_t * ctx = (t2w_dit_ctx_t *) h;
    *out = ctx->dims;
    return 0;
}

int t2w_dit_predict_buffer_sizes(t2w_dit_handle_t h,
                                  size_t * nelem_x_mu_cond_feat,
                                  size_t * nelem_t,
                                  size_t * nelem_spks,
                                  size_t * nelem_cnn_cache,
                                  size_t * nelem_att_cache,
                                  size_t * nelem_attn_mask) {
    if (!h) return -1;
    t2w_dit_ctx_t * ctx = (t2w_dit_ctx_t *) h;
    const t2w_dit_dims_t * d = &ctx->dims;
    if (nelem_x_mu_cond_feat) *nelem_x_mu_cond_feat = (size_t) d->batch * d->mel_channels * d->chunk_size;
    if (nelem_t)              *nelem_t              = (size_t) d->batch;
    if (nelem_spks)           *nelem_spks           = (size_t) d->batch * d->spk_dim;
    if (nelem_cnn_cache)      *nelem_cnn_cache      = (size_t) d->depth * d->batch * d->cnn_cache_channels * d->cnn_cache_t;
    if (nelem_att_cache)      *nelem_att_cache      = (size_t) d->depth * d->batch * d->num_heads * d->max_cache_len * (2 * d->head_dim);
    if (nelem_attn_mask)      *nelem_attn_mask      = (size_t) d->batch * d->chunk_size * (d->chunk_size + d->max_cache_len);
    return 0;
}

// ----------------------------------------------------------------------------
// 内部工具：用 dataPointer 包一个 fp32 MLMultiArray（零拷贝输入）
// ----------------------------------------------------------------------------

static MLMultiArray * make_array_fp32(const float * data, NSArray<NSNumber *> * shape, NSError ** err) {
    // 构造 row-major strides
    NSMutableArray<NSNumber *> * strides = [NSMutableArray arrayWithCapacity:shape.count];
    NSInteger stride = 1;
    for (NSInteger i = (NSInteger) shape.count - 1; i >= 0; --i) {
        [strides insertObject:@(stride) atIndex:0];
        stride *= [shape[i] integerValue];
    }
    // 注意：CoreML 不会写这块内存，但 dataPointer 接受的是 void*（非 const）。
    void * mutable_ptr = (void *) data;
    return [[MLMultiArray alloc] initWithDataPointer:mutable_ptr
                                               shape:shape
                                            dataType:MLMultiArrayDataTypeFloat32
                                             strides:strides
                                         deallocator:nil
                                               error:err];
}

// 检查 MLMultiArray 是否是 row-major contiguous（常规情况是；保底再 verify）。
static bool array_is_contiguous(MLMultiArray * arr) {
    NSArray<NSNumber *> * shape = arr.shape;
    NSArray<NSNumber *> * strides = arr.strides;
    if (shape.count != strides.count) return false;
    NSInteger expect = 1;
    for (NSInteger i = (NSInteger) shape.count - 1; i >= 0; --i) {
        if ([strides[i] integerValue] != expect) return false;
        expect *= [shape[i] integerValue];
    }
    return true;
}

// 把 CoreML 输出 MLMultiArray 拷贝到 row-major contiguous 的外部 fp32 buffer。
//
// CoreML 的输出张量经常带 padding（CoreML 把 inner dim pad 到 16/32/64 边界以提速），
// 例如 shape=[2,80,56] strides=[5120,64,1] —— 第 2 维 pad 到 64。这种情况下直接
// memcpy 会把 padding 当真实数据读，导致数值完全错乱。
//
// 实现：尾维 stride=1 时把它当 contiguous "行" 一段一段 memcpy；外层用 N-D
// 计数器迭代。最大 rank 8（远超我们的 4D）。
static void copy_array_to_fp32(MLMultiArray * arr, float * dst) {
    if (!arr || !dst) return;
    const NSInteger n_elem = arr.count;

    if (arr.dataType != MLMultiArrayDataTypeFloat32 &&
        arr.dataType != MLMultiArrayDataTypeFloat16) {
        NSLog(@"[t2w_dit] warn: unexpected output dtype %ld", (long) arr.dataType);
        return;
    }
    const bool is_fp16 = (arr.dataType == MLMultiArrayDataTypeFloat16);
    const size_t elem_size = is_fp16 ? sizeof(__fp16) : sizeof(float);
    const void * raw = arr.dataPointer;

    if (array_is_contiguous(arr) && !is_fp16) {
        memcpy(dst, raw, (size_t) n_elem * sizeof(float));
        return;
    }

    NSArray<NSNumber *> * shape   = arr.shape;
    NSArray<NSNumber *> * strides = arr.strides;
    const NSInteger ndim = (NSInteger) shape.count;
    if (ndim == 0 || ndim > 8) {
        NSLog(@"[t2w_dit] warn: unsupported rank %ld", (long) ndim);
        return;
    }

    int64_t sh[8] = {0}, st[8] = {0};
    for (NSInteger i = 0; i < ndim; ++i) {
        sh[i] = [shape[i]   longLongValue];
        st[i] = [strides[i] longLongValue];
    }
    const int64_t inner_dim    = sh[ndim - 1];
    const int64_t inner_stride = st[ndim - 1];
    const bool inner_contig    = (inner_stride == 1);

    // 外层迭代器：对前 ndim-1 维做 N-D 计数（row-major 输出顺序）。
    // 每步从源读 inner_dim 个元素到 dst[out_off..out_off+inner_dim]。
    int64_t idx[8] = {0};
    int64_t out_off = 0;
    while (true) {
        int64_t src_byte_off = 0;
        for (NSInteger i = 0; i < ndim - 1; ++i) {
            src_byte_off += (int64_t) idx[i] * st[i] * (int64_t) elem_size;
        }
        const char * src = (const char *) raw + src_byte_off;

        if (is_fp16) {
            // 逐元素转 fp32
            for (int64_t k = 0; k < inner_dim; ++k) {
                const __fp16 * p = (const __fp16 *) (src + k * inner_stride * (int64_t) elem_size);
                dst[out_off + k] = (float) (*p);
            }
        } else if (inner_contig) {
            memcpy(dst + out_off, src, (size_t) inner_dim * sizeof(float));
        } else {
            // inner stride != 1：极少见，逐元素拷
            for (int64_t k = 0; k < inner_dim; ++k) {
                const float * p = (const float *) (src + k * inner_stride * (int64_t) sizeof(float));
                dst[out_off + k] = *p;
            }
        }
        out_off += inner_dim;

        // 递增最后一个非内层维（ndim-2 → 0）
        if (ndim == 1) break;
        NSInteger d = ndim - 2;
        idx[d]++;
        while (idx[d] >= sh[d]) {
            if (d == 0) goto done;
            idx[d] = 0;
            d--;
            idx[d]++;
        }
    }
    done:
    (void) 0;
}

// ----------------------------------------------------------------------------
// predict
// ----------------------------------------------------------------------------

int t2w_dit_predict(t2w_dit_handle_t h,
                    const float * x,
                    const float * mu,
                    const float * t,
                    const float * spks,
                    const float * cond,
                    const float * cnn_cache_in,
                    const float * att_cache_in,
                    const float * attn_mask,
                    float * feat_out,
                    float * cnn_cache_out,
                    float * att_cache_out) {
    if (!h || !x || !mu || !t || !spks || !cond || !cnn_cache_in || !att_cache_in || !attn_mask) {
        NSLog(@"[t2w_dit] predict: null input");
        return -1;
    }
    if (!feat_out || !cnn_cache_out || !att_cache_out) {
        NSLog(@"[t2w_dit] predict: null output");
        return -1;
    }

    @autoreleasepool {
        t2w_dit_ctx_t * ctx = (t2w_dit_ctx_t *) h;
        const t2w_dit_dims_t * d = &ctx->dims;

        const auto t0 = std::chrono::high_resolution_clock::now();

        NSError * err = nil;
        NSArray * sh_x        = @[ @(d->batch), @(d->mel_channels), @(d->chunk_size) ];
        NSArray * sh_t        = @[ @(d->batch) ];
        NSArray * sh_spks     = @[ @(d->batch), @(d->spk_dim) ];
        NSArray * sh_cnn      = @[ @(d->depth * d->batch), @(d->cnn_cache_channels), @(d->cnn_cache_t) ];
        NSArray * sh_att      = @[ @(d->depth * d->batch), @(d->num_heads), @(d->max_cache_len), @(2 * d->head_dim) ];
        NSArray * sh_mask     = @[ @(d->batch), @(d->chunk_size), @(d->chunk_size + d->max_cache_len) ];

        MLMultiArray * a_x        = make_array_fp32(x,            sh_x,    &err);
        MLMultiArray * a_mu       = err ? nil : make_array_fp32(mu,           sh_x,    &err);
        MLMultiArray * a_t        = err ? nil : make_array_fp32(t,            sh_t,    &err);
        MLMultiArray * a_spks     = err ? nil : make_array_fp32(spks,         sh_spks, &err);
        MLMultiArray * a_cond     = err ? nil : make_array_fp32(cond,         sh_x,    &err);
        MLMultiArray * a_cnn_in   = err ? nil : make_array_fp32(cnn_cache_in, sh_cnn,  &err);
        MLMultiArray * a_att_in   = err ? nil : make_array_fp32(att_cache_in, sh_att,  &err);
        MLMultiArray * a_mask     = err ? nil : make_array_fp32(attn_mask,    sh_mask, &err);
        if (err || !a_x || !a_mu || !a_t || !a_spks || !a_cond || !a_cnn_in || !a_att_in || !a_mask) {
            NSLog(@"[t2w_dit] predict: input array build failed: %@", err.localizedDescription);
            return -5;
        }

        NSDictionary * features = @{
            @"x":            [MLFeatureValue featureValueWithMultiArray:a_x],
            @"mu":           [MLFeatureValue featureValueWithMultiArray:a_mu],
            @"t":            [MLFeatureValue featureValueWithMultiArray:a_t],
            @"spks":         [MLFeatureValue featureValueWithMultiArray:a_spks],
            @"cond":         [MLFeatureValue featureValueWithMultiArray:a_cond],
            @"cnn_cache_in": [MLFeatureValue featureValueWithMultiArray:a_cnn_in],
            @"att_cache_in": [MLFeatureValue featureValueWithMultiArray:a_att_in],
            @"attn_mask":    [MLFeatureValue featureValueWithMultiArray:a_mask],
        };
        MLDictionaryFeatureProvider * provider = [[MLDictionaryFeatureProvider alloc]
                                                    initWithDictionary:features error:&err];
        if (err || !provider) {
            NSLog(@"[t2w_dit] predict: provider build failed: %@", err.localizedDescription);
            return -2;
        }

        const auto t_pre = std::chrono::high_resolution_clock::now();

        id<MLFeatureProvider> result = [ctx->model predictionFromFeatures:provider error:&err];
        if (err || !result) {
            NSLog(@"[t2w_dit] predict: model predict failed: %@", err.localizedDescription);
            return -3;
        }

        const auto t_pred = std::chrono::high_resolution_clock::now();

        // 一次性 debug 输出实际拿到的 output names + shapes（仅当 prof 开启时）
        if (ctx->prof_enabled && ctx->n_calls == 0) {
            for (NSString * name in result.featureNames) {
                MLMultiArray * a = [result featureValueForName:name].multiArrayValue;
                if (a) {
                    NSMutableString * sh = [NSMutableString stringWithString:@"["];
                    for (NSInteger i = 0; i < (NSInteger) a.shape.count; ++i) {
                        if (i > 0) [sh appendString:@","];
                        [sh appendFormat:@"%@", a.shape[i]];
                    }
                    [sh appendString:@"]"];
                    NSMutableString * st = [NSMutableString stringWithString:@"["];
                    for (NSInteger i = 0; i < (NSInteger) a.strides.count; ++i) {
                        if (i > 0) [st appendString:@","];
                        [st appendFormat:@"%@", a.strides[i]];
                    }
                    [st appendString:@"]"];
                    fprintf(stderr, "[t2w_dit] output[%s]: count=%ld shape=%s strides=%s\n",
                            [name UTF8String], (long) a.count,
                            [sh UTF8String], [st UTF8String]);
                }
            }
        }

        MLMultiArray * o_feat = [result featureValueForName:@"feat"].multiArrayValue;
        MLMultiArray * o_cnn  = [result featureValueForName:@"cnn_cache_out"].multiArrayValue;
        MLMultiArray * o_att  = [result featureValueForName:@"att_cache_out"].multiArrayValue;
        if (!o_feat || !o_cnn || !o_att) {
            NSLog(@"[t2w_dit] predict: missing output (feat=%p cnn=%p att=%p)",
                  o_feat, o_cnn, o_att);
            return -4;
        }

        copy_array_to_fp32(o_feat, feat_out);
        copy_array_to_fp32(o_cnn,  cnn_cache_out);
        copy_array_to_fp32(o_att,  att_cache_out);

        const auto t1 = std::chrono::high_resolution_clock::now();

        const double ms_predict = std::chrono::duration<double, std::milli>(t_pred - t_pre).count();
        const double ms_total   = std::chrono::duration<double, std::milli>(t1     - t0  ).count();
        ctx->n_calls++;
        ctx->ms_predict += ms_predict;
        ctx->ms_total   += ms_total;

        if (ctx->prof_enabled) {
            const double ms_wrap = std::chrono::duration<double, std::milli>(t_pre - t0).count();
            const double ms_post = std::chrono::duration<double, std::milli>(t1 - t_pred).count();
            fprintf(stderr,
                    "[prof] t2w_dit call=%lld total=%.2fms (wrap=%.2f predict=%.2f post=%.2f)\n",
                    (long long) ctx->n_calls, ms_total, ms_wrap, ms_predict, ms_post);
        }
        return 0;
    }
}

// ----------------------------------------------------------------------------
// free
// ----------------------------------------------------------------------------

void t2w_dit_free(t2w_dit_handle_t h) {
    if (!h) return;
    t2w_dit_ctx_t * ctx = (t2w_dit_ctx_t *) h;
    if (ctx->n_calls > 0) {
        fprintf(stderr,
                "[t2w_dit] summary: n_calls=%lld total=%.2fms (avg=%.2fms) predict=%.2fms (avg=%.2fms)\n",
                (long long) ctx->n_calls, ctx->ms_total, ctx->ms_total / (double) ctx->n_calls,
                ctx->ms_predict, ctx->ms_predict / (double) ctx->n_calls);
    }
    ctx->model = nil;  // ARC release
    free(ctx);
}

#if __cplusplus
}  // extern "C"
#endif
