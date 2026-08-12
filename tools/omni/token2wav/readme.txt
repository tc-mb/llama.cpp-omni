1. 接口
  1. 初始化
bool init_from_prompt_cache_gguf()
使用此接口即可
始化方式第一种（最好使用此种方式，固定样例声音，并且加速）：加载模型 + 导入 prompt_cache
init_from_prompt_bundle()
不推荐使用，此接口做备用，用于更换示例音频
初始化方式第二种（如要更换示例音频使用此接口）：加载模型 + 导入 prompt_bundl
  
  2. 推理
feed_window()：
使用此接口即可
推理方式第一种：送入前已做好28个token拼接：若上层已在外部按窗口切好，送入28token即可，可选vector输出或callback形式(推荐使用callback)
feed_tokens()：
推理方式第二种：（不推荐）送入前若未做好28token拼接，滑窗形式，任意送入 token，内部凑 28 进行窗口推理，并按步进 25 滑动（不推荐使用此接口，会影响性能，仅做备用）可选vector输出或callback形式
  
  3. 其它
reset()
清空streaming所有状态

2. 接口位置
  综合，只需使用下方两接口即可，其余仅为备用
  init_from_prompt_cache_gguf+feed_window() 接口组合
  
  接口位于/llamacpp/tools/omni/token2wav.cpp
  实现位于/llamacpp/tools/omni/token2wav-impl.cpp
  
3. 使用示例（init_from_prompt_cache_gguf+feed_window() 接口组合）

4. Profile（性能度量，见 token2wav-profile.h）
  通过环境变量开启，默认关闭，对上层接口完全透明：
    OMNI_T2W_PROFILE=1    仅在进程退出前打印一次 [profile] summary 汇总
                          （init / token2mel / vocoder / total / callback 的
                           min/p50/p95/p99/max/mean，以及 audio / RTF）
    OMNI_T2W_PROFILE=2    在上面基础上，每次 push_tokens_window 都打印一行 [timing]
    OMNI_T2W_PRINT_GRAPH=1 第一次调用 token2mel 与 vocoder 的图计算时各 ggml_graph_print 一次

  典型建立 baseline 的做法（同目标硬件跑 10+ 次取中位数）：
    OMNI_T2W_PROFILE=1 ./token2wav-example 2> /tmp/t2w_profile.log
  首次做 kernel 级 profile 时可配合：
    nsys profile --trace=cuda,nvtx,osrt -o t2w \
        env OMNI_T2W_PROFILE=2 OMNI_T2W_PRINT_GRAPH=1 ./token2wav-example

5. Token2Mel 小算子优化回滚开关
  以下实验项默认保持 legacy 路径；只有显式设为 0 才启用优化：
    OMNI_T2W_LEGACY_CONV_STATE=0
    OMNI_T2W_LEGACY_ADALN_SILU=0
    OMNI_T2W_LEGACY_ADALN_CACHE=0
    OMNI_T2W_LEGACY_EST_ATTN_CPY=0
  fused QKV 也保持默认关闭，显式启用：
    OMNI_T2W_FUSED_QKV=1

  这些路径仍保留 canonical shape/type 检查；条件不满足时自动执行 legacy 图。

  AdaLN cache 在 canonical 16-block/512/F32/5-step CFM 中按 non-last/last call-id 分别在 device 上缓存
  固定 timestep 的 AdaLN modulation 输出；每个 call-id 首次使用由原计算图填充，后续 chunk 复用。

  canonical F32 estimator attention cache 会先打包 [current, old]，随后 slice(delta, L)
  得到逐字节相同的 old cache。在精确的 [128, L+delta, 640, 2] ->
  [128, L, 640, 2] shape proof 成立时，不把该恒等 writeback 加入 steady graph；
  max-cache 截断、类型或 shape 不匹配时自动使用原路径。

  canonical F32 CFM 默认把 modulate 的 Mul+Add+Add 和 gate-residual 的 Mul+Add
  分别融合为单个 AscendC kernel；非 canonical shape/layout 自动回退。需要恢复独立算子时：
    GGML_CANN_MODULATE_FUSION=off

  canonical F32 estimator attention 可把 time 维 concat 和紧随其后的
  [D,H,T,B] -> [D,T,H,B] materialize 合并为一次逐字节搬运；任何 shape、
  stride、别名或对齐不匹配都会回退。默认 auto，精确回滚为：
    GGML_CANN_ATTN_TIME_PACK=off

  causal Im2col 默认用 40 个 vector block 覆盖当前 Ascend 设备的双 vector core，
  相比旧 20-block 调度减少尾部串行。需要精确恢复旧调度时：
    GGML_CANN_IM2COL1D_CAUSAL_BLOCKS=20

  CTB causal Im2col 默认保留连续 [C,T,B] 输入，只做一次 time concat，并在
  AscendC 内通过 UB Gather 直接生成 legacy [K*C,OW,B] 排列，从图中删除两次
  transpose materialize。恢复旧图：
    OMNI_T2W_CAUSAL_CTB_IM2COL=off

  CTB concat+Im2col 默认在 canonical C=512/B=2/T=50|56 图中直接从
  2-frame cache 与当前 chunk 读取，跳过中间 concat；任一图结构、shape、
  内存别名不匹配会自动执行原图。精确回滚：
    GGML_CANN_IM2COL1D_CTB_SPLIT=off

  canonical CTB Gather 默认从内核只读区 DMA 载入 1536 项 offset 表，避免
  每个 vector core、每次调用重复执行 1536 次标量 SetValue。非 canonical
  shape 自动保留原 offset 生成逻辑。精确回滚：
    GGML_CANN_IM2COL1D_CTB_STATIC_OFFSETS=off

  HIFT/F0 中 canonical K=3/7/11、stride=1、对称 padding 的 F32 卷积默认复用
  native AscendC Im2col，避免长序列 ACLNN Im2col。非 canonical 参数自动走原图。精确回滚：
    OMNI_T2W_HIFT_NATIVE_IM2COL=off

  HIFT source_down0 的 K=30/stride=15/pad=7 长序列窗口默认使用专用
  32-word UB 行搬运；关闭后该调用点回到 ACLNN Im2col：
    GGML_CANN_IM2COL1D_HIFT_STRIDED=off

  HIFT/vocoder 的长序列 K=3/7/11 卷积默认按 4 个相邻输出位置复用同一
  GM->UB 输入窗口，减少重叠窗口的 HBM 读取；短序列保持原调度。精确回滚：
    GGML_CANN_IM2COL1D_VOCODER_TILE=off

  HIFT/F0/resblock 的连续 reshape 默认直接交给下游算子，避免无条件
  ggml_cont 生成重复 TensorMove；非连续 tensor 仍自动物化。精确回滚：
    OMNI_T2W_LEGACY_HIFT_CONT=1

  canonical F32 CFM 的 Norm -> Mul(weight) -> Add(bias) 默认直接调用 CANN
  原生带 affine 参数的 LayerNorm，省去两个独立 elementwise launch。需要精确回滚时：
    GGML_CANN_AFFINE_LAYERNORM=off
