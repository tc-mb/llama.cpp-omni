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
