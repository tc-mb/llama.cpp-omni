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