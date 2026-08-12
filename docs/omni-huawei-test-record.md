# 华为评测分支迁移与测试记录

## 1. 结论

本次已将经审查的 llama.cpp-omni 正确性修复和性能优化迁移到
`bench/huawei` 评测分支工作树，并在 Linux aarch64 + Ascend 910 上完成真实测试。

- 正式 RTS：3 次预热、20 次测量，20/20 成功。
- 全量 chunk pooled RTF：`1.10093 -> 0.87059`，降低 `20.92%`。
- 全部 chunk 平均 RTF：`1.14139 -> 0.89085`，降低 `21.95%`。
- 全部 chunk P95 RTF：`1.50200 -> 1.01247`，降低 `32.59%`。
- SPEAK 输入到 WAV 完整链路：`1287.11 ms -> 1109.53 ms`，降低 `13.80%`。
- decode 结束到 WAV：`575.63 ms -> 373.75 ms`，降低 `35.07%`。
- `./run_all.sh --no-build --smoke 2` 的四个任务全部成功。
- 14 个定向 C++/CANN 测试、11 个 Python wiring 测试全部通过。
- `evaluation/` 和四个指定文件的校验值保持不变。

旧基线和新版本的生成轨迹不完全相同，因此总体 RTF 对比用于衡量同一输入下的
实际端到端吞吐；具有相同 `6720 ms` 输出音频长度的逐项回滚测试用于定位优化收益。

## 2. 迁移内容

### 2.1 正确性与生命周期

- 统一 break、cancel、join、reset、reuse 生命周期，避免清理 KV 时仍有后台阶段访问。
- 修复 HTTP/SSL 对象生命周期、监听失败返回码、设备绑定和 WebSocket 失败路径清理。
- 修复媒体输入顺序，最终按图像/视频帧、音频、媒体后 prompt 进入会话。
- 补齐 EOG、生成上限、终止序列、sampler reset、参考音频会话和完整 terminal token。
- Python Token2Wav oneshot 请求改为确定性事务，补齐 seed、参考音频、缓存和精度模式协议。
- 修复 audition 的 mask、向上取整、mel crop、短音频和 BF16 转换边界。
- CLI 补齐采样、预测、输出目录、flash-attn 和默认 `npu:0` 参数。

### 2.2 默认启用的性能优化

- TTS 首次 prefill 跳过重复工作，保留 legacy 环境变量回滚。
- TTS head projection 使用长期 executor 和复用线程池。
- sampler 使用 `thread_local` workspace、partial TopK、稀疏 repetition penalty，并固定 seed 和 tie 顺序。
- CANN modulate fusion、attention time pack、affine LayerNorm。
- AscendC Im2col1D：CTB 直连/split/static offsets、causal 调度、HIFT/native/strided、vocoder tile。
- HIFT 连续布局消除和 Token2Wav 图内不必要 materialize/writeback 的安全删除。
- 调试产物默认关闭，避免热路径文件系统开销。

### 2.3 已迁移但保持显式 opt-in 的实验项

以下代码已迁移并完成真实链路测试，但默认保持保守路径，不用未经充分质量统计的
小样本收益改变正式评测语义：

- `OMNI_T2W_LEGACY_ADALN_CACHE=0`
- `OMNI_T2W_LEGACY_ADALN_SILU=0`
- `OMNI_T2W_LEGACY_CONV_STATE=0`
- `OMNI_T2W_LEGACY_EST_ATTN_CPY=0`
- `OMNI_T2W_FUSED_QKV=1`
- `GGML_CANN_SET_ROWS_F32_F16=auto`
- `GGML_CANN_KV_PAIR_UPDATE=auto`（与 SET_ROWS 一起启用）
- `OMNI_TTS_LEGACY_DUPLEX_CHUNK_RESET=0`
- APM baseline/replay API，以及 `OMNI_TTS_DEVICE`、`OMNI_PROJECTOR_DEVICE` 分阶段设备选择。

主 LLM TP2 配置已迁移，支持通过 `OMNI_LLM_DEVICES`、
`OMNI_LLM_SPLIT_MODE` 和 `OMNI_LLM_TENSOR_SPLIT` 显式启用。未设置这些变量时仍维持
原有单芯路径，不会静默改变设备拓扑。`OMNI_LLM_DEVICES=auto` 会运行时枚举 CANN
设备；只发现一张卡时明确回退到 `split_mode=none`，显式无效设备仍然报错，不回退
到 CPU。CANN 上的真实 TP2 使用 `OMNI_LLM_SPLIT_MODE=tensor`，由 llama.cpp Meta
backend 对每层权重和 KV 做张量切分；旧 `row` 模式因 CANN 不提供 legacy split
buffer，不能作为 TP2 证据。双卡真实性能与稳定性测试已经完成，结论和原始结果见第 8 节。

### 2.4 GitHub PR 覆盖边界

使用 `gh pr list --repo July-h5kf3/MiniCPM--competition --state all` 重新核对了
PR #1–#27。与 llama.cpp-omni F16/Graph-off 子赛道直接相关的内容已覆盖：

- PR #2–#4：CANN 并发互斥、Flash Attention 参数和 Token2Wav NPU 设备修复。
- PR #6：EOS/EOG 与异常生成循环修复。
- PR #8：赛题 Demo 与 RTF 基线适配；评测目录保持受保护分支原样。
- PR #10、#11、#13：causal/vocoder Im2col1D 与 vector gather 优化。
- PR #21、#22：双工 SPEAK 链路、NPU 设备解析和 SPEAK→WAV 指标。
- Open PR #25、#26：Token2Mel 算子融合和 TTS/Projector 分阶段设备选择；代码已迁移，
  多芯路径保持显式配置。
- PR #9：迁移 `ws_handler.cpp` 中多图片、音频、文本的 prefill 顺序及失败清理修复；
  PR 内其余 vLLM/Python 实现不直接搬入 C++ 评测分支。

以下 PR 不进入本次 F16、`ACL_GRAPH=off` 的正式默认组合：PR #5、#7、#15–#20
是 vLLM-Omni/Python 模型实现；PR #12 只优化 ACL Graph vocoder bucket。PR #14 的
Q8 W8A8 路径已完成迁移，但严格保持显式 opt-in，不改变 F16 正式默认。

### 2.5 迁移计划逐项复核

已按
`/workspace/MiniCPM--competition/.omx/plans/llama-omni-bench-huawei-optimization-migration.md`
重新核对实现、开关、构建接线和测试。结论如下：

| 计划分组 | 计划项 | 目标分支状态 | 验证 |
| --- | --- | --- | --- |
| Im2col 主栈 | causal CTB、split-input、static offsets、20→40 blocks、HIFT native/strided、vocoder tile、冗余 `ggml_cont` 消除 | 已迁移；主组合默认启用，均可回滚 | CANN 定向测试、逐项真实 A/B、3+20 RTS |
| Token2Mel 主线 | modulate/gate fusion、attention time pack、affine LayerNorm | 已迁移并默认启用 | policy/CANN 测试、逐项真实 A/B |
| Token2Mel 实验项 | AdaLN SiLU/cache、estimator writeback、conv-state、fused QKV、SET_ROWS F32→F16、K/V pair | 已迁移；严格显式 opt-in | policy 测试、真实链路逐项测试 |
| TTS/流水线 | first prefill、partial TopK/workspace/稀疏 penalty、持久 head executor、debug 默认关闭、same-turn KV | 已迁移；same-turn KV 保持 opt-in | 固定 seed/tie 测试、executor 测试、真实 A/B |
| 设备/APM | TTS/Projector 分阶段设备、APM baseline/replay | 已迁移；不设置时维持单芯正式路径 | APM audio-encode-only、duplex baseline/replay 计数与 hash；分阶段设备双卡 A/B 已排队 |
| 独立实验 | ACL Graph bucket、Q8 W8A8、主 LLM TP2 | Q8 W8A8 已迁移并完成真实单卡 A/B；TP2 已迁移并完成真实双卡 ABBA；均不改变正式默认 | Q8 正确性、Graph replay 与真实性能已验证；TP2 稳定但未通过收益门槛，继续保持显式 opt-in |

另外，源工作树中的 `Token2WavSession::switch_prompt_bundle` 已在目标分支现有
`token2wav.cpp` 和声明中存在；`src/llama-graph.cpp` 仅为空行差异；保护 CMake 中的
executor 接线改由根 `CMakeLists.txt` 完成。以上均已核对，不构成迁移遗漏。

## 3. 环境与可复现配置

- 时间：2026-08-11 至 2026-08-12（UTC）
- 系统：Linux 5.10.0，aarch64，openEuler
- NPU：Ascend 910；`npu-smi 25.5.1`
- CANN：`9.1.0-beta.3`
- CMake：`3.27.9`
- Python：`3.12.13`
- ffmpeg：`6.1.1`
- rubberband：`1.8.2`
- 目标基线 commit：`c9785ccca96501820e36744d6310e0c80af5c054`
- 正式主组合设备：物理 device `1`，单芯执行；Q8 独立实验使用 device `0`，TP2
  独立实验使用 device `0,1`
- 固定配置：F16、`CTX_SIZE=40960`、seed `42`、
  `GGML_CANN_WEIGHT_NZ=off`、`GGML_CANN_ACL_GRAPH=off`

关键输入 SHA-256：

- LLM：`d1e6984531bab1962d8bc73da4b6dffc5c2d9b0da336603943df04100e57c3de`
- TTS：`c7be3748a863dd6966ae7eed42600b7f41ca67affb03729ff245247f0e5ea088`
- RTS 视频：`31622e1efd9a7b197a340266037b45aeec13b3b27f010f1ea1d22d9c6e69405f`
- 旧基线 server：`e48f8acfcfcee98387fb148db6b86798d8d95cf5badcc5263fb0c29a73db95c6`
- 迁移后 server：`1430ff007e9308e530f78d5bcaaa490e583e96855369047788f30ea15003ee02`

## 4. 正式 3+20 RTS 结果

原始结果：

- 基线：`/workspace/MiniCPM--competition/result/llama_huawei_migration/baseline_ctx40960_20260811T215100Z`
- 候选：`/workspace/MiniCPM--competition/result/llama_huawei_migration/candidate_final_default_20260811T232059Z`

| 指标 | 旧基线 | 迁移后 | 变化 |
| --- | ---: | ---: | ---: |
| 全量 pooled RTF | 1.10093 | 0.87059 | -20.92% |
| 每次运行 pooled RTF 均值 / P50 / P95 | 1.10003 / 1.09820 / 1.12769 | 0.87060 / 0.86880 / 0.89181 | 均值 -20.86% |
| 全部 chunk 平均 RTF | 1.14139 | 0.89085 | -21.95% |
| 全部 chunk P95 RTF 均值 | 1.50200 | 1.01247 | -32.59% |
| core pooled RTF 均值 | 1.15048 | 0.92770 | -19.36% |
| SPEAK→WAV 均值 / P50 / P95 | 1287.11 / 1282.75 / 1311.78 ms | 1109.53 / 1100.00 / 1133.00 ms | 均值 -13.80% |
| SPEAK wall 均值 | 711.70 ms | 736.00 ms | +3.41% |
| decode→WAV 均值 | 575.63 ms | 373.75 ms | -35.07% |
| 失败率 | 0/20 | 0/20 | 无回退 |

迁移后阶段 RTF（20 次均值）：

| encode | LLM prefill | LLM decode | TTS | Token2Wav |
| ---: | ---: | ---: | ---: | ---: |
| 0.18279 | 0.01661 | 0.36132 | 0.17987 | 0.12998 |

迁移后 pooled RTF 的范围为 `0.85310–0.89950`；SPEAK→WAV 的范围为
`1097.0–1148.2 ms`。所有正式样本都计入统计，没有剔除慢样本。

## 5. 逐项真实端到端测试

每项均运行正式 direct judge，输入为同一 37-chunk 视频，生成真实 WAV。
“回滚”表示关闭默认优化；数值越低越好。单次差异小于约 1% 时按噪声范围记录，
不宣称独立收益。

### 5.1 默认优化及回滚

| 测试项 | 状态 | pooled RTF | chunk 平均 / P95 RTF | SPEAK→WAV | TTS / T2W RTF | 判断 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 默认候选（20 次） | 开启 | 0.8706 | 0.8909 / 1.0125 | 1109.5 ms | 0.1799 / 0.1300 | 正式结果 |
| CANN/Token2Wav 主组 | 全部回滚 | 1.0321 | 1.0524 / 1.2557 | 1229.0 ms | 0.1827 / 0.2763 | 默认主组明显有效 |
| Im2col/layout 主组 | 回滚 | 1.0277 | 1.0472 / 1.2018 | 1248.8 ms | 0.1726 / 0.2869 | 默认主组明显有效 |
| modulate fusion | 回滚 | 0.8951 | 0.9155 / 1.0468 | 1122.0 ms | 0.1855 / 0.1452 | 默认开启方向更优 |
| attention time pack | 回滚 | 0.8853 | 0.9053 / 1.0282 | 1130.0 ms | 0.1905 / 0.1319 | 默认开启方向更优 |
| affine LayerNorm | 回滚 | 0.8765 | 0.8960 / 1.0096 | 1104.1 ms | 0.1818 / 0.1350 | 噪声范围 |
| 首次 prefill | legacy | 1.3747 | 1.3682 / 1.5950 | 1621.7 ms | 0.2779 / 0.2802 | 明显变慢；输出轨迹不同 |
| sampler workspace/TopK/penalty | full-sort legacy | 1.4005 | 1.4107 / 1.6730 | 1770.2 ms | 0.3208 / 0.2996 | 明显变慢；同为 6720 ms 音频 |
| causal vector blocks | 旧 20 blocks | 0.8644 | 0.8829 / 0.9907 | 1100.2 ms | 0.1766 / 0.1273 | 未证明 40 blocks 独立收益 |
| causal CTB direct Im2col | 回滚 | 0.8901 | 0.9104 / 1.0443 | 1101.1 ms | 0.1759 / 0.1500 | 默认开启方向更优 |
| CTB split 直读 | 回滚 | 0.9037 | 0.9227 / 1.0588 | 1121.7 ms | 0.1945 / 0.1323 | 默认开启明显更优 |
| CTB static offsets | 回滚 | 0.8739 | 0.8934 / 1.0135 | 1101.9 ms | 0.1851 / 0.1277 | 噪声范围 |
| HIFT native Im2col | 回滚 | 0.8586 | 0.8786 / 0.9907 | 1100.3 ms | 0.1703 / 0.1316 | 未证明独立收益 |
| HIFT strided Im2col | 回滚 | 0.8581 | 0.8784 / 0.9944 | 1104.0 ms | 0.1788 / 0.1248 | 未证明独立收益 |
| vocoder tile | 回滚 | 0.8728 | 0.8908 / 0.9889 | 1103.0 ms | 0.1811 / 0.1314 | 噪声范围 |
| HIFT cont 消除 | legacy | 0.8673 | 0.8871 / 1.0013 | 1106.6 ms | 0.1769 / 0.1309 | 噪声范围 |

### 5.2 实验优化启用结果

| 测试项 | pooled RTF | chunk 平均 / P95 RTF | SPEAK→WAV | TTS / T2W RTF | 默认决定 |
| --- | ---: | ---: | ---: | ---: | --- |
| AdaLN cache | 0.8759 | 0.8969 / 1.0568 | 1119.7 ms | 0.1711 / 0.1369 | 保持关闭 |
| AdaLN SiLU | 0.8911 | 0.9083 / 1.0576 | 1127.3 ms | 0.1831 / 0.1342 | 保持关闭 |
| conv state | 0.8798 | 0.8999 / 1.0168 | 1121.0 ms | 0.1850 / 0.1275 | 保持关闭 |
| estimator writeback 消除 | 0.8687 | 0.8887 / 1.0040 | 1097.5 ms | 0.1858 / 0.1187 | 小幅、近噪声，保持关闭 |
| fused QKV | 0.8565 | 0.8764 / 0.9926 | 1099.7 ms | 0.1733 / 0.1247 | 有性能方向，质量统计不足，保持关闭 |
| SET_ROWS F32→F16 | 0.8593 | 0.8792 / 0.9989 | 1097.4 ms | 0.1726 / 0.1291 | 有性能方向，保持 opt-in |
| KV pair + SET_ROWS | 0.8680 | 0.8880 / 1.0079 | 1098.6 ms | 0.1727 / 0.1301 | 未优于 SET_ROWS 单项，保持 opt-in |
| same-turn duplex KV | 0.8591 | 0.8793 / 1.0101 | 1095.1 ms | 0.1699 / 0.1277 | 有性能方向，保持 opt-in |

APM baseline/replay 另按计划使用非保护测试入口验证。`--audio-encode-only` 完成 2/2
帧，首帧/热帧为 `209.173/12.036 ms`。同一两帧进入 duplex baseline 时 APM 为
`35.5/33.2 ms`，计数 `prepared=0 live=2 hit=0 miss=0`；replay 预计算后的运行时
encoder wall 均约 `0.1 ms`，计数 `prepared=2 live=0 hit=2 miss=0`。两条 replay 的
embedding hash 与 baseline 分别一致为 `1cb33cbebd5b51b4`、`086917cadac8866c`。
原始日志位于 `/workspace/MiniCPM--competition/result/llama_huawei_migration_apm/`。
该能力保持显式测试/API 路径，不改变正式在线默认。

Token2Wav ACL Graph bucket 使用同一个 `USE_ACL_GRAPH=ON` 构建做短 A/B。运行时 Graph
off/on 的 pooled RTF 为 `0.8634/0.8560`，chunk mean 为 `0.8834/0.8845`，P95 为
`1.0017/1.1252`，SPEAK→WAV 为 `1098.0/1074.0 ms`。虽然单次 pooled 和端到端时延
方向较好，但 chunk mean 未改善且 P95 回退 `12.33%`，未达到准入门槛，因此正式默认
继续 `GGML_CANN_ACL_GRAPH=off`，不追加 3+20。原始结果位于
`/workspace/MiniCPM--competition/result/llama_huawei_migration_graph/`。

逐项原始目录均位于：
`/workspace/MiniCPM--competition/result/llama_huawei_migration/`。目录名与表中测试项对应，
每个目录包含 manifest、console log、评测原始 JSON 和 `summary.json`。

## 6. 官方四任务 smoke

命令：

```bash
cd evaluation
EVAL_CONFIG=../.local-eval/config.env ./run_all.sh --no-build --smoke 2
```

结果目录：`evaluation/output/20260811_235613`

| 任务 | 状态 | 结果 | 耗时 |
| --- | --- | --- | ---: |
| Video-MME | OK | 2/2 有效，1/2 正确（50%） | 53.0 s |
| Daily-Omni | OK | 2/2 正确（100%） | 38.8 s |
| Seed-TTS | OK | 2 WAV，WER 4.545%，SIM 0.777 | 109.3 s |
| RTS | OK | core RTF 0.8601，SPEAK→WAV 1037.9 ms | 71.3 s |

此前两次 `--full` 资产检查记录位于 `evaluation/output/20260812_000128` 和
`evaluation/output/20260812_000722`，当时因 Video-MME 和 Daily-Omni 数据不完整，
未记为通过。

补齐数据后已重新启动正式全量评测，结果目录为
`evaluation/output/20260812_003522`。Video-MME 已完成 900 个视频、2700 个问题，
正式重跑无效答案后官方 Overall 为 `69.8%`。统一脚本当前继续执行 Daily-Omni；
截至本记录更新时已稳定处理超过 39/1197 条，未出现无效响应、进程崩溃或超时。
Daily-Omni 完成后，统一脚本将自动继续 Seed-TTS 和 RTS。完整四任务门禁仍在运行，
在其结束前不提前声明 full 通过。

另一次单卡 full 启动记录保存在 `evaluation/output/20260812_000128`。两次 full 尝试
都保留原始日志，不混入通过结论。四任务 smoke 与独立 3+20 RTS 均已完成；完整
Video-MME 视频和真实 Daily-Omni 全量 annotation 已补齐，当前 full 继续使用这些资产
执行，不再存在此前的数据缺失问题。

## 7. 构建与自动化验证

- 构建目标：`llama-omni-server`、`llama-omni-eval-cli`、
  `llama-omni-eval-daily-cli`、`llama-omni-tts-eval`，全部成功。
- CTest：14/14 通过，包括真实 CANN Im2col1D 和 Q8 W8A8 测试。
- `test_cpp_oneshot_wiring.py`：5/5 通过。
- `test_runtime_optimization_wiring.py`：7/7 通过，包含主 LLM TP2 配置接线和
  Meta split-state 地址复用回归检查。
- BF16 weight-norm Ascend 数值对照：固定 seed `42`、形状 `[6562,768]`，迁移公式与
  NPU 上 `torch._weight_norm` 的 5,039,616 个 BF16 元素逐位一致；旧 F32 合并顺序有
  1,555,565 个元素不同（`30.87%`）。
- `benchmarks/omni-huawei/validate.sh static`：通过。
- 最新静态验证目录：`benchmarks/omni-huawei/results/20260812T133302Z`。
- `git diff --check`：通过。

## 8. Q8 W8A8 与主 LLM TP2 追加迁移

Q8 W8A8 保持默认关闭，仅在 `GGML_CANN_Q8_W8A8=on` 时启用；默认仍走原有
W8A16 路径。迁移内容包括动态量化、QuantMatmulV5、Q8 权重布局注册、Graph
workspace/replay 事务和定向测试。NPU0 上三组正确性测试全部通过，W8A8 相对
W8A16 的平均绝对误差为 `0.00859515`、`0.00956655`、`0.00863347`，最大绝对误差
分别为 `0.01953125`、`0.03027344`、`0.01953125`。另使用 Graph-enabled 独立构建
验证了 eager、capture/replay、缓存淘汰、注册表切换及同设备资源路径。

对照 PR #14 最终合并头 `3a68b3f397c1f858baf4f17ae1fdfba6c52fdffb` 复核：
`graph-transaction.h`、`q8-w8a8.{h,cpp}` 和 `test-cann-q8-w8a8.cpp` 与上游文件
SHA-256 完全一致；修改文件中的 QuantMatmulV5、workspace plan、Graph snapshot、
per-device transaction gate、RoPE preload 与 active capture 接线均已进入目标实现。

真实吞吐测试使用 `MiniCPM-o-4_5-Q8_0.gguf`、CANN0、32-token decode，并分别执行
3 次预热和 20 次测量；Weight NZ 与 ACL Graph 均关闭，结果如下：

| 路径 | 平均耗时 | 平均吞吐 | 相对 W8A16 |
| --- | ---: | ---: | ---: |
| W8A16（Q8 优化关闭） | 1166.397 ms | 27.4349 token/s | 基线 |
| W8A8（Q8 优化开启） | 894.835 ms | 35.7609 token/s | 吞吐 `+30.35%`，耗时 `-23.28%` |

两组均为 20/20 有效。W8A16 吞吐范围为 `27.4121–27.4591 token/s`，W8A8 为
`35.5439–35.8012 token/s`；耗时 P95 分别为 `1167.256 ms` 和 `895.695 ms`。
原始 JSON 与日志位于
`/workspace/MiniCPM--competition/result/llama_huawei_migration_q8_3plus20_20260812T1105Z`。

随后使用同一目标构建、`MiniCPM-o-4_5-Q8_0.gguf` 和
`GGML_CANN_Q8_W8A8=on` 在 NPU0 运行 VideoMME smoke 2。进程正常加载并完成端到端
视频推理，2/2 响应有效、1/2 正确，任务退出码为 0；结果目录为
`evaluation/output/20260812_104457`。该小样本只证明端到端可运行和输出合法，不替代
全量质量结论。

主 LLM TP2 支持以下显式配置：

```bash
OMNI_LLM_DEVICES=CANN0,CANN1
OMNI_LLM_SPLIT_MODE=tensor
OMNI_LLM_TENSOR_SPLIT=0.5,0.5
```

实现会校验设备存在性、拒绝 CPU、校验 split mode，并拒绝非有限数、负数或超出设备上限
的 tensor split；`OMNI_LLM_DEVICES=auto` 使用运行时 CANN 设备枚举，单卡时自动关闭
tensor split。`tensor` 会进入 llama.cpp 的 `LLAMA_SPLIT_MODE_TENSOR` Meta backend；
`row` 和 `layer` 仍保留用于兼容多设备配置，但不宣称为 CANN 张量并行。未设置环境变量
时维持原有单芯行为。代码已通过 `llama-omni-server` 构建和静态接线测试。另在 NPU0 上以
`OMNI_LLM_DEVICES=CANN0`、`OMNI_LLM_SPLIT_MODE=none` 完成一次真实完整 RTS，日志确认
新设备路径命中；all pooled RTF 为 `0.8742`，SPEAK→WAV 平均 `1099.0 ms`，LLM decode
阶段 RTF 为 `0.3626`。原始结果位于
`/workspace/MiniCPM--competition/result/llama_huawei_migration_tp2/tp2_single_explicit_20260812T104114Z`。
双卡 TP2 首次长链路在 compute arena 地址复用后触发 Meta split-state 递归，表现为
`ggml_backend_meta_get_split_state` 无限递归并最终段错误。修复为只淘汰身份不匹配的
单项缓存，不再清空递归遍历仍依赖的整个缓存。相同输入回归已连续完成 69 个 chunk，
all pooled RTF 为 `0.8304`，SPEAK→WAV 平均 `1106.3 ms`，未再出现段错误；原始结果位于
`/workspace/MiniCPM--competition/result/llama_huawei_migration_tp2_fix/tp2_fix_B1_20260812T131159Z`。

随后在同一构建、同一视频、F16、seed 42、ACL Graph/Weight NZ 关闭条件下完成
单卡/双卡/双卡/单卡（A/B/B/A）对照。A 使用 CANN0 单卡，B 使用 CANN0+CANN1、
`tensor`、`0.5,0.5`；每段均运行完整 120 秒链路：

| 指标 | 单卡 A（两次均值） | TP2 B（两次均值） | TP2 相对变化 |
| --- | ---: | ---: | ---: |
| all pooled RTF | 0.6606 | 0.8248 | `+24.85%`（变慢） |
| chunk mean RTF | 0.6725 | 0.8388 | `+24.73%`（变慢） |
| chunk P95 RTF | 0.8219 | 0.9794 | `+19.17%`（变慢） |
| core pooled RTF | 0.7135 | 0.9739 | `+36.49%`（变慢） |
| SPEAK→WAV 平均耗时 | 899.4 ms | 1085.7 ms | `+20.71%`（变慢） |
| LLM prefill RTF | 0.0142 | 0.0672 | `+374.91%`（变慢） |
| LLM decode RTF | 0.1687 | 0.2854 | `+69.18%`（变慢） |

四段原始结果位于
`/workspace/MiniCPM--competition/result/llama_huawei_migration_tp2_abba/` 下时间戳
`20260812T131424Z`、`T131557Z`、`T131747Z`、`T131939Z` 的目录。两次 B 均确认同一
server PID 同时驻留两张物理 NPU，主 LLM 日志记录 `devices=CANN0,CANN1 split_mode=3`，
因此是真实 tensor parallel，而不是设备配置空转。

为验证通信瓶颈，另做了 CANN HCCL AllReduce 原型：串行 rank 调度会阻塞，并发 rank
调度可完整运行，但 all pooled RTF 为 `1.0471`、SPEAK→WAV 为 `1350.6 ms`，比通用
P2P TP2 更差。该无收益实现和新增依赖已撤销，仅保留结果证据：
`/workspace/MiniCPM--competition/result/llama_huawei_migration_tp2_hccl_threads/tp2_hccl_threads_B1_20260812T132802Z`。
由于 TP2 未达到预设 `>=2%` pooled 或关键阶段 `>=5%` 的准入门槛，未继续 3+20，
并保持默认单卡路径不变；TP2 仅作为可用、稳定的显式实验功能保留。

在正式 Daily-Omni 使用物理 NPU1 期间，另用空闲物理 NPU0 完成一次隔离的 RTS
全链路验证。37/37 个 chunk 全部完成，生成 12 个 WAV 事件；core pooled RTF 为
`0.8882`，chunk mean/P95 为 `0.888/0.960`，SPEAK→WAV 平均/中位耗时为
`1067.0/1109.4 ms`。阶段 RTF 为 encode `0.1757`、LLM prefill `0.0157`、
LLM decode `0.3779`、TTS `0.2121`、Token2Wav `0.1068`。结果位于
`/workspace/MiniCPM--competition/result/llama_huawei_migration_rts_parallel_20260812T1341Z`。
该运行用于补强真实全链路稳定性证据；由于与 Daily-Omni 并发，不替代最终
`run_all.sh --full` 的隔离门禁，也不作为新的性能基线。

新 `tensor` 路径的单卡回退也以 `OMNI_LLM_DEVICES=auto`、
`OMNI_LLM_SPLIT_MODE=tensor` 和 `tensor_split=0.5,0.5` 完成真实 RTS。日志最终记录
`devices=auto split_mode=0`，证明只发现一张 CANN 卡时不会创建伪 TP；all pooled RTF
为 `0.8648`，chunk mean/P95 为 `0.8853/1.0512`，SPEAK→WAV 为 `1122.8 ms`，
结果位于 `/workspace/MiniCPM--competition/result/llama_huawei_migration_tp2_tensor_fallback/tp2_tensor_auto_single_20260812T114801Z`。

## 9. 保护文件校验

- `evaluation/` tracked tree：`f7c02a99eae0cda6df4b1806814fa049495add6e17c8b557201dcfdd64f30e43`
- `tools/omni/omni-eval-cli.cpp`：`f1d1a0c8169cd9ca356251854411d6c439ceeff234ad79b0bb25a312d5cbd457`
- `tools/omni/omni-eval-daily-cli.cpp`：`60c211b7e7889cea1a116b0a5155cc3fe748fd7d62898dab99936cd759fd03c6`
- `tools/omni/omni-tts-eval.cpp`：`e0dfbd068638f5d3f4df49dee2ea54f18fb9a6db7c548da319ae315c54178fb3`
- `tools/omni/CMakeLists.txt`：`8e67311fa4ff47ada77475f3b6e0edfb9fb49f65685bc4ec0feabcd71bfd7ca1`

`validate.sh static` 已重新计算上述值，结果全部一致。
