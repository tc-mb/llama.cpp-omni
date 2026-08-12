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
- 13 个定向 C++/CANN 测试、10 个 Python wiring 测试全部通过。
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

本轮只有一张获准使用的空闲卡，因此未启用主 LLM 的 TP2；正式默认仍是单芯，
没有为跑分静默改变设备拓扑。

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

以下 PR 不进入本次 F16、`ACL_GRAPH=off` 的正式组合：PR #5、#7、#15–#20
是 vLLM-Omni/Python 模型实现；PR #12 只优化 ACL Graph vocoder bucket；PR #14 是
Q8 W8A8 opt-in 路径。它们不是遗漏，也不应混入本子赛道的 F16 正式默认。

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
| 设备/APM | TTS/Projector 分阶段设备、APM baseline/replay | 已迁移；不设置时维持单芯正式路径 | 语法/接口测试及单芯兼容验证 |
| 独立实验 | ACL Graph bucket、Q8 W8A8、主 LLM TP2 | 未进入本 PR 正式组合 | 计划要求与 F16/Graph-off 隔离；当前仅一张可用卡，且没有相应质量门禁证据 |

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
- 设备：物理 device `1`，单芯执行
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
`evaluation/output/20260812_003522`。截至本记录更新时，Video-MME 已完成
`680/900`，后台任务仍在运行；未出现进程崩溃或超时，已观察到 1 条非标准答案
`BC`，将按评测器原始结果如实计入有效率，不能提前声明 full 通过。Video-MME
结束后，统一脚本将继续执行 Daily-Omni、Seed-TTS 和 RTS。

另一次单卡 full 启动记录保存在 `evaluation/output/20260812_000128`。两次 full 尝试
都保留原始日志，不混入通过结论。四任务 smoke 与独立 3+20 RTS 均已完成；要补齐
full 验收，必须先提供完整 Video-MME 视频和真实 Daily-Omni 全量 annotation。

## 7. 构建与自动化验证

- 构建目标：`llama-omni-server`、`llama-omni-eval-cli`、
  `llama-omni-eval-daily-cli`、`llama-omni-tts-eval`，全部成功。
- CTest：13/13 通过，包括真实 CANN Im2col1D 测试。
- `test_cpp_oneshot_wiring.py`：5/5 通过。
- `test_runtime_optimization_wiring.py`：5/5 通过。
- `benchmarks/omni-huawei/validate.sh static`：通过。
- 最新静态验证目录：`benchmarks/omni-huawei/results/20260812T095039Z`。
- `git diff --check`：通过。

## 8. 保护文件校验

- `evaluation/` tracked tree：`f7c02a99eae0cda6df4b1806814fa049495add6e17c8b557201dcfdd64f30e43`
- `tools/omni/omni-eval-cli.cpp`：`f1d1a0c8169cd9ca356251854411d6c439ceeff234ad79b0bb25a312d5cbd457`
- `tools/omni/omni-eval-daily-cli.cpp`：`60c211b7e7889cea1a116b0a5155cc3fe748fd7d62898dab99936cd759fd03c6`
- `tools/omni/omni-tts-eval.cpp`：`e0dfbd068638f5d3f4df49dee2ea54f18fb9a6db7c548da319ae315c54178fb3`
- `tools/omni/CMakeLists.txt`：`8e67311fa4ff47ada77475f3b6e0edfb9fb49f65685bc4ec0feabcd71bfd7ca1`

`validate.sh static` 已重新计算上述值，结果全部一致。
