# 社区任务清单（Community Roadmap）

这份文档规定**如何维护 llama.cpp-omni 的公开任务清单**。它不是给人随便看看的说明，
而是一份**可执行规范**：任何人、任何 AI agent、在任何机器上，照这里的步骤做，结果都一致。

公开的清单以一条置顶 GitHub issue 的形式呈现，社区同学在该 issue 下声明接任务。

---

## 0. 一句话总览

```
tasks.toml   --(校验+渲染)-->  GitHub 置顶 issue 正文
  ^                                    |
  |                                    v
唯一真相源                        社区在此认领
```

**`tasks.toml` 是唯一真相源。公开 issue 的正文是生成物，不要直接编辑 issue 正文。**
直接改 issue 正文会在下次同步时被覆盖，而且会与 `tasks.toml` 产生不可调和的差异。

---

## 1. 文件分工

| 文件 | 作用 | 谁改 |
|------|------|------|
| `tasks.toml` | 任务清单唯一真相源 | 维护者 / AI agent |
| `labels.toml` | 所有标签的定义（名称、颜色、用途） | 维护者 |
| `README.md` | 本规范 | 维护者 |
| `../../../scripts/roadmap/render.py` | 校验 + 渲染 issue 正文 | 不要手改 |
| `../../../scripts/roadmap/apply-labels.py` | 幂等创建/更新标签 | 不要手改 |
| `../../../scripts/roadmap/sync-issue.py` | 创建或更新置顶 issue | 不要手改 |

脚本全部**默认 dry-run**，必须显式加 `--apply` 才会写线上。这是刻意设计的：
任何改动线上仓库的动作都要人工确认。

---

## 2. 怎么加一个任务

### 2.1 手动步骤（4 步，不要跳）

1. **分配 ID**：找出 `tasks.toml` 里现有的最大编号 `T-NNN`，新任务用 `NNN+1`。
   **如果文件里还没有任何任务，从 `T-001` 开始。**
   编号**连续递增、永不复用、永不重排**；删掉旧任务后留下的空缺不要回填。
2. **在 `tasks.toml` 末尾追加一个 `[[tasks]]` 块**，字段见第 3 节。
3. **跑校验**：

   ```bash
   python3 scripts/roadmap/render.py --check
   ```

4. **预览渲染结果**（不联网，不写任何东西）：

   ```bash
   python3 scripts/roadmap/render.py
   ```

### 2.2 一个完整的追加示例

假设现在最大编号是 `T-007`，要加一个 TTS 音色替换的中等难度任务：

```toml
[[tasks]]
id = "T-008"
title = "Document the TTS voice-swap workflow end to end"
difficulty = "medium"
domain = "tts"
note = "Users keep asking how to replace the C++ TTS voice. Needs a reproducible walkthrough covering prompt_cache.gguf."
```

要带一个已存在的 issue 就加 `issue` 字段：

```toml
[[tasks]]
id = "T-009"
title = "Investigate n_ctx sub-model scheduling for full-duplex on 24GB GPUs"
difficulty = "hard"
domain = "duplex"
issue = 88
note = "Long-running research item. Keep the Research stage checklist in the linked issue up to date."
```

### 2.3 字段顺序

建议固定为 `id` -> `title` -> `difficulty` -> `domain` -> `issue` -> `owner` -> `note`。
顺序本身不参与校验，但固定顺序能让 diff 保持整洁、便于 review。

---

## 3. 字段规范

| 字段 | 必填 | 类型 | 说明 |
|------|------|------|------|
| `id` | 是 | 字符串 | 格式 `T-NNN`（至少 3 位数字），全文件唯一，**一旦分配不可更改** |
| `title` | 是 | 字符串 | 一句话描述，单行，不要换行符。见第 4 节的措辞规则 |
| `difficulty` | 是 | 字符串 | 只能取 `easy` / `medium` / `hard`，定义见第 5 节 |
| `domain` | 是 | 字符串 | 只能取第 6 节列出的领域值 |
| `issue` | 否 | 整数 | 已建的 GitHub issue 编号（不带 `#`）。没有就整行省略 |
| `owner` | 否 | 字符串 | 认领者 GitHub 用户名（不带 `@`）。无人认领就整行省略 |
| `note` | 否 | 字符串 | 补充说明：前置条件、参考文件、为什么难。单行 |
| `umbrella` | 否 | 布尔 | `true` 表示伞形条目，下面可以挂子任务 |
| `parent` | 否 | 字符串 | 所属伞形条目的 ID |
| `kind` | 否 | 字符串 | `task`（默认）/ `research`，后者渲染出 `research` 标记 |

校验器会拒绝未知字段、未知枚举值、重复 ID、非法 `id` 格式，以及第 3.1 节的
所有伞形关系违规。**如果你加了字段但校验没报错，说明校验器有 bug，请开 issue。**

### 3.1 伞形条目与子任务

一个任务如果**本身就是一项大工程**（例如"同步上游"、"在 WebGPU 上跑通"），
不要试图把它拆成一条条平铺的任务，而是：

1. 先写**伞形条目**，`umbrella = true`，难度取整体难度
2. 再写**子任务**，各自 `parent = "T-0XX"`，各自有**自己的难度与领域**

```toml
[[tasks]]
id = "T-020"
title = "Catch up the engine with upstream llama.cpp"
difficulty = "hard"
domain = "build"
umbrella = true
note = "Umbrella. Do not attempt the whole thing in one pull request."

[[tasks]]
id = "T-021"
title = "Phase 1: catch up src/ and ggml/ with upstream"
difficulty = "medium"
domain = "ggml"
parent = "T-020"
note = "The least entangled layers. Land this first."
```

**硬规则（校验器会强制）**：

| 规则 | 原因 |
|------|------|
| 伞形条目必须排在它的子任务**之前** | 文件是按 ID 递增读的，这同时排除了循环引用 |
| `parent` 指向的条目必须声明 `umbrella = true` | 防止挂到普通任务下面 |
| 子任务**不能**自己再是伞形 | 只支持一层，避免出现树状复杂度 |
| 伞形条目**必须至少有一个子任务** | 否则它就是个普通任务，`umbrella` 标记是误导 |

**子任务的难度是独立计算的。** 一个 hard 伞形下面可以挂 easy 子任务 —— 而且这正是
期望的用法。渲染时会额外生成一个**"Subtask by difficulty"索引**，把子任务按难度
重新分组列出，否则挂在 hard 伞形下的 easy 子任务会从新人视野里消失。

### 3.2 跨仓库任务

`domain` 为 `demo` 或 `app` 的任务**不修改本仓库代码**，实际改动在
`OpenBMB/MiniCPM-o-Demo` 或 `OpenBMB/MiniCPM-V-Apps`。

- `note` 里**必须写明实际仓库**，否则贡献者点进去才发现仓库不对
- 认领需要在**对应仓库**开 issue 与 PR，本仓库的指派流程对它不生效
- 维护者需要和对应仓库的维护者协调

---

## 4. 措辞规则

`title` 会直接出现在公开 issue 里，注意：

- **用英语写**。这是国际开源项目，`tasks.toml` 的注释可以中文，但 `title` / `note` 面向社区，用英语。
- **写成"要达成什么"，不要写成"要改哪个文件"**。
  - 好：`Add a VRAM usage note to the --vision-batch-encode docs`
  - 差：`Edit docs/multimodal.md`
- **不要写内部代号**。用 `vision encoder` 而不是 `VPM`，用 `audio encoder` 而不是 `APM`。
  理由：社区同学不知道内部代号，他要能搜到才有用。
- **不要写"不难"、"很简单"之类的评价**。难度用 `difficulty` 字段表达。

---

## 5. 难度定义

按**贡献者视角**定义，不是按代码复杂度。四个等级，分界线是**"路径是否已知"**：

| 值 | 标签 | 判据 | 典型特征 |
|----|------|------|----------|
| `easy` | `difficulty: easy` | 只碰 1 个文件；不需要理解 omni 架构；有现成范例可抄；能本地自测 | 补文档、修参数校验、改善错误提示、整理复现脚本 |
| `medium` | `difficulty: medium` | 要理解一个子模块；改动跨 2-3 个文件；可能需要跑模型验证 | 某个后端路径修复、streaming 行为调整、依赖/Python 侧问题 |
| `hard` | `difficulty: hard` | **路径已知，只是工程量大的事**：移植、重构、搬到已有后端。需要数周投入，但不需要发明任何东西 | 追上游版本、拆分大文件、把某个阶段搬到现有后端 |
| `expert` | `difficulty: expert` | **原创开发，方法尚未确定**：没有现成答案可抄，要先做设计、大概率要先写提案，且有走不通的风险 | 多 session 全双工的 session/KV 生命周期设计、算子级加速方向探索 |

### 5.1 hard 与 expert 怎么区分

这一条最容易判错，所以单列。问自己**下面这个问题**：

> **社区里有没有别人已经做成的同类方案可以直接参考？**

- **有**（例如 llama.cpp 上游已经拆好了 server 模块、某个后端已经有同类算子）→ `hard`。
  工作量大、时间长、要读懂很多代码，但每一步都是已知的。
- **没有**（例如 omni 的双工协议下没有人做过并发 session 调度）→ `expert`。
  需要先回答问题、做设计决策、验证可行性，**可能做完调研发现此路不通**。

`expert` 任务的 note 里应当写明**为什么路径未知**，并提示认领者先和维护者讨论方案，
不要直接开始写代码。

### 5.2 关于"要不要再加一档"

目前四档的边界分别是"是否需要架构知识"（easy/medium）、"工作量是否跨周"（medium/hard）、
"路径是否未知"（hard/expert），三个分界互不重叠。

再加第五档（例如 `expert` 之上再分"研究性"）会和现有 `kind = research` 字段重复 ——
**nature（性质）和 difficulty（难度）是两个正交轴**：一个 medium 的调研（例如盘点 WebGPU
算子覆盖率）是 `research` 但难度不高。所以不要靠加难度档来表达"这是研究"，用 `kind`。

### 5.3 easy 不等于新人安全

有的改动很简单但需要深厚上下文。真正对新人安全的 `easy` 任务，
**额外再打一个 `good first issue` 标签** —— 那是 GitHub 的特殊标签，会被 GitHub 自己的
发现页曝光，自建标签拿不到这个流量。

---

## 6. 领域（domain）取值

`domain` 必须取自下表，且与仓库里真实的代码路径对应。

| 值 | 覆盖路径 | 标签 |
|----|----------|------|
| `vision` | `tools/omni/vision.*`、`tools/mtmd/mtmd-image.*`、`tools/mtmd/clip*` | `vision` |
| `audio` | `tools/omni/audition.*`、`tools/mtmd/mtmd-audio.*` | `audio` |
| `tts` | `tools/omni/token2wav/`、`tts-condition-graph.*`、`voxcpm2/`、`tools/tts/` | `tts` |
| `duplex` | 全双工调度、打断、上下文/session 管理 | `duplex` |
| `server` | `tools/server/**`，含 `server-omni.cpp` | `server` |
| `model` | `src/models/**`、`src/llama-model.cpp` | `model` |
| `ggml` | `ggml/**` | `ggml` |
| `conversion` | `conversion/`、`tools/omni/convert/`、`tools/omni/pyt2w/`、`gguf-py/` | `conversion` |
| `python` | `**/*.py`、`requirements/` | `python` |
| `build` | `cmake/`、`CMakeLists.txt`、`CMakePresets.json` | `build` |
| `docs` | `docs/**`、`media/**` | `documentation` |
| `devops` | `.github/`、`ci/` | `devops` |
| `demo` | 跨仓库：`OpenBMB/MiniCPM-o-Demo` | `demo` |
| `app` | 跨仓库：`OpenBMB/MiniCPM-V-Apps` | `app` |

**加新领域之前先问自己**：这个领域有专门的负责人吗？没有的话，加了也没人清队列。
vLLM 的规矩值得照搬：一个标签要有**受众**、**负责人**、**配套规则**，缺一就不建。

---

## 7. 状态如何维护（这部分几乎不用手工做）

设计上，**任务状态不在 `tasks.toml` 里**,而是在各自的 issue 里：

| 状态 | 谁维护 | 怎么反映到清单 |
|------|--------|----------------|
| 认领 | 维护者在该 issue 上设 assignee | 自动（`--github` 模式读取） |
| 进行中 | PR 链接留在 issue 里 | 自动 |
| 已完成 | 关闭该 issue | 自动渲染为 `[x]` |
| 14 天无进展 | 维护者解除 assignee | 自动 |

所以 `tasks.toml` **只在"加任务 / 删任务 / 改难度领域"时才需要改**。
你不需要为了勾选 checkbox 去编辑任何文件。

**重要**：往 `tasks.toml` 里加 `issue = NNN` 之前，确认该 issue 已存在并且**不是** `roadmap`
issue 自身。

---

## 8. 发布流程

**所有线上写操作都需要人工确认。** 脚本默认 dry-run，不加 `--apply` 什么都不写。

```bash
# 步骤 1：校验（必做，出错会给出具体行号）
python3 scripts/roadmap/render.py --check

# 步骤 2：本地预览渲染结果（不联网）
python3 scripts/roadmap/render.py --out /tmp/roadmap-body.md
head -60 /tmp/roadmap-body.md

# 步骤 3：把 issue 状态拉进来，看最终渲染（需要 gh 已登录）
python3 scripts/roadmap/render.py --github --out /tmp/roadmap-body.md

# 步骤 4：建标签（先看 dry-run 输出）
python3 scripts/roadmap/apply-labels.py
python3 scripts/roadmap/apply-labels.py --apply

# 步骤 5：创建/更新置顶 issue（先看 dry-run 输出）
python3 scripts/roadmap/sync-issue.py
python3 scripts/roadmap/sync-issue.py --apply
```

### 8.1 发布前置条件

1. **`roadmap` 标签必须已存在。** 否则 issue 不会被 `close-issue.yml` 豁免，
   会在 44 天后被 stale bot 自动关闭。
2. 需要有 `tc-mb/llama.cpp-omni` 的写权限。
3. `gh auth status` 显示已登录。

---

## 9. 给 AI agent 的确定性操作流程

如果你是一个 AI agent，被要求"加一个任务"，严格按下面执行。**不要跳步，不要自作主张改结构。**

```
输入：一条任务描述（自然语言，可能是中文）

1. 读 docs/contributing/roadmap/README.md（本文件）与 tasks.toml
2. 找到 tasks.toml 中最大的 NNN，新 ID = NNN + 1；文件为空则新 ID = T-001
3. 判断 difficulty：对照第 5 节的判据表，明确说出你选它的理由。
   在 hard 与 expert 之间犹豫时，用第 5.1 节的那一个问题来定：社区里有没有别人已经
   做成的同类方案可参考？有 → hard，没有 → expert
4. 判断 domain：对照第 6 节的表，必须是表里的值
5. 如果用户给了 issue 编号，加上 issue 字段；否则省略
6. 如果这是一个大工程（无法在一个人几周内交 PR），改成两步：
   先写伞形条目（umbrella = true，难度取整体难度），再写子任务（parent = 伞形 ID，
   各自独立难度）。伞形必须至少有一个子任务
7. 按第 2.3 节的字段顺序，在 tasks.toml 末尾追加 [[tasks]] 块
8. 把 title 和 note 翻译成英语（第 4 节）
9. 运行 python3 scripts/roadmap/render.py --check
10. 如果校验失败，按报错修正后重跑，直到通过
11. 运行 python3 scripts/roadmap/render.py --out /tmp/roadmap-body.md
12. 向用户报告：新增的 ID、difficulty 及理由、domain、校验结果
13. 不要运行任何带 --apply 的命令，不要提交，不要 push
```

**禁止事项**：

- 不要改动已有任务的 `id`
- 不要给已有任务重新排序
- 不要直接编辑 GitHub issue 正文（会被覆盖）
- 不要运行 `--apply`、`git commit`、`git push`（必须人工确认）
- 不要新增 `difficulty` 或 `domain` 的取值；需要新增时先改第 5/6 节并说明理由
- 不要做多层嵌套的伞形（子任务不能再是伞形）
- 不要在 `domain` 为 `demo` / `app` 的任务里省略实际仓库名（必须写在 note 里）

---

## 10. 维护节奏

| 频率 | 动作 | 谁 |
|------|------|----|
| 有任务想法时 | 追加到 `tasks.toml` + 校验 | 维护者 / AI |
| 每周 | 确认 `good first issue` 队列非空；处理新认领 | 维护者 |
| 有人认领 | 在对应 issue 设 assignee | 维护者 |
| 每 14 天 | 检查僵尸认领，无进展的解除 assignee | 维护者 |
| 每季度 | 收口：删掉已无意义的条目，重跑发布流程 | 维护者 |

**一条纪律**：清单上每一条都是你对社区的承诺 —— 有人照着做完了，你就得给出 review。
所以**只列你愿意 review 的**。起步阶段控制在 10 条左右，跑通一轮再扩。
