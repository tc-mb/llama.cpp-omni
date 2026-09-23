# Contributing to llama.cpp-omni

Thanks for your interest. This page is the entry point for new contributors.

> This page intentionally contains **no task details**. Everything here is a live
> link, so it stays accurate even if it is not updated for months. Task content
> lives in GitHub issues, which are always current.

---

## What this project is

`llama.cpp-omni` is a C++ inference engine for omni-modal models (vision, audio,
text, speech) built on top of [llama.cpp](https://github.com/ggml-org/llama.cpp).
It splits a model into separate GGUF modules: a vision encoder, an audio encoder,
the language model, and a text-to-speech head. It is best known for full-duplex
streaming, where audio input and output run at the same time.

If you are new to the code, start with the [README](../../README.md) and build it
once locally before picking up a task.

---

## Where to start

**Find a task**

- [Community task list](https://github.com/tc-mb/llama.cpp-omni/issues?q=is%3Aissue+is%3Aopen+label%3Aroadmap)
  - the roadmap, grouped by difficulty. **Start here.**
- [Good first issues](https://github.com/tc-mb/llama.cpp-omni/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)
  - small and newcomer-safe
- [Help wanted](https://github.com/tc-mb/llama.cpp-omni/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22)
  - we would like a hand with these
- [Open research areas](https://github.com/tc-mb/llama.cpp-omni/issues?q=is%3Aissue+is%3Aopen+label%3A%22research+%F0%9F%94%AC%22)
  - problems where we do not have an answer yet

**Or browse by area**

| Area | What it covers | Issues |
|------|----------------|--------|
| Vision | Vision encoder, image slicing, preprocessing | [vision](https://github.com/tc-mb/llama.cpp-omni/issues?q=is%3Aissue+is%3Aopen+label%3Avision) |
| Audio | Audio encoder, streaming audio input | [audio](https://github.com/tc-mb/llama.cpp-omni/issues?q=is%3Aissue+is%3Aopen+label%3Aaudio) |
| TTS | Speech synthesis, voice cloning, prompt cache | [tts](https://github.com/tc-mb/llama.cpp-omni/issues?q=is%3Aissue+is%3Aopen+label%3Atts) |
| Full-duplex | Scheduling, interruption, context management | [duplex](https://github.com/tc-mb/llama.cpp-omni/issues?q=is%3Aissue+is%3Aopen+label%3Aduplex) |
| Server | HTTP API, streaming endpoints | [server](https://github.com/tc-mb/llama.cpp-omni/issues?q=is%3Aissue+is%3Aopen+label%3Aserver) |
| Model | Architectures, new module support | [model](https://github.com/tc-mb/llama.cpp-omni/issues?q=is%3Aissue+is%3Aopen+label%3Amodel) |
| Backends | ggml operators and hardware paths | [ggml](https://github.com/tc-mb/llama.cpp-omni/issues?q=is%3Aissue+is%3Aopen+label%3Aggml) |
| Conversion | Model conversion and quantization | [conversion](https://github.com/tc-mb/llama.cpp-omni/issues?q=is%3Aissue+is%3Aopen+label%3Aconversion) |

**Or by difficulty**

[easy](https://github.com/tc-mb/llama.cpp-omni/issues?q=is%3Aissue+is%3Aopen+label%3A%22difficulty%3A+easy%22) -
[medium](https://github.com/tc-mb/llama.cpp-omni/issues?q=is%3Aissue+is%3Aopen+label%3A%22difficulty%3A+medium%22) -
[hard](https://github.com/tc-mb/llama.cpp-omni/issues?q=is%3Aissue+is%3Aopen+label%3A%22difficulty%3A+hard%22) -
[expert](https://github.com/tc-mb/llama.cpp-omni/issues?q=is%3Aissue+is%3Aopen+label%3A%22difficulty%3A+expert%22)

---

## How to claim a task

1. **Comment on the issue** linked from the task, saying how you plan to approach it.
2. A maintainer assigns it to you.
3. Open a pull request and link it in that issue.

**Please talk to us before writing code.** For anything above `easy`, we would
much rather agree on the approach first than review a large pull request that
went the wrong way.

Tasks are released if there is no pull request or draft pull request within
14 days of claiming. That is not a penalty - it just keeps tasks from being
parked. If you are stuck, say so and we will help break it down.

---

## Where to talk

We keep a simple split: **chat is for talking, GitHub is for deciding.**

| Where | Use it for |
|-------|-----------|
| **GitHub issues** | Claiming tasks, design discussion, decisions. **This is the only place that counts.** |
| [**Discord**](https://discord.com/invite/7q3ry8Ny8K) | Real-time chat, quick questions, getting unstuck. We use the OpenBMB community server - look for the Omni channel. |

Chat is fast, but it is not searchable forever and newcomers cannot find it.
**If a decision happens in chat, write the conclusion back into the issue.**
A decision that only lives in chat is lost.

Please write in English in issues and on Discord, so everyone can follow.

---

## Difficulty levels

All four levels mean the same thing everywhere in this repo. The line between
`hard` and `expert` is the useful one: `hard` work has a known path, `expert`
work does not.

| Level | What it means |
|-------|---------------|
| Easy | One file, no architecture knowledge needed, you can verify it locally. |
| Medium | You need to understand one submodule. Touches 2-3 files. May need a model run to verify. |
| Hard | Large but understood work: a port, a restructure, or offloading to an existing backend. Weeks of effort, but no invention required. |
| Expert | Original development. The approach is not known yet, so it needs design work and probably a written proposal before any code. Real risk of a dead end. |

`easy` does not automatically mean newcomer-safe. Tasks that are both are
additionally labelled `good first issue`.

For anything at `hard` or `expert`, please talk to a maintainer before starting -
on `expert` items the approach itself is usually still an open question.

---

## Other ways to help

- **Answer questions.** Many open issues are deployment questions. A good answer
  is a real contribution.
- **Improve the docs.** If something confused you, it will confuse the next
  person too.
- **Report bugs with a reproduction.** See the issue templates.

---

## For maintainers

The task list is generated. Do not edit the roadmap issue description by hand -
it is overwritten on the next sync.

- Spec and schema: [`docs/contributing/roadmap/README.md`](roadmap/README.md)
- Task data: [`docs/contributing/roadmap/tasks.toml`](roadmap/tasks.toml)
- Label definitions: [`docs/contributing/roadmap/labels.toml`](roadmap/labels.toml)

```bash
python3 scripts/roadmap/render.py --check          # validate
python3 scripts/roadmap/render.py --out /tmp/body.md
python3 scripts/roadmap/apply-labels.py            # dry run
python3 scripts/roadmap/sync-issue.py              # dry run
```

All three scripts are dry-run by default and require an explicit `--apply` to
write to GitHub.
