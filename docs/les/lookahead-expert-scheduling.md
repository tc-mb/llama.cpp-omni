# LES: Lookahead Expert Scheduling

English | [Chinese](lookahead-expert-scheduling.zh.md)

Running MoE on-device is only the starting point. The real challenge is **keeping the model's capability while generating at a usable speed**. Keeping expert weights in storage such as SSD lifts the memory limit, but if loading and computation do not overlap effectively, the repeated waits for loading slow generation down. Approaches that cap the number of experts or fix the expert set can be faster, but risk degrading model capability.

MiniCPM-V 4.7 35B-A3B addresses this with **LES (Lookahead Expert Scheduling)**, which co-optimizes expert loading, compute scheduling and weight access, and combines it with an **adapted DSpark speculative sampling** stage, reaching **lossless inference at 17.39 token/s on iPad Pro 11-inch M4 (16 GB)**.

## Core optimizations in LES

- **Expert prefetch and compute scheduling.** Cross-layer routing prediction loads the experts the next layer is likely to need ahead of time. Experts that the actual router selected and that are already loaded are computed first, while the remaining required experts are loaded in parallel, cutting load waits. Prediction is used only for prefetch and scheduling; computation still follows the model's own token-level routing, and any required expert missed by the prediction is loaded and used as usual. No additional training is needed.

- **Weight access optimization.** Expert loading is optimized along two axes, quantization and storage layout. Precise layer-wise mixed-precision quantization reduces weight storage and transfer cost while preserving model capability. Repacking the expert weights into contiguous storage reduces scattered reads and improves sequential read efficiency on flash. Together they lower what expert loading demands of on-device storage and bandwidth.

## Adapted DSpark speculative sampling

Speculative sampling is integrated into the on-device inference pipeline, working alongside LES expert scheduling and weight access optimizations to further improve decode efficiency.

## Results

Measured on **iPad Pro 11-inch M4 (16 GB)**. Test data covers MMMU, MathVista, CharXiv_reasoning_val, OCRBench, HumanEval, MBPP, AIME2025, AIME2026, Math500, and an internal text-dialogue Arena evaluation set.

The table shows measured results as each optimization is added. Decode speed is generated tokens divided by total decode time. The baseline is SSD expert loading with none of the optimizations below enabled. Both the gain over the previous stage and the gain over the baseline are listed.

| Optimization stage | Decode speed | vs previous | vs baseline |
| --- | ---: | ---: | ---: |
| Baseline: expert weights loaded from SSD, no optimizations below | 2.97 token/s | - | 1.00x |
| LES: lookahead expert scheduling and load optimization, including weight layout, fewer read calls, and direct writes into GPU-visible memory | 9.81 token/s | 3.30x | 3.30x |
| LES: plus precise layer-wise mixed-precision quantization | 14.92 token/s | 1.52x | 5.02x |
| Full LES + DSpark speculative sampling | **17.39 token/s** | **1.17x** | **5.86x** |

![LES](lookahead-expert-scheduling.png)
