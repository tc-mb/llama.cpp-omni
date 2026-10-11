# LES: Lookahead Expert Scheduling

English | [Chinese](lookahead-expert-scheduling.zh.md)

LES keeps the routed MoE experts of **MiniCPM-V 4.7 35B-A3B** in flash memory rather than RAM,
reading them on demand and overlapping those reads with GPU computation already in progress. Since
reading a single expert (~1.00 ms) is comparable in cost to computing one layer (~1.15 ms), serving
reads on demand leaves disk latency as the limiting factor.

## Highlights

- **Lookahead expert** - the hidden state of layer L predicts the top-k of layer L+1 one layer in advance; the predicted experts are prefetched from SSD during layer L's computation, requiring no additional training and leaving the output unchanged
- **Scheduling optimization** - fewer and larger reads, direct writes into GPU-visible memory, page cache bypass, wider per-layer expert windows, and prewarming of frequently used experts
- **Fine-grained quantization** - a parameter search assigns precision per tensor group, reducing the precision of byte-intensive tensors while preserving accuracy-critical ones
- **DSpark** - a speculative decoding stage operating alongside LES on the decode path

![LES](lookahead-expert-scheduling.png)
