# LES single-file GGUF layout

This document describes the GGUF layout used by the MoE-SSD runtime in this repository to keep
routed expert weights **inside the model file**, and the metadata that describes that region.

It covers the format only. The scheduling side (look-ahead prefetch, decode window) is described
in [lookahead-expert-scheduling.md](lookahead-expert-scheduling.md).

## 1. Why the experts are repacked

The runtime streams experts from disk and reads **one expert per `preadv`**. For that to be a
single syscall, the three tensors that make up one expert must be byte-adjacent:

```
expert e of layer L  ->  region_base[L] + e * expert_stride
                         |- ffn_gate_exps[e]   bytes_per_expert
                         |- ffn_up_exps[e]     bytes_per_expert
                         '- ffn_down_exps[e]   bytes_per_expert
```

A standard GGUF stores them family-major: `blk.N.ffn_up_exps` begins `n_expert` experts after
`blk.N.ffn_gate_exps`. One expert is then three scattered reads, and leaving the tensors in that
form also means the whole file has to be mapped, which is what a phone cannot afford.

Repacking expert-major and recording that layout in the file itself lets the streamed read path
work while the resident weights stay mapped without dragging the experts in.

The repack is a **byte move**: it does not decode, requantise or reorder anything inside an
expert. The same input weights produce the same output bytes.

## 2. File layout

```
+--------------------------------------------------------+
| gguf header                                            |
| KV: general.* , <arch>.* , les.moe.*                   |
+--------------------------------------------------------+
| tensor_info[]                                          |
|   resident:  token_embd, output, blk.N.{attn,ssm,norm..}|
|   packed:    blk.N.moe_exps_packed.weight  (I8)         |
+--------------------------------------------------------+
| data: resident weights        <- mmap, prefetched      |
| data: packed expert regions   <- preadv only, not mapped|
|       layer 0 : [e0 gate|up|down][e1 ...]...[e_last]   |
|       layer 1 : ...                                    |
+--------------------------------------------------------+
```

**Resident tensors come first.** The loader caps its load-time prefetch at the offset of the first
packed region, so a file whose resident tensors were emitted after the experts would defeat it.
The offset is derived from the tensor table at load time; it is not stored in metadata, so it
cannot go stale.

**Alignment.** The packer sets `general.alignment = 4096`, which makes every tensor offset a
multiple of the page size. This matters because `expert_stride` is a multiple of 4096 in practice:
with it, one expert read touches exactly `expert_stride / 4096` pages. With GGUF's default 32-byte
alignment the region base would land at an arbitrary page offset and every expert read would
straddle one extra page.

`expert_stride` also stays a multiple of 32 (the K-quant block size), so the per-family offsets
inside an expert keep the alignment the 256-bit slot reads want.

## 3. The carrier tensor

Each group of experts is carried by one tensor whose bytes are the region verbatim:

```
name   blk.{L}.moe_exps_packed.weight
type   GGML_TYPE_I8            (1 byte per element, no block)
ne     [expert_stride, 1, 1, n_expert]        (GGUF stores ne reversed)
```

`I8` is not a quantised type: one element is one byte, so the tensor's size is exactly the region's
size, it overlaps nothing, and any standard reader sees a structurally valid file. Its meaning comes
entirely from `les.moe.layout`; no architecture binds to that name.

Why a carrier instead of overlapping the offsets of the three quantised tensors: every tensor keeps
a self-consistent, disjoint extent, so `gguf-dump`, `gguf-split` and friends cannot read shifted
data. A single quantised tensor could not hold a group anyway — one tensor has one `ggml_type`,
while a group's tensor families may differ, and `gate`/`up` are `[n_embd, n_ff]` while `down` is
`[n_ff, n_embd]`.

## 4. Metadata

Three entries in the KV section:

```
les.moe.version      uint32    1                       structural version
les.moe.layout   string    { ... }                 the layout, as JSON
les.moe.requires     string[]  ["moe.expert_major_packed_v1"]
```

`les.moe.requires` is separate from the layout so an engine can reject a file **before**
parsing it, naming the capability it does not implement.

The layout is a single JSON document rather than one KV per field, because the structure is
nested (layers -> groups -> tensors) and because most fields are absent in the common case. Binary
payloads do not belong in it: a block that is large or binary becomes its own tensor, referenced by
name.

```json
{
  "les_version": 1,
  "layout": "expert_major_packed",
  "alignment": 4096,
  "defaults": { "n_expert_used": 8, "router": {"kind": "linear"} },
  "layers": [
    {
      "layer": 0,
      "n_expert": 256,
      "n_expert_used": 8,
      "n_ff_exp": 512,
      "groups": [
        {
          "expert_begin": 0,
          "expert_count": 256,
          "expert_stride": 1351680,
          "tensor_name": "blk.0.moe_exps_packed.weight",
          "tensors": [
            {"tensor_family": "ffn_gate_exps", "offset_in_expert": 0,      "bytes_per_expert": 450560, "type": "Q3_K"},
            {"tensor_family": "ffn_up_exps",   "offset_in_expert": 450560, "bytes_per_expert": 450560, "type": "Q3_K"},
            {"tensor_family": "ffn_down_exps", "offset_in_expert": 901120, "bytes_per_expert": 450560, "type": "Q3_K"}
          ]
        }
      ],
      "router":    {"kind": "linear", "tensor": "blk.0.ffn_gate_inp.weight", "ne": [2048, 256]},
      "predictor": {"kind": "lookahead", "scope": 1}
    }
  ]
}
```

| field | default | meaning |
| --- | --- | --- |
| `les_version` | required | must be `1` |
| `layout` | required | must be `expert_major_packed` |
| `alignment` | required | power of two, >= 2048; matches `general.alignment` |
| `layers[]` | required | one entry per layer carrying routed experts |
| `layers[].n_expert` | required | expert count of this layer |
| `layers[].n_expert_used` | `defaults.n_expert_used` | top-k of this layer |
| `layers[].n_ff_exp` | required | expert feed-forward width |
| `layers[].groups[]` | required | one group per contiguous expert range with a uniform layout |
| `groups[].expert_begin` / `expert_count` | — | first expert id and how many |
| `groups[].expert_stride` | — | bytes of one expert in this group |
| `groups[].tensor_name` | — | carrier tensor name |
| `groups[].tensors[].tensor_family` | — | `ffn_gate_exps`, `ffn_up_exps`, `ffn_down_exps` |
| `groups[].tensors[].offset_in_expert` / `bytes_per_expert` | — | position and length inside one expert |
| `groups[].tensors[].type` | — | `ggml_type` name |
| `layers[].router.kind` | `linear` | `linear` uses `tensor`; `none` means the layer has no router |
| `layers[].router.tensor` / `ne` | `blk.{L}.ffn_gate_inp.weight` | router weight and shape |
| `layers[].predictor.kind` | `none` | `none` (engine heuristic), `lookahead`, `static_table` |
| `layers[].predictor.scope` | `1` | how many layers ahead the prediction covers |

Offsets in the layout are **relative to the tensor named by `groups[].tensor_name`**, never absolute
file offsets. Absolute positions are resolved from the tensor table at load time, so a file can be
re-ordered or shipped as gguf shards without touching the layout.

## 5. Invariants

The packer enforces these and the loader re-checks them; a file that violates one is rejected
rather than loaded:

```
align                          power of two, >= 2048
groups                         sorted, contiguous, start at 0, end at n_expert
expert_stride                     == sum(tensors[].bytes_per_expert)
expert_stride, tensors[].offset_in_expert, bytes_per_expert    % 32 == 0
offset_in_expert + bytes_per_expert   <= expert_stride
family names                   unique within a group
carrier tensor                 type == I8
carrier tensor                 nbytes == expert_stride * (expert_count)
router.ne                      == [n_embd, n_expert] when kind == linear
packed region                  begins at or after every resident tensor's end
packed region base             % align == 0
```

`tools/moe/pack_experts.py verify` checks the same table, plus a byte-for-byte comparison against
the source files.

## 6. What the engine does with it

At load time, before any weights are touched:

1. read `les.moe.requires` and reject the file if a listed capability is unknown;
2. parse `les.moe.layout` and validate the invariants above;
3. take each carrier tensor out of `weights_map` and record its `(file, offset)`.

Taking the carrier out of `weights_map` is what makes the layout safe with mmap: the mapping range
is computed from the tensors that remain, so the mapped window covers the resident prefix only,
and the end-of-load cleanup unmaps the expert region. Expert bytes are read through a plain file
layout with `preadv`, which the streaming runtime already did for the split layout.

From there the expert runtime sees exactly what it saw before: one path per layer, one
`expert_stride`, and the three families inside that stride. `finalize_pack` validates those three
properties and the read path is unchanged.

`llama_les_probe()` exposes the per-layer slot sizes from the header alone, so a caller can size an
expert cache before committing to a ~15 GiB load.

## 7. Tooling

```
tools/moe/pack_experts.py pack    --model <les.gguf> --experts <dir> --out <packed.gguf>
tools/moe/pack_experts.py verify  --model <les.gguf> --experts <dir> --packed <packed.gguf>
tools/moe/pack_experts.py probe   --packed <packed.gguf>
```

`pack` copies resident tensors raw, concatenates each layer's expert shard verbatim into its carrier
tensor, and writes the layout. `verify` compares every resident tensor and every packed region
byte for byte against the sources. `probe` prints the per-layer slot budget from the header.

Peak memory during `pack` is one tensor at a time (it streams in 8 MiB chunks), but the writer keeps
a temporary copy on disk, so a conversion host needs roughly twice the output size free.

## 8. Compatibility

Files in this layout are not interchangeable with standard GGUF files of the same model: the
expert region is only meaningful to an engine that reads `les.moe.layout`. Tools that rewrite
GGUF structure (`llama-quantize`, `gguf-split`) will preserve the tensors but invalidate the
meaning of that region, so a file must be repacked after any such operation.

`les.moe.version` identifies the layout. An engine that does not know the version refuses the file.
