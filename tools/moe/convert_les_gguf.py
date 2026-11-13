#!/usr/bin/env python3
"""GGUF -> GGUF: rewrite a model's routed MoE experts into the LES layout.

为什么需要它
------------
LES 的专家流式读取要求「一个专家 = 一条 preadv」，也就是每个专家的
`gate|up|down` 三块必须字节连续。官方 gguf 是按 family 存的（`ffn_up_exps`
的第一个专家距离 `ffn_gate_exps` 有 n_expert 个专家那么远），直接读就是三次
分散读；让张量保持原样则等于对整个文件做 mmap。

本脚本做的是**重排**，不是重算：不解量化、不重新量化、不改任何数值。输出文件
的常驻张量与专家字节与输入逐字节相同，`verify` 校验这一点。

和 llama-quantize 的分工
-------------------------
量化（Q3_K / Q4_K / Q6_K 的混合精度）只能在 `llama-quantize` 里完成，因为那是
字节最终定型的阶段；`convert_hf_to_gguf.py` 的 `--outtype` 只支持 f32/f16/bf16/
q8_0/tq1_0/tq2_0，产不出量化档。

所以对**已经量化好**的 gguf（例如官方发布的 `mix_q3gud_im`），本脚本是把它转成
LES 布局的唯一手段 —— 重新跑一遍量化会改变字节，那不是重排。

用法
----
  convert_les_gguf.py pack   --model <in.gguf> --out <out.gguf>
  convert_les_gguf.py verify --model <in.gguf> --packed <out.gguf>
  convert_les_gguf.py probe  --packed <out.gguf>

峰值内存是一层专家（约 350 MB）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# tools/moe/convert_les_gguf.py -> parents[2] = 仓库根（再拼 gguf-py）。脚本挪位置要同步改深度。
LLAMA_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LLAMA_ROOT / "gguf-py"))

import numpy as np  # noqa: E402
from gguf import GGMLQuantizationType, GGUFReader, GGUFWriter  # noqa: E402

# 一个专家内的三段。合并形态（gate_up）单独算一族。
GATE_UP_FAMILY = "ffn_gate_up_exps"
SPLIT_FAMILIES = ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps")
ALL_FAMILIES = SPLIT_FAMILIES + (GATE_UP_FAMILY,)

# 一个族在一个专家内的对齐。K 量化的 block 是 32 元素，字节数天然是 32 的倍数；
# 引擎侧的 slot 读取依赖这个前提。
TENSOR_ALIGN = 32
# 张量数据的对齐，写进 general.alignment。
#
# 必须让每个专家的读落在页边界上：一个专家的字节数（expert_stride）在常见量化下
# 是 4096 的倍数，只有区间起点也对齐到 4096，每次读才正好覆盖整数个页。按 GGUF
# 默认的 32 字节对齐，起点会落在任意页偏移，每个专家的读都要多触一页并在首尾各
# 做一段半页拷贝。
REGION_ALIGN = 4096

LAYOUT_KIND = "expert_major_packed"
LAYOUT_VERSION = 1
CARRIER_NAME = "blk.{layer}.moe_exps_packed.weight"

# 本工具产出的文件需要引擎实现的能力。
REQUIRES = ["moe.expert_major_packed_v1"]


class PackError(SystemExit):
    pass


# ---------------------------------------------------------------------------------------------
# 读源文件
# ---------------------------------------------------------------------------------------------

def family_of(tensor_name: str) -> tuple[int, str] | None:
    """`blk.7.ffn_up_exps.weight` -> (7, "ffn_up_exps")；不是路由专家则 None。"""
    if not tensor_name.startswith("blk.") or not tensor_name.endswith(".weight"):
        return None
    body = tensor_name[len("blk."):-len(".weight")]
    layer_text, _, family = body.partition(".")
    if family not in ALL_FAMILIES or not layer_text.isdigit():
        return None
    return int(layer_text), family


def scan(source: GGUFReader) -> tuple[dict[int, list], list, str, int, int]:
    """分出「常驻张量」和「每层的专家族」。"""
    arch_field = source.fields.get("general.architecture")
    if arch_field is None:
        raise PackError("source gguf has no general.architecture")
    arch = str(arch_field.contents())

    experts: dict[int, list] = {}
    resident = []
    for tensor in source.tensors:
        parsed = family_of(tensor.name)
        if parsed is None:
            resident.append(tensor)
            continue
        layer, family = parsed
        experts.setdefault(layer, []).append((family, tensor))

    if not experts:
        raise PackError(
            "source gguf has no routed expert tensors "
            f"({', '.join(ALL_FAMILIES)}) — nothing to repack")

    # 专家轴必须是最后一维，且每个专家的字节数能被整除 —— 这是「专家字节连续」的前提。
    n_expert = None
    for layer, families in experts.items():
        for family, tensor in families:
            shape = [int(v) for v in tensor.shape]
            if len(shape) < 3:
                raise PackError(f"'{tensor.name}' has shape {shape}, expected an expert axis")
            if n_expert is None:
                n_expert = shape[-1]
            elif shape[-1] != n_expert:
                raise PackError(
                    f"'{tensor.name}' has {shape[-1]} experts but an earlier tensor has {n_expert}")
            if tensor.n_bytes % n_expert:
                raise PackError(
                    f"'{tensor.name}' byte size {tensor.n_bytes} is not divisible by {n_expert}")

    for layer, families in experts.items():
        names = [f for f, _ in families]
        if GATE_UP_FAMILY in names and len(names) > 1:
            raise PackError(
                f"layer {layer} has {GATE_UP_FAMILY} alongside separate gate/up tensors")
        if GATE_UP_FAMILY not in names and not {"ffn_gate_exps", "ffn_up_exps"} <= set(names):
            raise PackError(
                f"layer {layer} has {sorted(names)}; expected {GATE_UP_FAMILY} or both gate and up")

    # 一个保守的几何校验：每层每个专家的三段字节数必须是 32 的倍数。
    for layer, families in experts.items():
        for family, tensor in families:
            per_expert = tensor.n_bytes // n_expert
            if per_expert % TENSOR_ALIGN:
                raise PackError(
                    f"'{tensor.name}': {per_expert} bytes per expert is not a multiple of {TENSOR_ALIGN}")

    used_field = source.fields.get(f"{arch}.expert_used_count")
    n_expert_used = int(used_field.contents()) if used_field is not None else 0

    return experts, resident, arch, int(n_expert), n_expert_used


def layer_layout(families: list) -> list[tuple[str, "object", int, int]]:
    """一层的族排序并给出每个族在专家内的偏移。

    顺序固定为 gate | up | down（合并形态只有 gate_up），这样同一个模型无论在
    源文件里张量怎么排，产出都一样。
    """
    order = {name: index for index, name in enumerate(ALL_FAMILIES)}
    families = sorted(families, key=lambda item: order[item[0]])

    offset = 0
    out = []
    for family, tensor in families:
        per_expert = tensor.n_bytes // int(tensor.shape[-1])
        if offset % TENSOR_ALIGN:
            raise PackError(f"'{tensor.name}': offset {offset} inside an expert is misaligned")
        out.append((family, tensor, offset, per_expert))
        offset += per_expert
    return out


def interleave(families: list[tuple[str, "object", int, int]], n_expert: int) -> np.ndarray:
    """把一层的族按 expert-major 交织成一个 [n_expert, stride] 的字节块。

    源张量的专家轴是 ne[-1]（最外层），所以一个专家的字节是连续的：
    `data[e*per_expert : (e+1)*per_expert]`。这里只做切片与拼接。
    """
    stride = sum(per_expert for _, _, _, per_expert in families)
    out = np.empty((n_expert, stride), dtype=np.uint8)
    for _, tensor, offset, per_expert in families:
        raw = np.ascontiguousarray(tensor.data).view(np.uint8).reshape(n_expert, per_expert)
        out[:, offset:offset + per_expert] = raw
    return out


# ---------------------------------------------------------------------------------------------
# pack
# ---------------------------------------------------------------------------------------------

def build_layout(experts: dict[int, list], n_expert: int, n_expert_used: int) -> dict:
    layers = []
    for layer in sorted(experts):
        families = layer_layout(experts[layer])
        stride = sum(per_expert for _, _, _, per_expert in families)
        widest = next((t for f, t, _, _ in families if f in ("ffn_gate_exps", GATE_UP_FAMILY)),
                      families[0][1])
        shape = [int(v) for v in widest.shape]
        layers.append({
            "layer": layer,
            "n_expert": n_expert,
            "n_expert_used": n_expert_used,
            "n_ff_exp": shape[1] if len(shape) > 1 else 0,
            "groups": [{
                "expert_begin": 0,
                "expert_count": n_expert,
                "expert_stride": stride,
                "tensor_name": CARRIER_NAME.format(layer=layer),
                "tensors": [{
                    "tensor_family": family,
                    "offset_in_expert": offset,
                    "bytes_per_expert": per_expert,
                    "type": tensor.tensor_type.name,
                    "ne": shape[:2],
                } for family, tensor, offset, per_expert in families],
            }],
        })
    return {
        "layout": LAYOUT_KIND,
        "alignment": REGION_ALIGN,
        "defaults": {"n_expert_used": n_expert_used},
        "layers": layers,
    }


def check_layout(layout: dict, tensors: dict | None) -> None:
    """执行 docs/les/les-gguf-format.md 的不变量。

    `pack` 写盘前和 `probe` 读入后跑同一份校验 —— 只有一个规则表，才不会出现
    「工具产得出、引擎读不了」的缝隙。
    """
    if layout.get("layout") != LAYOUT_KIND:
        raise PackError(f"layout '{layout.get('layout')}' is not '{LAYOUT_KIND}'")
    layers = layout.get("layers")
    if not layers:
        raise PackError("layout declares no layers")

    carriers = set()
    for layer in layers:
        n_expert = int(layer["n_expert"])
        covered = 0
        for group in layer["groups"]:
            begin = int(group["expert_begin"])
            count = int(group["expert_count"])
            stride = int(group["expert_stride"])
            if begin != covered:
                raise PackError(
                    f"layer {layer['layer']}: group starts at expert {begin}, expected {covered}")
            if count <= 0 or begin + count > n_expert:
                raise PackError(
                    f"layer {layer['layer']}: group [{begin},+{count}) exceeds n_expert={n_expert}")
            if stride % TENSOR_ALIGN:
                raise PackError(
                    f"layer {layer['layer']}: expert_stride {stride} is not a multiple of {TENSOR_ALIGN}")

            names, total = set(), 0
            for tensor in group["tensors"]:
                family = tensor["tensor_family"]
                if family in names:
                    raise PackError(f"layer {layer['layer']}: duplicate tensor '{family}'")
                names.add(family)
                offset = int(tensor["offset_in_expert"])
                per_expert = int(tensor["bytes_per_expert"])
                if offset % TENSOR_ALIGN or per_expert % TENSOR_ALIGN:
                    raise PackError(
                        f"layer {layer['layer']}: '{family}' is not {TENSOR_ALIGN}-byte aligned")
                if offset + per_expert > stride:
                    raise PackError(
                        f"layer {layer['layer']}: '{family}' runs past expert_stride "
                        f"({offset}+{per_expert} > {stride})")
                total += per_expert
            if total != stride:
                raise PackError(
                    f"layer {layer['layer']}: tensors sum to {total} but expert_stride is {stride}")

            name = group["tensor_name"]
            carriers.add(name)
            if tensors is not None:
                info = tensors.get(name)
                if info is None:
                    raise PackError(f"layout references tensor '{name}' but the file has none")
                if info.tensor_type != GGMLQuantizationType.I8:
                    raise PackError(f"'{name}' is {info.tensor_type.name}, expected I8")
                want = stride * count
                if info.n_bytes != want:
                    raise PackError(f"'{name}': {info.n_bytes} bytes, expected {want}")

            covered = begin + count
        if covered != n_expert:
            raise PackError(
                f"layer {layer['layer']}: groups cover {covered} of {n_expert} experts")

    if tensors is None:
        return

    # 常驻张量必须全部排在专家区之前：加载期的预读上界取第一个专家区的偏移。
    first_carrier = min(int(tensors[name].data_offset) for name in carriers)
    if first_carrier % REGION_ALIGN:
        raise PackError(
            f"expert region starts at {first_carrier}, not a multiple of {REGION_ALIGN}; "
            f"tensor data would be written with general.alignment={REGION_ALIGN}")
    for name, info in tensors.items():
        if name in carriers:
            continue
        end = int(info.data_offset) + info.n_bytes
        if end > first_carrier:
            raise PackError(
                f"resident tensor '{name}' ends at {end}, past the expert region at "
                f"{first_carrier}; resident tensors must come first")


def force_no_thinking(template: str) -> tuple[str, str]:
    """Rewrite the generation prompt so the assistant turn is an empty think block.

    Returns (new_template, note). When the template has no recognisable generation
    prompt, it is returned unchanged and `note` says why.

    Why rewrite at all: the upstream GGUF ships a template whose generation prompt
    emits `<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n`, i.e. thinking is already
    off — but it does so through a hand-edited line carrying a filler comment, and it
    ignores `enable_thinking` entirely (the variable does not occur in the template).
    Both facts make the prompt fragile: a converter that reformats the template, or a
    caller that expects the standard switch, silently gets a different prompt.

    So the output file states it plainly: the assistant turn opens with an empty think
    block, unconditionally. Thinking is not a user-facing option for this model in this
    runtime, and a template that pretends otherwise is the source of the confusion.
    """
    marker = "{%- if add_generation_prompt %}"
    i = template.rfind(marker)
    if i < 0:
        return template, "no add_generation_prompt block found; template left as-is"

    tail = template[i:]
    j = tail.find("{%- endif %}")
    if j < 0:
        return template, "unterminated add_generation_prompt block; template left as-is"

    new_block = (
        "{%- if add_generation_prompt %}\n"
        "    {{- '<|im_start|>assistant\\n' }}\n"
        "    {# LES: the assistant turn opens with an empty think block. #}\n"
        "    {{- '<think>\\n\\n</think>\\n\\n' }}\n"
        "{%- endif %}"
    )
    return template[:i] + new_block + tail[j + len("{%- endif %}"):], "rewritten"


def pack(args: argparse.Namespace) -> int:
    src = Path(args.model).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    if not src.exists():
        raise PackError(f"source gguf not found: {src}")
    if out.exists() and not args.force:
        raise PackError(f"output exists: {out} (pass --force to overwrite)")

    source = GGUFReader(src)
    experts, resident, arch, n_expert, n_expert_used = scan(source)
    layout = build_layout(experts, n_expert, n_expert_used)
    check_layout(layout, None)

    writer = GGUFWriter(out, arch, use_temp_file=True)
    # 所有张量按 REGION_ALIGN 对齐，专家区的起点才会是页对齐的（见 REGION_ALIGN 的说明）。
    writer.add_custom_alignment(REGION_ALIGN)

    # KV：原样搬运，但去掉三类：
    #   GGUF.*           - reader 暴露的伪字段，不是真 KV
    #   general.alignment- 由本工具重新声明
    #   les.moe.*        - 由本工具重新声明
    from gguf import GGUFValueType

    n_kv = 0
    for name, field in source.fields.items():
        if name == "general.architecture" or name.startswith("GGUF."):
            continue
        if name == "general.alignment" or name.startswith("les.moe."):
            continue
        vtype = field.types[0]
        subtype = field.types[-1] if vtype == GGUFValueType.ARRAY else None

        if name == "tokenizer.chat_template":
            template, note = force_no_thinking(str(field.contents()))
            if note != "rewritten":
                print(f"warning: chat template not rewritten ({note})")
            writer.add_key_value(name, template, vtype, sub_type=subtype)
            n_kv += 1
            continue

        writer.add_key_value(name, field.contents(), vtype, sub_type=subtype)
        n_kv += 1

    writer.add_uint32("les.moe.version", LAYOUT_VERSION)
    writer.add_array("les.moe.requires", REQUIRES)
    writer.add_string("les.moe.layout", json.dumps(layout, separators=(",", ":")))

    # 常驻张量在前：加载期的预读上界 = 第一个专家区的偏移。
    for tensor in resident:
        writer.add_tensor(tensor.name, tensor.data, raw_shape=tensor.data.shape,
                          raw_dtype=tensor.tensor_type, tensor_endianess=source.endianess)

    total_expert_bytes = 0
    for layer_entry in layout["layers"]:
        layer = layer_entry["layer"]
        group = layer_entry["groups"][0]
        block = interleave(layer_layout(experts[layer]), n_expert)
        total_expert_bytes += block.nbytes
        writer.add_tensor(group["tensor_name"], block.reshape(n_expert, 1, 1, group["expert_stride"]),
                          raw_shape=(n_expert, 1, 1, group["expert_stride"]),
                          raw_dtype=GGMLQuantizationType.I8)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    print(f"arch={arch}  resident_tensors={len(resident)}  layers={len(layout['layers'])}")
    print(f"expert_bytes={total_expert_bytes / 2**30:.2f} GiB  output={out} "
          f"({out.stat().st_size / 2**30:.2f} GiB)")
    return 0


# ---------------------------------------------------------------------------------------------
# verify / probe
# ---------------------------------------------------------------------------------------------

def verify(args: argparse.Namespace) -> int:
    """逐字节比对：常驻张量对源文件，每个专家区对源文件的交织结果。"""
    src = Path(args.model).expanduser().resolve()
    packed_path = Path(args.packed).expanduser().resolve()

    source = GGUFReader(src)
    experts, resident, _, n_expert, n_expert_used = scan(source)

    packed = GGUFReader(packed_path)
    by_name = {t.name: t for t in packed.tensors}

    field = packed.fields.get("les.moe.layout")
    if field is None:
        print("packed file has no les.moe.layout")
        return 1
    layout = json.loads(str(field.contents()))
    try:
        check_layout(layout, by_name)
    except PackError as error:
        print(f"layout invalid: {error}")
        return 1

    bad = 0
    checked = 0

    for tensor in resident:
        other = by_name.get(tensor.name)
        if other is None:
            print(f"missing resident tensor: {tensor.name}")
            bad += 1
            continue
        if other.tensor_type != tensor.tensor_type or other.n_bytes != tensor.n_bytes:
            print(f"type/size differs: {tensor.name} {other.tensor_type.name}/{other.n_bytes} != "
                  f"{tensor.tensor_type.name}/{tensor.n_bytes}")
            bad += 1
            continue
        if not np.array_equal(np.asarray(tensor.data).view(np.uint8),
                              np.asarray(other.data).view(np.uint8)):
            print(f"bytes differ: {tensor.name}")
            bad += 1
            continue
        checked += 1

    for layer_entry in layout["layers"]:
        layer = layer_entry["layer"]
        group = layer_entry["groups"][0]
        want = interleave(layer_layout(experts[layer]), n_expert)
        info = by_name[group["tensor_name"]]
        have = np.asarray(info.data).view(np.uint8).reshape(want.shape)
        if not np.array_equal(want, have):
            row = int(np.argwhere((want != have).any(axis=1))[0][0])
            print(f"{group['tensor_name']}: differs from the source interleave "
                  f"(first differing expert: {row})")
            bad += 1
            continue
        checked += 1

    if bad:
        print(f"FAILED: {bad} mismatches ({checked} checked)")
        return 1
    print(f"OK: {checked} byte-identical ({len(resident)} resident tensors + "
          f"{len(layout['layers'])} expert regions)")
    return 0


def probe(args: argparse.Namespace) -> int:
    """只读 header：逐层预算 + 布局摘要。"""
    path = Path(args.packed).expanduser().resolve()
    reader = GGUFReader(path)

    field = reader.fields.get("les.moe.layout")
    if field is None:
        raise PackError(f"{path} declares no les.moe.layout")

    layout = json.loads(str(field.contents()))
    by_name = {t.name: t for t in reader.tensors}
    check_layout(layout, by_name)

    carrier_names = {g["tensor_name"] for x in layout["layers"] for g in x["groups"]}
    per_layer = [{
        "layer": x["layer"],
        "n_expert": x["n_expert"],
        "n_expert_used": x["n_expert_used"],
        "bytes_per_slot": sum(int(g["expert_stride"]) for g in x["groups"]),
    } for x in layout["layers"]]

    resident_bytes = min(int(by_name[name].data_offset) for name in carrier_names)

    out = {
        "path": str(path),
        "arch": str(reader.fields["general.architecture"].contents()),
        "layout_version": int(reader.fields["les.moe.version"].contents())
        if "les.moe.version" in reader.fields else None,
        "layout": layout.get("layout"),
        "n_layer": len(per_layer),
        "bytes_per_slot_total": sum(x["bytes_per_slot"] for x in per_layer),
        "resident_bytes": resident_bytes,
        "resident_page_aligned": resident_bytes % REGION_ALIGN == 0,
        "expert_bytes": sum(int(by_name[name].n_bytes) for name in carrier_names),
        "layers": per_layer,
    }
    print(json.dumps(out, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Repack the routed MoE experts of a GGUF into the LES single-file layout")
    commands = parser.add_subparsers(dest="command", required=True)

    p_pack = commands.add_parser("pack", help="official gguf -> LES single-file gguf")
    p_pack.add_argument("--model", required=True, help="source gguf with inline experts")
    p_pack.add_argument("--out", required=True, help="output gguf")
    p_pack.add_argument("--force", action="store_true", help="overwrite the output")
    p_pack.set_defaults(func=pack)

    p_verify = commands.add_parser("verify", help="byte-compare the repack against its source")
    p_verify.add_argument("--model", required=True)
    p_verify.add_argument("--packed", required=True)
    p_verify.set_defaults(func=verify)

    p_probe = commands.add_parser("probe", help="print the per-layer budget from the header")
    p_probe.add_argument("--packed", required=True)
    p_probe.set_defaults(func=probe)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
