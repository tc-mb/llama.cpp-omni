#!/usr/bin/env python3
"""把「主 gguf + experts/ 目录」重打包成**一个** gguf。

为什么
------
分裂布局是 1 个主 gguf 加 40 个专家分片（`layer_000.bin` … `layer_039.bin`）再加一个
`manifest.json`。它成立的前提是加载器只预读常驻那段：整文件预读（`MAP_POPULATE` +
全文件 `WILLNEED`）会把专家一起在启动时读满。分片本身只是重排器一层一个文件留下的形态。

重排（expert-major，每专家 gate|up|down 连续）是**必须保留**的：`llama-moe-slot.cpp`
的 slot 读取路径要求一个 `preadv` 读完一个专家，不满足直接 throw，没有退路。所以本工具
只搬字节，不解交织、不解量化、不重排。

格式见 docs/les/les-gguf-format.md。

用法
----
  pack_experts.py pack   --model <les.gguf> --experts <dir> --out <out.gguf>
  pack_experts.py verify --model <les.gguf> --experts <dir> --packed <out.gguf>
  pack_experts.py probe  --packed <out.gguf>
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

# tools/moe/pack_experts.py -> parents[2] = 仓库根（再拼 gguf-py）。脚本挪位置要同步改深度。
LLAMA_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LLAMA_ROOT / "gguf-py"))

import numpy as np  # noqa: E402
from gguf import GGMLQuantizationType, GGUFReader, GGUFWriter  # noqa: E402

ROUTED_FAMILIES = ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps", "ffn_gate_up_exps")
FAMILY_ORDER = {name: index for index, name in enumerate(ROUTED_FAMILIES)}

# 一个 group 里所有段的对齐。K 量化的 block 是 32 元素，gate/up/down 的字节数天然是 32 的倍数。
FAMILY_ALIGN = 32
# packed 区起始对齐。与 llama.cpp 的 GGUF_DEFAULT_ALIGNMENT 一致；预读区间以它为粒度。
#
# 它还有一个硬性作用：**必须让每个专家的读落在页边界上**。一个专家的字节数
# （这个量化的 stride）是 32 的倍数，但 packed 区起点如果只按 GGUF 默认的 32 字节对齐，
# 起点的 4096 余数就不是 0，于是每个专家的 preadv 都会跨页（多触一页、首尾各一段
# 半页拷贝）。旧布局里每层是一个独立文件、专家 k 在 k*stride，天然页对齐；新布局
# 只有把区起点也对齐到 4096 才能保持同样的性质。
REGION_ALIGN = 4096

PACKED_NAME = "blk.{layer}.moe_exps_packed.weight"

LES_VERSION = 1
LES_LAYOUT = "expert_major_packed"


class PackError(SystemExit):
    pass


# ---------------------------------------------------------------------------------------------
# manifest -> 每层一份 layout
# ---------------------------------------------------------------------------------------------

def load_manifest(experts_dir: Path) -> tuple[dict, list[dict]]:
    path = experts_dir / "manifest.json"
    if not path.exists():
        raise PackError(f"experts manifest 不存在: {path}")
    manifest = json.loads(path.read_text())
    entries = manifest.get("entries", [])
    if not entries:
        raise PackError("manifest 里没有任何 entry")
    return manifest, entries


def layer_layouts(entries: list[dict]) -> dict[int, list[dict]]:
    """按层归集 family，按专家内偏移排序，并校验「一层一个 stride、段连续」。"""
    by_layer: dict[int, list[dict]] = {}
    for entry in entries:
        family = entry["tensor_family"]
        if family not in FAMILY_ORDER:
            raise PackError(f"未知的 routed family '{family}'（{entry['tensor_name']}）")
        by_layer.setdefault(int(entry["layer"]), []).append(entry)

    for layer, fams in by_layer.items():
        fams.sort(key=lambda item: item["repacked_offset"])
        names = [f["tensor_family"] for f in fams]
        if len(set(names)) != len(names):
            raise PackError(f"layer {layer}: 同一个 family 出现多次 {names}")

        strides = {int(f["expert_stride"]) for f in fams}
        if len(strides) != 1:
            raise PackError(f"layer {layer}: 层内 expert_stride 不一致 {sorted(strides)}")
        stride = strides.pop()

        cursor = 0
        for fam in fams:
            offset, size = int(fam["repacked_offset"]), int(fam["bytes_per_expert"])
            if offset != cursor:
                raise PackError(
                    f"layer {layer}: '{fam['tensor_family']}' 起始 {offset} != 期望 {cursor}"
                    "（段必须首尾相接、无空洞，否则一个 preadv 读不完一个专家）")
            if offset % FAMILY_ALIGN or size % FAMILY_ALIGN:
                raise PackError(
                    f"layer {layer}: '{fam['tensor_family']}' offset={offset} bytes={size}"
                    f" 不是 {FAMILY_ALIGN} 的倍数")
            cursor = offset + size

        if cursor != stride:
            raise PackError(
                f"layer {layer}: expert_stride={stride} != 各段之和 {cursor}")

        if len({f["repacked_file"] for f in fams}) != 1:
            raise PackError(f"layer {layer}: 层内 family 分散在多个分片文件里")

    return by_layer


def group_spec(manifest: dict, fams: list[dict], expert_count: int) -> dict:
    """一层 -> descriptor 里的一个 group（以字节为单位的 layout 声明）。

    注意这里**不写**源分片文件名：descriptor 会随发布物一起出门，而分片是打包过程的
    内部细节，不该留下痕迹。源文件名由调用方自己按层查。
    """
    layer = int(fams[0]["layer"])
    stride = int(fams[0]["expert_stride"])
    return {
        "expert_begin": 0,
        "expert_end": expert_count,
        "unit_bytes": stride,
        "tensor": PACKED_NAME.format(layer=layer),
        "families": [
            {
                "name": f["tensor_family"],
                "offset": int(f["repacked_offset"]),
                "bytes": int(f["bytes_per_expert"]),
                "type": f["quant_type"],
                # manifest 的 shape 是 gguf 的 ne 顺序（ne[0], ne[1], …），
                # 专家轴在最后一维，这里只留矩阵的两维。
                "ne": [int(v) for v in f["shape"][:2]],
            }
            for f in fams
        ],
    }


# ---------------------------------------------------------------------------------------------
# pack
# ---------------------------------------------------------------------------------------------

def copy_kv(reader: GGUFReader, writer: GGUFWriter) -> int:
    from gguf import GGUFValueType

    count = 0
    for name, field in reader.fields.items():
        # GGUF.* 是 reader 暴露的伪字段（version / tensor_count / kv_count），不是真 KV。
        if name == "general.architecture" or name.startswith("GGUF."):
            continue
        # 本工具的输出是单文件；源文件若带分片信息，留着会让下游以为还要找兄弟文件。
        if name.startswith("split."):
            continue
        # 对齐由本工具重新声明（见 REGION_ALIGN 的说明），源文件的旧值不能跟着走。
        if name == "general.alignment":
            continue
        vtype = field.types[0]
        subtype = field.types[-1] if vtype == GGUFValueType.ARRAY else None
        writer.add_key_value(name, field.contents(), vtype, sub_type=subtype)
        count += 1
    return count


def add_packed_tensor(writer: GGUFWriter, source: Path, spec: dict) -> None:
    """把一层分片原样搬成一个 I8 载体张量。

    I8 是「1 byte / 元素、无 block」的非量化类型（GGML_QUANT_SIZES[I8] == (1, 1)），所以
    这个张量的 nbytes 自洽、与任何别的张量不重叠，标准工具读它不会拿到错位数据；含义
    完全由 descriptor 赋予。写进去的字节顺序就是分片里的字节顺序：专家 e 在
    `e * unit_bytes`，一条 C 序 reshape 就够，不做任何交织。
    """
    expected = spec["unit_bytes"] * (spec["expert_end"] - spec["expert_begin"])
    if source.stat().st_size != expected:
        raise PackError(
            f"{source.name}: 大小 {source.stat().st_size} != unit_bytes×n_expert {expected}")

    codes = np.fromfile(source, dtype=np.int8)
    # numpy 序 (n_expert, 1, 1, unit_bytes) 反向写出即 gguf 的 ne = [unit_bytes, 1, 1, n_expert]。
    codes = codes.reshape(spec["expert_end"] - spec["expert_begin"], 1, 1, spec["unit_bytes"])
    writer.add_tensor(spec["tensor"], codes, raw_shape=codes.shape,
                      raw_dtype=GGMLQuantizationType.I8)


def pack(args: argparse.Namespace) -> int:
    src = Path(args.model).expanduser().resolve()
    experts_dir = Path(args.experts).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    if not src.exists():
        raise PackError(f"源 gguf 不存在: {src}")

    manifest, entries = load_manifest(experts_dir)
    model = manifest["model"]
    expert_count = int(model["expert_count"])
    expert_used = int(model["expert_used_count"])
    by_layer = layer_layouts(entries)

    reader = GGUFReader(src)
    arch = str(reader.fields["general.architecture"].contents())

    resident = [t for t in reader.tensors if t.name not in {
        e["original_gguf_tensor_name"] for e in entries}]
    dropped = len(reader.tensors) - len(resident)
    if dropped:
        print(f"源 gguf 里仍有 {dropped} 个专家张量，将被跳过（以 experts/ 为准）")

    # descriptor 先算出来：它要写进 KV，而 KV 必须在张量之前写完。
    layers = []
    for layer in sorted(by_layer):
        fams = by_layer[layer]
        by_name = {f["tensor_family"]: f for f in fams}
        wide = by_name.get("ffn_gate_exps") or by_name.get("ffn_gate_up_exps") or fams[0]
        layers.append({
            "layer": layer,
            "n_expert": expert_count,
            "expert_used": expert_used,
            "n_ff_exp": int(wide["shape"][1]),
            "groups": [group_spec(manifest, fams, expert_count)],
            "router": {
                "kind": "linear",
                "tensor": f"blk.{layer}.ffn_gate_inp.weight",
                "ne": [int(wide["shape"][0]), expert_count],
            },
            # 1.0 的前瞻是引擎侧的启发式，文件里不含预测权重。
            "predictor": {"kind": "lookahead", "scope": 1},
        })
    descriptor = {
        "les_version": LES_VERSION,
        "layout": LES_LAYOUT,
        "align": REGION_ALIGN,
        "defaults": {"expert_used": expert_used, "router": {"kind": "linear"}},
        "layers": layers,
    }
    # 写盘之前先按引擎侧同一套规则自检一遍。
    validate_descriptor(descriptor, None)

    writer = GGUFWriter(out, arch, use_temp_file=True)
    # 所有张量按 4096 对齐（见 REGION_ALIGN 的说明）：这样 packed 区起点是页对齐的，
    # 区内每个专家的读也就页对齐，和旧的「一层一个文件」布局一致。
    # 代价是常驻张量之间多出一点 padding，总共约 1 MiB 量级。
    writer.add_custom_alignment(REGION_ALIGN)
    n_kv = copy_kv(reader, writer)

    # 常驻权重在前：加载时的预读上界 = 第一个 packed 张量的偏移。
    n_resident = 0
    for tensor in resident:
        writer.add_tensor(tensor.name, tensor.data, raw_shape=tensor.data.shape,
                          raw_dtype=tensor.tensor_type, tensor_endianess=reader.endianess)
        n_resident += 1

    total_packed = 0
    for entry in layers:
        spec = entry["groups"][0]
        shard = by_layer[entry["layer"]][0]["repacked_file"]
        add_packed_tensor(writer, experts_dir / shard, spec)
        total_packed += spec["unit_bytes"] * expert_count

    writer.add_uint32("les.moe.version", LES_VERSION)
    writer.add_array("les.moe.requires", ["moe.expert_major_packed_v1"])
    writer.add_string("les.moe.descriptor", json.dumps(descriptor, separators=(",", ":")))

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    size = out.stat().st_size
    print(f"arch={arch} KV={n_kv}+3 常驻张量={n_resident} packed 层={len(by_layer)}")
    print(f"专家字节={total_packed / 2**30:.2f} GiB  输出={out} ({size / 2**30:.2f} GiB)")
    return 0


# ---------------------------------------------------------------------------------------------
# verify / probe
# ---------------------------------------------------------------------------------------------

def verify(args: argparse.Namespace) -> int:
    """逐字节比对：常驻张量对源 gguf，packed 区对分片拼接。"""
    src = Path(args.model).expanduser().resolve()
    experts_dir = Path(args.experts).expanduser().resolve()
    packed_path = Path(args.packed).expanduser().resolve()

    manifest, entries = load_manifest(experts_dir)
    expert_count = int(manifest["model"]["expert_count"])
    by_layer = layer_layouts(entries)

    source = GGUFReader(src)
    packed = GGUFReader(packed_path)
    packed_by_name = {t.name: t for t in packed.tensors}
    dropped = {e["original_gguf_tensor_name"] for e in entries}

    bad = 0
    checked = 0
    for tensor in source.tensors:
        if tensor.name in dropped:
            continue
        other = packed_by_name.get(tensor.name)
        if other is None:
            print(f"缺张量: {tensor.name}")
            bad += 1
            continue
        if other.tensor_type != tensor.tensor_type or other.n_bytes != tensor.n_bytes:
            print(f"类型/长度不符: {tensor.name} {tensor.tensor_type.name}/{tensor.n_bytes}"
                  f" vs {other.tensor_type.name}/{other.n_bytes}")
            bad += 1
            continue
        if not np.array_equal(np.asarray(tensor.data), np.asarray(other.data)):
            print(f"字节不符: {tensor.name}")
            bad += 1
            continue
        checked += 1

    for layer in sorted(by_layer):
        spec = group_spec(manifest, by_layer[layer], expert_count)
        shard = experts_dir / by_layer[layer][0]["repacked_file"]
        info = packed_by_name.get(spec["tensor"])
        if info is None:
            print(f"缺 packed 张量: {spec['tensor']}")
            bad += 1
            continue
        expect_bytes = spec["unit_bytes"] * expert_count
        if info.n_bytes != expect_bytes:
            print(f"{spec['tensor']}: nbytes {info.n_bytes} != {expect_bytes}")
            bad += 1
            continue
        # 分片 346 MB，流式比对，避免两边同时整层驻留。
        mismatch = False
        with shard.open("rb") as handle, tempfile.TemporaryFile() as scratch:
            np.asarray(info.data).tofile(scratch)
            scratch.seek(0)
            remaining = expect_bytes
            while remaining:
                chunk = min(8 << 20, remaining)
                if handle.read(chunk) != scratch.read(chunk):
                    mismatch = True
                    break
                remaining -= chunk
        if mismatch:
            print(f"{spec['tensor']}: 与 {by_layer[layer][0]['repacked_file']} 字节不符")
            bad += 1
            continue
        checked += 1

    if bad:
        print(f"FAILED: {bad} 处不符（已核对 {checked} 项）")
        return 1
    print(f"OK: {checked} 项逐字节一致（常驻张量 + {len(by_layer)} 层 packed 区）")
    return 0


def validate_descriptor(descriptor: dict, tensors: dict | None) -> None:
    """执行 docs/les/les-gguf-format.md §5.4 列出的不变量。

    pack 写完前和 probe 读进来后都跑同一份校验 —— 两个方向只有一套规则，才不会有
    「工具产得出、引擎读不了」的缝隙。
    """
    if descriptor.get("les_version") != LES_VERSION:
        raise PackError(f"les_version {descriptor.get('les_version')} != {LES_VERSION}")
    if descriptor.get("layout") != LES_LAYOUT:
        raise PackError(f"layout '{descriptor.get('layout')}' 不是 '{LES_LAYOUT}'")

    align = int(descriptor.get("align", REGION_ALIGN))
    if align < 2048 or align & (align - 1):
        raise PackError(f"align={align} 必须是 >= 2048 的 2 的幂")

    packed_names = set()
    for layer in descriptor["layers"]:
        n_expert = int(layer["n_expert"])
        groups = layer["groups"]
        cursor = 0
        for index, group in enumerate(groups):
            begin, end = int(group["expert_begin"]), int(group["expert_end"])
            if begin != cursor:
                raise PackError(
                    f"layer {layer['layer']}: group {index} 起点 {begin} != {cursor}"
                    "（group 必须按 expert id 连续覆盖，不留空洞）")
            cursor = end

            unit = int(group["unit_bytes"])
            if unit % FAMILY_ALIGN:
                raise PackError(f"layer {layer['layer']}: unit_bytes={unit} 不是 {FAMILY_ALIGN} 的倍数")

            names, total = set(), 0
            for family in group["families"]:
                name, offset, size = family["name"], int(family["offset"]), int(family["bytes"])
                if name in names:
                    raise PackError(f"layer {layer['layer']}: family '{name}' 重复")
                names.add(name)
                if offset % FAMILY_ALIGN or size % FAMILY_ALIGN:
                    raise PackError(
                        f"layer {layer['layer']}: '{name}' offset={offset} bytes={size}"
                        f" 不是 {FAMILY_ALIGN} 的倍数")
                if offset + size > unit:
                    raise PackError(
                        f"layer {layer['layer']}: '{name}' 越过 unit_bytes（{offset}+{size} > {unit}）")
                total += size
            if total != unit:
                raise PackError(
                    f"layer {layer['layer']}: 各段之和 {total} != unit_bytes {unit}")

            info = tensors.get(group["tensor"]) if tensors is not None else None
            if tensors is not None and info is None:
                raise PackError(f"descriptor 引用了不存在的张量 {group['tensor']}")
            if info is not None:
                if info.tensor_type != GGMLQuantizationType.I8:
                    raise PackError(
                        f"{group['tensor']} 类型是 {info.tensor_type.name}，必须是 I8")
                expect = unit * (end - begin)
                if info.n_bytes != expect:
                    raise PackError(
                        f"{group['tensor']}: nbytes {info.n_bytes} != unit_bytes×experts {expect}")
            packed_names.add(group["tensor"])

        if cursor != n_expert:
            raise PackError(
                f"layer {layer['layer']}: groups 覆盖到 {cursor}，但 n_expert={n_expert}")

        router = layer.get("router", {})
        if router.get("kind", "linear") == "linear":
            ne = router.get("ne")
            if ne is None or [int(v) for v in ne] != [int(ne[0]), n_expert]:
                raise PackError(
                    f"layer {layer['layer']}: router.ne {ne} 与 n_expert={n_expert} 不符")

    # 常驻必须在先：预读上界取第一个 packed 张量的偏移，前面若夹着常驻张量就会被漏读。
    if tensors is None:
        return
    first_packed = min(int(tensors[name].data_offset) for name in packed_names)

    # packed 区起点必须页对齐，否则每个专家的 preadv 都跨页（见 REGION_ALIGN 的说明）。
    if first_packed % REGION_ALIGN:
        raise PackError(
            f"packed 区起点 {first_packed} 不是 {REGION_ALIGN} 的倍数；"
            f"每个专家读会跨页。写文件时要用 add_custom_alignment({REGION_ALIGN})")

    for name, info in tensors.items():
        if name in packed_names:
            continue
        if int(info.data_offset) + info.n_bytes > first_packed:
            raise PackError(
                f"常驻张量 '{name}' 结束于 {int(info.data_offset) + info.n_bytes}，"
                f"越过 packed 区起点 {first_packed}（常驻必须全部排在 packed 之前）")


def probe(args: argparse.Namespace) -> int:
    """只读 header：逐层预算 + 描述符摘要，给端侧在加载模型之前算窗口。"""
    path = Path(args.packed).expanduser().resolve()
    reader = GGUFReader(path)
    field = reader.fields.get("les.moe.descriptor")
    if field is None:
        raise PackError(f"{path} 里没有 les.moe.descriptor，不是 LES 格式")
    descriptor = json.loads(str(field.contents()))

    packed_by_name = {t.name: t for t in reader.tensors}
    validate_descriptor(descriptor, packed_by_name)

    per_layer = []
    for layer in descriptor["layers"]:
        total = sum(int(group["unit_bytes"]) for group in layer["groups"])
        per_layer.append({"layer": layer["layer"], "n_expert": layer["n_expert"],
                          "expert_used": layer["expert_used"], "bytes_per_slot": total})

    packed_names = {g["tensor"] for layer in descriptor["layers"] for g in layer["groups"]}
    resident_bytes = min(int(packed_by_name[name].data_offset) for name in packed_names)
    out = {
        "path": str(path),
        "arch": str(reader.fields["general.architecture"].contents()),
        "les_version": int(reader.fields["les.moe.version"].contents())
        if "les.moe.version" in reader.fields else descriptor.get("les_version"),
        "layout": descriptor.get("layout"),
        "n_layer": len(per_layer),
        "bytes_per_slot_total": sum(item["bytes_per_slot"] for item in per_layer),
        "resident_bytes": resident_bytes,
        "resident_page_aligned": resident_bytes % REGION_ALIGN == 0,
        "expert_bytes": sum(int(packed_by_name[n].n_bytes) for n in packed_names),
        "layers": per_layer,
    }
    print(json.dumps(out, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="把 LES 的专家分片重打包进一个 gguf")
    commands = parser.add_subparsers(dest="command", required=True)

    p_pack = commands.add_parser("pack", help="主 gguf + experts/ -> 单 gguf")
    p_pack.add_argument("--model", required=True, help="源主 gguf（专家已剥离的 -les 版）")
    p_pack.add_argument("--experts", required=True, help="experts 目录（含 manifest.json）")
    p_pack.add_argument("--out", required=True, help="输出 gguf")
    p_pack.set_defaults(func=pack)

    p_verify = commands.add_parser("verify", help="逐字节比对重打包结果")
    p_verify.add_argument("--model", required=True)
    p_verify.add_argument("--experts", required=True)
    p_verify.add_argument("--packed", required=True)
    p_verify.set_defaults(func=verify)

    p_probe = commands.add_parser("probe", help="只读 header，打印逐层预算")
    p_probe.add_argument("--packed", required=True)
    p_probe.set_defaults(func=probe)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
