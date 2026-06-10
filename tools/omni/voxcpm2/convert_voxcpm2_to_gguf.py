#!/usr/bin/env python3
"""
Convert VoxCPM / VoxCPM2 PyTorch weights to GGUF format for llama.cpp-omni.

Supports both VoxCPM-0.5B (v1) and VoxCPM2 (v2) architectures.
Auto-detects version from config.json architecture field.

Usage:
    # VoxCPM2 (safetensors)
    python convert_voxcpm2_to_gguf.py \\
        --model /path/to/model.safetensors \\
        --vae /path/to/audiovae.pth \\
        --config /path/to/config.json \\
        --output /path/to/output_dir

    # VoxCPM-0.5B (pytorch_model.bin)
    python convert_voxcpm2_to_gguf.py \\
        --model /path/to/pytorch_model.bin \\
        --vae /path/to/audiovae.pth \\
        --config /path/to/config.json \\
        --output /path/to/output_dir

Output:
    output_dir/
    ├── VoxCPM2-BaseLM-F16.gguf     (or VoxCPM-0.5B-BaseLM-F16.gguf)
    └── VoxCPM2-Acoustic-F16.gguf   (or VoxCPM-0.5B-Acoustic-F16.gguf)
"""

import argparse
import json
import os
import struct
import sys
from pathlib import Path
import numpy as np

# Add llama.cpp gguf-py to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.join(SCRIPT_DIR, "..", "..", "..")
GGUF_PY_DIR = os.path.join(REPO_ROOT, "gguf-py")
sys.path.insert(0, GGUF_PY_DIR)

import gguf
from gguf import GGUFWriter, GGMLQuantizationType, Keys, RopeScalingType

# ==============================================================================
# PyTorch → GGUF tensor name mapping for BaseLM (LLM_ARCH_MINICPM)
# ==============================================================================
BASE_LM_GGUF_TENSOR_MAP = {
    # Embedding & norm
    "base_lm.embed_tokens.weight": "token_embd.weight",
    "base_lm.norm.weight": "output_norm.weight",
    # lm_head tied with embed_tokens → output.weight (duplicated)
    # Per-layer
    "base_lm.layers.{i}.input_layernorm.weight": "blk.{i}.attn_norm.weight",
    "base_lm.layers.{i}.self_attn.q_proj.weight": "blk.{i}.attn_q.weight",
    "base_lm.layers.{i}.self_attn.k_proj.weight": "blk.{i}.attn_k.weight",
    "base_lm.layers.{i}.self_attn.v_proj.weight": "blk.{i}.attn_v.weight",
    "base_lm.layers.{i}.self_attn.o_proj.weight": "blk.{i}.attn_output.weight",
    "base_lm.layers.{i}.post_attention_layernorm.weight": "blk.{i}.ffn_norm.weight",
    "base_lm.layers.{i}.mlp.gate_proj.weight": "blk.{i}.ffn_gate.weight",
    "base_lm.layers.{i}.mlp.up_proj.weight": "blk.{i}.ffn_up.weight",
    "base_lm.layers.{i}.mlp.down_proj.weight": "blk.{i}.ffn_down.weight",
}

# ==============================================================================
# PyTorch → GGUF tensor name mapping for Acoustic components
# ==============================================================================
ACOUSTIC_GGUF_PREFIX_MAP = {
    # ResidualLM (8 layers, no_rope)
    "residual_lm.norm.weight": "residual_lm.output_norm.weight",
    "residual_lm.layers.{i}.input_layernorm.weight": "residual_lm.blk.{i}.attn_norm.weight",
    "residual_lm.layers.{i}.self_attn.q_proj.weight": "residual_lm.blk.{i}.attn_q.weight",
    "residual_lm.layers.{i}.self_attn.k_proj.weight": "residual_lm.blk.{i}.attn_k.weight",
    "residual_lm.layers.{i}.self_attn.v_proj.weight": "residual_lm.blk.{i}.attn_v.weight",
    "residual_lm.layers.{i}.self_attn.o_proj.weight": "residual_lm.blk.{i}.attn_output.weight",
    "residual_lm.layers.{i}.post_attention_layernorm.weight": "residual_lm.blk.{i}.ffn_norm.weight",
    "residual_lm.layers.{i}.mlp.gate_proj.weight": "residual_lm.blk.{i}.ffn_gate.weight",
    "residual_lm.layers.{i}.mlp.up_proj.weight": "residual_lm.blk.{i}.ffn_up.weight",
    "residual_lm.layers.{i}.mlp.down_proj.weight": "residual_lm.blk.{i}.ffn_down.weight",
    # LocEnc (12 layers, non-causal)
    "feat_encoder.encoder.norm.weight": "locenc.norm.weight",
    "feat_encoder.encoder.layers.{i}.input_layernorm.weight": "locenc.blk.{i}.attn_norm.weight",
    "feat_encoder.encoder.layers.{i}.self_attn.q_proj.weight": "locenc.blk.{i}.attn_q.weight",
    "feat_encoder.encoder.layers.{i}.self_attn.k_proj.weight": "locenc.blk.{i}.attn_k.weight",
    "feat_encoder.encoder.layers.{i}.self_attn.v_proj.weight": "locenc.blk.{i}.attn_v.weight",
    "feat_encoder.encoder.layers.{i}.self_attn.o_proj.weight": "locenc.blk.{i}.attn_output.weight",
    "feat_encoder.encoder.layers.{i}.post_attention_layernorm.weight": "locenc.blk.{i}.ffn_norm.weight",
    "feat_encoder.encoder.layers.{i}.mlp.gate_proj.weight": "locenc.blk.{i}.ffn_gate.weight",
    "feat_encoder.encoder.layers.{i}.mlp.up_proj.weight": "locenc.blk.{i}.ffn_up.weight",
    "feat_encoder.encoder.layers.{i}.mlp.down_proj.weight": "locenc.blk.{i}.ffn_down.weight",
    "feat_encoder.in_proj.weight": "locenc.in_proj.weight",
    "feat_encoder.in_proj.bias": "locenc.in_proj.bias",
    "feat_encoder.special_token": "locenc.cls_token.weight",
    # LocDiT (12 layers, non-causal + CFG)
    "feat_decoder.estimator.decoder.norm.weight": "locdit.norm.weight",
    "feat_decoder.estimator.decoder.layers.{i}.input_layernorm.weight": "locdit.blk.{i}.attn_norm.weight",
    "feat_decoder.estimator.decoder.layers.{i}.self_attn.q_proj.weight": "locdit.blk.{i}.attn_q.weight",
    "feat_decoder.estimator.decoder.layers.{i}.self_attn.k_proj.weight": "locdit.blk.{i}.attn_k.weight",
    "feat_decoder.estimator.decoder.layers.{i}.self_attn.v_proj.weight": "locdit.blk.{i}.attn_v.weight",
    "feat_decoder.estimator.decoder.layers.{i}.self_attn.o_proj.weight": "locdit.blk.{i}.attn_output.weight",
    "feat_decoder.estimator.decoder.layers.{i}.post_attention_layernorm.weight": "locdit.blk.{i}.ffn_norm.weight",
    "feat_decoder.estimator.decoder.layers.{i}.mlp.gate_proj.weight": "locdit.blk.{i}.ffn_gate.weight",
    "feat_decoder.estimator.decoder.layers.{i}.mlp.up_proj.weight": "locdit.blk.{i}.ffn_up.weight",
    "feat_decoder.estimator.decoder.layers.{i}.mlp.down_proj.weight": "locdit.blk.{i}.ffn_down.weight",
    "feat_decoder.estimator.in_proj.weight": "locdit.in_proj.weight",
    "feat_decoder.estimator.in_proj.bias": "locdit.in_proj.bias",
    "feat_decoder.estimator.out_proj.weight": "locdit.out_proj.weight",
    "feat_decoder.estimator.out_proj.bias": "locdit.out_proj.bias",
    "feat_decoder.estimator.cond_proj.weight": "locdit.cond_proj.weight",
    "feat_decoder.estimator.cond_proj.bias": "locdit.cond_proj.bias",
    "feat_decoder.estimator.time_mlp.linear_1.weight": "locdit.time_mlp.linear_1.weight",
    "feat_decoder.estimator.time_mlp.linear_1.bias": "locdit.time_mlp.linear_1.bias",
    "feat_decoder.estimator.time_mlp.linear_2.weight": "locdit.time_mlp.linear_2.weight",
    "feat_decoder.estimator.time_mlp.linear_2.bias": "locdit.time_mlp.linear_2.bias",
    "feat_decoder.estimator.delta_time_mlp.linear_1.weight": "locdit.delta_time_mlp.linear_1.weight",
    "feat_decoder.estimator.delta_time_mlp.linear_1.bias": "locdit.delta_time_mlp.linear_1.bias",
    "feat_decoder.estimator.delta_time_mlp.linear_2.weight": "locdit.delta_time_mlp.linear_2.weight",
    "feat_decoder.estimator.delta_time_mlp.linear_2.bias": "locdit.delta_time_mlp.linear_2.bias",
    # FSQ
    "fsq_layer.in_proj.weight": "fsq.in_proj.weight",
    "fsq_layer.in_proj.bias": "fsq.in_proj.bias",
    "fsq_layer.out_proj.weight": "fsq.out_proj.weight",
    "fsq_layer.out_proj.bias": "fsq.out_proj.bias",
    # Projections
    "enc_to_lm_proj.weight": "projections.enc_to_lm_proj.weight",
    "enc_to_lm_proj.bias": "projections.enc_to_lm_proj.bias",
    "lm_to_dit_proj.weight": "projections.lm_to_dit_proj.weight",
    "lm_to_dit_proj.bias": "projections.lm_to_dit_proj.bias",
    "res_to_dit_proj.weight": "projections.res_to_dit_proj.weight",
    "res_to_dit_proj.bias": "projections.res_to_dit_proj.bias",
    "fusion_concat_proj.weight": "projections.res_fusion_proj.weight",
    "fusion_concat_proj.bias": "projections.res_fusion_proj.bias",
    # StopPredictor
    "stop_proj.weight": "stop_predictor.linear1.weight",
    "stop_proj.bias": "stop_predictor.linear1.bias",
    "stop_head.weight": "stop_predictor.linear2.weight",
}

# ==============================================================================
# AudioVAE PyTorch weight_norm → GGUF tensor name mapping
# ==============================================================================
# AudioVAE uses weight_norm: weight = w_g * w_v / ||w_v||_2
# We need to merge w_g and w_v into a single weight tensor before writing.
# Keys ending in .weight_v will be merged with corresponding .weight_g.


def merge_weight_norm(weight_g: np.ndarray, weight_v: np.ndarray) -> np.ndarray:
    """Merge PyTorch weight_norm: weight = w_g * w_v / ||w_v||_2 (along dim=0 for Conv1d)"""
    # weight_g shape: [OC, 1, 1], weight_v shape: [OC, IC, K]
    # weight_norm dim=0 normalizes over all dims except 0
    v_norm = np.linalg.norm(
        weight_v.reshape(weight_v.shape[0], -1), axis=1, keepdims=True
    )
    v_norm = v_norm.reshape(weight_g.shape[0], 1, 1)  # [OC, 1, 1]
    merged = weight_g * weight_v / v_norm
    return merged


# AudioVAE mapping: PyTorch key (without .weight_g/.weight_v suffix) → GGUF key
def map_audiovae_key(pt_key: str) -> str:
    """Map AudioVAE pytorch key to GGUF key with audio_vae. prefix"""
    # Remove .module. prefix if present
    clean = pt_key.replace(".module.", ".")
    return f"audio_vae.{clean}"


# ==============================================================================
# Tensor type helpers
# ==============================================================================
def to_f32_if_norm(name: str, tensor: np.ndarray) -> np.ndarray:
    """Convert norm/rms weights to F32 to avoid ggml CPU backend f32×f16 type mismatch.
    The ggml binary_op only supports f16×f32→f32 but not f32×f16→f32.
    build_norm does ggml_mul(f32_rms_out, weight), so weight must be F32.
    """
    if "norm" in name.lower() or "rms" in name.lower():
        return tensor.astype(np.float32)
    return tensor


def ensure_dtype(name: str, tensor: np.ndarray, target_dtype: str) -> np.ndarray:
    """Convert tensor to target dtype, with norm tensors always forced to F32.

    Norm tensors must be F32 due to ggml binary_op constraint:
    ggml_mul(f32_rms_out, weight) requires weight to be F32.
    """
    # Always force norm tensors to F32
    tensor = to_f32_if_norm(name, tensor)

    # For non-norm tensors, convert to target dtype if needed
    target_np = np.float16 if target_dtype == "f16" else np.float32
    if tensor.dtype != target_np and "norm" not in name.lower() and "rms" not in name.lower():
        tensor = tensor.astype(target_np)
    return tensor


# ==============================================================================
# BF16 conversion helper
# ==============================================================================
def bf16_to_f16(data: np.ndarray) -> np.ndarray:
    """Convert BF16 numpy array to F16.

    BF16 is stored as uint16 in safetensors with dtype=BF16.
    We interpret the bytes as bfloat16 → float32 → float16.
    """
    if data.dtype == np.uint16:
        # Reinterpret uint16 as bf16 (top 16 bits of float32)
        as_uint32 = data.astype(np.uint32) << 16
        as_f32 = as_uint32.view(np.float32)
        return as_f32.astype(np.float16)
    elif data.dtype == np.float16:
        return data
    elif data.dtype == np.float32:
        return data.astype(np.float16)
    else:
        raise ValueError(f"Unsupported dtype for BF16 conversion: {data.dtype}")


def _token_content(value):
    if isinstance(value, dict):
        return value.get("content")
    if isinstance(value, str):
        return value
    return None


def _merge_pair_to_string(pair):
    return " ".join(
        "".join(chr(ord(c) + 256) if c == " " else c for c in part) for part in pair
    )


def _write_no_vocab_tokenizer(gguf_writer: GGUFWriter, lm_cfg: dict):
    gguf_writer.add_tokenizer_model("no_vocab")
    gguf_writer.add_bos_token_id(lm_cfg.get("bos_token_id", 1))
    gguf_writer.add_eos_token_id(lm_cfg.get("eos_token_id", 2))


def _write_voxcpm2_tokenizer(
    gguf_writer: GGUFWriter, tokenizer_dir: str, vocab_size: int, lm_cfg: dict
):
    """Write the HF BPE tokenizer metadata needed by llama.cpp tokenization."""
    tokenizer_path = Path(tokenizer_dir) / "tokenizer.json"
    tokenizer_config_path = Path(tokenizer_dir) / "tokenizer_config.json"
    if not tokenizer_path.is_file():
        print(
            f"  WARNING: tokenizer.json not found in {tokenizer_dir}; writing no_vocab tokenizer metadata"
        )
        _write_no_vocab_tokenizer(gguf_writer, lm_cfg)
        return

    with open(tokenizer_path, "r", encoding="utf-8") as f:
        tokenizer = json.load(f)
    if tokenizer_config_path.is_file():
        with open(tokenizer_config_path, "r", encoding="utf-8") as f:
            tokenizer_config = json.load(f)
    else:
        tokenizer_config = {}

    model = tokenizer.get("model", {})
    if model.get("type") != "BPE" or not model.get("byte_fallback", False):
        raise ValueError("VoxCPM2 tokenizer must be a byte-fallback BPE tokenizer")

    vocab = model.get("vocab", {})
    reverse_vocab = {int(tid): tok for tok, tid in vocab.items()}
    for tok in tokenizer.get("added_tokens", []):
        if (
            isinstance(tok, dict)
            and isinstance(tok.get("id"), int)
            and isinstance(tok.get("content"), str)
        ):
            reverse_vocab.setdefault(int(tok["id"]), tok["content"])

    special_ids = {
        int(tok["id"])
        for tok in tokenizer.get("added_tokens", [])
        if isinstance(tok, dict)
        and tok.get("special")
        and isinstance(tok.get("id"), int)
    }
    for entry in tokenizer_config.get("added_tokens_decoder", {}).values():
        if isinstance(entry, dict) and entry.get("special") and "content" in entry:
            tid = vocab.get(entry["content"])
            if tid is not None:
                special_ids.add(int(tid))
    for tid, entry in tokenizer_config.get("added_tokens_decoder", {}).items():
        if isinstance(entry, dict) and entry.get("special"):
            try:
                special_ids.add(int(tid))
            except ValueError:
                pass

    token_lookup = dict(vocab)
    token_lookup.update({tok: tid for tid, tok in reverse_vocab.items()})
    bos_token = _token_content(tokenizer_config.get("bos_token")) or "<s>"
    eos_token = _token_content(tokenizer_config.get("eos_token")) or "</s>"
    unk_token = _token_content(tokenizer_config.get("unk_token")) or model.get(
        "unk_token", "<unk>"
    )
    pad_token = _token_content(tokenizer_config.get("pad_token"))

    bos_id = token_lookup.get(bos_token, lm_cfg.get("bos_token_id", 1))
    eos_id = token_lookup.get(eos_token, lm_cfg.get("eos_token_id", 2))
    unk_id = token_lookup.get(unk_token, lm_cfg.get("unk_token_id", 0))
    pad_id = token_lookup.get(pad_token) if pad_token else None

    if bos_id is not None:
        special_ids.add(int(bos_id))
    if eos_id is not None:
        special_ids.add(int(eos_id))
    if unk_id is not None:
        special_ids.add(int(unk_id))
    if pad_id is not None:
        special_ids.add(int(pad_id))

    tokens = []
    scores = []
    toktypes = []
    for tid in range(vocab_size):
        token = reverse_vocab.get(tid)
        if token is None:
            tokens.append(f"[PAD{tid}]".encode("utf-8"))
            scores.append(-1000.0)
            toktypes.append(gguf.TokenType.UNUSED)
            continue

        tokens.append(token.encode("utf-8"))
        scores.append(-1000.0)
        if tid == unk_id:
            toktypes.append(gguf.TokenType.UNKNOWN)
        elif tid in special_ids:
            toktypes.append(gguf.TokenType.CONTROL)
        elif len(token) == 6 and token.startswith("<0x") and token.endswith(">"):
            toktypes.append(gguf.TokenType.BYTE)
        else:
            toktypes.append(gguf.TokenType.NORMAL)

    merges = model.get("merges", [])
    if merges and isinstance(merges[0], list):
        merges = [_merge_pair_to_string(pair) for pair in merges]

    gguf_writer.add_tokenizer_model("llama")
    gguf_writer.add_tokenizer_pre("default")
    gguf_writer.add_token_list(tokens)
    gguf_writer.add_token_scores(scores)
    gguf_writer.add_token_types(toktypes)
    if merges:
        gguf_writer.add_token_merges(merges)

    if bos_id is not None:
        gguf_writer.add_bos_token_id(int(bos_id))
    if eos_id is not None:
        gguf_writer.add_eos_token_id(int(eos_id))
    if unk_id is not None:
        gguf_writer.add_unk_token_id(int(unk_id))
    if pad_id is not None:
        gguf_writer.add_pad_token_id(int(pad_id))

    gguf_writer.add_add_bos_token(bool(tokenizer_config.get("add_bos_token", True)))
    gguf_writer.add_add_eos_token(bool(tokenizer_config.get("add_eos_token", False)))
    if chat_template := tokenizer_config.get("chat_template"):
        gguf_writer.add_chat_template(chat_template)

    print(
        f"  Tokenizer: {len(tokens)} tokens, {len(merges)} merges, bos={bos_id}, eos={eos_id}, unk={unk_id}"
    )


# ==============================================================================
# Safetensors loader (works without torch)
# ==============================================================================
class SafeTensorFile:
    """Lightweight safetensors reader that doesn't require torch."""

    def __init__(self, path: str, target_dtype: str = "f16"):
        self.path = path
        self.target_dtype = target_dtype  # "f16" or "f32"
        self._file = open(path, "rb")
        header_size = struct.unpack("<Q", self._file.read(8))[0]
        self.header = json.loads(self._file.read(header_size).decode("utf-8"))
        self._data_start = 8 + header_size

    def keys(self):
        return sorted(self.header.keys())

    def get_shape(self, key: str) -> list:
        return self.header[key]["shape"]

    def get_dtype(self, key: str) -> str:
        return self.header[key]["dtype"]

    def get_tensor(self, key: str) -> np.ndarray:
        if key not in self.header:
            raise KeyError(key)
        info = self.header[key]
        # Calculate data offset
        offset = self._data_start
        for k in sorted(self.header.keys()):
            if k == key:
                break
            tensor_start, tensor_end = self.header[k]["data_offsets"]
            offset += tensor_end - tensor_start
        else:
            tensor_start, tensor_end = info["data_offsets"]
            offset += tensor_start

        start, end = info["data_offsets"]
        self._file.seek(offset)
        raw = self._file.read(end - start)

        shape = info["shape"]
        dtype = info["dtype"]

        if dtype == "F32":
            return np.frombuffer(raw, dtype=np.float32).reshape(shape).copy()
        elif dtype == "F16":
            data = np.frombuffer(raw, dtype=np.float16).reshape(shape).copy()
            if self.target_dtype == "f32":
                return data.astype(np.float32)
            return data
        elif dtype == "BF16":
            data = np.frombuffer(raw, dtype=np.uint16).reshape(shape).copy()
            # Convert BF16 → F32 or BF16 → F16
            as_uint32 = data.astype(np.uint32) << 16
            as_f32 = as_uint32.view(np.float32)
            if self.target_dtype == "f16":
                return as_f32.astype(np.float16)
            return as_f32
        elif dtype == "I32":
            return np.frombuffer(raw, dtype=np.int32).reshape(shape).copy()
        elif dtype == "I64":
            return np.frombuffer(raw, dtype=np.int64).reshape(shape).copy()
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")

    def close(self):
        self._file.close()


# ==============================================================================
# Dict-based weight source (for pytorch_model.bin)
# ==============================================================================
class DictWeightSource:
    """Weight source backed by an in-memory dict of numpy arrays."""

    def __init__(self, weights: dict, target_dtype: str = "f16"):
        self.weights = weights
        self.target_dtype = target_dtype
        self.target_np = np.float16 if target_dtype == "f16" else np.float32

    def keys(self):
        return sorted(self.weights.keys())

    def get_shape(self, key: str) -> list:
        return list(self.weights[key].shape)

    def get_dtype_str(self, key: str) -> str:
        dtype = self.weights[key].dtype
        if dtype == np.float32:
            return "F32"
        elif dtype == np.float16:
            return "F16"
        elif dtype == np.int32:
            return "I32"
        elif dtype == np.int64:
            return "I64"
        return "F32"

    def get_tensor(self, key: str) -> np.ndarray:
        if key not in self.weights:
            raise KeyError(key)
        tensor = self.weights[key].copy()
        if tensor.dtype == np.float64:
            tensor = tensor.astype(np.float32)
        if tensor.dtype == np.float32 and self.target_dtype == "f16":
            tensor = tensor.astype(np.float16)
        elif tensor.dtype == np.float16 and self.target_dtype == "f32":
            tensor = tensor.astype(np.float32)
        return tensor

    def close(self):
        self.weights.clear()


def _infer_audiovae_encoder_dim(vae_weights: dict) -> int:
    """Infer AudioVAE encoder_dim from encoder.block.0.bias shape."""
    for key in ("encoder.block.0.bias", "audio_vae.encoder.block.0.bias"):
        tensor = vae_weights.get(key)
        if tensor is not None:
            return int(tensor.shape[0] if hasattr(tensor, 'shape') else len(tensor))
    return 128


def _infer_audiovae_decoder_dim(vae_weights: dict) -> int:
    """Infer AudioVAE decoder_dim from decoder.model.1.bias shape."""
    for key in ("decoder.model.1.bias", "audio_vae.decoder.model.1.bias"):
        tensor = vae_weights.get(key)
        if tensor is not None:
            return int(tensor.shape[0] if hasattr(tensor, 'shape') else len(tensor))
    return 2048


def _infer_audiovae_latent_dim(vae_weights: dict) -> int:
    """Infer AudioVAE latent_dim from encoder.fc_mu.bias shape."""
    for key in ("encoder.fc_mu.bias", "audio_vae.encoder.fc_mu.bias"):
        tensor = vae_weights.get(key)
        if tensor is not None:
            return int(tensor.shape[0] if hasattr(tensor, 'shape') else len(tensor))
    return 64


def _conv_stride_from_weight(tensor) -> int:
    """Infer convolution stride from weight kernel size: stride = kernel // 2."""
    if tensor is None:
        return -1
    shape = tensor.shape if hasattr(tensor, 'shape') else list(tensor.shape)
    if len(shape) < 3:
        return -1
    kernel = int(shape[-1])
    if kernel % 2 != 0:
        return -1
    return kernel // 2


def _get_weight_tensor(vae_weights: dict, base_key: str) -> object:
    """Get a weight tensor from vae_weights, handling both plain .weight and weight_norm (.weight_g / .weight_v) formats.

    Priority: .weight_v > .weight > .weight_g — because weight_v has the full [OC, IC, K] shape
    with the true kernel size, while weight_g is just [OC, 1, 1].
    """
    # Try plain weight first (safetensors / merged format)
    tensor = vae_weights.get(base_key + ".weight")
    if tensor is not None:
        return tensor
    # Try weight_v next: has the full kernel shape [OC, IC, K]
    tensor = vae_weights.get(base_key + ".weight_v")
    if tensor is not None:
        return tensor
    # Try weight_g as last resort
    return vae_weights.get(base_key + ".weight_g")


def _infer_audiovae_encoder_rates(vae_weights: dict) -> list:
    """Infer encoder_rates from AudioVAE encoder block stride=2 conv layers."""
    rates = []
    for i in range(1, 16):
        base_key = f"encoder.block.{i}.block.4"
        tensor = _get_weight_tensor(vae_weights, base_key)
        if tensor is None:
            tensor = _get_weight_tensor(vae_weights, f"audio_vae.{base_key}")
        if tensor is None:
            break
        stride = _conv_stride_from_weight(tensor)
        if stride <= 0:
            break
        rates.append(stride)
    return rates if rates else [2, 5, 8, 8]


def _infer_audiovae_decoder_rates(vae_weights: dict) -> list:
    """Infer decoder_rates from AudioVAE decoder model stride=2 conv layers."""
    rates = []
    for i in range(2, 32):
        base_key = f"decoder.model.{i}.block.1"
        tensor = _get_weight_tensor(vae_weights, base_key)
        if tensor is None:
            tensor = _get_weight_tensor(vae_weights, f"audio_vae.{base_key}")
        if tensor is None:
            break
        stride = _conv_stride_from_weight(tensor)
        if stride <= 0:
            break
        rates.append(stride)
    return rates if rates else [8, 6, 5, 2, 2, 2]


def infer_audio_vae_config(config: dict, vae_state_dict: dict) -> dict:
    """Build a complete AudioVAE config, inferring values from weights when missing."""
    if "audio_vae_config" in config and isinstance(config["audio_vae_config"], dict):
        return dict(config["audio_vae_config"])

    resolved = {}
    resolved["encoder_dim"] = _infer_audiovae_encoder_dim(vae_state_dict)
    resolved["latent_dim"] = _infer_audiovae_latent_dim(vae_state_dict)
    resolved["decoder_dim"] = _infer_audiovae_decoder_dim(vae_state_dict)
    resolved["encoder_rates"] = _infer_audiovae_encoder_rates(vae_state_dict)
    resolved["decoder_rates"] = _infer_audiovae_decoder_rates(vae_state_dict)

    # VoxCPM-0.5B defaults. cond_type, sr_bin_boundaries, out_sample_rate are left
    # unset so export_acoustic can apply version-appropriate defaults.
    resolved["sample_rate"] = 16000
    resolved["out_sample_rate"] = 16000

    print(f"  Inferred AudioVAE config: encoder_dim={resolved['encoder_dim']}, "
          f"decoder_dim={resolved['decoder_dim']}, latent_dim={resolved['latent_dim']}")
    print(f"    encoder_rates={resolved['encoder_rates']}, decoder_rates={resolved['decoder_rates']}")
    return resolved


def detect_model_version(config: dict) -> float:
    """Detect VoxCPM architecture version from config.

    Returns:
        0.5 = VoxCPM-0.5B (arch=voxcpm, patch=2, no fusion_concat_proj)
        1.5 = VoxCPM-1.5  (arch=voxcpm, patch=4, no fusion_concat_proj)
        2.0 = VoxCPM2     (arch=voxcpm2, patch=4, has fusion_concat_proj)

    The version float determines structural defaults (fusion path, no_rope, etc.),
    NOT the output file name.
    """
    arch = config.get("architecture", "")
    if isinstance(arch, str) and arch.strip().lower() == "voxcpm2":
        return 2.0

    # architecture == "voxcpm": distinguish 0.5B (patch=2) from 1.5 (patch=4)
    if config.get("patch_size") == 2:
        return 0.5
    return 1.5


def infer_model_name(config: dict, model_path_str: str, version: float) -> str:
    """Infer a human-friendly model name for output file naming.

    Priority:
      1. config.model_name / config.name fields
      2. Path hints (0.5B, 1.5, VoxCPM2, etc.)
      3. Version-based fallback
    """
    # 1. Explicit config fields
    for key in ("model_name", "name", "model_id"):
        val = config.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()

    # 2. Path hints
    path_lower = model_path_str.lower()
    patterns = [
        ("0.5b", "VoxCPM-0.5B"),
        ("0.5", "VoxCPM-0.5B"),
        ("1.5b", "VoxCPM-1.5"),
        ("1.5", "VoxCPM-1.5"),
        ("2b", "VoxCPM2"),
        ("voxcpm2", "VoxCPM2"),
        ("minicpm-o", "VoxCPM2"),
    ]
    for pattern, name in patterns:
        if pattern in path_lower:
            return name

    # 3. Version-based fallback
    if version == 0.5:
        return "VoxCPM-0.5B"
    if version == 1.5:
        return "VoxCPM-1.5"
    return "VoxCPM2"


# ==============================================================================
# BaseLM GGUF writer
# ==============================================================================
def export_baselm(
    sf, config: dict, output_path: str, tokenizer_dir: str, dtype: str = "f16", version: float = 2.0
):
    """Export BaseLM as standard LLM_ARCH_MINICPM GGUF."""
    lm_cfg = config["lm_config"]
    n_layer = lm_cfg["num_hidden_layers"]
    n_embd = lm_cfg["hidden_size"]
    n_ff = lm_cfg["intermediate_size"]
    n_head = lm_cfg["num_attention_heads"]
    n_kv_head = lm_cfg["num_key_value_heads"]
    n_embd_head_k = lm_cfg.get("kv_channels", n_embd // n_head)
    n_rot = n_embd_head_k
    rms_eps = lm_cfg.get("rms_norm_eps", 1e-5)
    rope_theta = lm_cfg.get("rope_theta", 10000.0)
    context_length = lm_cfg.get("max_position_embeddings", 32768)
    vocab_size = lm_cfg.get("vocab_size", 73448)
    scale_emb = lm_cfg.get("scale_emb", 12)
    dim_model_base = lm_cfg.get("dim_model_base", 256)
    scale_depth = lm_cfg.get("scale_depth", 1.4)
    use_mup = lm_cfg.get("use_mup", False)
    rope_scaling = lm_cfg.get("rope_scaling", None)

    # PyTorch: when use_mup=False, scale_emb is forced to 1.0 (no scaling).
    # Set embedding_scale=0.0 in GGUF so llama.cpp skips the scaling op.
    f_embedding_scale = float(scale_emb) if use_mup else 0.0
    # PyTorch applies residual scaling only when use_mup=True. A non-zero GGUF
    # value makes llama.cpp scale both attention and FFN residual branches.
    f_residual_scale = (scale_depth / np.sqrt(float(n_layer))) if use_mup else 0.0
    f_logit_scale = dim_model_base / n_embd

    print(
        f"  BaseLM: {n_layer} layers, n_embd={n_embd}, n_head={n_head}, n_kv_head={n_kv_head}"
    )
    print(f"          n_ff={n_ff}, vocab={vocab_size}, ctx={context_length}")
    print(
        f"          scale_emb={f_embedding_scale}, residual_scale={f_residual_scale:.4f}, logit_scale={f_logit_scale:.4f}"
    )

    arch = "minicpm4"
    gguf_writer = GGUFWriter(output_path, arch)

    # Metadata — Keys use {arch} placeholder, must format before passing
    gguf_writer.add_uint32(Keys.LLM.CONTEXT_LENGTH.format(arch=arch), context_length)
    gguf_writer.add_uint32(Keys.LLM.EMBEDDING_LENGTH.format(arch=arch), n_embd)
    gguf_writer.add_uint32(Keys.LLM.BLOCK_COUNT.format(arch=arch), n_layer)
    gguf_writer.add_uint32(Keys.LLM.FEED_FORWARD_LENGTH.format(arch=arch), n_ff)
    gguf_writer.add_uint32(Keys.LLM.VOCAB_SIZE.format(arch=arch), vocab_size)

    gguf_writer.add_uint32(Keys.Attention.HEAD_COUNT.format(arch=arch), n_head)
    gguf_writer.add_uint32(Keys.Attention.HEAD_COUNT_KV.format(arch=arch), n_kv_head)
    gguf_writer.add_float32(Keys.Attention.LAYERNORM_RMS_EPS.format(arch=arch), rms_eps)
    gguf_writer.add_uint32(Keys.Attention.KEY_LENGTH.format(arch=arch), n_embd_head_k)
    gguf_writer.add_uint32(Keys.Attention.VALUE_LENGTH.format(arch=arch), n_embd_head_k)

    gguf_writer.add_uint32(Keys.Rope.DIMENSION_COUNT.format(arch=arch), n_rot)
    gguf_writer.add_float32(Keys.Rope.FREQ_BASE.format(arch=arch), rope_theta)
    gguf_writer.add_bool(Keys.Rope.SCALING_FINETUNED.format(arch=arch), True)

    # MiniCPM-specific scales
    gguf_writer.add_float32(
        Keys.LLM.EMBEDDING_SCALE.format(arch=arch), f_embedding_scale
    )
    gguf_writer.add_float32(Keys.LLM.RESIDUAL_SCALE.format(arch=arch), f_residual_scale)
    gguf_writer.add_float32(Keys.LLM.LOGIT_SCALE.format(arch=arch), f_logit_scale)

    # LongRoPE scaling factors
    if rope_scaling and rope_scaling.get("type") == "longrope":
        gguf_writer.add_string(
            Keys.Rope.SCALING_TYPE.format(arch=arch), RopeScalingType.LONGROPE.value
        )
        gguf_writer.add_uint32(
            Keys.Rope.SCALING_ORIG_CTX_LEN.format(arch=arch),
            rope_scaling.get("original_max_position_embeddings", context_length),
        )
        long_factors = rope_scaling.get("long_factor", None)
        short_factors = rope_scaling.get("short_factor", None)
    else:
        gguf_writer.add_string(Keys.Rope.SCALING_TYPE, RopeScalingType.NONE.value)
        long_factors = None
        short_factors = None

    # LongRoPE per-layer factors (64 head_dims/2 values per factor)
    n_rope_factors = n_rot // 2  # 64 for n_rot=128

    _write_voxcpm2_tokenizer(gguf_writer, tokenizer_dir, vocab_size, lm_cfg)

    # Write tensors
    print("  Writing BaseLM tensors...")

    # Embedding
    embed = sf.get_tensor("base_lm.embed_tokens.weight")  # [vocab, embd]
    gguf_writer.add_tensor("token_embd.weight", ensure_dtype("token_embd.weight", embed, dtype))
    # lm_head tied with embedding → duplicate as output.weight
    gguf_writer.add_tensor("output.weight", ensure_dtype("output.weight", embed, dtype))
    # Final norm (F32 to avoid ggml f32×f16 type mismatch in build_norm)
    gguf_writer.add_tensor(
        "output_norm.weight",
        to_f32_if_norm("output_norm.weight", sf.get_tensor("base_lm.norm.weight")),
    )

    # Store LongRoPE factors as GGUF metadata (not tensors) to avoid per-layer
    # naming mismatch with llama.cpp's MiniCPM loader. VoxCPM2Transformer reads
    # them from these metadata keys, falling back to hardcoded defaults if absent.

    # Per-layer tensors
    for i in range(n_layer):
        pt_prefix = f"base_lm.layers.{i}"
        gguf_prefix = f"blk.{i}"

        gguf_writer.add_tensor(
            f"{gguf_prefix}.attn_norm.weight",
            to_f32_if_norm(
                "attn_norm", sf.get_tensor(f"{pt_prefix}.input_layernorm.weight")
            ),
        )
        gguf_writer.add_tensor(
            f"{gguf_prefix}.attn_q.weight",
            ensure_dtype("attn_q.weight", sf.get_tensor(f"{pt_prefix}.self_attn.q_proj.weight"), dtype),
        )
        gguf_writer.add_tensor(
            f"{gguf_prefix}.attn_k.weight",
            ensure_dtype("attn_k.weight", sf.get_tensor(f"{pt_prefix}.self_attn.k_proj.weight"), dtype),
        )
        gguf_writer.add_tensor(
            f"{gguf_prefix}.attn_v.weight",
            ensure_dtype("attn_v.weight", sf.get_tensor(f"{pt_prefix}.self_attn.v_proj.weight"), dtype),
        )
        gguf_writer.add_tensor(
            f"{gguf_prefix}.attn_output.weight",
            ensure_dtype("attn_output.weight", sf.get_tensor(f"{pt_prefix}.self_attn.o_proj.weight"), dtype),
        )
        gguf_writer.add_tensor(
            f"{gguf_prefix}.ffn_norm.weight",
            to_f32_if_norm(
                "ffn_norm",
                sf.get_tensor(f"{pt_prefix}.post_attention_layernorm.weight"),
            ),
        )
        gguf_writer.add_tensor(
            f"{gguf_prefix}.ffn_gate.weight",
            ensure_dtype("ffn_gate.weight", sf.get_tensor(f"{pt_prefix}.mlp.gate_proj.weight"), dtype),
        )
        gguf_writer.add_tensor(
            f"{gguf_prefix}.ffn_up.weight",
            ensure_dtype("ffn_up.weight", sf.get_tensor(f"{pt_prefix}.mlp.up_proj.weight"), dtype),
        )
        gguf_writer.add_tensor(
            f"{gguf_prefix}.ffn_down.weight",
            ensure_dtype("ffn_down.weight", sf.get_tensor(f"{pt_prefix}.mlp.down_proj.weight"), dtype),
        )

        # Per-layer LongRoPE factors — DISABLED
        # llama.cpp loads global rope_factors for MINICPM; per-layer factors
        # cause tensor count mismatch (313 vs 257 expected by loader).
        # if long_factors:
        #     gguf_writer.add_tensor(f"{gguf_prefix}.rope_factors_long",
        #                            np.array(long_factors, dtype=np.float32))
        #     gguf_writer.add_tensor(f"{gguf_prefix}.rope_factors_short",
        #                            np.array(short_factors, dtype=np.float32))

    # LongRoPE factors for llama.cpp MINICPM loader
    # MINICPM arch uses GLOBAL names (no blk.%d. prefix) for these tensors.
    # Layer 0 loads as TENSOR_NOT_REQUIRED; layers 1+ are TENSOR_DUPLICATED.
    if long_factors:
        gguf_writer.add_tensor("rope_factors_long.weight",
                               np.array(long_factors, dtype=np.float32))
        gguf_writer.add_tensor("rope_factors_short.weight",
                               np.array(short_factors, dtype=np.float32))

    gguf_writer.open_output_file(Path(output_path))
    gguf_writer.write_header_to_file()
    gguf_writer.write_kv_data_to_file()
    gguf_writer.write_tensors_to_file()
    gguf_writer.close()
    print(f"  → {output_path}")


# ==============================================================================
# Acoustic GGUF writer
# ==============================================================================
def export_acoustic(
    sf, vae_state_dict: dict, config: dict, output_path: str, dtype: str = "f16", version: float = 2.0
):
    """Export acoustic components as custom voxcpm-acoustic GGUF."""
    lm_cfg = config["lm_config"]
    encoder_cfg = config.get("encoder_config", {})
    dit_cfg = config.get("dit_config", {})
    vae_cfg = config.get("audio_vae_config", {})

    n_embd = lm_cfg["hidden_size"]

    # Version-dependent defaults
    if version == 0.5:      # VoxCPM-0.5B
        def_n_res_lm = 6
        def_n_enc = 4
        def_n_dit = 4
        def_patch = 2
        def_fsq_dim = 256
        def_no_rope = False
        def_dec_dim = 1536
        def_out_sr = 16000
        def_cond_type = "none"
        def_sr_bins = []
    elif version == 1.5:    # VoxCPM-1.5
        def_n_res_lm = 8
        def_n_enc = 8
        def_n_dit = 8
        def_patch = 4
        def_fsq_dim = 256
        def_no_rope = False
        def_dec_dim = 2048
        def_out_sr = 44100
        def_cond_type = "none"
        def_sr_bins = []
    else:                   # VoxCPM2 (2.0)
        def_n_res_lm = 8
        def_n_enc = 12
        def_n_dit = 12
        def_patch = 4
        def_fsq_dim = 512
        def_no_rope = True
        def_dec_dim = 2048
        def_out_sr = 48000
        def_cond_type = "scale_bias"
        def_sr_bins = [20000, 30000, 40000]

    n_residual_layers = config.get("residual_lm_num_layers", def_n_res_lm)
    n_encoder_layers = encoder_cfg.get("num_layers", def_n_enc)
    n_dit_layers = dit_cfg.get("num_layers", def_n_dit)
    patch_size = config.get("patch_size", def_patch)
    feat_dim = config.get("feat_dim", 64)
    fsq_latent_dim = config.get("scalar_quantization_latent_dim", def_fsq_dim)
    fsq_scale = config.get("scalar_quantization_scale", 9)
    residual_lm_no_rope = config.get("residual_lm_no_rope", def_no_rope)

    print(
        f"  Acoustic: ResLM={n_residual_layers}l, LocEnc={n_encoder_layers}l, "
        f"LocDiT={n_dit_layers}l, FSQ dim={fsq_latent_dim}"
    )
    print(f"            patch_size={patch_size}, feat_dim={feat_dim}, no_rope={residual_lm_no_rope}")

    gguf_writer = GGUFWriter(output_path, "voxcpm-acoustic")

    # Version
    gguf_writer.add_float32("voxcpm.model_version", version)

    # Metadata
    gguf_writer.add_uint32("voxcpm.patch_size", patch_size)
    gguf_writer.add_uint32("voxcpm.feat_dim", feat_dim)
    gguf_writer.add_uint32("voxcpm.residual_lm.n_layer", n_residual_layers)
    gguf_writer.add_uint32("voxcpm.residual_lm.n_embd", n_embd)
    gguf_writer.add_bool("voxcpm.residual_lm.no_rope", residual_lm_no_rope)
    gguf_writer.add_uint32("voxcpm.locenc.n_layer", n_encoder_layers)
    gguf_writer.add_uint32("voxcpm.locenc.n_embd", encoder_cfg.get("hidden_dim", 1024))
    gguf_writer.add_uint32("voxcpm.locdit.n_layer", n_dit_layers)
    gguf_writer.add_uint32("voxcpm.locdit.n_embd", dit_cfg.get("hidden_dim", 1024))
    gguf_writer.add_uint32("voxcpm.fsq.latent_dim", fsq_latent_dim)
    gguf_writer.add_float32("voxcpm.fsq.scale", float(fsq_scale))
    gguf_writer.add_float32(
        "voxcpm.cfm.sigma_min", dit_cfg.get("cfm_config", {}).get("sigma_min", 1e-6)
    )
    gguf_writer.add_float32(
        "voxcpm.cfm.cfg_rate",
        dit_cfg.get("cfm_config", {}).get("inference_cfg_rate", 2.0),
    )

    # AudioVAE config
    gguf_writer.add_uint32(
        "voxcpm.audiovae.encoder_dim", vae_cfg.get("encoder_dim", 128)
    )
    gguf_writer.add_uint32("voxcpm.audiovae.latent_dim", vae_cfg.get("latent_dim", 64))
    gguf_writer.add_uint32(
        "voxcpm.audiovae.decoder_dim", vae_cfg.get("decoder_dim", def_dec_dim)
    )
    vae_sample_rate = vae_cfg.get("sample_rate", 16000)
    vae_out_sample_rate = vae_cfg.get("out_sample_rate", def_out_sr)
    gguf_writer.add_uint32("voxcpm.audiovae.sample_rate", vae_sample_rate)
    gguf_writer.add_uint32("voxcpm.audiovae.out_sample_rate", vae_out_sample_rate)
    encoder_rates = vae_cfg.get("encoder_rates", [2, 5, 8, 8])
    decoder_rates = vae_cfg.get("decoder_rates", [8, 6, 5, 2, 2, 2])
    gguf_writer.add_array("voxcpm.audiovae.encoder_rates", list(encoder_rates))
    gguf_writer.add_array("voxcpm.audiovae.decoder_rates", list(decoder_rates))
    sr_bins = vae_cfg.get("sr_bin_boundaries", def_sr_bins)
    if sr_bins:
        gguf_writer.add_array("voxcpm.audiovae.sr_bin_boundaries", list(sr_bins))
    gguf_writer.add_string(
        "voxcpm.audiovae.cond_type", vae_cfg.get("cond_type", def_cond_type)
    )

    # LongRoPE frequency factors for LocEnc/LocDiT (read from lm_config)
    rope_scaling = lm_cfg.get("rope_scaling", None)
    if rope_scaling and rope_scaling.get("type") == "longrope":
        long_factors = rope_scaling.get("long_factor", None)
        if long_factors:
            gguf_writer.add_array(
                "voxcpm.rope.long_factor", [float(v) for v in long_factors]
            )

    print("  Writing acoustic tensors...")

    # ---- ResidualLM ----
    gguf_writer.add_tensor(
        "residual_lm.output_norm.weight",
        to_f32_if_norm("output_norm", sf.get_tensor("residual_lm.norm.weight")),
    )
    for i in range(n_residual_layers):
        pt_prefix = f"residual_lm.layers.{i}"
        gguf_prefix = f"residual_lm.blk.{i}"
        for pt_suffix, gguf_suffix in [
            (".input_layernorm.weight", ".attn_norm.weight"),
            (".self_attn.q_proj.weight", ".attn_q.weight"),
            (".self_attn.k_proj.weight", ".attn_k.weight"),
            (".self_attn.v_proj.weight", ".attn_v.weight"),
            (".self_attn.o_proj.weight", ".attn_output.weight"),
            (".post_attention_layernorm.weight", ".ffn_norm.weight"),
            (".mlp.gate_proj.weight", ".ffn_gate.weight"),
            (".mlp.up_proj.weight", ".ffn_up.weight"),
            (".mlp.down_proj.weight", ".ffn_down.weight"),
        ]:
            tensor = sf.get_tensor(f"{pt_prefix}{pt_suffix}")
            gguf_writer.add_tensor(
                f"{gguf_prefix}{gguf_suffix}", to_f32_if_norm(gguf_suffix, tensor)
            )

    # ---- LocEnc ----
    gguf_writer.add_tensor(
        "locenc.norm.weight",
        to_f32_if_norm("norm", sf.get_tensor("feat_encoder.encoder.norm.weight")),
    )
    gguf_writer.add_tensor(
        "locenc.in_proj.weight", sf.get_tensor("feat_encoder.in_proj.weight")
    )
    gguf_writer.add_tensor(
        "locenc.in_proj.bias", sf.get_tensor("feat_encoder.in_proj.bias")
    )
    # CLS token: squeeze [1,1,1,1024] → [1024]
    cls_token = sf.get_tensor("feat_encoder.special_token")
    cls_token = cls_token.reshape(-1)  # → [1024]
    gguf_writer.add_tensor("locenc.cls_token.weight", cls_token)

    for i in range(n_encoder_layers):
        pt_prefix = f"feat_encoder.encoder.layers.{i}"
        gguf_prefix = f"locenc.blk.{i}"
        for pt_suffix, gguf_suffix in [
            (".input_layernorm.weight", ".attn_norm.weight"),
            (".self_attn.q_proj.weight", ".attn_q.weight"),
            (".self_attn.k_proj.weight", ".attn_k.weight"),
            (".self_attn.v_proj.weight", ".attn_v.weight"),
            (".self_attn.o_proj.weight", ".attn_output.weight"),
            (".post_attention_layernorm.weight", ".ffn_norm.weight"),
            (".mlp.gate_proj.weight", ".ffn_gate.weight"),
            (".mlp.up_proj.weight", ".ffn_up.weight"),
            (".mlp.down_proj.weight", ".ffn_down.weight"),
        ]:
            tensor = sf.get_tensor(f"{pt_prefix}{pt_suffix}")
            gguf_writer.add_tensor(
                f"{gguf_prefix}{gguf_suffix}", to_f32_if_norm(gguf_suffix, tensor)
            )

    # ---- LocDiT ----
    gguf_writer.add_tensor(
        "locdit.norm.weight",
        to_f32_if_norm(
            "norm", sf.get_tensor("feat_decoder.estimator.decoder.norm.weight")
        ),
    )
    gguf_writer.add_tensor(
        "locdit.in_proj.weight", sf.get_tensor("feat_decoder.estimator.in_proj.weight")
    )
    gguf_writer.add_tensor(
        "locdit.in_proj.bias", sf.get_tensor("feat_decoder.estimator.in_proj.bias")
    )
    gguf_writer.add_tensor(
        "locdit.out_proj.weight",
        sf.get_tensor("feat_decoder.estimator.out_proj.weight"),
    )
    gguf_writer.add_tensor(
        "locdit.out_proj.bias", sf.get_tensor("feat_decoder.estimator.out_proj.bias")
    )
    gguf_writer.add_tensor(
        "locdit.cond_proj.weight",
        sf.get_tensor("feat_decoder.estimator.cond_proj.weight"),
    )
    gguf_writer.add_tensor(
        "locdit.cond_proj.bias", sf.get_tensor("feat_decoder.estimator.cond_proj.bias")
    )

    # Time MLP
    for mlp_name, gguf_name in [
        ("time_mlp", "time_mlp"),
        ("delta_time_mlp", "delta_time_mlp"),
    ]:
        for layer_n in ["1", "2"]:
            gguf_writer.add_tensor(
                f"locdit.{gguf_name}.linear_{layer_n}.weight",
                sf.get_tensor(
                    f"feat_decoder.estimator.{mlp_name}.linear_{layer_n}.weight"
                ),
            )
            try:
                gguf_writer.add_tensor(
                    f"locdit.{gguf_name}.linear_{layer_n}.bias",
                    sf.get_tensor(
                        f"feat_decoder.estimator.{mlp_name}.linear_{layer_n}.bias"
                    ),
                )
            except KeyError:
                pass  # bias may not exist

    for i in range(n_dit_layers):
        pt_prefix = f"feat_decoder.estimator.decoder.layers.{i}"
        gguf_prefix = f"locdit.blk.{i}"
        for pt_suffix, gguf_suffix in [
            (".input_layernorm.weight", ".attn_norm.weight"),
            (".self_attn.q_proj.weight", ".attn_q.weight"),
            (".self_attn.k_proj.weight", ".attn_k.weight"),
            (".self_attn.v_proj.weight", ".attn_v.weight"),
            (".self_attn.o_proj.weight", ".attn_output.weight"),
            (".post_attention_layernorm.weight", ".ffn_norm.weight"),
            (".mlp.gate_proj.weight", ".ffn_gate.weight"),
            (".mlp.up_proj.weight", ".ffn_up.weight"),
            (".mlp.down_proj.weight", ".ffn_down.weight"),
        ]:
            tensor = sf.get_tensor(f"{pt_prefix}{pt_suffix}")
            gguf_writer.add_tensor(
                f"{gguf_prefix}{gguf_suffix}", to_f32_if_norm(gguf_suffix, tensor)
            )

    # ---- FSQ ----
    gguf_writer.add_tensor(
        "fsq.in_proj.weight", sf.get_tensor("fsq_layer.in_proj.weight")
    )
    gguf_writer.add_tensor("fsq.in_proj.bias", sf.get_tensor("fsq_layer.in_proj.bias"))
    gguf_writer.add_tensor(
        "fsq.out_proj.weight", sf.get_tensor("fsq_layer.out_proj.weight")
    )
    gguf_writer.add_tensor(
        "fsq.out_proj.bias", sf.get_tensor("fsq_layer.out_proj.bias")
    )

    # ---- Projections ----
    for name in ["enc_to_lm_proj", "lm_to_dit_proj", "res_to_dit_proj"]:
        gguf_writer.add_tensor(
            f"projections.{name}.weight", sf.get_tensor(f"{name}.weight")
        )
        gguf_writer.add_tensor(
            f"projections.{name}.bias", sf.get_tensor(f"{name}.bias")
        )
    # residual fusion projection (V2 only; absent in V1/0.5B)
    try:
        gguf_writer.add_tensor(
            "projections.res_fusion_proj.weight", sf.get_tensor("fusion_concat_proj.weight")
        )
        gguf_writer.add_tensor(
            "projections.res_fusion_proj.bias", sf.get_tensor("fusion_concat_proj.bias")
        )
        print("    fusion_concat_proj: present")
    except KeyError:
        if version == 2.0:
            print("    WARNING: fusion_concat_proj missing for VoxCPM2 model")
        else:
            print("    fusion_concat_proj: absent (expected for 0.5B/1.5)")

    # ---- StopPredictor ----
    gguf_writer.add_tensor(
        "stop_predictor.linear1.weight", sf.get_tensor("stop_proj.weight")
    )
    gguf_writer.add_tensor(
        "stop_predictor.linear1.bias", sf.get_tensor("stop_proj.bias")
    )
    gguf_writer.add_tensor(
        "stop_predictor.linear2.weight", sf.get_tensor("stop_head.weight")
    )

    # ---- AudioVAE (from audiovae.pth) ----
    _write_audiovae_weights(gguf_writer, vae_state_dict, vae_cfg, dtype)

    gguf_writer.open_output_file(Path(output_path))
    gguf_writer.write_header_to_file()
    gguf_writer.write_kv_data_to_file()
    gguf_writer.write_tensors_to_file()
    gguf_writer.close()
    print(f"  → {output_path}")


def _write_audiovae_weights(gguf_writer: GGUFWriter, state_dict: dict, vae_cfg: dict, dtype: str = "f16"):
    """Write AudioVAE weights, merging weight_norm (weight_g + weight_v) to single tensors."""
    print("  Writing AudioVAE weights (merging weight_norm)...")
    target_np = np.float16 if dtype == "f16" else np.float32

    # Build weight map: {base_name: {weight_g: tensor, weight_v: tensor, ...}}
    # First collect all weight_norm pairs
    weight_map = {}

    for key, tensor in state_dict.items():
        # tensor from torch.load is a torch.Tensor
        np_tensor = tensor.cpu().numpy()

        # Handle weight_norm: weight_g + weight_v
        if key.endswith(".weight_g"):
            base = key[: -len(".weight_g")]
            weight_map.setdefault(base, {})["weight_g"] = np_tensor
        elif key.endswith(".weight_v"):
            base = key[: -len(".weight_v")]
            weight_map.setdefault(base, {})["weight_v"] = np_tensor
        elif key.endswith(".weight"):
            base = key[: -len(".weight")]
            weight_map.setdefault(base, {})["weight"] = np_tensor
        elif key.endswith(".bias"):
            base = key[: -len(".bias")]
            weight_map.setdefault(base, {})["bias"] = np_tensor
        elif key.endswith(".alpha"):
            base = key[: -len(".alpha")]
            weight_map.setdefault(base, {})["alpha"] = np_tensor
        elif "metadata" in key or "kwargs" in key:
            continue  # skip metadata
        else:
            # Unknown key - write as-is
            weight_map.setdefault(key, {})["direct"] = np_tensor

    # Process each weight
    written = set()
    for base_name, parts in sorted(weight_map.items()):
        gguf_name = map_audiovae_key(base_name)

        if "weight_g" in parts and "weight_v" in parts:
            # Merge weight_norm
            merged = merge_weight_norm(parts["weight_g"], parts["weight_v"])
            merged = merged.astype(target_np)
            gguf_writer.add_tensor(gguf_name, merged)
            written.add(base_name)
        elif "weight" in parts:
            gguf_writer.add_tensor(gguf_name, parts["weight"].astype(target_np))
            written.add(base_name)
        elif "direct" in parts:
            gguf_writer.add_tensor(gguf_name, parts["direct"].astype(target_np))
            written.add(base_name)

        # Bias
        if "bias" in parts:
            gguf_name_bias = map_audiovae_key(base_name) + ".bias"
            gguf_writer.add_tensor(gguf_name_bias, parts["bias"].astype(target_np))

        # Alpha (Snake activation parameter)
        if "alpha" in parts:
            alpha_tensor = parts["alpha"]
            # Fix alpha shape for ggml broadcast: [C] → [C, 1]
            if alpha_tensor.ndim == 1:
                alpha_tensor = alpha_tensor[:, np.newaxis]
            gguf_name_alpha = map_audiovae_key(base_name) + ".alpha"
            gguf_writer.add_tensor(gguf_name_alpha, alpha_tensor.astype(target_np))

    print(
        f"    Wrote {len(written)} AudioVAE weight tensors (incl. merged weight_norm)"
    )


# ==============================================================================
# Main
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Convert VoxCPM/VoxCPM2 weights to GGUF format for llama.cpp-omni"
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to model file: model.safetensors or pytorch_model.bin, or a directory containing one of these",
    )
    parser.add_argument("--vae", required=True, help="Path to audiovae.pth")
    parser.add_argument("--config", required=True, help="Path to config.json")
    parser.add_argument(
        "--output", default="./", help="Output directory for GGUF files (default: .)"
    )
    parser.add_argument(
        "--tokenizer-dir",
        default=None,
        help="Directory containing tokenizer.json/tokenizer_config.json (default: directory of --config)",
    )
    parser.add_argument(
        "--dtype",
        default="f16",
        choices=["f16", "f32"],
        help="Output weight dtype (default: f16). Use f32 for quantization input.",
    )
    args = parser.parse_args()

    dtype = args.dtype
    dtype_upper = dtype.upper()

    os.makedirs(args.output, exist_ok=True)

    # Load config
    with open(args.config, "r") as f:
        config = json.load(f)

    # Detect model version (architecture) and infer display name (for output files)
    version = detect_model_version(config)
    model_name = infer_model_name(config, str(args.model), version)
    print(f"Architecture: {config.get('architecture', 'unknown')} → {model_name} (v{version})")
    print(f"Output dtype: {dtype}")

    # Resolve model path: accept a directory or a direct file path
    model_path = Path(args.model)
    if model_path.is_dir():
        safetensors_path = model_path / "model.safetensors"
        pytorch_bin_path = model_path / "pytorch_model.bin"
    else:
        safetensors_path = model_path
        pytorch_bin_path = model_path

    # Load model weights (safetensors or pytorch pickle)
    import torch

    if safetensors_path.exists() and safetensors_path.suffix == ".safetensors":
        print(f"Loading safetensors: {safetensors_path}")
        sf = SafeTensorFile(str(safetensors_path), target_dtype=dtype)
        print(f"  {len(sf.keys())} tensors")
    elif pytorch_bin_path.exists() and pytorch_bin_path.suffix in (".bin", ".pt", ".pth"):
        print(f"Loading pytorch checkpoint: {pytorch_bin_path}")
        checkpoint = torch.load(str(pytorch_bin_path), map_location="cpu", weights_only=True)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif isinstance(checkpoint, dict):
            state_dict = checkpoint
        else:
            raise ValueError(f"Unexpected checkpoint format: {type(checkpoint)}")

        # Convert torch tensors to numpy
        print(f"  Converting {len(state_dict)} tensors to numpy...")
        np_weights = {}
        for key, value in state_dict.items():
            if hasattr(value, 'numpy'):
                t = value
                if t.dtype == torch.bfloat16:
                    t = t.float()
                np_weights[key] = t.numpy()
            elif isinstance(value, np.ndarray):
                np_weights[key] = value
            else:
                print(f"  WARNING: skipping {key} (type={type(value)})")
        sf = DictWeightSource(np_weights, target_dtype=dtype)
        print(f"  {len(sf.keys())} tensors")
    else:
        raise FileNotFoundError(
            f"No model file found at {args.model}. "
            f"Expected model.safetensors or pytorch_model.bin"
        )

    # Load AudioVAE
    print(f"Loading AudioVAE: {args.vae}")
    vae_data = torch.load(args.vae, map_location="cpu", weights_only=True)
    vae_state_dict = vae_data.get("state_dict", vae_data)

    # Infer AudioVAE config if missing (V1/0.5B)
    if "audio_vae_config" not in config:
        config["audio_vae_config"] = infer_audio_vae_config(config, vae_state_dict)

    # Export BaseLM
    print(f"\n=== Exporting {model_name} BaseLM ===")
    base_lm_path = os.path.join(args.output, f"{model_name}-BaseLM-{dtype_upper}.gguf")
    tokenizer_dir = args.tokenizer_dir or os.path.dirname(os.path.abspath(args.config))
    export_baselm(sf, config, base_lm_path, tokenizer_dir, dtype, version)

    # Export Acoustic
    print(f"\n=== Exporting {model_name} Acoustic ===")
    acoustic_path = os.path.join(args.output, f"{model_name}-Acoustic-{dtype_upper}.gguf")
    export_acoustic(sf, vae_state_dict, config, acoustic_path, dtype, version)

    sf.close()

    # Print file sizes
    print("\n=== Done ===")
    for name, path in [("BaseLM", base_lm_path), ("Acoustic", acoustic_path)]:
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"  {name}: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
