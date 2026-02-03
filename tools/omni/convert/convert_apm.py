import os
import sys
import json
from pathlib import Path

import torch
import numpy as np
import librosa

from gguf import GGUFWriter

# fmt: off
conv_map = {
        'self_attn.k_proj'              : 'attn.key',
        'self_attn.q_proj'              : 'attn.query',
        'self_attn.v_proj'              : 'attn.value',
        'self_attn.out_proj'            : 'attn.out',
        'self_attn_layer_norm'          : 'attn_ln',
        'encoder_attn.q_proj'           : 'cross_attn.query',
        'encoder_attn.v_proj'           : 'cross_attn.value',
        'encoder_attn.out_proj'         : 'cross_attn.out',
        'encoder_attn_layer_norm'       : 'cross_attn_ln',
        'fc1'                           : 'mlp.0',
        'fc2'                           : 'mlp.2',
        'final_layer_norm'              : 'mlp_ln',
        'encoder.layer_norm.bias'       : 'encoder.ln_post.bias',
        'encoder.layer_norm.weight'     : 'encoder.ln_post.weight',
        'encoder.embed_positions.weight': 'encoder.positional_embedding',
        'decoder.layer_norm.bias'       : 'decoder.ln.bias',
        'decoder.layer_norm.weight'     : 'decoder.ln.weight',
        'decoder.embed_positions.weight': 'decoder.positional_embedding',
        'decoder.embed_tokens.weight'   : 'decoder.token_embedding.weight',
        'proj_out.weight'               : 'decoder.proj.weight',
        'audio_projection_layer'        : 'audio_projector',
        }
# fmt on


def _replace_name(name):
    for pat in conv_map:
        name = name.replace(pat, conv_map[pat])
    return name


if len(sys.argv) < 4:
    print(
        "Usage: convert_apm.py dir-to-minicpmo-hf path-to-whipser dir-output"
    )
    sys.exit(1)

dir_model = Path(sys.argv[1])
dir_whisper = Path(sys.argv[2])
dir_out = Path(sys.argv[3])

fname_out = dir_out / "minicpmo-apm.gguf"
fout = GGUFWriter(path=fname_out, arch="whisper")

ftype = 1  # 1 = float16
# map from ftype to string
ftype_str = ["f32", "f16"]

fout.add_file_type(ftype)
fout.add_description("audio encoder for MiniCPM-omni")

use_f16 = True
hparams_json = json.load((dir_model / "config.json").open("r", encoding="utf8"))
hparams = hparams_json["audio_config"]
fout.add_uint32("encoder_attention_heads", hparams["encoder_attention_heads"])
fout.add_uint32("encoder_ffn_dim", hparams["encoder_ffn_dim"])
fout.add_uint32("encoder_layers", hparams["encoder_layers"])
fout.add_uint32("num_hidden_layers", hparams["num_hidden_layers"])
fout.add_uint32("d_model", hparams["d_model"])
fout.add_uint32("audio_pool_step", hparams_json["audio_pool_step"])
fout.add_uint32("use_f16", use_f16)

sample_rate = 16000
n_fft = 400
n_mels = 80

filters = librosa.filters.mel(sr=sample_rate, n_fft=n_fft, n_mels=n_mels)

fout.add_uint32("n_mel", filters.shape[0])
fout.add_uint32("n_fft", filters.shape[1])
fout.add_array("filters", filters.flatten().tolist())

state_dict = torch.load(dir_whisper)
new_state_dict = dict()
for k, v in state_dict.items():
    if k.startswith("apm."):
        k = k.replace("apm.", "encoder.")
    k = k.replace("layers.", "blocks.")
    k = _replace_name(k)

    data = v.squeeze().numpy()
    data = data.astype(np.float16)

    # reshape conv bias from [n] to [n, 1]
    if k in ["encoder.conv1.bias", "encoder.conv2.bias"]:
        data = data.reshape(data.shape[0], 1)
        print("  Reshaped variable: ", k, " to shape: ", data.shape)

    n_dims = len(data.shape)
    print(k, n_dims, data.shape)

    # looks like the whisper models are in f16 by default
    # so we need to convert the small tensors to f32 until we fully support f16 in ggml
    # ftype == 0 -> float32, ftype == 1 -> float16
    ftype_cur = 1
    if use_f16:
        if (
            n_dims < 2
            or k == "encoder.conv1.bias"
            or k == "encoder.conv2.bias"
            or k == "encoder.positional_embedding"
            or k == "decoder.positional_embedding"
        ):
            print("  Converting to float32")
            data = data.astype(np.float32)
            ftype_cur = 0
    else:
        data = data.astype(np.float32)
        ftype_cur = 0

    print(f"{k} - {ftype_str[ftype_cur]} - shape = {data.shape}")
    fout.add_tensor(name=k, tensor=data)

fout.write_header_to_file()
fout.write_kv_data_to_file()
fout.write_tensors_to_file()
fout.close()

print("Done. Output file: " + fname_out.as_posix())
