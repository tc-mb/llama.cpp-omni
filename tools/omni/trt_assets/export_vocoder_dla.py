#!/usr/bin/env python3
"""Export HiFi-GAN vocoder to ONNX, matching ggml C++ forward graph exactly."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "llama.cpp-omni/gguf-py"))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gguf import GGUFReader

GGUF_PATH = os.environ.get("GGUF_PATH", "/models/MiniCPM-o-4_5-gguf/token2wav-gguf/hifigan2.gguf")
ONNX_OUTPUT = os.environ.get("ONNX_OUTPUT", "/workspace/vocoder.onnx")
lrelu_slope = 0.01

# ====== 1. Load weights ======
reader = GGUFReader(GGUF_PATH)

def gguf_to_torch(tensor):
    s = list(tensor.shape)
    if len(s) <= 1:
        return torch.from_numpy(np.frombuffer(tensor.data, dtype=np.float32).copy())
    arr = np.frombuffer(tensor.data, dtype=np.float32).reshape(s).transpose()
    return torch.from_numpy(arr.copy())

W = {t.name: gguf_to_torch(t) for t in reader.tensors}
print(f"Loaded {len(W)} tensors, {sum(w.numel() for w in W.values()):,} params")

# ====== 2. Model ======

class Snake(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(ch))
    def forward(self, x):
        a = self.alpha.view(1, -1, 1)
        return x + (1.0 / a) * (torch.sin(a * x) ** 2)

class ResBlock1(nn.Module):
    def __init__(self, ch, k, dilations):
        super().__init__()
        c = ch
        self.c1 = nn.ModuleList([nn.Conv1d(c, c, k, dilation=d, padding=(k//2)*d) for d in dilations])
        self.c2 = nn.ModuleList([nn.Conv1d(c, c, k, dilation=d, padding=(k//2)*d) for d in dilations])
        self.s1 = nn.ModuleList([Snake(c) for _ in dilations])
        self.s2 = nn.ModuleList([Snake(c) for _ in dilations])
    def forward(self, x):
        for c1, c2, s1, s2 in zip(self.c1, self.c2, self.s1, self.s2):
            x = x + s2(c2(s1(c1(x))))
        return x

class HiFiGANGenerator(nn.Module):
    """Full HiFi-GAN decoder matching ggml build_graph_decode."""

    def __init__(self):
        super().__init__()
        self.conv_pre = nn.Conv1d(80, 512, 7, padding=3)

        self.up0 = nn.ConvTranspose1d(512, 256, 16, stride=8, padding=4)
        self.up1 = nn.ConvTranspose1d(256, 128, 11, stride=4, padding=4, output_padding=1)
        self.up2 = nn.ConvTranspose1d(128,  64,  7, stride=2, padding=3, output_padding=1)

        self.sd0 = nn.Conv1d(18, 256, 30, padding=15)
        self.sd1 = nn.Conv1d(18, 128,  6, padding=3)
        self.sd2 = nn.Conv1d(18,  64,  1)

        self.srb0 = ResBlock1(256,  7, [1, 3, 5])
        self.srb1 = ResBlock1(128,  7, [1, 3, 5])
        self.srb2 = ResBlock1( 64, 11, [1, 3, 5])

        # 9 resblocks: L0(256ch, k=3/7/11), L1(128ch, k=3/7/11), L2(64ch, k=3/7/11)
        self.rb = nn.ModuleList()
        for ch, ks in [(256, [3,7,11]), (128, [3,7,11]), (64, [3,7,11])]:
            for k in ks:
                self.rb.append(ResBlock1(ch, k, [1, 3, 5]))

        self.conv_post = nn.Conv1d(64, 18, 7, padding=3)
        self._loaded = False

    def load_gguf_weights(self):
        sd = {}

        def cp(src, dst_key):
            sd[dst_key] = W[src]

        cp('conv_pre.weight', 'conv_pre.weight'); cp('conv_pre.bias', 'conv_pre.bias')
        cp('ups.0.weight', 'up0.weight'); cp('ups.0.bias', 'up0.bias')
        cp('ups.1.weight', 'up1.weight'); cp('ups.1.bias', 'up1.bias')
        cp('ups.2.weight', 'up2.weight'); cp('ups.2.bias', 'up2.bias')

        for i, sd_n in enumerate(['sd0', 'sd1', 'sd2']):
            cp(f'source_downs.{i}.weight', f'{sd_n}.weight')
            cp(f'source_downs.{i}.bias',   f'{sd_n}.bias')

        for i, srb_n in enumerate(['srb0', 'srb1', 'srb2']):
            for j in range(3):
                for sgguf, spt, sn in [('convs1','c1','s1'), ('convs2','c2','s2')]:
                    cp(f'source_resblocks.{i}.{sgguf}.{j}.weight', f'{srb_n}.{spt}.{j}.weight')
                    cp(f'source_resblocks.{i}.{sgguf}.{j}.bias',   f'{srb_n}.{spt}.{j}.bias')
                    act = '2' if sn.startswith('s2') else '1'
                    cp(f'source_resblocks.{i}.activations{act}.{j}.alpha', f'{srb_n}.{sn}.{j}.alpha')

        for i in range(9):
            for j in range(3):
                cp(f'resblocks.{i}.convs1.{j}.weight', f'rb.{i}.c1.{j}.weight')
                cp(f'resblocks.{i}.convs1.{j}.bias',   f'rb.{i}.c1.{j}.bias')
                cp(f'resblocks.{i}.convs2.{j}.weight', f'rb.{i}.c2.{j}.weight')
                cp(f'resblocks.{i}.convs2.{j}.bias',   f'rb.{i}.c2.{j}.bias')
                cp(f'resblocks.{i}.activations1.{j}.alpha', f'rb.{i}.s1.{j}.alpha')
                cp(f'resblocks.{i}.activations2.{j}.alpha', f'rb.{i}.s2.{j}.alpha')

        cp('conv_post.weight', 'conv_post.weight'); cp('conv_post.bias', 'conv_post.bias')

        self.load_state_dict(sd, strict=True)
        self._loaded = True
        print("Weights loaded (exact ggml match)")

    def forward(self, mel, source_stft):
        """mel: [B, 80, Tm]  source_stft: [B, 18, Ts] → stft_18ch: [B, 18, Tout]"""
        x = self.conv_pre(mel)
        x = F.leaky_relu(x, lrelu_slope)

        # Level 0: 256 channels
        x = self.up0(x)
        si = self.sd0(source_stft); si = self.srb0(si)
        sl = min(x.shape[2], si.shape[2]); x, si = x[:,:,:sl], si[:,:,:sl]
        x = x + si
        x = sum(self.rb[i](x) for i in range(3)) / 3.0
        x = F.leaky_relu(x, lrelu_slope)

        # Level 1: 128 channels
        x = self.up1(x)
        si = self.sd1(source_stft); si = self.srb1(si)
        sl = min(x.shape[2], si.shape[2]); x, si = x[:,:,:sl], si[:,:,:sl]
        x = x + si
        x = sum(self.rb[i](x) for i in range(3, 6)) / 3.0
        x = F.leaky_relu(x, lrelu_slope)

        # Level 2: 64 channels
        x = self.up2(x)
        si = self.sd2(source_stft); si = self.srb2(si)
        sl = min(x.shape[2], si.shape[2]); x, si = x[:,:,:sl], si[:,:,:sl]
        x = x + si
        x = sum(self.rb[i](x) for i in range(6, 9)) / 3.0
        x = F.leaky_relu(x, lrelu_slope)

        return self.conv_post(x)  # [B, 18, Tout]


# ====== 3. Export ======
model = HiFiGANGenerator()
model.load_gguf_weights()
model.eval()

TMEL = 100  # fixed for TRT
TSTFT = TMEL * 2  # approximate — depends on upsampling ratios

dummy_mel = torch.randn(1, 80, TMEL)
dummy_src = torch.randn(1, 18, TSTFT)

with torch.no_grad():
    out = model(dummy_mel, dummy_src)
print(f"Input: mel {dummy_mel.shape} + src {dummy_src.shape} → Output: {out.shape}")

torch.onnx.export(
    model, (dummy_mel, dummy_src), ONNX_OUTPUT,
    input_names=["mel", "source_stft"],
    output_names=["stft_18ch"],
    opset_version=17,
)
sz = os.path.getsize(ONNX_OUTPUT) / 1048576
print(f"Exported: {ONNX_OUTPUT} ({sz:.1f} MB)")
