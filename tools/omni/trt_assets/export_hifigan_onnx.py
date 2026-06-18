import sys, os
sys.path.insert(0, "/workspace/llama.cpp-omni/gguf-py")
from gguf import GGUFReader
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

reader = GGUFReader("/models/MiniCPM-o-4_5-gguf/token2wav-gguf/hifigan2.gguf")
def g2t(t):
    s = list(t.shape)
    if len(s) <= 1: return torch.from_numpy(np.frombuffer(t.data,dtype=np.float32).copy())
    return torch.from_numpy(np.frombuffer(t.data,dtype=np.float32).reshape(s).transpose().copy())
W = {t.name: g2t(t) for t in reader.tensors}
print(f"Loaded {len(W)} tensors")

lrelu_slope = 0.01

class Snake(nn.Module):
    def __init__(s,c): super().__init__(); s.alpha = nn.Parameter(torch.ones(c))
    def forward(s,x): a=s.alpha.view(1,-1,1); return x+(1./a)*(torch.sin(a*x)**2)

class RB(nn.Module):
    def __init__(s,c,k,dl):
        super().__init__()
        s.c1=nn.ModuleList([nn.Conv1d(c,c,k,dilation=d,padding=(k//2)*d) for d in dl])
        s.c2=nn.ModuleList([nn.Conv1d(c,c,k,dilation=d,padding=(k//2)*d) for d in dl])
        s.s1=nn.ModuleList([Snake(c) for _ in dl]); s.s2=nn.ModuleList([Snake(c) for _ in dl])
    def forward(s,x):
        for c1,c2,s1,s2 in zip(s.c1,s.c2,s.s1,s.s2): x=x+s2(c2(s1(c1(x))))
        return x

class MainPath(nn.Module):
    def __init__(s):
        super().__init__()
        s.conv_pre = nn.Conv1d(80,512,7,padding=3)
        s.up0 = nn.ConvTranspose1d(512,256,16,stride=8,padding=4)
        s.up1 = nn.ConvTranspose1d(256,128,11,stride=4,padding=4,output_padding=1)
        s.up2 = nn.ConvTranspose1d(128,64,7,stride=2,padding=3,output_padding=1)
        s.rb = nn.ModuleList()
        for ch,ks in [(256,[3,7,11]),(128,[3,7,11]),(64,[3,7,11])]:
            for k in ks: s.rb.append(RB(ch,k,[1,3,5]))
        s.conv_post = nn.Conv1d(64,18,7,padding=3)

    def load(s):
        d={}
        cp=lambda src,dst:d.update({dst:W[src]})
        cp('conv_pre.weight','conv_pre.weight');cp('conv_pre.bias','conv_pre.bias')
        cp('ups.0.weight','up0.weight');cp('ups.0.bias','up0.bias')
        cp('ups.1.weight','up1.weight');cp('ups.1.bias','up1.bias')
        cp('ups.2.weight','up2.weight');cp('ups.2.bias','up2.bias')
        for i in range(9):
            for j in range(3):
                cp(f'resblocks.{i}.convs1.{j}.weight',f'rb.{i}.c1.{j}.weight')
                cp(f'resblocks.{i}.convs1.{j}.bias',f'rb.{i}.c1.{j}.bias')
                cp(f'resblocks.{i}.convs2.{j}.weight',f'rb.{i}.c2.{j}.weight')
                cp(f'resblocks.{i}.convs2.{j}.bias',f'rb.{i}.c2.{j}.bias')
                cp(f'resblocks.{i}.activations1.{j}.alpha',f'rb.{i}.s1.{j}.alpha')
                cp(f'resblocks.{i}.activations2.{j}.alpha',f'rb.{i}.s2.{j}.alpha')
        cp('conv_post.weight','conv_post.weight');cp('conv_post.bias','conv_post.bias')
        s.load_state_dict(d,strict=True);print('OK')

    def forward(s,mel):
        x=F.leaky_relu(s.conv_pre(mel),lrelu_slope)
        x=s.up0(x); x=sum(s.rb[i](x) for i in range(3))/3.; x=F.leaky_relu(x,lrelu_slope)
        x=s.up1(x); x=sum(s.rb[i](x) for i in range(3,6))/3.; x=F.leaky_relu(x,lrelu_slope)
        x=s.up2(x); x=sum(s.rb[i](x) for i in range(6,9))/3.; x=F.leaky_relu(x,lrelu_slope)
        return s.conv_post(x)

m = MainPath(); m.load(); m.eval()
x = torch.randn(1,80,100)
with torch.no_grad(): o = m(x)
print(f"Mel {list(x.shape)} -> STFT {list(o.shape)}")

torch.onnx.export(m, x, "/workspace/vocoder_main.onnx",
    input_names=["mel"], output_names=["stft_18ch"], opset_version=17)
sz=os.path.getsize("/workspace/vocoder_main.onnx")/1048576
print(f"ONNX: {sz:.1f} MB")
