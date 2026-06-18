import onnx, os
from onnxsim import simplify
m = onnx.load('/workspace/vocoder.onnx')
ms, ok = simplify(m)
print(f"Simplify: {ok}")
print(f"Nodes: {len(m.graph.node)} -> {len(ms.graph.node)}")
onnx.save(ms, '/workspace/vocoder_fixed.onnx')
sz = os.path.getsize('/workspace/vocoder_fixed.onnx') / 1048576
print(f"Size: {sz:.1f} MB")
