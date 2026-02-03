#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 MiniCPM-o 的 projector_semantic 权重转换为 GGUF 格式

projector_semantic 是 TTS 模块中的投影层，用于将 LLM hidden states (4096) 
投影到 TTS embedding space (768)

结构:
  forward(x): relu(linear1(x)) -> linear2

Usage:
    python convert_projector.py --model /path/to/model --output /path/to/projector.gguf
"""

import os
import struct
import argparse
import numpy as np

def write_string(f, s):
    """写入 GGUF 字符串"""
    encoded = s.encode('utf-8')
    f.write(struct.pack('<Q', len(encoded)))
    f.write(encoded)


def convert_projector(model_path: str, output_path: str):
    """转换 projector_semantic 到 GGUF 格式"""
    from safetensors import safe_open
    
    # 查找 safetensors 文件
    if os.path.isdir(model_path):
        safetensors_path = os.path.join(model_path, 'model.safetensors')
        if not os.path.exists(safetensors_path):
            # 尝试查找其他 safetensors 文件
            for f in os.listdir(model_path):
                if f.endswith('.safetensors'):
                    safetensors_path = os.path.join(model_path, f)
                    break
    else:
        safetensors_path = model_path
    
    if not os.path.exists(safetensors_path):
        raise FileNotFoundError(f"找不到 safetensors 文件: {safetensors_path}")
    
    print(f"加载模型权重: {safetensors_path}")
    f = safe_open(safetensors_path, framework='pt', device='cpu')
    
    # 提取 projector_semantic 权重
    weights = {}
    for key in f.keys():
        if 'tts.projector_semantic.' in key and 'norm' not in key:
            # 简化名称: tts.projector_semantic.linear1.weight -> linear1.weight
            new_key = key.replace('tts.projector_semantic.', '')
            tensor = f.get_tensor(key).float().numpy()
            weights[new_key] = tensor
            print(f"  {key} -> {new_key}: {tensor.shape}")
    
    if len(weights) == 0:
        raise ValueError("未找到 projector_semantic 权重")
    
    # GGUF 常量
    GGUF_MAGIC = 0x46554747  # "GGUF"
    GGUF_VERSION = 3
    GGML_TYPE_F32 = 0
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    print(f"\n创建 GGUF 文件: {output_path}")
    
    with open(output_path, 'wb') as out:
        # Header
        out.write(struct.pack('<I', GGUF_MAGIC))
        out.write(struct.pack('<I', GGUF_VERSION))
        
        n_tensors = len(weights)
        n_kv = 1  # 只有 general.architecture
        out.write(struct.pack('<Q', n_tensors))
        out.write(struct.pack('<Q', n_kv))
        
        # Metadata KV
        # general.architecture = "projector"
        write_string(out, "general.architecture")
        out.write(struct.pack('<I', 8))  # GGUF_TYPE_STRING = 8
        write_string(out, "projector")
        
        # Tensor info
        tensor_data = []
        current_offset = 0
        
        for name, tensor in weights.items():
            # 转换为 F32
            tensor = tensor.astype(np.float32)
            data = tensor.tobytes()
            tensor_data.append(data)
            
            # Name
            write_string(out, name)
            
            # n_dims
            n_dims = len(tensor.shape)
            out.write(struct.pack('<I', n_dims))
            
            # dims (GGUF 使用小端序，dims 顺序反过来)
            for dim in reversed(tensor.shape):
                out.write(struct.pack('<Q', dim))
            
            # type (F32)
            out.write(struct.pack('<I', GGML_TYPE_F32))
            
            # offset
            out.write(struct.pack('<Q', current_offset))
            
            current_offset += len(data)
            print(f"  {name}: shape={tensor.shape}, size={len(data)} bytes")
        
        # Padding to alignment (32 bytes)
        current_pos = out.tell()
        alignment = 32
        padding = (alignment - (current_pos % alignment)) % alignment
        out.write(b'\x00' * padding)
        
        # Tensor data
        for data in tensor_data:
            out.write(data)
    
    file_size = os.path.getsize(output_path)
    print(f"\n完成！文件大小: {file_size / 1024 / 1024:.2f} MB")
    
    return output_path


def verify_gguf(gguf_path: str):
    """验证生成的 GGUF 文件"""
    try:
        from gguf import GGUFReader
        print(f"\n验证 GGUF 文件: {gguf_path}")
        reader = GGUFReader(gguf_path)
        print(f"张量数量: {len(reader.tensors)}")
        for tensor in reader.tensors:
            print(f"  {tensor.name}: {tensor.shape}")
        return True
    except ImportError:
        print("警告: gguf 模块未安装，跳过验证")
        return True
    except Exception as e:
        print(f"验证失败: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='将 MiniCPM-o projector_semantic 转换为 GGUF 格式'
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        required=True,
        help='模型路径 (目录或 safetensors 文件)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='projector.gguf',
        help='输出 GGUF 文件路径 (默认: projector.gguf)'
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        default=True,
        help='验证生成的 GGUF 文件'
    )
    
    args = parser.parse_args()
    
    output_path = convert_projector(args.model, args.output)
    
    if args.verify:
        verify_gguf(output_path)


if __name__ == '__main__':
    main()
