#!/usr/bin/env python3
"""
MiniCPM-TTS 模型转换为 GGUF 格式

TTS 模型结构:
- emb_code: 音频 code embedding
- emb_text: 文本 embedding  
- model.layers: Transformer 层
- head_code: 音频 code 输出头 (带 weight_norm)
- projector_semantic: 语义投影层
- projector_spk: 说话人投影层
"""

import argparse
import json
import os
import struct
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

# GGUF 常量
GGUF_MAGIC = 0x46554747  # "GGUF" in little endian
GGUF_VERSION = 3

# GGML 类型
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1


def write_string(f, s):
    """写入字符串 (长度 + 内容)"""
    encoded = s.encode('utf-8')
    f.write(struct.pack('<Q', len(encoded)))
    f.write(encoded)


def write_metadata_kv(f, key, value_type, value):
    """写入元数据键值对"""
    write_string(f, key)
    f.write(struct.pack('<I', value_type))
    
    if value_type == 4:  # string
        write_string(f, value)
    elif value_type == 5:  # uint32
        f.write(struct.pack('<I', value))
    elif value_type == 6:  # int32
        f.write(struct.pack('<i', value))
    elif value_type == 7:  # float32
        f.write(struct.pack('<f', value))


def convert_tts_to_gguf(model_dir, output_path, output_type='f32'):
    """转换 TTS 模型到 GGUF"""
    
    safetensors_path = os.path.join(model_dir, 'model.safetensors')
    config_path = os.path.join(model_dir, 'config.json')
    
    # 读取配置
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # 读取张量 (使用 torch 处理 bfloat16)
    print(f"Loading model from {safetensors_path}")
    tensors_torch = load_file(safetensors_path)
    
    # 转换为 float32 numpy
    tensors = {}
    for key, tensor in tensors_torch.items():
        tensors[key] = tensor.float().numpy()
    
    # 处理 weight_norm 参数化
    # head_code.0.parametrizations.weight.original0 和 original1 需要合并
    if 'head_code.0.parametrizations.weight.original0' in tensors and \
       'head_code.0.parametrizations.weight.original1' in tensors:
        g = tensors['head_code.0.parametrizations.weight.original0']  # [6562, 1]
        v = tensors['head_code.0.parametrizations.weight.original1']  # [6562, 768]
        # weight_norm: w = g * v / ||v||
        v_norm = np.linalg.norm(v, axis=1, keepdims=True)
        weight = g * v / (v_norm + 1e-12)
        tensors['head_code.0.weight'] = weight.astype(np.float32)
        del tensors['head_code.0.parametrizations.weight.original0']
        del tensors['head_code.0.parametrizations.weight.original1']
        print(f"Reconstructed head_code.0.weight: {weight.shape}")
    
    # 张量名称映射 (TTS -> GGUF)
    tensor_map = {}
    for key in tensors:
        # 标准 Transformer 层
        new_key = key
        if key.startswith('model.'):
            new_key = key.replace('model.', 'tts.')
        elif key.startswith('emb_'):
            new_key = f'tts.{key}'
        elif key.startswith('head_'):
            new_key = f'tts.{key}'
        elif key.startswith('projector_'):
            new_key = f'tts.{key}'
        
        tensor_map[key] = new_key
    
    print(f"\nTotal tensors: {len(tensors)}")
    
    # 准备写入 GGUF
    n_tensors = len(tensors)
    n_kv = 10  # 元数据数量
    
    with open(output_path, 'wb') as f:
        # 写入 GGUF 头
        f.write(struct.pack('<I', GGUF_MAGIC))
        f.write(struct.pack('<I', GGUF_VERSION))
        f.write(struct.pack('<Q', n_tensors))
        f.write(struct.pack('<Q', n_kv))
        
        # 写入元数据
        write_metadata_kv(f, 'general.architecture', 4, 'minicpmtts')
        write_metadata_kv(f, 'general.name', 4, 'MiniCPM-TTS')
        write_metadata_kv(f, 'minicpmtts.context_length', 5, config.get('max_position_embeddings', 4096))
        write_metadata_kv(f, 'minicpmtts.embedding_length', 5, config.get('hidden_size', 768))
        write_metadata_kv(f, 'minicpmtts.block_count', 5, config.get('num_hidden_layers', 20))
        write_metadata_kv(f, 'minicpmtts.attention.head_count', 5, config.get('num_attention_heads', 12))
        write_metadata_kv(f, 'minicpmtts.attention.head_count_kv', 5, config.get('num_key_value_heads', 12))
        write_metadata_kv(f, 'minicpmtts.feed_forward_length', 5, config.get('intermediate_size', 3072))
        write_metadata_kv(f, 'minicpmtts.llm_hidden_size', 5, config.get('llm_hidden_size', 4096))
        write_metadata_kv(f, 'general.file_type', 5, GGML_TYPE_F32 if output_type == 'f32' else GGML_TYPE_F16)
        
        # 写入张量信息
        tensor_infos = []
        offset = 0
        for old_key in tensors:
            new_key = tensor_map[old_key]
            tensor = tensors[old_key]
            
            # 转换数据类型
            if output_type == 'f32':
                tensor = tensor.astype(np.float32)
                dtype = GGML_TYPE_F32
                element_size = 4
            else:
                tensor = tensor.astype(np.float16)
                dtype = GGML_TYPE_F16
                element_size = 2
            
            shape = tensor.shape
            n_dims = len(shape)
            
            # 写入张量名称
            write_string(f, new_key)
            
            # 写入维度数
            f.write(struct.pack('<I', n_dims))
            
            # 写入每个维度大小 (GGUF 使用小端序，维度逆序)
            for dim in reversed(shape):
                f.write(struct.pack('<Q', dim))
            
            # 写入数据类型
            f.write(struct.pack('<I', dtype))
            
            # 写入偏移量
            f.write(struct.pack('<Q', offset))
            
            # 计算数据大小
            data_size = tensor.size * element_size
            # 对齐到 32 字节
            aligned_size = (data_size + 31) // 32 * 32
            
            tensor_infos.append((new_key, tensor, aligned_size))
            offset += aligned_size
        
        # 对齐到 32 字节边界
        current_pos = f.tell()
        padding = (32 - (current_pos % 32)) % 32
        f.write(b'\x00' * padding)
        
        # 写入张量数据
        for name, tensor, aligned_size in tensor_infos:
            data = tensor.tobytes()
            f.write(data)
            # 填充对齐
            padding = aligned_size - len(data)
            if padding > 0:
                f.write(b'\x00' * padding)
    
    print(f"\nModel saved to {output_path}")
    print(f"File size: {os.path.getsize(output_path) / 1024 / 1024:.1f} MB")


def main():
    parser = argparse.ArgumentParser(description='Convert MiniCPM-TTS to GGUF')
    parser.add_argument('model_dir', type=str, help='Path to TTS model directory')
    parser.add_argument('--output', '-o', type=str, default=None, help='Output GGUF file path')
    parser.add_argument('--outtype', type=str, default='f32', choices=['f32', 'f16'], help='Output type')
    
    args = parser.parse_args()
    
    if args.output is None:
        model_name = Path(args.model_dir).name
        size_mb = 440  # 估计大小
        args.output = f"Tts-{size_mb}M-{args.outtype.upper()}.gguf"
    
    convert_tts_to_gguf(args.model_dir, args.output, args.outtype)


if __name__ == '__main__':
    main()

