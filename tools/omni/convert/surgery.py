import argparse
import os
import sys
import shutil
import importlib.util
import re

import torch
from safetensors import safe_open
from transformers import AutoTokenizer, AutoModel, AutoProcessor


def fix_relative_imports(model_dir: str) -> dict:
    """
    修复模型目录中 Python 文件的相对导入为绝对导入
    
    Args:
        model_dir: 模型目录路径
    
    Returns:
        备份字典 {file_path: original_content}
    """
    backups = {}
    
    # 需要修复的文件列表
    py_files = [f for f in os.listdir(model_dir) if f.endswith('.py')]
    
    for py_file in py_files:
        file_path = os.path.join(model_dir, py_file)
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 备份原始内容
        backups[file_path] = content
        
        # 替换相对导入为绝对导入
        # from .xxx import -> from xxx import
        # from . import -> import
        modified = re.sub(r'from \.([\w_]+)', r'from \1', content)
        modified = re.sub(r'from \. import', r'import', modified)
        
        if modified != content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(modified)
            print(f"  修复了相对导入: {py_file}")
    
    return backups


def restore_relative_imports(backups: dict):
    """恢复原始的相对导入"""
    for file_path, content in backups.items():
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
    print(f"  已恢复 {len(backups)} 个文件的原始导入")


def load_safetensors(safetensors_path: str) -> dict:
    """
    加载 safetensors 文件为字典
    
    Args:
        safetensors_path: safetensors 文件路径
    
    Returns:
        state_dict 字典
    """
    state_dict = {}
    with safe_open(safetensors_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            state_dict[key] = f.get_tensor(key)
    return state_dict


def main():
    parser = argparse.ArgumentParser(description="模型转换工具 - 支持 safetensors 格式")
    parser.add_argument(
        "--model",
        type=str,
        default="tools/omni/convert/MiniCPM-o-4_5",
        help="模型路径（包含 config.json、tokenizer 和 model.safetensors 的目录）",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="输出目录（如果为 None，则使用 tools/omni/convert/model）",
    )
    args = parser.parse_args()
    
    # 将模型目录添加到 sys.path，以便 transformers 能正确加载相关模块
    model_dir = os.path.abspath(args.model)
    if model_dir not in sys.path:
        sys.path.insert(0, model_dir)
        print(f"已将模型目录添加到 sys.path: {model_dir}")
    
    # 临时修复模型目录中的相对导入
    print("\n修复模型目录中的相对导入...")
    import_backups = fix_relative_imports(model_dir)
    
    # 确定输出目录
    if args.output_dir is None:
        # 使用固定的输出目录
        args.output_dir = os.path.join(os.path.dirname(__file__), "model")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 创建子文件夹
    llm_dir = os.path.join(args.output_dir, "llm")
    tts_dir = os.path.join(args.output_dir, "tts")
    vpm_dir = os.path.join(args.output_dir, "vpm")
    apm_dir = os.path.join(args.output_dir, "apm")
    
    os.makedirs(llm_dir, exist_ok=True)
    os.makedirs(tts_dir, exist_ok=True)
    os.makedirs(vpm_dir, exist_ok=True)
    os.makedirs(apm_dir, exist_ok=True)
    
    print(f"输出目录: {args.output_dir}")
    print(f"  - LLM: {llm_dir}")
    print(f"  - TTS: {tts_dir}")
    print(f"  - VPM: {vpm_dir}")
    print(f"  - APM: {apm_dir}")

    # 检查 safetensors 文件
    safetensors_path = os.path.join(args.model, "model.safetensors")
    if not os.path.exists(safetensors_path):
        raise FileNotFoundError(f"找不到 safetensors 文件: {safetensors_path}")
    
    print(f"\n正在加载 safetensors: {safetensors_path}")
    file_size = os.path.getsize(safetensors_path) / 1024 / 1024 / 1024
    print(f"  文件大小: {file_size:.2f} GB")
    
    # 加载 safetensors 文件
    checkpoint = load_safetensors(safetensors_path)
    print(f"  加载完成，共 {len(checkpoint)} 个张量")

    # 模型引入（使用 AutoModel，trust_remote_code=True 会自动加载模型目录中的代码）
    print(f"\n正在加载模型结构: {args.model}")
    model = AutoModel.from_pretrained(
        args.model, trust_remote_code=True, _attn_implementation="sdpa"
    )
    if hasattr(model, 'default_tts_chat_template'):
        print(f"model.default_tts_chat_template={model.default_tts_chat_template}")

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    processor.tokenizer = tokenizer
    model.processor = processor

    # safetensors 文件已经包含权重，直接使用
    print("权重已从 safetensors 加载")

    model.bfloat16()
    model.eval()

    # vpm
    print("\n正在提取 VPM 组件...")
    # get a list of mm tensor names
    mm_tensors = [k for k, v in checkpoint.items() if k.startswith("resampler")]

    # store these tensors in a new dictionary and torch.save them
    if len(mm_tensors) > 0:
        projector = {name: checkpoint[name].float() for name in mm_tensors}
        torch.save(projector, os.path.join(vpm_dir, "minicpmv.projector"))
        print(f"  VPM projector 已保存: {os.path.join(vpm_dir, 'minicpmv.projector')}")
        print(f"    张量数量: {len(projector)}")

    clip_tensors = [k for k, v in checkpoint.items() if k.startswith("vpm")]
    if len(clip_tensors) > 0:
        clip = {name.replace("vpm.", ""): checkpoint[name].float() for name in clip_tensors}
        torch.save(clip, os.path.join(vpm_dir, "minicpmv.clip"))
        print(f"  VPM clip 已保存: {os.path.join(vpm_dir, 'minicpmv.clip')}")
        print(f"    张量数量: {len(clip)}")
    
        # added tokvpm should be removed to be able to convert Mistral models
        if os.path.exists(os.path.join(args.model, "added_tokens.json")):
            with open(os.path.join(vpm_dir, "added_tokens.json"), "w") as f:
                f.write("{}\n")
            print(f"  added_tokens.json 已保存: {os.path.join(vpm_dir, 'added_tokens.json')}")
    
    # 复制 config.json 到 vpm 目录（convert_vpm.py 需要）
    config_src = os.path.join(args.model, "config.json")
    if os.path.exists(config_src):
        shutil.copy2(config_src, os.path.join(vpm_dir, "config.json"))
        print(f"  config.json 已复制到 VPM 目录")


    # apm
    print("\n正在提取 APM 组件...")
    # get a list of mm tensor names
    apm_mm_tensors = [
        k for k, v in checkpoint.items() if k.startswith("audio_projection_layer")
    ]

    # store these tensors in a new dictionary and torch.save them
    apm_projector = {name: checkpoint[name].float() for name in apm_mm_tensors}
    
    whisper_tensors = [k for k, v in checkpoint.items() if k.startswith("apm")]
    # append apm projector into a single file
    whisper_keys = list(whisper_tensors)
    whisper_keys.extend(list(apm_projector.keys()))
    if len(whisper_keys) > 0:
        whisper = {name: checkpoint[name].float() for name in whisper_keys if name in checkpoint}
        # 添加 projector 张量
        whisper.update(apm_projector)
        torch.save(whisper, os.path.join(apm_dir, "minicpmo.whisper"))
        print(f"  APM whisper 已保存: {os.path.join(apm_dir, 'minicpmo.whisper')}")
        print(f"    张量数量: {len(whisper)}")
    
    # 复制 config.json 到 apm 目录（convert_apm.py 需要）
    config_src = os.path.join(args.model, "config.json")
    if os.path.exists(config_src):
        shutil.copy2(config_src, os.path.join(apm_dir, "config.json"))
        print(f"  config.json 已复制到 APM 目录")
    
    
    # llm
    print("\n正在保存 LLM 模型...")
    config = model.llm.config
    config.auto_map = {}
    model.llm.save_pretrained(llm_dir)
    tokenizer.save_pretrained(llm_dir)
    
    # 清理 auto_map，避免 convert_hf_to_gguf 需要自定义代码
    import json
    llm_config_path = os.path.join(llm_dir, "config.json")
    with open(llm_config_path, 'r') as f:
        saved_config = json.load(f)
    
    saved_config['auto_map'] = {}
    
    with open(llm_config_path, 'w') as f:
        json.dump(saved_config, f, indent=2)
    print(f"  已清理 LLM config auto_map")
    
    # 清理 tokenizer_config.json，移除自定义代码引用
    tokenizer_config_path = os.path.join(llm_dir, "tokenizer_config.json")
    if os.path.exists(tokenizer_config_path):
        with open(tokenizer_config_path, 'r') as f:
            tok_config = json.load(f)
        
        # 移除 auto_map 和改为标准 tokenizer_class
        if 'auto_map' in tok_config:
            del tok_config['auto_map']
        if 'tokenizer_class' in tok_config:
            tok_config['tokenizer_class'] = 'PreTrainedTokenizerFast'
        
        with open(tokenizer_config_path, 'w') as f:
            json.dump(tok_config, f, indent=2)
        print(f"  已清理 tokenizer_config.json")
    
    print(f"  LLM 模型已保存: {llm_dir}")

    # tts
    print("\n正在保存 TTS 模型...")
    tts_config = model.tts.config
    tts_config.auto_map = {}
    
    # 将 TTS config 转换为 llama 兼容格式
    # 这是因为 convert_hf_to_gguf.py 需要识别 model_type
    print("  将 TTS config 转换为 llama 兼容格式...")
    tts_config.model_type = "llama"
    tts_config.architectures = ["LlamaForCausalLM"]
    
    # 确保关键参数存在（llama 格式需要）
    if not hasattr(tts_config, 'vocab_size'):
        tts_config.vocab_size = tts_config.num_audio_tokens + tts_config.num_text_tokens
    if not hasattr(tts_config, 'rms_norm_eps'):
        tts_config.rms_norm_eps = 1e-6
    if not hasattr(tts_config, 'rope_theta'):
        tts_config.rope_theta = 10000.0
    if not hasattr(tts_config, 'tie_word_embeddings'):
        tts_config.tie_word_embeddings = False
    
    print(f"    model_type: {tts_config.model_type}")
    print(f"    hidden_size: {tts_config.hidden_size}")
    print(f"    num_hidden_layers: {tts_config.num_hidden_layers}")
    print(f"    num_attention_heads: {tts_config.num_attention_heads}")
    print(f"    intermediate_size: {tts_config.intermediate_size}")
    print(f"    vocab_size: {tts_config.vocab_size}")
    
    model.tts.save_pretrained(tts_dir)
    tokenizer.save_pretrained(tts_dir)
    
    # save_pretrained 会使用原始 config，需要手动覆写为 llama 格式
    import json
    tts_config_path = os.path.join(tts_dir, "config.json")
    with open(tts_config_path, 'r') as f:
        saved_config = json.load(f)
    
    # 强制设置为 llama 格式
    saved_config['model_type'] = 'llama'
    saved_config['architectures'] = ['LlamaForCausalLM']
    
    with open(tts_config_path, 'w') as f:
        json.dump(saved_config, f, indent=2)
    print(f"  已将 TTS config 修改为 llama 格式")
    
    # 清理 TTS tokenizer_config.json
    tts_tok_config_path = os.path.join(tts_dir, "tokenizer_config.json")
    if os.path.exists(tts_tok_config_path):
        with open(tts_tok_config_path, 'r') as f:
            tts_tok_config = json.load(f)
        if 'auto_map' in tts_tok_config:
            del tts_tok_config['auto_map']
        if 'tokenizer_class' in tts_tok_config:
            tts_tok_config['tokenizer_class'] = 'PreTrainedTokenizerFast'
        with open(tts_tok_config_path, 'w') as f:
            json.dump(tts_tok_config, f, indent=2)
        print(f"  已清理 TTS tokenizer_config.json")
    
    print(f"  TTS 模型已保存: {tts_dir}")

    # 恢复原始的相对导入
    print("\n恢复模型目录中的原始导入...")
    restore_relative_imports(import_backups)

    print("\n" + "="*60)
    print("Done!")
    print(f"转换后的模型保存在: {args.output_dir}")
    print(f"  - LLM: {llm_dir}")
    print(f"  - TTS: {tts_dir}")
    print(f"  - VPM: {vpm_dir}")
    print(f"    - minicpmv.projector")
    print(f"    - minicpmv.clip")
    print(f"  - APM: {apm_dir}")
    print(f"    - minicpmo.whisper")
    print(f"\n现在可以将 {llm_dir} 转换为 LLaMA GGUF 文件。")
    print(f"使用 {os.path.join(vpm_dir, 'minicpmv.projector')} 准备 minicpmv-encoder.gguf 文件。")


if __name__ == "__main__":
    main()
