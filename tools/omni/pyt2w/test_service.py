#!/usr/bin/env python3
"""
测试 Token2Wav 服务的脚本
"""

import os
import sys
import json
import subprocess
import time

# 配置 - 使用相对路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "token2wav")
REF_AUDIO = os.path.join(SCRIPT_DIR, "../convert/gguf/token2wav-gguf/haitian_ref_audio/haitian_ref_audio.wav")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "test_output")
DEVICE = "cuda:0"  # 默认使用 GPU 0

# 测试 tokens（模拟从 TTS 生成的 audio tokens）
# 这是一个简单的测试，实际 tokens 应该从 LLM 生成
TEST_TOKENS = [4218, 4218, 4218] + [1000 + i for i in range(25)]  # 3 prefix + 25 tokens


def main():
    # 获取脚本目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    service_script = os.path.join(script_dir, "token2wav_service.py")
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"启动 Token2Wav 服务...")
    print(f"模型目录: {MODEL_DIR}")
    print(f"参考音频: {REF_AUDIO}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"设备: {DEVICE}")
    print()
    
    # 启动服务进程
    env = os.environ.copy()
    process = subprocess.Popen(
        [sys.executable, service_script],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        text=True,
        bufsize=1
    )
    
    def send_cmd(cmd):
        """发送命令并获取响应"""
        cmd_json = json.dumps(cmd)
        print(f">>> {cmd_json}")
        process.stdin.write(cmd_json + "\n")
        process.stdin.flush()
        
        response_line = process.stdout.readline()
        response = json.loads(response_line)
        print(f"<<< {json.dumps(response, ensure_ascii=False)}")
        print()
        return response
    
    try:
        # 等待服务就绪
        print("等待服务就绪...")
        ready_line = process.stdout.readline()
        ready = json.loads(ready_line)
        print(f"服务状态: {ready}")
        print()
        
        # 1. 初始化
        print("=" * 50)
        print("1. 初始化 Token2Wav")
        print("=" * 50)
        response = send_cmd({
            "cmd": "init",
            "model_dir": MODEL_DIR,
            "device": DEVICE,
            "float16": True,
            "n_timesteps": 10
        })
        
        if response.get("status") != "ok":
            print(f"初始化失败: {response}")
            return
        
        # 2. 设置参考音频
        print("=" * 50)
        print("2. 设置参考音频")
        print("=" * 50)
        response = send_cmd({
            "cmd": "set_ref_audio",
            "ref_audio_path": REF_AUDIO
        })
        
        if response.get("status") != "ok":
            print(f"设置参考音频失败: {response}")
            return
        
        # 3. 处理 tokens
        print("=" * 50)
        print("3. 处理 tokens")
        print("=" * 50)
        
        # 模拟滑动窗口处理
        for i in range(3):
            tokens = TEST_TOKENS[:]
            is_last = (i == 2)
            output_path = os.path.join(OUTPUT_DIR, f"wav_{i}.wav")
            
            response = send_cmd({
                "cmd": "process",
                "tokens": tokens,
                "last_chunk": is_last,
                "output_path": output_path
            })
            
            if response.get("status") != "ok":
                print(f"处理失败: {response}")
                break
        
        # 4. 重置缓存
        print("=" * 50)
        print("4. 重置缓存")
        print("=" * 50)
        response = send_cmd({"cmd": "reset"})
        
        # 5. 退出
        print("=" * 50)
        print("5. 退出服务")
        print("=" * 50)
        response = send_cmd({"cmd": "quit"})
        
    finally:
        # 等待进程结束
        process.wait(timeout=5)
        
        # 打印 stderr
        stderr = process.stderr.read()
        if stderr:
            print("=" * 50)
            print("服务日志 (stderr):")
            print("=" * 50)
            print(stderr)


if __name__ == "__main__":
    main()
