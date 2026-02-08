"""快速音频收集脚本：发送一段语音+图片，收集 TTS 输出并保存为 WAV

使用方式:
    cd llama.cpp-omni && PYTHONPATH=. .venv/base/bin/python tools/omni/app/tests/collect_audio.py

前置条件: server.py 已启动且 init_sys_prompt 已完成 (initialized=True)
"""
import argparse
import base64
import io
import json
import os
import struct
import time
import wave
from typing import List, Tuple

import numpy as np
import requests

SAMPLE_RATE = 24000
SERVER_URL = "http://127.0.0.1:9060"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "results/audio_collect")


def generate_test_image() -> str:
    """生成一张简单测试图（蓝天白云公园场景）"""
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (320, 240), (135, 206, 235))
    draw = ImageDraw.Draw(img)
    # 草地
    draw.rectangle([0, 160, 320, 240], fill=(34, 139, 34))
    # 太阳
    draw.ellipse([240, 20, 290, 70], fill=(255, 215, 0))
    # 树
    draw.rectangle([100, 100, 115, 160], fill=(139, 69, 19))
    draw.ellipse([75, 60, 140, 120], fill=(0, 128, 0))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode()


def generate_speech_audio(duration_s: float = 3.0) -> str:
    """生成模拟语音 WAV (带 440Hz tone 模拟有人说话)"""
    t = np.linspace(0, duration_s, int(SAMPLE_RATE * duration_s), endpoint=False)
    # 用简单正弦波 + 噪声模拟
    audio = 0.3 * np.sin(2 * np.pi * 440 * t) + 0.05 * np.random.randn(len(t))
    audio = np.clip(audio, -1.0, 1.0)
    pcm16 = (audio * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm16.tobytes())
    return base64.b64encode(buf.getvalue()).decode()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default=SERVER_URL)
    parser.add_argument("--rounds", type=int, default=30, help="generate 轮数")
    parser.add_argument("--output", default=OUTPUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # 检查 server
    health = requests.get(f"{args.server}/health", timeout=5).json()
    print(f"[HEALTH] {health}")
    if not health.get("initialized"):
        print("Server 未初始化，先 init...")
        resp = requests.post(f"{args.server}/omni/init_sys_prompt",
                             json={"media_type": 2, "duplex_mode": True, "language": "zh"},
                             timeout=60)
        resp.raise_for_status()
        print(f"[INIT] {resp.json()}")

    # 生成输入
    print("生成测试图片和语音...")
    image_b64 = generate_test_image()
    audio_b64 = generate_speech_audio(3.0)
    print(f"  图片: {len(image_b64)} bytes b64, 语音: {len(audio_b64)} bytes b64")

    # Prefill
    print("\n发送 prefill...")
    t0 = time.time()
    resp = requests.post(f"{args.server}/omni/streaming_prefill",
                         json={"audio": audio_b64, "image": image_b64}, timeout=30)
    resp.raise_for_status()
    print(f"  prefill done: {(time.time()-t0)*1000:.0f}ms")

    # Generate 循环，收集音频
    all_pcm: List[bytes] = []
    all_text: str = ""
    total_audio_s: float = 0.0

    print(f"\n开始 generate 循环 (最多 {args.rounds} 轮)...")
    for round_idx in range(args.rounds):
        t0 = time.time()
        resp = requests.post(f"{args.server}/omni/streaming_generate", stream=True, timeout=30)

        buffer = ""
        chunk_count = 0
        round_text = ""
        is_listen = False

        for raw_chunk in resp.iter_content(chunk_size=None, decode_unicode=True):
            if raw_chunk is None:
                continue
            buffer += raw_chunk
            while "\n\n" in buffer:
                idx_sep = buffer.index("\n\n")
                event_block = buffer[:idx_sep]
                buffer = buffer[idx_sep + 2:]
                for line in event_block.split("\n"):
                    if not line.startswith("data: "):
                        continue
                    try:
                        data = json.loads(line[6:])
                        if "chunk_data" in data:
                            cd = data["chunk_data"]
                            wav_b64 = cd.get("wav", "")
                            text = cd.get("text", "")
                            if wav_b64:
                                pcm_bytes = base64.b64decode(wav_b64)
                                all_pcm.append(pcm_bytes)
                                dur = len(pcm_bytes) // 2 / SAMPLE_RATE
                                total_audio_s += dur
                                chunk_count += 1
                            if text:
                                round_text += text
                                all_text += text
                        if "is_listen" in data:
                            is_listen = data["is_listen"]
                    except json.JSONDecodeError:
                        pass

        ms = (time.time() - t0) * 1000
        status = f"round={round_idx:2d} {ms:6.0f}ms chunks={chunk_count} audio={total_audio_s:.2f}s"
        if round_text:
            status += f' text="{round_text[:40]}"'
        if is_listen:
            status += " [LISTEN]"
        print(f"  {status}")

        # 如果连续 listen 且没有新文本，说明模型在等输入
        if is_listen and chunk_count == 0 and not round_text:
            # 再给几轮看看
            pass

        time.sleep(0.05)

    # 保存合并音频
    if all_pcm:
        merged = b"".join(all_pcm)
        out_path = os.path.join(args.output, "tts_output.wav")
        with wave.open(out_path, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(merged)
        duration = len(merged) / (SAMPLE_RATE * 2)
        print(f"\n✅ 音频已保存: {out_path}")
        print(f"   时长: {duration:.2f}s, 大小: {len(merged)//1024}KB")
    else:
        print("\n⚠️ 没有收到任何音频")

    if all_text:
        text_path = os.path.join(args.output, "tts_text.txt")
        with open(text_path, "w") as f:
            f.write(all_text)
        print(f"   文本: {all_text[:100]}...")
        print(f"   文本已保存: {text_path}")

    print(f"\n总计: {total_audio_s:.2f}s 音频, {len(all_text)} 字文本")


if __name__ == "__main__":
    main()
