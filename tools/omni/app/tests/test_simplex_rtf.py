"""单工 streaming RTF 测试

流程: init(simplex) → prefill(audio) → streaming_generate → 收集音频 chunks + 计时

测量:
  - 每个 audio chunk 的 RTF（inference_time / audio_duration）
  - 总体 RTF
  - 首响时间（first audio chunk latency）
"""
import os
import sys
import json
import time
import base64
import struct
import wave
import numpy as np
import requests
from typing import List, Dict, Optional

# ==================== 配置 ====================

SERVER_URL = "http://127.0.0.1:9060"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OMNI_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))  # tools/omni/
USER_AUDIO = os.path.join(OMNI_DIR, "assets/test_case/audio_test_case/audio_test_case_0000.wav")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "results/simplex_rtf")
SAMPLE_RATE = 24000  # TTS 输出采样率
TIMEOUT = 120


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_wav_to_base64(wav_path: str) -> str:
    """读取 WAV 文件并返回 base64 编码"""
    with open(wav_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def main() -> None:
    print("=" * 70)
    print("单工 Streaming RTF 测试")
    print(f"  服务器: {SERVER_URL}")
    print(f"  音频: {USER_AUDIO}")
    print("=" * 70)

    ensure_dir(OUTPUT_DIR)

    # 保存输入信息
    input_info = {
        "test": "test_simplex_rtf",
        "description": "单工模式 streaming RTF 性能测试",
        "input_audio": USER_AUDIO,
        "input_audio_basename": os.path.basename(USER_AUDIO),
        "input_image": None,
        "mode": "simplex",
    }
    with open(os.path.join(OUTPUT_DIR, "input_info.json"), "w", encoding="utf-8") as f:
        json.dump(input_info, f, ensure_ascii=False, indent=2)

    # Health check
    try:
        health = requests.get(f"{SERVER_URL}/health", timeout=5).json()
        print(f"\n[HEALTH] {json.dumps(health, ensure_ascii=False)}")
    except Exception as e:
        print(f"\n[ERROR] 无法连接: {e}")
        sys.exit(1)

    if not os.path.exists(USER_AUDIO):
        print(f"\n[ERROR] 音频不存在: {USER_AUDIO}")
        sys.exit(1)

    # Step 1: Init (simplex mode)
    print(f"\n{'='*70}")
    print("[1/3] 初始化 (单工模式)...")
    body = {"media_type": "omni", "duplex_mode": False, "language": "zh"}
    resp = requests.post(f"{SERVER_URL}/omni/init_sys_prompt", json=body, timeout=60)
    resp.raise_for_status()
    init_result = resp.json()
    print(f"  Result: {json.dumps(init_result, ensure_ascii=False)}")

    # Step 2: Prefill (audio only, no image for simplicity)
    print(f"\n[2/3] Prefill 音频...")
    audio_b64 = load_wav_to_base64(USER_AUDIO)
    print(f"  音频大小: {len(audio_b64)} bytes (base64)")

    t_prefill_start = time.time()
    resp = requests.post(
        f"{SERVER_URL}/omni/streaming_prefill",
        json={"audio": audio_b64},
        timeout=60,
    )
    resp.raise_for_status()
    prefill_ms = (time.time() - t_prefill_start) * 1000
    print(f"  Prefill 耗时: {prefill_ms:.0f}ms")

    # Step 3: Streaming Generate — 收集所有 audio chunks 并计时
    print(f"\n[3/3] Streaming Generate...")
    t_gen_start = time.time()

    resp = requests.post(
        f"{SERVER_URL}/omni/streaming_generate",
        headers={"Accept": "text/event-stream"},
        stream=True,
        timeout=TIMEOUT,
    )
    resp.raise_for_status()

    # 收集结果
    chunks: List[Dict] = []
    all_pcm: List[bytes] = []
    full_text: str = ""
    first_audio_ms: Optional[float] = None
    chunk_idx: int = 0
    last_chunk_time: float = t_gen_start

    buffer = ""
    for raw_chunk in resp.iter_content(chunk_size=None, decode_unicode=True):
        if raw_chunk is None:
            continue
        buffer += raw_chunk

        while "\n\n" in buffer:
            idx = buffer.index("\n\n")
            event_block = buffer[:idx]
            buffer = buffer[idx + 2:]

            for line in event_block.split("\n"):
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    continue
                try:
                    data = json.loads(payload)
                except json.JSONDecodeError:
                    continue

                if "chunk_data" in data:
                    now = time.time()
                    elapsed_ms = (now - t_gen_start) * 1000
                    inter_chunk_ms = (now - last_chunk_time) * 1000
                    last_chunk_time = now

                    cd = data["chunk_data"]
                    wav_b64 = cd.get("wav", "")
                    sr = cd.get("sample_rate", SAMPLE_RATE)
                    text = cd.get("text", "")

                    if text:
                        full_text += text

                    if wav_b64:
                        pcm_bytes = base64.b64decode(wav_b64)
                        audio_dur = len(pcm_bytes) / 2 / sr
                        all_pcm.append(pcm_bytes)

                        if first_audio_ms is None:
                            first_audio_ms = elapsed_ms

                        # 计算这个 chunk 的 RTF
                        # 用 inter_chunk_ms 作为推理时间（除了首个 chunk 用 elapsed）
                        infer_ms = elapsed_ms if chunk_idx == 0 else inter_chunk_ms
                        rtf = (infer_ms / 1000.0) / audio_dur if audio_dur > 0 else float("inf")

                        info = {
                            "chunk_idx": chunk_idx,
                            "elapsed_ms": round(elapsed_ms, 1),
                            "inter_chunk_ms": round(inter_chunk_ms, 1),
                            "audio_dur_s": round(audio_dur, 3),
                            "samples": len(pcm_bytes) // 2,
                            "rtf": round(rtf, 2),
                            "text": text[:30] if text else "",
                        }
                        chunks.append(info)

                        mark = "✅" if rtf < 1.0 else "⚠️"
                        print(f"  [{elapsed_ms:7.0f}ms] chunk#{chunk_idx:2d} "
                              f"| {audio_dur:.3f}s | Δ{inter_chunk_ms:7.0f}ms "
                              f"| RTF={rtf:.2f} {mark}"
                              f"{'  ' + text[:20] if text else ''}")
                        chunk_idx += 1

                if "is_listen" in data:
                    pass  # simplex 不关心 listen

    t_gen_end = time.time()
    gen_total_ms = (t_gen_end - t_gen_start) * 1000

    # 汇总
    total_audio_dur = sum(c["audio_dur_s"] for c in chunks)
    overall_rtf = (gen_total_ms / 1000.0) / total_audio_dur if total_audio_dur > 0 else float("inf")

    # 稳态 RTF（去掉首个 chunk）
    steady_chunks = [c for c in chunks if c["chunk_idx"] > 0]
    steady_audio_dur = sum(c["audio_dur_s"] for c in steady_chunks)
    steady_infer_ms = sum(c["inter_chunk_ms"] for c in steady_chunks)
    steady_rtf = (steady_infer_ms / 1000.0) / steady_audio_dur if steady_audio_dur > 0 else float("inf")

    print(f"\n{'='*70}")
    print("测试结果")
    print(f"{'='*70}")
    print(f"  文本输出: {full_text[:100]}...")
    print(f"  Prefill 耗时: {prefill_ms:.0f}ms")
    print(f"  Generate 总耗时: {gen_total_ms:.0f}ms")
    print(f"  音频 chunk 数: {len(chunks)}")
    print(f"  总音频时长: {total_audio_dur:.2f}s")
    print(f"  首响时间: {first_audio_ms:.0f}ms" if first_audio_ms else "  首响时间: N/A")
    print(f"  整体 RTF: {overall_rtf:.2f}x {'✅ 实时' if overall_rtf < 1.0 else '⚠️ 慢于实时'}")
    print(f"  稳态 RTF (去首chunk): {steady_rtf:.2f}x {'✅ 实时' if steady_rtf < 1.0 else '⚠️ 慢于实时'}")

    # 保存合并音频
    if all_pcm:
        merged = b"".join(all_pcm)
        wav_path = os.path.join(OUTPUT_DIR, "tts_output.wav")
        with wave.open(wav_path, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(merged)
        merged_dur = len(merged) / (SAMPLE_RATE * 2)
        print(f"\n  合并音频: {wav_path}")
        print(f"  时长: {merged_dur:.2f}s, 大小: {len(merged)//1024}KB")

    # 保存文本
    if full_text:
        text_path = os.path.join(OUTPUT_DIR, "response_text.txt")
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(full_text)
        print(f"  文本: {text_path}")

    # 保存详细日志
    log = {
        "prefill_ms": round(prefill_ms, 1),
        "gen_total_ms": round(gen_total_ms, 1),
        "first_audio_ms": round(first_audio_ms, 1) if first_audio_ms else None,
        "overall_rtf": round(overall_rtf, 2),
        "steady_rtf": round(steady_rtf, 2),
        "total_audio_dur_s": round(total_audio_dur, 2),
        "text": full_text,
        "chunks": chunks,
    }
    log_path = os.path.join(OUTPUT_DIR, "rtf_results.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)
    print(f"  详细日志: {log_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
