"""后端 API 端到端测试脚本

测试单工和双工模式下的完整流程：
  init_sys_prompt → streaming_prefill → streaming_generate

保存所有输入输出到 tmp/test/backend_api_test/ 供人工检查。
"""
import os
import sys
import json
import time
import base64
import struct
import wave
import requests

# ==================== 配置 ====================

SERVER_URL = "http://127.0.0.1:9060"
WORKSPACE = "/Users/sunweiyue/Desktop/lib/minicpm-o-4_5-macOS"
TEST_AUDIO = os.path.join(
    WORKSPACE, "llama.cpp-omni/tools/omni/assets/test_case/audio_test_case/audio_test_case_0000.wav"
)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "results/backend_api_test")
TIMEOUT = 120  # seconds


# ==================== 工具函数 ====================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def wav_file_to_base64(wav_path: str) -> str:
    """读取 WAV 文件并编码为 base64"""
    with open(wav_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def save_json(path: str, data: object) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_text(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def decode_pcm_base64_to_wav(pcm_b64: str, sample_rate: int, wav_path: str) -> None:
    """将 base64 编码的 PCM int16 数据保存为 WAV 文件"""
    pcm_bytes = base64.b64decode(pcm_b64)
    num_samples = len(pcm_bytes) // 2

    with open(wav_path, "wb") as f:
        # WAV header
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + len(pcm_bytes)))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(struct.pack("<I", 16))        # chunk size
        f.write(struct.pack("<H", 1))         # PCM format
        f.write(struct.pack("<H", 1))         # mono
        f.write(struct.pack("<I", sample_rate))
        f.write(struct.pack("<I", sample_rate * 2))  # byte rate
        f.write(struct.pack("<H", 2))         # block align
        f.write(struct.pack("<H", 16))        # bits per sample
        f.write(b"data")
        f.write(struct.pack("<I", len(pcm_bytes)))
        f.write(pcm_bytes)

    duration = num_samples / sample_rate
    print(f"  [保存WAV] {os.path.basename(wav_path)} ({num_samples} samples, {duration:.2f}s, {sample_rate}Hz)")


# ==================== API 调用 ====================

def api_health() -> dict:
    resp = requests.get(f"{SERVER_URL}/health", timeout=10)
    resp.raise_for_status()
    return resp.json()


def api_init(media_type: str, duplex_mode: bool, language: str = "zh") -> dict:
    """调用 /omni/init_sys_prompt"""
    body = {
        "media_type": media_type,
        "duplex_mode": duplex_mode,
        "language": language,
    }
    print(f"\n[INIT] POST /omni/init_sys_prompt")
    print(f"  body: {json.dumps(body, ensure_ascii=False)}")

    resp = requests.post(
        f"{SERVER_URL}/omni/init_sys_prompt",
        json=body,
        timeout=30,
    )
    resp.raise_for_status()
    result = resp.json()
    print(f"  status: {resp.status_code}")
    print(f"  response: {json.dumps(result, ensure_ascii=False)}")
    return result


def api_prefill(audio_b64: str, image_b64: str = None) -> dict:
    """调用 /omni/streaming_prefill"""
    body = {}
    if audio_b64:
        body["audio"] = audio_b64
    if image_b64:
        body["image"] = image_b64

    print(f"\n[PREFILL] POST /omni/streaming_prefill")
    print(f"  audio: {len(audio_b64)} chars base64" if audio_b64 else "  audio: None")
    print(f"  image: {len(image_b64)} chars base64" if image_b64 else "  image: None")

    resp = requests.post(
        f"{SERVER_URL}/omni/streaming_prefill",
        json=body,
        timeout=30,
    )
    resp.raise_for_status()
    result = resp.json()
    print(f"  status: {resp.status_code}")
    print(f"  response: {json.dumps(result, ensure_ascii=False)}")
    return result


def api_generate(output_dir: str) -> tuple:
    """调用 /omni/streaming_generate，解析 SSE 流，保存音频和文本
    
    Returns:
        (result_dict, sse_events_list)
    """
    print(f"\n[GENERATE] POST /omni/streaming_generate (SSE)")

    sse_events = []
    audio_chunks = []
    all_pcm: list = []       # 累积 PCM bytes 用于合并
    tts_sample_rate: int = 24000
    full_text = ""
    is_listen = False
    is_break = False
    chunk_count = 0

    resp = requests.post(
        f"{SERVER_URL}/omni/streaming_generate",
        headers={"Accept": "text/event-stream"},
        stream=True,
        timeout=TIMEOUT,
    )
    print(f"  status: {resp.status_code}")
    resp.raise_for_status()

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
                    sse_events.append(data)

                    if "chunk_data" in data:
                        chunk = data["chunk_data"]
                        chunk_count += 1
                        wav_b64 = chunk.get("wav", "")
                        sr = chunk.get("sample_rate", 24000)
                        text_piece = chunk.get("text", "")

                        if text_piece:
                            full_text += text_piece

                        if wav_b64:
                            pcm_bytes = base64.b64decode(wav_b64)
                            all_pcm.append(pcm_bytes)
                            tts_sample_rate = sr
                            audio_chunks.append({
                                "idx": data.get("chunk_idx", chunk_count - 1),
                                "wav_b64_len": len(wav_b64),
                                "sample_rate": sr,
                                "text": text_piece,
                            })
                            wav_path = os.path.join(
                                output_dir,
                                f"output_chunk_{data.get('chunk_idx', chunk_count - 1):03d}.wav"
                            )
                            decode_pcm_base64_to_wav(wav_b64, sr, wav_path)

                        print(f"  chunk #{data.get('chunk_idx', '?')}: "
                              f"wav={len(wav_b64)} chars, sr={sr}, "
                              f"text={repr(text_piece[:50]) if text_piece else 'None'}")

                    if data.get("break"):
                        is_break = True
                        print(f"  [BREAK] {json.dumps(data, ensure_ascii=False)}")

                    if data.get("done"):
                        is_listen = data.get("is_listen", False)
                        print(f"  [DONE] is_listen={is_listen}")

                except json.JSONDecodeError as e:
                    print(f"  [JSON ERROR] {e}: {line[:100]}")

    # 保存合并音频
    merged_wav_path = ""
    if all_pcm:
        merged_pcm = b"".join(all_pcm)
        merged_wav_path = os.path.join(output_dir, "tts_output.wav")
        with wave.open(merged_wav_path, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(tts_sample_rate)
            wf.writeframes(merged_pcm)
        audio_dur = len(merged_pcm) / (tts_sample_rate * 2)
        print(f"\n  合并音频: {merged_wav_path}")
        print(f"  时长: {audio_dur:.2f}s, 大小: {len(merged_pcm)//1024}KB")

    # 保存文本
    if full_text:
        text_path = os.path.join(output_dir, "response_text.txt")
        save_text(text_path, full_text)
        print(f"  文本: {text_path}")

    result = {
        "chunk_count": chunk_count,
        "full_text": full_text,
        "is_listen": is_listen,
        "is_break": is_break,
        "sse_event_count": len(sse_events),
        "audio_chunks_summary": audio_chunks,
        "merged_wav": merged_wav_path,
    }

    print(f"\n[GENERATE 结果]")
    print(f"  总 chunk 数: {chunk_count}")
    print(f"  完整文本: {repr(full_text[:200])}")
    print(f"  is_listen: {is_listen}")
    print(f"  is_break: {is_break}")
    if merged_wav_path:
        print(f"  合并音频: {merged_wav_path}")

    return result, sse_events


def api_stop() -> None:
    try:
        requests.post(f"{SERVER_URL}/omni/stop", timeout=5)
    except Exception:
        pass


# ==================== 测试流程 ====================

def test_simplex_audio(test_dir: str) -> None:
    """测试单工音频模式"""
    print("\n" + "=" * 60)
    print("测试: 单工音频模式 (Voice Chat)")
    print("=" * 60)

    ensure_dir(test_dir)

    # 保存输入信息
    input_info = {
        "test": "test_backend_api (simplex)",
        "description": "单工端到端 API 测试：init → prefill → streaming_generate → stop",
        "input_audio": TEST_AUDIO,
        "input_audio_basename": os.path.basename(TEST_AUDIO),
        "input_image": None,
        "mode": "simplex",
    }
    save_json(os.path.join(test_dir, "input_info.json"), input_info)

    # 1. 读取测试音频
    audio_b64 = wav_file_to_base64(TEST_AUDIO)
    save_text(os.path.join(test_dir, "input_audio.b64.txt"), audio_b64[:200] + "...[truncated]")
    print(f"\n输入音频: {TEST_AUDIO}")
    print(f"  base64 长度: {len(audio_b64)} chars")

    # 2. Init
    init_result = api_init("audio", False)
    save_json(os.path.join(test_dir, "01_init_response.json"), init_result)

    # 3. Prefill
    prefill_result = api_prefill(audio_b64)
    save_json(os.path.join(test_dir, "02_prefill_response.json"), prefill_result)

    # 4. Generate
    gen_result, sse_events = api_generate(test_dir)
    save_json(os.path.join(test_dir, "03_generate_result.json"), gen_result)
    # SSE events contain huge base64, save only metadata
    sse_summary = []
    for ev in sse_events:
        summary = dict(ev)
        if "chunk_data" in summary:
            cd = dict(summary["chunk_data"])
            if "wav" in cd:
                cd["wav"] = f"[{len(cd['wav'])} chars base64]"
            summary["chunk_data"] = cd
        sse_summary.append(summary)
    save_json(os.path.join(test_dir, "03_sse_events_summary.json"), sse_summary)

    # 5. Stop
    api_stop()
    print("\n[STOP] 已发送 stop 指令")


def test_duplex_audio(test_dir: str) -> None:
    """测试双工音频模式 (模拟 Video Call 的第一轮)"""
    print("\n" + "=" * 60)
    print("测试: 双工音频模式 (Video Call)")
    print("=" * 60)

    ensure_dir(test_dir)

    # 保存输入信息
    input_info = {
        "test": "test_backend_api (duplex)",
        "description": "双工端到端 API 测试：init(duplex) → prefill → streaming_generate → stop",
        "input_audio": TEST_AUDIO,
        "input_audio_basename": os.path.basename(TEST_AUDIO),
        "input_image": None,
        "mode": "duplex",
    }
    save_json(os.path.join(test_dir, "input_info.json"), input_info)

    # 1. 读取测试音频
    audio_b64 = wav_file_to_base64(TEST_AUDIO)
    save_text(os.path.join(test_dir, "input_audio.b64.txt"), audio_b64[:200] + "...[truncated]")

    # 2. Init (双工)
    init_result = api_init("omni", True)
    save_json(os.path.join(test_dir, "01_init_response.json"), init_result)

    # 关键检查: duplex_mode 是否被接受
    actual_duplex = init_result.get("duplex_mode")
    if actual_duplex is False:
        print("\n  ⚠️⚠️⚠️  后端拒绝了 duplex_mode=True，回退为单工模式!")
        print("  根因: server 预初始化时用了 simplex 模式，运行时不可切换")
        save_text(
            os.path.join(test_dir, "DUPLEX_REJECTED.txt"),
            "后端拒绝了 duplex_mode=True 请求。\n"
            "根因: server 预初始化时用了 simplex 模式 (omni_init duplex_mode=False)\n"
            "运行时调用 init_sys_prompt(duplex_mode=True) 被忽略。\n\n"
            f"Init response: {json.dumps(init_result, ensure_ascii=False, indent=2)}\n"
        )
    else:
        print(f"\n  ✅ 双工模式已启用 (duplex_mode={actual_duplex})")

    # 3. Prefill
    prefill_result = api_prefill(audio_b64)
    save_json(os.path.join(test_dir, "02_prefill_response.json"), prefill_result)

    # 4. Generate
    gen_result, sse_events = api_generate(test_dir)
    save_json(os.path.join(test_dir, "03_generate_result.json"), gen_result)
    sse_summary = []
    for ev in sse_events:
        summary = dict(ev)
        if "chunk_data" in summary:
            cd = dict(summary["chunk_data"])
            if "wav" in cd:
                cd["wav"] = f"[{len(cd['wav'])} chars base64]"
            summary["chunk_data"] = cd
        sse_summary.append(summary)
    save_json(os.path.join(test_dir, "03_sse_events_summary.json"), sse_summary)

    # 5. Stop
    api_stop()
    print("\n[STOP] 已发送 stop 指令")


def main() -> None:
    print("=" * 60)
    print("后端 API 端到端测试")
    print(f"服务器: {SERVER_URL}")
    print(f"测试音频: {TEST_AUDIO}")
    print(f"输出目录: {OUTPUT_DIR}")
    print("=" * 60)

    # 检查健康状态
    try:
        health = api_health()
        print(f"\n[HEALTH] {json.dumps(health, ensure_ascii=False)}")
        print(f"  当前 duplex_mode: {health.get('duplex_mode')}")
    except Exception as e:
        print(f"\n[ERROR] 无法连接服务器: {e}")
        sys.exit(1)

    if not os.path.exists(TEST_AUDIO):
        print(f"\n[ERROR] 测试音频不存在: {TEST_AUDIO}")
        sys.exit(1)

    # 1. 测试单工模式
    simplex_dir = os.path.join(OUTPUT_DIR, "simplex_audio")
    test_simplex_audio(simplex_dir)

    time.sleep(2)

    # 2. 测试双工模式
    duplex_dir = os.path.join(OUTPUT_DIR, "duplex_audio")
    test_duplex_audio(duplex_dir)

    print("\n\n" + "=" * 60)
    print("测试完成！检查输出:")
    print(f"  {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
