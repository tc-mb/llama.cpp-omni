"""omni_test_case 真实数据端到端测试

使用 assets/test_case/omni_test_case/ 下的滑雪视频帧 + 音频，
模拟双工视频通话场景，收集模型的文本和语音响应。

测试数据：
  - 9 帧 JPEG 图片（滑雪场景连续抽帧）
  - 9 段 1s WAV 音频（对应每帧的用户语音）

流程：
  1. init_sys_prompt (duplex/omni)
  2. 逐帧 prefill (audio + image)，每帧间隔 SEND_INTERVAL_S
  3. 并发 generate 收集文本 + 音频响应
  4. 保存合并 WAV、文本、性能统计

使用方法：
  # 先启动 server（双工模式）
  cd llama.cpp-omni && PYTHONPATH=. .venv/base/bin/python tools/omni/app/server.py --duplex

  # 另一终端运行测试
  cd llama.cpp-omni && PYTHONPATH=. .venv/base/bin/python tools/omni/app/tests/test_omni_realdata.py
"""
import argparse
import base64
import json
import os
import struct
import sys
import threading
import time
import wave
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests

# ==================== 路径 ====================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OMNI_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))  # tools/omni/
OMNI_TEST_CASE_DIR = os.path.join(OMNI_DIR, "assets/test_case/omni_test_case")
AUDIO_TEST_CASE_DIR = os.path.join(OMNI_DIR, "assets/test_case/audio_test_case")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "results/omni_realdata")

# ==================== 配置 ====================

SERVER_URL = "http://127.0.0.1:9060"
SEND_INTERVAL_S = 1.0       # 每帧发送间隔（模拟 1fps 视频通话）
TTS_SAMPLE_RATE = 24000      # TTS 输出采样率
TIMEOUT = 120                # HTTP 超时
POST_SEND_WAIT_S = 30        # 发完所有帧后等待 generate 收尾的时间


# ==================== 数据加载 ====================

def discover_test_frames(test_dir: str) -> List[Tuple[str, str]]:
    """扫描 omni_test_case 目录，返回 [(wav_path, jpg_path), ...] 按序号排列

    Args:
        test_dir: omni_test_case 目录路径

    Returns:
        按帧序号排序的 (wav, jpg) 路径列表

    Raises:
        FileNotFoundError: 目录不存在或无配对文件
    """
    if not os.path.isdir(test_dir):
        raise FileNotFoundError(f"测试数据目录不存在: {test_dir}")

    wavs = sorted([f for f in os.listdir(test_dir) if f.endswith(".wav")])
    jpgs = sorted([f for f in os.listdir(test_dir) if f.endswith(".jpg")])

    # 按基础名匹配配对
    wav_map = {os.path.splitext(f)[0]: f for f in wavs}
    jpg_map = {os.path.splitext(f)[0]: f for f in jpgs}
    common_keys = sorted(set(wav_map.keys()) & set(jpg_map.keys()))

    if not common_keys:
        raise FileNotFoundError(f"在 {test_dir} 中未找到配对的 wav+jpg 文件")

    pairs: List[Tuple[str, str]] = []
    for key in common_keys:
        pairs.append((
            os.path.join(test_dir, wav_map[key]),
            os.path.join(test_dir, jpg_map[key]),
        ))
    return pairs


def discover_audio_only_frames(test_dir: str) -> List[str]:
    """扫描 audio_test_case 目录，返回 [wav_path, ...] 按序号排列

    Args:
        test_dir: audio_test_case 目录路径

    Returns:
        按帧序号排序的 wav 路径列表
    """
    if not os.path.isdir(test_dir):
        return []
    wavs = sorted([f for f in os.listdir(test_dir) if f.endswith(".wav")])
    return [os.path.join(test_dir, f) for f in wavs]


def file_to_base64(path: str) -> str:
    """文件 → base64 编码"""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ==================== 共享状态 ====================

class TestState:
    """sendLoop 和 receiveLoop 之间的共享状态"""

    def __init__(self) -> None:
        self.is_active: bool = True
        self.start_time: float = 0.0
        self.lock: threading.Lock = threading.Lock()

        # 统计
        self.prefill_count: int = 0
        self.generate_rounds: int = 0
        self.audio_chunks_received: int = 0
        self.total_audio_duration_s: float = 0.0
        self.all_texts: List[str] = []
        self.all_pcm: List[bytes] = []

    def elapsed_s(self) -> float:
        return time.time() - self.start_time

    def log(self, tag: str, msg: str) -> None:
        print(f"  [{self.elapsed_s():6.1f}s] [{tag:<12}] {msg}", flush=True)


# ==================== Send Loop ====================

def send_loop(
    state: TestState,
    frames: List[Tuple[str, str]],
    mode: str,
) -> None:
    """逐帧发送 prefill（audio + image）

    Args:
        state: 共享状态
        frames: [(wav_path, jpg_path), ...] 或 [(wav_path, ""), ...]
        mode: "omni" 或 "audio"
    """
    for idx, (wav_path, jpg_path) in enumerate(frames):
        if not state.is_active:
            break

        # 首帧立即发，后续等间隔
        if idx > 0:
            time.sleep(SEND_INTERVAL_S)
        if not state.is_active:
            break

        audio_b64 = file_to_base64(wav_path)
        body: Dict[str, str] = {"audio": audio_b64}

        img_info = ""
        if jpg_path:
            image_b64 = file_to_base64(jpg_path)
            body["image"] = image_b64
            img_info = f" + {os.path.basename(jpg_path)}"

        try:
            t0 = time.time()
            resp = requests.post(
                f"{SERVER_URL}/omni/streaming_prefill",
                json=body,
                timeout=30,
            )
            resp.raise_for_status()
            ms = (time.time() - t0) * 1000

            with state.lock:
                state.prefill_count += 1

            state.log("PREFILL", f"#{idx} {os.path.basename(wav_path)}{img_info} → {ms:.0f}ms")
        except Exception as e:
            state.log("PREFILL_ERR", f"#{idx} {e}")

    state.log("SEND_DONE", f"全部 {state.prefill_count} 帧发送完毕")


# ==================== Receive Loop ====================

def receive_loop(state: TestState, output_dir: str) -> None:
    """持续调用 generate，收集文本和音频

    Args:
        state: 共享状态
        output_dir: 音频 chunk 保存目录
    """
    round_idx = 0

    while state.is_active:
        try:
            t0 = time.time()
            resp = requests.post(
                f"{SERVER_URL}/omni/streaming_generate",
                headers={"Accept": "text/event-stream"},
                stream=True,
                timeout=TIMEOUT,
            )
            resp.raise_for_status()

            chunk_count = 0
            round_audio_s = 0.0
            round_text = ""
            is_listen = False

            buffer = ""
            for raw_chunk in resp.iter_content(chunk_size=None, decode_unicode=True):
                if raw_chunk is None:
                    continue
                buffer += raw_chunk

                while "\n\n" in buffer:
                    sep_idx = buffer.index("\n\n")
                    event_block = buffer[:sep_idx]
                    buffer = buffer[sep_idx + 2:]

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
                            cd = data["chunk_data"]
                            wav_b64 = cd.get("wav", "")
                            sr = cd.get("sample_rate", TTS_SAMPLE_RATE)
                            text = cd.get("text", "")

                            if text:
                                round_text += text

                            if wav_b64:
                                pcm_bytes = base64.b64decode(wav_b64)
                                dur = len(pcm_bytes) / 2 / sr
                                chunk_count += 1
                                round_audio_s += dur

                                with state.lock:
                                    state.audio_chunks_received += 1
                                    state.total_audio_duration_s += dur
                                    state.all_pcm.append(pcm_bytes)

                                # 保存单个 chunk
                                chunk_path = os.path.join(
                                    output_dir,
                                    f"round{round_idx:02d}_chunk{chunk_count:03d}.wav",
                                )
                                _save_pcm_as_wav(pcm_bytes, sr, chunk_path)

                                state.log("AUDIO", f"R{round_idx}C{chunk_count} {dur:.3f}s")

                        if "is_listen" in data:
                            is_listen = data["is_listen"]

            gen_ms = (time.time() - t0) * 1000

            with state.lock:
                state.generate_rounds += 1
                if round_text:
                    state.all_texts.append(round_text)

            detail = f"R{round_idx} {gen_ms:.0f}ms | {chunk_count} chunks {round_audio_s:.2f}s"
            if round_text:
                detail += f' | "{round_text[:60]}"'
            if is_listen:
                detail += " [LISTEN]"
            state.log("GEN_DONE", detail)

            time.sleep(0.1)
            round_idx += 1

        except Exception as e:
            state.log("GEN_ERR", f"R{round_idx}: {e}")
            if state.is_active:
                time.sleep(1.0)
            round_idx += 1

    state.log("RECV_DONE", f"共 {state.generate_rounds} 轮 generate")


# ==================== 工具函数 ====================

def _save_pcm_as_wav(pcm_bytes: bytes, sample_rate: int, wav_path: str) -> None:
    """将 PCM int16 LE 字节保存为 WAV 文件"""
    with open(wav_path, "wb") as f:
        data_size = len(pcm_bytes)
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + data_size))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(struct.pack("<I", 16))
        f.write(struct.pack("<H", 1))       # PCM
        f.write(struct.pack("<H", 1))       # mono
        f.write(struct.pack("<I", sample_rate))
        f.write(struct.pack("<I", sample_rate * 2))
        f.write(struct.pack("<H", 2))       # block align
        f.write(struct.pack("<H", 16))      # bits per sample
        f.write(b"data")
        f.write(struct.pack("<I", data_size))
        f.write(pcm_bytes)


def _inject_runtime_config(server_url: str, send_interval: float, post_wait: float) -> None:
    """将命令行参数注入模块级变量（供 send_loop/receive_loop 使用）"""
    g = globals()
    g["SERVER_URL"] = server_url
    g["SEND_INTERVAL_S"] = send_interval
    g["POST_SEND_WAIT_S"] = post_wait


# ==================== 主流程 ====================

def run_omni_test(frames: List[Tuple[str, str]], mode: str, output_dir: str) -> Dict:
    """运行一次完整的 omni 测试

    Args:
        frames: [(wav_path, jpg_path), ...]
        mode: "omni" | "audio"
        output_dir: 结果输出目录

    Returns:
        测试结果字典
    """
    os.makedirs(output_dir, exist_ok=True)

    # Init
    media_type = "omni" if mode == "omni" else "audio"
    print(f"\n[INIT] mode={mode}, media_type={media_type}, duplex=True")
    resp = requests.post(
        f"{SERVER_URL}/omni/init_sys_prompt",
        json={"media_type": media_type, "duplex_mode": True, "language": "zh"},
        timeout=60,
    )
    resp.raise_for_status()
    init_result = resp.json()
    print(f"  → {json.dumps(init_result, ensure_ascii=False)}")

    # 启动并发 send + receive
    state = TestState()
    state.start_time = time.time()

    send_thread = threading.Thread(
        target=send_loop, args=(state, frames, mode), name="sendLoop", daemon=True,
    )
    recv_thread = threading.Thread(
        target=receive_loop, args=(state, output_dir), name="recvLoop", daemon=True,
    )

    print(f"\n{'='*60}")
    print(f"并发启动 sendLoop ({len(frames)} 帧) + receiveLoop")
    print(f"{'='*60}")

    send_thread.start()
    recv_thread.start()

    # 等 sendLoop 完成
    send_thread.join()

    # 发送完后等待 generate 收尾
    print(f"\n[MAIN] 发送完毕，等待 {POST_SEND_WAIT_S}s 收集剩余输出...")
    time.sleep(POST_SEND_WAIT_S)
    state.is_active = False

    recv_thread.join(timeout=10)

    # stop
    try:
        requests.post(f"{SERVER_URL}/omni/stop", timeout=5)
    except Exception:
        pass

    total_time = time.time() - state.start_time

    # 保存合并音频
    merged_wav_path = os.path.join(output_dir, "tts_output.wav")
    if state.all_pcm:
        merged = b"".join(state.all_pcm)
        with wave.open(merged_wav_path, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(TTS_SAMPLE_RATE)
            wf.writeframes(merged)
        audio_file_dur = len(merged) / (TTS_SAMPLE_RATE * 2)
        print(f"\n[AUDIO] 合并 WAV: {merged_wav_path}")
        print(f"  时长: {audio_file_dur:.2f}s, 大小: {len(merged)//1024}KB")
    else:
        audio_file_dur = 0.0
        print("\n[AUDIO] 未收到任何音频")

    # 保存文本
    full_text = "".join(state.all_texts)
    if full_text:
        text_path = os.path.join(output_dir, "response_text.txt")
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(full_text)
        print(f"[TEXT] {text_path}")
        print(f"  内容: {full_text[:200]}")

    # 统计
    rtf = total_time / state.total_audio_duration_s if state.total_audio_duration_s > 0 else float("inf")
    result = {
        "mode": mode,
        "n_frames": len(frames),
        "total_time_s": round(total_time, 1),
        "prefill_count": state.prefill_count,
        "generate_rounds": state.generate_rounds,
        "audio_chunks": state.audio_chunks_received,
        "total_audio_s": round(state.total_audio_duration_s, 2),
        "rtf": round(rtf, 2),
        "text": full_text,
        "merged_wav": merged_wav_path if state.all_pcm else "",
    }

    # 保存结果 JSON
    result_path = os.path.join(output_dir, "result.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="omni_test_case 真实数据端到端测试",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 双工 omni 模式（音频+视频，默认）
  python tools/omni/app/tests/test_omni_realdata.py

  # 纯音频模式
  python tools/omni/app/tests/test_omni_realdata.py --audio-only

  # 指定帧数（前 N 帧）
  python tools/omni/app/tests/test_omni_realdata.py --max-frames 3
""",
    )
    parser.add_argument("--server", default=SERVER_URL, help="服务器地址")
    parser.add_argument("--audio-only", action="store_true", help="纯音频模式（不发图片）")
    parser.add_argument("--max-frames", type=int, default=-1, help="最多使用 N 帧 (-1=全部)")
    parser.add_argument("--interval", type=float, default=SEND_INTERVAL_S, help="帧间隔(秒)")
    parser.add_argument("--wait", type=float, default=POST_SEND_WAIT_S, help="发完后等待(秒)")
    parser.add_argument("--output", default=OUTPUT_DIR, help="输出目录")
    args = parser.parse_args()

    server_url = args.server
    send_interval = args.interval
    post_wait = args.wait
    output_dir = args.output

    print("=" * 60)
    print("omni_test_case 真实数据端到端测试")
    print("=" * 60)

    # Health check
    try:
        health = requests.get(f"{server_url}/health", timeout=5).json()
        print(f"[HEALTH] {json.dumps(health, ensure_ascii=False)}")
    except Exception as e:
        print(f"[ERROR] 无法连接 {server_url}: {e}")
        sys.exit(1)

    # 加载测试数据
    if args.audio_only:
        # 纯音频模式：使用 audio_test_case 或 omni_test_case 的 wav（不带图）
        audio_paths = discover_audio_only_frames(AUDIO_TEST_CASE_DIR)
        if not audio_paths:
            # fallback: 用 omni_test_case 的 wav
            omni_frames = discover_test_frames(OMNI_TEST_CASE_DIR)
            audio_paths = [wav for wav, _ in omni_frames]

        if args.max_frames > 0:
            audio_paths = audio_paths[:args.max_frames]

        frames: List[Tuple[str, str]] = [(wav, "") for wav in audio_paths]
        mode = "audio"
        print(f"\n[DATA] 纯音频模式: {len(frames)} 帧")
        for wav, _ in frames:
            print(f"  {os.path.basename(wav)}")
    else:
        # omni 模式：音频 + 图片
        frames = discover_test_frames(OMNI_TEST_CASE_DIR)
        if args.max_frames > 0:
            frames = frames[:args.max_frames]
        mode = "omni"
        print(f"\n[DATA] Omni 模式 (音频+视频): {len(frames)} 帧")
        for wav, jpg in frames:
            print(f"  {os.path.basename(wav)} + {os.path.basename(jpg)}")

    print(f"\n[CONFIG] 帧间隔={send_interval}s, 收尾等待={post_wait}s")
    print(f"[CONFIG] 输出目录: {output_dir}")

    # 保存输入信息
    os.makedirs(output_dir, exist_ok=True)
    input_info = {
        "test": "test_omni_realdata",
        "description": f"多模态真实数据端到端测试（{mode} 模式）",
        "mode": mode,
        "input_frames": [
            {
                "audio": wav,
                "audio_basename": os.path.basename(wav),
                "image": jpg if jpg else None,
                "image_basename": os.path.basename(jpg) if jpg else None,
            }
            for wav, jpg in frames
        ],
        "send_interval_s": send_interval,
        "post_wait_s": post_wait,
    }
    with open(os.path.join(output_dir, "input_info.json"), "w", encoding="utf-8") as f:
        json.dump(input_info, f, ensure_ascii=False, indent=2)

    # 运行测试（将运行时参数注入模块级变量供 send_loop/receive_loop 使用）
    _inject_runtime_config(server_url, send_interval, post_wait)
    result = run_omni_test(frames, mode, output_dir)

    # 最终总结
    rtf_mark = "✅ 实时" if result["rtf"] < 1.0 else "⚠️ 慢于实时"
    print(f"\n{'='*60}")
    print("测试结果")
    print(f"{'='*60}")
    print(f"  模式:          {result['mode']}")
    print(f"  输入帧数:      {result['n_frames']}")
    print(f"  总耗时:        {result['total_time_s']}s")
    print(f"  Prefill 次数:  {result['prefill_count']}")
    print(f"  Generate 轮数: {result['generate_rounds']}")
    print(f"  音频 chunks:   {result['audio_chunks']}")
    print(f"  总音频时长:    {result['total_audio_s']}s")
    print(f"  RTF:           {result['rtf']}x {rtf_mark}")
    if result["text"]:
        print(f"  文本:          {result['text'][:100]}...")
    if result["merged_wav"]:
        print(f"\n  🔊 合并音频:   {result['merged_wav']}")
        print(f"     (用 macOS: open {result['merged_wav']})")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
