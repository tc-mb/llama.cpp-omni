"""双工模式音频流模拟测试（正确的并发模式 + 图片）

模拟前端 Video Call 的双工交互流：
  init(duplex) → 并发启动 sendLoop + receiveLoop

前端的真实行为：
  - duplexSendLoop: 每 2000ms 调一次 prefill(audio, image)
      - image = captureFrame() → 640x480 JPEG 0.7 base64
  - duplexReceiveLoop: 不断调 generate()，is_listen=True 时立即再调
  - 两个循环 **并发独立运行**

本脚本用两个线程模拟这个并发模式，每次 prefill 同时发送音频 + 图片。
"""
import os
import sys
import json
import time
import base64
import struct
import threading
import wave
import numpy as np
import requests
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

# ==================== 配置 ====================

SERVER_URL = "http://127.0.0.1:9060"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OMNI_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))  # tools/omni/
USER_AUDIO = os.path.join(OMNI_DIR, "assets/test_case/audio_test_case/audio_test_case_0000.wav")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "results/duplex_stream")

# 音频流参数
SEND_INTERVAL_S = 2.0        # 前端默认 2000ms
SAMPLE_RATE = 16000
TOTAL_STREAM_DURATION_S = 40  # 总流时长（秒）— 足够长让模型持续对话
IMAGE_SWITCH_TIME_S = 10.0    # 10s 时切换到第二张图
TIMEOUT = 120


# ==================== 图片生成 ====================

def generate_test_image_night_room() -> str:
    """场景 A: 夜晚房间 — 桌子、台灯、窗户看到星空"""
    from PIL import Image, ImageDraw
    
    w, h = 640, 480
    img = Image.new("RGB", (w, h), (40, 45, 55))
    draw = ImageDraw.Draw(img)
    
    # 窗户
    draw.rectangle([420, 30, 580, 200], fill=(20, 30, 60), outline=(120, 120, 120), width=3)
    draw.rectangle([430, 40, 500, 110], fill=(30, 40, 80))
    draw.rectangle([510, 40, 570, 110], fill=(30, 40, 80))
    draw.rectangle([430, 120, 500, 190], fill=(25, 35, 70))
    draw.rectangle([510, 120, 570, 190], fill=(25, 35, 70))
    for x, y in [(450, 60), (540, 80), (470, 150)]:
        draw.ellipse([x-2, y-2, x+2, y+2], fill=(255, 255, 200))
    
    # 桌子
    draw.rectangle([50, 320, 400, 340], fill=(120, 80, 40))
    draw.rectangle([70, 340, 90, 450], fill=(100, 65, 30))
    draw.rectangle([360, 340, 380, 450], fill=(100, 65, 30))
    
    # 台灯
    draw.rectangle([180, 240, 200, 320], fill=(80, 80, 80))
    draw.polygon([(140, 240), (240, 240), (220, 200), (160, 200)], fill=(255, 220, 100))
    draw.ellipse([130, 190, 250, 250], fill=(255, 240, 180, 40))
    
    # 书本 + 杯子
    draw.rectangle([260, 300, 340, 320], fill=(200, 50, 50))
    draw.rectangle([100, 290, 130, 320], fill=(200, 200, 220))
    
    # 地板 + 天花板灯
    draw.rectangle([0, 450, w, h], fill=(80, 60, 40))
    draw.ellipse([280, 5, 340, 30], fill=(255, 250, 230))
    
    import io
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=70)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def generate_test_image_park() -> str:
    """场景 B: 白天公园 — 蓝天白云、绿地、大树、长椅、小路"""
    from PIL import Image, ImageDraw
    
    w, h = 640, 480
    img = Image.new("RGB", (w, h), (135, 200, 250))  # 蓝天
    draw = ImageDraw.Draw(img)
    
    # 白云
    for cx, cy, rx, ry in [(150, 60, 60, 25), (170, 50, 40, 18),
                            (450, 80, 55, 22), (470, 70, 35, 16)]:
        draw.ellipse([cx-rx, cy-ry, cx+rx, cy+ry], fill=(255, 255, 255))
    
    # 远山
    draw.polygon([(0, 250), (100, 180), (200, 220), (320, 160), (450, 210),
                  (550, 170), (640, 230), (640, 280), (0, 280)], fill=(100, 160, 80))
    
    # 草地
    draw.rectangle([0, 270, w, h], fill=(80, 170, 60))
    # 浅色草地条纹
    for y in range(280, h, 30):
        draw.rectangle([0, y, w, y+8], fill=(90, 180, 70))
    
    # 大树（左侧）
    draw.rectangle([80, 200, 105, 370], fill=(100, 70, 30))   # 树干
    draw.ellipse([30, 100, 155, 220], fill=(50, 130, 40))      # 树冠
    draw.ellipse([45, 130, 140, 240], fill=(60, 140, 50))
    
    # 小树（右侧）
    draw.rectangle([500, 240, 515, 340], fill=(110, 75, 35))
    draw.ellipse([470, 170, 545, 260], fill=(55, 135, 45))
    
    # 小路（弯曲）
    draw.polygon([(280, 480), (360, 480), (340, 380), (310, 310),
                  (320, 280), (300, 280), (290, 310), (300, 380)],
                 fill=(190, 175, 140))
    
    # 长椅
    draw.rectangle([380, 330, 460, 340], fill=(140, 90, 40))  # 座面
    draw.rectangle([380, 340, 390, 370], fill=(120, 75, 30))  # 左腿
    draw.rectangle([450, 340, 460, 370], fill=(120, 75, 30))  # 右腿
    draw.rectangle([375, 310, 380, 340], fill=(140, 90, 40))  # 靠背左
    draw.rectangle([460, 310, 465, 340], fill=(140, 90, 40))  # 靠背右
    draw.rectangle([375, 310, 465, 318], fill=(140, 90, 40))  # 靠背横
    
    # 太阳
    draw.ellipse([530, 20, 590, 80], fill=(255, 230, 80))
    
    import io
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=70)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ==================== 共享状态 ====================

@dataclass
class SharedState:
    """sendLoop 和 receiveLoop 之间共享的状态"""
    is_in_call: bool = True
    
    # 统计
    prefill_count: int = 0
    generate_round_count: int = 0
    total_audio_chunks_received: int = 0
    total_audio_duration_s: float = 0.0
    all_texts: List[str] = field(default_factory=list)
    all_pcm: List[bytes] = field(default_factory=list)
    tts_sample_rate: int = 24000
    
    # 时序记录
    events: List[Dict] = field(default_factory=list)
    stream_start_time: float = 0.0
    
    # 锁
    lock: threading.Lock = field(default_factory=threading.Lock)
    
    def log_event(self, event_type: str, detail: str) -> None:
        elapsed = time.time() - self.stream_start_time
        entry = {
            "elapsed_s": round(elapsed, 2),
            "type": event_type,
            "detail": detail,
            "timestamp": datetime.now().strftime("%H:%M:%S.%f")[:12],
        }
        with self.lock:
            self.events.append(entry)
        print(f"  [{elapsed:6.1f}s] [{event_type:<12}] {detail}", flush=True)


# ==================== 工具函数 ====================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_wav_pcm16(wav_path: str) -> np.ndarray:
    """读取 WAV 文件返回 float32 数组 [-1, 1]"""
    import soundfile as sf
    data, sr = sf.read(wav_path, dtype='float32')
    if sr != SAMPLE_RATE:
        import librosa
        data = librosa.resample(data, orig_sr=sr, target_sr=SAMPLE_RATE)
    return data


def audio_chunk_to_wav_base64(audio_float32: np.ndarray, sr: int = SAMPLE_RATE) -> str:
    """将 float32 音频转为 WAV base64"""
    import io
    pcm16 = (np.clip(audio_float32, -1.0, 1.0) * 32767).astype(np.int16)
    buf = io.BytesIO()
    pcm_bytes = pcm16.tobytes()
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + len(pcm_bytes)))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<I", 16))
    buf.write(struct.pack("<H", 1))
    buf.write(struct.pack("<H", 1))
    buf.write(struct.pack("<I", sr))
    buf.write(struct.pack("<I", sr * 2))
    buf.write(struct.pack("<H", 2))
    buf.write(struct.pack("<H", 16))
    buf.write(b"data")
    buf.write(struct.pack("<I", len(pcm_bytes)))
    buf.write(pcm_bytes)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def decode_pcm_base64_to_wav(pcm_b64: str, sample_rate: int, wav_path: str) -> float:
    """将 base64 PCM int16 保存为 WAV，返回时长(秒)"""
    pcm_bytes = base64.b64decode(pcm_b64)
    num_samples = len(pcm_bytes) // 2
    duration = num_samples / sample_rate
    with open(wav_path, "wb") as f:
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + len(pcm_bytes)))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(struct.pack("<I", 16))
        f.write(struct.pack("<H", 1))
        f.write(struct.pack("<H", 1))
        f.write(struct.pack("<I", sample_rate))
        f.write(struct.pack("<I", sample_rate * 2))
        f.write(struct.pack("<H", 2))
        f.write(struct.pack("<H", 16))
        f.write(b"data")
        f.write(struct.pack("<I", len(pcm_bytes)))
        f.write(pcm_bytes)
    return duration


# ==================== 音频流准备 ====================

def prepare_audio_stream(audio_path: str) -> List[Tuple[str, np.ndarray]]:
    """准备音频流：用户音频 + zero pad 静音，按 SEND_INTERVAL_S 切块"""
    chunk_samples = int(SEND_INTERVAL_S * SAMPLE_RATE)
    total_samples = int(TOTAL_STREAM_DURATION_S * SAMPLE_RATE)
    
    user_audio = load_wav_pcm16(audio_path)
    print(f"用户音频: {len(user_audio)} samples ({len(user_audio)/SAMPLE_RATE:.2f}s)")
    
    full_stream = np.zeros(total_samples, dtype=np.float32)
    copy_len = min(len(user_audio), total_samples)
    full_stream[:copy_len] = user_audio[:copy_len]
    
    chunks: List[Tuple[str, np.ndarray]] = []
    for i in range(0, total_samples, chunk_samples):
        end = min(i + chunk_samples, total_samples)
        chunk = full_stream[i:end]
        start_s = i / SAMPLE_RATE
        end_s = end / SAMPLE_RATE
        has_audio = np.abs(chunk).max() > 0.001
        label = f"[{start_s:.0f}s-{end_s:.0f}s] {'SPEECH' if has_audio else 'SILENCE'}"
        chunks.append((label, chunk))
    
    return chunks


# ==================== SendLoop（prefill 循环） ====================

def send_loop(state: SharedState, audio_chunks: List[Tuple[str, np.ndarray]],
              image_a_b64: str, image_b_b64: str) -> None:
    """模拟 duplexSendLoop: 每 SEND_INTERVAL_S 发送 prefill(audio + image)

    在 IMAGE_SWITCH_TIME_S 时刻自动切换到第二张图片。
    """
    for idx, (label, chunk) in enumerate(audio_chunks):
        if not state.is_in_call:
            break
        
        # 等待发送间隔（第一个 chunk 也等，和前端一致）
        time.sleep(SEND_INTERVAL_S)
        if not state.is_in_call:
            break
        
        # 根据流已进行时间选择图片
        elapsed = time.time() - state.stream_start_time
        if elapsed < IMAGE_SWITCH_TIME_S:
            cur_image = image_a_b64
            img_tag = "IMG_A(夜晚房间)"
        else:
            cur_image = image_b_b64
            img_tag = "IMG_B(白天公园)"
        
        audio_b64 = audio_chunk_to_wav_base64(chunk)
        
        try:
            t0 = time.time()
            body = {"audio": audio_b64, "image": cur_image}
            resp = requests.post(
                f"{SERVER_URL}/omni/streaming_prefill",
                json=body,
                timeout=30,
            )
            resp.raise_for_status()
            prefill_ms = (time.time() - t0) * 1000
            
            with state.lock:
                state.prefill_count += 1
            
            state.log_event("PREFILL", f"#{idx} {label} +{img_tag} -> {prefill_ms:.0f}ms")
        except Exception as e:
            state.log_event("PREFILL_ERR", f"#{idx} {label}: {e}")
    
    state.log_event("SEND_DONE", f"发送完毕，共 {state.prefill_count} 次 prefill")


# ==================== ReceiveLoop（generate 循环） ====================

def receive_loop(state: SharedState) -> None:
    """模拟 duplexReceiveLoop: 不断调 generate

    注意：前端行为是 is_listen=True 时 continue，否则 break。
    但后端 end_of_turn 时不发 done 事件，导致 is_listen 默认 False。
    为了完整测量，这里 **始终 continue**，直到 is_in_call=False。
    """
    round_idx = 0
    
    while state.is_in_call:
        round_start = time.time()
        state.log_event("GEN_START", f"Round #{round_idx}")
        
        try:
            result = call_generate_sse(state, round_idx)
            
            with state.lock:
                state.generate_round_count += 1
            
            gen_ms = (time.time() - round_start) * 1000
            chunks_this_round = result["chunk_count"]
            audio_this_round = result["total_audio_s"]
            text_this_round = result["full_text"]
            
            detail = (f"Round #{round_idx}: {gen_ms:.0f}ms, "
                     f"{chunks_this_round} chunks, {audio_this_round:.2f}s audio, "
                     f"is_listen={result['is_listen']}")
            if text_this_round:
                detail += f" | \"{text_this_round[:50]}\""
            
            state.log_event("GEN_DONE", detail)
            
            # 始终继续，直到 is_in_call=False（由 send_loop 结束后设置）
            time.sleep(0.1)
            round_idx += 1
                
        except Exception as e:
            state.log_event("GEN_ERR", f"Round #{round_idx}: {e}")
            if state.is_in_call:
                time.sleep(1.0)
            round_idx += 1
    
    state.log_event("RECV_DONE", f"接收完毕，共 {state.generate_round_count} 轮 generate")


def call_generate_sse(state: SharedState, round_idx: int) -> Dict:
    """单次 generate 调用，解析 SSE"""
    gen_start = time.time()
    
    resp = requests.post(
        f"{SERVER_URL}/omni/streaming_generate",
        headers={"Accept": "text/event-stream"},
        stream=True,
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    
    chunks: List[Dict] = []
    full_text = ""
    is_listen = False
    first_chunk_time_ms: Optional[float] = None
    
    round_output_dir = os.path.join(OUTPUT_DIR, f"gen_round_{round_idx:03d}")
    ensure_dir(round_output_dir)
    
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
                try:
                    data = json.loads(line[6:])
                    
                    if "chunk_data" in data:
                        chunk_time = time.time()
                        chunk_ms = (chunk_time - gen_start) * 1000
                        if first_chunk_time_ms is None:
                            first_chunk_time_ms = chunk_ms
                        
                        cd = data["chunk_data"]
                        wav_b64 = cd.get("wav", "")
                        sr = cd.get("sample_rate", 24000)
                        text = cd.get("text", "")
                        chunk_idx = data.get("chunk_idx", len(chunks))
                        
                        audio_duration_s = 0.0
                        if wav_b64:
                            pcm_bytes = base64.b64decode(wav_b64)
                            num_samples = len(pcm_bytes) // 2
                            audio_duration_s = num_samples / sr
                            
                            wav_path = os.path.join(round_output_dir, f"chunk_{chunk_idx:03d}.wav")
                            decode_pcm_base64_to_wav(wav_b64, sr, wav_path)
                            
                            # 累积 PCM 用于最后合并
                            with state.lock:
                                state.all_pcm.append(pcm_bytes)
                                state.tts_sample_rate = sr
                        
                        if text:
                            full_text += text
                        
                        with state.lock:
                            state.total_audio_chunks_received += 1
                            state.total_audio_duration_s += audio_duration_s
                            if text:
                                state.all_texts.append(text)
                        
                        state.log_event("AUDIO_CHUNK", 
                            f"Gen#{round_idx} Chunk#{chunk_idx} "
                            f"T={chunk_ms:.0f}ms dur={audio_duration_s:.3f}s "
                            f"{text[:30] if text else ''}")
                        
                        chunks.append({
                            "chunk_idx": chunk_idx,
                            "time_ms": chunk_ms,
                            "audio_duration_s": audio_duration_s,
                            "text": text,
                        })
                    
                    if data.get("break"):
                        pass
                    
                    if "is_listen" in data:
                        is_listen = data["is_listen"]
                        if is_listen:
                            state.log_event("IS_LISTEN", f"Gen#{round_idx} is_listen=True")
                    
                except json.JSONDecodeError:
                    pass
    
    total_ms = (time.time() - gen_start) * 1000
    total_audio_s = sum(c["audio_duration_s"] for c in chunks)
    
    return {
        "first_chunk_ms": first_chunk_time_ms,
        "total_ms": total_ms,
        "total_audio_s": total_audio_s,
        "chunk_count": len(chunks),
        "full_text": full_text,
        "is_listen": is_listen,
        "chunks": chunks,
    }


# ==================== 主流程 ====================

def main() -> None:
    print("=" * 70)
    print("双工模式并发流测试 (正确模拟前端行为)")
    print(f"服务器: {SERVER_URL}")
    print(f"音频: {USER_AUDIO}")
    print(f"发送间隔: {SEND_INTERVAL_S}s, 总时长: {TOTAL_STREAM_DURATION_S}s")
    print("=" * 70)
    
    ensure_dir(OUTPUT_DIR)

    # 保存输入信息
    input_info = {
        "test": "test_duplex_stream",
        "description": "双工模式并发流测试（模拟 Video Call 前端行为：sendLoop + receiveLoop 并发）",
        "input_audio": USER_AUDIO,
        "input_audio_basename": os.path.basename(USER_AUDIO),
        "input_images": ["生成的测试图片: 夜晚房间 (0-10s)", "生成的测试图片: 白天公园 (10s+)"],
        "mode": "duplex",
        "send_interval_s": SEND_INTERVAL_S,
        "total_stream_duration_s": TOTAL_STREAM_DURATION_S,
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
    
    # 准备音频
    audio_chunks = prepare_audio_stream(USER_AUDIO)
    print(f"准备了 {len(audio_chunks)} 个音频 chunk (每 {SEND_INTERVAL_S}s):")
    for label, chunk in audio_chunks:
        rms = np.sqrt(np.mean(chunk ** 2))
        print(f"  {label}  RMS={rms:.6f}")
    
    # 初始化双工
    print(f"\n{'='*70}")
    print("初始化双工模式...")
    body = {"media_type": "omni", "duplex_mode": True, "language": "zh"}
    resp = requests.post(f"{SERVER_URL}/omni/init_sys_prompt", json=body, timeout=30)
    resp.raise_for_status()
    init_result = resp.json()
    print(f"  Init: {json.dumps(init_result, ensure_ascii=False)}")
    
    if not init_result.get("duplex_mode"):
        print("  duplex_mode 未启用！")
        sys.exit(1)
    
    # 生成两张模拟摄像头图片
    print("\n生成模拟摄像头画面 (640x480 JPEG)...")
    image_a_b64 = generate_test_image_night_room()
    image_b_b64 = generate_test_image_park()
    print(f"  图A (夜晚房间): {len(image_a_b64)} bytes")
    print(f"  图B (白天公园): {len(image_b_b64)} bytes")
    print(f"  切换时机: 流开始后 {IMAGE_SWITCH_TIME_S}s")
    
    # 保存图片供检查
    for tag, b64 in [("A_night_room", image_a_b64), ("B_day_park", image_b_b64)]:
        path = os.path.join(OUTPUT_DIR, f"test_image_{tag}.jpg")
        with open(path, "wb") as f:
            f.write(base64.b64decode(b64))
        print(f"  已保存: {path}")
    
    # 并发启动 sendLoop + receiveLoop
    state = SharedState()
    state.stream_start_time = time.time()
    
    print(f"\n{'='*70}")
    print("启动并发 sendLoop + receiveLoop (audio + image)")
    print(f"{'='*70}")
    
    send_thread = threading.Thread(target=send_loop, args=(state, audio_chunks, image_a_b64, image_b_b64), name="sendLoop")
    recv_thread = threading.Thread(target=receive_loop, args=(state,), name="recvLoop")
    
    send_thread.start()
    recv_thread.start()
    
    # 等待 sendLoop 完成
    send_thread.join()
    
    # sendLoop 结束后，再给 receiveLoop 15s 收集剩余音频
    print(f"\n[MAIN] sendLoop 结束，等待 receiveLoop 15s 收集剩余输出...")
    time.sleep(15)
    state.is_in_call = False
    
    recv_thread.join(timeout=10)
    if recv_thread.is_alive():
        print("\n[WARN] receiveLoop 超时，强制停止")
        recv_thread.join(timeout=5)
    
    # 停止
    state.is_in_call = False
    try:
        requests.post(f"{SERVER_URL}/omni/stop", timeout=5)
    except Exception:
        pass
    
    total_time = time.time() - state.stream_start_time
    
    # 总结
    print(f"\n\n{'='*70}")
    print("双工并发流测试总结")
    print(f"{'='*70}")
    print(f"  总运行时间: {total_time:.1f}s")
    print(f"  Prefill 次数: {state.prefill_count}")
    print(f"  Generate 轮数: {state.generate_round_count}")
    print(f"  收到音频 chunk 数: {state.total_audio_chunks_received}")
    print(f"  总生成音频时长: {state.total_audio_duration_s:.2f}s")
    
    if state.total_audio_duration_s > 0:
        rtf = total_time / state.total_audio_duration_s
        print(f"  整体 RTF: {rtf:.2f}x {'✅ 实时' if rtf < 1.0 else '⚠️ 慢于实时'}")
    
    if state.all_texts:
        full_text = "".join(state.all_texts)
        print(f"\n  模型输出文本: {full_text[:200]}")
    
    # 按类型统计事件
    prefill_events = [e for e in state.events if e["type"] == "PREFILL"]
    chunk_events = [e for e in state.events if e["type"] == "AUDIO_CHUNK"]
    
    if chunk_events:
        first_chunk = chunk_events[0]
        print(f"\n  首个音频 chunk 时间: 流开始后 {first_chunk['elapsed_s']}s")
        
        if len(chunk_events) > 1:
            intervals = []
            for i in range(1, len(chunk_events)):
                dt = chunk_events[i]["elapsed_s"] - chunk_events[i-1]["elapsed_s"]
                intervals.append(dt)
            avg_interval = sum(intervals) / len(intervals)
            print(f"  音频 chunk 间隔: avg={avg_interval:.1f}s, "
                  f"min={min(intervals):.1f}s, max={max(intervals):.1f}s")
    
    # 保存合并音频
    if state.all_pcm:
        merged_pcm = b"".join(state.all_pcm)
        merged_wav_path = os.path.join(OUTPUT_DIR, "tts_output.wav")
        sr = state.tts_sample_rate
        with wave.open(merged_wav_path, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(merged_pcm)
        audio_dur = len(merged_pcm) / (sr * 2)
        print(f"\n  合并音频: {merged_wav_path}")
        print(f"  时长: {audio_dur:.2f}s, 大小: {len(merged_pcm)//1024}KB")
        print(f"  (macOS 播放: open {merged_wav_path})")
    
    # 保存文本
    if state.all_texts:
        full_text = "".join(state.all_texts)
        text_path = os.path.join(OUTPUT_DIR, "response_text.txt")
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(full_text)
        print(f"  文本: {text_path}")

    # 保存事件日志
    log_path = os.path.join(OUTPUT_DIR, "events.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(state.events, f, ensure_ascii=False, indent=2)
    print(f"\n  事件日志: {log_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
