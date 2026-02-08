"""双工模式流水线时间轴分析

功能：
  1. 运行一次短时双工测试（15s 音频流 + 收尾等待）
  2. 收集客户端侧所有事件（带绝对时间戳）
  3. 从 server 终端日志中提取 C++ [TIMELINE] 和 Python [TIMELINE] 事件
  4. 合并所有事件，生成 HTML 泳道图可视化

使用方法：
  cd llama.cpp-omni && PYTHONPATH=. .venv/base/bin/python tools/omni/app/tests/test_duplex_timeline.py
"""
import os
import sys
import re
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
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "results/duplex_timeline")
# Cursor 终端文件路径 — server 的 stdout 被捕获在此
SERVER_TERMINAL_FILE = "/Users/sunweiyue/.cursor/projects/Users-sunweiyue-Desktop-lib-minicpm-o-4-5-macOS/terminals/901718.txt"

SEND_INTERVAL_S = 1.0   # 1s 周期：每秒发送 1s 音频 + 当前帧
SAMPLE_RATE = 16000
TOTAL_STREAM_DURATION_S = 15   # 15s 音频流
IMAGE_SWITCH_TIME_S = 8.0      # 8s 时切换图片
POST_SEND_WAIT_S = 15          # 发送完毕后等 15s 收集输出
TIMEOUT = 120


# ==================== 时间戳工具 ====================

def ts_now() -> str:
    """返回 HH:MM:SS.mmm 格式时间戳"""
    now = datetime.now()
    return now.strftime("%H:%M:%S.") + f"{now.microsecond // 1000:03d}"


# ==================== 事件收集 ====================

@dataclass
class TimelineEvent:
    """统一的时间轴事件"""
    timestamp: str       # HH:MM:SS.mmm
    source: str          # client / py_backend / cpp
    module: str          # vision_encoder / audio_encoder / llm / tts / t2w / prefill / generate
    event: str           # 事件名
    detail: str = ""     # 附加信息

    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp,
            "source": self.source,
            "module": self.module,
            "event": self.event,
            "detail": self.detail,
        }


class EventCollector:
    """线程安全的事件收集器"""
    def __init__(self) -> None:
        self._events: List[TimelineEvent] = []
        self._lock = threading.Lock()

    def add(self, source: str, module: str, event: str, detail: str = "") -> None:
        ts = ts_now()
        ev = TimelineEvent(timestamp=ts, source=source, module=module, event=event, detail=detail)
        with self._lock:
            self._events.append(ev)
        print(f"  [{ts}] [{source:<12}] [{module:<16}] {event} {detail}", flush=True)

    def get_all(self) -> List[TimelineEvent]:
        with self._lock:
            return list(self._events)


# ==================== 图片生成 ====================

def generate_test_image_night_room() -> str:
    """场景 A: 夜晚房间"""
    from PIL import Image, ImageDraw
    w, h = 640, 480
    img = Image.new("RGB", (w, h), (40, 45, 55))
    draw = ImageDraw.Draw(img)
    draw.rectangle([420, 30, 580, 200], fill=(20, 30, 60), outline=(120, 120, 120), width=3)
    draw.rectangle([50, 320, 400, 340], fill=(120, 80, 40))
    draw.rectangle([70, 340, 90, 450], fill=(100, 65, 30))
    draw.polygon([(140, 240), (240, 240), (220, 200), (160, 200)], fill=(255, 220, 100))
    draw.rectangle([260, 300, 340, 320], fill=(200, 50, 50))
    draw.rectangle([0, 450, w, h], fill=(80, 60, 40))
    import io
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=70)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def generate_test_image_park() -> str:
    """场景 B: 白天公园"""
    from PIL import Image, ImageDraw
    w, h = 640, 480
    img = Image.new("RGB", (w, h), (135, 200, 250))
    draw = ImageDraw.Draw(img)
    for cx, cy, rx, ry in [(150, 60, 60, 25), (450, 80, 55, 22)]:
        draw.ellipse([cx-rx, cy-ry, cx+rx, cy+ry], fill=(255, 255, 255))
    draw.rectangle([0, 270, w, h], fill=(80, 170, 60))
    draw.rectangle([80, 200, 105, 370], fill=(100, 70, 30))
    draw.ellipse([30, 100, 155, 220], fill=(50, 130, 40))
    draw.rectangle([380, 330, 460, 340], fill=(140, 90, 40))
    draw.ellipse([530, 20, 590, 80], fill=(255, 230, 80))
    import io
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=70)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ==================== 音频工具 ====================

def load_wav_pcm16(wav_path: str) -> np.ndarray:
    """读取 WAV 文件返回 float32 数组"""
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


def prepare_audio_stream(audio_path: str) -> List[Tuple[str, np.ndarray]]:
    """准备音频流：用户音频 + zero pad"""
    chunk_samples = int(SEND_INTERVAL_S * SAMPLE_RATE)
    total_samples = int(TOTAL_STREAM_DURATION_S * SAMPLE_RATE)
    user_audio = load_wav_pcm16(audio_path)
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
        label = f"[{start_s:.0f}-{end_s:.0f}s]{'SPEECH' if has_audio else 'SIL'}"
        chunks.append((label, chunk))
    return chunks


# ==================== Send Loop ====================

def send_loop(collector: EventCollector, audio_chunks: List[Tuple[str, np.ndarray]],
              image_a_b64: str, image_b_b64: str, is_active: threading.Event,
              start_idx: int = 0) -> None:
    """模拟 duplexSendLoop: 每 SEND_INTERVAL_S 发送 prefill(audio + image)
    
    Args:
        start_idx: 起始 chunk 编号（用于正确标记 #N），当 chunk #0 已同步发送时传 1
    """
    last_idx = start_idx
    # 计算图片切换的 chunk 边界：IMAGE_SWITCH_TIME_S / SEND_INTERVAL_S
    image_switch_chunk_idx = int(IMAGE_SWITCH_TIME_S / SEND_INTERVAL_S)
    for i, (label, chunk) in enumerate(audio_chunks):
        idx = start_idx + i
        last_idx = idx
        if not is_active.is_set():
            break
        time.sleep(SEND_INTERVAL_S)
        if not is_active.is_set():
            break

        # 基于全局 chunk 序号决定图片（而非 wall clock，避免 prefill 延迟干扰）
        cur_image = image_a_b64 if idx < image_switch_chunk_idx else image_b_b64
        img_tag = "IMG_A" if idx < image_switch_chunk_idx else "IMG_B"
        audio_b64 = audio_chunk_to_wav_base64(chunk)

        try:
            collector.add("client", "prefill", "send_start", f"#{idx} {label} {img_tag}")
            t0 = time.time()
            body = {"audio": audio_b64, "image": cur_image}
            resp = requests.post(f"{SERVER_URL}/omni/streaming_prefill", json=body, timeout=30)
            resp.raise_for_status()
            ms = (time.time() - t0) * 1000
            collector.add("client", "prefill", "send_done", f"#{idx} {ms:.0f}ms")
        except Exception as e:
            collector.add("client", "prefill", "send_error", f"#{idx} {e}")

    collector.add("client", "prefill", "loop_done", f"total={last_idx+1}")


# ==================== Receive Loop ====================

def receive_loop(collector: EventCollector, is_active: threading.Event,
                 audio_pcm_list: Optional[List[bytes]] = None) -> None:
    """模拟 duplexReceiveLoop: 不断调 generate

    Args:
        collector: 事件收集器
        is_active: 控制循环的 Event
        audio_pcm_list: 如果不为 None，收集所有 PCM 音频数据到此列表
    """
    round_idx = 0
    while is_active.is_set():
        try:
            collector.add("client", "generate", "call_start", f"round={round_idx}")
            t0 = time.time()
            resp = requests.post(
                f"{SERVER_URL}/omni/streaming_generate",
                headers={"Accept": "text/event-stream"},
                stream=True, timeout=TIMEOUT,
            )
            resp.raise_for_status()

            chunk_count = 0
            total_audio_s = 0.0
            full_text = ""
            is_listen = False
            buffer = ""

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
                                sr = cd.get("sample_rate", 24000)
                                text = cd.get("text", "")
                                if wav_b64:
                                    pcm_bytes = base64.b64decode(wav_b64)
                                    dur = len(pcm_bytes) // 2 / sr
                                    total_audio_s += dur
                                    chunk_count += 1
                                    if audio_pcm_list is not None:
                                        audio_pcm_list.append(pcm_bytes)
                                    collector.add("client", "generate", "audio_chunk",
                                        f"round={round_idx} chunk={chunk_count} dur={dur:.3f}s")
                                if text:
                                    full_text += text
                            if "is_listen" in data:
                                is_listen = data["is_listen"]
                                if is_listen:
                                    collector.add("client", "generate", "is_listen", f"round={round_idx}")
                        except json.JSONDecodeError:
                            pass

            ms = (time.time() - t0) * 1000
            collector.add("client", "generate", "call_done",
                f"round={round_idx} {ms:.0f}ms chunks={chunk_count} audio={total_audio_s:.2f}s listen={is_listen}")
            if full_text:
                collector.add("client", "generate", "text", f"round={round_idx} \"{full_text[:60]}\"")

            time.sleep(0.1)
            round_idx += 1

        except Exception as e:
            collector.add("client", "generate", "error", f"round={round_idx} {e}")
            if is_active.is_set():
                time.sleep(1.0)
            round_idx += 1

    collector.add("client", "generate", "loop_done", f"total_rounds={round_idx}")


# ==================== Server Log Parser ====================

def parse_server_log(log_lines: List[str]) -> List[TimelineEvent]:
    """从 server 终端日志中提取 [TIMELINE] 事件

    C++ 格式:  [CPP] HH:MM:SS.mmm [TIMELINE] event_name detail
    Python 格式: [TIMELINE] HH:MM:SS.mmm event_name detail
    """
    events: List[TimelineEvent] = []

    # C++ [TIMELINE] events: [CPP] HH:MM:SS.mmm [TIMELINE] event detail
    cpp_timeline_re = re.compile(
        r'\[CPP\]\s+(\d{2}:\d{2}:\d{2}\.\d{3})\s+\[TIMELINE\]\s+(\S+)\s*(.*)'
    )
    # Python [TIMELINE] events: [TIMELINE] HH:MM:SS.mmm event detail
    py_timeline_re = re.compile(
        r'\[TIMELINE\]\s+(\d{2}:\d{2}:\d{2}\.\d{3})\s+(\S+)\s*(.*)'
    )
    # C++ existing timestamped events (not [TIMELINE] tagged)
    cpp_ts_re = re.compile(
        r'\[CPP\]\s+(\d{2}:\d{2}:\d{2}\.\d{3})\s+(.*)'
    )

    # Module mapping for C++ events
    cpp_module_map: Dict[str, str] = {
        "vision_encode_start": "vision_encoder",
        "vision_encode_done": "vision_encoder",
        "vision_prefill_start": "vision_encoder",
        "vision_prefill_done": "vision_encoder",
        "audio_encode_start": "audio_encoder",
        "audio_encode_done": "audio_encoder",
        "audio_encode_failed": "audio_encoder",
        "llm_queue_push": "prefill",
    }

    # Existing C++ event patterns to capture
    cpp_event_patterns: List[Tuple[str, str, re.Pattern]] = [
        ("llm", "prefill_start", re.compile(r'LLM thread: start prefill')),
        ("llm", "prefill_done", re.compile(r'LLM thread: prefill done')),
        ("llm", "batch_process", re.compile(r'Batch processing (\d+) llm prefill')),
        ("llm", "decode_start", re.compile(r'stream_decode 开始')),
        ("llm", "decode_config", re.compile(r'LLM decode: max_tgt_len')),
        ("llm", "to_tts", re.compile(r"LLM->TTS: text='([^']*)'.*n_tokens=(\d+)")),
        ("llm", "end_token", re.compile(r'LLM: detected end token')),
        ("tts", "thread_start", re.compile(r'TTS thread \(duplex mode\) started')),
        ("tts", "chunk_process", re.compile(r'TTS Duplex: processing chunk_idx=(\d+)')),
        ("tts", "queue_info", re.compile(r'TTS Duplex: after queue.*speek_done=(\d+).*llm_finish=(\d+)')),
        ("t2w", "thread_start", re.compile(r'T2W thread \(Python\) started|T2W thread \(C\+\+\) started')),
        ("t2w", "wav_output", re.compile(r'T2W(?:\(Python\)|线程(?:\(C\+\+\))?): (wav_\d+\.wav) \| ([\d.]+)s audio \| ([\d.]+)ms inference \| RTF=([\d.]+)')),
        ("t2w", "first_audio", re.compile(r'🎉 首响时间.*?(\d+)ms')),
        ("prefill", "finish", re.compile(r'c\+\+ finish stream_prefill\(index=(\d+)\)')),
        ("prefill", "sys_init", re.compile(r'stream_prefill: n_past')),
        ("llm", "speak_token", re.compile(r'duplex.*<\|speak\|>')),
        ("llm", "listen_token", re.compile(r'duplex.*<\|listen\|>')),
    ]

    for line in log_lines:
        line = line.rstrip()

        # 1. C++ [TIMELINE] events
        m = cpp_timeline_re.search(line)
        if m:
            ts, event_name, detail = m.group(1), m.group(2), m.group(3).strip()
            module = cpp_module_map.get(event_name, "prefill")
            events.append(TimelineEvent(
                timestamp=ts, source="cpp", module=module, event=event_name, detail=detail
            ))
            continue

        # 2. Python [TIMELINE] events
        m = py_timeline_re.search(line)
        if m:
            ts, event_name, detail = m.group(1), m.group(2), m.group(3).strip()
            module = "py_backend"
            if "prefill" in event_name:
                module = "prefill"
            elif "generate" in event_name or "decode" in event_name:
                module = "generate"
            elif "wav" in event_name:
                module = "wav_delivery"
            elif "listen" in event_name or "end_of_turn" in event_name:
                module = "generate"
            events.append(TimelineEvent(
                timestamp=ts, source="py_backend", module=module, event=event_name, detail=detail
            ))
            continue

        # 3. Existing C++ timestamped events
        m = cpp_ts_re.search(line)
        if m:
            ts, msg = m.group(1), m.group(2).strip()
            for module, event_name, pattern in cpp_event_patterns:
                pm = pattern.search(msg)
                if pm:
                    detail_str = pm.group(0) if pm.lastindex is None else " ".join(pm.groups())
                    events.append(TimelineEvent(
                        timestamp=ts, source="cpp", module=module, event=event_name, detail=detail_str
                    ))
                    break

    return events


# ==================== HTML 可视化生成 ====================

def generate_timeline_html(all_events: List[Dict], output_path: str) -> None:
    """生成 HTML 泳道图时间轴"""
    html = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<title>双工模式流水线时间轴</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: 'SF Mono', 'Menlo', monospace; background: #1a1a2e; color: #eee; padding: 20px; }
h1 { text-align: center; margin-bottom: 20px; color: #00d4ff; font-size: 18px; }
.controls { text-align: center; margin-bottom: 15px; }
.controls button { background: #16213e; border: 1px solid #0f3460; color: #eee; padding: 6px 14px;
  margin: 0 5px; cursor: pointer; border-radius: 4px; font-size: 12px; }
.controls button:hover { background: #0f3460; }
.controls button.active { background: #e94560; border-color: #e94560; }
.timeline-container { overflow-x: auto; position: relative; }
.timeline { position: relative; min-height: 600px; }

/* 泳道 */
.lane { position: relative; height: 48px; margin-bottom: 2px; display: flex; align-items: center; }
.lane-label { width: 160px; min-width: 160px; font-size: 11px; font-weight: bold; padding-right: 10px;
  text-align: right; color: #aaa; }
.lane-track { position: relative; flex: 1; height: 100%; background: #16213e; border-radius: 3px; overflow: visible; }

/* 事件标记 */
.event { position: absolute; height: 28px; top: 10px; border-radius: 3px; font-size: 9px;
  display: flex; align-items: center; padding: 0 4px; cursor: pointer; white-space: nowrap;
  transition: opacity 0.2s; z-index: 2; }
.event:hover { z-index: 10; opacity: 1 !important; box-shadow: 0 0 8px rgba(255,255,255,0.3); }
.event.point { width: 8px; min-width: 8px; padding: 0; }
.event.span { min-width: 4px; }

/* 模块颜色 */
.event.vision_encoder { background: #e94560; }
.event.audio_encoder { background: #ff9800; }
.event.llm { background: #4caf50; }
.event.tts { background: #2196f3; }
.event.t2w { background: #9c27b0; }
.event.prefill { background: #00bcd4; }
.event.generate { background: #ff5722; }
.event.wav_delivery { background: #ffeb3b; color: #333; }
.event.client_prefill { background: #607d8b; }
.event.client_generate { background: #795548; }

/* Tooltip */
.tooltip { display: none; position: fixed; background: #0f3460; border: 1px solid #00d4ff;
  border-radius: 6px; padding: 10px; font-size: 11px; max-width: 400px; z-index: 100;
  pointer-events: none; box-shadow: 0 4px 12px rgba(0,0,0,0.5); }
.tooltip .ts { color: #00d4ff; font-weight: bold; }
.tooltip .src { color: #e94560; }
.tooltip .mod { color: #4caf50; }
.tooltip .ev { color: #ffeb3b; }
.tooltip .det { color: #aaa; margin-top: 4px; }

/* 时间轴刻度 */
.time-axis { display: flex; margin-left: 160px; height: 30px; align-items: flex-end; position: relative; }
.tick { position: absolute; font-size: 9px; color: #666; border-left: 1px solid #333;
  padding-left: 3px; height: 20px; display: flex; align-items: flex-end; }

/* 图例 */
.legend { display: flex; flex-wrap: wrap; justify-content: center; margin: 15px 0; gap: 10px; }
.legend-item { display: flex; align-items: center; gap: 4px; font-size: 11px; }
.legend-color { width: 14px; height: 14px; border-radius: 3px; }

/* 统计面板 */
.stats { background: #16213e; border-radius: 6px; padding: 15px; margin-top: 15px; }
.stats h3 { color: #00d4ff; margin-bottom: 10px; font-size: 14px; }
.stats table { width: 100%; border-collapse: collapse; font-size: 11px; }
.stats td, .stats th { padding: 4px 8px; border-bottom: 1px solid #333; text-align: left; }
.stats th { color: #aaa; }
</style>
</head>
<body>
<h1>MiniCPM-o 双工模式 流水线时间轴分析</h1>

<div class="legend">
  <div class="legend-item"><div class="legend-color" style="background:#e94560"></div>Vision Encoder</div>
  <div class="legend-item"><div class="legend-color" style="background:#ff9800"></div>Audio Encoder</div>
  <div class="legend-item"><div class="legend-color" style="background:#4caf50"></div>LLM</div>
  <div class="legend-item"><div class="legend-color" style="background:#2196f3"></div>TTS</div>
  <div class="legend-item"><div class="legend-color" style="background:#9c27b0"></div>Token2Wav</div>
  <div class="legend-item"><div class="legend-color" style="background:#00bcd4"></div>C++ Prefill</div>
  <div class="legend-item"><div class="legend-color" style="background:#ff5722"></div>Generate/Decode</div>
  <div class="legend-item"><div class="legend-color" style="background:#ffeb3b"></div>WAV Delivery</div>
  <div class="legend-item"><div class="legend-color" style="background:#607d8b"></div>Client Prefill</div>
  <div class="legend-item"><div class="legend-color" style="background:#795548"></div>Client Generate</div>
</div>

<div class="controls">
  <button onclick="zoomIn()">放大 +</button>
  <button onclick="zoomOut()">缩小 -</button>
  <button onclick="resetZoom()">重置</button>
  <span style="margin-left:20px; font-size:12px; color:#aaa;">
    总事件数: <span id="eventCount">0</span> | 时间范围: <span id="timeRange">-</span>
  </span>
</div>

<div class="time-axis" id="timeAxis"></div>
<div class="timeline-container">
  <div class="timeline" id="timeline"></div>
</div>
<div class="tooltip" id="tooltip"></div>

<div class="stats" id="statsPanel"></div>

<script>
const EVENTS = __EVENTS_JSON__;

// 泳道定义
const LANES = [
  { id: 'client_prefill', label: 'Client Prefill', filter: e => e.source === 'client' && e.module === 'prefill' },
  { id: 'client_generate', label: 'Client Generate', filter: e => e.source === 'client' && e.module === 'generate' },
  { id: 'prefill', label: 'C++ Prefill', filter: e => e.module === 'prefill' && e.source !== 'client' },
  { id: 'vision_encoder', label: 'Vision Encoder', filter: e => e.module === 'vision_encoder' },
  { id: 'audio_encoder', label: 'Audio Encoder', filter: e => e.module === 'audio_encoder' },
  { id: 'llm', label: 'LLM (Text)', filter: e => e.module === 'llm' },
  { id: 'tts', label: 'TTS (Duplex)', filter: e => e.module === 'tts' },
  { id: 't2w', label: 'Token2Wav', filter: e => e.module === 't2w' },
  { id: 'generate', label: 'Py Generate', filter: e => (e.module === 'generate' || e.module === 'wav_delivery') && e.source !== 'client' },
];

// 解析 HH:MM:SS.mmm 为毫秒偏移
function tsToMs(ts) {
  const m = ts.match(/(\\d{2}):(\\d{2}):(\\d{2})\\.(\\d{3})/);
  if (!m) return 0;
  return parseInt(m[1]) * 3600000 + parseInt(m[2]) * 60000 + parseInt(m[3]) * 1000 + parseInt(m[4]);
}

// 找出时间范围
let minMs = Infinity, maxMs = -Infinity;
EVENTS.forEach(e => {
  const ms = tsToMs(e.timestamp);
  if (ms < minMs) minMs = ms;
  if (ms > maxMs) maxMs = ms;
});
const durationMs = maxMs - minMs || 1;

document.getElementById('eventCount').textContent = EVENTS.length;
document.getElementById('timeRange').textContent =
  `${(durationMs / 1000).toFixed(1)}s (${EVENTS[0]?.timestamp || '-'} ~ ${EVENTS[EVENTS.length-1]?.timestamp || '-'})`;

let pixelsPerMs = 0.15; // 初始缩放
const TRACK_PADDING_LEFT = 0;

function zoomIn() { pixelsPerMs *= 1.5; render(); }
function zoomOut() { pixelsPerMs /= 1.5; render(); }
function resetZoom() { pixelsPerMs = 0.15; render(); }

// 模块到 CSS class 映射
function getEventClass(e) {
  if (e.source === 'client' && e.module === 'prefill') return 'client_prefill';
  if (e.source === 'client' && e.module === 'generate') return 'client_generate';
  return e.module;
}

// 渲染泳道图
function render() {
  const timeline = document.getElementById('timeline');
  const timeAxis = document.getElementById('timeAxis');
  timeline.innerHTML = '';

  const totalWidth = durationMs * pixelsPerMs + 200;

  // 时间轴刻度
  timeAxis.innerHTML = '';
  timeAxis.style.width = (totalWidth + 160) + 'px';
  const tickInterval = Math.max(500, Math.round(5000 / pixelsPerMs / 5) * 500); // 动态调整
  for (let ms = 0; ms <= durationMs + tickInterval; ms += tickInterval) {
    const tick = document.createElement('div');
    tick.className = 'tick';
    tick.style.left = (ms * pixelsPerMs) + 'px';
    const sec = ms / 1000;
    tick.textContent = sec >= 60 ? `${Math.floor(sec/60)}m${(sec%60).toFixed(1)}s` : `${sec.toFixed(1)}s`;
    timeAxis.appendChild(tick);
  }

  // 渲染各泳道
  LANES.forEach(lane => {
    const laneDiv = document.createElement('div');
    laneDiv.className = 'lane';

    const label = document.createElement('div');
    label.className = 'lane-label';
    label.textContent = lane.label;
    laneDiv.appendChild(label);

    const track = document.createElement('div');
    track.className = 'lane-track';
    track.style.width = totalWidth + 'px';

    const laneEvents = EVENTS.filter(lane.filter);
    laneEvents.forEach(e => {
      const ms = tsToMs(e.timestamp) - minMs;
      const left = ms * pixelsPerMs;

      const div = document.createElement('div');
      div.className = `event point ${getEventClass(e)}`;
      div.style.left = left + 'px';

      // 短标签
      let shortLabel = e.event;
      if (shortLabel.length > 12) shortLabel = shortLabel.substring(0, 12);
      div.textContent = shortLabel;
      div.style.minWidth = Math.max(8, shortLabel.length * 5.5) + 'px';
      div.style.paddingLeft = '2px';
      div.style.paddingRight = '2px';

      // Tooltip
      div.addEventListener('mouseenter', (evt) => {
        const tt = document.getElementById('tooltip');
        tt.style.display = 'block';
        tt.innerHTML = `
          <div><span class="ts">${e.timestamp}</span> (+${(ms/1000).toFixed(3)}s)</div>
          <div><span class="src">${e.source}</span> / <span class="mod">${e.module}</span></div>
          <div><span class="ev">${e.event}</span></div>
          ${e.detail ? '<div class="det">' + e.detail + '</div>' : ''}
        `;
        tt.style.left = Math.min(evt.clientX + 10, window.innerWidth - 420) + 'px';
        tt.style.top = (evt.clientY - 80) + 'px';
      });
      div.addEventListener('mouseleave', () => {
        document.getElementById('tooltip').style.display = 'none';
      });

      track.appendChild(div);
    });

    laneDiv.appendChild(track);
    timeline.appendChild(laneDiv);
  });

  renderStats();
}

function renderStats() {
  const panel = document.getElementById('statsPanel');
  // 计算一些关键指标
  const firstPrefillSend = EVENTS.find(e => e.source === 'client' && e.event === 'send_start');
  const firstAudioChunk = EVENTS.find(e => e.source === 'client' && e.event === 'audio_chunk');
  const firstLlmToTts = EVENTS.find(e => e.event === 'to_tts');
  const firstWavOutput = EVENTS.find(e => e.event === 'wav_output');
  const firstAudioResponse = EVENTS.find(e => e.event === 'first_audio');
  const visionEvents = EVENTS.filter(e => e.module === 'vision_encoder');
  const audioEncEvents = EVENTS.filter(e => e.module === 'audio_encoder');
  const llmToTtsEvents = EVENTS.filter(e => e.event === 'to_tts');
  const wavOutputEvents = EVENTS.filter(e => e.event === 'wav_output');
  const prefillFinishEvents = EVENTS.filter(e => e.event === 'finish' && e.module === 'prefill');

  let html = '<h3>关键指标</h3><table>';
  html += '<tr><th>指标</th><th>值</th></tr>';

  if (firstPrefillSend && firstAudioChunk) {
    const e2e = (tsToMs(firstAudioChunk.timestamp) - tsToMs(firstPrefillSend.timestamp)) / 1000;
    html += `<tr><td>端到端首音延迟 (Client)</td><td>${e2e.toFixed(3)}s</td></tr>`;
  }
  if (firstAudioResponse) {
    html += `<tr><td>C++ 首响时间</td><td>${firstAudioResponse.detail}</td></tr>`;
  }
  if (firstLlmToTts && firstPrefillSend) {
    const llmLatency = (tsToMs(firstLlmToTts.timestamp) - tsToMs(firstPrefillSend.timestamp)) / 1000;
    html += `<tr><td>首次 LLM→TTS 延迟</td><td>${llmLatency.toFixed(3)}s</td></tr>`;
  }

  // Vision encode durations
  const vStarts = visionEvents.filter(e => e.event.includes('start'));
  const vDones = visionEvents.filter(e => e.event.includes('done') && !e.event.includes('prefill'));
  if (vStarts.length > 0 && vDones.length > 0) {
    const durations = [];
    for (let i = 0; i < Math.min(vStarts.length, vDones.length); i++) {
      durations.push(tsToMs(vDones[i].timestamp) - tsToMs(vStarts[i].timestamp));
    }
    const avg = durations.reduce((a,b)=>a+b,0) / durations.length;
    html += `<tr><td>Vision Encode 平均耗时</td><td>${avg.toFixed(0)}ms (${durations.length}次)</td></tr>`;
  }

  // Audio encode durations
  const aStarts = audioEncEvents.filter(e => e.event.includes('start'));
  const aDones = audioEncEvents.filter(e => e.event.includes('done'));
  if (aStarts.length > 0 && aDones.length > 0) {
    const durations = [];
    for (let i = 0; i < Math.min(aStarts.length, aDones.length); i++) {
      durations.push(tsToMs(aDones[i].timestamp) - tsToMs(aStarts[i].timestamp));
    }
    const avg = durations.reduce((a,b)=>a+b,0) / durations.length;
    html += `<tr><td>Audio Encode 平均耗时</td><td>${avg.toFixed(0)}ms (${durations.length}次)</td></tr>`;
  }

  html += `<tr><td>LLM→TTS 事件数</td><td>${llmToTtsEvents.length}</td></tr>`;
  html += `<tr><td>WAV 输出数</td><td>${wavOutputEvents.length}</td></tr>`;
  html += `<tr><td>C++ Prefill 完成数</td><td>${prefillFinishEvents.length}</td></tr>`;
  html += `<tr><td>总事件数</td><td>${EVENTS.length}</td></tr>`;
  html += `<tr><td>时间跨度</td><td>${(durationMs/1000).toFixed(1)}s</td></tr>`;

  html += '</table>';
  panel.innerHTML = html;
}

render();
</script>
</body>
</html>"""
    # 注入事件数据
    events_json = json.dumps(all_events, ensure_ascii=False, indent=None)
    html = html.replace("__EVENTS_JSON__", events_json)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\n  HTML 时间轴已生成: {output_path}")


# ==================== 主流程 ====================

def main() -> None:
    print("=" * 70)
    print("双工模式流水线时间轴分析")
    print(f"服务器: {SERVER_URL}")
    print(f"音频流: {TOTAL_STREAM_DURATION_S}s, 图片切换: {IMAGE_SWITCH_TIME_S}s")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 保存输入信息
    input_info = {
        "test": "test_duplex_timeline",
        "description": "双工模式流水线时间轴分析（生成交互式 HTML 时间轴）",
        "input_audio": USER_AUDIO,
        "input_audio_basename": os.path.basename(USER_AUDIO),
        "input_images": ["生成的测试图片: 夜晚房间 (0-8s)", "生成的测试图片: 白天公园 (8s+)"],
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

    # 记录 server 终端日志起始行数
    server_log_start_line = 0
    if os.path.exists(SERVER_TERMINAL_FILE):
        with open(SERVER_TERMINAL_FILE, "r", encoding="utf-8", errors="ignore") as f:
            server_log_start_line = sum(1 for _ in f)
        print(f"\n[LOG] Server 终端日志起始行: {server_log_start_line}")
    else:
        print(f"\n[WARN] Server 终端文件不存在: {SERVER_TERMINAL_FILE}")

    # 准备音频
    audio_chunks = prepare_audio_stream(USER_AUDIO)
    print(f"准备了 {len(audio_chunks)} 个音频 chunk")

    # 初始化双工
    print(f"\n{'='*70}")
    print("初始化双工模式...")
    body = {"media_type": "omni", "duplex_mode": True, "language": "zh"}
    resp = requests.post(f"{SERVER_URL}/omni/init_sys_prompt", json=body, timeout=60)
    resp.raise_for_status()
    init_result = resp.json()
    print(f"  Init: {json.dumps(init_result, ensure_ascii=False)}")
    if not init_result.get("duplex_mode"):
        print("  duplex_mode 未启用！")
        sys.exit(1)

    # 等一下让 C++ 线程就绪
    time.sleep(1.0)

    # 生成模拟图片
    print("\n生成模拟摄像头画面...")
    image_a_b64 = generate_test_image_night_room()
    image_b_b64 = generate_test_image_park()
    print(f"  图A: {len(image_a_b64)} bytes, 图B: {len(image_b_b64)} bytes")

    # 启动事件收集
    collector = EventCollector()
    is_active = threading.Event()
    is_active.set()

    collector.add("client", "prefill", "test_start", f"duration={TOTAL_STREAM_DURATION_S}s")

    # ========== 同步发送第一个 prefill（确保模型有输入后再 generate）==========
    print(f"\n{'='*70}")
    print("同步发送首个 prefill (chunk #0)...")
    print(f"{'='*70}")
    first_label, first_chunk = audio_chunks[0]
    first_audio_b64 = audio_chunk_to_wav_base64(first_chunk)
    collector.add("client", "prefill", "send_start", f"#0 {first_label} IMG_A")
    t0 = time.time()
    body = {"audio": first_audio_b64, "image": image_a_b64}
    resp = requests.post(f"{SERVER_URL}/omni/streaming_prefill", json=body, timeout=30)
    resp.raise_for_status()
    ms = (time.time() - t0) * 1000
    collector.add("client", "prefill", "send_done", f"#0 {ms:.0f}ms")
    print(f"  首个 prefill 完成: {ms:.0f}ms")

    # ========== 首个 prefill 完成后，启动并发 sendLoop + receiveLoop ==========
    print(f"\n{'='*70}")
    print("启动并发 sendLoop + receiveLoop (从 chunk #1 开始)")
    print(f"{'='*70}")

    # send_loop 从 chunk #1 开始（chunk #0 已同步发送）
    remaining_chunks = audio_chunks[1:]
    send_thread = threading.Thread(target=send_loop,
        args=(collector, remaining_chunks, image_a_b64, image_b_b64, is_active),
        kwargs={"start_idx": 1}, name="sendLoop")
    audio_pcm_list: List[bytes] = []
    recv_thread = threading.Thread(target=receive_loop,
        args=(collector, is_active, audio_pcm_list), name="recvLoop")

    send_thread.start()
    recv_thread.start()

    # 等待 sendLoop 完成
    send_thread.join()
    print(f"\n[MAIN] sendLoop 结束，等待 receiveLoop {POST_SEND_WAIT_S}s 收集剩余输出...")
    time.sleep(POST_SEND_WAIT_S)
    is_active.clear()

    recv_thread.join(timeout=10)
    collector.add("client", "generate", "test_done", "")

    # 保存合并音频
    if audio_pcm_list:
        merged_pcm = b"".join(audio_pcm_list)
        audio_out_path = os.path.join(OUTPUT_DIR, "tts_output.wav")
        tts_sample_rate = 24000  # TTS 输出固定 24kHz（区别于输入的 16kHz SAMPLE_RATE）
        with wave.open(audio_out_path, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(tts_sample_rate)
            wf.writeframes(merged_pcm)
        audio_dur = len(merged_pcm) / (tts_sample_rate * 2)
        print(f"\n[AUDIO] TTS 音频已保存: {audio_out_path}")
        print(f"  时长: {audio_dur:.2f}s, 大小: {len(merged_pcm)//1024}KB, chunks: {len(audio_pcm_list)}")
    else:
        print("\n[AUDIO] 未收到任何 TTS 音频")

    # stop
    try:
        requests.post(f"{SERVER_URL}/omni/stop", timeout=5)
    except Exception:
        pass

    # 收集客户端事件
    client_events = [e.to_dict() for e in collector.get_all()]
    print(f"\n[收集] 客户端事件: {len(client_events)}")

    # 保存客户端事件
    client_events_path = os.path.join(OUTPUT_DIR, "client_events.json")
    with open(client_events_path, "w", encoding="utf-8") as f:
        json.dump(client_events, f, ensure_ascii=False, indent=2)

    # 读取 server 终端日志的新增部分
    server_events: List[Dict] = []
    if os.path.exists(SERVER_TERMINAL_FILE):
        print(f"[LOG] 读取 server 终端日志...")
        with open(SERVER_TERMINAL_FILE, "r", encoding="utf-8", errors="ignore") as f:
            all_lines = f.readlines()
        new_lines = all_lines[server_log_start_line:]
        print(f"  新增行数: {len(new_lines)}")

        # 保存原始 server log 供调试
        raw_log_path = os.path.join(OUTPUT_DIR, "server_log_raw.txt")
        with open(raw_log_path, "w", encoding="utf-8") as f:
            f.writelines(new_lines)
        print(f"  原始 log: {raw_log_path}")

        parsed_events = parse_server_log(new_lines)
        server_events = [e.to_dict() for e in parsed_events]
        print(f"  解析出 server 事件: {len(server_events)}")

        # 保存 server 事件
        server_events_path = os.path.join(OUTPUT_DIR, "server_events.json")
        with open(server_events_path, "w", encoding="utf-8") as f:
            json.dump(server_events, f, ensure_ascii=False, indent=2)

    # 合并所有事件，按时间排序
    all_events = client_events + server_events
    all_events.sort(key=lambda e: e["timestamp"])
    print(f"\n[合并] 总事件数: {len(all_events)}")

    # 保存合并事件
    merged_path = os.path.join(OUTPUT_DIR, "all_events.json")
    with open(merged_path, "w", encoding="utf-8") as f:
        json.dump(all_events, f, ensure_ascii=False, indent=2)

    # 生成 HTML
    html_path = os.path.join(OUTPUT_DIR, "timeline.html")
    generate_timeline_html(all_events, html_path)

    print(f"\n{'='*70}")
    print("完成！")
    print(f"  HTML 时间轴: {html_path}")
    print(f"  客户端事件: {client_events_path}")
    print(f"  Server 事件: {len(server_events)}")
    print(f"  总事件: {len(all_events)}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
