"""从 all_events.json 生成高质量 HTML 泳道图

用法:
  cd llama.cpp-omni && PYTHONPATH=. .venv/base/bin/python tools/omni/app/tests/generate_timeline_html.py
"""
import json
import re
import os
from typing import List, Dict, Optional, Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "results/duplex_timeline")
# 只显示到最后一个有意义事件后 2s，去除尾部空转
ACTIVE_CUTOFF_MARGIN_S = 2.0


# ==================== 时间工具 ====================

def ts_to_ms(ts: str) -> int:
    """HH:MM:SS.mmm → 毫秒"""
    m = re.match(r"(\d{2}):(\d{2}):(\d{2})\.(\d{3})", ts)
    if not m:
        return 0
    return int(m[1]) * 3600000 + int(m[2]) * 60000 + int(m[3]) * 1000 + int(m[4])


# ==================== 事件预处理 ====================

def preprocess_events(raw_events: List[Dict]) -> Tuple[List[Dict], int]:
    """预处理：配对 span、提取数据内容、计算相对时间

    Returns:
        (processed_items, t0_ms)  每个 item 是 point 或 span
    """
    if not raw_events:
        return [], 0

    # 找 client test_start 事件作为基准时间（而非第一个事件，避免被旧 server 事件污染）
    test_start_events = [e for e in raw_events if e.get("source") == "client" and e.get("event") == "test_start"]
    if test_start_events:
        t0 = ts_to_ms(test_start_events[0]["timestamp"])
        # 过滤掉早于 test_start 5s 以上的 server 事件（旧测试残留）
        t0_threshold = t0 - 5000  # 允许 5s 的偏差
        before_count = len(raw_events)
        raw_events = [e for e in raw_events if ts_to_ms(e["timestamp"]) >= t0_threshold]
        filtered_count = before_count - len(raw_events)
        if filtered_count > 0:
            print(f"  过滤掉 {filtered_count} 个旧事件 (早于 test_start 5s)")
    else:
        t0 = ts_to_ms(raw_events[0]["timestamp"])
    items: List[Dict] = []

    # ---- 配对 vision/audio encode spans ----
    pending_starts: Dict[str, List[Dict]] = {}  # key → [events]

    span_pairs = [
        ("vision_encode_start", "vision_encode_done", "vision_encoder", "Vision Encode"),
        ("audio_encode_start", "audio_encode_done", "audio_encoder", "Audio Encode"),
    ]
    # 收集用于配对的事件
    pair_start_names = {p[0] for p in span_pairs}
    pair_end_names = {p[1] for p in span_pairs}
    used_indices: set = set()

    for sp_start, sp_end, sp_module, sp_label in span_pairs:
        starts = [(i, e) for i, e in enumerate(raw_events) if e["event"] == sp_start]
        ends = [(i, e) for i, e in enumerate(raw_events) if e["event"] == sp_end]
        # 基于时间顺序配对：每个 start 匹配其后最近的 end（避免 zip 错位）
        used_end_indices: set = set()
        for (si, se) in starts:
            t_start_abs = ts_to_ms(se["timestamp"])
            # 找到第一个在 start 之后且未被使用的 end
            best_ei, best_ee = -1, None
            for (ei, ee) in ends:
                if ei in used_end_indices:
                    continue
                t_end_abs = ts_to_ms(ee["timestamp"])
                if t_end_abs >= t_start_abs:
                    best_ei, best_ee = ei, ee
                    break
            if best_ee is None:
                continue  # 没有匹配的 end，跳过
            used_end_indices.add(best_ei)
            t_start = t_start_abs - t0
            t_end = ts_to_ms(best_ee["timestamp"]) - t0
            dur = t_end - t_start
            if dur < 0 or dur > 30000:  # 异常值保护：负数或超过 30s
                continue
            idx_detail = re.search(r"index=(\d+)", se.get("detail", ""))
            idx_str = idx_detail.group(1) if idx_detail else ""
            items.append({
                "type": "span",
                "lane": sp_module,
                "t_start_ms": t_start,
                "t_end_ms": t_end,
                "label": f"{dur}ms",
                "data": se.get("detail", ""),
                "tooltip": f"{sp_label} #{idx_str}\n持续: {dur}ms\n{best_ee.get('detail', '')}",
                "css_class": sp_module,
            })
            used_indices.add(si)
            used_indices.add(best_ei)

    # ---- 配对 client prefill send_start → send_done ----
    c_starts = [(i, e) for i, e in enumerate(raw_events) if e["source"] == "client" and e["event"] == "send_start"]
    c_ends = [(i, e) for i, e in enumerate(raw_events) if e["source"] == "client" and e["event"] == "send_done"]
    for (si, se), (ei, ee) in zip(c_starts, c_ends):
        t_start = ts_to_ms(se["timestamp"]) - t0
        t_end = ts_to_ms(ee["timestamp"]) - t0
        dur = t_end - t_start
        # parse detail: "#0 [0-2s]SPEECH IMG_A"
        detail = se.get("detail", "")
        # extract image tag and audio tag
        img_tag = ""
        if "IMG_A" in detail:
            img_tag = "🌙 夜晚房间"
        elif "IMG_B" in detail:
            img_tag = "☀️ 白天公园"
        audio_tag = "🎤" if "SPEECH" in detail else "🔇"
        seg_match = re.search(r"\[(\d+-\d+s)\]", detail)
        seg = seg_match.group(1) if seg_match else ""
        items.append({
            "type": "span",
            "lane": "client_input",
            "t_start_ms": t_start,
            "t_end_ms": t_end,
            "label": f"{audio_tag}{seg} {img_tag}",
            "data": detail,
            "tooltip": f"Client Prefill {detail}\n耗时: {dur}ms",
            "css_class": "client_input",
        })
        used_indices.add(si)
        used_indices.add(ei)

    # ---- LLM prefill spans ----
    lp_starts = [(i, e) for i, e in enumerate(raw_events) if e["event"] == "prefill_start"]
    lp_ends = [(i, e) for i, e in enumerate(raw_events) if e["event"] == "prefill_done"]
    # 基于时间顺序配对（与 vision/audio encode 相同逻辑）
    lp_used_ends: set = set()
    for (si, se) in lp_starts:
        t_start_abs = ts_to_ms(se["timestamp"])
        best_ei, best_ee = -1, None
        for (ei, ee) in lp_ends:
            if ei in lp_used_ends:
                continue
            if ts_to_ms(ee["timestamp"]) >= t_start_abs:
                best_ei, best_ee = ei, ee
                break
        if best_ee is None:
            continue
        lp_used_ends.add(best_ei)
        t_start = t_start_abs - t0
        t_end = ts_to_ms(best_ee["timestamp"]) - t0
        dur = t_end - t_start
        if dur < 0 or dur > 30000:  # 异常值保护
            continue
        items.append({
            "type": "span",
            "lane": "llm",
            "t_start_ms": t_start,
            "t_end_ms": t_end,
            "label": f"Prefill {dur}ms",
            "data": "",
            "tooltip": f"LLM Prefill\n持续: {dur}ms",
            "css_class": "llm_prefill",
        })
        used_indices.add(si)
        used_indices.add(best_ei)

    # ---- Py Backend prefill spans (py_prefill_start → py_cpp_prefill_done) ----
    pp_starts = [(i, e) for i, e in enumerate(raw_events) if e["event"] == "py_prefill_start"]
    pp_ends = [(i, e) for i, e in enumerate(raw_events) if e["event"] == "py_cpp_prefill_done"]
    # 同时标记 py_cpp_prefill_call
    pp_calls = [(i, e) for i, e in enumerate(raw_events) if e["event"] == "py_cpp_prefill_call"]
    # 基于时间顺序配对
    pp_used_ends: set = set()
    pp_idx = 0
    for (si, se) in pp_starts:
        t_start_abs = ts_to_ms(se["timestamp"])
        best_ei, best_ee = -1, None
        for (ei, ee) in pp_ends:
            if ei in pp_used_ends:
                continue
            if ts_to_ms(ee["timestamp"]) >= t_start_abs:
                best_ei, best_ee = ei, ee
                break
        if best_ee is None:
            continue
        pp_used_ends.add(best_ei)
        t_start = t_start_abs - t0
        t_end = ts_to_ms(best_ee["timestamp"]) - t0
        dur = t_end - t_start
        if dur < 0 or dur > 30000:
            continue
        detail_str = best_ee.get("detail", "")
        items.append({
            "type": "span",
            "lane": "py_backend",
            "t_start_ms": t_start,
            "t_end_ms": t_end,
            "label": f"Prefill {dur}ms",
            "data": detail_str,
            "tooltip": f"Py Prefill #{pp_idx}\n持续: {dur}ms\n{se.get('detail','')}\n{detail_str}",
            "css_class": "py_prefill",
        })
        used_indices.add(si)
        used_indices.add(best_ei)
        pp_idx += 1
    for (ci, ce) in pp_calls:
        used_indices.add(ci)

    # ---- 逐个处理剩余事件 ----
    for i, e in enumerate(raw_events):
        if i in used_indices:
            continue

        t_ms = ts_to_ms(e["timestamp"]) - t0
        ev = e["event"]
        detail = e.get("detail", "")
        source = e["source"]
        module = e["module"]

        # 跳过噪声事件
        if ev in ("test_start", "test_done", "loop_done", "send_error",
                   "call_start", "call_done", "error",
                   "sys_init", "thread_start",
                   "decode_config", "batch_process",
                   "queue_info", "llm_queue_push"):
            continue

        # -- LLM to_tts: 仅保留有文本内容的输出
        if ev == "to_tts":
            text_content = detail.strip()
            text_match = re.match(r"(.+?)\s+\d+$", text_content)
            if text_match:
                text_content = text_match.group(1)
            if text_content == "0" or not text_content:
                # 空 token → 跳过（噪声事件）
                continue
            items.append({
                "type": "point",
                "lane": "llm",
                "t_ms": t_ms,
                "label": f'📝 "{text_content}"',
                "data": text_content,
                "tooltip": f'LLM→TTS 文本输出\n"{text_content}"\nt={t_ms/1000:.3f}s',
                "css_class": "llm_text",
            })
            continue

        # -- LLM end_token / decode_start: 跳过噪声（空转循环事件）
        if ev in ("end_token", "decode_start"):
            continue

        # -- TTS chunk_process
        if ev == "chunk_process":
            items.append({
                "type": "point",
                "lane": "tts",
                "t_ms": t_ms,
                "label": f"chunk#{detail}",
                "data": "",
                "tooltip": f"TTS Duplex 处理 chunk_idx={detail}\nt={t_ms/1000:.3f}s",
                "css_class": "tts",
            })
            continue

        # -- T2W wav_output: 音频数据
        if ev == "wav_output":
            parts = detail.split()
            if len(parts) >= 4:
                fname, dur_s, infer_ms, rtf = parts[0], parts[1], parts[2], parts[3]
                items.append({
                    "type": "point",
                    "lane": "t2w",
                    "t_ms": t_ms,
                    "label": f"🔊 {fname} {dur_s}s RTF={rtf}",
                    "data": f"{dur_s}s audio | {infer_ms}ms infer | RTF={rtf}",
                    "tooltip": f"Token2Wav 输出\n文件: {fname}\n音频: {dur_s}s\n推理: {infer_ms}ms\nRTF: {rtf}\nt={t_ms/1000:.3f}s",
                    "css_class": "t2w_wav",
                })
            continue

        # -- T2W first_audio
        if ev == "first_audio":
            items.append({
                "type": "point",
                "lane": "t2w",
                "t_ms": t_ms,
                "label": f"🎉 首响 {detail}ms",
                "data": f"首响时间: {detail}ms",
                "tooltip": f"C++ 首响时间: {detail}ms\nt={t_ms/1000:.3f}s",
                "css_class": "t2w_first",
            })
            continue

        # -- py_wav_send: WAV 传递数据
        if ev == "py_wav_send":
            items.append({
                "type": "point",
                "lane": "wav_delivery",
                "t_ms": t_ms,
                "label": f"📤 {detail}",
                "data": detail,
                "tooltip": f"Python→Client WAV\n{detail}\nt={t_ms/1000:.3f}s",
                "css_class": "wav_delivery",
            })
            continue

        # -- py events
        if ev == "py_is_listen":
            items.append({
                "type": "point",
                "lane": "py_backend",
                "t_ms": t_ms,
                "label": "👂 listen",
                "data": detail,
                "tooltip": f"Python 检测 is_listen=True\n{detail}\nt={t_ms/1000:.3f}s",
                "css_class": "py_listen",
            })
            continue

        if ev == "py_end_of_turn":
            items.append({
                "type": "point",
                "lane": "py_backend",
                "t_ms": t_ms,
                "label": "🔚 turn_end",
                "data": detail,
                "tooltip": f"Python 检测 end_of_turn\n{detail}\nt={t_ms/1000:.3f}s",
                "css_class": "py_end",
            })
            continue

        # py_backend: 跳过高频 decode call 和 generate_start（信息已在 client output 中）
        if ev in ("py_cpp_decode_call", "py_generate_start"):
            continue

        # py_prefill_start/call/done 已在 span 配对中处理
        if ev in ("py_prefill_start", "py_cpp_prefill_call", "py_cpp_prefill_done"):
            continue

        # -- Client audio_chunk
        if ev == "audio_chunk":
            dur_match = re.search(r"dur=([\d.]+)s", detail)
            dur_s = dur_match.group(1) if dur_match else "?"
            items.append({
                "type": "point",
                "lane": "client_output",
                "t_ms": t_ms,
                "label": f"🔊 {dur_s}s",
                "data": detail,
                "tooltip": f"Client 收到音频\n{detail}\nt={t_ms/1000:.3f}s",
                "css_class": "client_audio",
            })
            continue

        # -- Client text
        if ev == "text":
            text_match = re.search(r'"(.+)"', detail)
            text = text_match.group(1) if text_match else detail
            items.append({
                "type": "point",
                "lane": "client_output",
                "t_ms": t_ms,
                "label": f'📝 "{text}"',
                "data": text,
                "tooltip": f'Client 收到文本\n"{text}"\nt={t_ms/1000:.3f}s',
                "css_class": "client_text",
            })
            continue

        # -- Client is_listen: 跳过（高频噪声，每个 generate round 都有）
        if ev == "is_listen":
            continue

        # -- C++ prefill finish
        if ev == "finish":
            items.append({
                "type": "point",
                "lane": "cpp_prefill",
                "t_ms": t_ms,
                "label": f"✓ {detail[:30]}",
                "data": detail,
                "tooltip": f"C++ prefill 完成\n{detail}\nt={t_ms/1000:.3f}s",
                "css_class": "cpp_prefill_done",
            })
            continue

    # 排序
    def sort_key(item: Dict) -> int:
        return item.get("t_ms", item.get("t_start_ms", 0))
    items.sort(key=sort_key)

    # 折叠连续的"空转"事件
    # 空转事件组：连续多个属于同一组的事件会被折叠
    idle_groups = {
        "llm_idle": {"llm_empty", "llm_end", "llm_decode_start"},  # LLM 空转循环
        "py_idle": {"py_listen", "py_end", "py_gen"},  # Py backend 空转
        "client_idle": {"client_listen"},
    }
    # 建立 class → group 映射
    cls_to_group: Dict[str, str] = {}
    for group, classes in idle_groups.items():
        for cls in classes:
            cls_to_group[cls] = group

    collapsed: List[Dict] = []
    current_group: Optional[str] = None
    run_items: List[Dict] = []

    def flush_run() -> None:
        nonlocal run_items, current_group
        if not run_items:
            return
        if len(run_items) <= 4:
            # 少量事件不折叠
            collapsed.extend(run_items)
        else:
            # 保留首尾各 1 个 + 中间折叠标记
            collapsed.append(run_items[0])
            mid = run_items[len(run_items) // 2]
            n_folded = len(run_items) - 2
            collapsed.append({
                "type": "point",
                "lane": mid["lane"],
                "t_ms": mid["t_ms"],
                "label": f"···×{n_folded}",
                "data": "",
                "tooltip": f"{n_folded} 个空转事件已折叠\n从 {run_items[1].get('t_ms',0)/1000:.3f}s 到 {run_items[-2].get('t_ms',0)/1000:.3f}s",
                "css_class": mid.get("css_class", "llm_empty"),
            })
            collapsed.append(run_items[-1])
        run_items = []
        current_group = None

    for item in items:
        cls = item.get("css_class", "")
        group = cls_to_group.get(cls)
        if group:
            if current_group == group:
                run_items.append(item)
            else:
                flush_run()
                current_group = group
                run_items = [item]
        else:
            flush_run()
            collapsed.append(item)
    flush_run()

    return collapsed, t0


# ==================== HTML 生成 ====================

def generate_html(items: List[Dict], t0_ms: int, output_path: str) -> None:
    """生成 HTML"""

    # 计算时间范围
    all_times: List[int] = []
    for item in items:
        if item["type"] == "span":
            all_times.extend([item["t_start_ms"], item["t_end_ms"]])
        else:
            all_times.append(item["t_ms"])
    if not all_times:
        print("No events!")
        return
    t_min = min(all_times)
    t_max = max(all_times)
    duration_ms = t_max - t_min

    items_json = json.dumps(items, ensure_ascii=False)

    html = f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<title>MiniCPM-o 双工流水线时间轴</title>
<style>
:root {{
  --bg: #0d1117; --bg2: #161b22; --bg3: #21262d;
  --border: #30363d; --text: #c9d1d9; --text2: #8b949e;
  --accent: #58a6ff; --green: #3fb950; --red: #f85149;
  --orange: #d29922; --purple: #bc8cff; --pink: #f778ba;
  --cyan: #39d2c0; --yellow: #e3b341;
}}
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family: -apple-system, 'SF Mono', Menlo, monospace; background: var(--bg); color: var(--text); }}
.header {{ padding: 16px 24px; border-bottom: 1px solid var(--border); display: flex; align-items: center; justify-content: space-between; }}
.header h1 {{ font-size: 15px; color: var(--accent); font-weight: 600; }}
.header .meta {{ font-size: 11px; color: var(--text2); }}
.controls {{ padding: 8px 24px; border-bottom: 1px solid var(--border); display: flex; gap: 8px; align-items: center; }}
.controls button {{ background: var(--bg3); border: 1px solid var(--border); color: var(--text); padding: 4px 12px;
  border-radius: 6px; font-size: 11px; cursor: pointer; }}
.controls button:hover {{ background: var(--border); }}
.controls .info {{ font-size: 11px; color: var(--text2); margin-left: auto; }}

/* 主区域 */
.main {{ display: flex; overflow: hidden; }}
.lane-labels {{ width: 140px; min-width: 140px; border-right: 1px solid var(--border); padding-top: 32px; }}
.lane-label {{ height: 40px; display: flex; align-items: center; justify-content: flex-end;
  padding-right: 12px; font-size: 10px; font-weight: 600; color: var(--text2); border-bottom: 1px solid var(--border); }}
.lane-label .dot {{ width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; }}

.tracks-area {{ flex: 1; overflow-x: auto; overflow-y: hidden; position: relative; }}
.time-ruler {{ height: 32px; position: relative; border-bottom: 1px solid var(--border); }}
.tick {{ position: absolute; top: 0; height: 100%; border-left: 1px solid var(--border); }}
.tick.major {{ border-left-color: #30363d; }}
.tick.minor {{ border-left-color: #21262d; }}
.tick-label {{ position: absolute; bottom: 4px; left: 4px; font-size: 9px; color: var(--text2); white-space: nowrap; }}

.tracks {{ position: relative; }}
.track {{ height: 40px; position: relative; border-bottom: 1px solid var(--border); }}
.track:nth-child(odd) {{ background: rgba(22,27,34,0.5); }}

/* ====== Span 事件（时间段）====== */
.ev {{ position: absolute; font-size: 9px; cursor: default; z-index: 2;
  transition: z-index 0s, box-shadow 0.15s; }}
.ev.span {{ top: 4px; height: 32px; border-radius: 4px;
  display: flex; align-items: center; padding: 0 5px;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  border: 1px solid transparent; min-width: 3px; }}
.ev.span:hover {{ z-index: 20; box-shadow: 0 0 0 2px var(--accent); overflow: visible; }}
.ev.span .data-badge {{ display: inline-block; margin-left: 4px; padding: 1px 5px; border-radius: 3px;
  background: rgba(0,0,0,0.4); font-size: 8px; max-width: 120px; overflow: hidden; text-overflow: ellipsis; }}

/* ====== Point 事件（时间点）====== */
.ev.point {{ top: 0; height: 40px; background: none !important; border: none !important;
  padding: 0; overflow: visible; display: flex; align-items: center; }}
.ev.point .marker {{ width: 8px; height: 8px; border-radius: 2px;
  transform: rotate(45deg); position: absolute; top: 16px; left: 0; z-index: 3;
  flex-shrink: 0; }}
.ev.point .vline {{ position: absolute; left: 3px; top: 0; width: 1px;
  height: 40px; opacity: 0.2; pointer-events: none; }}
.ev.point .point-label {{ margin-left: 12px; max-width: 160px;
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
  font-size: 9px; opacity: 0.85; line-height: 40px; }}
.ev.point .point-label .data-badge {{ display: inline; margin-left: 4px; padding: 1px 4px; border-radius: 3px;
  background: rgba(0,0,0,0.35); font-size: 8px; }}
.ev.point:hover {{ z-index: 20; }}
.ev.point:hover .marker {{ transform: rotate(45deg) scale(1.5); box-shadow: 0 0 6px currentColor; }}
.ev.point:hover .vline {{ opacity: 0.5; }}
.ev.point:hover .point-label {{ overflow: visible; opacity: 1; max-width: none; }}

/* ====== Span 模块配色 ====== */
.ev.span.vision_encoder {{ background: rgba(248,81,73,0.25); border-color: var(--red); color: var(--red); }}
.ev.span.audio_encoder {{ background: rgba(210,153,34,0.25); border-color: var(--orange); color: var(--orange); }}
.ev.span.llm_prefill {{ background: rgba(63,185,80,0.3); border-color: var(--green); color: var(--green); }}
.ev.span.tts {{ background: rgba(88,166,255,0.2); border-color: var(--accent); color: var(--accent); }}
.ev.span.client_input {{ background: rgba(57,210,192,0.2); border-color: var(--cyan); color: var(--cyan); }}
.ev.span.py_prefill {{ background: rgba(57,210,192,0.1); border-color: #1b7c83; color: var(--cyan); }}

/* ====== Point 模块配色（通过 color 继承给 marker/label）====== */
.ev.point.llm_text {{ color: #7ee787; }}
.ev.point.llm_empty {{ color: #484f58; }}
.ev.point.llm_end {{ color: #f85149; }}
.ev.point.llm_decode_start {{ color: #3fb950; }}
.ev.point.tts {{ color: var(--accent); }}
.ev.point.t2w_wav {{ color: var(--purple); }}
.ev.point.t2w_first {{ color: #d2a8ff; font-weight: 700; }}
.ev.point.wav_delivery {{ color: var(--yellow); }}
.ev.point.client_audio {{ color: #d2a8ff; }}
.ev.point.client_text {{ color: #7ee787; }}
.ev.point.client_listen {{ color: #484f58; }}
.ev.point.py_listen {{ color: var(--orange); }}
.ev.point.py_end {{ color: var(--red); }}
.ev.point.py_gen {{ color: var(--accent); }}
.ev.point.cpp_prefill_done {{ color: var(--cyan); }}

/* Tooltip */
.tooltip {{ display: none; position: fixed; background: var(--bg2); border: 1px solid var(--accent);
  border-radius: 8px; padding: 10px 12px; font-size: 11px; line-height: 1.5; z-index: 100;
  pointer-events: none; box-shadow: 0 8px 24px rgba(0,0,0,0.5); max-width: 420px; white-space: pre-wrap; }}
.tooltip .t {{ color: var(--accent); font-weight: 700; }}

/* 统计面板 */
.stats {{ padding: 16px 24px; border-top: 1px solid var(--border); }}
.stats h3 {{ font-size: 13px; color: var(--accent); margin-bottom: 8px; }}
.stats table {{ border-collapse: collapse; font-size: 11px; }}
.stats td, .stats th {{ padding: 3px 12px 3px 0; }}
.stats th {{ color: var(--text2); font-weight: 500; }}
.stats .val {{ color: var(--green); font-weight: 600; }}
</style>
</head>
<body>

<div class="header">
  <h1>MiniCPM-o 双工模式 · 流水线时间轴</h1>
  <div class="meta" id="meta"></div>
</div>
<div class="controls">
  <button onclick="zoom(1.5)">放大 +</button>
  <button onclick="zoom(1/1.5)">缩小 −</button>
  <button onclick="pxPerMs=0.12;render()">重置</button>
  <div class="info" id="info"></div>
</div>

<div class="main">
  <div class="lane-labels" id="laneLabels"></div>
  <div class="tracks-area" id="tracksArea">
    <div class="time-ruler" id="ruler"></div>
    <div class="tracks" id="tracks"></div>
  </div>
</div>
<div class="tooltip" id="tt"></div>
<div class="stats" id="stats"></div>

<script>
const ITEMS = {items_json};
const DURATION_MS = {duration_ms};

const LANES = [
  {{ id: 'client_input',   label: 'Client 输入',   color: '#39d2c0' }},
  {{ id: 'vision_encoder', label: 'Vision Enc',     color: '#f85149' }},
  {{ id: 'audio_encoder',  label: 'Audio Enc',      color: '#d29922' }},
  {{ id: 'cpp_prefill',    label: 'C++ Prefill',    color: '#39d2c0' }},
  {{ id: 'llm',            label: 'LLM',            color: '#3fb950' }},
  {{ id: 'tts',            label: 'TTS',            color: '#58a6ff' }},
  {{ id: 't2w',            label: 'Token2Wav',      color: '#bc8cff' }},
  {{ id: 'py_backend',     label: 'Py Backend',     color: '#d29922' }},
  {{ id: 'wav_delivery',   label: 'WAV 传递',       color: '#e3b341' }},
  {{ id: 'client_output',  label: 'Client 输出',    color: '#a371f7' }},
];
const laneIndex = {{}};
const laneColor = {{}};
LANES.forEach((l, i) => {{ laneIndex[l.id] = i; laneColor[l.id] = l.color; }});

let pxPerMs = 0.12;

function zoom(factor) {{ pxPerMs *= factor; render(); }}

function render() {{
  const totalPx = DURATION_MS * pxPerMs + 100;

  // ---- Lane labels ----
  const labelsEl = document.getElementById('laneLabels');
  labelsEl.innerHTML = '';
  LANES.forEach(l => {{
    const d = document.createElement('div');
    d.className = 'lane-label';
    d.innerHTML = `<span class="dot" style="background:${{l.color}}"></span>${{l.label}}`;
    labelsEl.appendChild(d);
  }});

  // ---- Time ruler ----
  const ruler = document.getElementById('ruler');
  ruler.innerHTML = '';
  ruler.style.width = totalPx + 'px';
  // 动态刻度间隔
  const targetTickPx = 80;
  let tickMs = 1000;
  const candidates = [100, 200, 500, 1000, 2000, 5000, 10000];
  for (const c of candidates) {{
    if (c * pxPerMs >= targetTickPx * 0.5) {{ tickMs = c; break; }}
  }}
  for (let ms = 0; ms <= DURATION_MS + tickMs; ms += tickMs) {{
    const tick = document.createElement('div');
    tick.className = 'tick major';
    tick.style.left = ms * pxPerMs + 'px';
    const lbl = document.createElement('span');
    lbl.className = 'tick-label';
    lbl.textContent = (ms / 1000).toFixed(ms < 10000 ? 1 : 0) + 's';
    tick.appendChild(lbl);
    ruler.appendChild(tick);
    // minor ticks
    if (tickMs >= 1000) {{
      for (let m = tickMs / 5; m < tickMs; m += tickMs / 5) {{
        if (ms + m > DURATION_MS + tickMs) break;
        const mt = document.createElement('div');
        mt.className = 'tick minor';
        mt.style.left = (ms + m) * pxPerMs + 'px';
        ruler.appendChild(mt);
      }}
    }}
  }}

  // ---- Tracks ----
  const tracksEl = document.getElementById('tracks');
  tracksEl.innerHTML = '';
  tracksEl.style.width = totalPx + 'px';
  const trackDivs = [];
  LANES.forEach(() => {{
    const t = document.createElement('div');
    t.className = 'track';
    tracksEl.appendChild(t);
    trackDivs.push(t);
  }});

  // ---- Render events ----
  ITEMS.forEach(item => {{
    const laneIdx = laneIndex[item.lane];
    if (laneIdx === undefined) return;
    const trackDiv = trackDivs[laneIdx];

    const el = document.createElement('div');

    if (item.type === 'span') {{
      el.className = `ev span ${{item.css_class}}`;
      el.style.left = item.t_start_ms * pxPerMs + 'px';
      el.style.width = Math.max(3, (item.t_end_ms - item.t_start_ms) * pxPerMs) + 'px';
      let inner = item.label;
      if (item.data && item.data !== item.label) {{
        inner += `<span class="data-badge">${{escHtml(item.data.substring(0, 40))}}</span>`;
      }}
      el.innerHTML = inner;
      el.dataset.tip = item.tooltip + `\\n开始: ${{(item.t_start_ms/1000).toFixed(3)}}s\\n结束: ${{(item.t_end_ms/1000).toFixed(3)}}s`;
    }} else {{
      el.className = `ev point ${{item.css_class}}`;
      el.style.left = item.t_ms * pxPerMs + 'px';
      const c = laneColor[item.lane] || '#8b949e';
      let labelHtml = escHtml(item.label);
      if (item.data) {{
        labelHtml += `<span class="data-badge">${{escHtml(item.data.substring(0, 40))}}</span>`;
      }}
      el.innerHTML = `<div class="vline" style="background:${{c}}"></div>`
        + `<div class="marker" style="background:${{c}}"></div>`
        + `<span class="point-label">${{labelHtml}}</span>`;
      el.dataset.tip = item.tooltip;
    }}

    // Tooltip
    el.addEventListener('mouseenter', e => {{
      const tt = document.getElementById('tt');
      tt.style.display = 'block';
      tt.innerHTML = `<span class="t">${{el.dataset.tip.split('\\n')[0]}}</span>\\n${{el.dataset.tip.split('\\n').slice(1).join('\\n')}}`;
      const x = Math.min(e.clientX + 12, window.innerWidth - 440);
      const y = Math.max(e.clientY - 60, 10);
      tt.style.left = x + 'px';
      tt.style.top = y + 'px';
    }});
    el.addEventListener('mousemove', e => {{
      const tt = document.getElementById('tt');
      tt.style.left = Math.min(e.clientX + 12, window.innerWidth - 440) + 'px';
      tt.style.top = Math.max(e.clientY - 60, 10) + 'px';
    }});
    el.addEventListener('mouseleave', () => {{
      document.getElementById('tt').style.display = 'none';
    }});

    trackDiv.appendChild(el);
  }});

  // ---- Info ----
  document.getElementById('info').textContent = `${{ITEMS.length}} events · ${{(DURATION_MS/1000).toFixed(1)}}s · ${{pxPerMs.toFixed(3)}} px/ms`;
  document.getElementById('meta').textContent = `总时长 ${{(DURATION_MS/1000).toFixed(1)}}s · ${{ITEMS.length}} events`;

  renderStats();
}}

function escHtml(s) {{
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}}

function renderStats() {{
  const panel = document.getElementById('stats');
  const vSpans = ITEMS.filter(i => i.lane === 'vision_encoder' && i.type === 'span');
  const aSpans = ITEMS.filter(i => i.lane === 'audio_encoder' && i.type === 'span');
  const llmPrefills = ITEMS.filter(i => i.css_class === 'llm_prefill');
  const t2wWavs = ITEMS.filter(i => i.css_class === 't2w_wav');
  const llmTexts = ITEMS.filter(i => i.css_class === 'llm_text');
  const clientAudios = ITEMS.filter(i => i.css_class === 'client_audio');

  const avgDur = spans => spans.length ? (spans.reduce((a,s) => a + s.t_end_ms - s.t_start_ms, 0) / spans.length).toFixed(0) : '-';

  let h = '<h3>关键统计</h3><table>';
  h += `<tr><th>指标</th><th>值</th></tr>`;
  h += `<tr><td>Vision Encode</td><td class="val">${{avgDur(vSpans)}}ms avg (${{vSpans.length}}次)</td></tr>`;
  h += `<tr><td>Audio Encode</td><td class="val">${{avgDur(aSpans)}}ms avg (${{aSpans.length}}次)</td></tr>`;
  h += `<tr><td>LLM Prefill</td><td class="val">${{avgDur(llmPrefills)}}ms avg (${{llmPrefills.length}}次)</td></tr>`;
  h += `<tr><td>LLM 有效文本输出</td><td class="val">${{llmTexts.length}} chunks</td></tr>`;
  t2wWavs.forEach(w => {{
    h += `<tr><td>${{w.label}}</td><td class="val">${{w.data}}</td></tr>`;
  }});
  h += `<tr><td>Client 收到音频</td><td class="val">${{clientAudios.length}} chunks</td></tr>`;
  h += '</table>';
  panel.innerHTML = h;
}}

render();
</script>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"HTML 已生成: {output_path}")


# ==================== Main ====================

def parse_prefill_finish_from_log(log_path: str) -> List[Dict]:
    """从原始 server log 解析 finish stream_prefill 事件

    C++ 的 print_with_timestamp("\\n\\nc++ finish...") 导致时间戳在前 2 行：
      [CPP] HH:MM:SS.mmm
      [CPP]
      [CPP] c++ finish stream_prefill(index=N). n_past=X, ...
    """
    events: List[Dict] = []
    if not os.path.exists(log_path):
        return events
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    ts_re = re.compile(r"\[CPP\]\s+(\d{2}:\d{2}:\d{2}\.\d{3})\s*$")
    finish_re = re.compile(r"\[CPP\]\s+c\+\+ finish stream_prefill\(index=(\d+)\)\.\s*n_past=(\d+)")

    for i, line in enumerate(lines):
        m = finish_re.search(line)
        if m:
            idx_val = m.group(1)
            n_past = m.group(2)
            # 向前找时间戳 (2行内)
            ts = ""
            for j in range(max(0, i - 3), i):
                tm = ts_re.search(lines[j])
                if tm:
                    ts = tm.group(1)
            if ts:
                events.append({
                    "timestamp": ts,
                    "source": "cpp",
                    "module": "prefill",
                    "event": "finish",
                    "detail": f"index={idx_val} n_past={n_past}",
                })
    return events


def main() -> None:
    events_path = os.path.join(OUTPUT_DIR, "all_events.json")
    with open(events_path, "r", encoding="utf-8") as f:
        raw_events = json.load(f)
    print(f"加载 {len(raw_events)} 个原始事件")

    # 补充从 raw log 解析的 prefill finish 事件
    raw_log_path = os.path.join(OUTPUT_DIR, "server_log_raw.txt")
    finish_events = parse_prefill_finish_from_log(raw_log_path)
    if finish_events:
        print(f"补充 {len(finish_events)} 个 prefill finish 事件")
        raw_events.extend(finish_events)
        raw_events.sort(key=lambda e: e["timestamp"])

    items, t0 = preprocess_events(raw_events)

    # 计算活跃阶段截止时间：最后一个有意义事件 + margin
    meaningful_events = ["wav_output", "audio_chunk", "py_wav_send", "send_start",
                         "vision_encode_start", "to_tts", "first_audio"]
    last_meaningful_ms = 0
    for item in items:
        t_ms = item.get("t_ms", item.get("t_end_ms", 0))
        # 检查 label 和 css_class 来判断是否是有意义事件
        if item.get("css_class") in ("t2w_wav", "t2w_first", "client_audio", "client_text",
                                      "llm_text", "wav_delivery", "client_input",
                                      "vision_encoder", "audio_encoder"):
            last_meaningful_ms = max(last_meaningful_ms, t_ms)
        if item.get("lane") in ("client_input", "vision_encoder", "audio_encoder"):
            end = item.get("t_end_ms", item.get("t_ms", 0))
            last_meaningful_ms = max(last_meaningful_ms, end)

    cutoff_ms = last_meaningful_ms + int(ACTIVE_CUTOFF_MARGIN_S * 1000)
    print(f"活跃阶段截止: {cutoff_ms/1000:.1f}s (最后有意义事件 {last_meaningful_ms/1000:.1f}s + {ACTIVE_CUTOFF_MARGIN_S}s)")

    # 过滤
    items = [item for item in items if
             item.get("t_ms", item.get("t_start_ms", 0)) <= cutoff_ms]
    print(f"过滤后 {len(items)} 个可视化项")

    # 按 lane 统计
    from collections import Counter
    lane_counts = Counter(item["lane"] for item in items)
    for lane, cnt in lane_counts.most_common():
        print(f"  {lane}: {cnt}")

    html_path = os.path.join(OUTPUT_DIR, "timeline.html")
    generate_html(items, t0, html_path)


if __name__ == "__main__":
    main()
