#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
双工可行性报告生成器

读取 perf-duplex 产出的 JSON（frames + audio_chunks 时间线），计算关键指标，
给出「该机器能否支撑双工」的判定。

指标含义（详见 DUPLEX_PROFILING.md）：
  - LLM 判定实时性 : 每帧 push->判定(ms_total) 的 P50/P95/max。
                     P95 必须 < 进帧间隔(stream_interval_ms)。
  - 首响 e2e       : SPEAK 轮首帧 push -> 匹配到的首个 wav 落盘。
  - TTS RTF        : LLM 判定完成(t_done) -> 该轮末 wav / 音频时长；硬门槛 < 1.0。
  - e2e RTF        : 首帧 push -> 末 wav / 音频时长；仅展示，含 LLM 等待。

SPEAK 轮与音频轮按时间戳匹配（不用数组下标一一对应），避免中间无音频的
SPEAK 轮导致错位。

退出码:
  0 = 通过（可支撑双工）
  2 = 未通过实时性判据
  3 = 数据不完整（--no-tts / 无音频等），不能判定双工可行性

用法:
  python3 analyze_perf.py <perf_report.json> [--interval-ms 1000] [--md out.md]
"""

import argparse
import json
import sys


def percentile(values, p):
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    frac = k - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def segment_speak_turns(frames):
    """把连续的 is_speak 帧聚合成 SPEAK 轮次。返回 [(start_idx, end_idx)]。"""
    turns = []
    cur = None
    for i, fr in enumerate(frames):
        if fr.get("is_speak"):
            if cur is None:
                cur = [i, i]
            else:
                cur[1] = i
        else:
            if cur is not None:
                turns.append(tuple(cur))
                cur = None
    if cur is not None:
        turns.append(tuple(cur))
    return turns


def segment_audio_turns(audio):
    """优先按 audio_turn_id 分组；否则按 is_final 切分。返回 [chunk子列表]。"""
    if audio and all("audio_turn_id" in a for a in audio):
        groups = {}
        order = []
        for a in audio:
            tid = a["audio_turn_id"]
            if tid not in groups:
                groups[tid] = []
                order.append(tid)
            groups[tid].append(a)
        return [groups[tid] for tid in order]

    turns = []
    cur = []
    for a in audio:
        cur.append(a)
        if a.get("is_final"):
            turns.append(cur)
            cur = []
    if cur:
        turns.append(cur)
    return turns


def match_speak_audio_turns(frames, speak_turns, audio_turns):
    """
    按时间戳匹配 SPEAK 轮与音频轮，避免靠数组下标错位。

    对每个音频轮（按时间顺序）：取「尚未匹配、且 t_push <= 首 wav 落盘」的
    最晚一个 SPEAK 轮。被跳过的 SPEAK 轮记为无音频输出。
    """
    pairs = []
    unmatched_speak = []
    unmatched_audio = []
    next_s = 0

    for a_idx, a_turn in enumerate(audio_turns):
        if not a_turn:
            unmatched_audio.append(a_idx)
            continue
        t0 = a_turn[0]["t_complete_ms"]
        candidate = None
        while next_s < len(speak_turns):
            s0, _s1 = speak_turns[next_s]
            t_push = frames[s0]["t_push_ms"]
            if t_push <= t0:
                if candidate is not None:
                    unmatched_speak.append(candidate)
                candidate = next_s
                next_s += 1
            else:
                break
        if candidate is None:
            unmatched_audio.append(a_idx)
        else:
            pairs.append((candidate, a_idx))

    while next_s < len(speak_turns):
        unmatched_speak.append(next_s)
        next_s += 1

    return pairs, unmatched_speak, unmatched_audio


def speak_turn_label(frames, speak_turns, s_idx):
    s0, _s1 = speak_turns[s_idx]
    tid = frames[s0].get("speak_turn_id")
    if tid is not None and tid >= 0:
        return f"speak#{tid}"
    return f"speak@{s_idx}"


def analyze(report, interval_ms):
    meta = report.get("meta", {})
    frames = report.get("frames", [])
    audio = report.get("audio_chunks", [])

    interval = interval_ms if interval_ms is not None else meta.get("stream_interval_ms", 1000)
    if interval <= 0:
        interval = 1000

    use_tts = bool(meta.get("use_tts", True))

    lines = []

    def out(s=""):
        lines.append(s)

    out("=" * 64)
    out("MiniCPM-o 双工可行性报告")
    out("=" * 64)
    out(f"LLM           : {meta.get('llm_path', '?')}")
    out(f"Vision backend: {meta.get('vision_backend', '?')}   "
        f"use_tts: {meta.get('use_tts')}   media_type: {meta.get('media_type')}")
    out(f"n_threads     : {meta.get('n_threads', '?')}   "
        f"采样率: {meta.get('sample_rate_hint', '?')}Hz")
    out(f"进帧间隔(基准): {interval} ms")
    out("")

    n_speak = sum(1 for f in frames if f.get("is_speak"))
    n_listen = sum(1 for f in frames if not f.get("is_speak"))
    out(f"帧数: {len(frames)}  (SPEAK {n_speak} / LISTEN {n_listen})")
    out("")

    # ---- 1. LLM 判定实时性 ----
    ms_total = [f["ms_total"] for f in frames if f.get("ok")]
    ms_decode = [f["ms_decode"] for f in frames if f.get("ok")]
    p50 = percentile(ms_total, 50)
    p95 = percentile(ms_total, 95)
    mx = max(ms_total) if ms_total else 0.0
    out("[1] LLM 判定延迟 (push -> LISTEN/SPEAK, ms_total)")
    if ms_decode:
        out(f"    P50 {p50:.1f}ms | P95 {p95:.1f}ms | max {mx:.1f}ms | "
            f"avg decode {sum(ms_decode)/len(ms_decode):.1f}ms")
    else:
        out("    无数据")
    jud_llm = bool(ms_total) and p95 < interval
    out(f"    判据: P95({p95:.1f}ms) < 进帧间隔({interval}ms)  => "
        f"{'PASS' if jud_llm else 'FAIL'}")
    out("")

    speak_turns = segment_speak_turns(frames)
    audio_turns = segment_audio_turns(audio)
    pairs, unmatched_speak, unmatched_audio = match_speak_audio_turns(
        frames, speak_turns, audio_turns)

    out("[匹配] SPEAK 轮 <-> 音频轮 (按时间戳，非下标对齐)")
    out(f"    SPEAK 轮: {len(speak_turns)} | 音频轮: {len(audio_turns)} | "
        f"匹配成功: {len(pairs)}")
    if unmatched_speak:
        labels = [speak_turn_label(frames, speak_turns, i) for i in unmatched_speak]
        out(f"    无音频的 SPEAK 轮: {labels}")
    if unmatched_audio:
        out(f"    未匹配到 SPEAK 的音频轮 index: {unmatched_audio}")
    out("")

    # TTS 侧是否具备判定条件
    if not use_tts:
        tts_status = "skipped"   # --no-tts
        tts_reason = "use_tts=false (--no-tts)，跳过音频侧判定"
    elif n_speak == 0:
        tts_status = "skipped"   # 全程 LISTEN，没有可测的 speak/audio
        tts_reason = "全程 LISTEN，无 SPEAK 轮，跳过音频侧判定"
    elif not audio or not pairs:
        tts_status = "incomplete"
        tts_reason = "有 SPEAK 但无可用音频时间线，无法判定首响/RTF"
    else:
        tts_status = "ok"
        tts_reason = ""

    # ---- 2. 首响延迟 ----
    out("[2] 首响延迟")
    out("    e2e  = SPEAK 首帧 push -> 该轮首 wav（硬判据）")
    out("    tts  = SPEAK 首帧 t_done(LLM完成) -> 首 wav（仅展示）")
    first_resp_e2e = []
    first_resp_tts = []
    if tts_status != "ok":
        out(f"    {tts_reason}")
        jud_resp = None
    else:
        for s_idx, a_idx in pairs:
            s0, _s1 = speak_turns[s_idx]
            a_turn = audio_turns[a_idx]
            t_push = frames[s0]["t_push_ms"]
            t_done = frames[s0]["t_done_ms"]
            t_first = a_turn[0]["t_complete_ms"]
            d_e2e = t_first - t_push
            d_tts = t_first - t_done
            first_resp_e2e.append(d_e2e)
            first_resp_tts.append(d_tts)
            label = speak_turn_label(frames, speak_turns, s_idx)
            out(f"    {label}/audio#{a_idx}: e2e {d_e2e:.0f}ms | tts {d_tts:.0f}ms "
                f"(push@{t_push:.0f} done@{t_done:.0f} wav@{t_first:.0f})")
        out(f"    e2e P50 {percentile(first_resp_e2e,50):.0f}ms | "
            f"P95 {percentile(first_resp_e2e,95):.0f}ms")
        out(f"    tts P50 {percentile(first_resp_tts,50):.0f}ms | "
            f"P95 {percentile(first_resp_tts,95):.0f}ms")
        jud_resp = percentile(first_resp_e2e, 95) < interval
        out(f"    判据: 首响 e2e P95 < 进帧间隔({interval}ms) => "
            f"{'PASS' if jud_resp else 'FAIL'}")
    out("")

    # ---- 3. 音频 RTF ----
    out("[3] 音频 RTF")
    out("    TTS RTF = (末 wav - LLM t_done) / 音频时长  （硬判据，需 < 1.0）")
    out("    e2e RTF = (末 wav - 首帧 push) / 音频时长  （仅展示，含 LLM 等待）")
    total_audio_s = sum(a["duration_s"] for a in audio)
    tts_rtf_turns = []
    e2e_rtf_turns = []
    if tts_status != "ok":
        out(f"    {tts_reason}")
        jud_rtf = None
    else:
        for s_idx, a_idx in pairs:
            s0, _s1 = speak_turns[s_idx]
            a_turn = audio_turns[a_idx]
            audio_s = sum(c["duration_s"] for c in a_turn)
            if audio_s <= 0:
                continue
            t_push = frames[s0]["t_push_ms"]
            t_done = frames[s0]["t_done_ms"]
            t_first = a_turn[0]["t_complete_ms"]
            t_last = a_turn[-1]["t_complete_ms"]
            # 若音频回调早于 wait_next_frame 返回，用首 wav 时刻避免负 wall
            tts_wall_start = min(t_done, t_first)
            tts_wall_s = max(1e-6, (t_last - tts_wall_start) / 1000.0)
            e2e_wall_s = max(1e-6, (t_last - t_push) / 1000.0)
            tts_rtf = tts_wall_s / audio_s
            e2e_rtf = e2e_wall_s / audio_s
            tts_rtf_turns.append(tts_rtf)
            e2e_rtf_turns.append(e2e_rtf)
            label = speak_turn_label(frames, speak_turns, s_idx)
            out(f"    {label}/audio#{a_idx}: 音频 {audio_s:.2f}s | "
                f"TTS wall {tts_wall_s:.2f}s RTF {tts_rtf:.2f} | "
                f"e2e wall {e2e_wall_s:.2f}s RTF {e2e_rtf:.2f}")
        if tts_rtf_turns:
            avg_tts = sum(tts_rtf_turns) / len(tts_rtf_turns)
            avg_e2e = sum(e2e_rtf_turns) / len(e2e_rtf_turns)
            out(f"    平均 TTS RTF: {avg_tts:.2f} | 平均 e2e RTF: {avg_e2e:.2f}")
            jud_rtf = avg_tts < 1.0
            out(f"    判据: 平均 TTS RTF < 1.0 => {'PASS' if jud_rtf else 'FAIL'}")
        else:
            out("    匹配轮次音频时长均为 0，无法计算 RTF")
            jud_rtf = None
            tts_status = "incomplete"
            tts_reason = "匹配轮次无有效音频时长"
    out("")

    # ---- 4. 单 wav 时长分布 ----
    out("[4] 单个 wav chunk 时长分布 (验证「一帧!=1s音频」)")
    durs = [a["duration_s"] for a in audio]
    if durs:
        out(f"    chunk 数: {len(durs)} | 总音频 {total_audio_s:.2f}s")
        out(f"    时长 min {min(durs):.2f}s | P50 {percentile(durs,50):.2f}s | "
            f"max {max(durs):.2f}s")
        finals = [a["duration_s"] for a in audio if a.get("is_final")]
        if finals:
            out(f"    轮末 (is_final) 时长: {[f'{x:.2f}s' for x in finals]}")
        out("    说明: 满窗 chunk ≈1.0s，轮末 remainder 在 (0,1.0]s；"
            "单帧产出的音频量取决于该帧说了多少字。")
    else:
        out("    无音频输出。")
    out("")

    # ---- 总判定 ----
    out("=" * 64)
    checks = [
        ("LLM 判定实时性 (P95<间隔)", jud_llm if ms_total else None),
        ("首响 e2e (<间隔)", jud_resp),
        ("TTS RTF (<1.0)", jud_rtf),
    ]
    for name, v in checks:
        if v is None:
            tag = "SKIP"
        elif v:
            tag = "PASS"
        else:
            tag = "FAIL"
        out(f"  [{tag:>4}] {name}")

    hard = [v for _n, v in checks if v is not None]
    any_fail = any(v is False for _n, v in checks)

    if tts_status == "incomplete" or (tts_status == "skipped" and use_tts is False):
        # --no-tts 或音频数据缺失：不能宣称可支撑双工
        if any_fail:
            verdict = "fail"
            summary = "未通过实时性判据（且音频侧数据不完整）"
        else:
            verdict = "incomplete"
            summary = f"数据不完整，不能判定双工可行性 ({tts_reason or tts_status})"
    elif tts_status == "skipped" and n_speak == 0:
        # 全程 LISTEN：只看 LLM；通过也不等于「双工音频 OK」，标 incomplete
        if any_fail:
            verdict = "fail"
            summary = "未通过实时性判据"
        else:
            verdict = "incomplete"
            summary = "全程 LISTEN，未覆盖 SPEAK/TTS，不能判定双工可行性"
    elif not hard:
        verdict = "incomplete"
        summary = "无有效判据数据"
    elif all(hard):
        verdict = "pass"
        summary = "该机器可支撑双工"
    else:
        verdict = "fail"
        summary = "该机器暂不满足双工实时性"

    out("-" * 64)
    out(f"  最终判定: {summary}")
    out("=" * 64)

    return "\n".join(lines), verdict


def main():
    ap = argparse.ArgumentParser(description="双工可行性报告生成器")
    ap.add_argument("json_path", help="perf-duplex 产出的 JSON 路径")
    ap.add_argument("--interval-ms", type=int, default=None,
                    help="进帧间隔基准 (默认读 JSON meta.stream_interval_ms)")
    ap.add_argument("--md", default=None, help="额外把报告写到该 markdown 文件")
    args = ap.parse_args()

    with open(args.json_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    text, verdict = analyze(report, args.interval_ms)
    print(text)

    if args.md:
        with open(args.md, "w", encoding="utf-8") as f:
            f.write("```\n" + text + "\n```\n")
        print(f"\n[已写入 markdown: {args.md}]")

    if verdict == "pass":
        sys.exit(0)
    if verdict == "incomplete":
        sys.exit(3)
    sys.exit(2)


if __name__ == "__main__":
    main()
