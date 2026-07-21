#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
双工可行性报告生成器

读取 perf-duplex 产出的 JSON（frames + audio_chunks 时间线），计算关键指标，
给出「该机器能否支撑双工」的判定。

指标含义（详见 DUPLEX_PROFILING.md）：
  - LLM 判定实时性 : 每帧 push->判定(ms_total) 的 P50/P95/max。
                     P95 必须 < 进帧间隔(stream_interval_ms)，否则 pipeline
                     跟不上 1s 进帧速率，in-flight 队列会持续堆积。
  - 首响延迟       : 每个 SPEAK 轮次「首帧 push -> 该轮首个 wav 落盘」的时间。
  - 音频 RTF       : 生成 1s 音频所花的 wall time。<1.0 才能保证播放不饿死。
  - wav chunk 分布 : 说明单个 wav chunk 的实际音频时长范围。

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
    """把连续的 is_speak 帧聚合成 SPEAK 轮次。返回 [(start_idx, end_idx)] 列表。"""
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
    """把 audio_chunks 按 is_final 切成音频轮次。返回 [chunk子列表]。"""
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


def analyze(report, interval_ms):
    meta = report.get("meta", {})
    frames = report.get("frames", [])
    audio = report.get("audio_chunks", [])

    interval = interval_ms if interval_ms is not None else meta.get("stream_interval_ms", 1000)
    if interval <= 0:
        interval = 1000  # 背靠背压测时用 1000ms 作为实时基准

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
    out(f"    P50 {p50:.1f}ms | P95 {p95:.1f}ms | max {mx:.1f}ms | "
        f"avg decode {sum(ms_decode)/len(ms_decode):.1f}ms" if ms_decode else "    无数据")
    jud_llm = p95 < interval
    out(f"    判据: P95({p95:.1f}ms) < 进帧间隔({interval}ms)  => {'PASS' if jud_llm else 'FAIL'}")
    out("")

    # ---- 2. 首响延迟 ----
    speak_turns = segment_speak_turns(frames)
    audio_turns = segment_audio_turns(audio)
    out("[2] 首响延迟 (SPEAK 轮次首帧 push -> 该轮首个 wav 落盘)")
    first_resp = []
    n_match = min(len(speak_turns), len(audio_turns))
    if n_match == 0:
        out("    无 SPEAK 轮次或无音频输出（可能全程 LISTEN / 未开 TTS）")
    else:
        for k in range(n_match):
            s_start = speak_turns[k][0]
            t_push = frames[s_start]["t_push_ms"]
            t_first_wav = audio_turns[k][0]["t_complete_ms"]
            d = t_first_wav - t_push
            first_resp.append(d)
            out(f"    turn#{k+1}: 首帧 push@{t_push:.0f}ms -> 首 wav@{t_first_wav:.0f}ms "
                f"= {d:.0f}ms")
        out(f"    P50 {percentile(first_resp,50):.0f}ms | "
            f"P95 {percentile(first_resp,95):.0f}ms")
    jud_resp = bool(first_resp) and percentile(first_resp, 95) < interval
    if first_resp:
        out(f"    判据: 首响 P95 < 进帧间隔({interval}ms) => {'PASS' if jud_resp else 'FAIL'}")
    out("")

    # ---- 3. 音频 RTF ----
    out("[3] 音频生成 RTF (生成 1s 音频所花 wall time，需 < 1.0)")
    total_audio_s = sum(a["duration_s"] for a in audio)
    jud_rtf = True
    if audio and total_audio_s > 0:
        # 用每个音频轮次的 wall 跨度估算 RTF
        rtf_turns = []
        for k, t in enumerate(audio_turns):
            if not t:
                continue
            audio_s = sum(c["duration_s"] for c in t)
            # 该轮 wall 起点：优先用匹配的 speak 轮首帧 push，否则用首 chunk 落盘
            if k < len(speak_turns):
                wall_start = frames[speak_turns[k][0]]["t_push_ms"]
            else:
                wall_start = t[0]["t_complete_ms"]
            wall_end = t[-1]["t_complete_ms"]
            wall_s = max(1e-6, (wall_end - wall_start) / 1000.0)
            rtf = wall_s / audio_s if audio_s > 0 else float("inf")
            rtf_turns.append(rtf)
            out(f"    turn#{k+1}: 音频 {audio_s:.2f}s | wall {wall_s:.2f}s | RTF {rtf:.2f}")
        avg_rtf = sum(rtf_turns) / len(rtf_turns) if rtf_turns else float("inf")
        out(f"    平均 RTF: {avg_rtf:.2f}")
        jud_rtf = avg_rtf < 1.0
        out(f"    判据: 平均 RTF < 1.0 => {'PASS' if jud_rtf else 'FAIL'}")
    else:
        out("    无音频输出，跳过 RTF（全程 LISTEN 或未开 TTS）")
        jud_rtf = True
    out("")

    # ---- 4. 单 wav 时长分布（回答 demand 点 5） ----
    out("[4] 单个 wav chunk 时长分布 (验证「一帧≠1s音频」)")
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
    checks = {
        "LLM 判定实时性 (P95<间隔)": jud_llm,
        "首响 (<间隔)": jud_resp if first_resp else None,
        "音频 RTF (<1.0)": jud_rtf,
    }
    for name, v in checks.items():
        tag = "—" if v is None else ("PASS" if v else "FAIL")
        out(f"  [{tag:>4}] {name}")
    hard = [v for v in checks.values() if v is not None]
    verdict = all(hard) if hard else False
    out("-" * 64)
    out(f"  最终判定: {'✅ 该机器可支撑双工' if verdict else '❌ 该机器暂不满足双工实时性'}")
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

    sys.exit(0 if verdict else 2)


if __name__ == "__main__":
    main()
