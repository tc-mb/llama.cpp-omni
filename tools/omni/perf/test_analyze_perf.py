"""Model-free regressions for duplex coverage and playback viability."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from analyze_perf import analyze, playback_continuity


def report(times=(200, 1000, 1800, 2600)):
    return {
        "meta": {"use_tts": True, "stream_interval_ms": 1000},
        "frames": [{"ok": True, "is_speak": True, "ms_total": 100,
                    "ms_decode": 80, "t_push_ms": 0, "t_done_ms": 100}],
        "audio_chunks": [{"t_complete_ms": t, "duration_s": 1,
                          "is_final": i == len(times) - 1, "audio_turn_id": 0}
                         for i, t in enumerate(times)],
    }


class PlaybackTests(unittest.TestCase):
    def test_same_rtf_different_continuity(self):
        smooth = report()
        burst = report((200, 2300, 2400, 2600))
        self.assertEqual(analyze(smooth, None)[1], "pass")
        text, verdict = analyze(burst, None)
        self.assertEqual(verdict, "fail")
        self.assertIn("underrun_ms=1100.0", text)
        self.assertIn("minimum_startup_buffer_ms=1100.0", text)

    def test_buffer_cost_cannot_hide_in_first_audio(self):
        burst = report((200, 2300, 2400, 2600))
        self.assertEqual(playback_continuity(burst["audio_chunks"], 1100)["underrun_ms"], 0)
        text, verdict = analyze(burst, None, startup_buffer_ms=1100)
        self.assertEqual(verdict, "fail")
        self.assertIn("e2e 1300ms", text)

    def test_buffer_partial_and_full_compensation(self):
        chunks = report((200, 2300, 2400, 2600))["audio_chunks"]
        self.assertEqual(playback_continuity(chunks, 200)["underrun_ms"], 900)
        self.assertEqual(playback_continuity(chunks, 500)["underrun_ms"], 600)

    def test_several_gaps_and_variable_duration(self):
        chunks = [{"t_complete_ms": 100, "duration_s": .2},
                  {"t_complete_ms": 350, "duration_s": .1},
                  {"t_complete_ms": 650, "duration_s": .4}]
        actual = playback_continuity(chunks)
        self.assertEqual(actual["gap_count"], 2)
        self.assertEqual(actual["underrun_ms"], 250)
        self.assertEqual(actual["max_gap_ms"], 200)
        self.assertEqual(actual["minimum_startup_buffer_ms"], 250)

    def test_explicit_gap_tolerance(self):
        burst = report((200, 1250, 1800, 2600))
        self.assertEqual(analyze(burst, None)[1], "fail")
        self.assertEqual(analyze(burst, None, max_gap_ms=50)[1], "pass")

    def test_zero_duration_terminal_marker_is_not_audio(self):
        r = report((200,))
        r["audio_chunks"][0]["is_final"] = False
        r["audio_chunks"].append({"t_complete_ms": 300, "duration_s": 0,
                                  "audio_turn_id": 0, "is_final": True})
        self.assertEqual(analyze(r, None)[1], "pass")

    def test_zero_duration_prefix_cannot_fake_first_audio(self):
        r = report((1200,))
        r["audio_chunks"].insert(0, {"t_complete_ms": 200, "duration_s": 0,
                                    "audio_turn_id": 0, "is_final": False})
        text, verdict = analyze(r, None)
        self.assertEqual(verdict, "fail")
        self.assertIn("e2e 1200ms", text)


class CoverageTests(unittest.TestCase):
    def test_failed_frame_cannot_pass(self):
        r = report()
        r["frames"].append({"ok": False, "is_speak": False, "ms_total": 5000,
                            "ms_decode": 0, "t_push_ms": 1000, "t_done_ms": 6000})
        self.assertEqual(analyze(r, None)[1], "fail")

    def test_unmatched_speak_is_incomplete(self):
        r = report((200,))
        r["frames"].extend([
            {"ok": True, "is_speak": False, "ms_total": 100, "ms_decode": 80,
             "t_push_ms": 1000, "t_done_ms": 1100},
            {"ok": True, "is_speak": True, "ms_total": 100, "ms_decode": 80,
             "t_push_ms": 2000, "t_done_ms": 2100}])
        self.assertEqual(analyze(r, None)[1], "incomplete")

    def test_unmatched_audio_is_incomplete(self):
        r = report((200,))
        r["audio_chunks"].append({"t_complete_ms": 300, "duration_s": 1,
                                  "is_final": True, "audio_turn_id": 1})
        self.assertEqual(analyze(r, None)[1], "incomplete")

    def test_unclosed_audio_is_incomplete(self):
        r = report()
        r["audio_chunks"][-1]["is_final"] = False
        self.assertEqual(analyze(r, None)[1], "incomplete")

    def test_no_audio_paths_are_incomplete(self):
        for r in ({}, {"meta": {"use_tts": False}}, report()):
            r["audio_chunks"] = []
            self.assertEqual(analyze(r, None)[1], "incomplete")
        r = report()
        r["frames"][0]["is_speak"] = False
        self.assertEqual(analyze(r, None)[1], "incomplete")

    def test_slow_turn_cannot_hide_in_average(self):
        r = report((200,))
        r["audio_chunks"][0]["duration_s"] = 10
        r["frames"] += [
            {"ok": True, "is_speak": False, "ms_total": 100, "ms_decode": 80,
             "t_push_ms": 10000, "t_done_ms": 10100},
            {"ok": True, "is_speak": True, "ms_total": 100, "ms_decode": 80,
             "t_push_ms": 11000, "t_done_ms": 11100}]
        r["audio_chunks"].append({"t_complete_ms": 11200, "duration_s": .05,
                                  "is_final": True, "audio_turn_id": 1})
        self.assertEqual(analyze(r, None)[1], "fail")

    def test_legacy_final_marker_segmentation(self):
        r = report()
        for chunk in r["audio_chunks"]:
            del chunk["audio_turn_id"]
        self.assertEqual(analyze(r, None)[1], "pass")


class InvalidDataTests(unittest.TestCase):
    def test_nonfinite_negative_missing_or_wrong_type(self):
        for value in (float("nan"), float("inf"), 10**1000, -1, None, True, "200"):
            for collection, key in (("frames", "ms_total"), ("audio_chunks", "duration_s")):
                with self.subTest(value=value, key=key):
                    r = report()
                    r[collection][0][key] = value
                    self.assertEqual(analyze(r, None)[1], "incomplete")

    def test_duration_overflow_and_invalid_turn_label(self):
        r = report()
        r["audio_chunks"][0]["duration_s"] = 1e308
        self.assertEqual(analyze(r, None)[1], "incomplete")
        r = report()
        r["frames"][0]["speak_turn_id"] = "0"
        self.assertEqual(analyze(r, None)[1], "incomplete")

    def test_bad_structure(self):
        for r in ([], {"meta": []}, {"frames": {}}, {"frames": [None]}, {"audio_chunks": "a"}):
            self.assertEqual(analyze(r, None)[1], "incomplete")

    def test_out_of_order_and_partial_ids(self):
        r = report()
        r["audio_chunks"][1]["t_complete_ms"] = 100
        self.assertEqual(analyze(r, None)[1], "incomplete")
        r = report()
        del r["audio_chunks"][1]["audio_turn_id"]
        self.assertEqual(analyze(r, None)[1], "incomplete")
        r = report()
        r["audio_chunks"][1]["is_final"] = True
        self.assertEqual(analyze(r, None)[1], "incomplete")

    def test_invalid_parameters(self):
        for value in (-1, float("nan"), float("inf")):
            self.assertEqual(analyze(report(), None, startup_buffer_ms=value)[1], "incomplete")
            self.assertEqual(analyze(report(), None, max_gap_ms=value)[1], "incomplete")
            self.assertEqual(analyze(report(), value)[1], "incomplete")

    def test_cli_exit_codes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            for data, code in ((report(), 0), (report((200, 2300, 2400, 2600)), 2), ({}, 3)):
                path.write_text(json.dumps(data), encoding="utf-8")
                result = subprocess.run([sys.executable, str(Path(__file__).with_name("analyze_perf.py")),
                                         str(path)], capture_output=True, text=True, check=False)
                self.assertEqual(result.returncode, code, result.stderr)

    def test_cli_malformed_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_text('{"frames":', encoding="utf-8")
            result = subprocess.run([sys.executable, str(Path(__file__).with_name("analyze_perf.py")),
                                     str(path)], capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 3)
            self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
