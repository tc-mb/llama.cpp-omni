#!/usr/bin/env python3
"""Static protocol checks for the opt-in C++ precision dispatch path."""

import re
import unittest
from pathlib import Path


OMNI_CPP = Path(__file__).resolve().parents[1] / "omni.cpp"


class CppOneShotWiringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = OMNI_CPP.read_text(encoding="utf-8")

    def test_oneshot_request_has_complete_protocol_fields(self):
        start = self.source.rindex("static bool process_python_t2w_tokens_oneshot(")
        end = self.source.index("// 重置 Python T2W 缓存", start)
        body = self.source[start:end]

        self.assertIn(r'\"cmd\":\"process_oneshot\"', body)
        self.assertIn(r'\"ref_audio_path\"', body)
        self.assertIn(r'\"output_path\"', body)
        self.assertIn(r'\"seed\"', body)
        self.assertNotIn("last_chunk", body)
        self.assertEqual(body.count("append_python_t2w_json_string(cmd,"), 2)

    def test_precision_final_carries_full_ar_tokens(self):
        assignments = re.findall(
            r"if \(nonstream_precision\) \{\s*"
            r"(?:[^{}]|\{[^{}]*\})*?audio_tokens\.assign\(all_audio_tokens\.begin\(\),\s*"
            r"all_audio_tokens\.end\(\)\);",
            self.source,
        )
        self.assertGreaterEqual(len(assignments), 2)
        self.assertIn("merged_success = !merged_embeddings.empty();", self.source)

    def test_precision_generator_does_not_enqueue_tokens_before_terminal(self):
        start = self.source.index("static bool generate_audio_tokens_local_simplex(")
        end = self.source.index("void tts_thread_func_duplex(", start)
        body = self.source[start:end]

        precision_tail = body.rindex("if (nonstream_precision) {")
        precision_tail = body[precision_tail:]
        self.assertIn("retained %zu verified audio tokens for terminal one-shot", precision_tail)
        self.assertNotIn("t2w_thread_info->queue.push(t2w_out)", precision_tail.split(
            "// 🔧 [与 Python 对齐] chunk 结束时", 1
        )[0])

    def test_precision_dispatch_precedes_legacy_sliding_window(self):
        start = self.source.index("void t2w_thread_func_python(")
        end = self.source.index("void t2w_thread_func_cpp(", start)
        body = self.source[start:end]

        oneshot = body.index("process_python_t2w_tokens_oneshot(")
        streaming_loop = body.index("// Process windows using sliding window")
        self.assertLess(oneshot, streaming_loop)
        self.assertIn("oneshot_token_buffer.insert", body)
        self.assertIn("sample_rate, true", body)
        self.assertIn("std::vector<int32_t> token_buffer = {4218, 4218, 4218};", body)
        precision = body[oneshot:streaming_loop]
        self.assertLess(precision.index("read_wav_pcm16_data("),
                        precision.index("audio_output_cb("))
        self.assertLess(precision.index("audio_duration <= 0.0"),
                        precision.index("audio_output_cb("))
        self.assertLess(precision.index('"/generation_done.flag"'),
                        precision.index("audio_output_cb("))

    def test_precision_break_does_not_reset_stateless_oneshot_service(self):
        start = self.source.index("void t2w_thread_func_python(")
        end = self.source.index("void t2w_thread_func_cpp(", start)
        body = self.source[start:end]
        break_start = body.index("if (ctx_omni->break_event.load()) {")
        wait_start = body.index("std::unique_lock<std::mutex> lock(mtx);", break_start)
        break_body = body[break_start:wait_start]

        self.assertIn("if (nonstream_precision) {", break_body)
        self.assertIn("break_event.acknowledge(DuplexBreakStage::T2W", break_body)
        self.assertIn("else if (reset_python_t2w_cache(ctx_omni))", break_body)


if __name__ == "__main__":
    unittest.main()
