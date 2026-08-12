#!/usr/bin/env python3
"""
测试 Token2Wav 服务的脚本
"""

import os
import sys
import json
import random
import subprocess
import tempfile
import time
import types
import unittest
from unittest import mock

import numpy as np

# 配置 - 使用相对路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from token2wav_service import Token2WavService, dispatch_command

MODEL_DIR = os.path.join(SCRIPT_DIR, "token2wav")
REF_AUDIO = os.path.join(SCRIPT_DIR, "../convert/gguf/token2wav-gguf/haitian_ref_audio/haitian_ref_audio.wav")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "test_output")
DEVICE = "cuda:0"  # 默认使用 GPU 0

# 测试 tokens（模拟从 TTS 生成的 audio tokens）
# 这是一个简单的测试，实际 tokens 应该从 LLM 生成
TEST_TOKENS = [4218, 4218, 4218] + [1000 + i for i in range(25)]  # 3 prefix + 25 tokens
TEST_SEED = 42


class _FakeStreamToken2Wav:
    def __init__(self):
        self._core = None
        self.stream_cache = {"chunks": 0}
        self.hift_cache_dict = {"chunks": 0}
        self.draws = []

    def set_stream_cache(self, _ref_audio_path):
        self.stream_cache = {"chunks": 0}
        self.hift_cache_dict = {"chunks": 0}
        return self.stream_cache, self.hift_cache_dict

    def stream(self, **_kwargs):
        draw = (random.random(), float(np.random.random()))
        self.draws.append(draw)
        self.stream_cache["chunks"] += 1
        self.hift_cache_dict["chunks"] += 1
        return np.asarray([draw[0], draw[1]], dtype=np.float32)


class _FakeWaveform:
    def __init__(self, values):
        self._values = np.asarray(values, dtype=np.float32)

    def detach(self):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._values


class _FakeOneShotCore:
    def __init__(self):
        self.draws = []

    def forward(self, _tokens, _ref_audio_path, return_bytes=False):
        assert return_bytes is False
        draw = (random.random(), float(np.random.random()))
        self.draws.append(draw)
        return _FakeWaveform(draw)


class Token2WavSeedProtocolTests(unittest.TestCase):
    def setUp(self):
        class _FakeTensor:
            pass

        self.fake_torch = types.SimpleNamespace(
            Tensor=_FakeTensor,
            manual_seed=mock.Mock(),
            get_rng_state=mock.Mock(return_value="cpu-rng-state"),
            set_rng_state=mock.Mock(),
            cuda=types.SimpleNamespace(is_available=mock.Mock(return_value=False)),
            npu=types.SimpleNamespace(is_available=mock.Mock(return_value=False)),
        )
        torch_patch = mock.patch.dict(sys.modules, {"torch": self.fake_torch})
        torch_patch.start()
        self.addCleanup(torch_patch.stop)

    def _stream_service(self):
        service = Token2WavService()
        service.initialized = True
        service.ref_audio_path = "/tmp/ref.wav"
        service.token2wav = _FakeStreamToken2Wav()
        service.stream_cache = {"chunks": 0}
        service.hift_cache = {"chunks": 0}
        service.stream_cache_base = {"chunks": 0}
        service.hift_cache_base = {"chunks": 0}
        service._write_wav = mock.Mock()
        return service

    def test_stream_requires_seed_once_and_keeps_rng_continuous(self):
        service = self._stream_service()

        missing = service.process([1], False, "/ignored.wav")
        self.assertEqual(missing["code"], "missing_seed")
        self.assertEqual(service.token2wav.draws, [])

        first = service.process([1], False, "/ignored.wav", seed=17)
        second = service.process([2], False, "/ignored.wav")
        self.assertEqual(first["effective_seed"], 17)
        self.assertEqual(second["effective_seed"], 17)

        expected_rng = random.Random(17)
        self.assertAlmostEqual(service.token2wav.draws[0][0], expected_rng.random())
        self.assertAlmostEqual(service.token2wav.draws[1][0], expected_rng.random())

        mismatch = service.process([3], False, "/ignored.wav", seed=18)
        self.assertEqual(mismatch["code"], "stream_seed_mismatch")
        self.assertEqual(len(service.token2wav.draws), 2)

        final = service.process([3], True, "/ignored.wav", seed=17)
        self.assertEqual(final["effective_seed"], 17)
        self.assertIsNone(service.stream_session_seed)
        self.assertEqual(service.process([4], False, "/ignored.wav")["code"], "missing_seed")

        replay = service.process([4], True, "/ignored.wav", seed=17)
        self.assertEqual(replay["effective_seed"], 17)
        self.assertAlmostEqual(service.token2wav.draws[-1][0], service.token2wav.draws[0][0])

    def test_reset_ends_session_and_requires_a_new_seed(self):
        service = self._stream_service()
        self.assertEqual(service.process([1], False, "/ignored.wav", seed=7)["status"], "ok")
        self.assertEqual(service.reset()["status"], "ok")
        response = service.process([2], False, "/ignored.wav")
        self.assertEqual(response["code"], "missing_seed")

    def test_oneshot_requires_seed_and_replays_same_rng(self):
        service = Token2WavService()
        service.initialized = True
        core = _FakeOneShotCore()
        service.token2wav = types.SimpleNamespace(_core=core)
        service._write_wav = mock.Mock()

        missing = service.process_oneshot([1], "/does/not/matter.wav", "/ignored.wav")
        self.assertEqual(missing["code"], "missing_seed")

        with tempfile.NamedTemporaryFile(suffix=".wav") as ref_audio:
            first = service.process_oneshot([1], ref_audio.name, "/ignored.wav", seed=99)
            second = service.process_oneshot([1], ref_audio.name, "/ignored.wav", seed=99)

        self.assertEqual(first["effective_seed"], 99)
        self.assertEqual(second["effective_seed"], 99)
        self.assertEqual(core.draws[0], core.draws[1])

    def test_warmup_restores_python_and_numpy_rng_states(self):
        service = Token2WavService()
        service.initialized = True
        service.token2wav = _FakeStreamToken2Wav()

        random.seed(303)
        np.random.seed(303)
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        expected = (random.random(), float(np.random.random()))
        random.setstate(python_state)
        np.random.set_state(numpy_state)

        with tempfile.NamedTemporaryFile(suffix=".wav") as ref_audio:
            response = service.set_ref_audio(ref_audio.name)

        self.assertEqual(response["status"], "ok")
        self.assertEqual((random.random(), float(np.random.random())), expected)
        self.assertTrue(service.warmed_up)

    def test_seed_all_covers_cpu_cuda_and_npu_when_available(self):
        fake_cuda = types.SimpleNamespace(
            is_available=mock.Mock(return_value=True),
            manual_seed_all=mock.Mock(),
        )
        fake_npu = types.SimpleNamespace(
            is_available=mock.Mock(return_value=True),
            manual_seed_all=mock.Mock(),
        )
        fake_torch = types.SimpleNamespace(
            manual_seed=mock.Mock(),
            cuda=fake_cuda,
            npu=fake_npu,
        )

        with mock.patch.dict(sys.modules, {"torch": fake_torch}):
            Token2WavService()._seed_all(123)

        fake_torch.manual_seed.assert_called_once_with(123)
        fake_cuda.manual_seed_all.assert_called_once_with(123)
        fake_npu.manual_seed_all.assert_called_once_with(123)

    def test_rpc_dispatch_forwards_seed_without_defaulting(self):
        service = mock.Mock()
        service.init.return_value = {"status": "ok"}
        service.process.return_value = {"status": "ok"}
        service.process_oneshot.return_value = {"status": "ok"}

        dispatch_command(
            service,
            {
                "cmd": "init",
                "model_dir": "m",
                "device": "npu:0",
                "float16": False,
                "n_timesteps": 10,
                "flow_temperature": 0.7,
            },
        )
        service.init.assert_called_once_with(
            model_dir="m",
            device="npu:0",
            float16=False,
            n_timesteps=10,
            flow_temperature=0.7,
        )

        dispatch_command(
            service,
            {"cmd": "process", "tokens": [1], "last_chunk": False, "output_path": "x", "seed": 11},
        )
        service.process.assert_called_once_with(
            tokens=[1], last_chunk=False, output_path="x", seed=11
        )

        dispatch_command(
            service,
            {
                "cmd": "process_oneshot",
                "tokens": [2],
                "ref_audio_path": "r",
                "output_path": "y",
                "seed": 12,
            },
        )
        service.process_oneshot.assert_called_once_with(
            tokens=[2], ref_audio_path="r", output_path="y", seed=12
        )

    def test_invalid_seeds_are_rejected(self):
        service = self._stream_service()
        for seed in (True, -1, 1 << 32, "42"):
            with self.subTest(seed=seed):
                response = service.process([1], False, "/ignored.wav", seed=seed)
                self.assertEqual(response["code"], "invalid_seed")

    def test_flow_temperature_validation_and_decoder_override(self):
        service = Token2WavService()
        self.assertEqual(service._validate_flow_temperature(0.7), 0.7)
        for value in (True, 0.0, -0.1, 2.1, float("inf"), "0.7"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "flow_temperature"):
                    service._validate_flow_temperature(value)

        calls = []

        class _Decoder:
            def forward(self, **kwargs):
                calls.append(("forward", kwargs["temperature"]))

            def forward_chunk(self, **kwargs):
                calls.append(("forward_chunk", kwargs["temperature"]))

        service.flow_temperature = 0.7
        service.token2wav = types.SimpleNamespace(
            flow=types.SimpleNamespace(decoder=_Decoder())
        )
        service._patch_flow_temperature()
        service.token2wav.flow.decoder.forward(temperature=1.0)
        service.token2wav.flow.decoder.forward_chunk(temperature=1.0)
        self.assertEqual(calls, [("forward", 0.7), ("forward_chunk", 0.7)])


def main():
    # 获取脚本目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    service_script = os.path.join(script_dir, "token2wav_service.py")
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"启动 Token2Wav 服务...")
    print(f"模型目录: {MODEL_DIR}")
    print(f"参考音频: {REF_AUDIO}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"设备: {DEVICE}")
    print()
    
    # 启动服务进程
    env = os.environ.copy()
    process = subprocess.Popen(
        [sys.executable, service_script],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        text=True,
        bufsize=1
    )
    
    def send_cmd(cmd):
        """发送命令并获取响应"""
        cmd_json = json.dumps(cmd)
        print(f">>> {cmd_json}")
        process.stdin.write(cmd_json + "\n")
        process.stdin.flush()
        
        response_line = process.stdout.readline()
        response = json.loads(response_line)
        print(f"<<< {json.dumps(response, ensure_ascii=False)}")
        print()
        return response
    
    try:
        # 等待服务就绪
        print("等待服务就绪...")
        ready_line = process.stdout.readline()
        ready = json.loads(ready_line)
        print(f"服务状态: {ready}")
        print()
        
        # 1. 初始化
        print("=" * 50)
        print("1. 初始化 Token2Wav")
        print("=" * 50)
        response = send_cmd({
            "cmd": "init",
            "model_dir": MODEL_DIR,
            "device": DEVICE,
            "float16": True,
            "n_timesteps": 10
        })
        
        if response.get("status") != "ok":
            print(f"初始化失败: {response}")
            return
        
        # 2. 设置参考音频
        print("=" * 50)
        print("2. 设置参考音频")
        print("=" * 50)
        response = send_cmd({
            "cmd": "set_ref_audio",
            "ref_audio_path": REF_AUDIO
        })
        
        if response.get("status") != "ok":
            print(f"设置参考音频失败: {response}")
            return
        
        # 3. 处理 tokens
        print("=" * 50)
        print("3. 处理 tokens")
        print("=" * 50)
        
        # 模拟滑动窗口处理
        for i in range(3):
            tokens = TEST_TOKENS[:]
            is_last = (i == 2)
            output_path = os.path.join(OUTPUT_DIR, f"wav_{i}.wav")
            
            request = {
                "cmd": "process",
                "tokens": tokens,
                "last_chunk": is_last,
                "output_path": output_path
            }
            if i == 0:
                request["seed"] = TEST_SEED
            response = send_cmd(request)
            
            if response.get("status") != "ok":
                print(f"处理失败: {response}")
                break
        
        # 4. 重置缓存
        print("=" * 50)
        print("4. 重置缓存")
        print("=" * 50)
        response = send_cmd({"cmd": "reset"})
        
        # 5. 退出
        print("=" * 50)
        print("5. 退出服务")
        print("=" * 50)
        response = send_cmd({"cmd": "quit"})
        
    finally:
        # 等待进程结束
        process.wait(timeout=5)
        
        # 打印 stderr
        stderr = process.stderr.read()
        if stderr:
            print("=" * 50)
            print("服务日志 (stderr):")
            print("=" * 50)
            print(stderr)


if __name__ == "__main__":
    main()
