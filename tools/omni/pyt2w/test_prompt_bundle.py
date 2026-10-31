#!/usr/bin/env python3

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

# Keep the test runnable both from the repository root and from this directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from token2wav_service import (
    Token2WavService,
    normalize_prompt_bundle_arrays,
    write_prompt_bundle,
)


class _StepAudio2CacheToken2Wav:
    def __init__(self):
        self.cache = (
            np.arange(4, dtype=np.int64).reshape(1, 4),
            np.array([4], dtype=np.int32),
            np.ones((1, 192), dtype=np.float32),
            np.zeros((1, 8, 80), dtype=np.float32),
            np.array([8], dtype=np.int32),
        )
        self.ref_audio_path = None

    def set_stream_cache(self, ref_audio_path):
        self.ref_audio_path = ref_audio_path
        return object(), object()


class PromptBundleContractTest(unittest.TestCase):
    def test_validate_reference_audio_file_rejects_directory(self):
        service = Token2WavService()
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                service._validate_reference_audio_file(tmp)

    def test_validate_reference_audio_file_rejects_oversized_file(self):
        service = Token2WavService()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "oversized.wav")
            with path.open("wb") as output:
                output.truncate(64 * 1024 * 1024 + 1)
            with self.assertRaises(ValueError):
                service._validate_reference_audio_file(path)

    def test_validate_reference_audio_file_rejects_excessive_duration(self):
        service = Token2WavService()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "long.wav")
            path.write_bytes(b"RIFF" + b"\0" * 8)
            fake_soundfile = SimpleNamespace(
                info=lambda value: SimpleNamespace(
                    frames=16000 * 30 + 1,
                    samplerate=16000,
                    format="WAV",
                )
            )
            with patch.dict(sys.modules, {"soundfile": fake_soundfile}):
                with self.assertRaises(ValueError):
                    service._validate_reference_audio_file(path)

    def test_reset_restores_default_voice_after_dynamic_voice_switch(self):
        class FakeToken2Wav:
            def __init__(self):
                self.stream_cache = None
                self.hift_cache_dict = None

            def stream(self, **kwargs):
                return np.zeros(1, dtype=np.float32)

        service = Token2WavService()
        service.initialized = True
        service.token2wav = FakeToken2Wav()
        service._validate_reference_audio_file = lambda path: None
        service._set_stream_cache_with_soundfile = lambda token2wav, path: (
            f"stream:{path}",
            f"hift:{path}",
        )
        service._clone_cache = lambda cache: cache

        with tempfile.TemporaryDirectory() as tmp:
            default_path = str(Path(tmp, "default.wav"))
            dynamic_path = str(Path(tmp, "dynamic.wav"))
            Path(default_path).touch()
            Path(dynamic_path).touch()

            with patch.dict(sys.modules, {"torch": SimpleNamespace()}):
                self.assertEqual(service.set_ref_audio(default_path)["status"], "ok")
                self.assertEqual(service.set_ref_audio(dynamic_path)["status"], "ok")

            self.assertEqual(service.reset()["status"], "ok")
            self.assertEqual(service.ref_audio_path, default_path)
            self.assertEqual(service.stream_cache, f"stream:{default_path}")
            self.assertEqual(service.hift_cache, f"hift:{default_path}")

    def test_set_stream_cache_uses_soundfile_loader_for_torchaudio_compatibility(self):
        loaded = {}
        fake_torchaudio = SimpleNamespace(load=lambda path: (_ for _ in ()).throw(AssertionError))

        class FakeTorch:
            @staticmethod
            def from_numpy(value):
                loaded["waveform"] = value
                return value

        class FakeSoundfile:
            @staticmethod
            def read(path, dtype, always_2d):
                loaded["path"] = path
                loaded["dtype"] = dtype
                loaded["always_2d"] = always_2d
                return np.zeros((4, 1), dtype=np.float32), 16000

        class Token2Wav:
            def set_stream_cache(self, path):
                import torchaudio

                waveform, sample_rate = torchaudio.load(path)
                return waveform, sample_rate

        with patch.dict(
            sys.modules,
            {
                "soundfile": FakeSoundfile,
                "torch": FakeTorch,
                "torchaudio": fake_torchaudio,
            },
        ):
            service = Token2WavService()
            result = service._set_stream_cache_with_soundfile(
                Token2Wav(),
                "/tmp/reference.wav",
            )

        self.assertEqual(result[1], 16000)
        self.assertEqual(loaded["path"], "/tmp/reference.wav")
        self.assertEqual(loaded["dtype"], "float32")
        self.assertTrue(loaded["always_2d"])
        self.assertEqual(loaded["waveform"].shape, (1, 4))

    def test_extract_prompt_bundle_from_stepaudio2_cache_adds_lookahead_tokens(self):
        service = Token2WavService()
        token2wav = _StepAudio2CacheToken2Wav()

        with patch.dict(
            sys.modules,
            {
                "torchaudio": SimpleNamespace(
                    load=lambda path: (_ for _ in ()).throw(AssertionError)
                )
            },
        ):
            prompt_tokens, prompt_mel, speaker_embedding = (
                service._extract_prompt_bundle_from_stepaudio2_cache(
                    token2wav,
                    "/tmp/reference.wav",
                )
            )

        np.testing.assert_array_equal(
            prompt_tokens,
            np.array([[0, 1, 2, 3, 4218, 4218, 4218]], dtype=np.int64),
        )
        self.assertEqual(prompt_mel.shape, (1, 8, 80))
        self.assertEqual(speaker_embedding.shape, (1, 192))
        self.assertEqual(token2wav.ref_audio_path, "/tmp/reference.wav")

    def test_normalize_prompt_bundle_arrays_accepts_btc_and_squeezes_batch(self):
        tokens = np.arange(7, dtype=np.int64).reshape(1, 7)
        mel = np.arange(8 * 80, dtype=np.float16).reshape(1, 8, 80)
        spk = np.arange(192, dtype=np.float16).reshape(1, 192)

        normalized = normalize_prompt_bundle_arrays(tokens, mel, spk)

        self.assertEqual(normalized["prompt_tokens"].shape, (7,))
        self.assertEqual(normalized["prompt_tokens"].dtype, np.dtype(np.int32))
        self.assertEqual(normalized["prompt_mel"].shape, (8, 80))
        self.assertEqual(normalized["prompt_mel"].dtype, np.dtype(np.float32))
        self.assertEqual(normalized["speaker_embedding"].shape, (192,))
        self.assertEqual(normalized["speaker_embedding"].dtype, np.dtype(np.float32))

    def test_normalize_prompt_bundle_arrays_transposes_ct_layout(self):
        tokens = np.arange(7, dtype=np.int32)
        mel_ct = np.arange(80 * 8, dtype=np.float32).reshape(80, 8)
        spk = np.zeros(192, dtype=np.float32)

        normalized = normalize_prompt_bundle_arrays(tokens, mel_ct, spk)

        self.assertEqual(normalized["prompt_mel"].shape, (8, 80))
        np.testing.assert_array_equal(normalized["prompt_mel"][0], mel_ct[:, 0])

    def test_write_prompt_bundle_writes_manifest_and_binary_contract(self):
        tokens = np.arange(7, dtype=np.int32)
        mel = np.zeros((8, 80), dtype=np.float32)
        spk = np.ones(192, dtype=np.float32)

        with tempfile.TemporaryDirectory() as tmp:
            result = write_prompt_bundle(tmp, tokens, mel, spk)

            self.assertEqual(result["status"], "ok")
            manifest = json.loads(Path(tmp, "manifest.json").read_text())
            self.assertEqual(manifest["schema_version"], 1)
            self.assertEqual(manifest["prompt_token_count"], 7)
            self.assertEqual(manifest["prompt_mel_frames"], 8)
            self.assertEqual(
                np.fromfile(Path(tmp, "prompt_tokens_i32.bin"), dtype=np.int32).shape,
                (7,),
            )
            self.assertEqual(
                np.fromfile(Path(tmp, "prompt_mel_btc_f32.bin"), dtype=np.float32).shape,
                (8 * 80,),
            )
            self.assertEqual(
                np.fromfile(Path(tmp, "spk_f32.bin"), dtype=np.float32).shape,
                (192,),
            )

    def test_write_prompt_bundle_rejects_prompt_shape_mismatch(self):
        with self.assertRaises(ValueError):
            write_prompt_bundle(
                tempfile.mkdtemp(),
                np.arange(7, dtype=np.int32),
                np.zeros((7, 80), dtype=np.float32),
                np.zeros(192, dtype=np.float32),
            )

    def test_normalize_prompt_bundle_arrays_rejects_float_prompt_tokens(self):
        with self.assertRaises(ValueError):
            normalize_prompt_bundle_arrays(
                np.arange(7, dtype=np.float32),
                np.zeros((8, 80), dtype=np.float32),
                np.zeros(192, dtype=np.float32),
            )

    def test_normalize_prompt_bundle_arrays_rejects_integer_prompt_mel(self):
        with self.assertRaises(ValueError):
            normalize_prompt_bundle_arrays(
                np.arange(7, dtype=np.int32),
                np.zeros((8, 80), dtype=np.int32),
                np.zeros(192, dtype=np.float32),
            )

    def test_normalize_prompt_bundle_arrays_rejects_integer_speaker_embedding(self):
        with self.assertRaises(ValueError):
            normalize_prompt_bundle_arrays(
                np.arange(7, dtype=np.int32),
                np.zeros((8, 80), dtype=np.float32),
                np.zeros(192, dtype=np.int32),
            )

    def test_normalize_prompt_bundle_arrays_rejects_non_finite_prompt_mel(self):
        with self.assertRaises(ValueError):
            normalize_prompt_bundle_arrays(
                np.arange(7, dtype=np.int32),
                np.full((8, 80), np.nan, dtype=np.float32),
                np.zeros(192, dtype=np.float32),
            )

    def test_normalize_prompt_bundle_arrays_rejects_non_finite_speaker_embedding(self):
        with self.assertRaises(ValueError):
            normalize_prompt_bundle_arrays(
                np.arange(7, dtype=np.int32),
                np.zeros((8, 80), dtype=np.float32),
                np.full(192, np.inf, dtype=np.float32),
            )


if __name__ == "__main__":
    unittest.main()
