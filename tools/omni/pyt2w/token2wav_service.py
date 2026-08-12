#!/usr/bin/env python3
"""
Token2Wav 服务进程 - 用于 C++ 调用 Python 的 stepaudio2 Token2wav

协议：通过 stdin 接收 JSON 命令，通过 stdout 返回 JSON 响应

命令格式:
- init: {"cmd": "init", "model_dir": "/path/to/model", "device": "cuda:0", "float16": true, "n_timesteps": 5}
- set_ref_audio: {"cmd": "set_ref_audio", "ref_audio_path": "/path/to/ref.wav"}
- process（新 stream session 的首个 chunk 必须带 seed）:
  {"cmd": "process", "tokens": [1,2,3,...], "last_chunk": false, "output_path": "/path/to/output.wav", "seed": 42}
- process_oneshot（每次请求必须带 seed）:
  {"cmd": "process_oneshot", "tokens": [1,2,3,...], "ref_audio_path": "/path/to/ref.wav", "output_path": "/path/to/output.wav", "seed": 42}
- reset: {"cmd": "reset"}
- quit: {"cmd": "quit"}

响应格式:
- 成功: {"status": "ok", "message": "...", ...}
- 失败: {"status": "error", "message": "..."}

注意: CUDA_VISIBLE_DEVICES 必须在启动脚本前通过环境变量设置！
"""

import math
import os
import random
import sys

# 🔧 重定向库的 stdout 输出到 stderr，避免干扰 JSON 协议
# 保存原始 stdout 用于 JSON 通信
_original_stdout = sys.stdout
_original_stderr = sys.stderr

# 创建一个新的 stderr-like 对象来捕获库的打印输出
class StderrRedirector:
    def write(self, text):
        _original_stderr.write(text)
    def flush(self):
        _original_stderr.flush()

# 在导入其他库之前，临时重定向 stdout 到 stderr
sys.stdout = StderrRedirector()

import json
import time
import traceback
import numpy as np

# 禁用 tokenizers 的并行处理警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 恢复原始 stdout 用于 JSON 通信
sys.stdout = _original_stdout

def log(msg):
    """输出日志到 stderr（不影响 stdout 的 JSON 协议）"""
    print(f"[T2W-PY] {msg}", file=sys.stderr, flush=True)


class Token2WavService:
    _MAX_SEED = (1 << 32) - 1
    # CosyVoice registers a persistent, non-checkpointed rand_noise buffer
    # while constructing the flow model. Seed model construction explicitly
    # so identical request seeds remain reproducible across service processes.
    _MODEL_INIT_SEED = 0x4D43504D
    _WARMUP_SEED = 0x5EED5EED

    def __init__(self):
        self.token2wav = None
        self.stream_cache = None
        self.hift_cache = None
        self.stream_cache_base = None
        self.hift_cache_base = None
        self.ref_audio_path = None
        self.initialized = False
        self.device = "cuda:0"
        self.warmed_up = False
        self.stream_session_seed = None
        self.flow_temperature = 1.0

    @staticmethod
    def _validate_flow_temperature(value):
        """Validate the flow-matching noise scale used by both T2W paths."""
        if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
            raise ValueError("flow_temperature must be a finite number in (0, 2]")
        value = float(value)
        if not math.isfinite(value) or value <= 0.0 or value > 2.0:
            raise ValueError("flow_temperature must be a finite number in (0, 2]")
        return value

    def _patch_flow_temperature(self):
        """Override stepaudio2's hard-coded temperature=1.0 call sites.

        ``CausalConditionalCFM.forward`` and ``forward_chunk`` already expose
        the parameter, but the public flow wrapper fixes it to 1.0. Wrapping
        the decoder methods keeps one-shot and streaming on the same audited
        value without copying the upstream inference implementation.
        """
        flow = getattr(self.token2wav, "flow", None)
        decoder = getattr(flow, "decoder", None)
        if decoder is None:
            raise RuntimeError("Token2Wav flow decoder is unavailable")

        for method_name in ("forward", "forward_chunk"):
            original = getattr(decoder, method_name, None)
            if not callable(original):
                raise RuntimeError(f"Token2Wav flow decoder has no {method_name} method")

            def with_temperature(*args, _original=original, **kwargs):
                kwargs["temperature"] = self.flow_temperature
                return _original(*args, **kwargs)

            setattr(decoder, method_name, with_temperature)

    @classmethod
    def _validate_seed(cls, seed):
        """Validate the JSON seed contract and return a plain Python int."""
        if seed is None:
            return None, {
                "status": "error",
                "code": "missing_seed",
                "message": "seed is required for a new Token2Wav request/session",
            }
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
            return None, {
                "status": "error",
                "code": "invalid_seed",
                "message": "seed must be an unsigned 32-bit integer",
            }
        seed = int(seed)
        if seed < 0 or seed > cls._MAX_SEED:
            return None, {
                "status": "error",
                "code": "invalid_seed",
                "message": f"seed must be between 0 and {cls._MAX_SEED}",
            }
        return seed, None

    @staticmethod
    def _available_backend(torch_module, name):
        backend = getattr(torch_module, name, None)
        if backend is None:
            return None
        is_available = getattr(backend, "is_available", None)
        try:
            if callable(is_available) and not is_available():
                return None
        except Exception as exc:
            log(f"检查 torch.{name} RNG 可用性失败，跳过该后端: {exc}")
            return None
        return backend

    def _seed_all(self, seed: int):
        """Seed every RNG used by Token2Wav on the available device stack."""
        import torch

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        for backend_name in ("cuda", "npu"):
            backend = self._available_backend(torch, backend_name)
            if backend is None:
                continue
            seed_fn = getattr(backend, "manual_seed_all", None)
            if seed_fn is None:
                seed_fn = getattr(backend, "manual_seed", None)
            if not callable(seed_fn):
                raise RuntimeError(
                    f"torch.{backend_name} is available but exposes no RNG seeding API"
                )
            seed_fn(seed)

    def _capture_rng_state(self):
        """Capture global RNG state so warmup cannot perturb its caller."""
        import torch

        state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "devices": {},
        }
        for backend_name in ("cuda", "npu"):
            backend = self._available_backend(torch, backend_name)
            get_state = getattr(backend, "get_rng_state_all", None) if backend else None
            if callable(get_state):
                try:
                    state["devices"][backend_name] = get_state()
                except Exception as exc:
                    log(f"保存 torch.{backend_name} RNG 状态失败: {exc}")
        return state

    def _restore_rng_state(self, state):
        """Best-effort restoration paired with _capture_rng_state()."""
        import torch

        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch_cpu"])
        for backend_name, backend_state in state["devices"].items():
            backend = self._available_backend(torch, backend_name)
            set_state = getattr(backend, "set_rng_state_all", None) if backend else None
            if callable(set_state):
                try:
                    set_state(backend_state)
                except Exception as exc:
                    log(f"恢复 torch.{backend_name} RNG 状态失败: {exc}")

    def _end_stream_session(self):
        self.stream_session_seed = None

    def _resolve_stream_seed(self, seed):
        """Start a stream RNG once, or validate an already-active session."""
        if self.stream_session_seed is None:
            effective_seed, error = self._validate_seed(seed)
            if error is not None:
                return None, error
            self._seed_all(effective_seed)
            self.stream_session_seed = effective_seed
            return effective_seed, None

        if seed is None:
            return self.stream_session_seed, None

        supplied_seed, error = self._validate_seed(seed)
        if error is not None:
            return None, error
        if supplied_seed != self.stream_session_seed:
            return None, {
                "status": "error",
                "code": "stream_seed_mismatch",
                "message": (
                    f"active stream seed is {self.stream_session_seed}, "
                    f"but request supplied {supplied_seed}; reset before changing seed"
                ),
                "effective_seed": self.stream_session_seed,
            }
        # Repeated seed fields are accepted for migration convenience, but an
        # active session is never reseeded between chunks.
        return self.stream_session_seed, None

    def _set_stream_cache(self, ref_audio_path: str):
        """Initialize streaming state with the official Token2wav padding."""
        import torch

        core = getattr(self.token2wav, "_core", None)
        if core is None:
            return self.token2wav.set_stream_cache(ref_audio_path)

        prompt_cache = core._prepare_prompt(ref_audio_path)
        core.cache[ref_audio_path] = prompt_cache
        prompt_speech_tokens, _, spk_emb, prompt_mels, _ = prompt_cache

        right_pad_speech_tokens = torch.full(
            (prompt_speech_tokens.shape[0], 3),
            4218,
            device=prompt_speech_tokens.device,
            dtype=prompt_speech_tokens.dtype,
        )
        stream_cache = core.flow.setup_cache(
            torch.cat([prompt_speech_tokens, right_pad_speech_tokens], dim=1),
            prompt_mels,
            spk_emb,
            n_timesteps=self.token2wav.n_timesteps,
        )
        hift_cache = {
            "mel": torch.zeros(1, prompt_mels.shape[2], 0, device=prompt_mels.device),
            "source": torch.zeros(1, 1, 0, device=prompt_mels.device),
            "speech": torch.zeros(1, 0, device=prompt_mels.device),
        }
        self.token2wav.stream_cache = stream_cache
        self.token2wav.hift_cache_dict = hift_cache
        return stream_cache, hift_cache
        
    def init(
        self,
        model_dir: str,
        device: str = "cuda:0",
        float16: bool = True,
        n_timesteps: int = 5,
        flow_temperature: float = 1.0,
    ):
        """初始化 Token2Wav 模型"""
        try:
            self.flow_temperature = self._validate_flow_temperature(flow_temperature)
            # 🔧 在导入可能有输出的库之前，临时重定向 stdout 到 stderr
            import sys
            original_stdout = sys.stdout
            sys.stdout = sys.stderr
            
            try:
                # 🔧 设备格式转换: "gpu:0" -> "cuda:0", "gpu" -> "cuda:0"
                if device.startswith("gpu"):
                    if ":" in device:
                        gpu_id = device.split(":")[1]
                        device = f"cuda:{gpu_id}"
                    else:
                        device = "cuda:0"
                
                self.device = device
                
                # 🔧 注意: CUDA_VISIBLE_DEVICES 必须在 C++ fork 子进程时设置
                # 这里的设置已经太晚了（torch 可能已被导入），仅作为日志记录
                cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "not set")
                log(f"初始化 Token2Wav: model_dir={model_dir}, device={device}, float16={float16}, n_timesteps={n_timesteps}")
                log(f"CUDA_VISIBLE_DEVICES={cuda_visible}")
                
                import torch
                log(f"PyTorch CUDA available: {torch.cuda.is_available()}, device_count: {torch.cuda.device_count()}")

                init_rng_state = self._capture_rng_state()
                try:
                    self._seed_all(self._MODEL_INIT_SEED)
                    if device.startswith("npu"):
                        vllm_omni_root = os.environ.get("OMNI_VLLM_OMNI_ROOT", "").strip()
                        if vllm_omni_root and vllm_omni_root not in sys.path:
                            sys.path.insert(0, vllm_omni_root)
                        from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_token2wav import (
                            MiniCPMO45Token2wav,
                        )

                        self.token2wav = MiniCPMO45Token2wav(
                            model_dir,
                            float16=float16,
                            n_timesteps=n_timesteps,
                            device=device,
                        )
                        npu_available = hasattr(torch, "npu") and torch.npu.is_available()
                        log(f"使用 vLLM-Omni NPU Token2Wav, NPU available: {npu_available}")
                    else:
                        from stepaudio2 import Token2wav

                        self.token2wav = Token2wav(model_dir, float16=float16, n_timesteps=n_timesteps)
                finally:
                    self._restore_rng_state(init_rng_state)

                self._patch_flow_temperature()
                log(f"Flow matching temperature={self.flow_temperature}")
                
                # 🔧 修复 float16 模式下的 dtype bug
                # stepaudio2 库的 setup_cache 方法在 float16 模式下会出现输入是 float32 但权重是 float16 的问题
                if float16:
                    original_setup_cache = self.token2wav.flow.setup_cache
                    
                    @torch.inference_mode()
                    def patched_setup_cache(prompt_speech_tokens, prompt_mels, spk, n_timesteps):
                        # 将输入转换为 float16
                        if prompt_mels.dtype != torch.float16:
                            prompt_mels = prompt_mels.half()
                        if spk.dtype != torch.float16:
                            spk = spk.half()
                        return original_setup_cache(prompt_speech_tokens, prompt_mels, spk, n_timesteps)
                    
                    self.token2wav.flow.setup_cache = patched_setup_cache
                    log("已应用 float16 dtype 修复补丁")
                
                self.initialized = True
                self._end_stream_session()
                
                log("Token2Wav 初始化成功")
            finally:
                # 恢复原始 stdout
                sys.stdout = original_stdout
            
            return {
                "status": "ok",
                "message": "Token2Wav initialized",
                "model_init_seed": self._MODEL_INIT_SEED,
                "flow_temperature": self.flow_temperature,
            }
            
        except Exception as e:
            log(f"Token2Wav 初始化失败: {e}")
            traceback.print_exc(file=sys.stderr)
            return {"status": "error", "message": str(e)}
    
    def set_ref_audio(self, ref_audio_path: str):
        """设置参考音频，初始化流式缓存"""
        if not self.initialized:
            return {"status": "error", "message": "Token2Wav not initialized"}

        # A reference change is a stream-session boundary even when validation
        # of the new path subsequently fails.
        self._end_stream_session()
        
        try:
            # 🔧 临时重定向 stdout 到 stderr，避免库的打印输出干扰 JSON 协议
            import sys
            original_stdout = sys.stdout
            sys.stdout = sys.stderr
            
            try:
                import torch
                
                log(f"设置参考音频: {ref_audio_path}")
                
                if not os.path.exists(ref_audio_path):
                    return {"status": "error", "message": f"Reference audio not found: {ref_audio_path}"}
                
                self.ref_audio_path = ref_audio_path

                # The WS server reuses the same temporary filename across
                # sequential sessions. vLLM-Omni caches prompt features by
                # pathname, so invalidate that entry before reading the new
                # file contents or every later request would keep speaker #1.
                prompt_caches = [
                    getattr(self.token2wav, "cache", None),
                    getattr(getattr(self.token2wav, "_core", None), "cache", None),
                ]
                for prompt_cache in prompt_caches:
                    if isinstance(prompt_cache, dict):
                        prompt_cache.pop(ref_audio_path, None)
                
                # The NPU facade used the prompt's first three speech tokens as
                # lookahead.  Official stepaudio2 pads with three 4218 silence
                # tokens instead; preserve that contract across devices.
                self.stream_cache, self.hift_cache = self._set_stream_cache(ref_audio_path)
                
                # 深拷贝基础缓存，用于后续重置
                self.stream_cache_base = self._clone_cache(self.stream_cache)
                self.hift_cache_base = self._clone_cache(self.hift_cache)
                
                log("参考音频设置成功")
                
                # Warmup 只在服务进程首次设置参考音频时执行。后续每个请求只
                # 重建 voice-clone cache，避免把每条 Seed-TTS 的耗时放大一轮。
                if not self.warmed_up:
                    log("开始首次 warmup...")
                    warmup_start = time.time()

                    # Warmup owns a reserved RNG stream. Restore the previous
                    # global states afterwards; a formal stream will still be
                    # explicitly seeded by its first process request.
                    rng_state = self._capture_rng_state()
                    try:
                        self._seed_all(self._WARMUP_SEED)
                        dummy_tokens = [4218, 4218, 4218] + [1000] * 25
                        self.token2wav.stream_cache = self._clone_cache(self.stream_cache_base)
                        self.token2wav.hift_cache_dict = self._clone_cache(self.hift_cache_base)

                        _ = self.token2wav.stream(
                            generated_speech_tokens=dummy_tokens,
                            prompt_wav=ref_audio_path,
                            last_chunk=True,
                            return_waveform=True,
                        )
                        self.warmed_up = True
                    finally:
                        self._restore_rng_state(rng_state)
                        self.token2wav.stream_cache = self._clone_cache(self.stream_cache_base)
                        self.token2wav.hift_cache_dict = self._clone_cache(self.hift_cache_base)

                    warmup_time = time.time() - warmup_start
                    log(f"warmup 完成，耗时 {warmup_time*1000:.1f}ms")

                # 无论是否 warmup，都恢复到当前参考音频的初始 cache。
                self.stream_cache = self._clone_cache(self.stream_cache_base)
                self.hift_cache = self._clone_cache(self.hift_cache_base)
            finally:
                # 恢复原始 stdout
                sys.stdout = original_stdout
            
            return {"status": "ok", "message": "Reference audio set"}
            
        except Exception as e:
            log(f"设置参考音频失败: {e}")
            traceback.print_exc(file=sys.stderr)
            return {"status": "error", "message": str(e)}
    
    def _clone_cache(self, cache):
        """深拷贝缓存"""
        import torch
        if cache is None:
            return None
        if isinstance(cache, dict):
            return {k: self._clone_cache(v) for k, v in cache.items()}
        elif isinstance(cache, torch.Tensor):
            return cache.clone()
        elif isinstance(cache, (list, tuple)):
            return type(cache)(self._clone_cache(v) for v in cache)
        else:
            return cache

    def _trim_stream_cache(self):
        """Apply the official Token2wav streaming-cache retention policy."""
        import torch

        core = getattr(self.token2wav, "_core", None)
        cache = getattr(self.token2wav, "stream_cache", None)
        if core is None or not isinstance(cache, dict):
            return

        prompt_cache = core.cache.get(self.ref_audio_path)
        if prompt_cache is None:
            return
        prompt_mels = prompt_cache[3]
        prompt_frames = prompt_mels.shape[1]

        conformer_cache = cache.get("conformer_att_cache")
        if (
            isinstance(conformer_cache, torch.Tensor)
            and conformer_cache.ndim >= 4
            and conformer_cache.shape[3] > prompt_frames + 100
        ):
            cache["conformer_att_cache"] = torch.cat(
                [
                    conformer_cache[:, :, :, :prompt_frames, ...],
                    conformer_cache[:, :, :, -100:, ...],
                ],
                dim=3,
            )
    
    def process(self, tokens: list, last_chunk: bool, output_path: str, seed=None):
        """处理 tokens 并生成 WAV 文件"""
        if not self.initialized:
            return {"status": "error", "message": "Token2Wav not initialized"}
        
        if self.stream_cache is None:
            return {"status": "error", "message": "Reference audio not set"}

        try:
            effective_seed, seed_error = self._resolve_stream_seed(seed)
        except Exception as e:
            log(f"设置 stream RNG 失败: {e}")
            return {"status": "error", "code": "seed_failed", "message": str(e)}
        if seed_error is not None:
            return seed_error
        
        try:
            # 🔧 临时重定向 stdout 到 stderr
            import sys
            original_stdout = sys.stdout
            sys.stdout = sys.stderr
            
            try:
                import torch
                
                start_time = time.time()
                
                # 设置当前缓存到 token2wav 实例
                self.token2wav.stream_cache = self.stream_cache
                self.token2wav.hift_cache_dict = self.hift_cache
                
                # 调用流式生成
                wav_data = self.token2wav.stream(
                    generated_speech_tokens=tokens,
                    prompt_wav=self.ref_audio_path,
                    last_chunk=last_chunk,
                    return_waveform=True
                )
                
                # 更新缓存
                self._trim_stream_cache()
                self.stream_cache = self.token2wav.stream_cache
                self.hift_cache = self.token2wav.hift_cache_dict
                
                inference_time = time.time() - start_time
                
                # 保存 WAV 文件
                if wav_data is not None and len(wav_data) > 0:
                    # wav_data 是 numpy array，shape: [1, samples] 或 [samples]
                    if len(wav_data.shape) > 1:
                        wav_data = wav_data.squeeze()
                    
                    # 写入 WAV 文件
                    sample_rate = 24000
                    audio_duration = len(wav_data) / sample_rate
                    
                    self._write_wav(output_path, wav_data, sample_rate)
                    
                    log(f"生成 WAV: {output_path} | {audio_duration:.2f}s | {inference_time*1000:.1f}ms | RTF={inference_time/audio_duration:.2f}")
                    
                    result = {
                        "status": "ok",
                        "message": "WAV generated",
                        "output_path": output_path,
                        "audio_duration": audio_duration,
                        "inference_time_ms": inference_time * 1000,
                        "sample_rate": sample_rate,
                        "num_samples": len(wav_data),
                        "effective_seed": effective_seed,
                    }
                else:
                    result = {
                        "status": "ok",
                        "message": "No audio generated",
                        "output_path": None,
                        "effective_seed": effective_seed,
                    }
            finally:
                # 恢复原始 stdout
                sys.stdout = original_stdout

            if last_chunk:
                self._end_stream_session()
            return result
                
        except Exception as e:
            log(f"处理失败: {e}")
            traceback.print_exc(file=sys.stderr)
            # The model/cache/RNG may all have advanced partially. Do not let a
            # caller continue that stream under a misleading deterministic ID.
            self._end_stream_session()
            return {
                "status": "error",
                "message": str(e),
                "effective_seed": effective_seed,
            }

    def process_oneshot(self, tokens: list, ref_audio_path: str, output_path: str, seed=None):
        """Run the model's non-streaming Token2Wav path for HF parity checks."""
        if not self.initialized:
            return {"status": "error", "message": "Token2Wav not initialized"}
        effective_seed, seed_error = self._validate_seed(seed)
        if seed_error is not None:
            return seed_error
        if self.stream_session_seed is not None:
            return {
                "status": "error",
                "code": "stream_session_active",
                "message": "reset or finish the active stream before process_oneshot",
                "effective_seed": self.stream_session_seed,
            }
        if not os.path.isfile(ref_audio_path):
            return {"status": "error", "message": f"Reference audio not found: {ref_audio_path}"}
        if not tokens:
            return {"status": "error", "message": "No audio tokens supplied"}

        rng_state = None
        try:
            start_time = time.time()
            rng_state = self._capture_rng_state()
            self._seed_all(effective_seed)
            core = getattr(self.token2wav, "_core", None)
            if core is not None:
                waveform = core.forward(tokens, ref_audio_path, return_bytes=False)
                wav_data = waveform.detach().float().cpu().numpy()
                sample_rate = 24000
            else:
                import io
                import soundfile as sf

                wav_bytes = self.token2wav(tokens, ref_audio_path)
                wav_data, sample_rate = sf.read(io.BytesIO(wav_bytes), dtype="float32")
            wav_data = np.asarray(wav_data).squeeze()
            if wav_data.ndim > 1:
                wav_data = wav_data.mean(axis=1)
            self._write_wav(output_path, wav_data, int(sample_rate))
            inference_time = time.time() - start_time
            audio_duration = len(wav_data) / int(sample_rate)
            return {
                "status": "ok",
                "message": "One-shot WAV generated",
                "output_path": output_path,
                "audio_duration": audio_duration,
                "inference_time_ms": inference_time * 1000,
                "sample_rate": int(sample_rate),
                "num_samples": len(wav_data),
                "effective_seed": effective_seed,
            }
        except Exception as e:
            log(f"one-shot 处理失败: {e}")
            traceback.print_exc(file=sys.stderr)
            return {
                "status": "error",
                "message": str(e),
                "effective_seed": effective_seed,
            }
        finally:
            if rng_state is not None:
                self._restore_rng_state(rng_state)
    
    def _write_wav(self, path: str, wav_data: np.ndarray, sample_rate: int):
        """写入 WAV 文件"""
        import struct
        
        # 确保目录存在
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # 转换为 16-bit PCM
        wav_data = np.clip(wav_data, -1.0, 1.0)
        pcm_data = (wav_data * 32767.0).astype(np.int16)
        
        # 写入 WAV 文件
        num_channels = 1
        bits_per_sample = 16
        byte_rate = sample_rate * num_channels * (bits_per_sample // 8)
        block_align = num_channels * (bits_per_sample // 8)
        data_size = len(pcm_data) * (bits_per_sample // 8)
        
        with open(path, 'wb') as f:
            # RIFF header
            f.write(b'RIFF')
            f.write(struct.pack('<I', 36 + data_size))
            f.write(b'WAVE')
            
            # fmt chunk
            f.write(b'fmt ')
            f.write(struct.pack('<I', 16))  # chunk size
            f.write(struct.pack('<H', 1))   # audio format (PCM)
            f.write(struct.pack('<H', num_channels))
            f.write(struct.pack('<I', sample_rate))
            f.write(struct.pack('<I', byte_rate))
            f.write(struct.pack('<H', block_align))
            f.write(struct.pack('<H', bits_per_sample))
            
            # data chunk
            f.write(b'data')
            f.write(struct.pack('<I', data_size))
            f.write(pcm_data.tobytes())
    
    def reset(self):
        """重置流式缓存到初始状态"""
        if not self.initialized:
            return {"status": "error", "message": "Token2Wav not initialized"}

        self._end_stream_session()
        
        try:
            if self.stream_cache_base is not None:
                self.stream_cache = self._clone_cache(self.stream_cache_base)
                self.hift_cache = self._clone_cache(self.hift_cache_base)
                log("流式缓存已重置")
                return {"status": "ok", "message": "Cache reset"}
            else:
                return {"status": "error", "message": "No base cache to reset from"}
        except Exception as e:
            log(f"重置失败: {e}")
            return {"status": "error", "message": str(e)}


def dispatch_command(service, cmd):
    """Dispatch one decoded RPC command; kept separate for protocol tests."""
    cmd_type = cmd.get("cmd", "")
    if cmd_type == "init":
        return service.init(
            model_dir=cmd.get("model_dir", ""),
            device=cmd.get("device", "cuda:0"),
            float16=cmd.get("float16", True),
            n_timesteps=cmd.get("n_timesteps", 5),
            flow_temperature=cmd.get("flow_temperature", 1.0),
        )
    if cmd_type == "set_ref_audio":
        return service.set_ref_audio(cmd.get("ref_audio_path", ""))
    if cmd_type == "process":
        return service.process(
            tokens=cmd.get("tokens", []),
            last_chunk=cmd.get("last_chunk", False),
            output_path=cmd.get("output_path", ""),
            seed=cmd.get("seed"),
        )
    if cmd_type == "process_oneshot":
        return service.process_oneshot(
            tokens=cmd.get("tokens", []),
            ref_audio_path=cmd.get("ref_audio_path", ""),
            output_path=cmd.get("output_path", ""),
            seed=cmd.get("seed"),
        )
    if cmd_type == "reset":
        return service.reset()
    if cmd_type == "quit":
        log("收到退出命令")
        return {"status": "ok", "message": "Goodbye"}
    return {"status": "error", "message": f"Unknown command: {cmd_type}"}


def main():
    """主循环：从 stdin 读取命令，处理后写入 stdout"""
    log("Token2Wav 服务启动")

    service = Token2WavService()

    # 发送就绪信号
    print(json.dumps({"status": "ready", "message": "Token2Wav service ready"}), flush=True)

    while True:
        try:
            # 读取一行 JSON 命令
            line = sys.stdin.readline()
            if not line:
                log("stdin 关闭，退出")
                break

            line = line.strip()
            if not line:
                continue

            # 解析命令
            try:
                cmd = json.loads(line)
            except json.JSONDecodeError as e:
                response = {"status": "error", "message": f"Invalid JSON: {e}"}
                print(json.dumps(response), flush=True)
                continue

            response = dispatch_command(service, cmd)
            print(json.dumps(response), flush=True)
            if cmd.get("cmd", "") == "quit":
                break

        except Exception as e:
            log(f"主循环异常: {e}")
            traceback.print_exc(file=sys.stderr)
            response = {"status": "error", "message": str(e)}
            print(json.dumps(response), flush=True)

    log("Token2Wav 服务退出")


if __name__ == "__main__":
    main()
