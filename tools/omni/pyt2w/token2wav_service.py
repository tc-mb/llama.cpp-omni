#!/usr/bin/env python3
"""
Token2Wav 服务进程 - 用于 C++ 调用 Python 的 stepaudio2 Token2wav

协议：通过 stdin 接收 JSON 命令，通过 stdout 返回 JSON 响应

命令格式:
- init: {"cmd": "init", "model_dir": "/path/to/model", "device": "cuda:0", "float16": true, "n_timesteps": 5}
- set_ref_audio: {"cmd": "set_ref_audio", "ref_audio_path": "/path/to/ref.wav"}
- prepare_prompt_bundle: {"cmd": "prepare_prompt_bundle", "ref_audio_path": "/path/to/ref.wav", "output_dir": "/tmp/prompt-bundle"}
- process: {"cmd": "process", "tokens": [1,2,3,...], "last_chunk": false, "output_path": "/path/to/output.wav"}
- reset: {"cmd": "reset"}
- quit: {"cmd": "quit"}

响应格式:
- 成功: {"status": "ok", "message": "...", ...}
- 失败: {"status": "error", "message": "..."}

注意: CUDA_VISIBLE_DEVICES 必须在启动脚本前通过环境变量设置！
"""

import os
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
import hashlib
from pathlib import Path
import stat
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


PROMPT_BUNDLE_SCHEMA_VERSION = 1
PROMPT_SAMPLE_RATE = 16000
PROMPT_MEL_CHANNELS = 80
PROMPT_SPEAKER_DIMENSIONS = 192
PROMPT_PRE_LOOKAHEAD = 3
PROMPT_UP_RATE = 2
MAX_REFERENCE_WAV_BYTES = 64 * 1024 * 1024
MAX_REFERENCE_AUDIO_DURATION_SECONDS = 30


def _as_numpy(value):
    """Convert a torch tensor or array-like value to a detached NumPy array."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def normalize_prompt_bundle_arrays(prompt_tokens, prompt_mel, speaker_embedding):
    """Normalize frontend outputs to the C++ PromptBundle contract."""
    tokens = _as_numpy(prompt_tokens)
    mel = _as_numpy(prompt_mel)
    spk = _as_numpy(speaker_embedding)

    if not np.issubdtype(tokens.dtype, np.integer):
        raise ValueError(f"prompt_tokens must have an integer dtype, got {tokens.dtype}")
    if not np.issubdtype(mel.dtype, np.floating):
        raise ValueError(f"prompt_mel must have a floating dtype, got {mel.dtype}")
    if not np.issubdtype(spk.dtype, np.floating):
        raise ValueError(f"speaker_embedding must have a floating dtype, got {spk.dtype}")

    if tokens.ndim == 2:
        if tokens.shape[0] != 1:
            raise ValueError(f"prompt_tokens must have B=1, got shape {tokens.shape}")
        tokens = tokens[0]
    if tokens.ndim != 1 or tokens.size <= PROMPT_PRE_LOOKAHEAD:
        raise ValueError(f"prompt_tokens must be a non-empty 1D array, got shape {tokens.shape}")

    if mel.ndim == 3:
        if mel.shape[0] != 1:
            raise ValueError(f"prompt_mel must have B=1, got shape {mel.shape}")
        mel = mel[0]
    if mel.ndim != 2:
        raise ValueError(f"prompt_mel must be 2D after squeezing B, got shape {mel.shape}")
    if mel.shape[1] == PROMPT_MEL_CHANNELS:
        pass
    elif mel.shape[0] == PROMPT_MEL_CHANNELS:
        mel = mel.T
    else:
        raise ValueError(
            f"prompt_mel must contain {PROMPT_MEL_CHANNELS} channels, got shape {mel.shape}"
        )

    if spk.ndim == 2:
        if spk.shape[0] != 1:
            raise ValueError(f"speaker_embedding must have B=1, got shape {spk.shape}")
        spk = spk[0]
    if spk.ndim != 1 or spk.size != PROMPT_SPEAKER_DIMENSIONS:
        raise ValueError(
            "speaker_embedding must have shape "
            f"({PROMPT_SPEAKER_DIMENSIONS},), got shape {spk.shape}"
        )

    expected_mel_frames = (tokens.size - PROMPT_PRE_LOOKAHEAD) * PROMPT_UP_RATE
    if mel.shape[0] != expected_mel_frames:
        raise ValueError(
            "prompt shape mismatch: "
            f"T_token={tokens.size}, T_mel={mel.shape[0]}, "
            f"expected T_mel={expected_mel_frames}"
        )
    if not np.isfinite(mel).all():
        raise ValueError("prompt_mel contains non-finite values")
    if not np.isfinite(spk).all():
        raise ValueError("speaker_embedding contains non-finite values")

    return {
        "prompt_tokens": np.ascontiguousarray(tokens, dtype=np.int32),
        "prompt_mel": np.ascontiguousarray(mel, dtype=np.float32),
        "speaker_embedding": np.ascontiguousarray(spk, dtype=np.float32),
    }


def write_prompt_bundle(output_dir, prompt_tokens, prompt_mel, speaker_embedding):
    """Write a validated PromptBundle using the binary format consumed by C++."""
    normalized = normalize_prompt_bundle_arrays(prompt_tokens, prompt_mel, speaker_embedding)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokens = normalized["prompt_tokens"]
    mel = normalized["prompt_mel"]
    spk = normalized["speaker_embedding"]
    tokens.tofile(out_dir / "prompt_tokens_i32.bin")
    mel.tofile(out_dir / "prompt_mel_btc_f32.bin")
    spk.tofile(out_dir / "spk_f32.bin")

    manifest = {
        "schema_version": PROMPT_BUNDLE_SCHEMA_VERSION,
        "sample_rate": PROMPT_SAMPLE_RATE,
        "channels": 1,
        "prompt_token_count": int(tokens.size),
        "prompt_mel_frames": int(mel.shape[0]),
        "mel_channels": PROMPT_MEL_CHANNELS,
        "speaker_dimensions": PROMPT_SPEAKER_DIMENSIONS,
        "prompt_mel_layout": "BTC",
        "dtype": {
            "prompt_tokens": "int32",
            "prompt_mel": "float32",
            "speaker_embedding": "float32",
        },
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {
        "status": "ok",
        "output_dir": str(out_dir),
        "manifest_path": str(manifest_path),
        "prompt_token_count": int(tokens.size),
        "prompt_mel_frames": int(mel.shape[0]),
    }


class Token2WavService:
    def __init__(self):
        self.token2wav = None
        self.stream_cache = None
        self.hift_cache = None
        self.stream_cache_base = None
        self.hift_cache_base = None
        self.default_stream_cache = None
        self.default_hift_cache = None
        self.default_ref_audio_path = None
        self.ref_audio_path = None
        self.initialized = False
        self.device = "cuda:0"

    @staticmethod
    def _validate_reference_audio_file(ref_audio_path):
        path = Path(ref_audio_path)
        try:
            file_status = path.stat()
        except OSError as exc:
            raise ValueError(f"reference audio is not readable: {ref_audio_path}") from exc
        if not stat.S_ISREG(file_status.st_mode):
            raise ValueError(f"reference audio is not a regular file: {ref_audio_path}")
        if file_status.st_size < 12 or file_status.st_size > MAX_REFERENCE_WAV_BYTES:
            raise ValueError(
                "reference audio file size must be between 12 and "
                f"{MAX_REFERENCE_WAV_BYTES} bytes"
            )

        import soundfile as sf

        info = sf.info(str(path))
        if info.frames <= 0 or info.samplerate <= 0:
            raise ValueError("reference audio has invalid frame or sample-rate metadata")
        if info.frames > info.samplerate * MAX_REFERENCE_AUDIO_DURATION_SECONDS:
            raise ValueError(
                "reference audio exceeds "
                f"{MAX_REFERENCE_AUDIO_DURATION_SECONDS} seconds"
            )

    def _load_prompt_audio(self, ref_audio_path: str):
        import librosa

        audio, sample_rate = librosa.load(ref_audio_path, sr=PROMPT_SAMPLE_RATE, mono=True)
        if audio.size == 0:
            raise ValueError("reference audio is empty")
        return np.ascontiguousarray(audio, dtype=np.float32)

    def _model_device(self, torch):
        configured = getattr(self.token2wav, "device", None)
        if configured is not None:
            return torch.device(configured)
        try:
            return next(self.token2wav.flow.parameters()).device
        except (AttributeError, StopIteration):
            return torch.device(self.device)

    @staticmethod
    def _soundfile_torchaudio_load(path, *args, **kwargs):
        import soundfile as sf
        import torch

        audio, sample_rate = sf.read(
            path,
            dtype="float32",
            always_2d=True,
        )
        audio = np.ascontiguousarray(audio.T)
        return torch.from_numpy(audio), sample_rate

    def _set_stream_cache_with_soundfile(self, token2wav, ref_audio_path):
        """Use soundfile for reference WAV decoding inside StepAudio2."""
        import torchaudio

        original_load = torchaudio.load
        torchaudio.load = self._soundfile_torchaudio_load
        try:
            return token2wav.set_stream_cache(ref_audio_path)
        finally:
            torchaudio.load = original_load

    def _extract_prompt_bundle_from_stepaudio2_cache(self, token2wav, ref_audio_path):
        """Extract prompt inputs from the StepAudio2 Token2wav cache API."""
        if not hasattr(token2wav, "set_stream_cache"):
            raise RuntimeError("StepAudio2 Token2wav does not expose set_stream_cache")

        original_stdout = sys.stdout
        sys.stdout = sys.stderr
        try:
            self._set_stream_cache_with_soundfile(token2wav, ref_audio_path)
            cache = getattr(token2wav, "cache", None)
        finally:
            sys.stdout = original_stdout

        if not isinstance(cache, (tuple, list)) or len(cache) < 4:
            raise RuntimeError("StepAudio2 Token2wav did not populate its prompt cache")

        prompt_tokens, _, speaker_embedding, prompt_mel, _ = cache[:5]
        prompt_tokens = _as_numpy(prompt_tokens)
        if prompt_tokens.ndim == 1:
            prompt_tokens = np.concatenate(
                [prompt_tokens, np.full(3, 4218, dtype=prompt_tokens.dtype)]
            )
        elif prompt_tokens.ndim == 2 and prompt_tokens.shape[0] == 1:
            prompt_tokens = np.concatenate(
                [prompt_tokens, np.full((1, 3), 4218, dtype=prompt_tokens.dtype)],
                axis=1,
            )
        else:
            raise ValueError(
                "StepAudio2 prompt tokens must have shape (T,) or (1, T), "
                f"got {prompt_tokens.shape}"
            )

        return prompt_tokens, prompt_mel, speaker_embedding
        
    def init(self, model_dir: str, device: str = "cuda:0", float16: bool = True, n_timesteps: int = 5):
        """初始化 Token2Wav 模型"""
        try:
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
                
                from stepaudio2 import Token2wav
                self.token2wav = Token2wav(model_dir, float16=float16, n_timesteps=n_timesteps)
                
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
                
                log("Token2Wav 初始化成功")
            finally:
                # 恢复原始 stdout
                sys.stdout = original_stdout
            
            return {"status": "ok", "message": "Token2Wav initialized"}
            
        except Exception as e:
            log(f"Token2Wav 初始化失败: {e}")
            traceback.print_exc(file=sys.stderr)
            return {"status": "error", "message": str(e)}
    
    def set_ref_audio(self, ref_audio_path: str):
        """设置参考音频，初始化流式缓存"""
        if not self.initialized:
            return {"status": "error", "message": "Token2Wav not initialized"}
        
        try:
            self._validate_reference_audio_file(ref_audio_path)

            # 🔧 临时重定向 stdout 到 stderr，避免库的打印输出干扰 JSON 协议
            import sys
            original_stdout = sys.stdout
            sys.stdout = sys.stderr
            
            try:
                import torch
                
                log(f"设置参考音频: {ref_audio_path}")
                
                if not os.path.exists(ref_audio_path):
                    return {"status": "error", "message": f"Reference audio not found: {ref_audio_path}"}
                
                # 调用 set_stream_cache 设置缓存
                stream_cache, hift_cache = self._set_stream_cache_with_soundfile(
                    self.token2wav,
                    ref_audio_path,
                )
                
                # 深拷贝基础缓存，用于后续重置
                stream_cache_base = self._clone_cache(stream_cache)
                hift_cache_base = self._clone_cache(hift_cache)
                
                log("参考音频设置成功")
                
                # 🔧 Warmup: 用 dummy tokens 跑一次推理，预编译 CUDA kernels
                # 这样首次真正推理就不会有冷启动延迟
                log("开始 warmup (预编译 CUDA kernels)...")
                warmup_start = time.time()
                
                # 使用 audio_bos token (4218) 作为 dummy tokens
                dummy_tokens = [4218, 4218, 4218] + [1000] * 25  # 28 tokens
                
                # 设置缓存
                self.token2wav.stream_cache = self._clone_cache(stream_cache_base)
                self.token2wav.hift_cache_dict = self._clone_cache(hift_cache_base)
                
                # 跑一次推理
                _ = self.token2wav.stream(
                    generated_speech_tokens=dummy_tokens,
                    prompt_wav=ref_audio_path,
                    last_chunk=True,
                    return_waveform=True
                )
                
                # 重置缓存到初始状态
                self.stream_cache = self._clone_cache(stream_cache_base)
                self.hift_cache = self._clone_cache(hift_cache_base)

                warmup_time = time.time() - warmup_start
                log(f"warmup 完成，耗时 {warmup_time*1000:.1f}ms")

                self.stream_cache_base = self._clone_cache(stream_cache_base)
                self.hift_cache_base = self._clone_cache(hift_cache_base)
                if self.default_stream_cache is None:
                    self.default_stream_cache = self._clone_cache(stream_cache_base)
                    self.default_hift_cache = self._clone_cache(hift_cache_base)
                    self.default_ref_audio_path = ref_audio_path
                self.ref_audio_path = ref_audio_path
            finally:
                # 恢复原始 stdout
                sys.stdout = original_stdout
            
            return {"status": "ok", "message": "Reference audio set"}
            
        except Exception as e:
            log(f"设置参考音频失败: {e}")
            traceback.print_exc(file=sys.stderr)
            return {"status": "error", "message": str(e)}

    def prepare_prompt_bundle(self, ref_audio_path: str, output_dir: str):
        """Run the StepAudio2 frontend and write a C++ PromptBundle."""
        if not self.initialized:
            return {"status": "error", "message": "Token2Wav not initialized"}
        if not ref_audio_path or not os.path.exists(ref_audio_path):
            return {"status": "error", "message": f"Reference audio not found: {ref_audio_path}"}
        if not output_dir:
            return {"status": "error", "message": "Prompt bundle output_dir is empty"}

        try:
            self._validate_reference_audio_file(ref_audio_path)
            import torch

            with torch.inference_mode():
                if hasattr(self.token2wav, "frontend"):
                    prompt_audio = self._load_prompt_audio(ref_audio_path)
                    device = self._model_device(torch)
                    prompt_speech_16k = torch.from_numpy(prompt_audio).unsqueeze(0).to(device)
                    speech_tokens = torch.zeros((1, 1), dtype=torch.long, device=device)
                    model_input = self.token2wav.frontend.frontend_token2wav(
                        speech_tokens=speech_tokens,
                        speech_16k=None,
                        prompt_speech_16k=prompt_speech_16k,
                        resample_rate=self.token2wav.sample_rate,
                        prompt_speech=None,
                    )
                    prompt_tokens = model_input["flow_prompt_speech_token"]
                    prompt_mel = model_input["prompt_speech_feat"]
                    speaker_embedding = model_input["flow_embedding"]
                else:
                    (
                        prompt_tokens,
                        prompt_mel,
                        speaker_embedding,
                    ) = self._extract_prompt_bundle_from_stepaudio2_cache(
                        self.token2wav,
                        ref_audio_path,
                    )

                result = write_prompt_bundle(
                    output_dir,
                    prompt_tokens,
                    prompt_mel,
                    speaker_embedding,
                )
            log(
                "PromptBundle prepared: "
                f"tokens={result['prompt_token_count']}, "
                f"mel_frames={result['prompt_mel_frames']}, "
                f"output_dir={result['output_dir']}"
            )
            return result
        except Exception as e:
            log(f"PromptBundle prepare failed: {e}")
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
    
    def process(self, tokens: list, last_chunk: bool, output_path: str):
        """处理 tokens 并生成 WAV 文件"""
        if not self.initialized:
            return {"status": "error", "message": "Token2Wav not initialized"}
        
        if self.stream_cache is None:
            return {"status": "error", "message": "Reference audio not set"}
        
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
                        "num_samples": len(wav_data)
                    }
                else:
                    result = {"status": "ok", "message": "No audio generated", "output_path": None}
            finally:
                # 恢复原始 stdout
                sys.stdout = original_stdout
            
            return result
                
        except Exception as e:
            log(f"处理失败: {e}")
            traceback.print_exc(file=sys.stderr)
            return {"status": "error", "message": str(e)}
    
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
        
        try:
            reset_stream_cache = (
                self.default_stream_cache
                if self.default_stream_cache is not None
                else self.stream_cache_base
            )
            reset_hift_cache = (
                self.default_hift_cache
                if self.default_hift_cache is not None
                else self.hift_cache_base
            )
            if reset_stream_cache is not None:
                self.stream_cache = self._clone_cache(reset_stream_cache)
                self.hift_cache = self._clone_cache(reset_hift_cache)
                self.stream_cache_base = self._clone_cache(self.stream_cache)
                self.hift_cache_base = self._clone_cache(self.hift_cache)
                if self.token2wav is not None:
                    self.token2wav.stream_cache = self._clone_cache(self.stream_cache)
                    self.token2wav.hift_cache_dict = self._clone_cache(self.hift_cache)
                if self.default_ref_audio_path is not None:
                    self.ref_audio_path = self.default_ref_audio_path
                log("流式缓存已重置")
                return {"status": "ok", "message": "Cache reset"}
            else:
                return {"status": "error", "message": "No base cache to reset from"}
        except Exception as e:
            log(f"重置失败: {e}")
            return {"status": "error", "message": str(e)}


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
            
            cmd_type = cmd.get("cmd", "")
            
            # 处理命令
            if cmd_type == "init":
                response = service.init(
                    model_dir=cmd.get("model_dir", ""),
                    device=cmd.get("device", "cuda:0"),
                    float16=cmd.get("float16", True),
                    n_timesteps=cmd.get("n_timesteps", 5)
                )
            elif cmd_type == "set_ref_audio":
                response = service.set_ref_audio(cmd.get("ref_audio_path", ""))
            elif cmd_type == "prepare_prompt_bundle":
                response = service.prepare_prompt_bundle(
                    ref_audio_path=cmd.get("ref_audio_path", ""),
                    output_dir=cmd.get("output_dir", ""),
                )
            elif cmd_type == "process":
                response = service.process(
                    tokens=cmd.get("tokens", []),
                    last_chunk=cmd.get("last_chunk", False),
                    output_path=cmd.get("output_path", "")
                )
            elif cmd_type == "reset":
                response = service.reset()
            elif cmd_type == "quit":
                log("收到退出命令")
                response = {"status": "ok", "message": "Goodbye"}
                print(json.dumps(response), flush=True)
                break
            else:
                response = {"status": "error", "message": f"Unknown command: {cmd_type}"}
            
            # 发送响应
            print(json.dumps(response), flush=True)
            
        except Exception as e:
            log(f"主循环异常: {e}")
            traceback.print_exc(file=sys.stderr)
            response = {"status": "error", "message": str(e)}
            print(json.dumps(response), flush=True)
    
    log("Token2Wav 服务退出")


if __name__ == "__main__":
    main()
