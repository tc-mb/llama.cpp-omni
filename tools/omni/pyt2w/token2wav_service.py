#!/usr/bin/env python3
"""
Token2Wav 服务进程 - 用于 C++ 调用 Python 的 stepaudio2 Token2wav

协议：通过 stdin 接收 JSON 命令，通过 stdout 返回 JSON 响应

命令格式:
- init: {"cmd": "init", "model_dir": "/path/to/model", "device": "cuda:0", "float16": true, "n_timesteps": 5}
- set_ref_audio: {"cmd": "set_ref_audio", "ref_audio_path": "/path/to/ref.wav"}
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
    def __init__(self):
        self.token2wav = None
        self.stream_cache = None
        self.hift_cache = None
        self.ref_audio_path = None
        self.initialized = False
        self.device = "cuda:0"
        
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
                
                # 调用 set_stream_cache 设置缓存
                self.stream_cache, self.hift_cache = self.token2wav.set_stream_cache(ref_audio_path)
                
                # 深拷贝基础缓存，用于后续重置
                self.stream_cache_base = self._clone_cache(self.stream_cache)
                self.hift_cache_base = self._clone_cache(self.hift_cache)
                
                log("参考音频设置成功")
                
                # 🔧 Warmup: 用 dummy tokens 跑一次推理，预编译 CUDA kernels
                # 这样首次真正推理就不会有冷启动延迟
                log("开始 warmup (预编译 CUDA kernels)...")
                warmup_start = time.time()
                
                # 使用 audio_bos token (4218) 作为 dummy tokens
                dummy_tokens = [4218, 4218, 4218] + [1000] * 25  # 28 tokens
                
                # 设置缓存
                self.token2wav.stream_cache = self._clone_cache(self.stream_cache_base)
                self.token2wav.hift_cache_dict = self._clone_cache(self.hift_cache_base)
                
                # 跑一次推理
                _ = self.token2wav.stream(
                    generated_speech_tokens=dummy_tokens,
                    prompt_wav=ref_audio_path,
                    last_chunk=True,
                    return_waveform=True
                )
                
                # 重置缓存到初始状态
                self.stream_cache = self._clone_cache(self.stream_cache_base)
                self.hift_cache = self._clone_cache(self.hift_cache_base)
                
                warmup_time = time.time() - warmup_start
                log(f"warmup 完成，耗时 {warmup_time*1000:.1f}ms")
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
