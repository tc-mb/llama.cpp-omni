file(REMOVE_RECURSE "${TEST_ROOT}")
file(MAKE_DIRECTORY
    "${TEST_ROOT}/vision"
    "${TEST_ROOT}/audio"
    "${TEST_ROOT}/tts"
    "${TEST_ROOT}/token2wav-gguf"
)

foreach(relative_path IN ITEMS
    MiniCPM-o-4_5-F16.gguf
    MiniCPM-o-4_5-Q8_0.gguf
    MiniCPM-o-4_5-Q4_K_M.gguf
    vision/MiniCPM-o-4_5-vision-F16.gguf
    audio/MiniCPM-o-4_5-audio-F16.gguf
    tts/MiniCPM-o-4_5-tts-F16.gguf
    tts/MiniCPM-o-4_5-projector-F16.gguf
    token2wav-gguf/encoder.gguf
    token2wav-gguf/flow_matching.gguf
    token2wav-gguf/flow_extra.gguf
    token2wav-gguf/hifigan2.gguf
    token2wav-gguf/prompt_cache.gguf
)
    file(WRITE "${TEST_ROOT}/${relative_path}" "test")
endforeach()

# Keep this smoke fixture backend-portable; primary resolves to the first
# visible accelerator on CUDA, Metal, and other GGML backends.
file(WRITE "${TEST_ROOT}/omni-runtime-profile.json" [=[
{
  "schema_version": 1,
  "profile": "auto",
  "llm": {
    "model": "MiniCPM-o-4_5-F16.gguf",
    "quantization": "F16",
    "device": "primary",
    "n_gpu_layers": -2
  },
  "vision": {
    "model": "vision/MiniCPM-o-4_5-vision-F16.gguf",
    "device": "primary"
  },
  "audio": {
    "model": "audio/MiniCPM-o-4_5-audio-F16.gguf",
    "device": "primary"
  },
  "tts": {
    "model": "tts/MiniCPM-o-4_5-tts-F16.gguf",
    "device": "primary",
    "gpu_layers": -1
  },
  "projector": {
    "model": "tts/MiniCPM-o-4_5-projector-F16.gguf",
    "device": "primary"
  },
  "token2wav": {
    "model_dir": "token2wav-gguf",
    "device": "primary",
    "threads": 8
  },
  "runtime": {
    "n_ctx": 8192,
    "duplex": true,
    "async": true,
    "vpm_batch_encode": true
  }
}
]=])

execute_process(
    COMMAND "${OMNI_SERVER}"
        --profile auto
        --model-dir "${TEST_ROOT}"
        --print-effective-config
    RESULT_VARIABLE result
    OUTPUT_VARIABLE stdout
    ERROR_VARIABLE stderr
    TIMEOUT 20
)

if(NOT result EQUAL 0)
    message(FATAL_ERROR "llama-omni-server profile resolution failed (${result})\nstdout:\n${stdout}\nstderr:\n${stderr}")
endif()

if(NOT stdout MATCHES "profile=auto")
    message(FATAL_ERROR "effective config did not contain profile=auto\nstdout:\n${stdout}")
endif()
if(NOT stdout MATCHES "resolved_profile=static_config")
    message(FATAL_ERROR "effective config did not identify the static profile config\nstdout:\n${stdout}")
endif()
if(NOT stdout MATCHES "token2wav_model_dir=${TEST_ROOT}/token2wav-gguf")
    message(FATAL_ERROR "effective config did not contain the Token2Wav model directory\nstdout:\n${stdout}")
endif()
if(NOT stdout MATCHES "placement_plan_count=6")
    message(FATAL_ERROR "effective config did not contain all six module placements\nstdout:\n${stdout}")
endif()
if(NOT stdout MATCHES "placement=llm,")
    message(FATAL_ERROR "effective config did not contain the LLM placement\nstdout:\n${stdout}")
endif()
if(NOT stdout MATCHES "placement=token2wav,")
    message(FATAL_ERROR "effective config did not contain the Token2Wav placement\nstdout:\n${stdout}")
endif()

file(REMOVE "${TEST_ROOT}/omni-runtime-profile.json")
execute_process(
    COMMAND "${OMNI_SERVER}"
        --profile auto
        --model-dir "${TEST_ROOT}"
        --print-effective-config
    RESULT_VARIABLE missing_config_result
    OUTPUT_VARIABLE missing_config_stdout
    ERROR_VARIABLE missing_config_stderr
    TIMEOUT 20
)
if(missing_config_result EQUAL 0)
    message(FATAL_ERROR "missing profile config was accepted\nstdout:\n${missing_config_stdout}\nstderr:\n${missing_config_stderr}")
endif()
if(NOT missing_config_stderr MATCHES "profile config file")
    message(FATAL_ERROR "missing profile config error was not actionable\nstderr:\n${missing_config_stderr}")
endif()

execute_process(
    COMMAND "${OMNI_SERVER}"
        --profile auto
        --model-dir "${TEST_ROOT}"
        --model "${TEST_ROOT}/missing-explicit-model.gguf"
        --print-effective-config
    RESULT_VARIABLE missing_result
    OUTPUT_VARIABLE missing_stdout
    ERROR_VARIABLE missing_stderr
    TIMEOUT 20
)

file(REMOVE_RECURSE "${TEST_ROOT}")

if(missing_result EQUAL 0)
    message(FATAL_ERROR "a missing explicit LLM model was accepted\nstdout:\n${missing_stdout}\nstderr:\n${missing_stderr}")
endif()
if(NOT missing_stderr MATCHES "missing explicit LLM model")
    message(FATAL_ERROR "missing explicit LLM model error was not actionable\nstderr:\n${missing_stderr}")
endif()
