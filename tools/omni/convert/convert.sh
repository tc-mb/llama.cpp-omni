#!/bin/bash
set -e

# 从项目根目录 (llamacpp) 运行此脚本
# 用法: bash tools/omni/convert/convert.sh [模型目录]

MODEL_DIR="${1:-/cache/caitianchi/code/o45/release/MiniCPM-o-4_5-latest}"
OUTPUT_MODEL_DIR="tools/omni/convert/model"
OUTPUT_GGUF_DIR="tools/omni/convert/gguf"

echo "============================================================"
echo "MiniCPM-o 模型转换脚本"
echo "============================================================"
echo "模型目录: ${MODEL_DIR}"
echo "中间文件: ${OUTPUT_MODEL_DIR}"
echo "GGUF 输出: ${OUTPUT_GGUF_DIR}"
echo "============================================================"

# 创建输出目录
mkdir -p ${OUTPUT_GGUF_DIR}

# Step 1: 提取模型组件 (surgery)
echo ""
echo "=== Step 1: 提取模型组件 ==="
python tools/omni/convert/surgery.py --model "${MODEL_DIR}" --output-dir "${OUTPUT_MODEL_DIR}"

# Step 2: 转换 VPM (vision encoder + projector)
echo ""
echo "=== Step 2: 转换 VPM ==="
python tools/omni/convert/convert_vpm.py -m ${OUTPUT_MODEL_DIR}/vpm --output-dir ${OUTPUT_GGUF_DIR} --minicpmv_version 451

# Step 3: 转换 APM (audio encoder + projector)
echo ""
echo "=== Step 3: 转换 APM ==="
python tools/omni/convert/convert_apm.py ${OUTPUT_MODEL_DIR}/apm ${OUTPUT_MODEL_DIR}/apm/minicpmo.whisper ${OUTPUT_GGUF_DIR}

# Step 4: 转换 LLM
echo ""
echo "=== Step 4: 转换 LLM ==="
python convert_hf_to_gguf.py ${OUTPUT_MODEL_DIR}/llm --outtype f16
mv ${OUTPUT_MODEL_DIR}/llm/*.gguf ${OUTPUT_GGUF_DIR}/ 2>/dev/null || true

# 重命名 LLM 文件
if [ -f "${OUTPUT_GGUF_DIR}/Model-8.2B-F16.gguf" ]; then
    mv ${OUTPUT_GGUF_DIR}/Model-8.2B-F16.gguf ${OUTPUT_GGUF_DIR}/Llm-8.2B-F16.gguf
fi

# Step 5: 量化 LLM (可选)
echo ""
echo "=== Step 5: 量化 LLM (Q4_K_M) ==="
if [ -f "${OUTPUT_GGUF_DIR}/Llm-8.2B-F16.gguf" ]; then
    ./build/bin/llama-quantize ${OUTPUT_GGUF_DIR}/Llm-8.2B-F16.gguf ${OUTPUT_GGUF_DIR}/Llm-8.2B-Q4_K_M.gguf Q4_K_M
fi

# Step 6: 转换 TTS (use f32 for better precision)
echo ""
echo "=== Step 6: 转换 TTS ==="
echo "y" | python convert_hf_to_gguf.py ${OUTPUT_MODEL_DIR}/tts --outtype f32
mv ${OUTPUT_MODEL_DIR}/tts/*.gguf ${OUTPUT_GGUF_DIR}/ 2>/dev/null || true

# 重命名 TTS 文件
if [ -f "${OUTPUT_GGUF_DIR}/Model-440M-F32.gguf" ]; then
    mv ${OUTPUT_GGUF_DIR}/Model-440M-F32.gguf ${OUTPUT_GGUF_DIR}/Tts-440M-F32.gguf
fi

echo ""
echo "============================================================"
echo "转换完成！"
echo "============================================================"
echo "GGUF 文件位置: ${OUTPUT_GGUF_DIR}"
ls -lh ${OUTPUT_GGUF_DIR}/
echo ""
echo "Token2Wav 模型需要单独转换，请参考 token2wav 目录的说明"
