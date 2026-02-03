#!/bin/bash
# GGUF 转换脚本
# 用法: bash run_convert.sh

set -e  # 遇到错误就退出

# 路径配置
MODEL_DIR="tools/omni/convert/MiniCPM-o-4_5"
LLAMACPP_DIR="."
CONVERT_DIR="${LLAMACPP_DIR}/tools/omni/convert"
OUTPUT_DIR="${CONVERT_DIR}/gguf"
PYTHON="python"

echo "============================================"
echo "MiniCPM-o GGUF 转换脚本"
echo "============================================"
echo "源模型目录: ${MODEL_DIR}"
echo "LlamaCpp 目录: ${LLAMACPP_DIR}"
echo "输出目录: ${OUTPUT_DIR}"
echo ""

# 创建输出目录
mkdir -p "${OUTPUT_DIR}"

cd "${LLAMACPP_DIR}"

# Step 1: 运行 surgery.py 提取模型组件
echo "============================================"
echo "Step 1: 提取模型组件 (surgery.py)"
echo "============================================"
CUDA_VISIBLE_DEVICES="" ${PYTHON} tools/omni/convert/surgery.py --model "${MODEL_DIR}" 2>&1 | tee "${CONVERT_DIR}/surgery_latest.log"
echo "surgery.py 完成"
echo ""

# Step 2: 转换 VPM
echo "============================================"
echo "Step 2: 转换 VPM (convert_vpm.py)"
echo "============================================"
${PYTHON} tools/omni/convert/convert_vpm.py -m tools/omni/convert/model/vpm --output-dir "${OUTPUT_DIR}" --minicpmv_version 100045
echo "VPM 转换完成"
echo ""

# Step 3: 转换 APM
echo "============================================"
echo "Step 3: 转换 APM (convert_apm.py)"
echo "============================================"
${PYTHON} tools/omni/convert/convert_apm.py tools/omni/convert/model/apm tools/omni/convert/model/apm/minicpmo.whisper "${OUTPUT_DIR}"
echo "APM 转换完成"
echo ""

# Step 4: 转换 LLM
echo "============================================"
echo "Step 4: 转换 LLM (convert_hf_to_gguf.py)"
echo "============================================"
${PYTHON} convert_hf_to_gguf.py tools/omni/convert/model/llm
mv tools/omni/convert/model/llm/*.gguf "${OUTPUT_DIR}/" 2>/dev/null || true

# 重命名 LLM GGUF 文件
if [ -f "${OUTPUT_DIR}/Model-8.2B-F16.gguf" ]; then
    mv "${OUTPUT_DIR}/Model-8.2B-F16.gguf" "${OUTPUT_DIR}/MiniCPM-o-4_5-F16.gguf"
fi
echo "LLM 转换完成"
echo ""

# Step 5: 量化 LLM (可选)
echo "============================================"
echo "Step 5: 量化 LLM (Q8_0)"
echo "============================================"
if [ -f "${LLAMACPP_DIR}/build/bin/llama-quantize" ]; then
    ${LLAMACPP_DIR}/build/bin/llama-quantize "${OUTPUT_DIR}/MiniCPM-o-4_5-F16.gguf" "${OUTPUT_DIR}/MiniCPM-o-4_5-Q8_0.gguf" Q8_0
    echo "LLM 量化完成"
else
    echo "警告: llama-quantize 未找到，跳过量化步骤"
fi
echo ""

# Step 6: 转换 TTS
echo "============================================"
echo "Step 6: 转换 TTS (convert_hf_to_gguf.py) - F16"
echo "============================================"
echo "y" | ${PYTHON} convert_hf_to_gguf.py tools/omni/convert/model/tts --outtype f16
mv tools/omni/convert/model/tts/*.gguf "${OUTPUT_DIR}/" 2>/dev/null || true

# 重命名 TTS GGUF 文件
for f in "${OUTPUT_DIR}"/Model-*-F16.gguf; do
    if [ -f "$f" ]; then
        new_name=$(echo "$f" | sed 's/Model-/Tts-/')
        mv "$f" "$new_name"
        echo "重命名: $(basename $f) -> $(basename $new_name)"
    fi
done
echo "TTS 转换完成"
echo ""

# Step 7: 转换 Projector
echo "============================================"
echo "Step 7: 转换 Projector (convert_projector.py)"
echo "============================================"
${PYTHON} tools/omni/convert/convert_projector.py --model "${MODEL_DIR}" --output "${OUTPUT_DIR}/projector.gguf"
echo "Projector 转换完成"
echo ""

echo "============================================"
echo "转换完成！"
echo "============================================"
echo "输出文件列表:"
ls -la "${OUTPUT_DIR}"/*.gguf 2>/dev/null || echo "无 GGUF 文件"
echo ""
echo "完成时间: $(date)"
