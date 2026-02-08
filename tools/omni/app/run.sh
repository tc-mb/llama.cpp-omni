#!/usr/bin/env bash
# =============================================================================
# MiniCPM-o macOS App - 一键启动脚本
#
# 目录结构:
#   llama.cpp-omni/                ← REPO_ROOT
#   ├── .venv/base/                ← Python venv
#   ├── build/bin/omni_engine.so   ← pybind11 模块
#   └── tools/omni/
#       ├── app/                   ← 本脚本所在
#       │   └── server.py
#       └── models/                ← GGUF/CoreML 模型
#
# 用法：
#   cd llama.cpp-omni && bash tools/omni/app/run.sh
#   bash tools/omni/app/run.sh --duplex
#   bash tools/omni/app/run.sh --simplex --port 9060
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# app/ -> omni/ -> tools/ -> llama.cpp-omni/
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
VENV_DIR="$REPO_ROOT/.venv/base"
PYTHON="$VENV_DIR/bin/python"
PIP="$VENV_DIR/bin/pip"
PORT="${PORT:-9060}"

# 解析命令行参数中的 --port
EXTRA_ARGS=("$@")
for i in "${!EXTRA_ARGS[@]}"; do
  if [[ "${EXTRA_ARGS[$i]}" == "--port" ]] && [[ $((i+1)) -lt ${#EXTRA_ARGS[@]} ]]; then
    PORT="${EXTRA_ARGS[$((i+1))]}"
  fi
done

echo ""
echo "============================================"
echo "  MiniCPM-o macOS App Launcher"
echo "============================================"
echo "  REPO_ROOT: $REPO_ROOT"
echo ""

# ---------- 1. 创建 venv ----------
if [ ! -f "$PYTHON" ]; then
  echo "[1/3] Creating Python virtual environment..."
  mkdir -p "$(dirname "$VENV_DIR")"
  python3 -m venv "$VENV_DIR"
  echo "  -> $VENV_DIR"
else
  echo "[1/3] Virtual environment found: $VENV_DIR"
fi

# ---------- 2. 安装依赖 ----------
echo "[2/3] Installing dependencies..."
"$PIP" install --quiet --upgrade pip
"$PIP" install --quiet pybind11 numpy fastapi uvicorn
echo "  -> Dependencies OK"

# ---------- 3. 启动服务 ----------
echo "[3/3] Starting server..."
echo ""

# 2秒后自动打开浏览器
(
  sleep 3
  if command -v open &>/dev/null; then
    open "http://localhost:$PORT"
  elif command -v xdg-open &>/dev/null; then
    xdg-open "http://localhost:$PORT"
  fi
) &

cd "$REPO_ROOT" && PYTHONPATH=. exec "$PYTHON" tools/omni/app/server.py --port "$PORT" "$@"
