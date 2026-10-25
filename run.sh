#!/bin/bash
set -euo pipefail
cd /mnt/d/RobotReplanStudy
export HOME=/mnt/d/RobotReplanStudy/.home
export TMPDIR=/mnt/d/RobotReplanStudy/.tmp
export HF_HOME=/mnt/d/RobotReplanStudy/.cache/huggingface
export TORCH_HOME=/mnt/d/RobotReplanStudy/.cache/torch
export XDG_CACHE_HOME=/mnt/d/RobotReplanStudy/.cache
export PIP_CACHE_DIR=/mnt/d/RobotReplanStudy/.cache/pip
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPATH=/mnt/d/RobotReplanStudy
export PYTHONUNBUFFERED=1
PY=/mnt/d/RobotReplanStudy/.conda/bin/python
mkdir -p "$HOME" "$TMPDIR" "$HF_HOME" "$TORCH_HOME" results
"$PY" experiments/phase0.py
"$PY" experiments/demos.py
"$PY" experiments/sweep.py
