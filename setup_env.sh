#!/bin/bash
set -euo pipefail
export HOME=/mnt/d/RobotReplanStudy/.home
export TMPDIR=/mnt/d/RobotReplanStudy/.tmp
export PIP_CACHE_DIR=/mnt/d/RobotReplanStudy/.cache/pip
export CONDA_PKGS_DIRS=/mnt/d/RobotReplanStudy/.conda-pkgs
mkdir -p "$HOME" "$TMPDIR" "$PIP_CACHE_DIR" "$CONDA_PKGS_DIRS"
CONDA=/mnt/d/RobotReplanStudy/miniforge/bin/conda
"$CONDA" create --prefix /mnt/d/RobotReplanStudy/.conda python=3.12 pip -y
PY=/mnt/d/RobotReplanStudy/.conda/bin/python
"$PY" -m pip install --upgrade pip
"$PY" -m pip install "lerobot[smolvla,libero]" imageio imageio-ffmpeg pillow
"$PY" -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
