#!/bin/bash
# Reference-board GPU environment (xbabe3, RTX 3090) for TabFM. Idempotent. Log: logs/setup_env_gpu.log
set -euo pipefail
R=/home/ubuntu/jope-data/refboard
mkdir -p $R/tmp $R/hf $R/logs $R/weights
export TMPDIR=$R/tmp HF_HOME=$R/hf PIP_DISABLE_PIP_VERSION_CHECK=1
[ -x $R/venv-gpu/bin/python ] || python3 -m venv $R/venv-gpu
P=$R/venv-gpu/bin/pip
$P install -q torch
$P install -q numpy pandas scikit-learn tabfm==1.0.1 huggingface_hub safetensors
$P freeze > $R/venv-gpu.freeze.txt
$R/venv-gpu/bin/python -c "import torch;print('cuda', torch.cuda.is_available(), torch.__version__, torch.version.cuda)"
echo SETUP_GPU_DONE
