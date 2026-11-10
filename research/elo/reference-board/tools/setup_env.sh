#!/bin/bash
# Reference-board foundation-model environment on xbabe2 (CPU). Idempotent. Log: logs/setup_env.log
set -euo pipefail
R=/home/ubuntu/jope-data/refboard
export TMPDIR=$R/tmp HF_HOME=$R/hf PIP_DISABLE_PIP_VERSION_CHECK=1
[ -x $R/venv/bin/python ] || python3 -m venv $R/venv
P=$R/venv/bin/pip
$P install -q --index-url https://download.pytorch.org/whl/cpu torch
$P install -q numpy pandas scikit-learn tabpfn tabfm==1.0.1 huggingface_hub safetensors
$P install -q "structured-data-models @ git+https://github.com/NVIDIA/structured-data-models.git@50ff254e8044bf1a1b1c53cf6ba1b8eea1b34740"
$P freeze > $R/venv.freeze.txt
$R/venv/bin/python - <<'PY'
from huggingface_hub import hf_hub_download
for repo, files in (("Prior-Labs/tabpfn_3_5", ["tabpfn-v3.5-20260909.safetensors", "config.json", "LICENSE"]),
                    ("google/tabfm-1.0.0-pytorch", ["regression/model.safetensors", "regression/config.json", "config.json", "LICENSE"]),
                    ("nvidia/Kumo-Tabular", ["large/regressor.pt", "config.json"])):
    for f in files:
        try:
            print(repo, f, hf_hub_download(repo, f), flush=True)
        except Exception as e:
            print("FAILED", repo, f, type(e).__name__, str(e)[:200], flush=True)
PY
echo SETUP_DONE
