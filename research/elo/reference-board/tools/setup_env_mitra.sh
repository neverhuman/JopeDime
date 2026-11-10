#!/bin/bash
# Reference-board environment for Mitra v2 (xbabe2, CPU): AutoGluon 1.5.0 with its "mitra" extra, in an isolated venv.
# The system python's AutoGluon (used by autogluon_300s.v1) lacks the extra's dependencies (loguru, einx, ...); it is
# left untouched. Idempotent. Log: logs/setup_env_mitra.log. Added by H413 (2026-11-07).
set -euo pipefail
R=/home/ubuntu/jope-data/refboard
mkdir -p $R/tmp $R/logs
export TMPDIR=$R/tmp HF_HOME=$R/hf PIP_DISABLE_PIP_VERSION_CHECK=1
[ -x $R/venv-mitra/bin/python ] || python3 -m venv $R/venv-mitra
P=$R/venv-mitra/bin/pip
$P install -q --index-url https://download.pytorch.org/whl/cpu torch==2.7.0
$P install -q "autogluon.tabular[mitra]==1.5.0"
$P freeze > $R/venv-mitra.freeze.txt
$R/venv-mitra/bin/python -c "import torch, loguru, einx; from autogluon.tabular.models.mitra.sklearn_interface import MitraRegressor; import importlib.metadata as m; print('torch', torch.__version__, 'autogluon.tabular', m.version('autogluon.tabular'), 'cuda', torch.cuda.is_available())"
echo SETUP_MITRA_DONE
