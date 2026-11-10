#!/bin/bash
# Reference-board environment for TabICL v2 and TabDPT 1.3 (xbabe2, CPU), plus the weight files of the H413 entrants.
# Idempotent. Log: logs/setup_env_icl.log. Added by H413 (2026-11-07).
#   * venv-icl: an isolated venv (the existing venv / venv-gpu are not touched, so their registry pins stay valid).
#   * tabicl and tabdpt are installed from the exact source commits H257 used (ofm-20261008/src), copied first so the
#     original tree is never written to.
#   * weights: TabICL / TabDPT files copied from H257's local copies into $R/weights; Kumo-Tabular small and
#     Mitra v2 fetched into the board's HF cache ($R/hf), where the sdm / AutoGluon loaders read them offline.
#     setup prints every sha256; models.json pins them and `score_reference.py check` refuses drift.
set -euo pipefail
R=/home/ubuntu/jope-data/refboard
OFM=/home/ubuntu/jope-evidence/jopedime_probe_exp/ofm-20261008
mkdir -p $R/tmp $R/hf $R/logs $R/weights $R/src
export TMPDIR=$R/tmp HF_HOME=$R/hf PIP_DISABLE_PIP_VERSION_CHECK=1
want_tabicl=0dbff3ec8fc68c123c87af77b0ea8b25cd2d23f3
want_tabdpt=93670551adba28b186354fab9a1584c56c34aa76
for pair in "tabicl:$want_tabicl" "TabDPT-inference:$want_tabdpt"; do
  d=${pair%%:*}; c=${pair#*:}
  have=$(git -C "$OFM/src/$d" rev-parse HEAD)
  [ "$have" = "$c" ] || { echo "$d is at $have, expected $c" >&2; exit 2; }
  [ -n "$(git -C "$OFM/src/$d" status --porcelain)" ] && { echo "$d has local changes" >&2; exit 2; }
  [ -d "$R/src/$d" ] || git clone -q --no-local "$OFM/src/$d" "$R/src/$d"
  git -C "$R/src/$d" checkout -q --detach "$c"
done
[ -x $R/venv-icl/bin/python ] || python3 -m venv $R/venv-icl
P=$R/venv-icl/bin/pip
$P install -q --index-url https://download.pytorch.org/whl/cpu torch==2.15.0
$P install -q "$R/src/tabicl" "$R/src/TabDPT-inference"
$P freeze > $R/venv-icl.freeze.txt
for f in jingang__TabICL/tabicl-regressor-v2-20260212.ckpt Layer6__TabDPT/tabdpt1_3.safetensors; do
  dst=$R/weights/$(basename "$f")
  [ -f "$dst" ] || install -m 0444 "$OFM/models/$f" "$dst"
done
$R/venv/bin/python - <<'PY'
from huggingface_hub import hf_hub_download
# Kumo: the tag sdm itself loads (sdm/models/kumo/tabular/model.py revision="v1.0.1"), so offline loading finds it
for repo, files, rev in (("nvidia/Kumo-Tabular", ["small/regressor.pt"], "v1.0.1"),
                         ("autogluon/mitra-regressor-2", ["config.json", "model.safetensors"], None)):
    for f in files:
        print(repo, f, rev, hf_hub_download(repo, f, revision=rev), flush=True)
PY
sha256sum $R/weights/tabicl-regressor-v2-20260212.ckpt $R/weights/tabdpt1_3.safetensors \
  $OFM/models/jingang__TabICL/tabicl-regressor-v2-20260212.ckpt $OFM/models/Layer6__TabDPT/tabdpt1_3.safetensors \
  $OFM/models/nvidia__Kumo-Tabular/small/regressor.pt $OFM/models/autogluon__mitra-regressor-2/model.safetensors \
  $(readlink -f $R/hf/hub/models--nvidia--Kumo-Tabular/snapshots/*/small/regressor.pt) \
  $(readlink -f $R/hf/hub/models--autogluon--mitra-regressor-2/snapshots/*/model.safetensors)
$R/venv-icl/bin/python -c "import torch, tabicl, tabdpt; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
echo SETUP_ICL_DONE
