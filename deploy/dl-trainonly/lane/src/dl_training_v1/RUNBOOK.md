# Runbook

Every command (PLAN §2):
```
systemd-run --user --scope --slice=jopedime-dltrain.slice -p CPUQuota=600% -p MemoryMax=12G -p MemorySwapMax=0 -p TasksMax=256 \
  nice -n 10 env -C "$SRC" PYTHONPATH="$SRC" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 TMPDIR="$RUN/tmp" \
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 AWS_PROFILE=veox-jepson-prod AWS_REGION=us-east-2 "$PY" -m dltrain.<module> ...
```
Overlapping DL units split the cap (300%/6G each). P0: `$PY -m unittest discover -s tests -t . -v`, `$PY -m dltrain.s3io whoami|lifecycle|probe ...`.
