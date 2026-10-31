# H100: ASAM confirmation and drift control

**CPU preregistration; no H100 timing or outcomes yet.**
[PREREG.md](PREREG.md) was committed before implementation and execution.
The prospective panel uses fresh paired seeds 39–41 and four fixed arms:
H82 CONTROL, ASAM rho 0.5, ASAM rho 1.0, and a lower-LR CONTROL calibrated to
H82 ASAM's adapter drift. All start from offline r882 copies; DOPE stays frozen
and only the 29,211 non-anchor adapter parameters train.

The [historical calibration](calibration.json) uses the saved H82 adapter tensors,
not fresh-seed outcomes. Mean L2 drift was 0.44855066358 for CONTROL and
0.37970946839 for ASAM. Their fixed ratio 0.8465252628604802 gives DRIFT_MATCH
learning rate **8.465252628604803e-5**. This is a one-shot target; report actual
fresh-seed distances and do not retune.

The primary is median paired family utility gain over three seeds. Report the
10,000-draw family-cluster CI, strict LCB95 > 0.001546, Holm across three contrasts,
and the <=10% family-harm gate versus r882. Report per-seed/median Elo versus
CONTROL and r882, train/val gaps, L2 and unrevealed top-four overlap. The prereg
fixes an equivalence test for the drift-control interpretation. No serving
promotion or additional experiment is authorized by this CPU record.

The [guarded CPU dry run](cpu_dry_run.json) opened only H50 source/roster metadata:
1,219 train contexts / 205 families and metadata for 91 dev contexts / 19 families.
An audit hook refused every NPZ/JTT1 open. Zero payloads, optimizer updates,
CUDA initialization or FIFO tickets. This is local preparation, not live proof.

```sh
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 nice -n 19 \
  python3 -m pytest -q research/flywheel_selfplay/h100_asam_confirm/test_h100.py
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 nice -n 19 \
  python3 research/flywheel_selfplay/h100_asam_confirm/runner.py \
  --dry-run --out /home/ubuntu/jope_plan/evidence/H100/new-cpu-dry-run.json
```

After the CPU PR merges, launch `run_wave.py --cpu-pr PR_NUMBER` with the same
thread caps and nice 19. Its exclusive coordinator waits outside FIFO until
H93, H97 and H99 have merged terminal metrics, closed ledgers and no live
coordinator/ticket. It then joins the H100 FIFO tail per cell, releases each
lease, and retains every charge against 2700 seconds. No retries or ledger
reset; a failed attempt stops with missing outcomes unknown. Scratch copies
are isolated per attempt, hash-verified and read-only; source r882 is untouched.
Raw runtime output belongs under `~/jope-data/h100/h100-wave/`, outside Git.
