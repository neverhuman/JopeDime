# H234 — LambdaLoss (NDCG-weighted pairwise) for the r882 adapter continuation, xbabe1 CPU

**Status: CLOSED INCOMPLETE. No efficacy verdict.** The single attempt was stopped by its preregistered hard guard
(host MemAvailable < 20 GiB) after 1 of 15 cells and 120 of 1,800 optimizer updates. By the assignment's rule there
is no retry. The hypothesis is untested. No model, checkpoint, DOPE or serving state changed.

## Hypothesis

**H234 (hw_t2220) asks:** at the incumbent's 40-update budget, does an NDCG-weighted pairwise loss, which
concentrates the gradient on pairs that change the top of the ranking, lift DL-Val91 utility over r882 by more than
+0.002? The assignment proposed ListNet top-1. The inherited CTRL loss already contains a ListNet top-1 term (a
within-context softmax KL at temperature 0.2, weight 0.1), so under the assignment's own rule both arms switched to
**LambdaLoss** before any outcome. See [PREREG.md](PREREG.md).

## Setup

- **Prereg.** PR #2066, merged as `86c988c3`, at reviewed head `6b8d87b2` plus a main merge with an identical H234
  folder.
- **Arms.** All use the identical exact H170 CTRL AdamW recipe; only the loss differs.
  - CONTROL: the CTRL loss.
  - **LAMBDAMIX** (primary): CTRL + 0.5 · Lambda.
  - LAMBDA (secondary): Lambda only. Holm m = 2.
- **Machinery.** Merged H232 with its A1/A2 fixes. Fresh per-seed family folds 234141–234143, seeds 234101–234103,
  15 cells planned, 1,800 updates.
- **Decision panel.** DL-Val91: ADVANCE iff the median gain against r882 is > +0.002, the family LCB95 is > 0 and
  Holm p < 0.05.
- **Host.** xbabe1 CPU, taskset cores 44–47, nice 19.
- **Hard stops** (own process group only): RSS ≥ 4 GiB, MemAvailable < 20 GiB, free disk < 150 GB, more than 4
  threads, and the 04:30Z compute stop. Load5 > 60 would only have paused the worker.

## Outcome

- **Smokes.** Pre-merge `smoke-001` and merged-head `smoke-002` both passed (47 tensors, 29,211 parameters, DOPE
  `c51a56d0`, 0 updates).
- **Attempt `attempt-001`.**
  - Admitted at 22:29:49Z: load5 6.8, MemAvailable 28.9 GB, free 322 GB.
  - Native91 r882 passed bit-exact (1.1377941898164525).
  - Cell `A0-234101` completed: 3 arms × 40 updates = 120 updates and 12 snapshots.
  - **Stopped at 22:36:05Z.** Host MemAvailable fell to **19.92 GiB**, below the 20 GiB floor, with swap full (7/7
    GiB) and load5 7.2. The worker's own RSS was 3.14 GB, under its 4 GiB cap, so other tenants on xbabe1 drove the
    pressure (hw_t2200 notes the live flywheel writer).
  - The monitor sent SIGTERM to its own process group only. No other process was signalled, and
    `flywheel_gp_fused` and its run directory were never touched.
  - Unit ended 22:36:20Z: 376.7 s, peak RSS 3.35 GB, 0 load pauses.
- **Outcomes read.** None beyond the Native91 gate. The one completed cell's selection lock and assessment are in
  the archive but are not analysed: 1 of 15 cells cannot support the preregistered 3-seed statistic.
- **Frozen state.** DOPE `c51a56d0`, checked after every update; the DOPE assets were re-hashed OK after the stop.
  0 sealed, holdout or live-loop reads.

## Verdict

**INCOMPLETE: no efficacy verdict.** The LambdaLoss hypothesis is untested.

## Next step

- **A rerun needs a new assignment** (no retry here). It should either run where MemAvailable headroom is reliably
  above 20 GiB, or adopt a pause-on-memory guard as hw_t2200 asked. That guard was not in this preregistered
  monitor.
- **Memory headroom on xbabe1 was thin all evening:** H232's minimum was 20.44 GiB, and here it fell to 19.92 GiB.

## Custody

- **S3 (create-only, version-pinned readback):**
  `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H234/attempt-001/`. It holds the archive of
  the run root, logs, smokes, launch files and claim, plus `s3_manifest.json`.
- **Prereg.** `.../H234/prereg-6b8d87b2/`.
- **Receipts.** `~/jope_plan/ns/receipts/H234_claim.txt`.

**Code.** [runner.py](runner.py), [treatment.py](treatment.py) (LambdaLoss), [analysis.py](analysis.py),
[monitor.py](monitor.py) and [test_h234.py](test_h234.py) (16 data-free tests), executed at `86c988c3`.
