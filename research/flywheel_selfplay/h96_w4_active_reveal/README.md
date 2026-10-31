# H96 — W4 active reveal (two-plus-two) on new programs: CPU preparation

Status: **CPU preparation complete: preregistration, harness, real-data dry run and GPU queue entry. No W4 result.**
The scored roster run has not been started, so no W4 effect, Elo or verdict exists yet.

## What this delivers

1. **[PREREG.md](PREREG.md).** The W4 design on new programs, using the program-only prior that
   [H94](../h94_w6_prospective_hedge/README.md) found stronger than the context model on new programs.
   - Arms: `prog_top4` (comparator), `prog_2plus2`, a residual-swap placebo and `random4`.
   - Primary: `prog_2plus2 − prog_top4` under the common gate, with family Elo, on the frozen H67 roster × seeds
     701–703.
   - The use of H94's outcome to choose the comparator is disclosed.
2. **The harness, `run_w4.py`.** Its stages are `dry-run`, `run` and `analyze`.
   - It imports the merged [H81](../h81_w4_prospective_2plus2/README.md) helpers, pinned to their full sha256: input
     pins, the H79 selection and GP update rule, native fit, the atomic ledger and the fit store.
   - The only outcome access is a `reveal` callback, called on the first pair before the second pair is chosen.
   - A prior-signal guard on cached pools runs before any proposal is fitted.
   - Dry-run rows are tagged and refused by `analyze`.
3. **A CPU dry run on a small real shard.** Three DL-Train contexts were drawn from families outside the roster, so
   the roster's outcomes stay unseen. Seed 701, with the full pipeline and native fits.
4. **[GPU_QUEUE.md](GPU_QUEUE.md).** The `r882_top4` continuity arm is queued after H57 → Fused D → W1 under
   `qualification.lock`. No GPU was taken and no lock ticket is held.

## Dry-run receipt (`opus-h96-dryrun-20261031a`, xbabe2, nice 15)

| Check | Result |
|---|---|
| Pins (H50 packet, W1 folds/features, evaluator, proposer, Elo) and the H81/model hashes | pass |
| Prior-signal guard (roster cached pools, zero history) | `prog_top4` 1.8187 against `random4` 1.5048 (+0.314): **pass** |
| Dry-run contexts disjoint from the roster's families | 3 contexts, 0 overlap |
| Proposals | 16 per context-seed, 0 shortfalls |
| Arm rows complete | 12 / 12 (3 contexts × 4 arms) |
| Native fits | 22 distinct (7–8 per context-seed, with shared picks reused), 0 failures, 0 timeouts, 9.4 s summed wall |
| Selection behaviour | The GP update changed the second pair in 2 of 3 contexts. The swap placebo chose the same second pair as `prog_2plus2` in 3 of 3. |
| Wall | 12.9 s including the guard; 173 MB peak RSS |

The dry-run utilities are integrity evidence only and are not reported as a comparison.

**Observation for the scored run.** With only two residuals, swapping them can leave the GP ranking unchanged. In
that case the placebo equals the treatment, so `prog_2plus2 − prog_2plus2_swap` may be underpowered. The decision rule
already requires that contrast to be positive, so a tied placebo blocks advancement rather than passing it. The
design is not changed here.

**Cost projection.** At about 7.3 fits per context-seed, the scored run needs about 1,100 fits (cap 1,800). At about
0.43 s of native wall per fit, that is roughly 8 CPU-minutes of fitting, well within the W4 four-CPU-hour budget.

## Next step

1. **Scored CPU run.** Run `run_w4.py run` and then `analyze` on the roster. This is CPU only and needs no lock. The
   results go in a follow-up H folder, with S3 under `W4/`.
2. **r882 inference adapter.** This is a small reviewed follow-up before W1 releases the lock (see GPU_QUEUE.md).

## Reproduce

```sh
python3 -m pytest -q research/flywheel_selfplay/h96_w4_active_reveal/test_run_w4.py   # 6 synthetic tests (merged H81 rule)
cd research/flywheel_selfplay/h96_w4_active_reveal
nice -n 15 python3 -I run_w4.py dry-run --out ~/jope-data/h96/<run_id> --run-id <run_id> \
  --prog-models <H94 models_prog.pkl, sha256 5938a354...> --workers 8
# scored run (not yet executed): run_w4.py run ...; then run_w4.py analyze ...
```

The program-only models are at
`s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/W6/opus-h94-20261031a/models_prog.pkl`.

## Data scope and evidence

- **Data scope.**
  - DL-Train only: admitted H50 pools, W1 train-only features and the H58 DL-Train champion bank.
  - Zero DL-Val, outer-test, sealed or holdout reads.
  - DOPE c51a56d0 is not loaded, and no GPU is used.
- **Evidence.**
  - [dry_run_receipt.json](dry_run_receipt.json) holds the receipt and the S3 object versions.
  - The objects live under
    `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/W4/opus-h96-dryrun-20261031a/`: receipt,
    rows, fits, ledger, logs and sources. Each was written create-only with a version-pinned readback hash.
  - The source sha256 in the receipt (`9e8ca108…`) equals the committed runner.
