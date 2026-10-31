# H106 — proposal-distribution scorer and W9 memory-source replay: CPU preparation

Status: **CPU preparation complete: preregistration, harness and real-data dry run. No H106 result.** The prospective
roster run (2,400 native fits) waits for this preregistration to be on main.

## What this delivers

1. **[PREREG.md](PREREG.md).** Two primaries on fresh proposals of the H67 roster, Bonferroni m = 2:
   - **A** asks whether a scorer trained on logged reveals of proposer programs beats random4. H94 and H103 showed
     that cached-pool priors do not.
   - **B (W9, non-neural learner class)** asks whether memory acquired by the program prior makes a better common
     learner than uniformly acquired memory, at matched rows.

   The neural W9 cells stay with H75. The GPU queue (H57 → Fused D → W1) is unchanged.
2. **The harness, `run_h106.py`.**
   - **Stages:** `prep`, `dry-run`, `run` and `analyze`.
   - **Pinned imports:** the merged [H81](../h81_w4_prospective_2plus2/README.md) helpers (pins, inputs, native fit,
     atomic ledger and fit store), pinned by full sha256.
   - **Program recovery:** logged programs are rebuilt by replaying the proposer streams that produced them.
   - **Labelling:** all 16 proposals per context-seed are evaluated, so every arm is scored on a fully labelled batch.
     This avoids the restricted-subset problem raised in the #983 review.
   - **Completeness:** H96's completeness gate applies.
3. **A dry run on real data.** It used three DL-Train contexts from families outside the roster, at seed 801.

## Dry-run receipt (`opus-h106-dryrun-20261031a`, xbabe2, nice 15)

| Check | Result |
|---|---|
| Pins: H50 packet, W1 folds and features, evaluator, proposer, Elo; H81, program-model and reveal-log hashes | pass |
| Stream replay recovers every logged H94/H103 program | **pass**. A single miss would have stopped the build. |
| Memory rows, de-duplicated | UNIFORM 546, PRIOR 601, ADAPTIVE 439, CONTEXT 437; 50 contexts |
| Matched training rows per fold (four guided sources) | 338 / 351 / 361 / 312 / 385 (folds 0–4) |
| Scorers fitted | 5 memories × 5 folds; `scorers.pkl` sha256 `c12c4c78…` |
| Dry-run evaluation | 3 contexts × 16 proposals: **48 native fits, 0 failures, 0 timeouts**; 24 / 24 arm rows complete |
| Wall | 1:21 including memory replay and fitting; 170 MB peak |

The dry-run utilities are integrity evidence only and are never reported as a comparison.

**Cost projection.** At 1.17 s of native wall per fit, the 2,400-fit roster run needs about 47 CPU-minutes, or about
4 minutes of wall time with 12 workers.

## Next step

Once this preregistration is merged, run `run_h106.py run` and then `analyze` on the roster (CPU only, no lock).
Results go in a follow-up H folder, with S3 under `W9/`.

## Reproduce

```sh
python3 -m pytest -q research/flywheel_selfplay/h106_w9_proposal_scorer_replay/test_run_h106.py   # 6 synthetic tests
cd research/flywheel_selfplay/h106_w9_proposal_scorer_replay
nice -n 15 python3 -I run_h106.py dry-run --out ~/jope-data/h106/<run_id> --run-id <run_id> \
  --h94-run ~/jope-data/h94/opus-h94-20261031a --h103-run ~/jope-data/h103/opus-h103-20261031a \
  --prog-models <H94 models_prog.pkl, sha256 5938a354...> --workers 8
```

The reveal logs and the program-only models are custodied under
`s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/`:

- `W6/opus-h94-20261031a/rows.jsonl.gz`
- `W4/opus-h103-20261031a/rows_w4.jsonl.gz`
- `W6/opus-h94-20261031a/models_prog.pkl`

The runner pins the uncompressed logs by sha256.

## Data scope and evidence

- **Data scope.**
  - DL-Train only: logged native reveals, the admitted H50 pools, W1 train-only features and the H58 DL-Train
    champion bank.
  - Zero DL-Val, outer-test, sealed or holdout reads.
  - DOPE c51a56d0 is not loaded, and no GPU is used.
- **Evidence.**
  - [dry_run_receipt.json](dry_run_receipt.json) holds the receipt and the S3 object versions.
  - The objects live under
    `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/W9/opus-h106-dryrun-20261031a/`: receipt,
    memory, scorers, rows, fits, ledger, logs and sources. Each was written create-only with a version-pinned
    readback hash.
