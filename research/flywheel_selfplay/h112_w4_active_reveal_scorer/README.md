# H112 — W4 active reveal with the proposal-distribution scorer: CPU preparation

Status: **CPU preparation complete: preregistration, fold manifest, harness and real-data dry run. No H112 result.**
The roster run (≤ 2,400 native fits) waits for this preregistration to be on main.

## Why this shard

W4's original two-plus-two prep and run are done: H96 prep (#964) and [H103](../h103_w4_two_plus_two_scored/README.md)
(#996). H103 was NO_ADVANCE because its cached-pool prior had no signal among new programs, so the adaptive update
had nothing to refine. [H110](../h110_w9_scorer_replay_scored/README.md) supplied a prior that does have signal (+269
Elo over random4). H112 re-tests W4 with it rather than repeating H96.

## What this delivers

1. **[PREREG.md](PREREG.md).**
   - **Primary:** `s_2plus2 − s_top4` under the common gate.
   - **Advance rule:** it also requires the treatment to beat its residual-swap placebo.
   - **Kill rule:** taken from W4's PROPOSALS entry.
   - **Protocol:** fresh seeds 901–903; family-grouped W1 folds.
   - **Calibration:** a level-free residual update with an out-of-fold utility-scale slope.
2. **[fold_manifest.json](fold_manifest.json).** Built from the real S3-custodied W1 fold manifest over the H67
   roster.

   | Fold | Contexts | Families | Training-memory rows in fold |
   |---|---:|---:|---:|
   | 0 | 11 | 10 | 504 |
   | 1 | 9 | 8 | 456 |
   | 2 | 10 | 8 | 460 |
   | 3 | 14 | 12 | 637 |
   | 4 | 6 | 6 | 249 |

   Every family sits inside one fold.
3. **The harness, `run_h112.py`.**
   - **Pinned import:** it imports the merged [H106](../h106_w9_proposal_scorer_replay/README.md) module (memory
     build, proposer replay, learner target), which in turn pins the merged H81 helpers (GP posterior, lengthscale,
     native fit, ledger, fit store).
   - **Memory:** H110's fully labelled fits are added to the memory, with any inconsistent duplicate utility refused.
   - **Outcome access:** only through `reveal()` on the first pair.
   - **Completeness:** H96's completeness gate applies.
4. **A dry run** on three DL-Train contexts from families outside the roster, at seed 901.

## Dry-run receipt (`opus-h112-dryrun-20261031a`, xbabe2, nice 15)

| Check | Result |
|---|---|
| H106/H81 pins, reveal-log hashes and H110 `fits.jsonl` hash | pass |
| Memory build (every logged program recovered; consistent duplicates) | 2,306 rows |
| Priors (5 folds), out-of-fold utility slopes `b_k` | 0.836 / 0.895 / 0.908 / 0.902 / 0.877 |
| Dry-run evaluation | 3 contexts × 16 proposals: **48 native fits, 0 failures, 0 timeouts**; 15 / 15 rows complete (4 arms + ceiling) |
| Wall | 1:05 including memory replay and 25 GBM fits; 184 MB peak |

**Power note.** In all three dry-run contexts, `s_2plus2` revealed the same four programs as `s_top4`. The swap
placebo differed in one. The level-free update can leave the scorer's ranking unchanged when the pair's centred
residual is small relative to `mu`. If that is common on the roster, the primary will tie, and the advance rule
already treats a tie as NO_ADVANCE. The design is not changed here. The dry-run utilities are integrity evidence only.

## Not done here, and why

- **No CPU fan-out to xbabe1/3/4.** The native evaluator is x86-64, so xbabe4 (aarch64) is ineligible. xbabe1 hosts
  the live loop. A 2,400-fit run takes about 10 minutes on xbabe2.
- **No roster outcomes.** The scored run follows the merge of this preregistration.

## Reproduce

```sh
python3 -m pytest -q research/flywheel_selfplay/h112_w4_active_reveal_scorer/test_run_h112.py   # 6 synthetic tests
cd research/flywheel_selfplay/h112_w4_active_reveal_scorer
nice -n 15 python3 -I run_h112.py dry-run --out ~/jope-data/h112/<run_id> --run-id <run_id> \
  --h94-run ~/jope-data/h94/opus-h94-20261031a --h103-run ~/jope-data/h103/opus-h103-20261031a \
  --h110-run ~/jope-data/h110/opus-h110-20261031a --workers 8
# scored run after merge: run_h112.py run ...; then run_h112.py analyze ...
```

## Data scope and evidence

- **Data scope.**
  - DL-Train only.
  - Zero DL-Val, outer-test, sealed or holdout reads.
  - DOPE c51a56d0 is not loaded, and no GPU is used.
- **Evidence.**
  - [dry_run_receipt.json](dry_run_receipt.json) holds the receipt and the S3 object versions.
  - The objects live under
    `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/W4/opus-h112-dryrun-20261031a/`: receipt,
    memory, fold manifest, priors, rows, fits, ledger, logs and sources. Each was written create-only with a
    version-pinned readback hash.
