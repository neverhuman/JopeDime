# H114 — W6 cross-family outcome transfer on new families: CPU preparation

Status: **CPU preparation complete: preregistration, panel manifest, harness and a 16-fit real-data dry run. No H114
result.** The 608-fit panel run waits for this preregistration to be on main.

## What this delivers

1. **[PREREG.md](PREREG.md).**
   - **Primaries:** does a context-conditioned proposal scorer, trained on 2,306 donor outcomes from the H67 roster,
     choose better new programs on **38 families outside the roster** than the generic proposal scorer (H110 recipe)
     and current raw-lm12 retrieval? Bonferroni m = 2.
   - **Advance rule:** it also requires the treatment to beat a context-shuffled placebo, and the gain to hold on
     programs whose templates no donor has seen.
   - **Kill and withhold:** these follow W6's PROPOSALS entry. The design sits inside W6's 1,200-fit ceiling.
2. **[panel_manifest.json](panel_manifest.json).** It lists 38 contexts in 38 families, none in the roster, spread
   over W1 folds 0–4 (7/10/9/7/5). It records 0 cross-family table and dataset aliases (W1 manifest).
3. **The harness, `run_h114.py`.**
   - **Stages:** `prep`, `dry-run`, `run` and `analyze`.
   - **Pinned import:** it imports the merged [H106](../h106_w9_proposal_scorer_replay/README.md) module (memory
     build, proposer replay, learner target), which pins the merged H81 helpers.
   - **Safety refusals:** it refuses a panel that overlaps the roster or repeats a family, refuses cross-family
     aliases, and refuses inconsistent donor utilities.
   - **Labelling:** it evaluates all 16 proposals per context, so every arm is scored on a fully labelled batch.
4. **A dry run** on one DL-Train context outside the roster and outside the panel, at seed 1001.

## Dry-run receipt (`opus-h114-dryrun-20261031a`, xbabe2, nice 15)

| Check | Result |
|---|---|
| Pins (H106/H81, reveal logs, H110 fits, program-model hash); alias census 0/0 | pass |
| Donor memory | 2,306 rows, 50 contexts; the placebo permutation moves all 50 donor contexts |
| Scorers | `G_SCORER`, `X_SCORER` and `X_SHUF`; `scorers.pkl` sha256 `fd567b9f…` |
| Dry-run evaluation | 16 native fits, 0 failures; 13 / 13 rows complete (6 arms × 2 panels + ceiling) |
| Template census | 9 of 16 proposals have templates no donor has seen (the unseen-template panel qualifies); 2 of 16 programs appear exactly in donor memory |
| Wall | 0:24; 206 MB peak |

The dry-run utilities are integrity evidence only.

## Reproduce

```sh
python3 -m pytest -q research/flywheel_selfplay/h114_w6_xfamily/test_run_h114.py   # 6 synthetic tests
cd research/flywheel_selfplay/h114_w6_xfamily
nice -n 15 python3 -I run_h114.py dry-run --out ~/jope-data/h114/<run_id> --run-id <run_id> \
  --h94-run ~/jope-data/h94/opus-h94-20261031a --h103-run ~/jope-data/h103/opus-h103-20261031a \
  --h110-run ~/jope-data/h110/opus-h110-20261031a --workers 8
# panel run after merge: run_h114.py run ...; then run_h114.py analyze ...
```

## Data scope and evidence

- **Data scope.**
  - DL-Train only.
  - Zero DL-Val, outer-test, sealed or holdout reads.
  - DOPE c51a56d0 is not loaded, and no GPU is used.
- **Evidence.**
  - [dry_run_receipt.json](dry_run_receipt.json) holds the receipt and the S3 object versions.
  - The objects live under
    `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/W6/opus-h114-dryrun-20261031a/`: receipt,
    panel manifest, scorers, rows, census, fits, ledger, logs and sources. Each was written create-only with a
    version-pinned readback hash.
