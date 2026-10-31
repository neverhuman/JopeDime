# H120 — W4 offline acquisition screen (zero native fits)

> **Numbering.** Opened as H119 (#1046). sol_v1_dsc08 had claimed H119 first, so per the hwave overseer this is H120. The
> content is unchanged, and the prep receipts were regenerated under the H120 runner.

Status: **complete. Verdict: CLOSE W4 adaptive reveal at 4 of 16.** No reveal rule beats the scorer's top four on
the 300 fully labelled batches. The best rule, `UCB_2PLUS2`, adds +0.0007 (adjusted LCB −0.0005), and 36 of 44
families tie. The integrity gate reproduced H116 exactly (300/300), the panel is complete and the screen used
**0 native fits**.

## Outcome (44 families, 50 contexts, 6 batches each; completeness gate passed)

Merged runner `run_h120.py` (sha256 `cc5219f5…`, main `9e46eea4c`) ran the `screen` stage exactly as preregistered.
The squash merge of #1046 (`569324b77`) changed the runner only by formatting and by adding fail-closed completeness
checks, with no change to any rule, endpoint or gate.

| Contrast (primary) | Mean [95% CI] | Adj. LCB (.05/3) | Median | Losing > .01 | W/T/L | Gate |
|---|---|---:|---:|---:|---|---|
| `GP_2PLUS2 − S_TOP4` | +0.00008 [−0.00051, +0.00063] | −0.00062 | 0 | 0.0% | 3/39/2 | fail |
| `GP_1PLUS3 − S_TOP4` | −0.0473 [−0.152, +0.009] | −0.194 | 0 | 9.1% | 9/24/11 | fail |
| `UCB_2PLUS2 − S_TOP4` | +0.00069 [−0.00041, +0.00220] | −0.00047 | 0 | 0.0% | 4/36/4 | fail |

| Rule | Family-mean best of four | Family Elo (`S_TOP4` = 1500) | 95% family-bootstrap CI |
|---|---:|---:|---|
| `ORACLE4` (ceiling, not rated) | 2.0452 | — | — |
| `UCB_2PLUS2` | 2.0313 | 1504.7 | [1475.6, 1538.1] |
| `GP_2PLUS2` | 2.0307 | 1519.0 | [1496.3, 1547.1] |
| **`S_TOP4`** | **2.0306** | **1500.0** | anchor |
| `GP_1PLUS3` | 1.9833 | 1467.9 | [1402.4, 1525.4] |
| `random4` | 1.8709 | 1211.8 | [1062.3, 1317.3] |

- Elo is the canonical family-grouped Bradley–Terry tracker (tie 1e-4, prior 0.01, 400 iterations). Most pairs tie:
  `GP_2PLUS2` beats `S_TOP4` in 3 contexts and loses 2 of 50.
- The Elo CIs are **descriptive and were not preregistered**. They come from 1,000 family-cluster resamples (seed
  20261031) of the same tracker, and the point estimates reproduce the run's metrics exactly. Every rule's interval
  contains 1500 except `random4`'s.
- The best-of-four value differs from `S_TOP4`'s in 7 (`GP_2PLUS2`), 11 (`UCB_2PLUS2`) and 43 (`GP_1PLUS3`) of 300
  batches.

## Verdict and reading

- **W4 adaptive reveal at 4 of 16 is CLOSED.** Under the preregistered decision, no rule passes, so no budgeted
  prospective W4 test is requested. The two-plus-two update showed no signal on cached pools (H79), failed
  prospectively twice (H103, H116) and fails offline here.
- **There is little left to gain.** `S_TOP4` already closes 92% of the random-to-oracle gap (1.871 → 2.031 of
  2.045). The rules recover at most 0.0007 of the remaining 0.015.
- **Revealing one program and then trusting the posterior hurts.** `GP_1PLUS3` loses −0.047, with an 11-family loss
  side. A likely reading is that one clipped residual is too little evidence to reorder the other fifteen.
- **The win stays in the prior.** The proposal-distribution scorer's top four beats random by about +288 Elo here,
  consistent with H110, H116 and H118. Future W4 effort belongs in the scorer, not in adaptive reveal.

## Next step

No W4 native fits. The remaining owner-gated question is the live-loop ablation from H118: rank the live proposer's
candidates by the generic proposal scorer instead of raw retrieval or the cached prior.

## Budget disclosure

W4's cumulative cap in PROPOSALS is **≤ 2,400 native fits and four CPU-hours**. W4 has used:

| Run | Native fits |
|---|---:|
| H96 dry run | 22 |
| H103 | 850 |
| H112 dry run | 48 |
| H116 | 1,614 |
| **Total** | **2,534** |

That is **134 fits (5.6%) over the cap**; CPU use is 0.71 h, within four CPU-hours. Cause: H112's preregistration
set a 2,400-fit cap for that run instead of the cumulative remainder (1,528). The error is rd_opus's and is recorded
in `owner_decisions.md`. Any further W4 native fit needs an owner budget call. H120 spends none.

## Hypothesis

Before requesting more budget, test whether any other reveal rule beats the scorer's top four. In
[H116](../h116_w4_scorer_2plus2_scored/README.md), the GP two-plus-two update changed the revealed set in only 11 of
150 units. The screen uses the **300 fully labelled proposal batches** already evaluated by
[H110](../h110_w9_scorer_replay_scored/README.md) (seeds 801–803) and H116 (seeds 901–903). Each rule's `reveal`
returns cached outcomes only for programs the rule itself asked for, and the harness refuses any reveal outside a
rule's own picks.

## Preregistration

| Item | Fixed value |
|---|---|
| Prior | The merged [H112](../h112_w4_active_reveal_scorer/PREREG.md) prior exactly: memory, per-fold scorer and out-of-fold slope `b_k`. Uncertainty `sd` = `b_k` × the standard deviation of the 4 inner-fold models. |
| `S_TOP4` | The prior's top four (anchor, Elo 1500) |
| `GP_2PLUS2` | The H112 rule, unchanged merged code |
| `GP_1PLUS3` | Reveal the top one; its clipped (±1) residual drives the same RBF-GP posterior; reveal the top 3 of the rest |
| `UCB_2PLUS2` | First pair = top two by `mu + sd`; then the H112 level-free update; reveal the top two of the rest |
| `random4` / `ORACLE4` | H112's seeded four (so it reproduces H116) / the best possible four (ceiling only) |
| Endpoint | Best utility among the four. Six batches per context are averaged, then contexts within each of 44 families. |
| Primaries | `GP_2PLUS2 − S_TOP4`, `GP_1PLUS3 − S_TOP4` and `UCB_2PLUS2 − S_TOP4`. Bonferroni m = 3, one-sided .05/3 LCB > +.001, median ≥ 0, ≤ 10% of families losing > .01; a 10,000-draw family bootstrap, seed 20261031. |
| Decision | A passing rule becomes a **CANDIDATE** for a budgeted prospective W4 test (an owner call). If none passes, **W4 adaptive reveal at 4-of-16 is CLOSED**. |
| Integrity gate | On the 150 H116 batches, `S_TOP4` and `GP_2PLUS2` must reproduce H116's published values **exactly**, or the run stops |
| Caveat | Adaptive development: these batches' outcomes were seen in aggregate by H110/H116. A screen pass is not a confirmation. |

## Prep evidence ([prep_receipt.json](prep_receipt.json); screen results in [metrics.json](metrics.json))

| Check | Result |
|---|---|
| Pins: H112 module (sha256 `de7d5882…`), H110/H116/H118 `fits.jsonl`, H116 `rows_h112.jsonl` | pass |
| Prior rebuilt | 2,306 memory rows; slopes 0.836 / 0.895 / 0.908 / 0.902 / 0.877, identical to H112 |
| **Integrity gate** | **300 of 300 values reproduced exactly** (150 batches × `S_TOP4`/`GP_2PLUS2`); passed |
| Dry run | 3 H118 batches (non-roster panel, seed 1001), all rules screened; harness integrity only |
| Native fits / GPU | **0 / 0** |
| Wall | Integrity 138 s, dry run 105 s (xbabe2, nice 15) |

## Reproduce

```sh
python3 -m pytest -q research/flywheel_selfplay/h120_w4_offline_acquisition/test_run_h120.py   # 8 synthetic tests
cd research/flywheel_selfplay/h120_w4_offline_acquisition
A="--out <dir> --h94-run ~/jope-data/h94/opus-h94-20261031a --h103-run ~/jope-data/h103/opus-h103-20261031a \
   --h110-run ~/jope-data/h110/opus-h110-20261031a --h112-run ~/jope-data/h112/opus-h112-20261031a \
   --h115-run ~/jope-data/h115/opus-h115-20261031a"
nice -n 15 python3 -I run_h120.py integrity $A    # then dry-run
nice -n 15 python3 -I run_h120.py screen $A       # writes rows_h120.jsonl and metrics.json (1:44 wall)
```

## Data scope and evidence

- **Data scope.**
  - DL-Train only.
  - Zero DL-Val, outer-test, sealed or holdout reads.
  - DOPE is not loaded, and no GPU is used.
- **Evidence.**
  - Prep: S3 `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/W4/opus-h120-prep-20261031a/`.
  - Screen: S3 `…/W4/opus-h120-screen-20261031a/` holds the manifest, metrics, receipt, all 300 rows, the log, the Elo
    CI script and output, and the executed runner and tests. The version ids are in [metrics.json](metrics.json).
  - Each object was written create-only with a version-pinned readback hash.
  - The screen ran on xbabe2 at nice 15 with 4 threads: 1:44 wall, 190 MB peak RSS, 0 native fits and 0 GPU.
