# H120 — W4 offline acquisition screen: preregistration and CPU preparation (zero native fits)

> **Numbering.** Opened as H119 (#1046). sol_v1_dsc08 had claimed H119 first, so per the hwave overseer this is H120. The
> content is unchanged, and the prep receipts were regenerated under the H120 runner.

Status: **prep complete: preregistration, harness, integrity gate passed and dry run. No screen result.** The screen
itself (`run_h120.py screen`) runs after this preregistration is on main. It costs **0 native fits**.

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

## Prep evidence ([prep_receipt.json](prep_receipt.json))

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
python3 -m pytest -q research/flywheel_selfplay/h120_w4_offline_acquisition/test_run_h120.py   # 6 synthetic tests
cd research/flywheel_selfplay/h120_w4_offline_acquisition
A="--out <dir> --h94-run ~/jope-data/h94/opus-h94-20261031a --h103-run ~/jope-data/h103/opus-h103-20261031a \
   --h110-run ~/jope-data/h110/opus-h110-20261031a --h112-run ~/jope-data/h112/opus-h112-20261031a \
   --h115-run ~/jope-data/h115/opus-h115-20261031a"
nice -n 15 python3 -I run_h120.py integrity $A    # then dry-run; the screen stage follows the merge
```

## Data scope and evidence

- **Data scope.**
  - DL-Train only.
  - Zero DL-Val, outer-test, sealed or holdout reads.
  - DOPE is not loaded, and no GPU is used.
- **Evidence.** S3 `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/W4/opus-h120-prep-20261031a/`
  holds the receipts, logs and sources. Each was written create-only with a version-pinned readback hash.
