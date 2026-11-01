# H197 closeout: per-context gradient clipping

**Status: INCOMPLETE / NOT_EVALUABLE.** The native reference passed, but attempt 3 stopped during the first FIT cell before B/A scoring. No scientific verdict is available.

## Hypothesis and registered setup

The preregistered hypothesis was that per-context adapter-gradient clipping (PCGC40) could reduce the influence of a small number of high-gradient, low-loss contexts and improve held-family utility over the exact r882 reference. The frozen design uses three independent TRAIN-family fold seeds, five cyclic A/B/FIT folds per seed, four matched 40-update arms (CTRL40, PCGC40, GCLIP40, PCGC_SHUF40), and a fixed step-40 candidate with no B-based selection. Only the 47-tensor/29,211-parameter adapter trains; the native loss, targets, and kernel stay fixed. See [PREREG.md](PREREG.md) and [fold_manifest.json](fold_manifest.json).

## Preflight and attempts

The preregistration is [PR #1616](https://github.com/neverhuman/JopeDime/pull/1616), merged at `33aa7fd2544cb4395d1c06f3e083be87a7502960`. A mandatory verify-only preflight passed on source `c61b23010759d179f626bc7378057239988b0f59`: all 10 local source, 11 input, and 15 asset pins verified; the pinned checkpoint and feature cache loaded; native r882 DL-Val replay was bit-exact on 91/91 contexts (`1.1377941898164525`) with exact cost bytes; all 1,219 TRAIN and 91 DL-Val pools loaded after the reference passed; optimizer updates were zero. The run used a separate preflight output and did not claim an attempt.

- **Attempt 1:** Resource guard passed, then the worker failed after 1.77 seconds because the H189 helper import resolved outside the checkout. No model loaded and no updates occurred. The root-path and module isolation fix is [PR #1617](https://github.com/neverhuman/JopeDime/pull/1617), merged at `ca3c938e1`.
- **Attempt 2:** Resource guard passed, then H182 source verification rejected the H197 `PREREG.md` pin because it resolved local pins relative to H189. The verifier hashed pinned files, but did not deserialize the checkpoint or load a model. The local-pin and preflight fix is [PR #1622](https://github.com/neverhuman/JopeDime/pull/1622), merged at `ff9ed17390db799cce0c0a59ea4d823f38bdd2e3`.
- **Attempt 3:** The required preflight passed, then the guarded CPU run passed the exact reference and completed 160 FIT optimizer updates in seed `197101`, fold `0` (40 each for CTRL40, PCGC40, GCLIP40, and PCGC_SHUF40). It stopped while writing the PCGC_SHUF40 trace: its factor permutation was a NumPy array and the JSON writer rejected it. The three preceding arm traces/receipts and all four adapter snapshots exist; the shuffled trace is truncated and has no receipt. The attempt failure record shows `backward_started=720`, `backward_completed=720`, `step_started=160`, `step_completed=160`, `protected_reads=0`, and no audit refusals. The monitor reported no resource violation. No B panels, B lock, A scores, or cross-seed outcomes were produced.

The trace serializer is corrected in the closeout source and covered by a fixture. Attempt 3 is consumed and was not resumed. Attempts 1–3 claims, launch/preflight receipts, monitor samples, stdout, failure record, and partial FIT files remain outside Git under `/home/ubuntu/jope_plan/ns/`.

## Family-grouped metrics and decision

The registered outputs—per-seed equal-family utility gains, median and paired-family 95% CI over three seeds, bootstrap p-value, positive-seed count, harm rate, exact `-2.0` picks, clipping summaries, and family-bootstrap Elo/lift versus r882—are **not computed**. They require completed FIT states and B-before-A scoring across all cells. The 160 updates are a partial execution record, not a candidate comparison. The registered decision is **NOT_EVALUABLE**; no adoption or serving action is supported.

## Verification and next step

The closeout fixes NumPy permutation serialization for a future authorized run. Focused H197 fixtures, Ruff, all five `jope-records` checks, source pins, Python compilation, and the import check pass. No attempt 4 is authorized by this closeout; any further run requires a new ENG decision. No large science panels were produced, so no S3 result custody was created.
