# H161 — Robust Per-Context Normalized/Huber Loss + Family-Balanced Sampling on `r882` Continuation

**Status: `COMPLETE` (`RULE_NEUTRAL` / `NO_ADVANCE`).** Executed in a single supervised CPU wave on `xbabe2` (`pid = 523125`, `nice = 19`, `exit_status = 0`, `2,400` AdamW updates across `60` cells, `0 / 60` non-zero checkpoint selections under the strict `H148` family cross-fit selector). See [PREREG.md](PREREG.md) and [metrics.json](metrics.json).

---

## 1. Hypothesis & Setup

- **Hypothesis**: On `1,219` real-S3 `DL-Train` contexts (`205` dataset families), raw context losses span several orders of magnitude and family context counts range from `1` to `76`. We tested whether:
  1. **`CLIP`** — normalizing each context's loss $L_c(\theta)$ by the detached fit-fold median `r882` baseline loss $s_{\text{fit}}(A) = \max(\text{median}_{c \in A} L_c(\theta_{r882}), 10^{-3})$ and applying Huber ($\delta = 1.0$) to normalized excess above $1.0$:
     $$\tilde{\ell}_c(\theta) = \frac{L_c(\theta)}{s_{\text{fit}}(A)}, \qquad \ell^{\text{CLIP}}_c(\theta) = s_{\text{fit}}(A)\cdot\begin{cases}\tilde{\ell}_c(\theta) & \tilde{\ell}_c(\theta) \le 1\\ 1 + \text{Huber}_{\delta=1.0}(\tilde{\ell}_c(\theta)-1) & \tilde{\ell}_c(\theta) > 1\end{cases}$$
  2. **`FAMBAL`** — replacing context-uniform sampling (`16` contexts/step) with equal-family sampling (`8` distinct fit-fold families $\times 1$ context per microbatch, accumulated over `G = 2` microbatches per optimizer update so each step sees `16` contexts across `16` distinct families at identical `#updates = 40`), or
  3. **`CLIP_FAMBAL`** — combining both interventions,
  produces continuation checkpoints from `r882` (`d25b1db7`, `29,211` trainable adapter parameters across `47` tensors; DOPE backbone `c51a56d0` frozen) that pass the `H148` 5-fold family cross-fit selector (`gain > +0.001 + 1e-12` AND `10,000`-draw paired family-bootstrap 95% LCB `> 0` on 41-family selection fold `B`) and beat `r882` out-of-fold.
- **5-Fold Family Cross-Fit (`fold_seed = 161140`, `boot_seed = 161141`)**:
  - `205` `DL-Train` families (`1,219` contexts) partitioned into `5` disjoint folds (`41` families each; `[238, 250, 240, 260, 231]` contexts).
  - For each outer assess fold $k \in \{0,1,2,3,4\}$: `fit` $A = \{(k+1)\bmod 5, (k+2)\bmod 5, (k+3)\bmod 5\}$ (`123` families), `select` $B = \{(k+4)\bmod 5\}$ (`41` families), `assess` $C = \{k\}$ (`41` families).
  - `3` seeds (`161101, 161102, 161103`) $\times 5$ folds $\times 4$ arms (`CTRL`, `CLIP`, `FAMBAL`, `CLIP_FAMBAL`) $= 60$ cells (`40` AdamW updates/cell, `lr = 1e-4`, `wd = 1e-4`, candidate checkpoints $\{0, 10, 20, 30, 40\}$; total `2,400` optimizer updates).
- **Exact `r882` Baseline Gate (Pre-Execution Hard Gate)**:
  - `91 / 91` `DL-Val` contexts (`19` families) matched the canonical `r882` artifact (`max_context_error = 0.0`, `mean_error = 0.0`, context mean `1.1377941898164525`, equal-family mean `1.153615276991245`, `0` `-2` picks; `0.0` bitwise discrepancy between uncached and frozen-anchor-cached scoring).
  - DOPE backbone (`c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`) and frozen non-adapter state (`8d1070faf574ee1cd2a4e630887e1662b43189b3f89b04982143ef2704818aae`) verified identical before and after every cell (`61 / 61` checks).

---

## 2. Primary Out-of-Fold (`DL-Train`, `1,219` Contexts / `205` Families) & Secondary (`DL-Val`, `91` Contexts / `19` Families) Results

Under the strict `H148` selector rule on the 41-family selection fold `B`, **`0 / 15` cells selected a non-zero step in every arm (`0 / 60` total)**. Every cell retained step `0` (`r882`), producing `RULE_NEUTRAL` (`NO_ADVANCE`):

| Arm | Selected `> 0` (`15` cells) | Out-of-Fold `DL-Train` Family Utility (`205` fam) | $\Delta$ vs `r882` Median `[95% CI]` | Holm $p$ vs `r882` | `DL-Train` Context Utility (`1,219` ctx) | `DL-Train` `-2` Picks | `DL-Train` Canonical BT Elo `[95% CI]` | `DL-Val` Family Utility (`19` fam) | `DL-Val` `-2` Picks |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `r882` (step `0`) | — | `1.385355` | — | — | `1.423073` | `17` | `1500.0 [1500.0, 1500.0]` | `1.153615` | `0` |
| `CTRL` | `0 / 15` | `1.385355` | `+0.000000 [+0.000000, +0.000000]` | `1.0000` | `1.423073` | `17` | `1500.0 [1500.0, 1500.0]` | `1.153615` | `0` |
| `CLIP` | `0 / 15` | `1.385355` | `+0.000000 [+0.000000, +0.000000]` | `1.0000` | `1.423073` | `17` | `1500.0 [1500.0, 1500.0]` | `1.153615` | `0` |
| `FAMBAL` | `0 / 15` | `1.385355` | `+0.000000 [+0.000000, +0.000000]` | `1.0000` | `1.423073` | `17` | `1500.0 [1500.0, 1500.0]` | `1.153615` | `0` |
| `CLIP_FAMBAL` | `0 / 15` | `1.385355` | `+0.000000 [+0.000000, +0.000000]` | `1.0000` | `1.423073` | `17` | `1500.0 [1500.0, 1500.0]` | `1.153615` | `0` |

---

## 3. Selector-Fold (`B`) & Parameter-Drift (`H151`) Diagnostics

Even though `0 / 60` cells cleared the `41`-family bootstrap `LCB > 0` gate, the logged trajectories on selection fold `B` (`15` fold-seed cells per arm) and the `H151` L2 adapter drift telemetry isolate the exact effects of `CLIP` and `FAMBAL`:

- **Detached Fit-Fold Median `r882` Loss Scales $s_{\text{fit}}(A)$**:
  - Fold `0`: `1.243913`; Fold `1`: `1.338549`; Fold `2`: `1.522554`; Fold `3`: `1.315936`; Fold `4`: `1.238511`.

| Arm | Step-40 L2 Drift Median `[Min, Max]` | Drift Ratio vs `CTRL` | Cells with Best Step (`10..40`) Gain `> 0` on `B` | Median Best-Step Gain on `B` | Max Best-Step Gain on `B` | Step `10` Median Gain on `B` | Step `20` Median Gain on `B` | Step `30` Median Gain on `B` | Step `40` Median Gain on `B` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `CTRL` | `0.442175 [0.429413, 0.457697]` | `1.0000x` | `6 / 15` | `+0.000000` | `+0.028195` | `-0.011267` | `-0.017582` | `-0.032031` | `-0.024187` |
| `CLIP` | `0.433611 [0.417121, 0.450119]` | `0.9806x` | `6 / 15` | `+0.000000` | `+0.028484` | `-0.007193` | `-0.016451` | `-0.032211` | `-0.028472` |
| `FAMBAL` | `0.446848 [0.438437, 0.451904]` | `1.0106x` | `7 / 15` | `+0.000000` | `+0.027165` | `-0.006985` | `-0.017527` | `-0.023866` | `-0.027574` |
| `CLIP_FAMBAL` | `0.438649 [0.429401, 0.444935]` | `0.9920x` | **`8 / 15`** | **`+0.000294`** | `+0.027470` | **`-0.006104`** | `-0.025645` | `-0.027742` | `-0.025853` |

- **Why `0 / 60` cells passed `LCB > 0` despite positive point gains (`+0.027` to `+0.028`)**:
  - In `CLIP_FAMBAL`, a majority of cells (`8 / 15`, `median_best_step_gain = +0.000294`) achieved positive best-step gain on selection fold `B`, and step-10 median degradation on `B` was reduced by `45.8%` vs `CTRL` (`-0.006104` vs `-0.011267`).
  - Furthermore, step-40 L2 parameter drift was virtually identical across all four arms (`0.9806x` to `1.0106x` `CTRL`), proving that neither `CLIP` nor `FAMBAL` acts merely as a hidden step-size reduction. (Because no treatment arm passed primary efficacy, conditional post-hoc `CTRL_DRIFT` was not triggered: `ctrl_drift_executed = false`.)
  - However, on a 41-family selection fold `B` (e.g., outer fold `A3` where `select` is fold `4`), the `+0.0272` to `+0.0285` equal-family gain at `step = 10` is driven by flipping a single family out of a `-2.0` invalid cliff while the remaining `40` families experience tiny negative drift (`~ -0.001`). In a `10,000`-draw family bootstrap over `41` families, the $\approx 36.3\%$ of bootstrap resamples that omit that single cliff-flip family have a slightly negative mean delta, placing the 2.5th percentile (`LCB`) at `-0.000504` (`FAMBAL`), `-0.001483` (`CLIP_FAMBAL`), `-0.001914` (`CLIP`), and `-0.002021` (`CTRL`). The strict `LCB > 0` gate therefore abstains (`step = 0`) in every cell.

---

## 4. Execution & Custody Receipts

- **Prelaunch Boundary Refusal Receipt**:
  - At initial merged prereg commit `14f22d182ada9b37145bf84080ae88c15d482407`, the first launch aborted before context loading (`optimizer_updates = 0`, `scored_contexts = 0`) because `runner.py` wrote `out.parent / "launch.json"` after `sys.addaudithook` while only `out` (`.../run`) was allowlisted. That prelaunch refusal receipt was preserved in `/home/ubuntu/jope-rescue-20260921/h161-rd_argon-20261101/custody-prelaunch-refusal/` and S3, and fixed in PR [#1376](https://github.com/neverhuman/JopeDime/pull/1376) (`cea36d4d8a3711d275279647beda060aa078f29e`).
- **Single Supervised Scientific Wave (`cea36d4d8a3711d275279647beda060aa078f29e`)**:
  - Command: `python3 -B /home/ubuntu/JopeDime-rd/research/flywheel_selfplay/h161_robust_fambal/monitor.py --custody /home/ubuntu/jope-data/h161/custody-run`
  - Telemetry (`9,366` samples over `2,983.15s` wall / `2,980.24s` worker): `pid = 523125`, `nice = 19`, `exit_status = 0`, max `3` OS threads (`<= 4`), peak RSS `3.49 GB` (`3,491,217,408 B < 6.00 GB`), min free disk `634.13 GB` (`>= 40 GB`), max user-visible disk used `91.93%` (`< 95%`), max scratch `20.30 MB` (`< 3 GB`), max host load `44.48` (`<= 85.0`), min `MemAvailable` `68.37 GB` (`63.67 GiB >= 16 GiB`), `0` protected/refused reads.
- **S3 Custody Archive & Restore Verification (`21 / 21` objects)**:
  - Bucket/Prefix: `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H161/argon-h161-20261101a/`
  - Key VersionIds & SHA-256:
    - `custody-run/run/metrics.json`: VersionId `x4WuccjBDZuc38vMOqvlXkUlEu5HP90J`, SHA-256 `23c5d07d18537f9792f48c58fde36e14fac742c5ce949e240443be9514163645`
    - `custody-run/run/baseline.json`: VersionId `EspaAeGHYQDoxSCGH5g0piffXTnNyONx`, SHA-256 `6c4d1337c3deea49b48890b03d62799ed04d737a078170672bfc3651317e80a5`
    - `custody-run/run/audit.json`: VersionId `rCbYeTQiUhdaU9a1hDwkl.16gOT2PvSV`, SHA-256 `0fcb8a6111f7a662ec8e2b317db6165329162f7dddb4106831fcaabba3148a75`
    - `custody-run/run/panels.npz`: VersionId `hLz0e5GMfb9ULbWtTor2PxYrRhye6mGp`, SHA-256 `16e2d79c7b14d2e7b088ee9011d4241a14bfbd04138597045c65261e795c1d4a`
    - `custody-run/run/cells-60.tar.gz`: VersionId `oMmIajL3L6jR_.K4aUqz0FQ79mFnDwwR`, SHA-256 `859b01807ba12491e9423fd08e93014541b8431d4a952263387a3d63dc150c6c`
  - Full inventory receipt: `/home/ubuntu/jope_plan/evidence/H161/s3-custody-receipt.json` (and local backup `/home/ubuntu/jope-rescue-20260921/h161-rd_argon-20261101/`); scratch `/home/ubuntu/jope-data/h161` cleaned after verification.

---

## 5. Decision & Next Steps

- **Verdict**: `RULE_NEUTRAL` (`NO_ADVANCE`). Neither `CLIP`, `FAMBAL`, nor `CLIP_FAMBAL` produced a checkpoint that cleared the `41`-family selection-fold bootstrap `LCB > 0` gate at `lr = 1e-4` over `10..40` steps, so `r882` remains the undefeated anchor (`serving_promotion = false`).
- **Next Steps**:
  1. Do not promote per-context median Huber clipping or equal-family minibatching alone on unregularized `lr = 1e-4` `r882` continuation.
  2. Because `CLIP_FAMBAL` achieved positive best-step selection-fold gain in `8 / 15` cells (`+0.000294` median) at `0.992x` `CTRL` parameter drift while failing `LCB > 0` due to slight dense-family drift (`~ -0.001`) against single-family `-2` cliff rescues, pair `CLIP_FAMBAL` with explicit non-target ranking-anchor KL / margin-locking on already-valid families so cliff rescues do not trade off against small regressions on the remaining `40` selection families.
