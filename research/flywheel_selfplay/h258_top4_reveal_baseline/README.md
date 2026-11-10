# H258 (`h258_top4_reveal_baseline`) — Equal-Budget `TOP4_PICK` Search Baseline vs `ACTIVE_2PLUS2_HURDLE` (`r882` & `r3175`)

- **Wave ID**: `h258-top4-reveal-baseline-20261103`
- **Status**: `COMPLETE`
- **Verdict**: **`NO_ADVANCE`** (`ACTIVE_2PLUS2_HURDLE` loses decisively to the trivial equal-budget baseline `TOP4_PICK` on both `r882` and `r3175`; adoption of `ACTIVE_2PLUS2_HURDLE` is **REJECTED** in favor of direct best-of-revealed selection `TOP4_PICK`)
- **Preregistration Commit (`origin/v1/next`, `0` outcomes read)**: `df196c1f9e570e544a0002c2651b4ee75847154b` (`2026-11-03T14:17:19Z`, `<= 14:45Z` deadline)
- **Owner / Slot**: `rd_argon` (`slot-7`, `xbabe2` cores `57-60`, `nice 19`, `<= 4` threads, peak RSS `1.43 GiB <= 4.0 GiB`, `0` GPU)

---

## 1. Hypothesis & Motivation

In `H252` and `H254`, `ACTIVE_2PLUS2_HURDLE` achieved `+0.015174` median family utility over `GATED_025` (`+24.75` Elo) on `DL-Val` (`Native91`, `91` contexts / `19` families). However, the `14:05Z` red-team audit identified two critical structural questions before any policy adoption:

1. **Weak comparator in `H252`/`H254`**: `GATED_025` reveals `4` *random* candidates from `idx4` (`top-16` by `z_prior`) and applies a noisy affine shift to the `1152`-candidate pool score (`Elo 1500.00` on `DL-Val`, `1500.33` on `DL-Train` OOF). Because the scored quantity is `utility = -fitness` (`common_h252.py:137`), any policy that reveals `4` true candidate evaluations on gated contexts (`top_zgap < 0.25`) can simply **pick the best revealed candidate** (`argmin` of revealed gated cost `fitness if gate else 2.0`).
2. **Primary Question (`H258`)**: Does `ACTIVE_2PLUS2_HURDLE` beat the trivial equal-budget (`4` reveals on gated contexts `top_zgap < 0.25`, `0` elsewhere) search baseline **`TOP4_PICK`** (reveal the `4` lowest-`r882`-cost candidates, pick `argmin` revealed gated cost; tie -> `r882` order)? And does that comparison survive on the current flywheel incumbent **`r3175`** (`round-3175.pt`, SHA256 `1d6ea900ebed8d2a05c3b3f37576adc15aadd2b53c585bfd5cb71a103e9ff97b`)?

---

## 2. Experimental Setup

- **Frozen Snapshot**: Byte-identical `common_h252.py` (`abe974ed10a0cd0d4b08440015a72a7b0adcebb22118c234732b812e9f0b0f92`) and `stage1_hurdle_sos_critic.py` (`fc0dd0e2fa543f0540d677609989c1785ef33b07f6cf7e54f4f8bedbd0e5e797`) from `H254`, with frozen DOPE backbone `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`.
- **Gate & Reveal Budget**: Gate `top_zgap < 0.25`. All equal-budget reveal arms (`GATED_025`, `ACTIVE_2PLUS2_HURDLE`, `TOP4_PICK`, `TOP2_RAND2_PICK`, `ACTIVE_2PLUS2_PICK`) reveal **exactly `4` unique candidates** on gated contexts (`91/91` contexts on `DL-Val`, `1,200/1,219` contexts on `DL-Train` OOF) and **`0` candidates** on ungated contexts (`0/91` on `DL-Val`, `19/1,219` on `DL-Train` OOF).
- **Evaluated Arms (`r882` primary + `r3175` incumbent transfer)**:
  - `R882` / `R3175`: `0` reveals; single-pick argmin of incumbent predicted cost.
  - `GATED_025` / `GATED_025@r3175`: Frozen `H252`/`H254` baseline (`4` seeded random reveals from `idx4` on gated contexts -> 1-parameter shift on full pool).
  - `ACTIVE_2PLUS2_HURDLE` / `ACTIVE_2PLUS2_HURDLE@r3175`: Frozen `H252`/`H254` candidate (`top-2` by incumbent cost + `2` hurdle-critic picks from `idx4 \ top2` -> 2-stage affine shift on full pool).
  - `TOP4_PICK` / `TOP4_PICK@r3175`: Equal-budget trivial baseline (reveal `4` lowest-incumbent-cost candidates on gated contexts; pick `argmin` revealed gated cost `fitness if gate else 2.0`, tie -> incumbent order; `0` reveals on ungated contexts).
  - `TOP2_RAND2_PICK` / `TOP2_RAND2_PICK@r3175`: Descriptive equal-budget baseline (reveal `top-2` by incumbent cost + `2` seeded random from `idx4 \ top2` on gated contexts; pick `argmin` revealed gated cost).
  - `ACTIVE_2PLUS2_PICK` / `ACTIVE_2PLUS2_PICK@r3175`: Diagnostic equal-budget decomposition (uses the **exact same** `rev4` revealed set as `ACTIVE_2PLUS2_HURDLE` on gated contexts, but picks `argmin` revealed gated cost instead of shifting the full unrevealed pool).
  - `TOP4_PICK_UNGATED` / `TOP4_PICK_UNGATED@r3175`: Descriptive unequal-budget reference (reveals `top-4` on all contexts regardless of gate).
- **Seeds & Statistical Protocol**:
  - Fresh 5-fold family-grouped `DL-Train` OOF split seed `258140` (`1,219` contexts / `205` families; `0` family overlap across folds), evaluation seeds `258101, 258102, 258103`, paired family bootstrap seed `258150` (`10,000` draws), Holm step-down `m = 2` across Primary (`ACTIVE_2PLUS2_HURDLE - TOP4_PICK`) and Secondary (`TOP4_PICK - GATED_025`) on `DL-Val` (`Native91`, `91` contexts / `19` families).
  - `400`-iteration MM Bradley-Terry Elo (`scale = 400 / ln(10)`, `1e-3` uniform prior pseudo-games) anchored at `R882 = 1500.0`.

---

## 3. Baseline Regression & Hygiene Audits

1. **`R882` Bit-Exact Regression Gate (`PASS` prior to arm evaluation)**:
   - `R882` `DL-Val` (`Native91`) mean context utility = **`1.1377941898164525`** (`expected = 1.1377941898164525`, `abs_diff = 0.0`, **`91 / 91` bit-exact context matches**).
   - `R3175` `DL-Val` (`Native91`) mean context utility = **`1.1366829009431036`**.
2. **Audit (a) — Zero Leakage & Sealed Boundary (`PASS`)**:
   - `has_meta_test = False` verified on all `1,310` contexts (`91` `DL-Val` + `1,219` `DL-Train`); `0` sealed, holdout, or meta-test reads.
   - `0` `DL-Val` contexts (`0 / 91`) and `0` `DL-Val` families (`0 / 19`) appear in `pool_priors.npz` or `promotions.jsonl` (`3,449` training-only promotions used).
   - `0` family overlap across the 5 `DL-Train` OOF folds (`fold_seed = 258140`).
   - Checkpoints (`r882`, `r3175`) and DOPE (`c51a56d0...`) SHA256 hashes verified identical before and after execution (`0` optimizer updates, `CUDA_VISIBLE_DEVICES=""`).
3. **Audit (b) — Exact Reveal-Budget & Snapshot Invariance (`PASS`)**:
   - Across all `3,930` context-seed evaluations (`(91 + 1,219) * 3`), every equal-budget arm (`GATED_025`, `ACTIVE_2PLUS2_HURDLE`, `TOP4_PICK`, `TOP2_RAND2_PICK`, `ACTIVE_2PLUS2_PICK`) used `min = 4, max = 4` unique reveals on all `3,873` gated evaluations (`91 * 3` `DL-Val` + `1,200 * 3` `DL-Train`) and `min = 0, max = 0` reveals on all `57` ungated evaluations (`19 * 3` `DL-Train`).
   - `ACTIVE_2PLUS2_HURDLE` reproduced `eval_active_2plus2_hurdle` from the frozen `H254` snapshot bit-for-bit on **`273 / 273` `DL-Val` evaluations (`3` seeds × `91` contexts)** for `r882` and **`273 / 273` `DL-Val` evaluations** for `r3175` (`DL-Train` OOF loops called `score_h258_context(verify_snapshot=False)` where `snap_match` defaults `True`, so `DL-Train` OOF was not snapshot-verified).
4. **Audit (c) — Permuted-Fitness Negative Control (`PASS`)**:
   - When revealed `fitness` / `gate` values are permuted within the `4` revealed candidates on `DL-Val`, `PERM_CONTROL - GATED_025` yields median family gain **`-0.009188`** (`95% CI [-0.033623, +0.001590]`, covers `0`).

### 3.1 Post-prereg code changes (no scoring logic change)

Between the preregistration commit (`df196c1f9e570e544a0002c2651b4ee75847154b`) and the closeout commit (`8869cc439c5f218bdef8b5276254b7acd595dff3`), `runner.py` was updated (`git diff df196c1f9 8869cc439 -- research/flywheel_selfplay/h258_top4_reveal_baseline/runner.py`) with the following infrastructure and memory-envelope fixes while `0` scoring expressions were altered:
- **(a) Batched pool loading + `malloc_trim` / `cooperate` pauses (memory only)**: Replaced all-at-once `load_pools` with 50-context batches that pop unused `program` (`412`-dim), `pids`, and `proj16` arrays and invoke `gc.collect()` + `libc.malloc_trim(0)` and `cooperate(out_dir)` so peak RSS stayed at `1.43 GiB <= 4.0 GiB`.
- **(b) `POOL_PRIORS` path fix**: Corrected `CACHE_DIR / "h252" / "pool_priors.npz"` (which did not exist on disk) to `CACHE_DIR / "pool_priors.npz"` (`/home/ubuntu/jope-data/h58/cache/pool_priors.npz`, SHA256 `6db567ef2c10d2c6835be5b4506d3861e6180f231fd75425140423718d12cf00`), which is the exact file pinned in the `H254` manifest.
- **(c) `DOPE` check**: Changed from hashing a standalone `dope.pt` file path to asserting the `dope_sha256` metadata field (`c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`) in `r882` / `r3175` plus the split manifest asset SHA (`a36510b9921c7e4e83a4769262328561bf06a7c7a0ba01779a6139c441e91933`); because H258 performs `0` optimizer updates, the checkpoint/backbone freeze is unaffected.
- **(d) `ref_ctx_map` parsing fix**: Updated `R882_REF_JSON` context parsing to read `[c["cid"] for c in ref_obj["evaluation_contract"]["contexts"]]` (list of dicts rather than bare strings).
- **(e) Chunked Elo bootstrap**: Computed the `10,000` bootstrap MM Bradley-Terry Elo draws in chunks of `500` instead of a single `(10000, 205, 14, 14)` tensor to avoid a `3.2 GiB` transient allocation.
- **(f) `verify_snapshot` made optional**: Added `verify_snapshot: bool = True` to `score_h258_context` (`True` on `DL-Val`, `False` on `DL-Train` OOF to avoid redundant double evaluation on `3,657` training context-seed pairs).
- **Scoring invariance**: The `ACTIVE_2PLUS2_HURDLE`, `GATED_025`, and `TOP4_PICK` scoring expressions are byte-identical (moved, not changed).

---

## 4. Key Metrics

### 4.1 `DL-Val` (`Native91`: `91` contexts / `19` families; `91/91` gated) — Arm Summary (`r882` & `r3175`)

| Arm | Incumbent | Reveal Budget (Gated / Ungated) | Median Context Utility | Median Family Utility | MM BT Elo (`R882=1500` Joint) | 95% Elo CI |
| --- | --- | --- | --- | --- | --- | --- |
| `R882` | `r882` | `0 / 0` | `1.137794` | `1.153615` | `1500.00` | `[1500.00, 1500.00]` |
| `GATED_025` | `r882` | `4 / 0` | `1.137794` | `1.153615` | `1500.00` | `[1500.00, 1500.00]` |
| `ACTIVE_2PLUS2_HURDLE` | `r882` | `4 / 0` | `1.153636` | `1.168790` | `1528.69` | `[1503.41, 1556.17]` |
| `TOP2_RAND2_PICK` | `r882` | `4 / 0` | `1.236128` | `1.247809` | `1765.84` | `[1711.35, 1828.04]` |
| `ACTIVE_2PLUS2_PICK` (diag) | `r882` | `4 / 0` | `1.234169` | `1.245932` | `1777.41` | `[1726.00, 1835.93]` |
| **`TOP4_PICK`** | **`r882`** | **`4 / 0`** | **`1.240360`** | **`1.252000`** | **`1846.46`** | **`[1797.37, 1904.66]`** |
| `TOP4_PICK_UNGATED` (unequal)* | `r882` | `4 / 4` | `1.240360` | `1.252000` | `1846.46` | `[1797.37, 1904.66]` |
| `R3175` | `r3175` | `0 / 0` | `1.136683` | `1.152551` | `1494.47` | `[1475.69, 1509.22]` |
| `GATED_025@r3175` | `r3175` | `4 / 0` | `1.137713` | `1.153538` | `1497.79` | `[1480.64, 1510.28]` |
| `ACTIVE_2PLUS2_HURDLE@r3175` | `r3175` | `4 / 0` | `1.148076` | `1.163464` | `1512.71` | `[1485.27, 1538.44]` |
| `TOP2_RAND2_PICK@r3175` | `r3175` | `4 / 0` | `1.236119` | `1.247800` | `1767.76` | `[1712.45, 1829.12]` |
| `ACTIVE_2PLUS2_PICK@r3175` (diag) | `r3175` | `4 / 0` | `1.233405` | `1.245200` | `1778.05` | `[1724.54, 1837.26]` |
| **`TOP4_PICK@r3175`** | **`r3175`** | **`4 / 0`** | **`1.240323`** | **`1.251964`** | **`1846.46`** | **`[1797.78, 1903.42]`** |
| `TOP4_PICK_UNGATED@r3175` (unequal)* | `r3175` | `4 / 4` | `1.240323` | `1.251964` | `1846.46` | `[1797.78, 1903.42]` |

*\*Note: On `DL-Val` (`Native91`), all `91 / 91` contexts satisfy `top_zgap < 0.25`, so `TOP4_PICK` and `TOP4_PICK_UNGATED` perform the exact same reveals (`4` per context).*

### 4.2 `DL-Val` (`Native91`) — Preregistered Primary, Secondary & Diagnostic Contrasts

| Contrast (`DL-Val`, `19` families) | Role | Median Family Gain | 95% Bootstrap CI | Holm `m=2` 97.5% CI | Raw `p` (1-sided) | Holm `p` (`m=2`) | Pos Seeds | Family W/T/L (Seeds 1 / 2 / 3) | Harm `>0.01` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **`ACTIVE_2PLUS2_HURDLE - TOP4_PICK`** | **Primary (`m=2`)** | **`-0.083210`** | **`[-0.133392, -0.046824]`** | **`[-0.142342, -0.043140]`** | `1.0000` | `1.0000` | `0 / 3` | `0/0/19`, `0/0/19`, `0/0/19` | `89.47%` |
| **`TOP4_PICK - GATED_025`** | **Secondary (`m=2`)** | **`+0.098384`** | **`[+0.059734, +0.148274]`** | **`[+0.055752, +0.156378]`** | `0.00010` | `0.00020` | `3 / 3` | `19/0/0`, `19/0/0`, `19/0/0` | `0.00%` |
| `TOP4_PICK - R882` (4 vs 0 reveals) | Unequal-budget ref | `+0.098384` | `[+0.059734, +0.148274]` | `[+0.055752, +0.156378]` | `0.00010` | — | `3 / 3` | `19/0/0`, `19/0/0`, `19/0/0` | `0.00%` |
| `ACTIVE_2PLUS2_HURDLE - GATED_025` | H254 reproduction | `+0.015174` | `[+0.005182, +0.026381]` | `[+0.003693, +0.028205]` | `0.00010` | — | `3 / 3` | `9/8/2`, `10/7/2`, `9/8/2` | `0.00%` |
| `ACTIVE_2PLUS2_HURDLE - ACTIVE_2PLUS2_PICK` | Shift vs Pick (`rev4`) | `-0.076644` | `[-0.113200, -0.044213]` | `[-0.119265, -0.040614]` | `1.0000` | — | `0 / 3` | `0/1/18`, `0/1/18`, `0/1/18` | `84.21%` |
| `ACTIVE_2PLUS2_PICK - TOP4_PICK` | Critic vs Top4 (`rev4`) | `-0.006068` | `[-0.030003, +0.010652]` | `[-0.032892, +0.012606]` | `0.7178` | — | `1 / 3` | `5/0/14`, `10/0/9`, `10/0/9` | `26.32%` |
| `TOP2_RAND2_PICK - TOP4_PICK` | Random-2 vs Top-3,4 | `-0.004191` | `[-0.026291, +0.009918]` | `[-0.028468, +0.011891]` | `0.7240` | — | `1 / 3` | `4/0/15`, `10/0/9`, `8/0/11` | `31.58%` |
| **`ACTIVE_2PLUS2_HURDLE@r3175 - TOP4_PICK@r3175`** | **`r3175` Primary** | **`-0.088499`** | **`[-0.139671, -0.050637]`** | **`[-0.147326, -0.046696]`** | `1.0000` | `1.0000` | `0 / 3` | `0/0/19`, `0/0/19`, `0/0/19` | `89.47%` |
| **`TOP4_PICK@r3175 - GATED_025@r3175`** | **`r3175` Secondary** | **`+0.098426`** | **`[+0.060517, +0.149209]`** | **`[+0.056244, +0.157046]`** | `0.00010` | `0.00020` | `3 / 3` | `19/0/0`, `19/0/0`, `19/0/0` | `0.00%` |
| `TOP4_PICK@r3175 - R3175` (4 vs 0 reveals) | `r3175` Unequal ref | `+0.099413` | `[+0.060985, +0.149415]` | `[+0.056695, +0.157524]` | `0.00010` | — | `3 / 3` | `19/0/0`, `19/0/0`, `19/0/0` | `0.00%` |
| `R3175 - R882` (0 reveals) | Incumbent check | `-0.001064` | `[-0.006536, +0.002692]` | `[-0.007385, +0.003122]` | `0.6390` | — | `0 / 3` | `2/15/2`, `2/15/2`, `2/15/2` | `5.26%` |
| `TOP4_PICK@r3175 - TOP4_PICK` | Cross-incumbent Top4 | `-0.000036` | `[-0.000353, +0.000245]` | `[-0.000353, +0.000245]` | `0.6500` | — | `0 / 3` | `1/17/1`, `1/17/1`, `1/17/1` | `0.00%` |

---

### 4.3 `DL-Train` OOF (`1,219` contexts / `205` families; `1,200` gated, `19` ungated; `fold_seed = 258140`)

| Arm | Incumbent | Median Context Utility | Median Family Utility | MM BT Elo (`R882=1500` Joint) | 95% Elo CI |
| --- | --- | --- | --- | --- | --- |
| `R882` | `r882` | `1.423073` | `1.385355` | `1500.00` | `[1500.00, 1500.00]` |
| `GATED_025` | `r882` | `1.423014` | `1.385282` | `1500.33` | `[1499.08, 1501.12]` |
| `ACTIVE_2PLUS2_HURDLE` | `r882` | `1.494245` | `1.466442` | `1524.76` | `[1518.31, 1531.99]` |
| `TOP2_RAND2_PICK` | `r882` | `1.590063` | `1.562425` | `1765.55` | `[1747.36, 1784.00]` |
| `ACTIVE_2PLUS2_PICK` (diag) | `r882` | `1.597481` | `1.571093` | `1772.99` | `[1753.66, 1791.64]` |
| **`TOP4_PICK`** | **`r882`** | **`1.606115`** | **`1.569233`** | **`1793.96`** | **`[1774.62, 1814.70]`** |
| `TOP4_PICK_UNGATED` (unequal) | `r882` | `1.614066` | `1.578716` | `1798.82` | `[1779.16, 1819.76]` |
| `R3175` | `r3175` | `1.419535` | `1.381525` | `1503.07` | `[1498.68, 1507.82]` |
| `GATED_025@r3175` | `r3175` | `1.421353` | `1.386693` | `1502.60` | `[1498.29, 1507.13]` |
| `ACTIVE_2PLUS2_HURDLE@r3175` | `r3175` | `1.493041` | `1.466126` | `1526.18` | `[1518.95, 1534.13]` |
| `TOP2_RAND2_PICK@r3175` | `r3175` | `1.591240` | `1.563656` | `1766.18` | `[1748.02, 1784.67]` |
| `ACTIVE_2PLUS2_PICK@r3175` (diag) | `r3175` | `1.598270` | `1.572064` | `1770.60` | `[1752.03, 1791.31]` |
| **`TOP4_PICK@r3175`** | **`r3175`** | **`1.606920`** | **`1.570568`** | **`1795.56`** | **`[1776.03, 1816.15]`** |
| `TOP4_PICK_UNGATED@r3175` (unequal) | `r3175` | `1.614871` | `1.580051` | `1800.44` | `[1780.86, 1821.22]` |

#### Key `DL-Train` OOF Contrasts (`205` families):
- **`ACTIVE_2PLUS2_HURDLE - TOP4_PICK`**: median family gain **`-0.102791`** (`95% CI [-0.127965, -0.080421]`, median context gain `-0.111870`, `0/3` seeds, W/T/L `0/6/199`, harm `>0.01` = `86.83%`).
- **`TOP4_PICK - GATED_025`**: median family gain **`+0.183951`** (`95% CI [+0.136392, +0.245238]`, median context gain `+0.183101`, `3/3` seeds, W/T/L `199/6/0`, harm `>0.01` = `0.00%`).
- **`ACTIVE_2PLUS2_HURDLE - ACTIVE_2PLUS2_PICK`**: median family gain **`-0.104655`** (`95% CI [-0.128478, -0.081265]`, W/T/L `0/7/198`).
- **`ACTIVE_2PLUS2_PICK - TOP4_PICK`**: median family gain **`+0.001860`** (`95% CI [-0.018875, +0.021052]`, median context gain `-0.008634`, `p = 0.4882`, W/T/L `98/8/99`).
- **`ACTIVE_2PLUS2_HURDLE@r3175 - TOP4_PICK@r3175`**: median family gain **`-0.104441`** (`95% CI [-0.130023, -0.081251]`, W/T/L `0/5/200`).
- **`TOP4_PICK_UNGATED - TOP4_PICK` (unequal budget: `1,219` vs `1,200` contexts revealed)**: median family gain **`+0.009483`** (`95% CI [+0.001013, +0.024365]`, W/T/L `12/193/0`).

---

## 5. Mechanism Decomposition & Verdict

### Why `ACTIVE_2PLUS2_HURDLE` Loses by `-0.083` (`DL-Val`) and `-0.103` (`DL-Train` OOF) to `TOP4_PICK`
`H258` cleanly isolates two mechanisms through the diagnostic arm `ACTIVE_2PLUS2_PICK`:
1. **Indirect Pool-Shifting vs Direct Best-of-Revealed Selection (`-0.0766` on `DL-Val`, `-0.1047` on `DL-Train` OOF)**:
   - On gated contexts, `ACTIVE_2PLUS2_HURDLE` reveals `4` true candidate evaluations (`rev4`), uses their residuals only to estimate a 2-stage affine shift (`shift1`, `shift2`) on the predicted costs of all `1,152` pool candidates, and then selects `argmin(s_full)` across the entire `1,152`-candidate pool — frequently selecting an **unrevealed** candidate whose true cost is much worse than the best already-revealed candidate in `rev4`.
   - Simply changing the selection rule on the **exact same `rev4` set** from `argmin(s_full)` (`ACTIVE_2PLUS2_HURDLE`) to `argmin` of the `4` revealed gated costs (`ACTIVE_2PLUS2_PICK`) increases `DL-Val` median family utility from `1.168790` to `1.245932` (`+0.076644` median family gain, `+248.72` Elo) and `DL-Train` OOF median family utility from `1.466442` to `1.571093` (`+0.104655` median family gain, `+248.22` Elo).
2. **Hurdle-Critic Slot Selection (`rev4`) vs Simple `r882` Top-4 (`-0.0061` on `DL-Val`, `+0.0019` on `DL-Train` OOF)**:
   - Once direct best-of-4 selection is used, replacing the 3rd and 4th `r882` candidates (`TOP4_PICK`) with the Stage-1 hurdle critic's 2 picks from `idx4 \ top2` (`ACTIVE_2PLUS2_PICK`) produces no statistically significant gain (`-0.006068` on `DL-Val`, `95% CI [-0.030003, +0.010652]`; `+0.001860` on `DL-Train` OOF, `95% CI [-0.018875, +0.021052]`, context gain `-0.008634`, Elo `1772.99` vs `TOP4_PICK`'s `1793.96`).

### Verdict: **`NO_ADVANCE`**
- All preregistered `ADVANCE` gates for `ACTIVE_2PLUS2_HURDLE` vs `TOP4_PICK` fail (`median_family_gain = -0.083210 <= 0`, `holm_lcb = -0.133392 <= 0`, `holm_p = 1.0 >= 0.05`, `0/3` positive seeds).
- **Decision**:
  1. **Reject `ACTIVE_2PLUS2_HURDLE` and `GATED_025`**: Indirect affine pool-shifting from 4 reveals throws away ~85% of the utility gain available from the 4 revealed evaluations themselves.
  2. **Establish `TOP4_PICK` as the canonical 4-reveal search baseline**: Any future `k`-reveal test-time search or active-query policy must beat `TOPk_PICK` (reveal top-`k` by incumbent predicted cost, pick `argmin` revealed gated cost) at matched reveal budget `k`.
  3. **Separate zero-reveal checkpoint evaluation (`k = 0`) from `k`-reveal search (`k = 4`)**: Checkpoint promotion gates remain strictly zero-reveal (`R882` / `R3175` at `k = 0`), while `TOP4_PICK` defines the `k = 4` search benchmark (`1.252000` family / `1.240360` context utility on `Native91`).

---

## 6. Durable S3 Custody & Execution Receipts

- **Preregistration S3 Custody**:
  - `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H258/prereg-df196c1f9e570e544a0002c2651b4ee75847154b/prereg_s3_manifest.json` (`VersionId = Z91vuhz0CzFHICyqVRcfJfvsxl4mxAPI`, `sha256 = ab73edab87f272ecfb9707480db6550fd07871737d501b0cad19d5b291d02e23`)
- **Primary Run (`attempt-006`) S3 Custody**:
  - `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H258/attempt-006/s3_manifest.json` (`VersionId = 06zqRmxpVe_5B0zDThaHnBTLONe.IwHj`, `sha256 = 56e7018f7c95c7072589c8c738f97138e7bc305a2ce63dd6e0ae17f75a680fe8`)
  - `metrics.json`: `VersionId = O8CbOry_nDoCXwtRyDLH4Q4K0ZxOKv32` (`sha256 = 7bc52fd90b01e2c1c7c1abc2fa66d0156129f78050e90afe0bc980a1c26b7eaf`)
  - `baseline.json`: `VersionId = Qp1o0r827CIYzjuJXpsU1bA8WUbVgGjE` (`sha256 = a7d8f15447c45dc3f7812836a9f514f8c687387a519d14a3cb2fe9994ef3c3ec`)
  - `panels.npz`: `VersionId = FynDoFyCnRMb.c2j8hArSd6RafIr7z.I` (`sha256 = f447610e4a9e5c4cf41d52c40209dc8f01fb9ab7124df6b8e145416d9c6976a0`)
  - `attempt-006.resource_summary.json`: `VersionId = bCM6TQfvmUmwase9FNwln3o0iiqwTs34` (`sha256 = cba29aec895e5bf07d99d28484568cfbbba110ea6ea26122ea2824deebb33230`)
  - `smoke-001/smoke.json`: `VersionId = 2CdvvatIxCrDQh.uj0J3sbP.kbjUyGjP` (`sha256 = 081333a5052d491d9572027725a81ecc9f852f96546f64c9453e1d38462f4f72`)
- **Compute Resource Receipt (`attempt-006`)**:
  - `status = PASS`, `exit_code = 0`, wall `160.46s` (including `78.89s` cooperative pauses across `3` host memory dips below `30 GiB`), peak RSS `1,536,159,744 B` (`1.43 GiB <= 4.0 GiB`), peak threads `2 <= 4`, CPU affinity `[57, 58, 59, 60]`, `0` GPU.
