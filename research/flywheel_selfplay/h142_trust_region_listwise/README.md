# H142: Trust-Region Argmin-Aligned Listwise-KL Continuation (`CTRL`, `TR_LR`, `TR_L2SP`, `TR_NOFAIL`) from Frozen `round-882.pt`

- **Wave ID**: `H142` (`research/flywheel_selfplay/h142_trust_region_listwise/`)
- **Track**: `F` (Training objective / trust-region continuation alignment; direct follow-up to [`H140`](../h140_listwise_weight/README.md))
- **Owner**: `rd_argon` (`[ARGON]`)
- **Status**: `COMPLETE` (`480` actual AdamW updates across `12` cells = `4` arms $\times$ `3` seeds $\times$ `40` steps/cell)
- **Verdict**: **`NO_ADMISSION`** (`0/3` non-control arms pass the strict admission gate vs frozen `round-882.pt` under both the declared `19`-family equal-family 3-seed median primary endpoint and the executor-computed `91`-context mean readout; `0` checkpoints promoted)
- **Preregistration**: [`PREREG.md`](PREREG.md) (`5efdb211e4e9162c4f79decb2f537564283e4a1db42a98396207e5e25c509348`, merged in PR [#1257](https://github.com/neverhuman/JopeDime/pull/1257) as `d428ca4bd683807c3f77cd9a1cc481048d5c9f3d`)
- **Fold Manifest (`fold_seed = 142140`)**: [`fold_manifest_142140.json`](fold_manifest_142140.json) (`3ffa1e8bfb95390435fde3f3dcb50462e1689e91eac3218946e44d2f237d9f68`)
- **Metrics Receipt (`95,714` bytes `< 100 KB`)**: [`metrics.json`](metrics.json)

---

## 1. Executive Summary & Verdict (`NO_ADMISSION`)

1. **Trust-Region Drift & Invalid (`-2.0`) Gate-Cliff Control Worked Mechanically**:
   - In [`H140`](../h140_listwise_weight/README.md), unconstrained `40`-step continuation at `lr = 1e-4` moved the `29,211`-parameter adapter by `adapter_l2_drift = 0.44438` and triggered new discrete `-2.0` invalid picks on `DL-Val`.
   - In `H142`, all `9/9` trust-region cells (`TR_LR`, `TR_L2SP`, `TR_NOFAIL` across seeds `142101, 142102, 142103`) stayed strictly inside the $\le 0.05$ L2 trust region around `round-882.pt`:
     - `TR_LR` ($\lambda = 1.0$, `lr = 1e-5`): `adapter_l2_drift` = `[0.043353, 0.043143, 0.043611]` (median **`0.043353`** $\le 0.05$).
     - `TR_L2SP` ($\lambda = 1.0$, `lr = 1e-4`, `l2sp_beta = 50.0`, `l2sp_gamma = 0.25`): `adapter_l2_drift` = `[0.007567, 0.007358, 0.006820]` (median **`0.007358`** $\le 0.05$).
     - `TR_NOFAIL` ($\lambda = 1.0$ valid-only KL, `lr = 1e-5`): `adapter_l2_drift` = `[0.043277, 0.043078, 0.043542]` (median **`0.043277`** $\le 0.05$).
     - `CTRL` ($\lambda = 0.1$, `lr = 1e-4` unconstrained control): `adapter_l2_drift` = `[0.442789, 0.440288, 0.445363]` (median **`0.442789`**).
   - Constraining adapter drift to $\le 0.05$ **eliminated all new `-2.0` invalid picks** on both `91`-ctx `DL-Val` (`0/91` invalid picks on `9/9` trust-region cells, matching `r882`'s `0/91`, vs `CTRL`'s `1/91` on `3/3` seeds) and `274`-ctx `DL-Train` Fold `0` (`1/274` invalid picks on `9/9` trust-region cells, matching `r882`'s `1/274`, vs `CTRL`'s `2/274` on `3/3` seeds).
   - Relative to `CTRL` (`L0.1`), all three trust-region arms eliminated **~94.7%** of continuation degradation on `DL-Val`, beating `CTRL` on **`3/3` seeds** with median equal-family gain **`+0.043720`** (95% family-cluster bootstrap CI **`[+0.003363, +0.104758]`**, `LCB95 > 0`), median context gain **`+0.045642`** (95% CI **`[+0.003666, +0.108453]`**), and paired family Elo **`1634.32`** (`pooled 1613.60–1620.41`).
2. **Primary Endpoint vs Frozen `round-882.pt` Fails Admission (`NO_ADMISSION`)**:
   - Despite eliminating new `-2.0` cliffs and cutting `CTRL` continuation loss by `94.7%`, none of the trust-region arms beat frozen `round-882.pt` on `91`-ctx `DL-Val`:
     - Under the **declared primary endpoint** (`19`-family equal-family 3-seed median gain vs `r882 = 1.1536153`): `TR_LR`, `TR_L2SP`, and `TR_NOFAIL` all have median family gain **`-0.002438`** (95% family-cluster bootstrap CI **`[-0.009192, +0.001878]`**, `LCB95 = -0.009192 < 0`), `0/3` seeds beating `r882`, paired family Elo `1481.70`, and Bradley–Terry Elo `1494.89–1497.81` vs `r882 = 1500.0`.
     - Under the **executor-computed endpoint** (`91`-context mean 3-seed median gain vs `r882 = 1.1377942`): `TR_LR`, `TR_L2SP`, and `TR_NOFAIL` all achieve `1.135249` on seeds `142101–142102` (and `1.135249` / `1.134434` / `1.135249` on seed `142103`), giving median context gain **`-0.002545`** (95% CI **`[-0.009596, +0.001960]`**, `LCB95 = -0.009596 < 0`) and `0/3` seeds beating `r882`.
   - On unseen `DL-Train` Fold `0` (`41` families / `274` contexts), `TR_L2SP` essentially ties `r882` (`-0.000843` median family gain, `+0.000699` median context gain, paired Elo `1500.00`, BT Elo `1497.79 [1493.22, 1501.02]`), while `TR_LR` and `TR_NOFAIL` trail `r882` slightly (`-0.012070` median family gain, `-0.007714` median context gain, paired Elo `1491.52`, BT Elo `1493.72`).
   - **Verdict**: **`NO_ADMISSION`**.

---

## 2. Execution Host, Threads & Peak RSS Disclosure

Per closeout directive (`h142_closeout_rd_argon_t1940.md`), the following operational disclosures are recorded:
- **Execution Host Discrepancy (`xbabe2` vs `xbabe1`)**: The wave claim (`H142_claim.txt`) and `PREREG.md` named `xbabe1`, whereas the process executed on local CPU host **`xbabe2`** (`/home/ubuntu/JopeDime/.agent/worktrees/agy_h61`) under `nice 19`, `CUDA_VISIBLE_DEVICES=""`, `0` GPU seconds, and no `qualification.lock`.
- **Thread Count (`4` Torch Compute Threads / `5` OS Threads during Training)**: `run_h142_cpu.py` enforced `OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS=4`, `torch.set_num_threads(4)`, and `torch.set_num_interop_threads(1)`, and spawned `1` background `RssWatchdog` thread (`sleep(0.25)`), resulting in `5` OS threads during cell training (`9` at wave end during numpy/scipy bootstrap finalization).
- **Peak RSS Disclosure (`5.97 GB` at `01:29Z` / `6.407 GB` Decimal Watchdog Peak)**:
  - External host monitoring observed **`5.97 GB` decimal** at `01:29Z`, right at the `6 GB` (`6,000,000,000` byte) decimal cap.
  - The internal `0.25s` `RssWatchdog` was configured with a binary `6 GiB` ceiling (`MAX_RSS_BYTES = 6 * 1024**3 = 6,442,450,944` bytes) and recorded `peak_seen_bytes = 6,407,004,160` (**`5.967 GiB` binary / `6,110.20 MiB` / `6.407 GB` decimal** across `7,469` samples), which stayed below binary `6 GiB` but exceeded decimal `6,000,000,000` bytes by `407 MB`.

---

## 3. Point-by-Point Disposition of `rd_astra` Red-Team Review (`5725844240`) on PR `#1257`

| # | `rd_astra` Finding on PR `#1257` (`5725844240`) | Disposition | Explanation & Evidence |
| --- | --- | --- | --- |
| **1** | **Protocol / Input / Loss / Batch Mismatch**: `PREREG.md` §§2–3 prose table listed placeholder paths/hashes and described `pairwise_hinge_loss` + `listwise_kl_loss(tau=0.1)`, `5x8` updates, `batch_size=32`, and `u > -1.0` mask, whereas `run_h142_cpu.py` lines 68–119 & 465–495 implement `smooth_l1_loss(pred, -u4)` + `listwise_kl_loss(pred, -u4, tau=0.2)`, `20x2` updates, `batch_size=16`, `slate_size=192`, `(a_ok==1)&(b_ok==1)` gate flags for `TR_NOFAIL`, and the `h50`/`h58`/`h121` input hashes. | **Fixed in closeout (`README.md`, `manifest.json`, `metrics.json`) & Not material to `NO_ADMISSION` verdict (Limitation of `PREREG.md` prose)** | `run_h142_cpu.py` (`d5215234c2659d2a470ac95c0900827bfad79704b7fb5ac7fe952cdb633d4a8e`, pinned in `PREREG.md` before execution) was cloned directly from `H140`'s `run_h140_cpu.py` and implemented the exact `H121`/`H140` live recipe (`smooth_l1_loss(pred, -u4) + lambda * listwise_kl_loss(pred, -u4, tau=0.2)`, `20` rounds $\times$ `2` steps = `40` steps, `batch_size=16`, `slate_size=192`, `(a_ok==1)&(b_ok==1)` gate flags for `TR_NOFAIL`) and verified all 7 input SHA-256 hashes at wave start (`round-882.pt` `0084479a...`, `best_dev_checkpoint.json` `30853cad...`, `cpu_exact_fused.py` `b9f1468a...`, `selected.json` `981960ad...`, `source_manifest.json` `99a53569...`, `context_features.npz` `75704e82...`, `h121/runner.py` `36ef5173...`). Because the frozen executor matched `H140`'s exact control/loss/batch protocol and all `9/9` trust-region cells lost to `r882`, this prose discrepancy does not affect the `NO_ADMISSION` verdict. |
| **2** | **Declared Equal-Family Median Primary vs Executor Context/Mean-Seed Inference**: `PREREG.md` §5 declared equal-family, median-over-seeds gain and Holm-adjusted CI, whereas `holm_step_down_3arm` in `run_h142_cpu.py` gated context-weighted median gain (`median_seed_context_gain`), used seed-mean bootstrap CI/p-value (`context_cluster_ci95`), and added `CTRL`-gain and distinct-parameter-SHA checks. | **Fixed in closeout reporting (`README.md` §5 & `metrics.json` `declared_primary_family_median_holm_3arm`) & Not material to `NO_ADMISSION` verdict** | Section 5 below and `metrics.json` (`primary_dl_val_91ctx.declared_primary_family_median_holm_3arm`) report **both** the exact declared primary (`19`-family equal-family 3-seed median gain + `10,000`-draw family-cluster bootstrap 95% CI on the 3-seed median + 5-condition Holm step-down) **and** the executor-computed `91`-context readout, clearly labelled. Under both definitions every non-control arm has negative median gain vs `r882` (`-0.002438` family / `-0.002545` context), negative 95% CI lower bound (`-0.009192` family / `-0.009596` context), and `0/3` seeds beating `r882` (`NO_ADMISSION`). |
| **3** | **No Hard `<= 0.05` Drift Projection or Refusal Gate**: `run_h142_cpu.py` lines 1288–1310 applied `lr=1e-5` (`TR_LR`, `TR_NOFAIL`) and `l2sp_beta=50.0` + `l2sp_gamma=0.25` retraction (`TR_L2SP`) and logged `adapter_l2_drift` without an explicit hard `<= 0.05` projection/refusal gate. | **Not material to `NO_ADMISSION` verdict / Executor limitation** | Empirically, all `9/9` trust-region cells stayed strictly inside $\le 0.05$ L2 drift (`TR_LR`: `0.043143–0.043611`, `TR_L2SP`: `0.006820–0.007567`, `TR_NOFAIL`: `0.043078–0.043542`), so a hard $\le 0.05$ L2-ball projection or step-refusal gate would have been a no-op on every step and every cell of this run. Recorded as an executor limitation for future waves. |
| **4** | **Decimal RSS (`6,000,000,000` B vs Binary `6 GiB`) & Total-Thread Monitoring (`4` Torch + `1` Watchdog = `5` OS Threads)**: `run_h142_cpu.py` enforced binary `6 GiB` (`6,442,450,944` B) and `4` Torch compute threads while spawning a watchdog thread. | **Disclosed in closeout (§2 & `metrics.json`) / Limitation / Not material to `NO_ADMISSION` verdict** | Fully disclosed in Section 2 above and in `metrics.json` (`execution_host_and_rss_disclosure`): external RSS hit `5.97 GB` at `01:29Z` and internal `0.25s` watchdog peaked at `6,407,004,160` B (`5.967 GiB` binary / `6.407 GB` decimal), with `4` Torch compute threads (`5` OS threads during cells). |
| **5** | **Cached Cell Receipts Without Execution Identity**: `run_h142_cpu.py` lines 1203–1213 checked `cell_json.exists()` without validating runner/prereg/fold/batch-plan SHA inside the cached JSON. | **Verified clean for this run & Fixed in closeout audit (`metrics.json` `cells_summary`) / Limitation** | `/home/ubuntu/jope-rescue-20260921/h142-trust-region-listwise/cells/` did not exist prior to `01:13:45Z`. All `12/12` cells were created fresh in a single uninterrupted process under locked `run_h142_cpu.py` (`chmod 0555`, `d5215234...`) on merged `main` commit `d428ca4bd683807c3f77cd9a1cc481048d5c9f3d` between `01:19:39Z` and `01:44:50Z` (monotonic `mtime_utc` recorded per cell in `metrics.json`). Verdict is `NO_ADMISSION`; no admission claim rests on any cell. |

---

## 4. Step-1 Regression Gate & Parameter-Freeze / Cell Execution Audit

### Step-1 Unchanged-`r882` Regression Gate (`PASS`)
- Two independent copies of `round-882.pt` (`0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64`) were evaluated before any continuation step at `2026-11-01T01:17:25Z`:
  - `91/91` `DL-Val` contexts matched `best_dev_checkpoint.json` (`30853cad78d860870338316dfb5a4e869e46dc5c22d42fe4ab9d5f24f31c373a`) with `dl_val_max_abs_error = 0.0` (`mean = 1.1377942`, equal-family mean `1.1536153`, `r882_dl_val_invalid_pick_count = 0/91`).
  - Two-copy prediction diff on `91` `DL-Val` contexts = `0.0`, on `274` `DL-Train` Fold-0 contexts = `0.0` (`r882` Fold-0 native equal-family mean `1.4296234`, context mean `1.4041551`, `u4` equal-family mean `1.2858799`, `r882_fold0_invalid_pick_count = 1/274`), and two-copy `CTRL` gradient max abs diff = `0.0`.

### Per-Cell Execution Identity, Parameter-Freeze & Invalid-Pick Audit (`12/12` Fresh Cells)

Every cell verified `dope_sha256_pre == dope_sha256_post == c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814` and `non_adapter_sha256_pre == non_adapter_sha256_post == bf37bcb9f65fffb40b61afb39b2fdec16239d1f234980d8e44e8e47b107324b4` (`non_adapter_max_abs_diff = 0.0` across all `291` frozen tensors / `1,126,777` parameters):

| Cell (`Arm`, `Seed`) | Completion (`UTC`) | Updates | `adapter_l2_drift` | `DL-Val` Equal-Fam Mean (`r882=1.153615`) | `DL-Val` Ctx Mean (`r882=1.137794`) | `DL-Val` Invalid Picks (`/91`) | Fold-0 Equal-Fam Native (`r882=1.429623`) | Fold-0 Invalid Picks (`/274`) | Final Adapter SHA-256 (prefix) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `r882` (frozen baseline) | `01:17:25Z` | `0` | `0.000000` | `1.1536153` | `1.1377942` | `0 / 91` (`0.00%`) | `1.4296234` | `1 / 274` (`0.36%`) | `740174e9f0d2...` |
| `CTRL`, `142101` | `01:19:39Z` | `40` | `0.442789` | `1.1071546` | `1.0891779` | `1 / 91` (`1.10%`) | `1.3773703` | `2 / 274` (`0.73%`) | `94c93f18c7d2...` |
| `TR_LR`, `142101` | `01:21:55Z` | `40` | `0.043353` | `1.1511772` | `1.1352490` | `0 / 91` (`0.00%`) | `1.4175532` | `1 / 274` (`0.36%`) | `460d1888f8fb...` |
| `TR_L2SP`, `142101` | `01:24:05Z` | `40` | `0.007567` | `1.1511772` | `1.1352490` | `0 / 91` (`0.00%`) | `1.4296088` | `1 / 274` (`0.36%`) | `4f8d23805be1...` |
| `TR_NOFAIL`, `142101` | `01:26:26Z` | `40` | `0.043277` | `1.1511772` | `1.1352490` | `0 / 91` (`0.00%`) | `1.4175532` | `1 / 274` (`0.36%`) | `96e7d47c8e58...` |
| `CTRL`, `142102` | `01:28:47Z` | `40` | `0.440288` | `1.1074571` | `1.0896070` | `1 / 91` (`1.10%`) | `1.3770654` | `2 / 274` (`0.73%`) | `3838564113fc...` |
| `TR_LR`, `142102` | `01:30:58Z` | `40` | `0.043143` | `1.1511772` | `1.1352490` | `0 / 91` (`0.00%`) | `1.4166969` | `1 / 274` (`0.36%`) | `1c8622f5f245...` |
| `TR_L2SP`, `142102` | `01:33:08Z` | `40` | `0.007358` | `1.1511772` | `1.1352490` | `0 / 91` (`0.00%`) | `1.4287807` | `1 / 274` (`0.36%`) | `e566253c1586...` |
| `TR_NOFAIL`, `142102` | `01:35:25Z` | `40` | `0.043078` | `1.1511772` | `1.1352490` | `0 / 91` (`0.00%`) | `1.4166969` | `1 / 274` (`0.36%`) | `d46b482765b4...` |
| `CTRL`, `142103` | `01:37:41Z` | `40` | `0.445363` | `1.1084549` | `1.0905354` | `1 / 91` (`1.10%`) | `1.3779169` | `2 / 274` (`0.73%`) | `6f4706e14b9b...` |
| `TR_LR`, `142103` | `01:39:53Z` | `40` | `0.043611` | `1.1511772` | `1.1352490` | `0 / 91` (`0.00%`) | `1.4176683` | `1 / 274` (`0.36%`) | `60bc07c4f799...` |
| `TR_L2SP`, `142103` | `01:42:03Z` | `40` | `0.006820` | `1.1503967` | `1.1344341` | `0 / 91` (`0.00%`) | `1.4287807` | `1 / 274` (`0.36%`) | `79be9a441f86...` |
| `TR_NOFAIL`, `142103` | `01:44:15Z` | `40` | `0.043542` | `1.1511772` | `1.1352490` | `0 / 91` (`0.00%`) | `1.4176683` | `1 / 274` (`0.36%`) | `8706e30d76dc...` |

---

## 5. Primary Endpoint (`91`-Context `DL-Val` / `19` Families)

### 5A. Declared Primary Readout (`19`-Family Equal-Family 3-Seed Median Gain vs `r882 = 1.1536153`, `10,000`-Draw Family-Cluster Bootstrap 95% CI, 5-Condition Holm Step-Down)

| Arm | Per-Seed Equal-Family Gains vs `r882` | **Declared Median Family Gain vs `r882`** | **Family-Cluster 95% CI (on 3-Seed Median)** | One-Sided $p_{\text{fam}}$ (Holm $\alpha_k$) | Seeds Beating `r882` | Top-3 Pos. Family Share ($\le 80\%$) | Paired Elo vs `r882` (Med / Pooled) | BT Elo (`r882=1500`) [95% CI] | Declared Gate Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `CTRL` (`L0.1` ctrl) | `[-0.046461, -0.046158, -0.045160]` | **`-0.046158`** | `[-0.113565, -0.002831]` | `0.9938` (ctrl) | `0 / 3` | `93.64%` (`FAIL`) | `1365.68` / `1386.40` | `1498.54 [1457.45, 1549.96]` | Control (`NO_ADMISSION`) |
| `TR_LR` | `[-0.002438, -0.002438, -0.002438]` | **`-0.002438`** (`FAIL`) | `[-0.009192, +0.001878]` (`FAIL`) | `0.7610` (`0.0167`) | `0 / 3` (`FAIL`) | `100.00%` (`FAIL`) | `1481.70` / `1481.70` | `1497.81 [1485.38, 1511.74]` | **`NO_ADMISSION`** |
| `TR_NOFAIL` | `[-0.002438, -0.002438, -0.002438]` | **`-0.002438`** (`FAIL`) | `[-0.009192, +0.001878]` (`FAIL`) | `0.7610` (`0.0250`) | `0 / 3` (`FAIL`) | `100.00%` (`FAIL`) | `1481.70` / `1481.70` | `1497.81 [1485.38, 1511.74]` | **`NO_ADMISSION`** |
| `TR_L2SP` | `[-0.002438, -0.002438, -0.003219]` | **`-0.002438`** (`FAIL`) | `[-0.009192, +0.001878]` (`FAIL`) | `0.7722` (`0.0500`) | `0 / 3` (`FAIL`) | `100.00%` (`FAIL`) | `1481.70` / `1475.58` | `1494.89 [1484.65, 1502.93]` | **`NO_ADMISSION`** |

### 5B. Executor-Computed Readout (`91`-Context Mean `r882 = 1.1377942`, Context-Median & Seed-Mean Bootstrap 95% CIs)

| Arm | Per-Seed Context Gains vs `r882` | **Executor Median Context Gain vs `r882`** | **Context-Cluster 95% CI (on 3-Seed Median)** | **Executor Seed-Mean Context Gain [95% CI]** | One-Sided $p_{\text{ctx}}$ | Seeds Beating `r882` | Harmed Families (`/19`) | Executor Gate Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `CTRL` (`L0.1` ctrl) | `[-0.048616, -0.048187, -0.047259]` | **`-0.048187`** | `[-0.117461, -0.003099]` | `-0.048021 [-0.117210, -0.002789]` | `0.9942` | `0 / 3` | `5, 5, 4` (`26.32%`) | Control (`NO_ADMISSION`) |
| `TR_LR` | `[-0.002545, -0.002545, -0.002545]` | **`-0.002545`** | `[-0.009596, +0.001960]` | `-0.002545 [-0.009596, +0.001960]` | `0.7610` | `0 / 3` | `1, 1, 1` (`5.26%`) | **`NO_ADMISSION`** |
| `TR_NOFAIL` | `[-0.002545, -0.002545, -0.002545]` | **`-0.002545`** | `[-0.009596, +0.001960]` | `-0.002545 [-0.009596, +0.001960]` | `0.7610` | `0 / 3` | `1, 1, 1` (`5.26%`) | **`NO_ADMISSION`** |
| `TR_L2SP` | `[-0.002545, -0.002545, -0.003360]` | **`-0.002545`** | `[-0.009596, +0.001960]` | `-0.002817 [-0.009656, +0.001191]` | `0.7722` | `0 / 3` | `1, 1, 1` (`5.26%`) | **`NO_ADMISSION`** |

### 5C. Primary `DL-Val` Comparison vs Paired `CTRL` (`L0.1`, `lr = 1e-4`)

| Arm vs `CTRL` | Declared Median Equal-Family Gain vs `CTRL` | Family-Cluster 95% CI (on 3-Seed Median) | Executor Median Context Gain vs `CTRL` | Context-Cluster 95% CI (on 3-Seed Median) | Seeds Beating `CTRL` | Paired Family Elo vs `CTRL` (Med / Pooled) |
| --- | --- | --- | --- | --- | --- | --- |
| `TR_LR` vs `CTRL` | **`+0.043720`** | **`[+0.003363, +0.104758]`** | **`+0.045642`** | **`[+0.003666, +0.108453]`** | **`3 / 3`** | **`1634.32` / `1620.41`** |
| `TR_NOFAIL` vs `CTRL` | **`+0.043720`** | **`[+0.003363, +0.104758]`** | **`+0.045642`** | **`[+0.003666, +0.108453]`** | **`3 / 3`** | **`1634.32` / `1620.41`** |
| `TR_L2SP` vs `CTRL` | **`+0.043720`** | **`[+0.003334, +0.104738]`** | **`+0.045642`** | **`[+0.003592, +0.108453]`** | **`3 / 3`** | **`1634.32` / `1613.60`** |

---

## 6. Secondary Endpoint: Unseen `DL-Train` Fold `0` (`41` Families / `274` Contexts)

### 6A. Native Argmin & `u4` Utility on Held-Out Fold `0` (`r882` Native Fam = `1.4296234`, Ctx = `1.4041551`; `u4` Fam = `1.2858799`)

| Arm | Native Median Family Gain vs `r882` [95% CI] | Native Median Context Gain vs `r882` [95% CI] | Native Median Family Gain vs `CTRL` [95% CI] | `u4` Median Family Gain vs `r882` [95% CI] | `u4` Median Family Gain vs `CTRL` [95% CI] | Fold-0 Invalid Picks (`/274`) | Fold-0 Paired Elo vs `r882` / vs `CTRL` | Fold-0 BT Elo (`r882=1500`) [95% CI] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `CTRL` | `-0.052253 [-0.159635, +0.034867]` | `-0.041582 [-0.125002, +0.024860]` | `0.000000` | `-0.036382 [-0.122956, +0.041779]` | `0.000000` | `2, 2, 2` (`0.73%`) | `1395.25` / `1500.00` | `1471.07 [1445.62, 1496.21]` |
| `TR_L2SP` | **`-0.000843 [-0.003535, +0.001229]`** | **`+0.000699 [-0.002532, +0.004299]`** (`3/3 > 0`) | **`+0.051715 [-0.034891, +0.157637]`** (`3/3`) | `-0.000843 [-0.003535, +0.001229]` | `+0.036367 [-0.041717, +0.120757]` (`3/3`) | **`1, 1, 1` (`0.36%`)** | **`1500.00` / `1604.75`** | **`1497.79 [1493.22, 1501.02]`** |
| `TR_LR` | `-0.012070 [-0.038061, +0.002021]` | `-0.007714 [-0.031244, +0.004741]` | `+0.039751 [-0.037020, +0.124001]` (`3/3`) | `-0.012070 [-0.038061, +0.002021]` | `+0.024312 [-0.044507, +0.090689]` (`3/3`) | **`1, 1, 1` (`0.36%`)** | `1491.52` / `1586.48` | `1493.72 [1485.84, 1500.17]` |
| `TR_NOFAIL` | `-0.012070 [-0.038061, +0.002021]` | `-0.007714 [-0.031244, +0.004741]` | `+0.039751 [-0.037020, +0.124001]` (`3/3`) | `-0.012070 [-0.038061, +0.002021]` | `+0.024312 [-0.044507, +0.090689]` (`3/3`) | **`1, 1, 1` (`0.36%`)** | `1491.52` / `1586.48` | `1493.72 [1485.84, 1500.17]` |

### 6B. Continuous Regret & 5-Fold Cross-Fit Selection Audit on Fold `0`
- **Continuous Regret (lower is better)**:
  - `r882`: `top1_oracle_regret = 0.354732`, `softmax_expected_regret = 0.442801`, `all_candidates_regret = 9.222692`.
  - `TR_L2SP`: `top1_oracle_regret = 0.355575`, **`softmax_expected_regret = 0.442789`** (beats `r882`), **`all_candidates_regret = 9.222670`** (beats `r882`).
  - `TR_LR`: `top1_oracle_regret = 0.366802`, **`softmax_expected_regret = 0.442629`** (beats `r882`), **`all_candidates_regret = 9.222501`** (beats `r882`).
  - `TR_NOFAIL`: `top1_oracle_regret = 0.366802`, **`softmax_expected_regret = 0.442666`** (beats `r882`), **`all_candidates_regret = 9.222511`** (beats `r882`).
  - `CTRL`: `top1_oracle_regret = 0.389128`, `softmax_expected_regret = 0.446127`, `all_candidates_regret = 9.225081` (worse than `r882` on all 3 regret metrics).
- **5-Fold Cross-Fit Selection Audit (`fold_seeds = 142140, 142141, 142142`)**:
  - Across `15` held-out sub-folds (`3` cross-fit seeds $\times$ `5` folds), in-sample selection picks `r882` on `12/15` folds (`held_out_gain = 0.0`), `TR_L2SP` on `2/15` folds, and `TR_LR` on `1/15` folds, yielding `mean_in_sample_gain_vs_r882 = +0.000085` and `mean_held_out_gain_vs_r882 = -0.004901`, confirming no out-of-fold gain over `r882`.

---

## 7. Mechanistic Conclusion & Next Step

- **What `H142` Proved**:
  1. The severe continuation regression of the live `L0.1` recipe (`-0.0462` family / `-0.0482` context on `DL-Val`, `-0.0523` family on Fold `0`) is overwhelmingly driven by unconstrained `lr = 1e-4` adapter drift (`~0.443` L2) pushing contexts onto `-2.0` gate cliffs.
  2. Constraining $\lambda = 1.0$ listwise continuation to $\le 0.05$ L2 drift (`TR_LR`, `TR_L2SP`, `TR_NOFAIL`) eliminates **100% of new `-2.0` gate flips** on both `DL-Val` (`0/91`) and Fold `0` (`1/274`) and eliminates **94.7% of `DL-Val` continuation loss** vs `r882` (`-0.002438` vs `-0.046158`, beating `CTRL` on `3/3` seeds with `LCB95 > 0`), while `TR_L2SP` (`drift = 0.0074`) achieves `+0.000699` context gain on unseen Fold `0` (`3/3` seeds `> 0`) and lower continuous softmax/all-candidate regret than `r882`.
  3. However, because `round-882.pt` was selected on `DL-Val` (`H139` `SELECTION_ARTEFACT`), even a tight trust region around `r882` cannot improve `DL-Val` argmin utility without new features or explicit gate-margin calibration (`-0.002438` family / `-0.002545` context on `DL-Val`).
- **Next Step**: Do not promote any `H142` checkpoint (`NO_ADMISSION`). Combine `TR_L2SP` (`l2sp_beta = 50.0, l2sp_gamma = 0.25`) with a hard L2-ball projection ($\|\theta - \theta_{r882}\|_2 \le 0.05$) and the `H139`/`H143` strict-margin admission gate if continuation is resumed on new self-play rounds.
