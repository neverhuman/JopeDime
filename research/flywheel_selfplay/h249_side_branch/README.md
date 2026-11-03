# H249: Zero-Init Low-Rank Residual Side Branch on Frozen `r882` Adapter (`h249-side-branch-20261103`)

**Status: complete. Verdict: `NO_ADVANCE`.** The design, arms, hyperparameters, fold partition (`fold_seed = 249140`), and decision rule (`Holm m = 2`) were preregistered in [`PREREG.md`](PREREG.md) (PR `#2099`, merged as `94151e444ffe4561a24f830f7068775b4ccd6659`, with outcome-blind `schedules.json` audit-boundary fix in PR `#2101`, merged as `c97ed930a789045cec2575d5c76629464fde62b7`) before any outcome was read. `attempt-002` ran all `45` cells (`15` cells × `3` arms × `40` updates = **`1,800` optimizer updates**) on `xbabe2` CPU (`taskset -c 52-55`, `nice -n 19`) in `2,070.68 s` (`~34.5 min`), peak RSS `3.14 GiB`, `0` pauses, `0` protected reads.

- **Strict-selector decision (`DL-Val91`, `91` contexts, `19` families)**:
  - **`SIDE4` (`r = 4`, `6,232` params, primary)**: ties `r882` exactly (`median_gain = 0.0000 [0.0000, 0.0000]`, per-seed `[0.0, 0.0, 0.0]`, `0/3` positive seeds, `Holm p = 1.0`, Elo `1500.0`; strict selector kept `r882` in `15/15` cells).
  - **`SIDE16` (`r = 16`, `24,928` params, secondary)**: ties `r882` at median (`median_gain = 0.0000 [0.0000, 0.0000]`, per-seed `[0.0, -0.000488, 0.0]`, `0/3` positive seeds, `Holm p = 1.0`, median Elo `1500.0`; strict selector kept `r882` in `14/15` cells, selecting step `30` in `A3-249102`).
  - **`CONTROL` (exact H170 `CTRL` full-adapter AdamW, `29,211` params)**: `median_gain = 0.0000 [0.0000, 0.0000]`, per-seed `[0.0, -0.000839, 0.0]`, `0/3` positive seeds (kept `r882` in `14/15` cells, selecting step `10` in `A2-249102`).
- **Unconstrained Step-40 diagnostic (before strict fallback)**:
  - Freezing `r882` and training only the zero-initialised low-rank side branch **eliminates 95% of full-adapter step-40 utility loss**:
    - **`SIDE4` at step 40 (`DL-Val91`)**: `-0.0014 [-0.0040, +0.0006]` vs `r882`, beating `CONTROL` by **`+0.0289 [+0.0043, +0.0615]`** (`3/3` seeds, `p = 0.0050`), with effective merged-weight L2 drift **`0.0681`** (`5.6×` smaller than `CONTROL`'s `0.3791` and nearly `2×` smaller than H247 `MUON`'s `0.122`).
    - **`SIDE16` at step 40 (`DL-Val91`)**: `-0.0009 [-0.0036, +0.0016]` vs `r882`, beating `CONTROL` by **`+0.0275 [+0.0046, +0.0594]`** (`3/3` seeds, `p = 0.0032`), with effective merged-weight L2 drift **`0.1536`**.
    - **`CONTROL` at step 40 (`DL-Val91`)**: `-0.0300 [-0.0635, -0.0044]` vs `r882` (`8` harm families per seed).
    - On `1,219`-context `DL-Train` OOF at step 40, `SIDE4` (`-0.00005 [-0.00040, +0.00031]`) beats `CONTROL` (`-0.0362 [-0.0630, -0.0109]`) by **`+0.0362 [+0.0104, +0.0626]`** (`3/3` seeds, `p = 0.0027`), and `SIDE16` (`-0.0041`) beats `CONTROL` by **`+0.0321 [+0.0084, +0.0572]`** (`3/3` seeds, `p = 0.0046`).
    - Neither `SIDE4` nor `SIDE16` ever flips the H152 gate-cliff context `#27` (`1.1227` across all seeds at both `strict` and `step40`).
- **Bottom line**: Function-preserving low-rank residual branches strongly contain continuation drift and prevent step-40 collapse, but like H247 `MUON`, they still cannot extract positive out-of-fold ranking utility beyond `r882` from the same 40-update `CTRL` objective and training pools.

---

## Hypothesis

Every prior 40-update continuation that directly moved the `47` `r882` adapter tensors (`29,211` parameters) drifted away from `r882` and lost utility at step 40 (e.g., H247 `CONTROL` `-0.0263`), while zero-update weight-space interpolations and trims (H241–H244) rarely moved candidate argmins. **H249** tests whether a **function-preserving capacity expansion**—attaching a zero-initialised low-rank residual side branch $\Delta W = (B A) \cdot (\alpha / r)$ (`A ~ Kaiming-uniform`, `B = 0`, $\alpha = r$) to all `16` 2-D adapter weight matrices while keeping all `47` `r882` adapter tensors strictly frozen (`requires_grad = False`)—can learn additive ranking improvements over `r882` (> `+0.002` median `DL-Val91` family gain) without destructive drift.

---

## Setup

- **Frozen Backbone & Checkpoint**:
  - `DOPE` backbone (`c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`) frozen and verified before and after execution.
  - Read-only `r882` copy (`/home/ubuntu/jope-data/h249/inputs/round-882.pt`, SHA256 `0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64`; initial adapter hash `9c3ba940d9eb60aa2a39dd57c229c0952927ea1c2f22c372ae9764f43baa8669`).
  - `update-0` (`B = 0`) verified bit-identical (`torch.equal`) to `r882` on `Native91` (`1.1377941898164525`) before any update.
- **Arms (`Holm m = 2`)**:
  - **`CONTROL`**: exact H170 `CTRL` full-adapter AdamW (`lr = 1e-4`, `betas = (0.9, 0.999)`, `eps = 1e-8`, decoupled `wd = 1e-4`; `29,211` trainable parameters across `47` tensors).
  - **`SIDE4` (primary)**: rank $r = 4$, $\alpha = 4.0$ ($\alpha / r = 1.0$) on all `16` 2-D adapter matrices (`6,232` trainable parameters across `32` tensors; all `47` `r882` tensors `requires_grad = False`), trained with AdamW (`lr = 1e-4`, `betas = (0.9, 0.999)`, `eps = 1e-8`, decoupled `wd = 1e-2` on branch only).
  - **`SIDE16` (secondary)**: rank $r = 16$, $\alpha = 16.0$ ($\alpha / r = 1.0$) on all `16` 2-D adapter matrices (`24,928` trainable parameters across `32` tensors; all `47` `r882` tensors `requires_grad = False`), trained with the same branch-only AdamW (`lr = 1e-4`, `wd = 1e-2`).
- **Protocol**:
  - `205` `DL-Train` families (`1,219` contexts) partitioned into `5` family-grouped folds under fresh `fold_seed = 249140`; `3` seeds (`249101`, `249102`, `249103`), `boot_seed = 249150`.
  - `15` outer cells × `3` arms × `40` updates (`steps = [0, 10, 20, 30, 40]`, global grad clip `1.0`, identical H156 minibatches and data order) = `1,800` updates.
  - Strict inner `SELECT` selector (`gain > +0.001` AND family `LCB95 > 0`, else step `0` `r882`), `fsync`-locked to `{cell}-selection.json` before any outer `A` or `DL-Val91` assessment.
- **Deviation Note (`hw_t1100_argon_h249_fix.md`)**:
  - `attempt-001` (`INCOMPLETE_INFRA`, `120` updates, `0` completed cells, `0` outer/DL-Val91 outcomes read) exited at line 746 when `runtime_audit.py` refused `sha(HERE / "schedules.json")` because the capsule directory lived under `/home/ubuntu/jope-data`. Fixed outcome-blind in PR `#2101` (`c97ed930...`) and rerun from scratch across all `45` cells as `attempt-002`.

---

## Outcome and Metrics

### 1. Primary (`DL-Val91`, `91` contexts, `19` families) & Secondary (`DL-Train` Strict OOF, `1,219` contexts, `205` families) — Strict Selector

| Panel | Contrast | Median Family Gain `[CI95]` | Per-Seed Family Gains (`249101`, `249102`, `249103`) | Pos Seeds | One-Sided `p` / `Holm p` | Harm Families (`< -1e-4`) | Context Mean (`249101–249103`) | Median Elo `[CI95]` |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`DL-Val91` (Primary)** | `SIDE4 - r882` | **`+0.000000`** `[+0.000000, +0.000000]` | `0.000000, 0.000000, 0.000000` | `0/3` | `1.0000` / **`1.0000`** | `0, 0, 0` | `1.137794, 1.137794, 1.137794` | `1500.0` `[1500.0, 1500.0]` |
| **`DL-Val91` (Secondary)** | `SIDE16 - r882` | **`+0.000000`** `[+0.000000, +0.000000]` | `0.000000, -0.000488, 0.000000` | `0/3` | `1.0000` / **`1.0000`** | `0, 2, 0` | `1.137794, 1.137285, 1.137794` | `1500.0` `[1500.0, 1500.0]` |
| **`DL-Val91` (Control)** | `CONTROL - r882` | **`+0.000000`** `[+0.000000, +0.000000]` | `0.000000, -0.000839, 0.000000` | `0/3` | `1.0000` / — | `0, 4, 0` | `1.137794, 1.136918, 1.137794` | `1500.0` `[1500.0, 1500.0]` |
| **`DL-Val91`** | `SIDE4 - CONTROL` | `+0.000000` `[+0.000000, +0.000000]` | `0.000000, +0.000839, 0.000000` | `1/3` | `1.0000` / — | `0, 2, 0` | — | — |
| **`DL-Val91`** | `SIDE16 - CONTROL` | `+0.000000` `[+0.000000, +0.000000]` | `0.000000, +0.000352, 0.000000` | `1/3` | `1.0000` / — | `0, 2, 0` | — | — |
| **`DL-Train` OOF** | `SIDE4 - r882` | **`+0.000000`** `[+0.000000, +0.000000]` | `0.000000, 0.000000, 0.000000` | `0/3` | `1.0000` / `1.0000` | `0, 0, 0` | `1.423073, 1.423073, 1.423073` | `1500.0` `[1500.0, 1500.0]` |
| **`DL-Train` OOF** | `SIDE16 - r882` | **`+0.000000`** `[+0.000000, +0.000000]` | `0.000000, +0.000500, 0.000000` | `1/3` | `1.0000` / `1.0000` | `0, 2, 0` | `1.423073, 1.424238, 1.423073` | `1500.0` `[1500.0, 1500.0]` |
| **`DL-Train` OOF** | `CONTROL - r882` | **`+0.000000`** `[+0.000000, +0.000000]` | `0.000000, -0.000055, 0.000000` | `0/3` | `1.0000` / — | `0, 7, 0` | `1.423073, 1.422382, 1.423073` | `1500.0` `[1500.0, 1500.0]` |

- **Plateau gap (`DL-Val91` context mean vs `1.1389572`)**: `r882` and `SIDE4` sit at `-0.001163` in all `3` seeds; `SIDE16` sits at `[-0.001163, -0.001672, -0.001163]`; `CONTROL` sits at `[-0.001163, -0.002039, -0.001163]`.

### 2. Unconstrained Diagnostics (Step 40 & Best Inner-`SELECT` Step)

| Diagnostic Readout | `CONTROL - r882` | `SIDE4 - r882` | `SIDE16 - r882` | `SIDE4 - CONTROL` | `SIDE16 - CONTROL` |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Step 40, `DL-Val91`** | `-0.0300` `[-0.0635, -0.0044]` | **`-0.0014`** `[-0.0040, +0.0006]` | **`-0.0009`** `[-0.0036, +0.0016]` | **`+0.0289`** `[+0.0043, +0.0615]`, **`3/3`** (`p=.0050`) | **`+0.0275`** `[+0.0046, +0.0594]`, **`3/3`** (`p=.0032`) |
| **Step 40, `DL-Train` OOF** | `-0.0362` `[-0.0630, -0.0109]` | **`-0.00005`** `[-0.00040, +0.00031]` | `-0.0041` `[-0.0107, +0.0009]` | **`+0.0362`** `[+0.0104, +0.0626]`, **`3/3`** (`p=.0027`) | **`+0.0321`** `[+0.0084, +0.0572]`, **`3/3`** (`p=.0046`) |
| **Best `SELECT` step, `DL-Val91`** | `-0.0031` `[-0.0075, +0.0002]` | `-0.0006` `[-0.0020, +0.0003]` | `-0.0011` `[-0.0040, +0.0009]` | `+0.0025` `[-0.0003, +0.0062]`, `3/3` | `+0.0022` `[-0.0001, +0.0056]`, `3/3` |
| **Best `SELECT` step, `DL-Train` OOF** | `-0.0012` `[-0.0102, +0.0033]` | `-0.00006` `[-0.00040, +0.00024]` | `-0.0019` `[-0.0076, +0.0015]` | `+0.0011` `[-0.0034, +0.0101]`, `2/3` | `-0.0007` `[-0.0089, +0.0092]`, `1/3` |
| **Step-40 Elo (`DL-Val91`)** | `1491.78` `[1452.27, 1545.91]` | `1496.35` `[1487.20, 1503.66]` | `1497.26` `[1485.36, 1510.06]` | — | — |
| **Step-40 Harm Families (`DL-Val91`)** | `8, 8, 8` | **`2, 2, 2`** | **`3, 3, 3`** | — | — |
| **Step-40 Harm Families (`OOF`)** | `90, 89, 87` | **`7, 8, 6`** | `20, 24, 16` | — | — |
| **Step-40 Merged-Weight L2 Drift** (`median [min, max]`) | `0.3791` `[0.3531, 0.3957]` | **`0.0681`** `[0.0631, 0.0723]` | **`0.1536`** `[0.1439, 0.1707]` | — | — |
| **Step-40 Branch L2 Norm $\|(A, B)\|_2$** (`median [min, max]`) | `0.0000` `[0.0000, 0.0000]` | `4.6459` `[4.5841, 4.7235]` | `9.2812` `[9.2124, 9.3189]` | — | — |

### 3. H152 Gate-Cliff Context `#27` (`43022ae6a8bc...`, family `fam-9f67f144...`)

Reported separately and never excluded from any statistic (`r882 = +1.122695`):
- **Strict selector**: `+1.122695` in all `3` seeds for `r882`, `CONTROL`, `SIDE4`, and `SIDE16` (never flipped).
- **Step 40**: `+1.122695` in all `3` seeds for `SIDE4` and `SIDE16` (**never flips**); `CONTROL` scores `+1.122695` in seeds `249101`/`249102` and `+1.123721` in seed `249103`.

---

## Verdict

**`NO_ADVANCE`.**
- Under the preregistered strict selector, both `SIDE4` (`0.0000 [0.0000, 0.0000]`, `Holm p = 1.0`, `0/3` seeds) and `SIDE16` (`0.0000 [0.0000, 0.0000]`, `Holm p = 1.0`, `0/3` seeds) tie `r882` on `DL-Val91`.
- No serving or live-flywheel promotion (`serving_promotion = False`). `DOPE` `c51a56d0...` is unchanged.

---

## Next Step

**Close out adapter-only 40-update continuation parameterizations on the fixed H156 `CTRL` objective.**
- Across full-adapter optimizers (`CTRL` AdamW, OneCycle, SAM, Lion, AdaBelief, RAdam, Muon, Grokfast), subset freezes (H239), zero-update weight-space soups/trims (H241–H244), and now function-preserving zero-init low-rank side branches (`SIDE4`, `SIDE16`), every regularizer that reduces effective weight drift (`SIDE4` drift `0.068` -> step-40 loss `-0.0014`; H247 `MUON` drift `0.122` -> step-40 loss `-0.0027`; vs `CTRL` drift `0.379` -> step-40 loss `-0.0300`) prevents continuation damage by staying near `r882`, without discovering a higher-utility policy on `DL-Val91`.
- If a future wave introduces **new supervision signal** (expanded training families or a non-`CTRL` target), `SIDE4` (rank-4 frozen-backbone residual branch) and H247 `MUON` provide the two lowest-drift continuation vehicles (`0.068` and `0.122` L2 drift) to test that new signal safely.

---

## Files and Custody

- **In this directory**:
  - [`PREREG.md`](PREREG.md): preregistration contract (`SHA256 = 78e57e6894262dfcf585406a5596a407d79d42c918ecd69ee6604c6432796af9`).
  - [`fold_manifest.json`](fold_manifest.json): frozen `fold_seed = 249140` 5-fold family partition (`SHA256 = 5daacad390fc73f8527db6b7f12b6eea6b295d0b0a9f1024a8f9587df0d6172f`).
  - [`manifest.json`](manifest.json): input/asset/source pin manifest (`SHA256 = 86eb8d47b681eae8ead018c7fed6a22abe7a999236a1969bf2e93fb56a458b58`).
  - [`schedules.json`](schedules.json), [`provenance.json`](provenance.json), [`treatment.py`](treatment.py), [`analysis.py`](analysis.py), [`runner.py`](runner.py), [`monitor.py`](monitor.py), [`test_h249.py`](test_h249.py) (`13` data-free tests).
  - [`metrics.json`](metrics.json): compact closeout metrics (`60,031 B`) with S3 custody pointers.
- **S3 Custody (`s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H249/attempt-002/`)**:
  - `s3_manifest.json`: `VersionId = w8adjDgYiJkjYMgj7h3ZZ7DHoXhGLShj`, `SHA256 = 1eb7774ed223ed80b938210dfa121c205ddeebac36d997d136c31c7a0a626fca` (`5,417 B`).
  - `metrics.json` (uncompacted): `VersionId = vJDq_9.L0fPYpQVBm9Ro686sv7ilnNhT`, `SHA256 = fb0d320217844e2f7f0796d8351e10366c96df7a63e85012f384e0d8c8ff0c94` (restore-verified).
  - `full_metrics.json`: `VersionId = Vyw0HV6wSEdl0qmEEp6xdVn4IWmjD4Gp`, `SHA256 = cc67d7a4908f499d730b1d9460168d89f87c3cb89feba361f86073b3c4ef68fe`.
  - `panels.npz`: `VersionId = Ajkj1Q0mKVvN5HqiprW5uvSusc1RWZFZ`, `SHA256 = ca5cd1673239088437f8e3461aa39f3db582b7137b17fb6107d07d85fa4ff40a`.
  - `custody-attempt-002.tar.gz`: `VersionId = obLmY6B1DM2pPii1MJrNfW62w0A2NllC`, `SHA256 = 672c5402a685fbe9e95a3cb90abfd0bd375cc52b97e29686d354f2bf3e6865e8` (includes all cell JSONs, selection locks, `.npz` snapshots, resource logs, `smoke-001`, and `attempt-001` `INCOMPLETE_INFRA` record).
