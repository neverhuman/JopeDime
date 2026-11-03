# Wave H246: Fresh-Seed `STABLE_G025` Pick-Stability Vote-Gate Rerun (`h246-stable-g025-rerun-20261103`)

- **Status**: `COMPLETE` (`75/75` HistGBM fits across `45` fit units; `3` fresh seeds `246101–246103`)
- **Primary Verdict**: `NO_ADVANCE` (`combined_verdict = NO_ADVANCE + NO_PLATEAU_WIN`)
- **Preregistration**: [`PREREG.md`](PREREG.md) (`SHA256 = d733f8c1c1d7bf50f73bbbdb92daf8b1d7b424ed35071dfeda5c311fee684531`), merged in PR `#2092` at `d4a1c1b0292863d7e659bc2b834cf019107c4512` (`reviewed_head = 689770aea723f7380d02c4ed789297d2cc820381`) before any outcome read.
- **Fold Manifest**: [`fold_manifest.json`](fold_manifest.json) (`SHA256 = 3be3efbb8f107061849b09e0565cc03656472bea45958aedabe26f7ae522d60e`, fold seeds `246141–246143`, max best-match Jaccard vs `H203–H245` `= 0.192801 <= 0.20`).
- **Stage-1 Lock**: `stage1_lock.json` (`SHA256 = f31a381fbd11512dfb9ba9003a128a9c8976493c15f9b7928b842de1bd295490`) written and `fsync`-locked after all `75` HistGBM fits and FIT-only `(a, b)` Platt calibration + `tau_cell` selection completed, before any pooled-`A` OOF or `DL-Val` (`flywheel91`) outcome read.
- **Metrics**: [`metrics.json`](metrics.json) (`SHA256 = 6b42061ef43aeb317f00fdb3c1abd1b004ca072354f3dba2566c564d99d5b452`)
- **Manifest**: [`manifest.json`](manifest.json)
- **Source & Tests**: [`analysis.py`](analysis.py), [`runner.py`](runner.py), [`monitor.py`](monitor.py), [`test_h246.py`](test_h246.py), [`metrics.schema.json`](metrics.schema.json)

## 1. Executive Summary & Scientific Verdict

Wave **H246** reran the unchanged **H233** `STABLE_G025` outer 3-member pick-stability unanimity vote-gate recipe inside the frozen **H217** `GATED_025` cost-blend architecture (`0` neural network optimizer updates; frozen `DOPE` `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814` verified before and after; offline read-only `r882` `0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64`; `has_meta_test` rows excluded) on fresh seeds `246101–246103` (`fold_seeds = 246141–246143`). The **only** operational change vs H233 (`CLOSED INCOMPLETE` at `44/75` fits due to an instant `MemAvailable < 20 GiB` hard abort during a transient host spike) was the H239 A1 cooperative unit/member-boundary resource guard and sequential member fitting (`<= 1` HistGBM estimator and `<= 1` prediction dict live at a time, with `gc.collect() + malloc_trim(0)` after each member), which reduced peak RSS to `3.41 GiB` (`3,577,496 KiB < 4 GiB` cap) and completed all `75/75` fits cleanly in `14,924.94s` with `0` pause transitions.

- **Pooled-`A` OOF (`205` families, `1,219` contexts)**:
  - `STABLE_G025` improves significantly over `R882`: median equal-family native gain **`+0.012711`** (`CI95 [+0.000745, +0.033812]`, `3/3` seeds positive `[+0.014409, +0.012711, +0.001089]`, one-sided `p(> +0.002) = 0.039196 < 0.05`, harm `3/205 = 1.46% <= 10%`, Elo `1502.20` (`lift +2.20`, `CI95 [+0.17, +3.73]`)).
  - `STABLE_G025` also outperforms the cell-matched shuffle control `STABLE_SHUF` (`+0.005972`, `CI95 [+0.000146, +0.016390]`, `p(> +0.002) = 0.053495`, harm `8/205 = 3.90%`, Elo `1501.28`) on median gain (`+0.012711 > +0.005972`, paired `STABLE_G025 - STABLE_SHUF` median `+0.000341`, `2/3` seeds positive) and cuts family harm from `8/205` (`3.90%`) to `3/205` (`1.46%`).
  - However, `STABLE_G025` does **not** beat the unweighted `G025` replication anchor (`+0.019754`, `CI95 [+0.002893, +0.040570]`, `3/3` seeds positive `[+0.019754, +0.023057, +0.001709]`, `p(> +0.002) = 0.007999`, harm `8/205 = 3.90%`, Elo `1504.03`): paired `STABLE_G025 - G025` median gain is **`-0.005345`** (`CI95 [-0.015103, +0.001258]`, `0/3` seeds positive, `beats_g025 = false`). Requiring 3-member pick unanimity closes `~68–76%` of `G025` switching openings (`open_rate_among_switching` drops from `[23.69%, 18.58%, 14.76%]` to `[7.48%, 5.34%, 3.56%]`), discarding more beneficial `G025` switches than harmful ones.
- **`DL-Val` (`flywheel91`, `19` families, `91` contexts)**:
  - `R882` baseline verified exact at `1.1377941898164525` (`equal_family = 1.153615`) before any stage-1 fit.
  - `STABLE_G025` achieves context-mean `1.1378358603845307` (`+0.000042` vs `R882`, `CI95 [-0.000233, +0.000295]`, `-0.001121` vs plateau best `1.1389572403748425`), equal-family median gain **`-0.000068`** (`CI95 [-0.000212, +0.000141]`, `1/3` seeds positive `[+0.000259, -0.000072, -0.000068]`, Holm `p = 1.0`, harm `0/19 = 0.0%`, Elo `1500.0`).
  - By comparison, `G025` achieves `DL-Val` context-mean `1.1383601269877812` (`+0.000566` vs `R882`, equal-family median gain `+0.000522`, `2/3` seeds positive, Elo `1507.31`), and `BAG_VOTE_UNGATED` (report-only) achieves `1.1387817973781513` (`+0.000988` vs `R882`, equal-family median gain `+0.001038`, `3/3` seeds positive, Elo `1507.31`).
- **Decision**: **`NO_ADVANCE + NO_PLATEAU_WIN`**. No serving promotion.

## 2. Primary & Secondary Endpoint Summary

### 2.1 Pooled-`A` OOF (`205` families, `1,219` contexts)

| Arm | Role | Median Equal-Family Native Gain vs `R882` | Family-Bootstrap `CI95` | Per-Seed Gains (`246101, 246102, 246103`) | Pos Seeds | One-Sided `p(> +0.002)` | Harm Families (`< -0.01`) | Native Elo vs `R882` (`CI95`) | `-2.0` Count (`s1,s2,s3`) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `R882` | Control | `0.000000` | — | `[0.0, 0.0, 0.0]` | — | — | `0/205 (0.00%)` | `1500.00` | `[17, 17, 17]` |
| `G025` | Replication anchor | `+0.019754` | `[+0.002893, +0.040570]` | `[+0.019754, +0.023057, +0.001709]` | `3/3` | `0.007999` | `8/205 (3.90%)` | `1504.03 [1498.98, 1508.74]` | `[14, 13, 16]` |
| `STABLE_G025` | **Sole Primary (`m=1`)** | **`+0.012711`** | **`[+0.000745, +0.033812]`** | **`[+0.014409, +0.012711, +0.001089]`** | **`3/3`** | **`0.039196`** | **`3/205 (1.46%)`** | **`1502.20 [1500.17, 1503.73]`** | **`[15, 16, 17]`** |
| `STABLE_SHUF` | Same-rate shuffle control | `+0.005972` | `[+0.000146, +0.016390]` | `[+0.005972, +0.012371, +0.004927]` | `3/3` | `0.053495` | `8/205 (3.90%)` | `1501.28 [1496.99, 1505.39]` | `[15, 16, 16]` |
| `BAG_VOTE_UNGATED` | Report-only | `+0.015480` | `[+0.004548, +0.036333]` | `[+0.015480, +0.018689, +0.009724]` | `3/3` | `0.000400` | `9/205 (4.39%)` | `1504.87 [1500.02, 1508.10]` | `[15, 15, 15]` |

### 2.2 `DL-Val` (`flywheel91`, `19` families, `91` contexts)

| Arm | Context-Mean Native | Context-Mean Gain vs `R882` (`CI95`) | Gain vs Plateau Best (`1.1389572`) | Equal-Family Median Gain vs `R882` (`CI95`) | Per-Seed Equal-Family Gain | Pos Seeds | One-Sided `p(> +0.002)` | Harm Families | Native Elo vs `R882` (`CI95`) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `R882` | `1.1377942` | `0.000000` | `-0.001163` | `0.000000` | `[0.0, 0.0, 0.0]` | — | — | `0/19 (0.0%)` | `1500.00` |
| `G025` | `1.1383601` | `+0.000566 [-0.000075, +0.001614]` | `-0.000597` | `+0.000522 [-0.000466, +0.001040]` | `[+0.000522, +0.001320, -0.000216]` | `2/3` | `0.966003` | `0/19 (0.0%)` | `1507.31 [1489.04, 1529.30]` |
| `STABLE_G025` | **`1.1378359`** | **`+0.000042 [-0.000233, +0.000295]`** | **`-0.001121`** | **`-0.000068 [-0.000212, +0.000141]`** | **`[+0.000259, -0.000072, -0.000068]`** | **`1/3`** | **`1.000000`** | **`0/19 (0.0%)`** | **`1500.00 [1489.04, 1507.31]`** |
| `STABLE_SHUF` | `1.1376730` | `-0.000121 [-0.000518, +0.000145]` | `-0.001284` | `-0.000062 [-0.000422, +0.000222]` | `[-0.000385, +0.000099, -0.000062]` | `1/3` | `1.000000` | `0/19 (0.0%)` | `1507.31 [1489.04, 1518.28]` |
| `BAG_VOTE_UNGATED` | `1.1387818` | `+0.000988 [-0.000244, +0.002352]` | `-0.000175` | `+0.001038 [-0.000294, +0.002473]` | `[+0.001038, +0.000699, +0.001101]` | `3/3` | `0.522748` | `0/19 (0.0%)` | `1507.31 [1485.38, 1521.95]` |

## 3. Resource Discipline & S3 Custody

- **Host & Affinity**: `xbabe2` CPU only, `taskset -c 52-55`, `nice -n 19`, `OMP_NUM_THREADS=4`, `OPENBLAS/MKL/NUMEXPR/RAYON_NUM_THREADS=1`, `CUDA_VISIBLE_DEVICES=""` (`CUDA_initialized = false`, `GPU_used = false`).
- **Smoke Check**: `/home/ubuntu/jope-data/h246/smoke-20261103T0525Z` (`205.67s`, `4` HistGBM fits, `0` `DL-Val` reads, `0` protected reads, peak RSS `0.90 GiB`).
- **Primary Run**: `/home/ubuntu/jope-data/h246/run-20261103T0529Z` (`wall_seconds = 14,924.94s`, `fit_wall_seconds = 14,501.53s`, `process_cpu_seconds = 46,289.78s`, `max_rss_kib = 3,577,496` (`3.41 GiB < 4 GiB` cap), `paused_seconds = 0.0`, `transitions = []`).
- **S3 Custody (`s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H246/run-20261103T0529Z/`, all verified by `VersionId`)**:
  - `s3_manifest.json`: `VersionId = rGW.vP13jYveakBNtySfFnP3qUB3L5_Z`, `SHA256 = 2811c6a8542862c42449c2969d6fb04885c279b0f5a6d9f2cd6bc354ec5b7ac3` (`4,653` bytes)
  - `metrics.json`: `VersionId = 1KoTxAdAjL_oyiHgXGZ9JsvkALARB4xT`, `SHA256 = 6b42061ef43aeb317f00fdb3c1abd1b004ca072354f3dba2566c564d99d5b452` (`28,986` bytes)
  - `stage1_lock.json`: `VersionId = o2mfrxnqlHZRQeKQLLqd_WQCgoAu8Dz9`, `SHA256 = f31a381fbd11512dfb9ba9003a128a9c8976493c15f9b7928b842de1bd295490` (`32,491` bytes)
  - `h246-run-20261103T0529Z-bundle.tar.gz`: `VersionId = X4YiDUYp7C3dAjzhRjIqRFJcFeBQayOi`, `SHA256 = 683cfbc56c8e0c934e2a634ff77cb6ec6e3bf44262b0b323e73f01760372721a` (`235,326,509` bytes)
