# H130 — Harm-constrained adapter fine-tuning (`HINGE` & `CVAR` vs `r882` and `CONTROL`)

Status: **COMPLETE — `NO_ADMISSION`** (`candidate_breakthrough = false`). Preregistered in [PREREG.md](PREREG.md) (PR [#1133](https://github.com/neverhuman/JopeDime/pull/1133), merged at `9e57ce36f4269ea0a1af7ffa6cc1c6c5d1566734`) before any real-data training or `fold 0` assessment scoring occurred. Machine-readable protocol and closeout state are in [manifest.json](manifest.json) and [metrics.json](metrics.json); the frozen 5-fold `DL-Train` family split (`fold_seed = 130094`) is in [fold_manifest_130094.json](fold_manifest_130094.json) (`sha256 = 98196cbc017c10d12db1f8c4390c163a11d3faddf1c457b79108a6386ed785ea`).

## Design & Execution Summary

- **Input Closure**: `1,219` `DL-Train` contexts (`205` families) from pinned H50 input (`/home/ubuntu/jope-data/h50/input`): `fold 0` = `41` assessment families (`222` contexts); `folds 1..4` = `164` fit families (`997` contexts). Zero `DL-Val`, outer-test, sealed, holdout, or meta-test reads.
- **Step 1 — Unchanged-`r882` Regression-First Gate (`PASS`)**: Two fresh hash-verified `r882` copies (`0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64`) with frozen `DOPE` (`c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`) matched with `0.0` max absolute difference across all `222` `fold 0` predictions, utilities (`r882_fold0_unrev_family_utility = 1.392421`, `context_utility = 1.402575`, `native_family_utility = 1.416685`), and one `CONTROL` forward+backward gradient probe (`regression_receipt.json` `sha256 = 1616ea5f88f6d58ef2ae37c137f651dc0accbe694e3d6568ed64421998cbde1b`).
- **Step 2 — Frozen Fit-Fold Per-Family Baseline (`PASS`)**: Computed once across `997` fit-fold contexts (`164` fit families, `0` assessment families touched, `0` optimizer updates): `median_loss = 1.408269` (`min = 0.634294`, `max = 53.724483`, `fit_family_baseline.json` `sha256 = 0367146a1178ac811ff9e4d3154050acb113376e1beec094f6016ba01d59c4f8`).
- **Step 3 — 9-Cell CPU Wave (`3` arms x paired seeds `94, 95, 96`, `40` AdamW updates/cell = `360` actual optimizer updates)**: Executed on CPU only (`nice 15`, `4` threads, peak RSS `4.82 GiB`, `0.0` GPU seconds, `1022.88 s` wall time).

## Primary Endpoint: `HINGE` and `CVAR` vs Unchanged `r882` (`fold 0`, `41` families)

| Arm | Seed 94 Fam U | Seed 95 Fam U | Seed 96 Fam U | Median Seed Gain vs `r882` (`1.392421`) | 10,000-Draw Bootstrap `LCB95` (`95% CI`) | Holm 1-Sided $p$ ($lpha$) | Pooled Family Median Gain | Pooled Family Harm (`< -0.01`) | Pooled Paired Elo vs `r882=1500` (`95% CI`) | Joint BT Elo vs `r882=1500` | Gate Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `CONTROL` | `1.307063` | `1.317730` | `1.323300` | `-0.074691` | `-0.146212` (`[-0.161210, -0.011873]`) | `0.9933` (ref) | `-0.000560` | `39.02%` (`16/41`) | `1395.25` (`[1303.46, 1480.21]`) | `1469.52` | Reference |
| `HINGE` | `1.306911` | `1.322860` | `1.325656` | `-0.069562` | `-0.143503` (`[-0.159028, -0.009228]`) | `0.9908` (`0.025`) | `-0.000003` | `34.15%` (`14/41`) | `1410.51` (`[1322.09, 1491.52]`) | `1472.49` | **`NO_ADMISSION`** |
| `CVAR` | `1.311373` | `1.329345` | `1.334425` | `-0.063077` | `-0.134829` (`[-0.151545, -0.006571]`) | `0.9914` (`0.050`) | `0.000000` | `34.15%` (`14/41`) | `1443.00` (`[1356.83, 1522.63]`) | `1480.33` | **`NO_ADMISSION`** |

## Secondary Endpoint: `HINGE` and `CVAR` vs `CONTROL` (`fold 0`, `41` families)

| Comparison | Per-Seed Family Mean Deltas (`94, 95, 96`) | Median Seed Gain | Bootstrap `LCB95` (`95% CI`) | Pooled Family Harm (`< -0.01`) | Adapter L2 Drift Range from `r882` | Pooled Paired Elo (`95% CI`) | Joint BT Elo (`CONTROL=1500`) |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: |
| `CVAR` vs `CONTROL` | `+0.004311, +0.011614, +0.011125` | `+0.011125` | `-0.008537` (`[-0.011956, +0.030380]`) | `9.76%` (`4/41`) | `0.3556 – 0.3869` (vs `0.4474 – 0.4551`) | `1551.21` (`[1514.13, 1589.49]`) | `1510.82` (`R882: 1530.48`) |
| `HINGE` vs `CONTROL` | `-0.000152, +0.005129, +0.002357` | `+0.002357` | `-0.000681` (`[-0.000850, +0.008266]`) | `2.44%` (`1/41`) | `0.4287 – 0.4437` (vs `0.4474 – 0.4551`) | `1505.65` (`[1483.04, 1528.31]`) | `1502.97` (`R882: 1530.48`) |
| `HINGE` vs `CVAR` | `-0.004463, -0.006485, -0.008769` | `-0.006485` | `-0.023497` (`[-0.027429, +0.013377]`) | `17.07%` (`7/41`) | — | `1451.67` (`[1416.51, 1485.87]`) | — |

## Key Takeaway

Penalizing batch-family regressions relative to frozen `r882` fit-fold losses—especially via the top-20% batch-family `CVAR` penalty—meaningfully curbed adapter L2 drift (`0.356–0.387` vs `0.447–0.455` in `CONTROL`) and improved held-out assessment utility over `CONTROL` across all three seeds (`median_seed_gain = +0.011125`, `pooled_paired_elo = 1551.21` vs `CONTROL=1500`, `9.76%` harmed families vs `CONTROL`). However, 40-step offline fit-fold adapter optimization under all three objectives still regressed relative to unchanged `r882` (`CVAR` `-0.063077`, `HINGE` `-0.069562`, `CONTROL` `-0.074691`), so neither arm meets the primary admission gates against `r882` (`NO_ADMISSION`). No serving promotion is implied.
