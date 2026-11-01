# H182 — fresh-adapter ensemble-size dose

## Hypothesis and setup

H182 tested whether a larger ensemble of independently initialized fresh adapters improves held-out family utility over the unchanged `r882` policy. The preregistered arms were nested `ENS5`, `ENS10`, and primary `ENS20`, with candidate scores averaged before selection. Every member used the same 40-update recipe and trained only the private 29,211-parameter adapter copy.

The run used the pinned real-S3 TRAIN cohort: 1,219 contexts from 205 families. Five family-grouped folds used seed `182140` (41 A families per fold, 123 FIT families, unused B families), with outer seeds `182101–182103` and 20 distinct member seeds per outer seed. All 300 member trajectories completed for 12,000 optimizer updates. The four-update FIT-only timing pilot projected the full five-fold design within the four-hour cap, so all five folds and all 40 updates/member were retained.

The exact native `r882` baseline passed bit-exactly on all 91 DL-Val contexts at mean utility `1.1377941898164525` before TRAIN outcome pools were opened. The run was CPU-only on xbabe2 under the preregistered `nice 19`, four-thread, memory, disk, host CPU, and wall-clock guards. The frozen DOPE composite remained pinned to `c51a56d0…770ba814`.

## Results

The strict verdict is **NO_ADVANCE**. ENS20's median paired gain over R882 was `+0.016757`, with 3/3 positive seed gains, but its paired-family bootstrap 95% interval was `[-0.015297, +0.050534]` and its Holm-adjusted one-sided p-value was `0.457454`. ENS20 therefore failed both the required positive lower confidence bound and Holm thresholds. This is diagnostic only; it does not promote a model or authorize serving an ensemble.

| Arm | Median utility | Median gain vs R882 | Paired-family bootstrap 95% CI | Holm m=3 p | Median Elo vs R882 |
| --- | ---: | ---: | ---: | ---: | ---: |
| R882 | 1.385355 | — | — | — | 1500.00 |
| ENS5 | 1.392876 | +0.007520 | [-0.021093, +0.044898] | 0.457454 | 1505.31 |
| ENS10 | 1.400556 | +0.015200 | [-0.015163, +0.049962] | 0.457454 | 1509.02 |
| ENS20 | 1.402112 | +0.016757 | [-0.015297, +0.050534] | 0.457454 | 1506.66 |

ENS20 seed gains were `[+0.016757, +0.020022, +0.010906]`. Pairwise member-choice disagreement was `0.342494` at K=5, `0.341980` at K=10, and `0.343170` at K=20; an independent recount reproduced these values from all persisted raw member scores and choices. ENS20 exact `-2.0` counts were 18, 16, and 17 for seeds 182101, 182102, and 182103, respectively.

The H175 MDE80 planning heuristic classifies this null as underpowered because the absolute ENS20 median gain is below `0.042341`; this is not a measured H182 power estimate. No AUC or speed metric was used.

## Audit, custody, and next step

All 300 cell receipts and all 300 member-level A-score shards passed independent checks against the consolidated score panel. The raw-score `argmin` choices, 300 distinct FIT batch-plan hashes, per-fold member initialization hashes, utility panels, bootstrap, Holm adjustment, Elo, `-2` counts, and verdict were independently recounted. Audit reported 0 protected reads and 12,000 optimizer updates; the CPU supervisor recorded no resource violations. `metrics.json` is 6,622 bytes.

The 628-member evidence archive and custody manifest are stored in versioned S3. Exact-version byte readback verified both objects:

- Archive: `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H182/H182_attempt1_20261101T1318Z/H182-evidence.zip`, VersionId `esDiSDGmFLPoJPZwI82KSMudXqtEEIob`, SHA-256 `c29931a4ad2096b9afcfafd02782c4a2abbbe7dfa5dc214fe3b0f7ecb9fe537c`.
- Manifest: same prefix, `H182-manifest.json`, VersionId `f13256Ja2cGNeubP2OPUywT.y3gRthva`, SHA-256 `a857d12b803df58205e1f5dcc194f01edb839617b2761703dc51e2ad8dfd1b39`.

Next step: retain `NO_ADVANCE`; make no serving or promotion change. Any future promotion requires the H177/H178 calibration rules and a separate owner ruling.
