# H316: family-normalized inner objective

**COMPLETE — NORM_NOT_ADVANCED (TRAIN-only, shadow).** No live or neural-model change.

Hypothesis: per-family utility scale normalization and a family-median inner objective would reduce concentration in a few huge-gain families. [PREREG.md](PREREG.md) was pushed at `cf3eadb6baf0699f812a5bac99542f929cac5d6b` before any H316 fit. Its fixed decision states: **“Ties never advance.”**

## Setup and interpretation

Paired unchanged H303 BASE versus NORM on H308 P0/P1/P2, four family outer folds, roots 30001–30003 and the same 18 configurations/member streams. H303/H302/H308 CPU sources are byte-pinned to H310 code `819a45aa5e`; loader and model reload SHA checks remained enabled. No dev91, DL-Val feature/utility rows, DL-Test, fresh v1–v7, sealed, holdout or meta-test inputs. This reused DL-Train design is not independent confirmation.

As preregistered, **inner-TRAIN** means in-sample predictions of the paired inner members: inner validation families do not exist in inner-TRAIN and cannot supply a TRAIN MAD. Each training family uses unscaled slot0 MAD with floor 0.001; normalized context gains are q10/q90-winsorized within inner fold/config, repeated training records are averaged within family, then the family median selects the configuration. Ties within 1e-4 favor fewer switches, then higher k/a and lower tau. BASE stays out-of-fold H303 selection. The preregistration's “205 outer-TRAIN family scores” wording refers to the 205-family TRAIN population; each outer fit uses its actual proper family subset, never all 205 families.

The regression **passed exactly** for all 36 partition/fold/root pick hashes, all nine whole-panel picks, and all three published medians: P0 0.0643163334285782, P1 0.04821480030677507, P2 0.04360142642936269. Only then did NORM start. Each arm scored every outer fold once after its fsynced fit lock (72 complete score markers). Guarded fit-consumer outer-packet accesses and guard denials were zero. Setup prebuilt the TRAIN packet population, including each task's outer packets; the zero-access claim is **not** a claim of zero physical feature bytes loaded for those packets before fitting. The mixed-cache loader read zero DL-Val feature rows; its trailing identifier metadata is permitted.

## Outcome on raw utility

Median across nine cells of family-equal mean gain: BASE **+0.048215**, whole-family 95% CI [0.025273358495823812, 0.07642710309905959]; NORM **+0.050287**, CI [0.024752560122009968, 0.08871571563390201]. Literal family-median gain (also reported separately): BASE +0.003624, CI [0.0004378293769140007, 0.007781921492181709]; NORM +0.006326, CI [0.0024202517587761617, 0.011715240473271216].

Paired NORM−BASE drop-top-10 median **+0.000342**, 95% CI [-0.000773491924396365, 0.0025324438125799635], one-sided LCB95 **-0.000500** (10,000 paired whole-family draws, RNG316160; same families across arms/cells, each arm re-drops its own ten). Median harmed shares: BASE 9.268%, NORM 10.244%. Decision gates: `{"gain_retention": true, "harmed_nonincrease": false, "paired_drop_top10_positive": false}`. Configurations matched 13/36.

Median Elo: BASE **1725.66**, NORM **1702.31**. **reference = r882 slot0 (DL-Train outer)**; **r3175 is not measured on TRAIN**. Elo retains H302's family W/T/L tolerance1e-4; descriptive `family_wtl` in metrics uses the preregistered1e-9 and its separate `elo_family_wtl_eps_1e4` is explicit. All gains/drop-top/harm/invalid metrics are unnormalized.

| Partition | Seed | Arm | Family mean gain | Family median | Drop top10 | Harmed | Invalid | Switch rate | Elo |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| P0 | 30001 | BASE | +0.041289 | +0.000000 | +0.013174 | 6.341% | 13 | 17.966% | 1725.66 |
| P0 | 30002 | BASE | +0.064316 | +0.001588 | +0.008633 | 8.780% | 14 | 24.610% | 1722.48 |
| P0 | 30003 | BASE | +0.072258 | +0.006984 | +0.016621 | 9.268% | 11 | 26.251% | 1727.10 |
| P0 | 30001 | NORM | +0.045899 | +0.001874 | +0.019442 | 7.805% | 13 | 21.165% | 1719.23 |
| P0 | 30002 | NORM | +0.067025 | +0.005700 | +0.010024 | 11.220% | 14 | 28.138% | 1703.23 |
| P0 | 30003 | NORM | +0.073254 | +0.008983 | +0.017668 | 10.244% | 11 | 28.548% | 1702.31 |
| P1 | 30001 | BASE | +0.048215 | +0.003000 | +0.010198 | 11.220% | 17 | 26.825% | 1697.13 |
| P1 | 30002 | BASE | +0.048658 | +0.003624 | +0.014063 | 10.244% | 15 | 22.559% | 1693.84 |
| P1 | 30003 | BASE | +0.038294 | +0.003253 | +0.012702 | 8.293% | 16 | 23.380% | 1734.68 |
| P1 | 30001 | NORM | +0.048232 | +0.003040 | +0.010216 | 11.220% | 17 | 26.907% | 1698.68 |
| P1 | 30002 | NORM | +0.050287 | +0.004635 | +0.014405 | 10.244% | 14 | 23.544% | 1701.44 |
| P1 | 30003 | NORM | +0.037344 | +0.006552 | +0.005828 | 10.244% | 17 | 29.368% | 1704.72 |
| P2 | 30001 | BASE | +0.043601 | +0.006326 | +0.010678 | 10.732% | 16 | 25.841% | 1716.88 |
| P2 | 30002 | BASE | +0.031840 | +0.003624 | +0.008774 | 7.805% | 17 | 23.954% | 1740.78 |
| P2 | 30003 | BASE | +0.054721 | +0.008145 | +0.023605 | 10.732% | 15 | 27.482% | 1726.57 |
| P2 | 30001 | NORM | +0.036306 | +0.010316 | +0.003201 | 16.098% | 17 | 35.029% | 1685.60 |
| P2 | 30002 | NORM | +0.053720 | +0.006803 | +0.016276 | 8.293% | 17 | 26.497% | 1742.18 |
| P2 | 30003 | NORM | +0.066513 | +0.006326 | +0.015009 | 11.220% | 17 | 26.825% | 1682.13 |

Drop-top-1/5/10, W/T/L, per-cell confidence intervals, invalid reference counts, all choices, lock hashes and pick hashes are in [metrics.json](metrics.json). H315 conditional overlap: `{"agreement_by_cell": [0.9007383100902379, 0.8851517637407711, 0.867104183757178, 0.7998359310910582, 0.8785890073831009, 0.8301886792452831, 0.844954881050041, 0.9179655455291222, 0.8318293683347006], "both_switch_contexts": 1580, "checked_commit": "072eb6798139d400a1cb617ae4c2d3c803c6b25b", "checked_utc": "2026-11-04T12:38:51.278083+00:00", "h315_verdict": "MARGIN_NOT_ADVANCED", "identical_cells": 0, "median_agreement": 0.867104183757178, "picks_sha256": "e9e6b16a7381ff14191d2a4d31ce31d4b623c39297e7383ee6aeb871e08ca615", "same_candidate_switched_contexts": 1540, "status": "CLOSED_PICKS_COMPARED", "version_id": "jCll6z9FGwmpNg.cocb305LcVzFYuieR"}`. No wait and no active H315 scratch reads.

## Verification, resource limits and custody

The compute journal and all completed artifacts are retained. The transient unit had already been garbage-collected at status readback, so its exact process exit status is recorded as unavailable; default not-found status fields are not treated as observed exit evidence.

Fourteen targeted tests pass: objective scale/floor/clipping/family weighting/conservative ties, paired bootstrap/decision negative controls, poisoned mixed-tail/prefix refusal, outer access guard, missing-lock refusal and second-score refusal. Actual new LightGBM members and byte-verified reloads are archived; DOPE c51a56d0 and neural weights stayed frozen. Two own-unit operational restarts before any outer score retained completed members: per-task S3-verified compression bounds scratch, and telemetry tolerates its own verified file reclamation. Neither reset the deadline or changed a scientific choice.

xbabe2 CPU only, nice19, numeric threads1, cores52/58/60/61/62/63 disjoint H314/H315, systemd MemoryMax6G/MemorySwapMax0. Sampled peak host CPU 41.16%, peak load 62.13, minimum MemAvailable 67.90GiB, peak scratch 1342203661bytes. Guards stop own compute at CPU>85%, load>85, MemAvailable<20GiB, scratch>2GB or13:15Z; no GPU calls. Individual completed BASE inner-model archives were exact-version GET+SHA verified before raw-copy removal. Final source/model/lock/input/output archive and exact-version/member verification are recorded in [s3_manifest.json](s3_manifest.json); no models enter Git. Final archive streams directly to S3 to avoid a duplicate local gigabyte. Own scratch is reclaimed only after complete verification. No PR/main/fastlane/CI or foreign checkout/job changes.

For replay, use the pinned CPU sources and S3 archive; `run_h316.py` expects host-local TRAIN inputs, `H316_FS` pointing to the pinned H302/H303/H308 sibling directories and `H316_CUSTODY` containing the published-prereg receipt. No new scoring is authorized by this closeout. [manifest.json](manifest.json) records code, input summary, runtime limits and operational deviations; the complete per-pool/prefix input hashes are archived.

Next: Do not advance NORM or score dev91. Preserve the negative result and seek a separately preregistered objective or new independent families; fresh confirmation remains blocked by H313 NO_NEW_CATALOG.
