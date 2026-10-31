# H57 — DEEP-ES direct-utility search

## Hypothesis and setup

H57 tested whether eight generations of antithetic evolution strategies (ES), initialized from the pinned r882 adapter, improve the final mean adapter on an untouched family-grouped fold. The adapter was trained on fold A of DL-Train (164 families) and judged against r882 on fold B (41 families). The six candidate cells were `sigma001` and `sigma003` at seeds 15, 16, and 17; each used 16 antithetic pairs (population 32), centered-rank fitness shaping, and Adam at 0.01. The final mean theta was evaluated; no generation or sampled candidate was selected. A drift-only `null_sigma001` cell used seed 15 and randomized rank assignments. Its role was descriptive and it was not eligible for admission.

All candidate and null cells completed the fixed eight generations. The preregistered endpoint was paired fold-B family utility gain versus r882. Positive gain and positive Elo favor the candidate; ties are not wins. Admission required a one-sided family-cluster bootstrap LCB95 strictly above 0.001, pooled family-median gain at least zero, and at most 10% of families below −0.01.

## Outcome and verdict

Neither sigma arm was admitted. Both pooled mean LCB95 values were negative, and both exceeded the 10% harmful-family limit. The pooled family median was zero for each arm, so ties did not satisfy the strict mean-LCB gate. No terminal DL-Val reads were allowed or performed. Do not extend H57 from these results alone.

| Arm | Pooled mean gain | Mean LCB95 (one-sided) | Pooled family median [95% CI] | Families below −0.01 | Admission |
|---|---:|---:|---:|---:|---|
| sigma001 | 0.01558 | −0.02721 | 0.00000 [0.00000, 0.00410] | 21.95% | No |
| sigma003 | 0.01397 | −0.02735 | 0.00000 [−0.00482, 0.00141] | 31.71% | No |

Per-seed fold-B family mean gains were `0.01898, 0.03077, −0.00301` for sigma001 and `0.03358, 0.00394, 0.00437` for sigma003 (seeds 15, 16, 17). The seed spread is material; positive means on some seeds do not override the pooled family-cluster admission result.

### Elo diagnostic

Reported Elo is family-grouped Bradley–Terry Elo relative to r882=1500, summarized as the median of the three seed-level Elo deltas. The 95% interval is a descriptive bootstrap over the three seed summaries (10,000 resamples); it is not the admission interval and has very low replication.

| Arm | Seed 15 | Seed 16 | Seed 17 | Median Elo delta [seed-bootstrap 95% CI] |
|---|---:|---:|---:|---:|
| sigma001 | −0.71 | +37.72 | +7.91 | +7.91 [−0.71, +37.72] |
| sigma003 | +0.56 | +1.98 | −4.10 | +0.56 [−4.10, +1.98] |

Family-grouped W/T/L counts versus r882, with ties kept separate, were sigma001 `125/431/98` and sigma003 `122/415/117`. These are diagnostics, not the admission endpoint.

### Drift-only null

The null's generation-eight fold-A population fitness mean was 1.43690. The corresponding sigma001 seed values ranged from 1.41672 to 1.45318 (mean 1.43172); sigma003 ranged from 1.43878 to 1.45887 (mean 1.44877). On this in-sample population metric, the null was comparable to sigma001 and below sigma003's mean. This does **not** establish that the null is as good as either candidate on the admission endpoint: the null was deliberately not scored on fold B and cannot be admitted. These training-fitness values did not drive a decision.

## Integrity, timing, and red-team checks

- The independent audit passed all seven cells, fixed order and eight-generation horizon, six fold-B candidate reads, source/checkpoint pins, final adapter receipts, and admission recomputation.
- DOPE checksum `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814` matched before and after every candidate/null generation and fold-B scoring.
- Candidate/null receipts state that fold B and validation payloads were not opened during those cells. Fold B was read only after all seven cells completed. No sealed/holdout payloads or outer rows were opened. No terminal DL-Val reads occurred.
- H59 held the qualification lock between H57 sigma003/seed16 completion at 04:26:57Z and H57 sigma003/seed17 start at 04:33:37Z. H57 logged a 399.84-second wait. After the null cell ended at 04:51:56Z, H57 waited while H59 and then H61 used the lock; H61's fold-B pass ended at 05:16:48Z, H59 resumed at 05:16:53Z, and H57 fold-B scoring started at 05:25:54Z after 2,036.75 seconds of cumulative wait. The lock handoffs did not add to cell wall or consume the cell-wall cap.
- Accepted outcome and fold-B GPU cell wall was 4,029.287 seconds of the 5,400-second cap; total lock wait was 2,442.788 seconds and is reported separately. The retry wave had zero void-cell wall. Attempt 1's 565.914-second disk-guard infra-void remains separately receipted and excluded under the owner ruling; it produced no candidate or decision data.
- The family-grouped Elo W/T/L treats ties as ties, not wins. Utility gain and Elo direction are reported explicitly; no AUC is used.

## Next step

H57 is closed with no admitted arm and no terminal validation. Follow the owner queue: H75 closed-loop evolution, then W1, then H71/W5. Any H57 change would need a new preregistration and cannot be justified by selecting a seed from these results.

## Evidence

- Compact metrics: [`metrics.json`](metrics.json) (20,290 bytes, below 50 KiB).
- Retry-run custody manifest: `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/h57/h57-outcomes-retry-20261031T0338Z-closeout/manifest.json` (SHA-256 `1b977602d5b3e29f9439601d5c7f7e298b6b850ba21c1b8cc8d3d18283566eda`); verification receipt: `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/h57/h57-outcomes-retry-20261031T0338Z-closeout/custody-verification.json`.
- Execution source: [`PREREGISTRATION.md`](../../../../../research/flywheel_selfplay/h57/PREREGISTRATION.md), [`amendment-2026-10-31.md`](../../../../../research/flywheel_selfplay/h57/amendment-2026-10-31.md), and [`infra-void-amendment-2026-10-31.md`](../../../../../research/flywheel_selfplay/h57/infra-void-amendment-2026-10-31.md).
- Outcome root: `~/jope-data/h57/h57-wave-retry-20261031T0338Z/`; final wave receipt `outcomes/wave.json`, audit `outcomes/audit.json`, admission `outcomes/admission.json`.
