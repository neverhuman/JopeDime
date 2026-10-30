# FUP-01: H38 / H40B Elo on held-out real S3 cohorts

**Outcome: scoring not computable from the available approved artifacts.** No Elo, confidence interval, match rate, or R² is reported. This is an evidence-backed preflight result, not a zero score and not a failed model evaluation.

## Requested output, in order

| Measure | Result |
|---|---|
| Matched held-out real S3 coverage (H38, H40B, baseline) | **Unavailable**: no common eligible predictions or frozen family-to-fold roster |
| Elo with confidence interval | **Not computable**: no common per-cohort outcomes for the three methods |
| R² | **Not computable**: neither candidate has eligible predictions on a common held-out cohort |

The 120-row `jope-fused-net-v3` panel in `/home/ubuntu/jope_plan/elo/fused_panel` cannot supply the missing H38/H40B entries. Its manifest identifies the method as `jope-fused-net-v3`, adapter `h45`, checkpoint SHA-256 `5f5640d18443b838b4777216128ad0c03504eb28ceb2198deec8145730961dfa`; it is not an H38/H40B comparison. The panel is an existing artifact, not a newly executed FUP evaluation.

## Preflight findings

1. **H38 candidate identity is not a runnable table predictor.** H38 selected `F2_SMART_LM80_STD_SVD32` as a ranking/screen configuration. The reconciled H38 record has `generalization_claim=false`, `post_hoc_replay_only=true`, and `selection_exposed_stage2_20=true`. It contains ranking/utility metrics, not held-out regression R² or per-dataset predictions. The existing evaluator inventory (`~/jope_plan/elo/fused_g45/README.md`) reports `jope-h38-f2smart: 0/42`: no exported fitted table selector/inference entrypoint was available. Golden45 is not a substitute for the requested real S3 cohort.
2. **H40B has no eligible evaluation output.** The checked-in H40 record has status `FAILED_WALL_CLOCK_CAP_EXCEEDED_RECONCILED_NO_RETRY` and `metrics: null`. The H-wave baseline record describes H40B as cached-pool replay with zero native fits and repeated development/holdout inspection; it is not independent candidate scoring.
3. **S3 cohort membership cannot be treated as a held-out family roster.** The S3 catalog evidence says there is no authoritative family/lineage key or frozen family-to-fold roster. It explicitly disallows using dataset IDs, collection names, or dedupe fingerprints as substitute family IDs. The sealed/meta-test family set remains untouched.
4. **A baseline comparison would not repair the missing candidate rows.** Existing `fused_panel` receipts score a separate method. They provide no matched H38/H40B predictions, so the pairwise observations required by `elo.py` do not exist.

## Method and stop rule

The existing Elo implementation in `~/jope_plan/elo/elo.py` converts each dataset's method metrics into pairwise win/tie/loss outcomes, fits regularized logistic Elo centered at 1500, and reports each method's number of covered datasets. A confidence interval additionally requires a declared resampling unit. For this request the valid unit must respect held-out S3 family grouping; the authoritative family-to-fold roster is absent. Therefore no Elo fit, bootstrap, or R² aggregation was run. Missing candidate predictions are unknown coverage, not losses; no denominator or match rate was fabricated.

A valid rerun needs: (a) a pinned, runnable H38 candidate inference artifact; (b) a pinned H40B candidate artifact with its provenance and allowed evaluation scope; (c) an authorized frozen real-S3 family-to-fold roster that excludes sealed/meta-test families; and (d) baseline predictions on the same cohort and split. Then run the existing evaluator on the common rows, report match rate first, family-clustered Elo with a predeclared interval procedure second, and R² last. This report does not create or train candidate models.

## Evidence identities

- Repository source head at assessment: `fe9cc58bddabe533e17b02e7c9044dde178181d3`.
- H38 reconciled record SHA-256: `0afee6439cd69f63edc1a827621b337c77120c9c03c092635b8663433cc4c545` (`docs/research/studies/dope-neural-ranking-20261021/h38/results_v2_reconciled_noncompute.json`).
- H40 record SHA-256: `5d7e0d4034b12b1c7e284db7eed27c3984365f15ff263963c5dd5d356daf70c5` (`docs/research/studies/dope-neural-ranking-20261021/h40/results.json`).
- H-wave baseline assessment SHA-256: `622054b98f961fd0d5bf893fb0e1a45cf415314de30a0aa9259550ef5361569b` (`~/jope_plan/ns/H_BASELINE.md`).
- Existing S3 panel scorer SHA-256: `10763b22c2806e7ff188c997aac960baf26e2d6e248791c1febb9a458e54a8e4` (`~/jope_plan/elo/fused_panel/score_panel.py`).
- Existing S3 panel shard-3 manifest SHA-256: `37c5a9bdffa77f206ee9bcd6078b44a1e65878550111ce550c5b5e20500c80cd`; manifest records 120 dataset IDs, method `jope-fused-net-v3`, adapter `h45`, CPU, 1 thread, nice 19, 120 successful rows and zero errors.

No S3 payload was fetched or opened for this preflight, no sealed/meta-test data was read, no GPU or training was used, and DOPE remained frozen at `c51a56d0`.
