# H179 deep ensemble

## Result

H179 completed its single CPU-only guarded run on 2026-11-01. The unchanged `r882` baseline passed bit-exactly on 91/91 contexts before the outcome pool was opened. The run assessed 205 training families across five outer folds, three seeds, 90 trajectories, and 3,600 optimizer updates.

The strict preregistered verdict is **NO_ADVANCE**. `ENS5` had median paired gain `+0.013694` over `r882` (all three seed gains positive), but its corrected paired-family bootstrap 95% interval was `[-0.016451, +0.047647]` and Holm-adjusted one-sided p-value was `0.608939`; neither confirmatory condition passed. This is diagnostic only and does not promote a serving model. The H175 MDE80 comparison is a planning heuristic, not a measured H179 power result.

The same corrected bootstrap gives `CONT` gain `-0.027584 [-0.053836, -0.002630]`, `FRESH1` `-0.003537 [-0.037167, +0.036050]`, and `ENS3` `+0.005469 [-0.027152, +0.041432]`. `ENS5` Elo versus `r882` is `1509.46 [1495.98, 1525.00]`.

## Inference and custody

The run source was merged at `3776e5b50ac461afd625a841267f0f1d58495624` (PR #1509), which corrected the paired bootstrap to reduce across seeds (`axis=0`) for each family-resampling draw. An independent recount from the persisted family/context panels reproduced the utilities, paired gains, intervals, p-values, Holm adjustment, Elo, `-2` counts, and verdict. Member-level choices/scores were not persisted, so the runner's `0.339595` member-pair disagreement scalar cannot be independently recalculated from the retained files.

After the run, the paired-bootstrap expression was extracted into a helper with unchanged `axis=0` semantics, and the regression fixture was strengthened to use distinct per-seed variances and assert a 10,000-element non-degenerate bootstrap vector; all nine H179 tests pass. This helper/test follow-up was not part of the run's source closure. The exact run launch and raw output are retained in the H179 versioned S3 custody archive identified by `custody.json`.

No serving promotion or follow-up training run is authorized by this result.
