# H19 result: correction range and loss allocation are plausible bottlenecks

Six frozen H18 bases were replayed on their own TRAIN slates:850contexts,162components,2550seed-context decisions. No fits, optimizer updates, S3GETs, CAL/OLDDEV or model-opposite-fold pool reads. Producer46.045s; independent NumPy saved-score/source QA4.326s. QA is arithmetic/source verification, not an independent neural forward replay. All six original TRAIN32 controls and score scales matched.

The independent FP32 score-box envelope has median4candidates (mean6.331,max92) of1152.326/2550queries have only the base choice; top32 misses envelope candidates in47/2550. It includes33.398%of full TRAIN oracle headroom; about66.6%is outside. Equal-component mean score-coordinate ranking-gradient L1share outside is64.041%; this is not a parameter-gradient attribution.

These are TRAIN-only diagnostic upper bounds, not achieved lift or the exact set realizable by shared adapter weights. The wider correction could overfit or harm choices. No validation outcome chose a new bound. H18 primary null/adverse findings remain fixed.

Highest-value next fitting hypothesis: H20 uses one fixed10x larger bound (1.0times the same TRAIN-score scale) with twelve fresh trained/random adapters and unchanged frozen bases, loss, inputs, schedules, optimizer and update count. Compare trained-wide against BOTH base and random-wide on the full pool, after all fits close. Do not replace either primary with a favorable narrow-vs-wide or seed result. No shortlist/Ksearch, early stopping or CAL/OLDDEV tuning. Its2400updates are planned, not executed.

See results.json for aggregate measurements/source pins. Full private score arrays and reproducible source remain in verified research evidence custody; bulk is outside product Git. Product documentation import remains queued behind unrelated ownership.
