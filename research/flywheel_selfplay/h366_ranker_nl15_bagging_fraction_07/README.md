# H366 — NL15 ranker bagging_fraction 0.8 to 0.7

PREREGISTERED; scientific UNKNOWN. BASE and NL15 regressions exact;27archiveparts/2691members,972NL15nativeconfigs/108vetoSHAs and1219TRAIN inputidentities verified;54data-free checks PASS. Owner sol_v1_dsc01.
Hypothesis: reduce row-specific outlier/sentinel chasing with a moderate
fraction reduction. One BF07 arm changes existing ranker subsample.8->.7 only,
retaining H352 leaves15/depth3, freq1, featurefraction.8, LR/trees/seeds/veto.
No added keyword or bagging seed; querybagging stays0.
Primary comparator: archived H352 NL15; secondary: archived BASE.
Preregistration commit: `e6fce0d3dc6a69f42d9855830d31964db78ff232`, published before treatment fits.

Both fresh exact-version regressions precede treatment. Full36locks before
once-only outer scoring. Three roots mean three training seeds30001–30003.
Whole-family10k bootstrap RNG366191; locked +.002/positiveLCB/retention/harm/
positive2of3root decision. Outcome/keymetrics/verdict UNKNOWN pending fullcohort.
Corrected guard CPU>85% averaged60s or load>108 on3 consecutive60s samples;
MemAvailable20GiB/scratch2GB/fixed16:30Z compute stop. Close16:45Z Nov5 fixed.

Sequential sweep H348–H366 on the same TRAIN folds/roots. Any ADVANCE needs
separately authorized one-shotdev91 confirmation, outside H366. No speedverdict.
reference = r882 slot0 (DL-Train outer); r3175 not measured on TRAIN.
DOPEc51a56d0 unloaded/unchanged, CPU selection layer only. Live writer remains
intentionally stopped at round5049; no writer restart or configuration change.
