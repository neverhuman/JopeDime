# H163 preregistration: accepted adapter prediction ensemble

- Owner: `luna_ns_hbaseline`; claim: `/home/ubuntu/jope_plan/ns/receipts/H163_claim.txt`.
- Inputs: read-only H160 copies of R782/R785/R859/R882; DOPE composite must remain `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`.
- Cohorts: H148 real-S3 TRAIN only (1,219 contexts / 205 families); disjoint DL-Val (91 contexts / 19 families) secondary. No sealed, holdout or meta-test access.
- Updates: exactly zero. No optimizer, gradient, backward pass, or parameter write.
- Execution: one CPU-only xbabe2 attempt, nice 19, at most four total OS threads and RSS <6 GB; stop terminally on load >85% of cores, MemAvailable <16 GiB, free disk <40 GB, or used disk >=95%. No GPU or qualification lock.

## Prediction and arms

Each model emits a candidate **cost score**; lower scores win by `argmin`. Measured outcome utility is a separate quantity and is higher-is-better. Model score vectors must have identical candidate ID sets and are aligned to R882's native candidate array order. Exact ties use native stable first-index behavior (`numpy.argmin`), never a sorted-ID tie break. Outcomes are read only after a choice has been locked.

Arms: R882 baseline; ENS2 equal mean of per-context population z-scored R882/R859 costs; ENS4 equal mean across R882/R859/R785/R782; ENS4_W weighted z-scored mean with R882 weight 0.5 or 0.7 and remaining weight divided equally, where the weight is selected on fit families only (equal-family utility; exact tie chooses 0.5); RANKAVG4 averages within-context average ranks (lowest cost rank 1; exact score ties receive average rank).

## Cross-fit and gate

Use the H148 five-fold family-grouped split over sorted family IDs with **fixed fold seed 163140**. For outer assessment fold A, B=(A+1)%5 is the selector fold and the remaining three folds are fit-only. Assert family disjointness. Seeds 163101, 163102, 163103 randomize selector/bootstrap procedures only; they do not change fold membership or the prediction panels. Bootstrap base seed is 163141. If deterministic panels repeat, disclose that and do not imply independent prediction data.

For each alternative, the B-fold equal-family utility gain over R882 must be strictly >0.001 (within 1e-12 of threshold rejects) and paired 10,000-draw family-bootstrap 95% LCB >0. Otherwise use R882 on A. Persist the selection lock before reading/scoring A outcomes. ENS4_W chooses its weight on the three fit folds only, then is subject to the same B gate.

Primary: one 1,219-context out-of-fold panel per arm and seed; equal-family gain vs R882, per-seed and median gain, paired family bootstrap 95% CI and one-sided p, and canonical H145 Bradley-Terry Elo (400 MM iterations, baseline 1500). ADVANCE only if median primary gain >+0.001 and CI LCB >0. Report changed choices and -2 count for all arms. Secondary: selected policy on DL-Val relative to exact R882 mean 1.1377941898164525. Exact 91/91 R882 regression gate is the first scored operation; any mismatch terminally consumes the sole attempt before TRAIN outcome scoring. No speed, latency, or throughput metrics.

R882 historically saw these TRAIN contexts; inference is conditional continuation evidence, not unseen-family evidence. No serving or flywheel promotion follows from H163. Store outputs in `/home/ubuntu/jope-data/h163` and evidence in `/home/ubuntu/jope_plan/evidence/H163`; Git contains only source and compact manifests/metrics.
