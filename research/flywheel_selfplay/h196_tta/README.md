# H196: zero-update test-time augmentation of r882

**INCOMPLETE**. No scientific verdict and no adoption.

Attempt 1 stopped at the disk guard before any treatment result and was closed in PR [#1623](https://github.com/neverhuman/JopeDime/pull/1623). ENG explicitly authorized one fresh attempt 2 after capacity recovery. Scientific source, seeds, folds and guards were unchanged; separate S3 keys preserve both attempts. This is an infrastructure rerun, not outcome selection.

Attempt 2 ran from preregistration PR [#1615](https://github.com/neverhuman/JopeDime/pull/1615), merged source `7fe61149b7ed5e07f93444765a85b136b8ee0317`. The monitor stopped the attempt with reasons `["sustained load>85%: three readings five seconds apart"]` after 7145.009678954026 seconds. See [metrics.json](metrics.json) for the full terminal resource receipt.

Operational cause: sustained load>85%: three readings five seconds apart. The owner projected roughly ten minutes per B cell before A began; the attempt-2 budget was never extended.

Hypothesis: averaging 16 frozen-r882 cost predictions after FIT-scaled Gaussian input noise improves candidate selection. Fixed sigma 0.05; seeds 196101–196103, independent family-fold seeds 196141–196143, paired-family bootstrap seed196150; NOISE16 sole primary (Holm m=1), PERM8 and NOISE1 descriptive. The intended pooled-A analysis covers 1,219 TRAIN contexts / 205 families, with all 15 B cells fsynced and locked first. Complete setup is in [PREREG.md](PREREG.md).

Native91 passed all 91 utilities bit-for-bit and mean **1.1377941898164525**. Native TRAIN reference scoring also completed. All three FIT permutation probes found a non-bitexact first comparison (maximum absolute cost difference 2.384185791015625e-7), requiring full PERM8 scoring; details are in metrics.json. Completed B/A cells: **9 / 0**; full B lock: **False**. These are progress counts, not complete treatment comparisons.

Three-seed gains, median, 95% CI, Holm p, harm, exact -2 treatment counts and Elo/lift are **unavailable**, not zero. No qualifying verdict is inferred from partial observations. The planned gate remains LCB95>0, p<0.05 and harm<=10% (strictly negative family seed-mean deltas), without the H177 +0.001 threshold. DL-Val treatment analysis, if completed, would be descriptive only.

Source forbids optimizer construction/backward; the failure receipt records zero updates and zero backward calls. No terminal adapter-after or DOPE-after hash is claimed. Resource limits remained nice19, at most four process-tree threads, RSS below6GB, three hours, user-visible disk<=93.8%, MemAvailable>=16GiB and local output<300MB. Observed peak RSS 3.335GB, threads 2, local output 12.817MB, disk 93.563%; minimum MemAvailable 64.755GiB. Peak load 128.187 exceeded the 108.8 threshold. The monitor terminated only its owned worker. No live model or serving pointer changed.

Exact-version partial replay verified 24 S3 objects / 5939 array members and 1310 original pool hashes, including native91 identity, cohort membership, array shapes and FIT comparisons. The prefix inventory matched registered receipts. No extra model inference or local per-candidate dump was used. See [custody.json](custody.json) for version IDs and hashes.

Local closeout validation: 22 targeted data-free tests, Ruff, five jope-records gates, source-pin and report checks. These validate implementation and evidence custody; they do not turn the incomplete experiment into a scientific pass. Historically exposed TRAIN/Val and overlapping roles across cells remain limitations. No sealed/holdout data were read.

Next step: retain r882. Before any A result was observed, ENG preauthorized one fresh attempt 3 after a merged budget-only amendment to seven hours (hw_t1338_h196_rdastra.md). Attempt 2 kept its original three-hour cap. The next attempt must preserve the science, arms, seeds, folds and guards, start from scratch, and reuse no attempt-2 B selections. No partial efficacy claim or adoption.
