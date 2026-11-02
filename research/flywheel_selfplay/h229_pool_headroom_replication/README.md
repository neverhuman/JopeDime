# H229 — Fresh-seed candidate-pool headroom replication

**PREREGISTERED, NOT RUN.** Test frozen-r882 NATIVE64 vs NATIVE16, with
FPS16of64 secondary. Reuse H227 source1b463762b90c2f5f4db8143483026802e2947d03,
fresh229101–103 seeds and229141–143 grouped folds; no reused H227 outcomes.

See [PREREG.md](PREREG.md), [manifest.json](manifest.json),
[fold_manifest.json](fold_manifest.json), [metrics.json](metrics.json).
TRAIN1219/DL-Val91 real-S3 only; zero updates; baseline91/91 bit-exact,
r882 mean1.1377941898164525 and frozen DOPE checked before/after.
Primary median equal-family gain, paired19-family CI and Holm2; absolute
NATIVE64 plateau lift vs r882 strictly>0.002 plus positive primary CI/gain
and adjusted p<0.05. FPS cannot rescue; Elo/oracle descriptive only.

CPU xbabe1 nice19,4cores/math1. RAM<12GiB OR load5>70 pauses dispatch;
resume iff RAM>=16GiB AND load5<=60, recheck60s. Hard stops RSS>6GiB,
RAM<6GiB, free<80GB, output>2GB, deadline. Exact field/value logs mandatory.
One TRAIN smoke then one primary; prereg19:15Z, compute22:30Z,
closeout23:00Z November2 UTC. Unknown/incomplete efficacy remains null.

Next: targeted local checks/review/squash; source-bound TRAIN smoke and single
primary; terminal output-only readout and create-only versioned S3 custody.
