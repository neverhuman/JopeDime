# H227 — Candidate-generation headroom

**PREREGISTERED, NOT RUN.** Test whether increasing the frozen r882 candidate
pool from parent+15 to parent+63 improves selected-program utility; compare an
equal-cost farthest-point16 subset of the same64. Oracle-best utility diagnoses
generation versus selection headroom and never influences selection.

See [PREREG.md](PREREG.md), [manifest.json](manifest.json),
[fold_manifest.json](fold_manifest.json), and [metrics.json](metrics.json).
Zero optimizer updates. Frozen r882 COPY/DOPE, approved real-S3 H139/H195
cohort, strict91/91 baseline parity before any proposal. Three fresh seeds
and grouped folds; paired family bootstrap and Holm m=2. ADVANCE requires
strict plateau gain>0.002, positive family CI lower and adjusted p<0.05.
Unknown/missing results remain INCOMPLETE/null. Elo and oracle gaps are
reported separately; no speed/AUC verdict.

The current source planner has16/8 generated proposals and4 native fits;
this offline16-program control binds exploration width and does not claim
complete live-budget replay. No live run/custody access is permitted.

CPU xbabe1, nice19,4 cores/math1,6GiB RSS,10GiB available RAM,50GB free,
output<2GB. Pause load5>70/recheck60s/resume<=60; other guards are hard stops.
Prereg17:15Z, full-path TRAIN smoke17:45Z (up to2 engineering attempts),
one primary attempt, compute21:30Z and closeout22:00Z on2026-11-02.

Next: targeted data-free tests/review/merge, full-path TRAIN smoke, then the
single source-bound primary measurement. Large artifacts remain outside Git.
