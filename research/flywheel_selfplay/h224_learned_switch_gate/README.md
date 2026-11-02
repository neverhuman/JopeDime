# H224 — learned FIT-only switch gate

**PREREGISTERED, NOT RUN.** Hypothesis: a logistic gate using five outcome-free
switch features can retain more BLEND_025 gain than H217's scalar zgap gate
while controlling harm and passing the DL-Val +.002 plateau margin.

Arms: frozen R882, primary LGATE, unchanged GATED_025 reference, and LGATE_SHUF
within-family label placebo. [PREREG.md](PREREG.md) fixes nested FIT OOF labels,
FIT-only C/threshold selection,45 unique HistGBM fits/60 cell model uses,
zero adapter updates, native91-first regression and the global stage1 lock.

Fresh224 seeds/folds cover205 TRAIN families/1,219 contexts. Maximum mean
best-match fold Jaccard is0.192792 against H203/H210/H217/H218/H223.
See [fold_manifest.json](fold_manifest.json), [provenance.json](provenance.json)
and [manifest.json](manifest.json). No sealed, holdout or meta-test reads.

Next: merge preregistration, pass the four-fit non-scored smoke, then finish the
fixed CPU-only attempt and publish the result by18:30Z on2026-11-02.

## Pre-outcome amendment A2 — load pause (2026-11-02T14:00:10.506879+00:00)

Owner `hw_t1400_loadguard.md` replaces load-triggered abort with pause. The
supervisor sends SIGSTOP only to its own H224 worker PID when load5>60, keeps
all hard guards active, and checks the pause decision every60 seconds. Resume
with SIGCONT only when load5<=55. A hard RSS/MemAvailable/free-disk/deadline
failure resumes a stopped worker only to terminate its owned process group.
The absolute18:00Z compute stop and18:30Z closeout do not move during pauses.
No H224 claim, baseline, fit or outcome existed at this amendment. Science,
seeds, fits, logistic choices and all other limits are unchanged.
