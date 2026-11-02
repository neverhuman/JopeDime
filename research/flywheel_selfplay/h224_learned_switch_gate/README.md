# H224 — learned FIT-only switch gate

**RUNNING; no treatment outcome read at amendment A3.** Hypothesis: a logistic gate using five outcome-free
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

Attempt1 launched at14:05Z on2026-11-02; native91 passed bit-exactly and the
four-fit FIT-only smoke passed. Attempt1 retains its18:00Z compute stop and
18:30Z closeout; the conditional continuation below applies only if needed.

## Pre-outcome amendment A2 — load pause (2026-11-02T14:00:10.506879+00:00)

Owner `hw_t1400_loadguard.md` replaces load-triggered abort with pause. The
supervisor sends SIGSTOP only to its own H224 worker PID when load5>60, keeps
all hard guards active, and checks the pause decision every60 seconds. Resume
with SIGCONT only when load5<=55. A hard RSS/MemAvailable/free-disk/deadline
failure resumes a stopped worker only to terminate its owned process group.
The absolute18:00Z compute stop and18:30Z closeout do not move during pauses.
No H224 claim, baseline, fit or outcome existed at this amendment. Science,
seeds, fits, logistic choices and all other limits are unchanged.

## Pre-outcome amendment A3 — one continuation (2026-11-02T15:38:43.385940+00:00)

Owner `hw_t1530_astra_h224_cont.md` authorizes one conditional continuation
after the unchanged18:00Z attempt-1 stop, reusing only verified fit pairs.
If attempt1 completes45 fits, skip it. Otherwise the sole continuation has a
22:30Z absolute compute stop and23:00Z closeout, with identical scientific
code/seeds/folds/core IDs and resource guards. No further attempt is allowed.
[CONTINUATION.md](CONTINUATION.md) specifies admission, immutable harness
custody, artifact reuse and cumulative accounting. No treatment outcome has
been read at this amendment; attempt1 and [PREREG.md](PREREG.md) remain intact.
