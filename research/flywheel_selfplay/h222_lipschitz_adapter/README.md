# H222 — Lipschitz smoothing of the adapter

Preregistered, not run. Test CTRL vs spectral normalization (SN) and a context
input-Jacobian penalty (JAC), with lambda chosen only by fresh TRAIN-family CV.
Frozen r882 COPY, frozen DOPE, real-S3 TRAIN1219/205 and DL-Val91/19 families.
Seeds222101–222103; fold seeds222141–222143. See [PREREG.md](PREREG.md),
[manifest.json](manifest.json) and [fold_manifest.json](fold_manifest.json).

The39 full fits use the same40-update r882 recipe. Candidate validation waits
until all choices and final states lock. Report paired family-bootstrap gain vs
CTRL/r882, all seed signs, harm and Elo. CANDIDATE requires strict positive CIs
and gains on every seed and the median against both references; ties fail.

CPU xbabe2, nice19, at most4 cores, RSS<4GiB, load5<=60, free disk>=300GB.
Compute stops17:00Z; closeout by17:30Z on2026-11-02. One merged-source TRAIN
smoke precedes one full attempt. No scientific outcome or optimizer update yet.
The next step is exact-head tests/review/merge, then source-bound smoke/run.
Large artifacts remain outside Git. [metrics.json](metrics.json) is unmeasured.
