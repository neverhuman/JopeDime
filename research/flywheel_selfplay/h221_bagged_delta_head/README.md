# H221 — bagged utility-delta head with a safe gate

**PREREGISTERED, NOT RUN.** Tests whether five family-bootstrap heads reduce
prediction variance and improve safe adoption while r882/DOPE stay frozen.
Arms: R882, DELTA_SAFE, primary BAG5_SAFE, and BAG5_SHUF_SAFE placebo.

[PREREG.md](PREREG.md) fixes H213's head and B gate, fresh221 seeds/folds,
165 fits /21,120 updates, cache and role boundaries, native91-first regression,
strict primary criteria, and one-attempt resource/deadline limits.
[manifest.json](manifest.json), [provenance.json](provenance.json),
[fold_manifest.json](fold_manifest.json), and [fold_jaccard.json](fold_jaccard.json)
bind source and real-S3 data. No trained DL-Val score or serving promotion.
Next: merge preregistration, regress native91, then run the fixed study.
