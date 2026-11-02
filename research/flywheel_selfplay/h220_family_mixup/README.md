# H220 — within-family mixup

**PREREGISTERED, NOT RUN.** Tests whether mixing adapter inputs and targets
within FIT families reduces training harm relative to r882 and matched CONTROL.
MIX uses Beta(0.4,0.4), probability0.5; no optional cross-family arm.

[PREREG.md](PREREG.md) defines exact pairing/mixing, fresh220 seeds and folds,
30 fits /1,200 full updates, native91 regression before training, strict primary
utility/Holm1/harm/seed gates, and the one-attempt resource/deadline contract.
[manifest.json](manifest.json), [provenance.json](provenance.json),
[fold_manifest.json](fold_manifest.json), and [fold_jaccard.json](fold_jaccard.json)
bind reproducible inputs, source and partitions. No trained DL-Val evaluation,
sealed/holdout/meta-test read or serving promotion. Next: merge, regress, run.
