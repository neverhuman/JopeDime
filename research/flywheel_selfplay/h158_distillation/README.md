# H158 — R782-to-r882 listwise distillation

**PREREGISTERED; not executed.** Transfer a frozen R782 teacher distribution into the r882 adapter at lr1e-5; test whether excluding teacher loss on gate-failing fit contexts reduces harm. CTRL, KD03, KD07 and KD07_NOVETO train40 steps on three seeds,480 actual updates planned; R882 is unchanged.

[PREREG.md](PREREG.md) fixes the context veto, matched-slate KL, family split, KD-only Holm lower-bound rule and CPU limits. [manifest.json](manifest.json) pins source and inputs; [fold_manifest.json](fold_manifest.json) defines fit/assessment families. Both held-family and full TRAIN readouts are TRAIN-exposed/descriptive. No serving promotion.
