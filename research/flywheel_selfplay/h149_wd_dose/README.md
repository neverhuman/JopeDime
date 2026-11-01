# H149 — weight-decay dose follow-up

**Preregistered, not executed.** H144's nominal decay was too small to separate its adapters' drift or validation choices. H149 compares CTRL (wd0.0001), WD_5 (5.0) and WD_25 (25.0), retaining the H144 lr0.0001 /40-update recipe and frozen r882 initialization.

[PREREG.md](PREREG.md) fixes all three seeds, family fold, 19-family median-seed bootstrap/Holm2 and the manipulation check: median drift within inclusive5% of CTRL is DOSE_INVALID and cannot be read as a null. [fold_manifest.json](fold_manifest.json) excludes fold0 from fit; the full1219-context TRAIN secondary is explicitly TRAIN-exposed/descriptive only. [manifest.json](manifest.json) pins source and inputs. [metrics.json](metrics.json) contains no results.

Next: constructor-only smoke, self-review plus ARGON read-only review, merge, then one CPU run. No live-loop change or promotion.
