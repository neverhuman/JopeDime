# H186 — kernel dose and shape sweep

**PREREGISTERED; not executed.** Hypothesis: changing one revealed-history kernel constant improves r882 without fitting any weights, motivated by H184's kernel attribution.

[PREREG.md](PREREG.md) fixes seven arms, three family-fold reshuffles, imported H177 pooled selection with Holm m=7, B lock before A/treatment DL-Val, and zero optimizer updates. TOPK4/16 are prespecified structural no-ops with four revealed candidates. Pooled A and identical seed outcomes cannot count as independent confirmation. H178 calibrated four-arm selection, not a new seven-arm empirical experiment.

Inputs/source pins: [manifest.json](manifest.json), [provenance.json](provenance.json), [fold_manifest.json](fold_manifest.json). [metrics.json](metrics.json) records execution status. No serving promotion. Next: review and merge preregistration, run once under CPU guards, preserve/replay raw scores, and report all registered outcomes.
