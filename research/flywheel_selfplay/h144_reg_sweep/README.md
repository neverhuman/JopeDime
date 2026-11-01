# H144 — regularization sweep

PREREGISTERED, not executed. Hypothesis: stronger decoupled weight decay can reduce r882 continuation cliffs. CTRL, WD_1E2 and WD_1E1 each receive40 updates on three fresh seeds. DROP arms are prospectively excluded because the effective fused adapter has no active dropout; frozen DOPE is unchanged.

[PREREG.md](PREREG.md) fixes all arms, data, loss, sampler, equal-family median-seed estimand, bootstrap, Holm-four correction and resource limits. [fold_manifest.json](fold_manifest.json) fixes family fold144140. [manifest.json](manifest.json) pins inputs and executable source. [metrics.json](metrics.json) has no outcomes before execution.

Exact91-context baseline comes first; only29,211 adapter parameters train. Primary is19-family DL-Val; held-out DL-Train Fold0 is descriptive. Zero protected reads/live writes; CPU xbabe2 only after H142 exits, nice19, total threads<=4, RSS<6GB decimal. Next: merged-source run, verified S3 custody and closeout. No serving promotion.
