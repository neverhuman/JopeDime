# H230 — COSWR × GATED_025 stack

**PREREGISTERED, NOT RUN.** Hypothesis: a COSWR-trained adapter plus the frozen
GATED_025 selector may clear the strict+.002 native and DL-Val plateau margins.
Neither predecessor alone established that result.

Four arms: unchanged R882, COSWR, G025, STACK. Three fresh230 seeds/folds cover
205 TRAIN families/1,219 contexts from the pinned real-S3 H124 packet. Only the
adapter trains; DOPE stays frozen. Fifteen40-update adapters and45 shared
HistGBMs are locked before treatment assessment. The first TRAIN smoke reuses
its registered artifacts; no extra attempt/update is authorized.

[PREREG.md](PREREG.md) defines the frozen recipes, FIT-only tau choice,
in-sample adapter/FIT limitation, primary/co-primary rules, boot interruption
handling and fixed resources. [manifest.json](manifest.json),
[fold_manifest.json](fold_manifest.json), [provenance.json](provenance.json)
and data-free tests retain identities and admission evidence.

Next: merge by22:00Z on2026-11-02, pass native91 and the TRAIN smoke, then run
the sole CPU attempt. Compute stops03:30Z on2026-11-03; closeout by04:00Z.
No outcome is available yet; no qualification or serving claim.
