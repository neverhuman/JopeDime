# H268 — Context-normalized reranker labels

Hypothesis: robust per-context relevance labels reduce H265 seed instability while improving equal-family single-pick utility. Setup: frozen H265 feature/model/selector code; CONTROL/RERANK4/CTXNORM/CTXNORM_SHUF;135 cells over three fold partitions and three model seeds. Zero neural updates; cached r882 scores retain native support conditioning, and the reranker adds no new test-time reveals.

Status: PREREGISTERED, no H268 outcomes read. See [PREREG.md](PREREG.md), [manifest.json](manifest.json) and [fold_manifest.json](fold_manifest.json). Regression tests and source integrity must pass before one bounded CPU run. Full outcomes go to versioned S3 before compact metrics packaging. No verdict yet; next step is the preregistered run and independent saved-evidence audit.
