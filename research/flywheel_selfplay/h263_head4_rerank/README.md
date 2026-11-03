# H263 — trust-region head-4 reranking

PREREGISTERED; no outcomes read. Hypothesis: FIT-only pairwise ordering within frozen r882's top4 plus the H142 anchor improves single-pick utility. Four arms CONTROL/L2SP/HEAD4/HEAD4_NOTR,15cells each,40updates/cell, fresh folds/seeds and globally locked strict H257 selection.

See [preregistration](PREREG.md), [manifest](manifest.json), [folds](fold_manifest.json), [local metrics schema](metrics.schema.json), [metrics](metrics.json), [data provenance](provenance.json), [runner](runner.py), [analysis](analysis.py), [treatment](treatment.py) and [resource monitor](monitor.py).

Primary HEAD4-r882; secondary HEAD4_NOTR-r882; Holm m2 and3/3positive. Native91 exact prerequisite. Primary adds no oracle reveal/reselection, but retains Native91's fixed four support inputs. Five independently selected cell utilities are averaged per seed; no single deployable checkpoint claim. TOP4_PICK final only, descriptive. Ties retain r882: fallback identity, not equivalence.

CPU xbabe2 48,50-52/nice19/numeric1/aggregate4/RSS4GiB/H239-A1/load guard. Prereg17:15Z,compute18:45Z,closeout19:30Z 2026-11-03; one run/no outcome retry. Origin/v1/next only; no hostedCI/main/fastlane/PR2071. S3 H263/astra-t1635 exact-version member-verified custody.

Acceptance: `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 python3 -m pytest -q . -p no:cacheprovider`, `ruff check .`, `ruff format --check .`, JSON/schema, pinned-source and Git scope verification. Next step: publish frozen protocol, qualify Native91, smoke, then run the bounded wave and audit its terminal evidence.
