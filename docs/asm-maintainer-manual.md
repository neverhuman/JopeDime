# ASM maintainer and user manual

**Private maintainer edition 3, rebuilt with the current core-technology explanation.** The core book adds an ELI5 explanation, current executable pipeline, network diagrams and the unmeasured transfer-reporting contract, while preserving the dated PR83 sections and historical observations. PR88 subsequently integrated the learning and delivery interfaces; see the [current integration runbook](experimental-learning-preparation.md). Source and CPU fixture support do not establish live learning qualification.

**Publication record.** This manual distinguishes implemented behavior, measured qualification, historical results and pending acceptance. The [publication manifest](internal/asm-manual/publication.json) records the implementation snapshot, dated observations and document build independently.

- [Read Volume 1: maintainer manual](asm-maintainer-manual.pdf).
- [Read Volume 2: evidence reference](asm-evidence-reference.pdf).
- [Read the complete Markdown book](internal/asm-manual/MANUAL.md).
- [Open the detailed property-list checklist](asm-maintainer-status.plist), or its [JSON companion](internal/asm-manual/data/roadmap.json).
- [Inspect the rendering receipt and input hashes](internal/asm-manual/render-receipt.json).

The book covers the thirty frozen datasets and all 522 feature records, the broader 3,859-dataset catalog, local and remote storage inventories, encodings and neural models, actual historical training/target progress, loss curves, binary packing, continuation commands, language support, measured examples, historical ranking/Elo plots, maintenance, and a 43-item acceptance roadmap with 24 separately marked follow-on proposals. Sealed target bodies and model weights are not included.

**For the learning story, start with chapters 7–8 and 13, then appendices B–D in Volume 2.** They distinguish real optimizer updates from better teachers, improved conditional program generation, and a complete measured flywheel. Current product version, accepted checkpoint, historical continuation, and pending experimental interfaces have separate identities.

The retained recovery history includes all 63,000 reported steps from 6,001 through 69,000, with 378,000 exact metric values in a compressed companion. Loss plots mark corpus changes and accepted step 60,000; internal self-play Elo and adaptive search comparisons are explicitly separated from benchmark-valid evidence. Packing measurements and proposed smaller formats have separate qualification requirements.

## Section Markdown

| Section | Volume |
| --- | --- |
| [1 — Current status, versions, north star and measured limitations](internal/asm-manual/sections/status.md) | 1 |
| [2 — Architecture, encodings, program representation and model lineages](internal/asm-manual/sections/architecture.md) | 1 |
| [3 — Installation, preparation, assets, workers and doctor](internal/asm-manual/sections/installation.md) | 1 |
| [4 — Data, splits, groups, families and permitted input boundaries](internal/asm-manual/sections/data.md) | 1 |
| [5 — Regression search, fitting, prediction and focused optimization](internal/asm-manual/sections/search.md) | 1 |
| [6 — Teacher admission, HWM identities and corpus updates](internal/asm-manual/sections/teachers.md) | 1 |
| [7 — Training, continuation, export and compatibility](internal/asm-manual/sections/training.md) | 1 |
| [8 — Measuring repeated learning and deciding improvement](internal/asm-manual/sections/learning.md) | 1 |
| [9 — Complete fitted inference and language support](internal/asm-manual/sections/inference.md) | 1 |
| [10 — Benchmark execution, ranking, Elo and interpretation](internal/asm-manual/sections/benchmark.md) | 1 |
| [11 — Scheduling, cancellation, recovery and accounting](internal/asm-manual/sections/scheduling.md) | 1 |
| [12 — Maintenance, privacy, release and preservation](internal/asm-manual/sections/maintenance.md) | 1 |
| [13 — Prioritized implementation plan and extension interfaces](internal/asm-manual/sections/plan.md) | 1 |
| [A — All thirty datasets and all 522 feature records](internal/asm-manual/sections/datasets.md) | 2 |
| [B — Historical training, checkpoint lineage and original loss trajectories](internal/asm-manual/sections/training-history.md) | 2 |
| [C — Historical targets, rankings, Elo and attribution limits](internal/asm-manual/sections/targets.md) | 2 |
| [D — Dated data estate, recovery evidence and binary formats](internal/asm-manual/sections/data-estate.md) | 2 |
| [E — Full task catalogue, source/schema index, receipts and glossary](internal/asm-manual/sections/catalogue.md) | 2 |

Historical ladders are not current ASM panel results. The current one-dataset operational proof demonstrates the implemented path but loses to its generic comparison arm. Full-panel performance, new bounded training and complete selected fitted-model export qualification remain pending. Public distribution rules remain separate from this private manual.
