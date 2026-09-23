# ASM maintainer and user manual

**Private maintainer edition, 23 September 2026.** This manual distinguishes implemented behavior, measured qualification, historical results and pending acceptance. Its implementation snapshot is reviewed PR83: source `68f56404c3e199e06413bfd645d3fe0803d7b96f`, merged main `a75e543d2166017cb0139e0a251d51d9a0c98e08`, with identical source trees.

- [Read the PDF](asm-maintainer-manual.pdf).
- [Read the complete Markdown book](internal/asm-manual/MANUAL.md).
- [Open the detailed property-list checklist](asm-maintainer-status.plist), or its [JSON companion](internal/asm-manual/data/roadmap.json).
- [Inspect the rendering receipt and input hashes](internal/asm-manual/render-receipt.json).

The book covers the thirty frozen datasets and all 522 feature records, the broader 3,859-dataset catalog, local and remote storage inventories, encodings and neural models, installation and operation, language support, measured examples, historical ranking/Elo plots, maintenance, and a 38-item acceptance roadmap. Sealed target bodies and model weights are not included.

## Section Markdown

| Section | Contents |
| --- | --- |
| [01 — Current status](internal/asm-manual/sections/01-status.md) | What works, evidence levels, measured results, cost and remaining acceptance |
| [02 — Architecture](internal/asm-manual/sections/02-architecture.md) | Components, encodings, neural model dimensions, checkpoint identities and compatibility |
| [03 — Operations](internal/asm-manual/sections/03-operations.md) | Current and pending commands, installation, runtime preparation and recovery |
| [04 — Training](internal/asm-manual/sections/04-training.md) | Admission, task routing, resource budgets, optimizer proof, reload and continuation |
| [05 — Languages and examples](internal/asm-manual/sections/05-languages.md) | Complete recipe matrix, fitted-state distinctions and fourteen executed example commands |
| [06 — Benchmark methods](internal/asm-manual/sections/06-benchmark-methodology.md) | Frozen folds, metrics, class order, ranking/Elo, exposure and accounting |
| [07 — Data estate](internal/asm-manual/sections/07-data-estate.md) | Panel/corpus statistics, locations, bytes, provenance and metadata gaps |
| [08 — Visual evidence](internal/asm-manual/sections/08-visual-evidence.md) | Current measured loss, dataset coverage, historical Elo/ranks/relative loss and timing |
| [09 — Maintenance](internal/asm-manual/sections/09-maintenance.md) | Checkout ownership, resource coordination, reviewed delivery and preservation |
| [10 — Detailed roadmap](internal/asm-manual/sections/10-roadmap.md) | Thirty-eight stable acceptance items with dependencies, commands, resources and evidence |
| [11 — Evidence and glossary](internal/asm-manual/sections/11-evidence-glossary.md) | Source/CI/review/main identities, receipt locations and terminology |
| [12 — Dataset appendix](internal/asm-manual/sections/12-dataset-statistics.md) | Every frozen dataset, class/category frequencies and all feature statistics |
| [13 — Source map](internal/asm-manual/sections/13-source-map.md) | Exact-revision source links, pending source pins, companion files and rebuild commands |

Historical ladders are not current ASM panel results. The current one-dataset operational proof demonstrates the implemented path but loses to its generic comparison arm. Full-panel performance, new bounded training and complete selected fitted-model export qualification remain pending. Public distribution rules remain separate from this private manual.
