# InnovationZero — publication sources

*InnovationZero: Learning to Propose Algorithms Before Testing Them* asks whether a model can read a
dataset and commit a working algorithm for it before executing any candidate. This directory holds
the sectioned manuscript, its technical supplement, a byte-exact archive of everything it compares,
and the checks that bind every plotted number to an archived receipt.

| Volume | Source | Contents |
| --- | --- | --- |
| `InnovationZero.pdf` | `main.tex`, `sections/`, `appendices/` | The paper: setting, challenge, frontier comparison (Elo, error magnitude, token yield, tokens per second, fitting cost), model, pilot, protocol, related work; appendices with every family's outcome, the pilot's per-dataset wins, the emitted programs, receipts, disclosure boundary, reproduction and claim ledger |
| `InnovationZero-Supplement.pdf` | `supplement.tex`, `supplement/` | S1 context, S2 method audit of the native objective, S3 statistical and resource contract, S4 effort reconciliation, S5 the organizer's red-team record |
| `InnovationZero-Archive.pdf` | `archive.tex` (entries generated from the manifests) | The challenge packet and master file, the four frontier sources and the ten emitted programs, each with bytes, lines and SHA-256 |

## Build

```sh
make -C docs/paper/innovationzero          # or: just paper
make -C docs/paper/innovationzero check    # sources only, no TeX
make -C docs/paper/innovationzero self-test
```

Outputs go to `target/innovationzero-paper/`. The build needs XeLaTeX, latexmk and BibTeX (TeX Live
2023+), Poppler, jq, curl, unzip and sha256sum; it is bash throughout (ADR-0006). Fonts — Libertinus
Serif and Math, Inter, JetBrains Mono, all SIL OFL — are fetched once from the release archives pinned
by SHA-256 in `fonts.lock` and cached outside the tree.

`scripts/check.sh` refuses the build if any archived artifact's bytes or SHA-256 differ from
`evidence/artifact-manifest.json`; if the plotted Elo, cost or source-size values differ from the
archived receipts; if `data/frontier-*.csv` do not regenerate from the archived match matrix
(`scripts/derive-frontier.sh`); if an input, figure or citation does not resolve or a section file is
not included; or if the manuscript uses the repository's restricted vocabulary
(`deploy/ci/ip-scrub.sh`). After typesetting it fails on overfull boxes, unresolved references,
missing glyphs and unembedded fonts. `make self-test` breaks a disposable copy thirteen ways and
requires each refusal to fire.

## Layout

- `sections/00-abstract.tex` … `10-conclusion.tex`; `appendices/A-frontier.tex` … `G-claims.tex`;
  `supplement/S1-context.tex` … `S5-organizer.tex` — one file per section.
- `figures/` — TikZ/pgfplots sources; plots read `data/`.
- `data/` — plotted tables: `frontier-entrants.csv` (transcribed from receipts, checked),
  `frontier-pairwise.csv`, `frontier-family-strip.csv`, `frontier-ratios.csv` (derived, checked),
  `pilot-*.csv`.
- `artifacts/` — exact copies of the challenge, the four sources, their receipts and the four-entrant
  benchmark outputs, with provenance in `evidence/artifact-manifest.json`.
- `evidence/` — `claims.json` (machine-readable claim ledger), `elo-summary.json`, `literature.md`
  (primary-source verification of the references), `pilot-2026-10-22/` (identical-cuts pilot: score
  table, Elo, compile and emission receipts, the ten emitted programs) and
  `gemini-authoring-statement-20261022.md` (a later self-report, archived verbatim and reconciled).
- `EDITORIAL_DECISIONS.md` — how the competing drafts and proposals were reconciled.

## Status

The paper reports a closed-world challenge, four frontier entrants on a 64-family exploratory panel,
and a pilot on ten datasets. It specifies, but does not report, the decisive track-C experiment: no
neural proposer in this repository has yet committed a first program on held-out data. Evidence
labels in the text (EST, PRE, PEND) and Appendix G say which is which. Private kernel-language
productions, token mappings and decomposition rules are not part of this release; emitted programs
are published in full.


The combined publication retains PR #411's task-specific objective section in
`sections/05b-objectives.tex`, its implementation/baseline distinctions, exploratory six-table
results and cross-objective protocol. The historical effort reference now targets Supplement S4;
the commitment reference targets the current definition. A nonsignificant small-pilot comparison
does not establish equivalence. Source conflicts were resolved without changing archived artifacts,
plotted frontier inputs or the three original author commits.


CI runs the reusable `paper.yml` from the existing admitted `ci.yml` and trusted-main
`full-main.yml` callers. Every PR/manual CI run executes `make check self-test` without
TeX, font downloads or PDF typesetting. The aggregate requires its source-bound receipt,
check output and verified payload hashes. The paper job uses the native event checkout
(synthetic merge on PRs), refuses unless its tree exactly matches the reviewed source
head, and retains both commit identities and their equal trees. Full manual CI checks
out the actual source commit. Main-push CPU qualification also requires these source
checks. Queueing or a missing receipt fails qualification.

Generate all three volumes explicitly with `just paper` when preparing a publication.
This runs the source/refusal checks and PDF rendering, including layout, embedded-font
and reference checks; rendering is not a product merge or CPU qualification requirement.
For a reproducible toolchain without installing host packages, use
`bash deploy/ci/publication-tools.sh just paper`. That wrapper extracts the SHA-256/size-pinned
official Ubuntu packages in `deploy/ci/publication-deps.lock.json` into the lane target cache
and generates an English XeLaTeX format in separate writable state. It needs no sudo,
host package installation, maintainer scripts, runner-image changes, corpus access or
credentials. Existing Chromium lock callers keep their original default. The package
cache and extracted runtime are verified on reuse.
