# InnovationZero publication

**InnovationZero: Learning to Propose Algorithms Before Testing Them** is a complete
research manuscript, methods supplement and immutable source archive. It is labeled
**protocol and evidence audit**: the central first-proposal generalization experiment
has not been reported as completed. Publication polish does not change that evidence
status.

## Build and read

From the repository root:

```sh
make -C docs/paper/innovationzero check self-test pdf
```

The three outputs are written to the lane's reused `target/innovationzero-paper/`:

| Output | Contents |
| --- | --- |
| `InnovationZero.pdf` | Main research manuscript, two-column Palatino text and math, vector figures and references |
| `InnovationZero-Supplement.pdf` | Methods, objective lineage, statistics, historical effort and research context |
| `InnovationZero-Archive.pdf` | Complete organizer master, exact participant packet, and all four frozen Rust sources |

Keep the three PDFs beside each other so cross-volume links resolve. Their source
files are deliberately separate: edit one section without editing the complete paper.
The archival files in [`artifacts/`](artifacts/README.md) are the byte-level authority;
the source PDF adds page breaks, line numbers and wrapping for reading.

### Dependencies

The build uses Python 3.9 or newer, Make, `latexmk`, pdfLaTeX, XeLaTeX, BibTeX and
Poppler (`pdfinfo`, `pdftotext`, `pdffonts`). On Debian/Ubuntu:

```sh
sudo apt-get install make python3 latexmk texlive-latex-extra \
  texlive-fonts-recommended texlive-xetex fonts-dejavu-core fonts-urw-base35 poppler-utils
```

There is no network access during the build, no shell escape, no executed embedded
Rust source, and no model, training or benchmark dependency. Figures are committed
vector PDFs and SVGs. Regenerating them uses the separately documented Python figure
script and its plotting dependency; that is not needed to typeset a fresh checkout.

### Verification and reproducibility

- `make check` verifies TeX imports, included figures, bibliography keys, the absence
  of unresolved editorial placeholders, and every archived file's byte count and
  SHA-256 digest. Elo ratings, intervals and pairwise values must exactly match the archived result receipt; numerical finiteness, panel scope and oriented ratio calculations are checked. It requires all four complete source artifacts and both prompt
  variants. Validation checks source identity, not scientific correctness.
- `make self-test` exercises eight refusal paths using disposable fixtures: altered
  size, altered digest, duplicate identity, path escape, unresolved citation and
  missing section, numerical rating drift and a falsely qualified experimental arm. It never changes the real archive.
- `make pdf` runs both checks required by the build and then rejects missing PDFs,
  unresolved references/citations, missing glyphs, overfull boxes, unembedded fonts
  and PDFs without extractable text. Review rendered pages as well: programmatic
  checks cannot establish visual quality or scientific adequacy.
- Output timestamps use `SOURCE_DATE_EPOCH=1792655756`, the pinned main evidence
  snapshot's commit timestamp. Set this variable explicitly to use another release
  epoch. The build writes `build-environment.txt` with the current source HEAD and
  tool versions. Exact PDF-byte reproducibility is scoped to matching tools/fonts;
  different TeX distributions can paginate or serialize differently.
- `make clean` removes only the known products of this paper build. It never
  recursively deletes `target/`, other lanes or preserved evidence.

## Editorial structure

| Source | Purpose |
| --- | --- |
| `main.tex`, `style.sty` | Main assembly and shared typography; no venue branding or invented author roster |
| `sections/00-abstract.tex` | Standalone abstract paragraph |
| `sections/01-introduction.tex` | Scientific question, historical motivation and contributions |
| `sections/02-setting.tex` | Commitment, information and fitting boundaries |
| `sections/03-method.tex` | Existing model lineages, protected interface, support and learning objectives |
| `sections/04-evidence.tex` | Code/report distinctions and exploratory four-artifact comparison |
| `sections/05-protocol.tex` | Prospective family-level experiments and Move42 criteria |
| `sections/06-accounting.tex` | Token categories, yield and symmetric cost amortization |
| `sections/07-discussion.tex` | Limitations, disclosure and future scope |
| `sections/08-conclusion.tex` | Evidence-bounded conclusion |
| `appendices/*.tex` | Methods, statistics, effort and contextual detail |
| `references.bib` | Shared bibliography |
| `evidence/artifact-manifest.json` | Exact file hashes, sizes, source revisions and snapshot distinctions |
| `evidence/claims.json` | Claim-level evidence and qualification |
| `evidence/elo-summary.json` | Figure data copied from the separately pinned exploratory result |
| `scripts/figures.py` | Deterministic vector figure generation |

## Evidence and release discipline

The main evidence snapshot is `bdf9441aaa17f87dcb37053d33b185ba2217bf17`.
Additional four-source results are pinned separately to PR #407 at
`584713ed3f512057361460567bb56d620162286a`; inclusion does not assert that this branch
was merged into the main snapshot. The evidence section explains training-family,
preprocessing, adaptive-seed, custody and hardware limitations next to the result.
The source archive preserves receipt wording even when the manuscript corrects its
interpretation. Never silently edit an original receipt to repair a narrative error.

The chosen experimental proposal uses failure-inclusive utility clipped to `[-1,1]`
and a superiority margin of **0.02 utility units**, not a claimed 2% RMSE gain.
Proposed thresholds, a proposed 512-dimensional solution representation and Move42
must remain labeled until their registered evidence exists. No hypothetical winning
curves, token-efficiency claims or unobserved experiments are drawn as results.

This directory preserves public challenge/source material without exposing private
GP productions, token mappings or proprietary decomposition rules. Releasing output
programs is not equivalent to releasing the generator's training implementation.
Before a venue submission, confirm the author list, venue style, references, access
arrangements and the highest supported claim against the current evidence manifest.
