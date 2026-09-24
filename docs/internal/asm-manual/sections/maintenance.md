# Maintenance, privacy, release and preservation

## Questions that should remain answerable after a maintainer handoff

Which reviewed source and runtime produced this artifact? Which exact rows and original labels were available? Was this a recipe, a checkpoint or a saved fitted predictor? Which score selected it, under what comparison identity? Which failures, missing dependencies and cancelled jobs were retained? Did any nominal test labels influence selection? Did resume preserve the original deadline? Does the export predict from saved state, and which backend actually passed? Did training perform positive updates and change parameters? Can another maintainer locate and authenticate the receipt without reading credentials or sealed data?

The roadmap ([§](plan.md#compact-task-index)) translates the unanswered parts into named acceptance items. Unknowns stay explicit until the corresponding evidence exists.

## Repository authority and coordination

The canonical product checkout is `/home/ubuntu/JopeDime`, upstream `veox-ai/JopeDime`. `/home/ubuntu/jain-split/jope-dime` is a different research repository; `/home/ubuntu/jope-dime` may be its symlink. Similar names are not interchangeable source, checkpoint or promotion authorities.

Before source changes, inspect the open PR queue, worktrees, branch/full HEAD, dirty state and latest `AGENT_CHAT.md`. Under `flock` on `AGENT_CHAT.md.lock`, reread overlapping claims and append checkout, branch, full HEAD, owner, paths and verification. The append lock does not permit two writers in one checkout. An idle session or old claim is not a stopped-head handoff.

Keep one writer per checkout and one implementation PR queue. Preserve pending work and authorship. A resumed campaign retains its resource limits and deadline.

Repository operation remains serialized: read the current queue/ledger, claim paths under `AGENT_CHAT.md.lock`, keep one writer per checkout, and preserve pending work. Changes require exact-head CI, independent eligible review/native approval, guarded rebase merge and verification of remote `main` plus `main-guard`. A changed head invalidates an earlier head's approval. Preserve evidence before deleting closed lanes or build directories; `target/` can contain irreplaceable receipts.

## Worktrees and builds

Extra copies belong under `<repo>/.agent/worktrees/`, at most four lanes. `.claude/worktrees` stays a symlink into that shared root. Verify it before worktree operations and keep `.agent/` locally excluded from Git status.

Each lane reuses its own `target/`. Before Cargo:

```sh
export CARGO_INCREMENTAL=0
export CARGO_PROFILE_DEV_DEBUG=line-tables-only
export CARGO_BUILD_JOBS=4
```

Do not set `CARGO_TARGET_DIR` globally or create a new target per task. CI selects persistent PR and trusted-main storage separately. Resolve binaries from Cargo artifact messages: guessing a hashed test executable can select an old binary built with different features.

Never put source, worktrees or builds in `/tmp`. A target directory can contain evidence. Cargo markers do not prove that every descendant is disposable.

## Coverage and reviewed lifecycle

### Coverage inventories and CI lanes

Read `deploy/ci/coverage.md`, mapping and generated inventory together. Discovery counts packages, features and families; execution receipts count what one invocation actually collected and ran. Invocations overlap. Path exclusions, ignored tests and unavailable dependencies do not become passing tests.

Fast CI has eight lanes plus aggregate. Full-main diagnostics cover additional families and have recorded failures/unavailable capabilities. A fast gate does not establish all-feature, GPU or live-S3 qualification. Regenerate inventories and scores through their owning proof scripts; never hand-edit generated evidence.

### Writer and reviewer identities

The established writer is `jepsontaylor`; the designated independent Claude reviewer uses `neverhuman`. Verify them for every PR. The reviewer must differ from the opener and every associated commit author and committer. Use the full paginated inventory and record unassociated commits.

```sh
/home/ubuntu/.local/bin/gh-role writer api user --jq .login
/home/ubuntu/.local/bin/gh-role eligible reviewer veox-ai/JopeDime PR_NUMBER
```

The helper prevents identity mistakes but cannot prove independence. Changing credentials does not turn a writer into a reviewer. Never print tokens or credential-bearing environments.

### Freeze, review and merge

Freeze the full head; dispatch fresh manual CI; retain source SHA, run ID, attempt and dispatch time. Verify current-attempt jobs/aggregate and nonzero meaningful coverage. Supply the independent reviewer with diff, acceptance commands and evidence. A native approval must explicitly bind the final head and remain undismissed. Changed source repeats checks and review.

```sh
/home/ubuntu/.local/bin/gh-role writer pr merge PR_NUMBER \
  --rebase --match-head-commit FULL_REVIEWED_SHA
```

Immediately recheck head, eligibility, CI and approval before merging. Afterwards verify `merged_at`, fetch remote main, compare the reviewed tree and verify main-guard. Preserve any ledger tail or other dirty work before restoring clean verified main. Protection enforcement has been procedural; a protection API 403 does not prove a preventive branch lock. Main-guard detects unreviewed pushes after they occur.

### Main-guard and full-main scheduling

Main-guard and the full-main diagnostic share the single xbabe2 `jope-full-cpu` slot and run serially. A main push queues both; full-main has a 120-minute timeout, so main-guard can wait behind it. Read back job/run states and retain the exact source/run/attempt before changing the queue. In the PR83 lifecycle the owner-authorized full-main diagnostic was cancelled while still queued, before any job started, and the owned full timer was restored to allow main-guard to run. That is a recorded unexecuted diagnostic, not passing coverage or general permission to cancel running work. Follow the current allocation owner and runner runbook; never stop another running job to accelerate a merge. Sources: `.github/workflows/main-guard.yml:17`, `.github/workflows/full-main.yml:24`, `docs/runbooks/runner.md:15`, `:131`; the PR83 lifecycle cancellation receipt.

## Privacy boundaries: the private manual and public fitted bundles

This is a maintainer document for the private product repository. Its explicitly requested GP examples and internal operational paths explain the implementation; they do not expand the allowlist for public inference bundles.

This is a maintainer document containing internal model and discovery design. The user's request authorizes this explanation and the companion private GP examples. It does not enlarge the allowlist for distributed inference artifacts or authorize publishing private weights, discovery histories, vocabulary assets or training corpora. Hashes identify dependencies; they are not the dependencies themselves.

Allowlisted public fitted bundles contain inference code, necessary fitted input dictionaries, class/schema metadata and their hashes. They exclude the private recipe, discovery history and scoring references. The private prepared plan and worker directory remain private. Canary/text-policy checks supplement the explicit file allowlist; passing static checks does not replace actual complete-prediction parity.

## Release and installation are independent from model promotion

The current tag workflow runs for `v*`, checks tag = workspace = Python version, builds a Linux binary, writes build provenance and checksums, builds the Python wheel, and records model/dataset references by hash. The committed build command selects default CLI features; it does **not** include `--features asm`. Therefore the existence of this workflow does not prove that a released binary can execute the private ASM path. No newly published release is asserted here. The current private source build in [§](installation.md#development-build-of-the-cli) explicitly selects `asm`.

[Release workflow](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/.github/workflows/release.yml#L71) and [version consistency](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/deploy/ci/version-consistency.sh#L1) are source evidence. A future reviewed version bump can use `just bump X.Y.Z`; that recipe edits manifests and the lockfile and invokes Cargo. It is a developer mutation recipe, not a command executed for this manual or a decision to release any particular version. [Exact bump implementation](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/justfile#L63).

The checksum-verifying private binary installer and CLI-only source-build helper remain product28 follow-ons. Their concrete commands and cold/warm separation are in [§](installation.md#build-and-install). Installing a new CLI does not admit new data, train a model, or promote a checkpoint. Likewise, `models pull` or a storage manifest does not change the accepted policy pins.

## Preservation and lane closure

Before deletion inspect `git status`, `git stash list`, `git log --all --not --remotes`, and local versus remote tags. A branch-only log misses commits elsewhere.

Preserve unique source/evidence for the tree being removed, restore independently into the reusable bare verification directory without alternates, and verify the exact commit, tree and connectivity. A valid bundle checksum does not prove the desired detached HEAD was included.

After preservation and a fresh consumer check, remove a closed lane with `git worktree remove`. Indefinite redundant copies are not an alternative to the required reclamation step. Unknown contents remain protected. Follow `deploy/maintenance/README.md`; its reporting timer proposes candidates and never authorizes deletion. Backup, replication and retention operations remain separately authorized: closing a lane does not authorize them, and neither does this manual.

## Rebuilding and updating this manual

Rebuilding the book is itself a runbook. Its summary:

| Runbook part | Rebuilding this manual |
|---|---|
| Purpose and status | Render both volumes from the committed Markdown, figures and data. **available**: on 2026-09-23 `build_manual.py` rendered both edition-2 volumes of the owner layout, with every gate passing and no warnings. Each build is identified by its own `render-receipt.json`. |
| Prerequisites and resources | Python 3.12 with the direct versions in `tools/requirements-docs.txt`, Pango/Fontconfig, DejaVu fonts and Poppler; local CPU only. The build opens no network resource, model, dataset or sealed label. |
| Manifest-creation command | None. `book.json`, `publication.json` and `aliases.json` are reviewed inputs edited directly; no committed command generates them. |
| Execution | The figure scripts when their data changed, then the renderer ([§](#rebuilding-the-book), [§](#regenerating-the-training-and-target-figures)). |
| Artifacts and success | Two PDFs under `docs/`, `manual.html`, `MANUAL.md`, `structure.json` and `render-receipt.json`. Success is a zero exit with every gate passed, followed by inspection of the rendered pages and extracted text. |
| Failure and resume | A failed gate exits nonzero and can leave a PDF without a render receipt; such a PDF is not a build. Fix the input and rerun from the start: there is no partial resume. |
| Limitations | PDF bytes are not reproducible bit for bit; the renderer does not check that a figure is newer than its data; it runs no model. |

### Rebuilding the book

The 2026-09-23 builds used Python 3.12 and the observed direct versions in `tools/requirements-docs.txt`. That file is not a transitive lock or a verified cold installer. A fresh environment also needs working Pango/Fontconfig, DejaVu fonts, and Poppler's `pdfinfo`, `pdftotext` and `pdftoppm`. The render receipt records exact Python/library versions and font-file hashes. The following environment-creation recipe is provided for rebuilding; it was not executed as a fresh installation during this document audit:

```sh
MANUAL_ENV=/home/ubuntu/.venvs/jope-manual
python3 -m venv "$MANUAL_ENV"
"$MANUAL_ENV/bin/python" -m pip install \
  -r docs/internal/asm-manual/tools/requirements-docs.txt
. "$MANUAL_ENV/bin/activate"
```

From the repository root, render the audited retained inputs with these local-only commands:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 docs/internal/asm-manual/tools/make_inventory_figures.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 docs/internal/asm-manual/tools/make_figures.py
python3 docs/internal/asm-manual/tools/build_manual.py
pdfinfo docs/asm-maintainer-manual.pdf
pdfinfo docs/asm-evidence-reference.pdf
pdftotext -layout docs/asm-maintainer-manual.pdf /path/to/durable/review/volume-1.txt
pdftotext -layout docs/asm-evidence-reference.pdf /path/to/durable/review/volume-2.txt
```

The two figure scripts above do not regenerate the training and target figures. A complete rebuild also runs the four scripts of [§](#regenerating-the-training-and-target-figures) before `build_manual.py`, because the renderer does not check that a figure is newer than its data. `make_figures.py` still defines a fixed host path to the ASM custody root (`tools/make_figures.py:39-41`), but never uses it.

Rendering uses Python Markdown, BeautifulSoup and WeasyPrint with system DejaVu fonts; their observed versions and font hashes are retained in the render receipt. Plotting uses Matplotlib and NumPy. The renderer rejects missing/nonlocal images and failed resource fetches, and checks that inputs did not change during rendering. It does not run models, fit predictors, fetch datasets or open sealed labels. Inspect the resulting pages and extracted text before treating a successful render as a complete document.

Byte-identical PDF rebuilding is not promised. `SOURCE_DATE_EPOCH` fixes font-subset timestamps, but the observed WeasyPrint 68.1 tagged-table identifiers contain process-specific values. Preserved first-edition repeat renders had identical normalized logical objects and decoded non-container streams after accounting for those identifiers, while their PDF hashes differed. Keep each output with its own render receipt and verify input hashes, content and layout; do not treat a changed PDF hash alone as changed measurements. This observation concerns the renderer, not numerical predictor reproducibility.

### What the renderer reads, checks and writes

`tools/build_manual.py` takes the layout from `book.json` (schema `asm-manual.book/v1`):

- **Chapters and appendices.** `chapters` is a flat list of section files in reading order, numbered from 1 when rendered. `appendices` lists files whose letters must run A, B, C and onwards. The renderer refuses a file listed twice, a file in `sections/` that `book.json` omits, and a listed file that does not exist.
- **Volumes.** An appendix with a `volume` key is published in that volume's PDF. Edition 2 publishes chapters 1 to 13 as Volume 1, `docs/asm-maintainer-manual.pdf`, and appendices A to E as Volume 2, `docs/asm-evidence-reference.pdf`. Volume 1 ends with a signpost page that keeps each appendix's letter, title and sections and links into Volume 2. Volume 2 has its own title page and two-level contents.
- **Position-independent anchors.** Section files are named by topic, not by position. A heading's anchor is `c-<file stem>--<GitHub heading slug>`, and a chapter title's is `c-<file stem>`, so moving a chapter does not change its anchors. Write a cross-reference as a Markdown link whose target is `file.md#slug`, or `#slug` within one file. The renderer fills a link whose text is exactly `§` with the target's number and title, and refuses any internal link it cannot resolve.
- **Aliases.** The required `aliases.json` maps all 266 published older destinations, such as edition 1's `chapter-NN` and `chapter-NN-M` anchors, to current headings. Each alias becomes an additional anchor on its heading. Destinations moved to Volume 2 also retain a named landing in the original Volume-1 PDF, with an explicit link to the exact semantic target. An alias that names no heading, or that repeats an existing heading anchor, is refused.
- **Strict heading check.** Each file must begin with exactly one `#` title. Headings go at most four levels deep and may not skip a level. Headings under one parent need distinct titles; a heading followed directly by a heading of the same or a shallower level, or by nothing, is refused as empty; and a heading may not have exactly one subdivision. Fewer than two sections in a file is only a warning. Headings are numbered by the renderer, never by hand, and heading-like lines inside fenced code are not treated as headings.
- **Size gate.** Each PDF must be at most 5 MiB (5,242,880 bytes), the repository's per-file cap. A larger volume stops the build.
- **Pinned inputs.** Every Markdown, JSON, JSON Lines, CSV, Python, CSS, PNG, SVG, property-list, gzip and text file under `docs/internal/asm-manual/`, except the generated outputs, is hashed together with `docs/asm-maintainer-status.plist` and the landing page `docs/asm-maintainer-manual.md`.
- **Outputs.** Besides the PDFs: `manual.html` (Volume 1), the combined `MANUAL.md`, `structure.json` with every numbered heading, figure and alias, and `render-receipt.json` (schema `jope.manual-render/v2`). The receipt's `volumes` list gives each PDF's name, page count, SHA-256 and bytes; its top-level `pages`, `pdf_sha256` and `pdf_bytes` describe Volume 1 only. It also records heading counts per level, the alias count, warnings, the Python, library and font identities and every input hash, and it is written only after every gate has passed.

### Regenerating the training and target figures

Regenerate the training and target figures before the PDF after changing an audited source CSV:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 docs/internal/asm-manual/tools/make_training_figures.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 docs/internal/asm-manual/tools/make_target_figures.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 docs/internal/asm-manual/tools/make_legacy_training_figure.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 docs/internal/asm-manual/tools/make_per_step_figure.py
python3 docs/internal/asm-manual/tools/build_manual.py
```

These commands render retained observations; they do not train a model or reconstruct unavailable metrics. The original-history verifier needs the separately preserved source files named in each CSV. Plot rebuilding uses the committed CSVs. No missing source or changed hash may be converted into a successful historical verification.

### The publication record and its three identities

Three identities describe a copy of this book. They change independently, and none implies the others:

| Identity | What it names | Rule and edition-2 value |
|---|---|---|
| Implementation snapshot | The reviewed source that the book documents | The publication manifest pins PR86 source `0079ceddbaba739d736e80f6fe9b999955c4ad3a`, merged 2026-09-24T00:04:12Z as main `2b375ef11fc9298ffa67d4d03e0bb83053201802`, tree `01ecd51afc77e753ee4fe1f3fbe3c130803bbf43`. These documentation corrections preserve the PR83 application implementation; its original reviewed identities remain in chapter 1. |
| Qualification observations | Each measurement the book reports | Each keeps its own observation time and receipt, such as the fourteen-command example run of 23 September 2026 ([§](inference.md#measured-execution-of-the-private-examples)). Rebuilding the book repeats none of them. |
| Document build | The edition, its pinned inputs and its PDFs | Recorded by the build itself: the input hashes and each volume's PDF hash, bytes and pages in `render-receipt.json`. |

The renderer requires `docs/internal/asm-manual/publication.json` and prints it in the front matter as the publication record, with one subsection per identity. It validates the schema, UTC timestamps, implementation identity format and both PDF filenames, and refuses undeclared PDFs under `docs/`. The snapshot subsection gives source, main, merge time, tree and an optional note. The observations form a table of observation, observed UTC time and receipt. The document-build subsection states the edition and its date, `SOURCE_DATE_EPOCH`, the number of pinned inputs and a SHA-256 digest of their path and hash list. PDF hashes stay in `render-receipt.json`, because a PDF cannot contain its own hash. `publication.json` is a reviewed input edited directly and is itself pinned: the renderer derives none of it from Git and refuses to build without it. The declared `build_utc` sets `SOURCE_DATE_EPOCH`; covers and status counts derive from the manifest and task inventory. Run `python3 docs/internal/asm-manual/tools/verify_publication.py` after rendering to reject stale or missing input/output hashes, size violations and inconsistent task priorities or statuses across Markdown, JSON and plist.

### Updating the book

Refresh inventories and receipts, then section Markdown, figures, checklist, `publication.json` and both PDFs. Keep a rendering input manifest. A checkpoint does not update performance figures by implication; merged training source does not prove actual learning. Historical Elo remains historical after regeneration.

A manual change lands through the same reviewed lifecycle as source ([§](#freeze-review-and-merge)): fresh exact-head CI and an independent non-contributor native review of the final head. Every committed file, including the rendered PDF, must stay within the repository's 5 MiB (5,242,880-byte) file cap (`deploy/ci/check.sh:46`), and because the rebase merge lands every commit of the branch, the cap applies to intermediate commits as well. The earlier publication, PR84 (published head `e06c0551`), was closed as superseded for that reason: the independent non-contributor review 5295763455, a COMMENTED review at `e06c0551`, found two files above the cap in three ancestor commits. Its clean-history successor, commit `2022017a8` on `docs/asm-maintainer-manual-final`, is a single commit on top of main `a75e543d2166…` that carries a 203-page PDF (`69418c48…`). It was published as PR85 and merged at 2026-09-23T22:33:36Z as main `bdfc95c63d905c647e41175e79dc30cdc8600287`, after independent native approval 5297559032 at that exact head and full CI run 35927921851; the following main-guard run 35929010658 passed. This second edition restructures and corrects that published edition. The original authoring commits `7b9398b76` and `bda953f62` are preserved; their changes and the unfinished content pass were integrated on `docs/asm-manual-edition2-integration`, retaining the separately merged PR86 corrections. Edition 2 is submitted as PR87. Each candidate head needs fresh exact-head CI and independent native approval before a reviewed merge; publication is not claimed from local rendering.

Before delivery check local links, source identities, totals, metric directions, captions, capability status and rendered pages. Extract PDF text and inspect dense tables, examples and figures. Preserve unavailable statistics and measurements explicitly rather than filling them with zero.
