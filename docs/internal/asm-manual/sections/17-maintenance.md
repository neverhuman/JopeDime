# Maintaining, reviewing and preserving the product

## Questions that should remain answerable after a maintainer handoff

Which reviewed source and runtime produced this artifact? Which exact rows and original labels were available? Was this a recipe, a checkpoint or a saved fitted predictor? Which score selected it, under what comparison identity? Which failures, missing dependencies and cancelled jobs were retained? Did any nominal test labels influence selection? Did resume preserve the original deadline? Does the export predict from saved state, and which backend actually passed? Did training perform positive updates and change parameters? Can another maintainer locate and authenticate the receipt without reading credentials or sealed data?

The roadmap ([§](03-roadmap.md#compact-task-index)) translates the unanswered parts into named acceptance items. Unknowns stay explicit until the corresponding evidence exists.

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

### Worked example: the PR83 lifecycle record

| Gate | Exact observation |
| --- | --- |
| PR | `veox-ai/JopeDime#83`, merged |
| Manual CI dispatch | Run 35887284772, attempt 1, source `68f56404…`, success |
| PR CI | Run 35887255898, attempt 1, source `68f56404…`, success |
| Required CI jobs | Eight lanes and aggregate succeeded in both runs |
| CI test inventory | 937 Rust passed, 14 skipped; 221 Python passed per run |
| First independent native review | 5293748183, `neverhuman`, APPROVED, exact full reviewed head, 16:29:01 UTC |
| Second independent native review | 5293788772, same eligible account, APPROVED, same head, 16:32:43 UTC; includes residual disclosures |
| Author eligibility | All twelve commits associated with `jepsontaylor` as author and committer; reviewer differs |
| Merge | Guarded rebase with `--match-head-commit` at the reviewed head, completed at 16:34:47 UTC as main `a75e543d2166…` with the reviewed tree `176b75bc36d7…` |
| Remote-main verification | Fetched main and compared complete Git tree; canonical clean main observed before documentation branch |
| Main guard | Run 35889731910, attempt 1, success at resulting main; guard job 107278860508 |

Two review processes overlapped during coordination. Both were independent of the source writer and both approvals are retained; they are not two different human reviewers. One process's supplementary disclosures identify outstanding work rather than silently treating it as fixed. The writer did not select another credential to approve its own source.

The full-main diagnostic run 35889732032 that the PR83 merge queued was cancelled while still queued to preserve the admitted local allocation and let main-guard run. It did not pass or execute. Earlier full-main run 35852022466 failed and remains historical diagnostic evidence requiring repair. Passing fast CI does not replace that outstanding broad qualification.

## Shared compute

Fast CI runs on xbabe1. The canonical Ubuntu machine is local xbabe2 for full CPU work. The user's SSH alias `xbabe2` points elsewhere; use the runbook's local form for this machine.

The exclusive full runner and numerical campaigns share a host lock. Drain owned listeners and verify no active full job before numerical allocation. Do not seize the host by stopping another job. Record allocation ownership and restore the runner after numerical work releases it. Keep full CPU CI separate from the one-GPU-hour training campaign and the frozen panel.

Missing cgroups or dependencies cause explicit failures/unavailable outcomes; they do not authorize an unbounded fallback. Worker control readbacks, not prose, establish the limits actually enforced.

## Preservation and lane closure

Before deletion inspect `git status`, `git stash list`, `git log --all --not --remotes`, and local versus remote tags. A branch-only log misses commits elsewhere.

Preserve unique source/evidence for the tree being removed, restore independently into the reusable bare verification directory without alternates, and verify the exact commit, tree and connectivity. A valid bundle checksum does not prove the desired detached HEAD was included.

After preservation and a fresh consumer check, remove a closed lane with `git worktree remove`. Indefinite redundant copies are not an alternative to the required reclamation step. Unknown contents remain protected. Follow `deploy/maintenance/README.md`; its reporting timer proposes candidates and never authorizes deletion.

## Rebuilding and updating this manual

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
pdftotext -layout docs/asm-maintainer-manual.pdf /path/to/durable/review/manual.txt
```

The two figure scripts above do not regenerate the training and target figures. A complete rebuild also runs the four scripts of [§](#regenerating-the-training-and-target-figures) before `build_manual.py`, because the renderer does not check that a figure is newer than its data. `make_figures.py` still defines a fixed host path to the ASM custody root (`tools/make_figures.py:39-41`), but never uses it.

Rendering uses Python Markdown, BeautifulSoup and WeasyPrint with system DejaVu fonts; their observed versions and font hashes are retained in the render receipt. Plotting uses Matplotlib and NumPy. The renderer rejects missing/nonlocal images and failed resource fetches, and checks that inputs did not change during rendering. It does not run models, fit predictors, fetch datasets or open sealed labels. Inspect the resulting pages and extracted text before treating a successful render as a complete document.

Byte-identical PDF rebuilding is not promised. `SOURCE_DATE_EPOCH` fixes font-subset timestamps, but the observed WeasyPrint 68.1 tagged-table identifiers contain process-specific values. Preserved first-edition repeat renders had identical normalized logical objects and decoded non-container streams after accounting for those identifiers, while their PDF hashes differed. Keep each output with its own render receipt and verify input hashes, content and layout; do not treat a changed PDF hash alone as changed measurements. This observation concerns the renderer, not numerical predictor reproducibility.

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

### Updating the book

Refresh inventories and receipts, then section Markdown, figures, checklist and PDF. Keep a rendering input manifest. A checkpoint does not update performance figures by implication; merged training source does not prove actual learning. Historical Elo remains historical after regeneration.

A manual change lands through the same reviewed lifecycle as source ([§](#freeze-review-and-merge)): fresh exact-head CI and an independent non-contributor native review of the final head. Every committed file, including the rendered PDF, must stay within the repository's 5 MiB (5,242,880-byte) file cap (`deploy/ci/check.sh:46`), and because the rebase merge lands every commit of the branch, the cap applies to intermediate commits as well. The earlier publication, PR84 (published head `e06c0551`), is being superseded for that reason: the independent non-contributor review 5295763455, a COMMENTED review at `e06c0551`, found two files above the cap in three ancestor commits. Its clean-history successor is local commit `2022017a8` on `docs/asm-maintainer-manual-final`, a single commit on top of main `a75e543d2166…` that carries a 203-page PDF (`69418c48…`); as of 2026-09-23 22:00 UTC it was not yet published and no successor PR was open. This second edition is a restructure and correction of `2022017a8`, prepared in a separate lane for the manual owner to integrate; it needs its own fresh exact-head CI and independent non-contributor review.

Before delivery check local links, source identities, totals, metric directions, captions, capability status and rendered pages. Extract PDF text and inspect dense tables, examples and figures. Preserve unavailable statistics and measurements explicitly rather than filling them with zero.
