# InnovationZero publication integration

The replacement bundle SHA-256 is d437f46a90979b04980aba9e51ccb1771c20c39e346619c9cc3af72d87a0f716.
Its three original commits end at c361aa27a4dae7534afae788eb0c62befdb5cc7f and remain ancestors;
the supplied PDFs are preserved separately. The combined manuscript retains PR411 objective
conditioning, implementation/baseline distinctions, six exploratory tables and references. Old
cross-references now target the current definition and Supplement S4. Nonsignificance is not
described as equivalence. No archived artifact, frontier plot input or numerical recipe changed.

The hosted publication job had no serving runner. Reusable paper.yml now runs from the existing
admitted ci.yml and trusted-main full-main.yml callers. Existing lane admission, image, runner
credentials, protection, numerical tests and test partitions are unchanged. The new job runs all
source checks, all13 refusal cases and all3 PDF builds. PR/manual aggregate requires its result,
current source/run/attempt identity, clean checkout, exact command, expected PDF/log/tool files
and every retained file hash. CPU aggregate also requires the publication job.

Security review: the optional explicit Ubuntu dependency lock preserves the existing Chromium
default and official HTTPS mirror allowlist, package SHA-256/size checks and extracted-tree
inventory validation. All157 publication packages were selected from verified Ubuntu noble
signed InRelease/Packages indices. Only dpkg-deb extraction runs, with no apt installation,
maintainer scripts or sudo. Generated English XeLaTeX configuration and formats live outside
the immutable extracted tree in task-local target state. Pinned font archives are verified before
each extraction; cached font bytes are refreshed from those archives. No production credentials,
corpus, training worker or private release key enter this job.

Local checks: original9 integrity tests plus an explicit-lock/mirror/schema/architecture control
pass;24 owning records checks, actionlint, shellcheck, static test-partition proof, paper-trace,
source checks and13 refusal cases pass. The initial integrated build produced45/12/235 pages,
all fonts embedded and no unresolved references, overfull boxes or missing glyphs. This first
build was from a dirty private integration and is local evidence only. The new tool bootstrap
also ran in the pinned093036ba runner image with a read-only root and no network. Final exact-head
CI, source-bound receipts and visual PDF review are separate requirements before merge.

Evidence custody: /home/ubuntu/jope-evidence/pr-release-drive-20261022T2319Z. This is the
implementing agent's review, not an independent numerical or human peer review. Track C and
authoring-telemetry limitations remain explicit in the manuscript.


The first complete isolated build refused missing `bibtex`: Ubuntu registers its public name
through update-alternatives, absent in extraction-only operation. The bootstrap now binds that
name to the locked `bibtex.original` in separate task-local PATH state. No maintainer script runs.
Initial hygiene also refused the two newly discovered reusable caller jobs; both are explicitly
assigned to the existing paper lane, and the README coverage count is derived from that map.
Both reproducing failures are retained; complete corrected container build and hygiene remain
separate acceptance commands.

The corrected container command first reused existing PDFs (latexmk reported no work); that is a cache/readback check, not a fresh image compilation. Preserve those PDFs/logs, clean only the publication build products through its owning target, and rebuild all three volumes from source in the pinned read-only image before calling the image build qualified.


The fresh read-only093036ba image build completed at da532e2f30afaf52e9e3d91ced04b916849e19d5,
with no network and2CPU/8GiB. Each volume compiled from source after preserving previous products
and invoking the owning clean target:45/12/235 pages, all PDF checks passed. All3 output hashes
match the visually inspected previews exactly. The corrected full local hygiene command also
passed368/368 unit checks,99 Lambda Rust cases and42 bump checks; its unchanged9912 untracked
legacy target-* entries are explicitly disclosed, not treated as clean hosted source.

PR413 merged as01850a92892f85a384a30db1dc3e4b4beb98d3d2 with reviewed-tree equality and a passing
main-guard. Its integration into this publication branch was conflict-free and changed no paper
source, archive or plot input. Post-integration source/13refusal checks, coverage validation,
paper/README numbers, actionlint and shellcheck pass. Fresh final-head PR/manual CI is still
required; the earlier private local proofs do not substitute for it.


The first native PR publication job120181058823 refused before creating a receipt because explicit
head checkout differed from the synthetic merge GITHUB_SHA. The shared receipt guard remains
unchanged. Publication now uses the native event checkout and independently requires its complete
tree to equal the reviewed source head. Both SHA identities, equal trees, run/attempt/event are
retained and verified by aggregate. Manual CI still uses the actual source commit. The failure and
failed evidence publication are retained; no PDF/reference/font check was relaxed.
