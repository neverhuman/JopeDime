# JopeDime current direction

**2026-09-28 wrap-up train and v0.2.0.** Eight pull requests, #115 to #122, landed on main between
2026-09-27T09:56:32Z and 2026-09-28T19:29:10Z. Clean-base part B (#123) followed at
2026-09-28T21:39:07Z. This pull request, which carries the v0.2.0 release tooling, sits on #123's
merge and closes the wrap-up train the owner approved at 2026-09-27T22:20Z: release first, then
the live three-host S3 demonstration as the next milestone.

The same rules held for all eight, and for #123 (whose one failed CI attempt was at an earlier
head, below):
- **Merge.** Each one merged by GitHub rebase at the exact reviewed head, and each merged tree is
  byte-identical to the reviewed head's tree. `jepsontaylor` performed every merge.
- **Review.** Every approval and every changes-requested review came from `neverhumanbot`
  (219957145). Every commit that landed on main has Jepson Taylor
  `<130782313+jepsontaylor@users.noreply.github.com>` as both author and committer. Claude
  appears only as a Co-Authored-By trailer.
- **CI.** A `pull_request` run and a `workflow_dispatch` run of `ci.yml` passed on attempt 1 at
  each final head. A full run is nine jobs: `changes`, `hygiene`, `rust-check`, `rust-test`,
  `emitters`, `inputs`, `python`, `e2e-smoke` and `aggregate`. Lanes skipped by the path matrix
  are named below.
- **Issues.** None of the eight closes an issue.
- **Custody.** `…/claude-w0-w12/` stands for
  `/home/ubuntu/jope-rescue-20260921/regression-remediation-20260926/claude-w0-w12/`.

Merge order, main SHA after each merge, and final main:
`cee1fc8bd242688972c352133847793c6927279f` (#115) →
`f46a9048fbb2697d6d1f4463a688083ea9dfcce1` (#116) →
`dd49b23bc94678106d33544e138f22e8979e8344` (#118) →
`7aec174e055af6ffe41f71446876e0b0e6f632ce` (#117) →
`4ff4e7b697dcd5b7097dacbae9e99c07ffef8e28` (#119) →
`01ef9ddcb31133bea27188e29af9978c2d36b8c4` (#121) →
`4d9a27e64d31c57b6573a84e35f22c69ebb2b1b6` (#120) →
`a8e4ce8f1f7cdf2f518e867147c4f0b038d966c5` (#122) →
`0737da51102ad502676c1b1c3c06549a3ad0442e` (#123) → this pull request's merge commit.

A rebase merge writes new commits, so no commit of this pull request can name that merge commit.
It is `origin/main` immediately after the merge, and the v0.2.0 release readback under
`~/jope-evidence/release-v0.2.0/` records its SHA.

- **[#115](https://github.com/neverhuman/JopeDime/pull/115): the record of #114's landing.**
  It added the MASTER_PLAN record that follows this one.
  - Merge: one commit, main `cee1fc8bd242688972c352133847793c6927279f`.
  - Reviewed head: `a2e45bca35a31db33bbd4b7a24d1038be8142c1a`. The tree is
    `070bf1c6969c01963fcd8f86991e758ba16e5c10` on both sides.
  - Review: approval 5329823074.
  - PR run 36310255122: `changes`, `hygiene` and `aggregate` passed; the six other lanes were
    skipped by path.
  - Manual run 36310333178: all nine jobs passed.
  - Manual dispatches 36310263359 and 36310263463 were cancelled. They are not evidence.
  - After merge: main-guard 36310954778 passed. full-main-cpu 36310954813 failed. No run of
    that workflow succeeded until #116.
  - Custody: `…/claude-w0-w12/pr115-merge-20260927T095621Z/` and
    `…/claude-w0-w12/independent-review-pr115/`.
- **[#116](https://github.com/neverhuman/JopeDime/pull/116): the known full-main-cpu failures
  are routed to the lanes that can prove them.**
  - It carries Codex's `health/full-main-green`. The 8 host-data tests go to
    `deploy/ci/real-corpus-lane.sh`. The CPU fit-service contracts are ignored only under
    `neural`; that list grew from 74 to 80 with #114's six new contracts.
    `deploy/ci/check-test-routing.sh` guards both lists.
  - No test was deleted, and no fence was weakened.
  - Merge: four commits, main `f46a9048fbb2697d6d1f4463a688083ea9dfcce1`.
  - Reviewed head: `05a5cf84c90bca27f00c0d67dda624e5d63634fd`. The tree is
    `3cb9737c1b6f6f10bd412815c386a78359b85db0`.
  - Review: approval 5331659297.
  - CI: PR run 36340840488 and manual run 36340841630 passed all nine jobs.
  - After merge: main-guard 36343656758 passed, and full-main-cpu 36343656796 passed. That was
    the workflow's first successful run: the 53 runs created before it ended in 48 failures and
    5 cancellations.
  - Custody: `…/claude-w0-w12/pr116-merge-20260927T191447Z/` and
    `…/claude-w0-w12/independent-review-pr116-05a5cf84c/`.
- **[#118](https://github.com/neverhuman/JopeDime/pull/118): targeted single-dataset search in
  `jope pipeline`.**
  - What it adds:
    - seeded generation 0 and an incumbent;
    - deterministic threads and islands (parallel search groups over the same cuts);
    - `evolve` rounds chained in a verified `lineage.json`, with an unseeded control;
    - a sealed test split scored at most once per dataset and program in its registry (a local
      record, not access control);
    - `hwm ingest --source pipeline`;
    - an append-only champion bank whose export feeds the next `evolve --seeds`.
  - The local demonstration improved on validation folds that every round selected on. On the
    sealed test split the control's champion scored better. The PR says so.
  - Merge: 32 commits, main `dd49b23bc94678106d33544e138f22e8979e8344`.
  - Reviewed head: `9fa240906991bd10f5538c29edc66c7b99d28b29`. The tree is
    `d5ddc3358218de0711d39e98c71fdd3284816771`.
  - Review:
    - 5334007364 requested changes at `d7e89f21197f446618bd1c80e03e90890f185010`. Refused
      arguments had to leave nothing written, and the determinism claim had to be limited to
      phases that the budget, not the deadline, ends.
    - 5334625316 requested changes at `053c7862f755a17ae7434561e626e4d4c096b686`. Two
      concurrent test-split scores both recorded no earlier looks. The fix is a per-dataset lock.
    - 5334865651 approved.
  - CI: PR run 36387309556 and manual run 36387306913 passed all nine jobs.
  - After merge: main-guard 36388986311 and full-main-cpu 36388986268 passed.
  - Custody: `…/claude-w0-w12/pr118-merge-20260928T065616Z/`,
    `…/claude-w0-w12/independent-review-pr118-*/` and `…/claude-w0-w12/pr118-review-*`.
- **[#117](https://github.com/neverhuman/JopeDime/pull/117): Python removal P1.**
  - The Python that CI and the engine depend on is ported to Rust:
    - family growth, in `jope-engine`;
    - `jope-ci cargo-artifact`;
    - the four records guards, as `jope-records` verbs;
    - `jope-ci build-report`.
  - 703 files with no live consumer are archived, with a manifest and a custody tarball.
    ADR-0006 records the Rust-only stack.
  - Merge: 50 commits, main `7aec174e055af6ffe41f71446876e0b0e6f632ce`.
  - Reviewed head: `083485a76c1ab13e7ed73ac20f608e6332f6e12f`. The tree is
    `d2e989ce2d00eeb4f061896728e622369d4f94db`.
  - Review: four requests for changes, then approval. Each finding was fixed across its whole
    class, and each fix is backed by differential fixtures captured from the original tools. In
    every harness there are 0 cases where the original refuses and the port accepts.
    - 5332299199 at `a997ab82500ce9708fe88054c8147a38ae4d9bd7`: the coverage validator
      accepted malformed lanes and dispositions.
    - 5333747872 at `e8640b7b07ab7977d20b99b450bd99456e7fe4ca`: POSIX `//` roots and
      Python's 4300-digit integer limit.
    - 5334300806 at `bc22bc6ca229c6f95153e0a8b2372506a0191de1`: two cases the original
      rejects and the port accepted.
    - 5337854418 at `bcdadfc77caf34ef32fd8fe279427ee4b617c909`: inexact arithmetic on
      input-derived values.
    - 5338691773 approved.
  - CI: PR run 36421749004 and manual run 36421748661 passed all nine jobs.
  - After merge: main-guard 36424942633 and full-main-cpu 36424942246 passed.
  - Custody: `…/claude-w0-w12/pr117-merge-20260928T125447Z/`,
    `…/claude-w0-w12/independent-review-pr117-*/` and `…/claude-w0-w12/pr117-review-*`.
- **[#119](https://github.com/neverhuman/JopeDime/pull/119): clean-base part A.**
  - Changes:
    - three unreferenced leftovers and the dead `ops/ci` directory are removed;
    - `actions/checkout` is pinned to v7.0.1 in `ci.yml`, `full-main.yml` and
      `main-guard.yml`;
    - the stash-only ledger block and plan are archived under `docs/`;
    - all 33 feat-math temp-dir test callers now use `tempfile`.
  - `AGENT_CHAT.md` is unchanged.
  - Merge: 10 commits, main `4ff4e7b697dcd5b7097dacbae9e99c07ffef8e28`.
  - Reviewed head: `d582c30bfab7a14dad44cc05764bda680c76e28d`. The tree is
    `2876e389b09e849e0b1246fe80b424f6a418acd4`.
  - Review, two approvals:
    - 5338771284, submitted at 12:42:58Z. It was given at
      `01f3b6f1872eca8ca4281d6b090251090eb2eec9`, before the rebase onto #117; its body names
      that head. The API now reports its `commit_id` as the rebased head.
    - 5339999982, submitted at 14:19:47Z, after the rebased head was pushed (its PR run was
      created at 13:32:47Z). The merge relied on this one.
  - CI: PR run 36429372310 and manual run 36429371243 passed all nine jobs.
  - After merge: main-guard 36435189783 and full-main-cpu 36435189776 passed.
  - Custody: `…/claude-w0-w12/pr119-merge-20260928T141950Z/`,
    `…/claude-w0-w12/independent-review-pr119-01f3b6f18/` and
    `…/claude-w0-w12/independent-review-pr119-d582c30bf/`.
- **[#121](https://github.com/neverhuman/JopeDime/pull/121): D5, all forty synthetic
  fixtures.**
  - The fixtures were regenerated by this tree's generator, not copied. On main, the
    `datasets/synthetic` tree is `43e962350afe29fd1c93c0ba3128bd8bcfc8e58e`, which is the tree of
    Codex's `15a1d3102c7c45e79396749c8ba1fd516cfc34e9`. It holds 40 fixtures plus
    `catalog.json`, whose sha256 is
    `651d28c8456bc160ed33960110142cdb8e9941d2a727348416db999f99d22e93`.
  - The ten real `test-10` datasets are kept, and `MAX_DATASETS` stays 10.
  - `champions --local` and `train-step --local` use one verified snapshot. They search the 8
    supported regression fixtures and report the other 32 as unsupported.
  - No Python was added or edited.
  - Merge: four commits, main `01ef9ddcb31133bea27188e29af9978c2d36b8c4`.
  - Reviewed head: `e0b906b239b107701364adb41d522f7a7c1ff051`. The tree is
    `65d8c3f2ae2e59dbccfa322ebc36d42f558fe947`.
  - Review: approval 5340876669.
  - PR run 36442700858: seven jobs passed; `python` and `emitters` were skipped by path.
  - Manual run 36442700618: all nine jobs passed.
  - After merge: main-guard 36445370527 and full-main-cpu 36445370746 passed.
  - Custody: `…/claude-w0-w12/pr121-merge-20260928T154011Z/` and
    `…/claude-w0-w12/independent-review-pr121-e0b906b23/`.
- **[#120](https://github.com/neverhuman/JopeDime/pull/120): W7, the bounded multi-host
  native training loop.**
  - This lands Codex's `rr2/w7-live` through cherry-picks with `-x`: W7-T01, T03 to T07 and
    W7-D1 to D6.
  - #114's read-once API is kept, and `full-lane.sh` calls the Rust `cargo-artifact` port.
  - No live three-host run is claimed.
  - Review 5339604155 requested changes at `3d12729a2181bcbe5c153770c60095be9c4414fe`: the
    TLS handshake and framed-I/O deadlines were checked only between `complete_io` calls.
    `transport::DeadlineSocket` now applies the absolute deadline to every read and write, and
    the courtesy close is capped at 1 s.
  - The branch was updated without rewriting history. Merge commit
    `1c19341fcd5e387863e0b71aa1aae5814d63dd98` brought in main
    `01ef9ddcb31133bea27188e29af9978c2d36b8c4`, and the corrections were cherry-picked on top.
    The previously pushed head stayed an ancestor.
  - GitHub's rebase merge replayed the 25 non-merge commits of the 26 and dropped the merge
    commit. Main became `4d9a27e64d31c57b6573a84e35f22c69ebb2b1b6`.
  - Reviewed head: `58a919e214b4c46ac6614156d7bc498084cb17b3`. The tree is
    `7c544624efdb3af542a33365a54401ec974bb984` on both sides.
  - Approval: 5341334345, at 16:00:21Z, after the 15:56Z push.
  - CI: PR run 36447376345 and manual run 36447429049 passed all nine jobs.
  - Codex ran the review, CI and merge (AGENT_CHAT 2026-09-28T15:42Z).
  - After merge: main-guard 36449887715 passed. full-main-cpu 36449887665 **failed**. The failure
    reproduced in #122 is 11 tests: 6 in `jope-parallel::bounded_census` and 5 in
    `bounded_smartcluster`. They read the runner container's empty `/etc/machine-id`, and PR
    CI's `rust-test` lane does not run those suites.
  - Custody:
    - `/home/ubuntu/jope-rescue-20260921/pr120-codex-resume-20260928/`;
    - `/mnt/jope-archive/jopedime-pr120-custody-20260928/`, the honesty lane archive;
    - `…/claude-w0-w12/independent-review-pr120-3d12729a2/`;
    - Claude's evidence, in `…/claude-w0-w12/runs/w7-*`.
- **[#122](https://github.com/neverhuman/JopeDime/pull/122): W7's host-identity tests become
  hermetic.**
  - `collect::HostSources` has no `Default`. The product binary hard-wires
    `HostSources::system()`, which reads `/etc/machine-id` with the same strict refusal.
  - The tests go through the example `jope-native-loop-fixture` and checked-in fixture host
    files. No test was weakened or skipped.
  - Test support gained three things: a stale-fixture guard (mtime and dep-info only), coordinator
    diagnostics, and a port reservation outside the ephemeral range.
  - Authorship: Claude wrote eight commits, through `4af8afe7547f7fbdb9b0881742b96778965e04cc`,
    and handed off. Codex added the ninth, which only corrects the freshness guard's scope wording
    in comments and docs. That commit is main `a8e4ce8f1f7cdf2f518e867147c4f0b038d966c5` itself.
  - Merge: nine commits, main `a8e4ce8f1f7cdf2f518e867147c4f0b038d966c5`.
  - Reviewed head: `00541d319493c772c053821ae6b3f8b0462c5423`. The tree is
    `37ce3d5cbfe638bc22bfd6a3eeb0fd1a71ec328c`.
  - Review: approval 5343417155.
  - CI: PR run 36469868551 and manual run 36469978500 passed all nine jobs.
  - Codex ran the review, CI and merge.
  - After merge: main-guard 36472565789 passed, and full-main-cpu 36472565658 passed on attempt 1
    (completed 2026-09-28T20:11:15Z), so the #120 failure is fixed on main.
  - Custody:
    - `/home/ubuntu/jope-rescue-20260921/fix-census-codex-lifecycle-20260928/`;
    - `/mnt/jope-archive/jopedime-pr122-custody-20260928/`;
    - Claude's evidence, in `…/claude-w0-w12/runs/fix-machine-id-*`.
- **[#123](https://github.com/neverhuman/JopeDime/pull/123): clean-base part B.** It completes
  the clean-base plan (D-A) on a main that holds #117 and #119 to #122. Main has 536 fewer
  tracked files: 2,880 at `a8e4ce8f1f7cdf2f518e867147c4f0b038d966c5`, 2,344 at the merge (557
  deleted, 21 added).
  - Changes:
    - it retires `deploy/kernel-v1` (304 files, among them second copies of `jope-parallel`,
      `feat-math`, `feat-cli` and `feat-core`) and `docs/legacy` (250 files), after custody in
      `/home/ubuntu/jope-evidence/clean-base-20260927/`: a tarball, a bundle restored
      independently with `git clone --no-local --bare` by two verifiers, `SHA256SUMS` and
      `VERIFICATION.txt`;
    - `docs/archive/clean-base-manifest.json` is generated and gives each of its 564 entries a
      `git checkout <sha> -- <path>` restore;
    - `repository = "https://github.com/neverhuman/JopeDime"` in `[workspace.package]`, and the
      `jope` binary with its direct crates as the workspace `default-members`;
    - `actions/checkout` is pinned to v7.0.1 in `weekly.yml`;
    - the `jope-records repo-layout` guard joins the hygiene lane, with a shrink-only allowlist;
    - `check-test-routing.sh` checks both directions.
  - Review 5344298377 requested changes at `fc39cfb59118d8622eccbf58fea45d3c3c249254`: the
    duplicate-package rule read `[package]` names with a regex, so a copy whose name was a
    single-quoted TOML string passed at exit 0, and manifests the regex could not read were
    skipped silently. The fix, two commits added without a rewrite
    (`7982c479fd673084ed1490b3edcd4d72f315544e` and `dd4675c097901ad323323adc7361ef67a6b0ab0e`),
    reads every tracked Cargo manifest as TOML 1.0, adds the rule `bad-manifest` for a manifest
    the guard cannot read, and reports every tracked path that is not UTF-8 (`bad-path`) instead
    of skipping it. Neither rule takes an exception. It added 7 jope-records tests (3,789 to
    3,796 in the inventory).
  - Merge: 12 commits, main `0737da51102ad502676c1b1c3c06549a3ad0442e`, at
    2026-09-28T21:39:07Z.
  - Reviewed head: `dd4675c097901ad323323adc7361ef67a6b0ab0e`. The tree is
    `67677e988df377cfd157219b047875b1d8d508bc` on both sides.
  - Review: 5344298377 requested changes at `fc39cfb59118d8622eccbf58fea45d3c3c249254`, as
    above. 5344815205 approved at `dd4675c097901ad323323adc7361ef67a6b0ab0e`.
  - CI at `fc39cfb59118d8622eccbf58fea45d3c3c249254`:
    - PR run 36479466799 failed on attempt 1. `hygiene` and `changes` stopped at the
      runner-provenance step on the xbabe1 runners, because the job's cgroup limits did not equal
      the approved CPU budget, and `aggregate` failed with them. The PR attributes this to a host
      `systemctl daemon-reload` at 20:03Z that reset the idle containers' swap limit. Attempt 2
      passed all nine jobs.
    - Manual run 36479467996 passed all nine jobs.
  - CI at `dd4675c097901ad323323adc7361ef67a6b0ab0e`: PR run 36485227333 and manual run
    36485226033 passed all nine jobs on attempt 1.
  - After merge: main-guard 36487524917 passed, and full-main-cpu 36487524712 passed on attempt 1
    (completed 2026-09-28T22:24:59Z).
  - Custody: `…/claude-w0-w12/pr123-merge-20260928T213854Z/`,
    `…/claude-w0-w12/independent-review-pr123-fc39cfb59/` and
    `…/claude-w0-w12/independent-review-pr123-dd4675c09/`.
- **This pull request: the v0.2.0 release tooling, CHANGELOG, README and this record.** Rebased
  on #123's merge, `0737da51102ad502676c1b1c3c06549a3ad0442e`.
  - Changes:
    - the lock-only dependency bump that was PR #5's intent, on current main: `cc` 1.4.5 to
      1.5.1, `clap` 4.6.6 to 4.6.7 and `crc32fast` 1.5.1 to 1.5.2;
    - `deploy/release/release.sh` (bash and jq) and its refusal tests,
      `deploy/release/test-release.sh`, which runs in the hygiene lane (owner decision D-D);
    - `docs/release/README.md` and the public key `docs/release/signing.pub`, sha256
      `de84945ec5326877d202d0738e655492a8b2c3b794873a2eb248939c4d326e4e`;
    - `release.yml` made dispatch-only; it fails closed at its first step;
    - version 0.2.0 for the workspace and every product crate;
    - `CHANGELOG.md` for 0.2.0, generated with git-cliff and curated by pull request;
    - the README's 0.2.0 status, stats, releases section and layout, and the lines that #117 to
      #123 and the release made false in `SECURITY.md`, `CONTRIBUTING.md` and `docs/`;
    - this record.
  - Merge: this pull request's merge commit, `origin/main` immediately after the merge; its SHA
    is recorded in the release readback, not here. No tag exists yet: `release.sh --publish`
    creates `v0.2.0` on that commit. The Python package stays 0.1.0 and unreleased, and no wheel
    ships.

Deferred to the next milestone. These are recorded, not dropped:
1. **The live three-host S3 training demonstration.**
   - First, P6-0 search throughput. #118's run measured about 3 of 8 cores busy; the work is
     parallel folds and steady-state generations.
   - Then the P6-1 to P6-7 host setup.
   - Then windows A to E: 80% CPU search islands on three hosts, 3-GPU training, and S3 progress
     and restore.
   - The live W7 three-host run follows its runbooks under its own admission window.
2. **A self-hosted `release.yml`:** a release lane, the Rust admission hook (ADR-0006 U4) and the
   `jope-ci release` commands. Until then, releases come from `deploy/release/release.sh`.
3. **Python P2 and P3:** the remaining CI tools move to Rust, and then the torch lineage, the
   `python` lane and `pyproject` are retired.
4. **The remaining W-series gaps and open issues:**
   - #48, the C++/JS emit binding;
   - #62, live S3 from the command line. Its last status (2026-09-27) lists the live champion
     pull/push route, the unused `s3` feature of `jope-arena`, a receipted live `jope.hwm/v1`
     round-trip through the `jope-learning` mirror and sync, and where `JOPE_DATASETS_CACHE`
     actually puts fetched files;
   - #47, the gitleaks history scan, whose findings depend on the clone's refs. #114 made the
     hygiene scan cover `HEAD` and all of its history; still open are an audit of the 243
     findings on custody refs that are not ancestors of the source head (225 vendored
     `generic-api-key`, 18 `jope-env-files`) and a scheduled scan of every retained ref;
   - W7 cross-host exchange and continuation;
   - the other open W1–W12 items.
5. **Two W7 test-support items:**
   - the transient `bounded_smartcluster` failure (post2) is not root-caused;
   - the port-reservation window left open by #122 needs a production change.
6. **Smaller items:**
   - the jankurai measurement and its caps;
   - the WP5 cli-split;
   - the 35 `std::env::temp_dir` calls in the test code of 25 files outside feat-math, held on
     the repo-layout R5 allowlist (3 of them in the pinned fit-authority source, which stays as
     it is);
   - the README's generated arena block cites
     `python/src/jope/research/index_combined_regression_evolution.py:183` and
     `docs/learning.md:46` cites `promotion_gate.py:22-27`, both archived by #117. The arena text
     comes from the sha256-pinned `assets/regression/protocol-corrections-tabular-regression-v2.json`
     through the `jope-records readme-benchmarks` generator, so the fix goes through it (PY-P2);
     the generated block is not edited by hand;
   - owner: `docs/runbooks/rotate-credentials.md` gives the OIDC trust policies' `sub` claims
     as `repo:veox-ai/JopeDime:…`, with a `refs/tags/v*` binding for `release.yml`, which no
     longer requests `id-token: write`. Re-check the policies' repository and ref bindings
     against `neverhuman/JopeDime`; they cannot be read from here;
   - untracking `AGENT_CHAT.md`, which waits for the owner and Codex.

Next allowed action:
1. Cut v0.2.0 on xbabe2, from a clean registered worktree at this pull request's merge commit
   (`origin/main` immediately after the merge, after a fresh fetch): run
   `deploy/release/release.sh v0.2.0 "$(git rev-parse origin/main)"` (the dry run), inspect its
   evidence, then run the same command with `--publish`. `release.sh` refuses unless `HEAD` and
   a freshly fetched `origin/main` both equal that SHA. Read back the receipts under
   `~/jope-evidence/release-v0.2.0/`: `readback.json` must show cosign `Verified OK` against the
   committed key, `sha256sum -c` passing, the manifest `git_sha` equal to
   `git rev-parse v0.2.0^{commit}`, the binary printing `jope 0.2.0`, glibc at most 2.36 and a
   failing tampered copy. Record them with the main-guard and full-main-cpu runs on that commit.
2. Close the `bench` and `demo` lanes under custody: bundle, restore independently, then
   `git worktree remove`. Delete the landed branches. Confirm that `gh pr list --state open` is
   empty and that `git log --all --not --remotes` shows only custody refs.
3. Update the records of issues #47, #48 and #62, which stay open.
4. Then start the next milestone, with P6-0 throughput as its own reviewed PR. The live three-host
   S3 and GPU windows need their own admission. This record does not start them.

**2026-09-27 W1–W12 source gap closure.** [PR #114](https://github.com/neverhuman/JopeDime/pull/114)
landed the remaining source- and fixture-scope W1–W12 work on top of #111 and #113:
- W1 input boundaries;
- W2 semantic manifest, typed context and authenticated bytes;
- W3 revocation consumers, the authenticated W2 handoff with its producer side, requalification
  refusal, fitted-artifact binding and HWM recovery;
- W4 per-draw outcomes;
- W5 frozen inputs and evaluator process groups;
- W8 publication, reconcile and S3 preflight;
- W9 package seal and install;
- W12 claim inventory, conformance pack and duplicate-cell refusal;
- the never-loop Hyperion fix;
- the owner's conservative policy defaults, `OWNER-DECISIONS.md` D1–D20.

It merged by rebase, 45 commits, at 2026-09-27T09:39:08Z as main
`1b301a9c52b8ef409a85e76d878a0bbcb6bb0f40`. The merged tree is byte-identical to the reviewed head
`9dc747155a34f1f5194ded2a5abb9c11899d06a4`.

Review was by independent Codex processes as `neverhumanbot` (219957145); every commit author and
committer is `jepsontaylor`:
- Changes were requested at `2160a28c9`: context substitution, a later context window, S3 URL
  normalization, and a commit whose email mapped to the reviewer, re-authored with its tree
  unchanged.
- Changes were requested at `09b9676b3`: HTTPS S3 routes.
- All findings were reproduced first and then fixed.
- Approvals followed at `94b1616fb`, `4d475b320`, `31383aa48` and `93bfcb956`, and finally review
  5329763785 at `9dc747155`.

PR run 36309043047 and manual run 36309054082 passed all eight jobs and aggregate on attempt 1 at
that head. It closes #57 and #58. #47, #48 and #62 stay open with the reasons in
`docs/internal/core-ip/findings/pr114-issue-followups-20260927.md`.

Earlier exact-head attempts failed only on GitHub network steps, for two causes:
- Runner containers resolved through the LAN router, which failed 18 of 90 lookups from xbabe1.
  This is fixed on the xbabe1 and xbabe2 hosts: resolved now serves `docker0`, and receipts are in
  `~/jope-evidence/runner-dns-fix-20260927/`.
- Intermittent SYN loss on the LAN/NAT path. The final commit gives `changes`, `hygiene` and
  `aggregate` room for one stalled connection and retries the aggregate download at most three
  times. No gate or pass criterion changed.

Task status: 92 tasks, of which 70 are implemented, 67 fixture-proven, 35 reproduced and 22 blocked.
None is CI-verified or live-qualified, and all 195 findings remain open. The blocked tasks are the
live gates W7-T07/T08/T09, W8-T07/T08, W9-T07, W10 (except T09), W11-T02/T05–T08 and W12-T06/T08.

Custody:
- `/home/ubuntu/jope-rescue-20260921/regression-remediation-20260926/claude-w0-w12/pr114-merge-20260927T093857Z/`
- `independent-review-pr114*/`
- `runs/`

Owner decisions of 2026-09-27:
- **D-A:** Codex lands the queued source branches.
- **D-B:** the Python policy-v3 training lineage is retired in favour of the native trainer.
- **D-C:** the following infrastructure actions are authorized, each logged with receipts and
  reversible:
  - the scoped S3 role `JopeDimeW8Publisher`, which now exists and is verified;
  - root host setup on xbabe1–3;
  - retiring xbabe2's out-of-repo qualification daemons, which is done;
  - pausing competing workloads in admitted windows.
- **D-D:** the next release is v0.2.0, from the self-hosted runner with a repo-held signature.

Next allowed action: land the queued branches in order, each rebased on `1b301a9c5` with records
regenerated:
1. `health/full-main-green`;
2. `rr2/w7-live`;
3. `rr2/python-p1`.

Under D-A the owner changed the PR-first order in writing: Claude proceeds in parallel, on paths
disjoint from those branches, with the following work, each as its own reviewed PR:
- clean the base;
- run and fix jankurai;
- finish removing Python;
- close the remaining source gaps;
- the targeted GP evolution demo;
- the live three-host S3 training demonstration;
- the v0.2.0 release.

This record does not start any of them.

**2026-09-26 core-IP regression remediation.** [PR #111](https://github.com/neverhuman/JopeDime/pull/111)
delivered W0 and the recorded source- and fixture-scope slices across W1–W12 of
`docs/internal/core-ip/work-orders`. Source integration items and live gates remain open;
`STATUS.md` records them per task.
It merged as squash `7b86622ea694bc069338c268f23e17c29f76e8eb` at 2026-09-26T15:12:59Z. The tree is
byte-identical to the reviewed head `bddb8fed16e4c73cb0277998f37ab4a8871e6a3e`, and main-guard
36251156798 passed.

It was squashed, not rebased, because GitHub refuses rebase merges of more than 100 commits (121
here, `canBeRebased=false`). The squash body lists every commit. The remote head branch was
deleted on merge (`delete_branch_on_merge`), but the full history stays reachable at GitHub's
`refs/pull/111/head` (= `bddb8fed1`) and in the local branch `feat/regression-remediation`.

Review was by independent Codex processes acting as `neverhumanbot` (219957145), against `jepsontaylor` as author
and committer of all 121 commits:
- Review 5326209256 requested changes at `0b303b293` with two P1 defects: grant expiry bypassed
  during coordinator recovery, and an outer-score grant that re-read its preparation after
  verifying it.
- Both were reproduced and fixed, together with sibling gaps, in `4ac185bb7`, `5ad932450` and
  `bddb8fed1`.
- Review 5326304313 approved `bddb8fed1`.

PR run 36250125057 and manual run 36250695256 passed all eight jobs and aggregate at that head.

Task status is 70 implemented, 66 fixture-proven, 22 blocked and 0 planned. None is CI-verified or
live-qualified, and all 195 findings remain open. The blocked tasks are live or dependent release
gates, for example three-host W7, live S3 W8, W9-T07, loop/transfer W10, GPU timing W11 and
W12-T06/T08. Per-task state and open items are in
`docs/internal/core-ip/work-orders/STATUS.md`.

The PR added no Python and deleted seven files. The core-IP verifiers now live in the Rust crate
`jope-records`. The owner's repo-wide Python migration go-ahead is pending (owner decision 10 on the
PR).

Custody for the landing:
`/home/ubuntu/jope-rescue-20260921/regression-remediation-20260926/claude-w0-w12/`, specifically
`pr111-lifecycle/`, `independent-review-pr111*/` and `runs/`.

Next allowed action: close the lanes that served this PR (`bench`, `demo`, `honesty`) under the
custody rules. Then take the recorded owner decisions and the live-gate prerequisites; this record
does not start them.

**2026-09-22 CPU foundation delivery.** Publication repo: private `veox-ai/JopeDime`.
Issue #52 is closed by [PR #55](https://github.com/veox-ai/JopeDime/pull/55).
Its source head `c5a5b1138b4e878327eb2f429ac75be9b8a9374c` received independent
Claude review and native `neverhuman` approval after verifying `jepsontaylor` was
both PR author and every GitHub-associated commit author. All eight manual jobs
passed in [run 35756345708](https://github.com/veox-ai/JopeDime/actions/runs/35756345708).
Rust executed 751 passing tests with 13 nextest skips; Python passed 207 with 15
deselected. JUnit omits those Rust skips. These are limited gate results, not full
CPU qualification. PR #54 was closed as superseded; its discussion is retained.
PR synthetic merge refs and source-head dispatch evidence are recorded separately.

The guarded rebase merge produced remote main
`8eaafa3915c937c49581191fd00680c9d3f01cc0`; main-guard
[35757914440](https://github.com/veox-ai/JopeDime/actions/runs/35757914440) passed.
Canonical returned clean to that main before claiming `fix/cargo-artifacts-46`.
Lifecycle evidence is outside the checkout under
`/home/ubuntu/jope-rescue-20260921/implementation-20260922/foundation/issue52`.
Writer access is recovered. Protection API still returns 403; enforcement is procedural.

Issue #46 is closed by [PR #56](https://github.com/veox-ai/JopeDime/pull/56). Local proof at
`96ec9a996` passes both the canonical reused `target/` and the registered
`artifact-proof` lane's stable external target directory with spaces. Each layout
passed nine artifact protocol tests, twelve root tests plus one doctest, emitter
parity and pipeline smoke, and a cached Cargo build reported `fresh: true`.
The old external-root test failed at main before the change. Logs, input hashes,
commands, statuses, environment and 16 GiB/16 CPU budgets are retained under
`foundation/issue46/{normal,external}` in the same custody tree. An initial scratch
fixture placed under the repository marker failed its no-marker precondition;
that failure is preserved separately. Final source `161ddeb58b2fcf1768651f8dc5b7aff6ebdc61ac`
passed all eight manual jobs in run35760152193 and all PR jobs35760158987. Independent
Claude/native neverhuman approval5281400221 binds that head. Main is now
`c7ae554a85f0e2d35d282aa87ee72a539d6cb4c2`; main-guard35761248043 passed.
Nextest normal/external proofs passed at the final head after correcting its
workspace-relative default report store and rejecting stale reports. The proof lane
and external build were archived, independently restored/verified, then removed
with fresh dependency checks. See `foundation/issue46/lifecycle-receipt.json` and
`proof-closure.json`. Canonical returned clean to verified main before this phase.
GPU/S3 regression and release/container wrappers have static validation only here;
their required capabilities are not claimed as exercised by CPU smoke proofs.

Coverage foundation [PR #59](https://github.com/veox-ai/JopeDime/pull/59) merged from
`a13f8cee4fa10cbde4b56305a2e6d842f089d749` after all eight manual35763473661 and
PR35763657978 jobs passed and independent Claude/native neverhuman approval5281705335.
Main `aa1e5986824b41632fce482bb06a64c02a21d65d`, main-guard35764421202 passed;
canonical returned clean before the next claim. Custody: `foundation/coverage/lifecycle-receipt.json`.
The review corrected a feature-gated target wrongly assigned to default commands.

Aggregation [PR #60](https://github.com/veox-ai/JopeDime/pull/60) merged from
`f08bcb6486fff1c6472a5a07224d110444853b86` after both nine-job manual35767660672 and
PR35767666966 runs passed, including aggregate, and independent Claude/native
neverhuman approval5282087456. Main is `8ea1289fec019caa21facda4dea26bfb328bbbe8`;
main-guard passed (exact receipt: `foundation/aggregation/lifecycle-receipt.json`).
Canonical returned clean before claiming `ci/full-main-cpu`. The first aggregate
correctly stayed red when nested product receipts were misclassified; that defect
was fixed and the real failed-run artifacts replayed before final hosted acceptance.

[PR #61](https://github.com/veox-ai/JopeDime/pull/61) delivered the CPU foundation from
exact source `ac26f6596e24a2af6e627b87e87d9bd18a90514e`, independently reviewed by
Claude with native neverhuman approval 5282585692. Manual 35773238753/1 and
PR 35772555348/2 passed all nine jobs. Guarded rebase merge produced
`fdb2e8e3e3d152590d9246b5226eca417f5c6cbc` with the identical reviewed tree.
The preliminary genuine docs-only run 35769025859 passed changes, hygiene and
aggregate with six declared path exclusions. Failed pre-deployment provenance runs
and the first aggregate failure remain evidence, not superseded success claims.

The reviewed runner source was deployed after draining owned listeners. Actual readback
verified two fast slots on xbabe1 at 16 GiB/16 CPUs each and one exclusive full slot on
xbabe2 at 48 GiB/16 CPUs, no swap, separate PR/trusted-main work and cache volumes,
read-only root/policy/hooks, no GPU/Docker/PAT/models mounts or training/publishing/S3
credentials. Other JopeDime roles are disabled; the held 24 GiB/28 CPU proposal was not
used. Provenance remains mandatory. Full jobs have 120 minutes and at least 50 GB disk
reserve. Main-guard shares the full slot serially, so it can queue behind the baseline.
See the [runner runbook](docs/runbooks/runner.md) and custody `foundation/full-main`.

Current static discovery at that source assigns 22 packages, 18 features, 214 Cargo targets,
218 Python test-named files, 6 workflows/16 jobs and 32 benchmark sources. These are discovery
counts, not tests executed. Fast CI executed 54 protocol tests, 751 nextest passes with 13 skips
from 764 listed, 207 Python passes with 15 deselected from 222, four systemd tests and four
emitter tests with 278 filtered. Counts overlap across invocations. Non-test commands
have unknown/not-applicable counts, never invented zeros. Collection, execution and
explicit deferred/research dispositions remain distinct in
[coverage/receipt semantics](deploy/ci/coverage.md).

The first push-triggered full-main run,
[35775436225](https://github.com/veox-ai/JopeDime/actions/runs/35775436225), attempt 1,
completed at `fdb2e8e3e3d152590d9246b5226eca417f5c6cbc`. All11families produced fresh,
verified receipts and a retained inventory: eight passed, two failed, one unavailable.
The overall job correctly failed. Downloaded artifacts replayed through the verifier
with matching evidence digests; no missing/invalid family results were accepted.

| Family | Observed outcome at that main |
| --- | --- |
| Default workspace | 2,523 passed, 29 failed, 37 skipped; 2,589 listed; exit 100 |
| Maintained Python | 207 passed, 15 deselected; 222 collected |
| Doctests | 8 passed |
| Scratch feature variants | 1,879 passed, 30 failed, 16 skipped; 1,925 listed; ten-minute family limit, exit 124 |
| Policy/testing | 23 passed, 5 skipped; 28 listed |
| S3 feature, offline only | 34 passed; no live S3 credentials or qualification |
| CPU-neural | Unavailable, exit 78: configured `/usr/include/cuda_runtime.h` missing; counts unknown |
| Emitters | 4 passed, 278 filtered plus emitted-code parity |
| Input verification and pipeline smoke | Both passed; non-test counts not applicable |
| Build artifact measurement | Passed; initial artifact `fresh:false`; subsequent cached reuse is a post-merge acceptance check |

The scratch timeout interrupted `feat-math::golden fast_profile_portfolio_head` after
466.505 seconds of that test; compilation and the rest of the family used the remaining
budget. Counts are per invocation and overlap. Ignored/deselected tests, research scripts,
GPU and live-service dependencies remain explicit non-execution in the mapping/inventory.
Main-guard35775436296 passed at that exact main. Negative admission run35775538345
proved rejection in both domains: actual Worker hook invocation followed by exit137
in0.467seconds (fast) and0.674seconds (main), no bypass-step execution, and no OOM,
service stop or timeout explaining the exits. Installed hashes/policies and deterministic
authorization replay bind the inference; denial stderr itself was lost with the Workers.
The expected-rejection run was cancelled after preserving both proofs; it is not passing CI.
The workflow runs on every main push and manual dispatch of main, and remains diagnostic.

Proposed next repair PRs, derived from the 29 default failures and the other family outcomes:

- **Repair campaign spec and fixture discovery**: four spec-drift failures and nine missing
  or host-dependent fixture/module/catalog failures. Preserve dataset identities and sealed
  input boundaries; do not copy research data into the learning worker's input closure.
- **Repair campaign status and UI assertions**: three failures in island counts, selfplay
  alarms and requeued-attempt history.
- **Repair engine cache, history and queue contracts**: five failures in native cache
  accounting/invalidation, displayed learning metrics and unsupported-task queue handling.
- **Make fit-authority test setup explicit**: eight failing engine fit/view contracts;
  preserve the production fit fence and require valid test authority rather than bypass it.
- **Bound the scratch-feature portfolio proof**: investigate the measured timeout and
  preserve the approved 120 minute full-job/resource limits when revising family budgets.
- **Repair CPU-neural SDK discovery**: reconcile the declared CPU feature's header/runtime
  prerequisites; an unavailable SDK is not a passing test result or GPU qualification.

These are measured failure groupings, not yet established root causes. The held lint patch
belongs to the later lint repair PR; reproduce its warning census before strict enforcement.
Custody includes `push-35775436225-verified.json`, `default-repair-observations.json`,
exact source reviews/CI/deployment fingerprints, and the original failed logs.

Full Rust/Python repairs, strict Clippy enforcement, persistent managed runners and
mandatory full-CPU gates remain the following phase. GPU/S3 qualification and learning
campaigns remain outstanding. Qualification and soak clocks are unstarted. The approval
procedure and verified credential paths are now explicit in
[AGENTS.md](AGENTS.md#approval-procedure-on-this-host).

Post-merge completion is recorded outside the checkout at
`/home/ubuntu/jope-rescue-20260921/implementation-20260922/foundation/completion.json`,
including the final main SHA, documentation review/CI, a subsequent job's fresh cache-reuse
receipt, and the clean canonical handoff. This source record does not preclaim those outcomes.

Stopped-head handoffs preserve `runner-prov` at
`b8acc6202824698e6480ae2fa92f1a89a6e04626` and `lint` at
`187c2f14dfca1c10d6670b81e6db7cebbe7fcc6a`, with existing authorship.
The 24 GiB/28 CPU runner proposal is withdrawn as written and remains undeployed.
The lint patch stays deferred; its reported 388 warnings require reproduction.
The deployed launcher now rejects unrecorded effective resource overrides; actual
container resources are verified by provenance. Handoffs: `foundation/held-lanes-handoff.json`.

PR49's actual scheduled report succeeded at **04:00–04:02 UTC**, with32 roots,
six unchanged installed fingerprints and report-only behavior. Independent Claude
review approved exact head `252d466a3d3c48bce2ee2432fb198eeda9f249ca`; all eight
manual CI jobs passed, and main-guard passed at the merge commit. The final immutable
receipt is `/home/ubuntu/jope-rescue-20260921/completion-20260921t201512z/final-merge-readback.json`,
linked in [the PR closeout](https://github.com/veox-ai/JopeDime/pull/49#issuecomment-5771273200).
Historical preservation limitations remain explicit in the cleanup record.

Other sessions merged #44/#50/#51/#53. This session does not claim their complete
independent-review or lane-evidence preservation lifecycle. #44's observed
cross-cut budget defect is tracked by [issue52](https://github.com/veox-ai/JopeDime/issues/52):
a four-attempt budget produced ten attempts on each of three cuts. The current
focused correction reserves comparison work within the total allowance, records
phase costs and requires complete cut coverage before selecting a program.
Implementation and qualification are separate; see [the contract](docs/pipeline.md).
Evidence lives under `/home/ubuntu/jope-rescue-20260921/implementation-20260922/pipeline-budget`.

No learning qualification or soak has started. The remaining delivery sequence is
CI coverage/aggregation/full-main, deterministic security/#47, emitter binding/#48, dataset/fold
contracts, S3 conditional recovery, benchmark/task adapters, isolated GPU execution,
canonical training/resume, successor proposals/promotion and the bounded controller,
followed by the one-hour qualification and24-hour soak. Ownership is recorded in
[AGENT_CHAT.md](AGENT_CHAT.md). The Bulletfarm-inspired contract improvements are
merged in49; its private tools and conflicting clone/build policies were not imported.

## Queue

PRs **#5–#16 were closed unmerged** on 2026-09-20 (administrative withdrawal,
not a merge or supersession). Source branches are retained. The full-history
bundle and per-PR repair checklists live in the builder's
`jope-pr-resolution/20260920` evidence tree (bundle SHA-256
`3bafaa52b9a2c6a6996a2dacc22b9c1bff9248b7833bf07d7cd9882fbcba978e`). See that
directory's `REPORT.md`.

PRs **A–F** ([#17](https://github.com/veox-ai/JopeDime/pull/17)–[#22](https://github.com/veox-ai/JopeDime/pull/22))
are merged, and so is everything through #39, including #34 — see the landing board. **Zero open PRs is not “done.”** Recover selected work **one PR at
a time**. Do not reopen the twelve-PR pile. Live S3 is not claimed. Bound
`jope-policy` checkpoint training is not claimed.

## ASM demonstration delivery — 2026-09-22

The latest explicit user direction is the private [ASM demonstration](docs/asm.md):
a fixed ten-regression/ten-binary/ten-multiclass panel, fold-zero screening and top-five
recipe confirmation over five development folds. Preserve labels, private discovery
and untouched final evaluation. The full acceptance list and unchanged runtime limits
are recorded there; wins are an empirical target, not an implementation claim.

Prerequisites PR63/65 are merged with exact-head full CI, independent native review
and successful main-guard. Canonical was verified clean at remote main
95e92af34fc95854f102965b3f4fbf01d08beeb7; four extra lanes were preserved and removed,
leaving one registered checkout. Lifecycle/custody receipts live under
`/home/ubuntu/jope-rescue-20260921/asm-demonstration-20260922`.

Current focused branch: `feat/asm-experiment-contract`. Work in progress: task refusal,
capability inventory, experiment/evaluation contracts and frozen input panel. Execution,
training and fitted export qualification remain subsequent serial PRs. The panel and
training clocks have not started; a source implementation is not a measured win.

## Product target

```text
verified training inputs → encoders → neural GP proposals + GP exploration →
  measured programs and bound emitted code → durable S3 targets and comparisons → immutable corpus →
  policy training → export and qualification → next bounded GP generation
```

Git holds tiny fixtures only (D5: the ten licensed test datasets from `wpd1`,
≈1 MB, and the forty synthetic fixtures from Codex `15a1d31`, ≈0.8 MB, are both
admitted). Every other dataset and all weights stay in S3.
This is the requested product goal, not a claim that the loop is qualified.
Preserve dataset labels and private admission; keep outer test folds and sealed
meta-test families outside learning. Accepted serving state changes only through
its bound promotion protocol. The existing centroid router is a separate baseline.
The Sep22 user-approved plan freezes the encoder for the first regression experiment.
The policy proposes GP programs; a held-out SOTA claim requires separate frozen evaluation.

## Serial recovery (Grok's lane, one open PR at a time; now merged through F)

| PR | Goal | Status |
| --- | --- | --- |
| A | Honest README / layout / MASTER_PLAN matching that SHA | Merged #17 `b388ada` |
| B | Ten Git fixtures + offline verify | Merged #18 `049410f` |
| C | Encode path (recover `wp3/hyperion-policy`, CPU/Candle) | Merged #19 `089c127` |
| D | Search/fit/emit CLI on feat-math | Merged #20 `8326c36` |
| E | Local champion bank; live S3 fail-closed | Merged #21 `86c88bd` |
| F | Local 128-d centroid router train-step + `jope route` | Merged #22 `1f0fc5a` |
| G | Live S3 champion pull/push | #34 merged `6a85fa398504f99e7787a58d7d3efd623bb6100d`; pull/recovery and server-enforced conditional replacement remain follow-ups. [The mirror implementation](crates/jope-learning/src/mirror.rs) records its compare-then-put limitation. The merge alone does not prove live S3 restoration. |
| H | Docs/ADRs after the above are true | Not started |

Grok's lane deferred: Dependabot #5, workflow copy (#7), 365-file Python dump
(#8), timings (#10), census regen (#12), path mega-move (#13), LibTorch (#14).
Those items are queued on the integrator's landing board below (it holds the
`workflow`-scope PAT and the AWS credentials that blocked them here); none is a
wholesale import — each lands as its own reviewed PR. The deferred forty-fixture
import is on that board as `rr2/d5-forty-fixtures`. It regenerates the thirty
missing fixtures instead of importing them wholesale, and it keeps the ten
`test-10` datasets (D5).

## Blockers (do not fake)

- `full-main-cpu` runs on every push to `main`; its first success came after #116 (see the
  2026-09-28 record). The `xbabe2-gpu` runner that `weekly.yml` targets is not provisioned.
  Keep runner credentials separate and preserve independent reviewer eligibility for each
  subsequent PR.
- Private-repo rulesets API returns HTTP 403 until the org is upgraded (D3,
  owner action in progress). Until then the merge rule below is process-only.
- No `--admin` merge. Worktrees only in `.agent/worktrees/`, at most four,
  with the current serial queue policy in [AGENTS.md](AGENTS.md). No credentials in git.

Coordinate in [`AGENT_CHAT.md`](AGENT_CHAT.md).

## Owner decisions — 2026-09-21

D2–D5 remain in effect. The Sep22 user-directed continuation clarifies D1 to
check both PR and commit authors and uses one implementation PR at a time;
the current operating rules are in [AGENTS.md](AGENTS.md).

| # | Decision |
| --- | --- |
| D1 | Agents merge after exact-head CI and independent review by an identity different from both PR and GitHub-associated commit authors (`jepsontaylor` / `neverhuman`, when eligible), with guarded merge and remote `main` readback. |
| D2 | PR gate at `70506066` = the seven jobs `ci.yml` defines: `changes`, `hygiene`, `rust-check`, `emitters`, `inputs`, `python` (#38), `rust-test` (#39) — all path-gated. **`rust-check` runs `cargo check --workspace --all-targets --locked` plus a systemd test and an MSRV check; it runs neither clippy nor rustfmt** — those exist only in the local `justfile`, which is why #37 had to land rustfmt drift that no lane would have caught. Full nextest, LibTorch `features`, sanitizers, security, artifacts, Jankurai = visible non-blocking main/nightly lanes, each promoted to blocking once it passes on main. Jankurai is a no-regression ratchet, never a merge gate. |
| D3 | Owner upgrades `veox-ai` to Team; branch protection (required checks + 1 approval + no force-push) is enabled the moment the API stops returning 403. |
| D4 | `veox-jopedime-{datasets,models,learning,archive}-use2` (+usw2 replicas) = product buckets the crates target. `jopedime-714044927654-us-east-2-{dev,prod,sealed}` = preservation custody. Nothing deleted; `catalogs/s3.json` records both. |
| D5 | Both the ten real test datasets (`wpd1`) and the forty synthetic fixtures (Codex `15a1d31`) are admitted to Git. |

## Merge rule (per PR)

1. Required contexts = every job that `.github/workflows/ci.yml` defines **at the reviewed head** (on `main` at `3e061605`: `changes`, `hygiene`, `rust-check`, `emitters`, `inputs`, `python`, `rust-test`, `e2e-smoke`). A lane legitimately excluded by its declared path matrix is recorded as `skipped-by-path`, not green. Every required execution must pass at the exact head; absent, cancelled or failed is not green. The configured jobs are not proof of execution. Heavy functional lanes become required after their successful main qualification; Jankurai remains informational under D2.
2. Review must come from a session that did not write the PR and a GitHub identity distinct from both the PR author and every GitHub-associated commit author. Follow the PR-scoped identity checks in `AGENTS.md`; check `gh api user --jq .login` in the posting environment. #40 and #41 were opened under the wrong account and replaced by #42 and #43. An agent-operated approval proves an independent read at the exact head, not human sign-off. Trust-surface PRs (workflows, lockfiles, credentials, protection manifests) also receive a security review.
3. `gh pr merge <n> --rebase --match-head-commit <reviewed-full-sha>`, then verify fresh remote `main` and `main-guard`. Preserve shared source-branch history when updating its base.
4. Resolve the existing queue, then one implementation PR at a time. Fix review findings on its existing branch; record explicit replacement/disposition for superseded work. The four managed-lane limit does not authorize four concurrent implementation PRs.

## Landing board (Claude integrator; old PR → new PR → merged SHA)

Open rows are pinned to the head and review they describe, exactly as merged
rows are pinned to a merge SHA. A row about a moving PR is unfalsifiable and
goes stale within the hour; live state belongs on the PR, where GitHub is the
source of truth.

A merge and its landing row land in the same push, and any change to this file
re-reads `git ls-remote origin main` and `gh pr list --state merged` at **push**
time rather than draft time. This revision exists because #42 merged two
minutes before the previous one was pushed, so a board opened to fix a stale
readback shipped one of its own. Neither command takes a second, and neither
can be satisfied by remembering.

| Old PR | Branch | Head | New PR | Merged |
| --- | --- | --- | --- | --- |
| — | `coord/integrator-lane-20260921` | | #23 | `4753a9a` 04:06:47Z |
| #7 | `ci/lanes` | 38aeaa1 | #24 | `932f7b7` 04:06:52Z (rebase; + inputs lane; review found the strict-parity step ran 0 tests — fixed) |
| #10 | `wp1/test-timings` | 5f66e3f | #29 | `0c14405` 05:08:29Z (rebase; 3 nextest exclusions module-qualified after review) |
| #14 | `wp8/neural-receipt` | 44501a9 | #25 | `dc5ccf9` 2026-09-21T03:58:58Z (rebase; reviewer `neverhuman`) |
| #12 | `wp10/parity-census` | c483544 | #28 | `d1317a1` 04:16:19Z (rebase; + rows for the #19 Hyperion bins) |
| #9 | `wp7/model-manifests` | f62095c | #26 | `44443ab` 04:08:07Z (rebase; manifests byte-identical to S3) |
| #6 | `wp3/hyperion-policy` | a043f8b | — | recovered by Grok's #19 (`089c127`): `main` is a strict superset; no PR needed |
| #11 | `wpd1/datasets` | d672e76 | #27 | `a6df267` 04:44:49Z (rebase; object_store 0.14 for RUSTSEC-2026-0194/0195; scoped webpki-root-certs exception; lock additive) |
| #15 | `wp7/models-cli` | b497325 | #33 | `5c66edb` 04:54:50Z (rebase; `jope models`/`datasets` umbrella on Grok's verbs; blob-lock deadlock fixed) |
| #8 | `wp6/python` | d49662f | #31 | `99981e6` 04:42:33Z (rebase; ruff format + pidfd fallback) |
| #13 | `wp4/paths` | 96a61c8 | #32 | `ff5da5b` 05:04:22Z (rebase; abs-path hygiene step now blocking for crates/deploy/.github) |
| #16 | `wp9/docs-release` | 9bec536 | #30 | `aa57388` 04:33:58Z (rebase; README kept as Grok's honest front door, quickstart → docs/quickstart.md, status banners) |
| — | `wp12/arena` | 53f5ec4 | #35 | `53ea45d` 08:21:44Z (rebase; rustls provider guard fails closed, ureq 3 proxy/redirect deltas disclosed) |
| — | board | | #36 | `ebf055d` 06:36:46Z (landing rows through #33) |
| — | rustfmt drift | | #37 | `dc46e86` 06:47:00Z (two drifted hunks in `jope-engine eval_registry/query`) |
| — | python lane | | #38 | `d43bc5f` 07:49:28Z (fixes the systemd test #32 broke; **adds the hosted `python` lane that would have caught it**) |
| — | `wp11/learning-store` | 44aa4b69 | #34 | `6a85fa398504f99e7787a58d7d3efd623bb6100d` 2026-09-21T17:30:58Z; remaining live-recovery limits are recorded above |
| — | rust-test lane | | #39 | `7050606` 11:11:35Z (4 rounds; the hosted gate ran no Rust tests beyond two `feat-math` targets, and the scope guard itself failed open twice before the reviewer's negative probes closed it) |
| — | ADR-0003 test paths | | #42 | `5feb2fa0` 11:50:13Z (three tests red since the ADR-0003 relocation, fixed; quarantine 15 → 12 and the misfiled `queue.rs` panic given its own cause. #40 was the same branch opened from the wrong identity and is closed) |
| — | `wp13/pipeline-run` | 16ed3505 | #44 | `584112293221563aa1acfe1b9a06bf3c987f1e6f` 2026-09-22T03:11:35Z; CPU GP pipeline. Codex owns budget/selection correction [#52](https://github.com/veox-ai/JopeDime/issues/52) and independent validation of the pipeline/CI assertions after #49. Artifact binding #48 remains open. The final review disclosed author involvement in two CI commits; this readback does not certify full review independence. |
| — | `wp5/cli-split` | d0b4819 | | needs a mechanical re-do on current main, not a replay (#32 rewrote 67 of the files it moves) |
| — | `wp8/runner-lanes` | 51857841 | #45 | `c572f1778562768ebc7c29fbfeff047f871cba6c` 2026-09-21T16:55:46Z |
| — | strict Python parity prerequisites | d76cdf1e | #50 | `baa4c1f24e7b2ab7f38af5061feea6029d27cc94` 2026-09-22T03:10:24Z |
| — | generated README benchmark results | 9e44a206 | #51 | `25242513173e1a8b01f7d1e9f73775b7df14110e` 2026-09-22T03:20:41Z |
| — | feat-math Clippy cleanup | c0e8d473 | #53 | `3e0616057bc3234d568deceb884ea1c69ea8451b` 2026-09-22T04:09:11Z; external merge, remaining workspace lint work still requires its own measured evidence |
| #5 | dependabot cargo group | 1151493 | the v0.2.0 release PR | its merge commit, whose SHA the v0.2.0 release readback records: #5's intent re-cut on current main as a lock-only bump (`cc` 1.4.5 to 1.5.1, `clap` 4.6.6 to 4.6.7, `crc32fast` 1.5.1 to 1.5.2), one commit of the release PR |
| — | `rr2/d5-forty-fixtures` | e0b906b239b107701364adb41d522f7a7c1ff051 | #121 | `01ef9ddcb31133bea27188e29af9978c2d36b8c4` 2026-09-28T15:40:24Z (rebase; four commits; reviewed tree `65d8c3f2ae2e59dbccfa322ebc36d42f558fe947` equals the merged tree; approval 5340876669; PR run 36442700858 and manual run 36442700618 green; the forty fixtures, 8 supported and 32 reported unsupported; the `test-10` datasets are kept) |

Recovery of #5–#16: bundle `pr-5-through-16.bundle` (sha256 3bafaa52…) is also
at `s3://veox-jopedime-archive-use2/git-bundles/20260921/`.

The #44 review disclosure and checks are retained in
`/home/ubuntu/jope-rescue-20260921/implementation-20260922/queue/external-merges-20260922.json`;
the #52 acceptance is linked in its row. Corrective validation cannot retroactively
make the earlier merge independent.

## What the gate actually proves (2026-09-21, measured)

Read this before trusting a green check. Every number here names where it was
measured; a figure without a source is a defect in this section.

- Until #38 and #39 the hosted gate ran **almost** no tests: `hygiene`,
  `rust-check`, `emitters`, `inputs`. The one exception is real and worth
  naming rather than rounding away — `emitters` ran `cargo test -p feat-math
  --lib -- cxx::` and `--test portable_parity`. Nothing else. No other crate's
  tests and no Python ran at all, so a green PR meant "it compiles and the
  emitters agree". #38 added the `python` lane, and exists *because* #32 broke
  an existing Python test that nothing ran. #39 added the Rust half.
- **`rust-check` is `cargo check`, not a lint gate.** It runs `cargo check
  --workspace --all-targets --locked`, the `systemd_unit_paths` test and an
  MSRV check. Clippy and rustfmt live only in the local `justfile`, so
  formatting drift reaches `main` unchallenged — #37 is exactly that, landed
  after the fact. Adding `cargo fmt --all --check` to the gate is cheap and
  owed, and is already written on `ci/rust-test-followup-20260921`
  (`f1cbf820`) awaiting a slot. Clippy cannot join it: `cargo clippy
  --workspace --all-targets --locked -- -D warnings` at `70506066` on this
  host fails in `feat-math` — **31 errors in the lib, 37 in the lib test** —
  and exits `RC=101`. So the `~31 inside` half of the figure this board quoted
  at `9628a652` reproduces almost exactly, and the `143 outside feat-math` half
  is **unmeasured rather than refuted**: every diagnostic in that run names a
  `crates/feat-math/` path, and once the build failed cargo stopped scheduling
  the rest (`warning: build failed, waiting for other jobs to finish...`),
  having already checked `jope-paths`, `numpy-pcg64`, `jope-arena` and
  `jope-datasets` clean. So no workspace figure is quoted here. A count for the
  rest is one `--exclude feat-math` away; nobody has run it.
- **The hosted `rust-test` lane, measured on the first green run that retained
  an artifact** (`35586778652`, artifact `rust-test-evidence` = `10633028231`,
  independently audited by the qualification reviewer; the lane was already
  green at `35580809723` 09:00Z and `35583896833` 09:32Z, neither of which kept
  anything — which is the whole reason retention landed): **651 tests executed and passed**, target **8.98 GB**, free disk
  11.5 GB before and 3.9 GB after, nextest **626 s** inside an **822 s** job.
  Full product inventory 664 = 651 executed + 9 ignored + 4 filtered by the
  `pr` profile. The tree's own comments said 9.4 GB / 617 tests / "about a
  minute" — local numbers from before the lane ever ran hosted, corrected on
  `ci/rust-test-followup-20260921`.
- A full-workspace `cargo test --no-run` links ~72 GB against a runner's ~14 GB
  (documented identically in `deploy/ci/rust-test-scope.sh` and `ci.yml`) — it
  SIGBUSes rather than failing a test. #39 therefore gates a **checked
  partition**: 11 product crates on pull requests, 9 deferred to self-hosted,
  with `deploy/ci/rust-test-scope-check.py` asserting against `cargo metadata`
  that the two agree with the script's own output and leave nothing unplaced.
  A crate in neither list would be gated by nothing; `jope-arena` did that
  once, and the guard caught `jope-pipeline` doing it next — on the unmerged
  `wp13/pipeline-run` (#44, `8b4cf466`), so do not look for that crate on
  `main`.
- **A full-workspace run is red on `main` for reasons unrelated to any branch.**
  18 failures across `jope-parallel`, `jainnode` and `jain-smartcluster-daemon`,
  measured by the integrator session on this host with `cargo nextest run -P pr
  --locked --workspace` on 2026-09-21 and reproduced on `main` itself. **This
  has not been reproduced in CI and has no receipt in the tree** — treat it as
  an integrator measurement, not a verified fact, and re-measure before relying
  on it. Note the `pr` profile *as gated* is green: 651/651. Note also that
  `deploy/ci/known-broken-tests.txt` contains **zero** `jainnode` and
  `jain-smartcluster-daemon` entries: every entry in it is a `jope-parallel`
  binary, so the file is scoped to that crate and is not a census of what fails
  on `main`. It should say so.
- **The quarantine, counted rather than estimated** (15 entries at
  `70506066`): the `cause A` block holds 9, but the fit-authority fence
  explains **5** of them (2 "namespace capability inactive", 3 "fit authority
  fenced"). The other 4 are 3 × the ADR-0003 relocation (cause A0, fixed in the
  #42, merged) and 1 × a `queue.rs:133` panic that was misfiled under the fence.
  `cause B` (behavioural assertions) holds 5 and `cause C` (racy, passes
  serially) holds 1. #42 took the file to 12 and unfiled the panic.

## Correction to the product plan: the engine cannot measure

`crates/jope-engine/src/fit_boundary.rs` requires `JOPE_FIT_SERVICE_SOCKET`
unconditionally — no socket, no fitting — and
`eval_registry/authority.rs::activation_block` returns a block for every
non-test scope ("production capabilities have no verifier yet"), so **no
shipped binary can measure through the engine**. That is not a bug to route
around quietly: it is a deliberate fence, and it explains **5** of the 15
quarantined tests — not the 9 the `cause A` block appears to hold, which is
why that block was unfiled in #42.

Consequence for the pipeline: **WP-P1 (`jope pipeline run`) builds on
`feat_math::evolve` + `feat_math::fit` + `feat_math::transpile`, not on
`jope-engine`'s held entry points.** `crates/jope-cli/src/kernel.rs` is the
unfenced path today, but its `SEARCH_CANDIDATES` is a three-candidate loop, not
a genetic program — the GP is `feat_math::evolve`
(`random_canonical_pipeline` / `mutate_canonical_pipeline` /
`crossover_canonical_pipelines`) and it is complete. Lifting the fence is a
separate, reviewed change with its own verifier, never a side effect of a
pipeline PR.

## Recorded conflict — owner to settle (2026-09-21)

Two agents hold different owner directions dated 2026-09-21:

- **Integrator (Claude, this board's D1–D5):** rapid PRs; agents merge after CI +
  cross-identity review; fast PR gate; both fixture sets admitted; org upgrade
  for protection. Owner-approved plan copy:
  `/home/ubuntu/jope-migration-evidence/claude/jopedime-unstall-20260921/PLAN.md`.
- **Coordinator (`jdime_merger_cursor`, review on #23 at 03:50Z):** a
  "stopped-writer plan" requiring explicit owner releases from every writer,
  stopped preservation workers/supervisors and restoration first, one
  implementation/ledger owner, exactly ten synthetic scenarios, and
  protection/qualification before any merge. It reports that Grok has released
  its JopeDime lane and that the preservation owner has not.

**Owner ruling (2026-09-21 ≈03:58Z, given directly to the integrator session
with all three directions quoted verbatim): D1–D5 control.** Codex's foundation
work lands through the same reviewed-PR gate one small PR at a time with the
≥95 audit as a ratchet; Cursor keeps the reviewer/coordinator role; the
preservation lane and its live workers (PIDs 115122, 3692153, 3692154, 3777416
at 03:20Z) are not touched by anyone. The stopped-writer and audit-before-merge
holds do not apply to this repository from that time. Posted on the canonical
board, the canonical chat and the cross-tree wire at 03:58:38Z.

Merge history for the record: #1–#16 were closed unmerged (2026-09-18/20);
#17–#22 (Grok's A–F) are the first merges in this repository, 2026-09-21
01:24–03:16Z, `main` `4235db6` → `1f0fc5a`.

## Known stale — do not cite

Claims that were true once and are not now. They are listed rather than
silently deleted, because they have been quoted at each other across three
sessions.

| Claim, and where it still appears | Why it is wrong |
| --- | --- |
| "617 tests", "9.4 GB", "about a minute" — `deploy/ci/rust-test-scope.sh` and `.github/workflows/ci.yml` at `70506066`; still in `deploy/ci/rust-test-scope.sh` and the `test-pr` comment of `justfile` | Local numbers from before the lane ever ran hosted. Measured: 651 tests, 8.98 GB, 626 s. Corrected on `ci/rust-test-followup-20260921`; the tree carries the old ones until that lands. |
| "the hosted gate ran no Rust tests at all" — this board's earlier draft and the #39 discussion | `emitters` ran `cargo test -p feat-math --lib -- cxx::` and `--test portable_parity`. The rest of the claim holds. |
| "the fence explains 9 of the quarantined tests" — this board's earlier draft | It explains 5. The `cause A` block held 9 because 3 relocation failures and 1 panic were misfiled under it. |
| "`rust-check` runs clippy and fmt" — the D2 row before this revision | It runs `cargo check`, a systemd test and an MSRV check. `cargo fmt --all --check` now runs in the hosted `hygiene` lane (`deploy/ci/check.sh`); clippy is still in no hosted lane. |
| "Full-main CPU workflow and runner delivery remain unimplemented and unqualified" — this file's Blockers before the 2026-09-28 record | `full-main-cpu` runs on every push to `main`. Its first success, 36343656796, came after #116; 36472565658 passed on `a8e4ce8f1f7cdf2f518e867147c4f0b038d966c5`. |
| "Live S3 is **not** wired" — the README before the v0.2.0 release PR | The dataset and model stores and the learning mirror reach `s3://` through `object_store`. What fails closed is the champion route (#62) and `train-step --env`; no live S3 training run has been made. |
| "A `v0.1.0` tag through `release.yml` with attested `jope` binary, wheel, SBOMs and receipt" — `docs/public-readiness-checklist.md` before the v0.2.0 release PR | Releases come from `deploy/release/release.sh` (owner decision D-D), with a cosign signature over `SHA256SUMS`; `release.yml` is dispatch-only; there is no wheel and no attestation. The first release is v0.2.0. |
| In the stale `jain-split` mirror: "Eight-A100 recovery" in the header of `/home/ubuntu/jain-split/jope-dime/AGENT_CHAT.md:12` (not this repository's file, whose header says the opposite); and `main` at `458ef58` **with** "zero Actions workflows / runs" in that tree's `FINAL_GITHUB_PORT_PLAN.md:116` and `CODEX_GITHUB_PORT_PLAN.md:27-28`. "13 open PRs" appears in neither and was this appendix's own misattribution | Pre-2026-09-21 state of the stale `jain-split` mirror, which is now a pointer. |
| PR #40, PR #41 | Opened from the reviewer's identity and closed unmerged; the same branches are #42 and #43. Not withdrawals of the work. |

### CPU foundation record (PR61, 2026-09-22)

Source, reviewed deployment and live run evidence are recorded at the top of this document.
They remain separate from full CPU qualification, which requires the subsequent repair phase.
