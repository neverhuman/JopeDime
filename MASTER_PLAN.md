# JopeDime current direction

**2026-09-22 coverage-foundation readback.** Publication repo: private `veox-ai/JopeDime`.
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

The current claimed branch is `ci/coverage-foundation`. Static mapping in
[coverage.json](deploy/ci/coverage.json) assigns22packages,18features,214Cargo targets,
213Python test-named files,4workflows/12jobs and32benchmark sources. These are static
counts, not executed tests. Real maintained pytest collection reports222 tests;
collection errors now remain failures and static mode requires explicit selection.
Receipt foundations bind source/checkout SHA, run/attempt, command, status, counts
and evidence digests; wiring and strict aggregation remain the next serial PR.
See [coverage/receipt semantics](deploy/ci/coverage.md). This phase still requires
final-head hosted CI, independent review and guarded merge.

The user-approved bounded foundation remains serial: explicit coverage/receipts;
trustworthy aggregation/operator status; then a diagnostic push-triggered full-main
CPU baseline. Full Rust/Python repair, strict Clippy, persistent runner migration and
mandatory full-CPU gates remain the following phase. Qualification and soak are
unstarted. Fast jobs are limited to two at 16 GiB/16 CPUs each; full CPU is exclusive
among JopeDime jobs on its host at 48 GiB/16 CPUs, 120 minutes and a 50 GB disk reserve.

Stopped-head handoffs preserve `runner-prov` at
`b8acc6202824698e6480ae2fa92f1a89a6e04626` and `lint` at
`187c2f14dfca1c10d6670b81e6db7cebbe7fcc6a`, with existing authorship.
The 24 GiB/28 CPU runner proposal is withdrawn as written and remains undeployed.
The lint patch stays deferred; its reported 388 warnings require reproduction.
The runner delivery must also bind effective resource overrides, which the current
installed-script digest does not cover. Handoffs: `foundation/held-lanes-handoff.json`.

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
(#8), timings (#10), census regen (#12), path mega-move (#13), LibTorch (#14),
forty-fixture wholesale import. Those items are queued on the integrator's
landing board below (it holds the `workflow`-scope PAT and the AWS
credentials that blocked them here); none is a wholesale import — each lands
as its own reviewed PR.

## Blockers (do not fake)

- Coverage foundation still needs final-head hosted CI, independent review and guarded merge.
  Writer access has recovered; keep runner credentials separate and preserve
  independent reviewer eligibility for each subsequent PR.
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
| #5 | dependabot cargo group | 1151493 | | re-trigger after main settles; lockfile = trust surface, two lenses |

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
| "617 tests", "9.4 GB", "about a minute" — `deploy/ci/rust-test-scope.sh` and `.github/workflows/ci.yml` at `70506066` | Local numbers from before the lane ever ran hosted. Measured: 651 tests, 8.98 GB, 626 s. Corrected on `ci/rust-test-followup-20260921`; the tree carries the old ones until that lands. |
| "the hosted gate ran no Rust tests at all" — this board's earlier draft and the #39 discussion | `emitters` ran `cargo test -p feat-math --lib -- cxx::` and `--test portable_parity`. The rest of the claim holds. |
| "the fence explains 9 of the quarantined tests" — this board's earlier draft | It explains 5. The `cause A` block held 9 because 3 relocation failures and 1 panic were misfiled under it. |
| "`rust-check` runs clippy and fmt" — the D2 row before this revision | It runs `cargo check`, a systemd test and an MSRV check. Neither linter is in any hosted lane. |
| In the stale `jain-split` mirror: "Eight-A100 recovery" in the header of `/home/ubuntu/jain-split/jope-dime/AGENT_CHAT.md:12` (not this repository's file, whose header says the opposite); and `main` at `458ef58` **with** "zero Actions workflows / runs" in that tree's `FINAL_GITHUB_PORT_PLAN.md:116` and `CODEX_GITHUB_PORT_PLAN.md:27-28`. "13 open PRs" appears in neither and was this appendix's own misattribution | Pre-2026-09-21 state of the stale `jain-split` mirror, which is now a pointer. |
| PR #40, PR #41 | Opened from the reviewer's identity and closed unmerged; the same branches are #42 and #43. Not withdrawals of the work. |
