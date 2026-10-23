# JopeDime agent contract

The operating contract for contributors and automated agents working in this repository.

The 2026-09-21 audit found that `main` carried **no root `AGENTS.md`** — one existed only on an
unmerged branch, so an agent working from a fresh clone had no contract at all. That gap is a direct cause of the
disk incident recorded in [`MASTER_CLEAN_UP.md`](MASTER_CLEAN_UP.md).

---

## Direction and evidence

The product repository is `neverhuman/JopeDime`; its current goal is the bounded neural GP learning
loop in [MASTER_PLAN.md](MASTER_PLAN.md#product-target). Dataset labels stay intact; outer test
folds and sealed meta-test families stay outside the learning worker's input closure.
See [ADR-0005](docs/adr/ADR-0005-canonical-learning-loop.md) for the model lineage,
[learning status](docs/learning.md) and the [capability matrix](docs/parity/capability-matrix.md)
for recorded capabilities, and [MASTER_PLAN.md](MASTER_PLAN.md) for the delivery record.

Read those records as dated observations. Refresh the relevant Git, CI and runtime evidence before
acting; stale plans or historical PASS entries do not establish current capability. The latest
explicit user direction controls scope and budgets. A configured command or workflow, a merged
implementation, and a qualified live learning loop are separate milestones.

## Which checkout

**The canonical local checkout is `/home/ubuntu/JopeDime`**. Work on its claimed PR branch;
return it to verified remote `main` after the reviewed lifecycle.

Do not confuse this repository with `~/jain-split/jope-dime` (also reachable as `~/jope-dime`, a
symlink). That is a different repository on a different upstream — the local forge at
`git.neverhuman.org` — and it is the research monorepo, not this product repo.

## Claim before writing; hand off with evidence

Before the first write and after resuming a session, read the open PR queue, registered worktrees,
and the latest entries in [AGENT_CHAT.md](AGENT_CHAT.md). Hold `flock` on `AGENT_CHAT.md.lock`,
reread the claims, then append the checkout, branch, full HEAD, owner, paths and intended
verification before editing. An overlapping claim waits for the current owner's stopped-head
handoff; age or an idle session alone does not release it. An explicit later user assignment can
transfer ownership: preserve the predecessor's work and record the transfer and observed state
under the same lock before writing. Cleanup work also follows the
synchronized [MASTER_CLEAN_UP.md](MASTER_CLEAN_UP.md) record, whose second copy is in the research
repository; claim that repository separately before changing its copy. Recheck branch, HEAD,
dirty state and overlapping writers immediately before a checkout or index mutation. The append
lock serializes the record; it does not itself lock a checkout for the whole task.

Keep corrections on the claimed PR branch. At handoff, record the stopped HEAD, dirty files,
commands and outcomes, durable receipt locations, remaining blockers and ownership disposition.
A campaign claim links its pinned manifest and approved resource limits and deadline. A fresh
context resumes that state and its existing budget; it does not start a second lane or reset a
campaign deadline. JopeDime uses these existing records, not Bulletfarm's `bf board`/`bf claim`.

## Where work happens

**Never clone into `/tmp`.** `/tmp` is emptied at boot. On 2026-09-21 it held **442 GiB across 180
clones** of this repository, including commits that existed nowhere else and were one reboot from
being lost permanently. Use the reusable bare verification directory under custody for independent restoration.

**After its PR closes, preserve and restore-verify unique source and evidence, then remove the lane.**
**At most 4 concurrent lanes.**

**Worktrees live in exactly one place: `<repo>/.agent/worktrees/`** — owner ruling, 2026-09-21.
One shared directory for every agent (Claude, Codex, Grok) rather than a per-tool location each.
`.claude/worktrees` is a **symlink** to `.agent/worktrees`, so Claude Code's own worktree tool lands
in the shared directory without any configuration. **Do not replace that symlink with a real
directory**, and do not introduce `.codex/worktrees` or similar — that is the sprawl this replaces.

**Bootstrap before any worktree tool in a fresh checkout.** These ignored directories and
the symlink are not supplied by Git. From the repository root, create `.agent/worktrees`
and `.claude` with `rtk proxy mkdir -p .agent/worktrees .claude`. If there is no entry at
`.claude/worktrees` (a dangling symlink counts as existing), create it with
`rtk proxy ln -sT ../.agent/worktrees .claude/worktrees`. Before every worktree tool use,
verify it is a symlink and `rtk proxy readlink -f .claude/worktrees` resolves to this
repository's `.agent/worktrees`. A real directory or a different target requires
preservation and reconciliation before proceeding; never replace it with `ln -f`.
Its absence or wrong type is invisible to `git status` because `.claude/` is ignored.

- **Never `/tmp`** — not worktrees, not clones, not build output. It is emptied at boot.
- **At most 4 concurrent lanes per repository**, one build directory each, reused.
- **Remove a lane when its PR closes** with `git worktree remove`, never `rm -rf` (which orphans
  the admin entry).

This supersedes the previous blanket ban, which was phrased as a prohibition on `git worktree add`
and so was walked straight past by Claude Code's `EnterWorktree` **tool**. A rule stated as a
*location* is enforceable; one that names a single invocation is not.

## Build output

- **One build directory per lane, reused.** Never a fresh `--target-dir` per task. The alternative
  produced 119 separate Cargo build directories totalling 809 G.
- Verify or explicitly set `CARGO_INCREMENTAL=0` and
  `CARGO_PROFILE_DEV_DEBUG=line-tables-only` before Cargo; not every agent inherits them.
- **Keep `CARGO_TARGET_DIR` unset globally.** Issue #46 was fixed by PR #56 with normal and
  external-directory proofs: wrappers resolve executables from Cargo artifact messages and
  `jope-paths` preserves explicit root-selection precedence. CI now selects its persistent
  target per job through `.github/actions/cargo-target`; preserve one reused target per lane
  and the separation between PR and trusted-main storage domains.

## `target/` is partly an evidence store — never `rm -rf` it

`target/health/` holds generated score files that are **evidence**: regenerate them through their
owning proof lane, never hand-edit, and never blanket-delete the tree they live in.

The maintenance helper reports candidates only. Cargo markers do not certify contents;
source and evidence at every depth need preservation and independent verification before removal.

## Preserve before you delete — and then delete

Both halves are required. "Preserve everything" with no counterpart is how a disk reaches 91 %.

1. **Preserve first.** Bundle it, restore the bundle in a reusable independent `git clone --no-local --bare` under custody, and
   confirm **the specific commit is present** — a bundle can verify and still lack the commit you
   care about. This is not hypothetical: during the 2026-09-21 cleanup a detached worktree `HEAD`
   (`b2f72ec`) was absent from a `--all` bundle and would have been lost.
2. **Then delete.** A complete verified recovery route plus a fresh dependency check permits logged reclamation.

**Only trees you are going to delete need a bundle.** Bundling a tree you are keeping consumes the
disk you are trying to reclaim.

**Count unpushed commits with `git log --all --not --remotes`.** The branch-only form
(`git log --not --remotes`) examines only the checked-out branch and will report "zero unpushed
commits" for a repository that holds dozens on other local branches.

## Pull requests

**Owner ruling, 2026-10-21: the same agent may implement, review and merge its PR.**
This supersedes the earlier requirement for a different agent, session or GitHub identity.
An outside reviewer and a native GitHub approval are optional; their absence is not a blocker.
Keep review substantive and bind it to the exact full head SHA. Preserve existing authorship.

Land work through the reviewed lifecycle: branch → PR → CI green at the exact head →
agent review → merge → verify the remote state. `main-guard` checks that every commit
on `main` belongs to a merged pull request; it does not certify review quality.

Before starting new implementation work, check the open pull-request queue. If a PR is awaiting
review, checks, correction, or merge, resolve it before opening another lane. The current delivery
plan uses one implementation PR at a time; the four-lane ceiling is not a concurrency target.

Each focused PR states the behavior it changes, the acceptance command, and the evidence produced.
For behavioral fixes, preserve a reproducing failure and prove the corrected behavior; refusing
an operation and successfully executing its authorized path are different acceptance cases.
For documentation, `git diff --check` and link/import verification supplement the same exact-head
CI requirement; a declared path exclusion is recorded as `skipped-by-path`, not passed.
Any changed head needs fresh checks and review. Resolve blocking findings before merging.
Trust-surface changes, including workflows, lockfiles, credentials and protection settings,
also receive a security review; the implementing agent may perform it.

### Approval procedure on this host

Use the established writer identity for commits, pushes and PRs. Verify the live identity rather
than relying on historical account names or Git config. The `xbabe2` helper selects a configured
account for one child process without changing the shared active login:

```sh
/home/ubuntu/.local/bin/gh-role writer <gh arguments>
```

An optional separate reviewer may use `gh-role reviewer` after verifying its identity and
eligibility. The installed helper's native-approval eligibility restrictions still apply.
Do not alter credentials, switch accounts or rewrite attribution to manufacture a self-approval.
Same-agent review is recorded honestly as such; GitHub does not allow a PR author to submit a
native approval on their own PR. A head-bound `COMMENT` review or equivalent durable review
receipt records the agent's findings without requiring that approval.

1. Claim the checkout as above and preserve existing commit authorship. Verify the posting
   identity with `gh-role writer api user --jq .login`. Freeze the final full head SHA and retain
   the PR, commit inventory and complete review diff.
2. Dispatch `gh-role writer workflow run ci.yml --repo neverhuman/JopeDime --ref BRANCH`.
   Record dispatch time, then identify the new manual run by branch, creation time and frozen
   source SHA. Retain run ID, full source SHA and attempt; never substitute an older success.
   All declared full manual lanes, including both Rust test shards, frontend tests and aggregate,
   must pass at that head. A path-gated PR run may legitimately skip declared lanes; retain its
   synthetic checkout SHA separately. Missing, stale, cancelled, failed and zero-test results
   are not passes. Use **Re-run all jobs** for another attempt.
3. Review the complete candidate and relevant context, including substantive test evidence and
   unresolved discussions. The same implementing agent may perform this review. Record the
   exact full SHA, findings and disposition in a durable receipt or native `COMMENT` review
   with `commit_id: FULL_SHA`. Do not describe a self-review as an independent or human review.
   Preserve any existing independent review evidence as additional evidence.
4. Immediately before merging, recheck the current head, current-attempt CI, review disposition,
   discussion resolution and effective protection. Preserve API readbacks outside the checkout.
   The 2026-10-21 owner-authorized settings require zero approving reviews, no last-push approval
   and no extra approval for unattributed changes in both main protection layers. Required
   status checks, strict up-to-date checks, conversation resolution and push protections remain
   enforced. Read settings back before claiming enforcement; do not bypass them with `--admin`.
5. Merge with the guarded command, then verify `merged_at`, resulting remote main and successful
   `main-guard`. GitHub's rebase merge does not authorize rewriting the shared source branch:

   ```sh
   gh-role writer pr merge PR_NUMBER --repo neverhuman/JopeDime --rebase --match-head-commit FULL_SHA
   ```

   If GitHub cannot safely rebase a branch with conflict resolution, use the enabled squash
   strategy with the same `--match-head-commit FULL_SHA` guard and all checks above. Preserve
   the shared source branch and verify that the merged tree exactly matches the reviewed tree.

   Return the canonical checkout to verified remote main while preserving live coordination
   appends and other owners' dirty files. Retain lifecycle receipts before removing closed lanes
   under the custody rules.

### Credential and key locations

These are host-local paths, verified by file metadata on 2026-09-22. Keep secret bytes out of
Git, chat, review prompts, receipts and command output. Do not print these files or run token
retrieval as a standalone displayed command. Account access and eligibility must still be
checked at use time; `GH_TOKEN`/`GITHUB_TOKEN` can override the stored CLI account.

| Purpose | Location | How it is used |
| --- | --- | --- |
| GitHub CLI writer/reviewer authentication | `/home/ubuntu/.config/gh/hosts.yml` (0600) | Stored `jepsontaylor` and `neverhuman` accounts. Retrieve through `gh auth token --hostname github.com --user ACCOUNT` into a child environment; do not parse or copy the file. A configured `GH_CONFIG_DIR` changes the location. |
| GitHub SSH writer key | `/home/ubuntu/.ssh/id_ed25519_github` (0600) | Selected by `Host github.com` in `/home/ubuntu/.ssh/config`. Git transport only; it does not select the API approval identity. |
| GitHub SSH alternate key | `/home/ubuntu/.ssh/id_ed25519_neverhuman` (0600) | Selected by SSH aliases `github-neverhuman` and `github-mirror`. Git transport does not select the API identity; same-agent review needs no native self-approval. |
| Claude CLI authentication | `/home/ubuntu/.claude/.credentials.json` (0600) | Authentication for an optional Claude process; separate from GitHub review credentials. |
| Runner registration PAT | `/etc/jope-runner/github-pat` (root:root, 0600 on runner hosts) | Host launcher only, for single-use JIT registration. Never mount into a job or use as the PR review credential. |
| GitHub identity helper | `/home/ubuntu/.local/bin/gh-role` (0700) | `gh-role writer` → `jepsontaylor`, `gh-role reviewer` → `neverhuman`. The supported way to select an identity; see the approval procedure above. xbabe2 only. |
| GitHub writer token file | `/home/ubuntu/.config/jopedime/secrets/gh.env` (0600) | A **bare token on one line with no `KEY=`**, despite the suffix. Sourcing it attempts to execute the token as a command and can expose it in shell diagnostics. Never source it; prefer the CLI credential store through the helper. |
| AWS credential for JopeDime S3 | profile `veox-jepson-prod` in `/home/ubuntu/.aws/config` (0600), resolved by `credential_process` `/home/ubuntu/.aws/veox-jepson-new-cred.sh` (0700) | Verified 2026-09-25: principal `arn:aws:iam::714044927654:user/jepson`, with `AdministratorAccess` through group `admin`. **Administrator access, not an append-only writer.** `veox-jepson`, `veox-jepson-new`, and `veox-jepson-prod` share this provider; they are not separate dev/prod credentials. Verify permissions at use time. `veox-alton` (`/home/ubuntu/.aws/veox-alton-cred.sh`, 0700) also reaches the same account. |
| AWS sandbox credential | profile `veox-jepson-sandbox` → `/home/ubuntu/.aws/veox-jepson-cred.sh` (0700) | A **different account** (`228886154691`) holding different buckets. The prior datasets-bucket probe returned `AccessDenied`; it is not the selected product profile. |
| `aws.env` | `/home/ubuntu/.config/jopedime/secrets/aws.env` (0600) | **A human-readable note, not an env file**: the earlier inspection found no assignments. Never source it. Use the profile above. |

As of 2026-09-25, append-only dev/prod writer credentials and separate dataset-curator
credentials are **not provisioned—no verified path available**. Claude's credential design
is a draft; illustrative profile names are not installed credentials. The shared administrator
provider above does not supply those restricted roles. This inventory documents current facts
only; it does not authorize credential provisioning, AWS permission changes, storage rollout
or campaign compute.

The following optional launcher fragment illustrates credential delivery to a separate review process; it is not required for same-agent review.
It is not the helper's full validation implementation. On another host, verify account and
host selection explicitly before use. It does not change the shared writer's login:

```python
env = os.environ.copy()
env.pop("GH_TOKEN", None)
env.pop("GITHUB_TOKEN", None)
token = subprocess.check_output(
    ["gh", "auth", "token", "--hostname", "github.com", "--user", "neverhuman"],
    env=env, text=True,
).strip()
env["GH_TOKEN"] = token
# Launch the separate reviewer with env=env; never print env or token.
# The reviewer first verifies: gh api user --jq .login
```

This is a launcher fragment (`os` and `subprocess` imports required), not a self-approval
command. Keep its prompt and redacted review receipt in custody outside the checkout.
Verify the parent still authenticates as the writer afterward. For runner host access,
use the [runner runbook](docs/runbooks/runner.md): this Ubuntu `xbabe2` is local (`sudo -n`),
whereas this user's SSH alias `xbabe2` points to `agent@192.168.68.89` using
`/home/ubuntu/.ssh/xbabe2_ed25519`. Do not confuse that alias with the local CPU runner host.

### S3 buckets and how to reach them

The buckets listed below were inventoried from metadata (current object versions, no object
bodies) in AWS account `714044927654` on 2026-09-23. Sizes are decimal (1 GB = 10^9 B).
Per-prefix counts, prefix status (canonical, staging or historical) and the inventory hashes
are in [`catalogs/s3.json`](catalogs/s3.json). The named profile aliases resolved to one
credential on 2026-09-22; this is not a permanent access guarantee. Local custody and
training state also exist.

| Bucket | What it holds (2026-09-23) |
| --- | --- |
| `s3://veox-jopedime-datasets-use2` | The dataset corpus. `JOPE_DATASETS_URL` defaults here (`crates/jope-datasets/src/lib.rs`). Content-addressed: `blobs/ catalog/ collections/ datasets/`. 15,442 objects, 3.23 GB. It holds only the T1 real tier (3,859 datasets); T2 and T3 are not uploaded. No replica. |
| `s3://veox-jopedime-learning-use2` | The learning store, `JOPE_LEARNING_URL` (no default; `jope-learning` mirrors only when `--store` or `JOPE_LEARNING_URL` is set). It holds **only `staging/`**: 645,281 objects, 430.65 GB of second copies (`lineage-a/`, `lineage-b/accepted60000-restored/`, `mirror-weights/`). [`docs/learning.md`](docs/learning.md) declares part of lineage A unrecoverable. `staging/` is not a canonical prefix and must not be promoted as one. |
| `s3://veox-jopedime-models-use2` | Model bundles (`crates/jope-models`; `JOPE_MODELS_URL` defaults here): `blobs/` and `bundles/` (3 bundles), plus the `staging/` source copies. 967 objects, 3.93 GB. |
| `s3://veox-jopedime-archive-use2` | Git bundles (`git/`: 5 objects, 22.17 GB; `git-bundles/20260921/`: 3 objects) and the `staging/node-high-value/` second copy (180,539 objects, 10.88 GB). 180,547 objects, 33.08 GB in total. There is no `evidence/` or `derived/` prefix yet. No replica. |
| `s3://veox-jopedime-models-usw2` | Replica target of `models-use2` (rule `models-all`). On 2026-09-23 its object counts and bytes matched at inventoried prefix depths 1–3; this does not prove payload identity or tested recovery. |
| `s3://veox-jopedime-learning-usw2` | Replica target for `learning-use2`'s `hwm/`, `waves/`, `checkpoints/serving/` and `training-head` only. The inventory found it **empty**, with no source objects under those prefixes. The configured rules do not cover `staging/`; other copies were not exhaustively inventoried. |
| `s3://jopedime-714044927654-us-east-2-{dev,prod,sealed}` | Preservation custody, a separate family in the same account: dev 186,305 objects / 118.12 GB, prod 186,304 / 117.85 GB, sealed 8,679 / 246.06 GB. These are not the configured product corpus buckets; do not substitute them for the defaults. They have no Object Lock configuration and no replication. |

The six inventoried `veox-jopedime-*` buckets had versioning and Object Lock enabled with
**no default retention**. Per-object retention and legal holds were not read. The
`veox-jopedime-datasets-usw2` HeadBucket probe returned HTTP 404 on 2026-09-23.
In this inventory, “no replica” means no configured bucket replication destination was found;
it does not establish the absence of manually preserved copies elsewhere.

Relevant environment: `JOPE_DATASETS_URL`, `JOPE_LEARNING_URL`, `JOPE_DATASETS_ANON=1` (unsigned
requests, for a public bucket), and `JOPE_AWS_PROFILE` / `JOPE_AWS_ALLOW_IMDS`.

**The pre-PR63 failure mode.** The unadapted `object_store` builder supports native
environment, OIDC and container providers, then falls back to EC2 instance metadata. It never executes `credential_process`,
which is how every profile on this host supplies credentials, and the `[default]` profile has
none. With only a profile selected you therefore get a ~13 s wait and an error naming
`169.254.169.254` — which reads as a network fault and is in fact "no credentials":

```text
error: get datasets/<id>/manifest.json: Generic S3 error: Error performing
PUT http://169.254.169.254/latest/api/token in 12.9s, after 10 retries
```

With PR63 present, set `AWS_PROFILE=veox-jepson-prod` (`crates/jope-datasets/src/credentials.rs` resolves it through
the AWS CLI and fails by name when it cannot), or put credentials in the environment yourself with
`eval "$(aws configure export-credentials --format env)"`. On this host that yields a long-lived
static key pair with no session token, so do not write it to disk, bake it into an image or hand
it to a CI job.

**Runner containers have no AWS credentials**, and a long-lived writer credential must not be
given to one. Corpus-dependent work runs on the host until a scoped read-only role exists.

See [`docs/datasets.md`](docs/datasets.md), [`docs/learning.md`](docs/learning.md),
[`catalogs/training-artifacts.json`](catalogs/training-artifacts.json),
[`docs/runbooks/training-artifact-locations.md`](docs/runbooks/training-artifact-locations.md),
[`docs/runbooks/ingest-winners.md`](docs/runbooks/ingest-winners.md) and
[`docs/runbooks/rotate-credentials.md`](docs/runbooks/rotate-credentials.md) for the details;
they are not repeated here.

## What counts as verification

Bind evidence to the source commit, command and exit status, execution time, relevant environment,
input identities and resource budget. Record tests collected, executed, failed, skipped or excluded
and why. Missing runners, cancelled jobs, zero collected tests and unavailable required capabilities
are not passing coverage. Derive coverage totals from inventories and run receipts.

Status records distinguish source implementation, fixture/local proof, hosted CI, live operational
qualification and serving promotion. Preserve the remaining requirement for partial behavior.
Do not mark a whole capability complete from a passing subset, a checkpoint file, or an author's
summary. Training evidence includes actual optimizer updates and changed parameters; a target
improvement needs replay under the same HWM comparison identity (`task`, `dataset_hash`, `cut_id`,
evaluator, vocabulary and protocol). Resume preserves the
declared resource limits and deadline, and unknown outcomes remain unknown rather than zero.

## Change discipline

Use focused commits, preserve reproducibility, and do not rewrite shared branches. Update private feature branches onto current `origin/main` before handoff. For shared
PR branches, use a history-preserving update; never force-push or rewrite shared history. If a generated artifact is stale, delete or regenerate it
only through its owning proof lane.

## Large data

Bulk data does not live inside a git working tree — git does not protect untracked files, they can
never be committed at that size, and their presence makes every `git status` and every
untracked-file rescue walk them.

## Custody, maintenance and lane closure

One writer per checkout. Use the canonical checkout or a claimed, registered
worktree under `<repo>/.agent/worktrees/`; no sibling clones or tool-specific roots.
Preserve `.claude/worktrees` as the symlink to that shared root. At most four managed
lanes per repository, each with one reused build directory. No clones, worktrees
or build output under `/tmp`. Bulk data stays outside working trees.

When a lane's PR closes, preserve unique source and evidence first, verify recovery,
then remove the lane with `git worktree remove`. Reuse one independent **bare**
verification directory under custody outside working trees, populated with
`git clone --no-local --bare` or equivalent independent bundle restoration. Check
exact commits, trees and connectivity; no alternates or dependency on the original.
Remove verified generated restoration contents after each batch; keep immutable
archives and receipts. A valid checksum, a local commit and a merged change are
three different states.

The daily maintenance policy reports candidates older than 48 hours at **04:00 UTC**,
with `Persistent=true`; it never deletes. `cargo-build-dirs.sh` rejects `--delete`.
Cargo markers (`.rustc_info.json`, `debug/release/.fingerprint`) identify candidates,
not disposable contents. Neither `target` nor generic `CACHEDIR.TAG` identifies
Cargo output. Evidence may occur at any depth, including `debug`, `release`, and
`tmp`. Preserve and independently verify all relevant content and recheck active
consumers before an explicit logged removal. Unknown content is retained. See
`/home/ubuntu/JopeDime/deploy/maintenance/README.md` for installed reporting behavior.

Before running Cargo, verify or explicitly set `CARGO_INCREMENTAL=0` and
`CARGO_PROFILE_DEV_DEBUG=line-tables-only`. Claude's settings retain these values;
other agents may not inherit them. Keep `CARGO_TARGET_DIR` unset globally. Preserve
JopeDime CI's job-scoped target selection and separate PR/trusted-main storage domains.

## One-time owner-directed emergency integration — 2026-10-23

The owner approved consolidating PRs #416, #420–#426 and then instructed completion
after reviewing the merge-first plan. For this integration only, expensive native
and manual matrix validation is deferred until after merge. Main is UNVERIFIED
until actual merged-source manual CI and repairs complete. This is an explicit
exception to the usual premerge full-run requirement, not evidence of passing CI.
Short format, locked workspace/all-target compilation, actionlint, source
preservation and exact-head substantive review remain admission requirements.
The integrator may temporarily remove only required status checks in BOTH main
protection layers for this one exact-head PR merge and must immediately restore
and read back both complete policies, including on failure. PR/discussion/linear
history/force-push/deletion controls remain. Preserve original branches, commits,
authors and scientific fixtures. Ordinary PR rules resume after this merge.
Automatic test/build triggers are paused; maintained jobs remain manually callable.
CUDA qualification remains local/manual only. Runtime deployment requires actual
merged source and retains separate qualification gates. Signed release publication
waits for validation and remains assigned to the existing release owner.
See [the source inventory](docs/internal/ci/emergency-integration-20261023.json).
