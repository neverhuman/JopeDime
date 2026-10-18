# Algorithmic Self-Play Integration PR Acceptance & Gate Matrix (`I63`)

## 1. Purpose and Authoritative Completion Rule

This document is the reviewer-facing acceptance and gate matrix (`I63`) for the single integration pull request (`PR #358`, branch `feat/selfplay-spec-integration-20261018` targeting `main` on `neverhuman/JopeDime`) rooted at merged `I01` base commit `1a8a66c502ca2cede77c197b4f5b5bbc24ff398e`.

- **Authoritative Completion Rule (`I63`)**: *"One PR, exact-head source tests and genuine independent review; receiving evidence before any completion claim."*
- **Continuing Lifecycle Obligation**: Authoring `PR_ACCEPTANCE.md` establishes the bounded decision contract, ordered commit ledger, and pre-merge/post-merge verification matrix; final `I63` closure requires exact-head CI, genuine independent review posted by the independent reviewer process, rebase merge (`--rebase --match-head-commit`), and receiving-`main` verification (`main-guard.yml` + `full-main.yml`).
- **Immutable v4 Planning Packet Pin**: All 39 frozen v4 planning payload files (`docs/plans/algorithmic-selfplay-20261017/PACKET_MANIFEST.v4.json`, SHA-256 `921643bc3ae0b7d75bbab13434f4d8211bc3c8638154c45ac00fc7df184750d5`) remain byte-for-byte untouched. Programme-wide 66-task definitions (`I01`..`I66`) and validation matrix rows live in `docs/plans/algorithmic-selfplay-20261017/VALIDATION_MATRIX.md` and `tasks.json`.

---

## 2. State Taxonomy (No State Conflation)

Every task and gate in the integration PR distinguishes five orthogonal qualification states:

| State Class | Meaning | What Qualifies It | What Cannot Qualify It |
| --- | --- | --- | --- |
| `PENDING` / `IN_PROGRESS` | Implementation, exact-head verification, or independent review is still underway. | Claimed worktree/branch + active lease in `COORD.md` / `AGENT_CHAT.md`. | Never counted as accepted or verified. |
| `FIXTURE_VERIFIED` | Scoped unit/integration/doctest/conformance fixtures pass (`-- --list` then `-- --test-threads=1`) with positive and negative witnesses at the exact commit SHA. | Durable receipts under `/home/ubuntu/jope-evidence/spec-integration-20261017/work-packets/evidence/<TASK>/<FULL_SHA>/`. | Does not establish live S3 durability, GPU numerical qualification, serving readiness, or scientific improvement. |
| `ENGINEERING_INTEGRATED` | Ordered commit landed on the single integration branch (`feat/selfplay-spec-integration-20261018`), serialized coverage/inventory/identity gates pass (`/usr/bin/just required`), exact-head CI (`ci.yml`) is green, and an independent reviewer has approved the exact head SHA. | Native GitHub `APPROVE` bound to `commit_id: <FULL_SHA>` + full-attempt `ci.yml` receipt. | Does not complete receiving-`main` closure until merged and verified on `main`. |
| `MERGED_AND_RECEIVING_MAIN_VERIFIED` | Merged into `main` via `--rebase --match-head-commit <FULL_SHA>` with receiving `main-guard.yml` and `full-main.yml` (`cpu-1`, `cpu-2`, `aggregate`) green at the resulting `main` commit. | Original merge command/API receipt + receiving `main-guard` and `full-main` API readbacks + custody bundle restored in `verify-bare`. | Does not grant `training_authorized=true` or claim scientific/serving promotion. |
| `UNAVAILABLE_BY_AUTHORITY` / `SKIPPED_BY_PATH` | Gated by held GPU/training authority (`training_authorized=false`, `require_qualified_gpu_experiment`), non-default Cargo feature (`neural`, `testing`), or CI path filter (`skipped-by-path`). | Explicit disclosure of the gate, feature, or path exclusion. | Never labeled `PASS` or `VERIFIED`. |

---

## 3. Historical Baseline & Gate Dispositions (`PR #356` and `PR #357` / `I01`)

### 3.1 `PR #356` (Immutable v4 Planning Packet Publication)

- **Merged Commit on `main`**: `22939ca18a088a1e75eb9270fa54da9a87083440`
- **Receiving `main` Verification**:
  - `main-guard.yml`: run `39084695817` (`completed` / `success`)
  - `full-main.yml`: run `39084695810` (`completed` / `success`)
- **Render & Whitespace Honesty (`I63` Cases 4 & 5)**:
  - `VALIDATION_MATRIX.md` renders all **66** distinct task rows (`I01`..`I66`) in GitHub Flavored Markdown with real table newlines (zero literal `\n` row separators); verified in `docs/plans/algorithmic-selfplay-20261017/RENDER_CHECK.json`.
  - Unscoped `git diff --check` across the `PR #356` planning diff exited `2` due to `23` trailing double-space markdown hardbreaks preserved inside the byte-identical imported reference specifications (`docs/plans/algorithmic-selfplay-20261017/WHITESPACE_EXCEPTION.json`), while the authored-doc-only check exited `0`. On `feat/selfplay-spec-integration-20261018` (`origin/main..HEAD`), those reference files are unchanged, so `git diff --check origin/main..HEAD` must exit `0` with zero new whitespace warnings.

### 3.2 `PR #357` (`I01` — Pin one executable baseline and one immutable v4 packet)

- **Reviewed PR Head**: `407c4248601d6e54145fec581093e0fce59ff084` (`5` ordered commits on `feat/selfplay-i01-executable-baseline`)
- **Merged `main` Commit**: `1a8a66c502ca2cede77c197b4f5b5bbc24ff398e` (parent `22939ca18a088a1e75eb9270fa54da9a87083440`, Git tree `1c6d5882718434ae6f109854bff4597896f1210f` identical to `407c4248601d6e54145fec581093e0fce59ff084^{tree}`)
- **Exact-Head CI at `407c4248601d6e54145fec581093e0fce59ff084`**:
  - `pull_request` `ci.yml`: run `39087080072` (attempt `1`, synthetic merge SHA `f6a377431637e401ee1c8d726da9382d4bf64eb3`, `completed` / `success`)
  - `workflow_dispatch` `ci.yml`: run `39087091490` (attempt `1`, exact source SHA `407c4248601d6e54145fec581093e0fce59ff084`, all scheduled jobs + `aggregate` `completed` / `success`)
- **Explicit Gate Dispositions (`G1`, `G2`, `G3`)**:
  - **`G1` (Squash Merge Recorded Exception)**: `PR #357` was merged with `/home/ubuntu/.local/bin/gh-role writer pr merge 357 --repo neverhuman/JopeDime --squash --match-head-commit 407c4248601d6e54145fec581093e0fce59ff084` per the 01:57Z prompt instruction before the 02:27Z owner correction restored `--rebase --match-head-commit` as the sole authorized merge method. Recorded in `/home/ubuntu/jope-evidence/spec-integration-20261017/i01-executable-baseline/pr357-merge-method-exception-receipt.json`; all 5 pre-squash commits and reviewed head `407c4248601d6e54145fec581093e0fce59ff084` are preserved in bundle `i01-pr357-407c4248.bundle` (SHA-256 `d33e521f038eef2a4dccadc45d83ad6cbed4e2f696a5b7c1b5837965d93f3f36`) and restored in `/home/ubuntu/jope-evidence/spec-integration-20261017/verify-bare` under `refs/custody/pr357/*`. Every subsequent merge (`PR #358` onward) MUST use `--rebase --match-head-commit`.
  - **`G2` (Native Approval Process Provenance Exception on `PR #357`, Binding Rule for `PR #358`)**: On `PR #357`, the independent Claude reviewer (`2269522.jope_review_claude`) wrote its `APPROVE` verdict for `407c4248601d6e54145fec581093e0fce59ff084` to `/home/ubuntu/jope-evidence/spec-integration-20261017/reviews/i01-claude/review-407c4248.{json,md}` (`no GitHub posting` per its launch brief), and the coordinator session submitted native GitHub `APPROVE` review `4132833670` (`neverhumanbot`) following the 01:57Z prompt. That split is recorded as a historical procedural exception on `PR #357` and does **not** satisfy `I63` Case 6 for future PRs. For `PR #358`, the independent Claude reviewer process itself must submit the native GitHub `APPROVE` review bound to `commit_id: <FULL_SHA>` and record the returned GitHub `review_id` in its own review receipt so the GitHub API readback matches the reviewer process's receipt directly.
  - **`G3` (Receiving-`main` Closure Verified)**: Receiving `main` (`1a8a66c502ca2cede77c197b4f5b5bbc24ff398e`) passed `main-guard.yml` run `39089266033` (attempt `1`, `success`) and `full-main.yml` run `39089266014` (attempt `1`, `success` across `rust-test` `cpu-1`/`cpu-2`, `determinism`, and `aggregate`), recorded in `/home/ubuntu/jope-evidence/spec-integration-20261017/i01-executable-baseline/{pr357-lifecycle-receipt.json,receiving-full-main-cpu-39089266014.json}`.

---

## 4. Active Single Integration PR Ledger (`PR #358`)

- **Repository**: `neverhuman/JopeDime`
- **Pull Request**: `PR #358` (`https://github.com/neverhuman/JopeDime/pull/358`)
- **Integration Branch**: `feat/selfplay-spec-integration-20261018`
- **Base Commit (`origin/main`)**: `1a8a66c502ca2cede77c197b4f5b5bbc24ff398e`
- **Coordinator / Integrator Lane**: `2830191.jopedime_agy` (operating under Captain `2724936.captain_jopedime`)

### Ordered Task & Commit Matrix on `feat/selfplay-spec-integration-20261018`

| Order | Task ID | Title | Direct DAG Deps | Status | Commit / Lane | Primary Owned Paths & Acceptance Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | `I01` | Pin one executable baseline and one immutable v4 packet | None | `MERGED_AND_RECEIVING_MAIN_VERIFIED` | `1a8a66c502ca2cede77c197b4f5b5bbc24ff398e` (`PR #357`) | `crates/jope-records/src/selfplay_baseline.rs`, `docs/plans/algorithmic-selfplay-20261017/BASELINE.{md,v1.json}`. Evidence: `/home/ubuntu/jope-evidence/spec-integration-20261017/i01-executable-baseline/`. |
| 1 | `I02` | Freeze one additive wire-contract catalogue | `I01` | `FIXTURE_VERIFIED` (committed on `PR #358`) | `0c06b14539a9a700091d046eef14f1c90f773482` (`2830191.jopedime_agy`) | `schemas/jope.canonical-json.v1.schema.json`, `crates/jope-records/src/wire_contract.rs`, `crates/jope-records/tests/wire_contract.rs`, `crates/jope-records/tests/fixtures/wire-contract-v1/*`, `docs/contracts/algorithmic-selfplay-identity.md`. Zero `Cargo.lock` changes (`compat_freeze` `13/13` pass). Evidence: `/home/ubuntu/jope-evidence/spec-integration-20261017/work-packets/evidence/I02/0c06b14539a9a700091d046eef14f1c90f773482/`. |
| 2 | `I46` | Verify immutable publication by content, not size | `I02` | `FIXTURE_VERIFIED` (committed on `PR #358`) | `a4efb4c79ebd373e289952d2a39a14af1f33a901` (`2830191.jopedime_agy`) | `crates/jope-datasets/src/{verified_object.rs,store.rs,lib.rs}`, `crates/jope-models/src/{publish.rs,cli.rs}`, `crates/jope-learning/src/mirror.rs`, `crates/jope-datasets/tests/verified_object.rs`, `crates/jope-models/tests/{publication_integrity.rs,roundtrip.rs}`. Pre-fix reproducer + post-fix receipts under `/home/ubuntu/jope-evidence/spec-integration-20261017/work-packets/evidence/I46/a4efb4c79ebd373e289952d2a39a14af1f33a901/`. |
| 3 | `I63` | Make the one integration PR reviewable and continuously current | `I01`, `I02` | `IN_PROGRESS` (continuing lifecycle gate on `PR #358`) | `feat/selfplay-spec-integration-20261018` (`2830191.jopedime_agy`) | `docs/plans/algorithmic-selfplay-20261017/PR_ACCEPTANCE.md` + durable lifecycle receipts under `/home/ubuntu/jope-evidence/spec-integration-20261017/work-packets/evidence/I63/`. |
| 4 | `I03` | Remove unavailable-score and nested-CV ambiguity | `I02` | `IN_PROGRESS` (worker lane `jope_i03_codex`) | `lane/selfplay-i03-20261018` -> integrates onto `PR #358` | `crates/jope-learning/src/pools/{mod.rs,fold_safe_eval.rs}`, `crates/jope-learning/tests/fold_safe_eval.rs`, `crates/jope-learning/tests/fixtures/fold-safe-eval-v1/*` (no `crates/feat-math/src/model.rs` edit; preserves evaluator & lineage identities). |
| 5 | `I07` | Generate one capability and operator registry | `I02` | `IN_PROGRESS` (worker lane `jope_i07_codex`) | `lane/selfplay-i07-20261018` -> integrates onto `PR #358` | `crates/feat-math/assets/operator_registry.v1.toml`, `crates/feat-math/src/kernel/registry_v2/{mod.rs,generated.rs}`, `crates/feat-math/tests/registry_agreement.rs` (`crates/feat-math/src/registry.rs` untouched per `FROZEN_FILES`). |

---

## 5. Seven Identity and Guard Surfaces

Every commit integrated onto `feat/selfplay-spec-integration-20261018` is checked against all seven repository guard surfaces before push:

| Surface | Authoritative Source Anchor (`1a8a66c502ca2cede77c197b4f5b5bbc24ff398e`) | Enforcement Rule |
| --- | --- | --- |
| 1. ADR-0004 Evaluator Identity | `crates/feat-math/tests/fixtures/compat_freeze/evaluator_identity.v1.json:9` | Any change to listed `feat-math` sources or `Cargo.lock` changes the evaluator identity digest. `I02`, `I46`, `I63`, and `I03` make zero `Cargo.lock` or evaluator-hashed file edits (`compat_freeze` and `historical_v1` run in compare mode). Planned transitions (`I07` -> `I09` -> `I10`) require task-named transition fixtures, `historical_v1.rs` assertions, and ADR-0004 known-table updates serialized by `2830191.jopedime_agy`. |
| 2. Teaching / Corpus / Eval-Query Lineage | `crates/feat-math/tests/fixtures/compat_freeze/lineage_sources.v1.json:4` | `crates/feat-math/src/model.rs` and `crates/jope-engine/src/task_contract.rs` are not edited in `I02`, `I03`, `I07`, `I46`, or `I63`. |
| 3. `FROZEN_FILES` | `crates/feat-math/tests/compat_freeze.rs:2705` | All files in `FROZEN_FILES` (including `crates/feat-math/src/registry.rs` at line `2712` and emitted pins at lines `2715-2725`) are strictly read-only. |
| 4. Witness / Authority Digest | `crates/jope-engine/src/eval_registry/witness.rs:122,143` | Untouched until `I11` (which requires explicit owner-signed opt-in and retains legacy `60000`/`6000` defaults when absent). |
| 5. Clover GP Audit | `crates/jope-learning/src/clover/expansion/config.rs:37,61-145` | The 12 audited files (`gp.rs`, `grammar.rs`, `tournament.rs`, `manifest.rs`, `selection_replay.rs`, `authority.rs`, `featmath.json`, `search.rs`, `islands.rs`, `seeds.rs`, `evolve.rs`, `Cargo.lock`) must match `HEAD` byte-for-byte when `jope-learning --lib` runs and preserve all required search/authority literals. |
| 6. Retention Directory & Content Guard | `crates/jope-learning/tests/retention.rs:918,933-945` | Exact `crates/jope-learning/src/retention/*.rs` file roster and banned-substring checks remain strict and untouched. |
| 7. Coverage & Test Inventory Discovery | `crates/jope-records/src/coverage.rs:440,523` | Every new integration test target or example is registered by `2830191.jopedime_agy` in `deploy/ci/coverage.json`, `docs/internal/core-ip/findings/test-inventory.json`, `docs/parity/test-inventory.json`, and `README.md`, and verified via `coverage validate`, `verify-test-inventory`, `paper-trace`, and `repo-layout`. |

---

## 6. Profile Registry and Runtime Test-Feature Reconciliation (`F1`, `F2`, `F3`)

1. **Profile Registry (`F1`, `crates/feat-math/src/kernel/profile.rs:727-748`)**:
   - Registered `(id, revision)` pairs (`8` total): `legacy-wrappers-v1` (rev `1`), `legacy-wrappers-v2` (rev `1`), `kernel2-baseline` (revs `1, 2, 3`), `kernel2-experimental` (revs `1, 2, 3`).
   - Shipped `kernel2` wrapper set (`profile.rs:748`): `kernel2-baseline` (rev `3`) and `kernel2-experimental` (rev `3`).
   - Product CLI default spelling is `legacy` (`crates/jope-pipeline/src/kernel2/messages.rs:4`); formal binding from CLI `legacy` to `legacy-wrappers-v1` vs `legacy-wrappers-v2` is not assumed without explicit code binding.
2. **Policy Test Feature Gates (`F2`, `crates/jope-policy/Cargo.toml:39,43`)**:
   - `challenger_tests` (`crates/jope-policy/src/lib.rs:114`) is a default `cfg(test)` library module (`cargo test --locked -p jope-policy --lib challenger_tests`).
   - Integration targets `jope_dime` and `decode_outcomes` require `--features testing` (`cargo test --locked -p jope-policy --features testing --test jope_dime` and `--test decode_outcomes`).
3. **Neural Gates, Held GPU Routes, and `#[ignore]` Census (`F3`)**:
   - `18` whole-file `#![cfg(feature = "neural")]` test files plus `3` partially gated test files (`crates/jope-engine/tests/{corpus.rs:29,request_integrity.rs:204,training_inputs.rs:184}`) require `--features neural` and an admitted LibTorch environment; default-feature CI does not execute them.
   - `6` held numerical routes call `require_qualified_gpu_experiment` (`ranked_proposals.rs:739`, `neural.rs:1262,1392`, `controller.rs:913,1045`, `training.rs:600`), delegating to `task_contract.rs:307` (`training_authorized=false` remains enforced).
   - Ignored test reconciliation at `1a8a66c502ca2cede77c197b4f5b5bbc24ff398e`: `docs/parity/test-inventory.json` records `72` default-feature runtime ignored tests, while static `git grep` of anchored `#[ignore]` attributes finds `86` source attributes (`72` default-feature matched + `3` `asm` + `5` `neural` + `5` `testing` + `1` `not(debug_assertions)`).

---

## 7. Decision Table: Required Positive Proofs and Refusal Cases (`I63`)

| Acceptance Obligation | Required Positive Proof | Concrete Refusal Case (Must Fail Closed) |
| --- | --- | --- |
| **1. One current integration lane** | Fresh PR/claim/head receipt (`PR #358` on `feat/selfplay-spec-integration-20261018`); disjoint worker patches (`I03`, `I07`, `I46`, `I63`) serialize shared hooks through `2830191.jopedime_agy`. | Pushing to a closed/merged PR (`#357`), opening a second competing/stacked PR, or a post-push `headRefOid` mismatch against local `HEAD`. |
| **2. Current and exact source** | Full base (`1a8a66c502ca2cede77c197b4f5b5bbc24ff398e`), head SHA, tree SHA, and dependency receipts re-verified after every commit or rebase onto `main`. | Relying on a CI run or review approval from an ancestor commit after any new commit, amend, or rebase changes `HEAD`. |
| **3. Complete CI attempt** | Exact-head manual `ci.yml` (`workflow_dispatch`) run at frozen `HEAD` with all 10 current-attempt jobs (`changes`, `hygiene`, `rust-check`, both `rust-test` shards, `emitters`, `inputs`, `python`, `e2e-smoke`, `aggregate`) green, plus current `pull_request` CI run (`source_sha` and `synthetic_checkout_sha` recorded separately). | Mixing partial job reruns from an older attempt, missing a `rust-test` shard, zero tests collected on a required target, or labeling a `skipped-by-path` job as `PASS`. |
| **4. Genuine independent approval** | Complete paginated commit author/committer inventory (`gh api --paginate repos/neverhuman/JopeDime/pulls/358/commits`) showing the reviewer (`neverhumanbot`) is absent from all pages and not the PR opener (`jepsontaylor`), plus the independent Claude reviewer process's own receipt naming the exact native GitHub `APPROVE` `review_id` and `commit_id: <FULL_SHA>`. | Reviewer appearing on page $\ge 2$ of commit authors/committers, unadjudicated `UNASSOCIATED` commit author, coordinator-posted self-approval, `COMMENT`-only review, dismissed review, or approval on a stale SHA. |
| **5. Render and whitespace honesty** | `VALIDATION_MATRIX.md` GFM HTML render verified with all 66 rows (`I01`..`I66`) and zero literal `\n` row delimiters; `PACKET_MANIFEST.v4.json` SHA-256 unchanged (`921643bc3ae0b7d75bbab13434f4d8211bc3c8638154c45ac00fc7df184750d5`); `git diff --check origin/main..HEAD` exits `0`. | Any modification to the 39 frozen v4 packet files, literal `\n` table delimiters, or any new whitespace error in authored files on `origin/main..HEAD`. |
| **6. Original protected merge lifecycle** | Exact pre-merge readback of open state, head/base SHA, current-attempt CI, and native approval, followed by capturing the original `/home/ubuntu/.local/bin/gh-role writer pr merge 358 --repo neverhuman/JopeDime --rebase --match-head-commit <FULL_SHA>` argv, exit code, stdout/stderr, and API response. | Merging with `--squash`, `--merge`, or `--admin`; omitting `--match-head-commit`; or synthesizing a merge receipt after the fact from Git topology. |
| **7. Receiving `main` verified** | Post-merge verification of `merged_at`, resulting `origin/main` commit/tree/parents, and current-attempt `main-guard.yml` + `full-main.yml` (`cpu-1`, `cpu-2`, `aggregate`) `completed` / `success` receipts at the new `main` SHA. | Claiming delivery completion from PR CI alone while `main-guard` or `full-main-cpu` is pending, cancelled, or red, or resetting the protected dirty canonical checkout `/home/ubuntu/JopeDime`. |
| **8. Accurate completion accounting** | Only tasks with complete fixture receipts, `/usr/bin/just required` pass, exact-head CI, and independent review are marked engineering-accepted; `science`, `serving`, and `live_s3` remain `PENDING` / `UNVERIFIED` until their own authorized gates run. | Treating `PR_ACCEPTANCE.md` authorship alone as `I63` completion, or claiming live S3 / GPU training qualification from local CPU fixtures. |

---

## 8. Pre-Push and Pre-Merge Verification Commands (`PR #358`)

### 8.1 Scoped and Repository-Wide Local Gate (Inside Admitted Resource Envelope)

```sh
export CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only CARGO_BUILD_JOBS=4
unset CARGO_TARGET_DIR
cargo test --locked -p jope-records --lib selfplay_baseline::tests -- --list
cargo test --locked -p jope-records --lib selfplay_baseline::tests -- --test-threads=1
cargo test --locked -p jope-records --test wire_contract -- --list
cargo test --locked -p jope-records --test wire_contract -- --test-threads=1
cargo test --locked -p jope-datasets --test verified_object -- --list
cargo test --locked -p jope-datasets --test verified_object -- --test-threads=1
cargo test --locked -p jope-models --test publication_integrity -- --list
cargo test --locked -p jope-models --test publication_integrity -- --test-threads=1
/usr/bin/just required
git diff --check origin/main..HEAD
```

### 8.2 Exact-Head GitHub API Audit Commands (Coordinator Only)

```sh
/home/ubuntu/.local/bin/gh-role writer pr view 358 --repo neverhuman/JopeDime --json state,author,headRefOid,baseRefOid,headRefName
/home/ubuntu/.local/bin/gh-role writer api --paginate repos/neverhuman/JopeDime/pulls/358/commits \
  --jq '.[] | {sha, login: (.author.login // "UNASSOCIATED"), committer: (.committer.login // "UNASSOCIATED"), git_author: .commit.author}'
/home/ubuntu/.local/bin/gh-role eligible reviewer neverhuman/JopeDime 358
/home/ubuntu/.local/bin/gh-role writer api --paginate repos/neverhuman/JopeDime/pulls/358/reviews \
  --jq '.[] | {id, reviewer: .user.login, state, commit_id, submitted_at}'
```
