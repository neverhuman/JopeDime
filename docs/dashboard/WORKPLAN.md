# Live training observatory — implementation and acceptance plan

Owner: Codex chat `01a13a98-d6d0-7903-91a0-4a66046deff6`. Captain lease:
`/home/ubuntu/xbabe2_kicker/captains-20261017/CAPTAIN_TRAINING_DASHBOARD_DISJOINT_LEASE_20261019T1322Z.md`.
Checkout: `/home/ubuntu/JopeDime/.agent/worktrees/fleet-main-296d07e1`.
Branch: `feat/training-observatory-dashboard`. Initial base: `c956b163105f93cb8a2f589e402ed6655e1a1906`.

## Objective, boundaries and completion

Deliver a usable fleet-wide JopeDime dashboard, Rust API and Vite/TypeScript/React UI,
showing real training curves, storage, dataset census, results, code/checkpoint identities,
and a searchable GP library with source drill-down. Preserve unknown values and scientific limits.

Non-goals: change the optimizer, authorize training, promote checkpoints, adjudicate sealed
data, implement autonomous Admiralty execution, replace Admiralty, delete old evidence, or
add a paid monitoring service. A read-only dashboard has no dispatch/merge/promotion authority.

Dependencies: existing owned run receipts and catalogs; Node >=24; Rust >=1.90 for this crate;
workspace toolchain remains its separately declared version. AWS CLI/profile needed only
for an explicit optional S3 metadata refresh. Live fleet inputs require owner-approved snapshots.

Complete when API/data tests, frontend tests/typecheck/build and browser smoke pass with actual
run/library inputs; live URL and source/config/build identities are recorded; a single PR is
reviewed at its exact head under current repository rules and receiving-main validation is recorded.
Unknown fleet nodes, daily metric lag and incomplete archive publication must be disclosed.
Stop a particular integration on a conflicting ownership lease, linked/unbounded/unqualified
input, AWS account mismatch, unavailable authority, or unacceptable fleet pressure. Continue
disjoint implementation/tests. Never substitute synthetic live metrics or bypass a gate.

## Read first, in order

1. `/Users/bentaylor/.codex/RTK.md`; repository `AGENTS.md`, latest `AGENT_CHAT.md` under flock.
2. `MASTER_PLAN.md`, `docs/adr/ADR-0005-canonical-learning-loop.md`, `docs/learning.md`.
3. `docs/plans/major-release-20261018/MAJOR_RELEASE_TRACKER.json`, current workplan and receipts.
4. `catalogs/s3.json`, `catalogs/training-artifacts.json`, `docs/runbooks/training-artifact-locations.md`.
5. `docs/research/waves/INDEX.jsonl` and per-wave manifest/card/metrics/context contracts.
6. Existing `crates/jope-engine/src/dashboard.rs`, `dashboard_gp.rs`, native neural progress/checkpoint
   producer and original corpus admission/split limits. Existing engine paths remain read-only.
7. Fresh captain/Admiral rulings and written dashboard lease; reconcile PR383 and active campaign.

## Ownership and concurrency

Exclusive paths: `crates/jope-observatory/**`, `web/training-dashboard/**`, `docs/dashboard/**`;
narrow workspace/lock/ignore changes. Inventory registration only if required by repository hygiene.
No engine, feat-math, training, catalog, runbook, master-plan or Admiralty writes. Captain owns
campaign and source-review queue; existing Claude writer owns PR383 and research publication.
The main checkout, GP branch, paper tree, symlinks and ignored evidence are preserved.
Do not create another screen, clone or worktree. Reuse this lane's target; cap builds at four
jobs, nice10 and fleetcap. No GPU use. No parallel packet with overlapping checkout/index writes.

## Ordered work packets

1. **Data contract / Rust crate.** Add workspace member `crates/jope-observatory`.
   `Config`, `Observation`, `read_bounded`, `campaign`, `curve`, `storage`, `gp_catalog`, `waves`,
   `snapshot` in `src/lib.rs`. Inputs are explicit operator paths; no request-controlled paths.
   Enforce file/count bounds, refuse symlink components, hash compact inputs, use null for unknown.
   Check `progress.config_sha256 == SHA256(config bytes)` and known progress schema before any
   optimizer counter/curve is accepted. Check GP source hash before deduplication by program identity.
2. **HTTP delivery.** `src/main.rs`: loopback-only Axum listener, five-second shared sampler,
   GET `/api/v1/snapshot`, GET `/api/v1/gp`, GET `/healthz`, compiled static assets. Restrict Host,
   no cross-origin permission, no writes or arbitrary file downloads. Keep library rows out of
   repeated snapshot polls. Browser request count never starts AWS/SSH/compute jobs.
3. **S3 metadata.** `src/bin/collect-storage.rs`: explicit finite collector, expected-account check,
   configured catalog buckets only, bounded daily CloudWatch metadata, no object bodies or sealed
   data. Align storage types to a common daily timestamp per bucket; no missing-bucket zeroes.
   Retain captured-at versus measured-at; physical replicas/versions differ from unique payloads.
   Atomic output replacement. Do not install a new timer without coordination.
4. **React app.** Typed `src/types.ts`; `App.tsx` with overview, training, fleet, storage/datasets,
   GP library, results/lineage. SVG train/dev curves separate changed panels/objectives; no smoothed
   fabricated values. Progressive checkpoint budget combines epochs and step limit. Preserve last
   observation on fetch failure with visible stale banner/backoff. Search/filter/page GP rows and
   inspect original DSL/source/hash; never execute DSL. Responsive layout, keyboard controls,
   accessible status/error and dialog semantics. Catalog/producer claims visibly distinguished.
5. **Real connection.** External config and telemetry under owned evidence directory, not Git;
   current campaign roots read-only. Hash-pin actual library snapshot. Link approved fleet node
   samples, optional storage refresh, source identity receipt and current compiled UI. Do not alter
   active campaign outputs. Default loopback port18894 avoids Admiralty18890. SSH tunnel to view.
6. **Proof and delivery.** Run commands below, preserve original failures, capture API response
   SHA256/actual counts, browser desktop/mobile and drill-down evidence. Rebase private dashboard
   branch to current main after PR383; one PR, exact-head independent review and protected ordinary
   rebase merge through captain. No self-approval or forced history. Record receiving-main result
   and future refresh/deploy owner; attach created PR to this chat.

## Edge cases and validation

Missing/malformed/oversized JSON, linked directories, growing files, identity mismatch, future or
stale timestamps, absent loss history, changed objective/panel, empty/zero counts, incomplete
daily metrics, protected bucket denial, duplicate programs, stale dataset catalog, and unknown
fleet workload attribution must remain explicit. A finished epoch can still await full validation
or terminal/archive receipt. Current loss is not an HWM win or untouched holdout measurement.

```sh
CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only cargo test -p jope-observatory -j4
cargo fmt --all --check
CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only cargo clippy -p jope-observatory --all-targets -j4 -- -D warnings
cd web/training-dashboard
npm ci
npm run typecheck
npm test
npm run build
```

Run remote commands through `rtk` and fleetcap per host instructions. Use one existing target.
Also run `git diff --check`, applicable `jope-records verify-test-inventory`/hygiene, and exact-head
repository CI as required. A filtered, skipped or zero-test lane is not a complete pass.
HTTP probes: valid snapshot schema, real run/config match, GP unique counts; POST405;
unknown API404; bad Host403; linked/traversal source refusal; server remains responsive.
Browser: select run, inspect curve/checkpoint, search/family/page/inspect GP, open wave metric,
simulate fetch failure and verify last observation marked stale. Mobile width has no page overflow.

## Receipts and handoff

Local evidence: this chat's `dashboard-build-<UTC>/`; remote config/runtime/evidence outside Git
under `/home/ubuntu/jope-evidence/observatory/`. Each receipt records UTC, source full SHA,
dirty caveat, input hashes, commands, exit codes, counts, URL, binary/UI hashes and scientific
limits. Write `CODEX_DONE_...` or precise `CODEX_BLOCKED_...` in captain coordination directory,
with PR, stopped head, dirty paths, owner dispositions and next action. Never call the dashboard
or release fully complete before review/merge/deployment and remaining data feeds are qualified.
