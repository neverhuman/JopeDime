# W12 claims inventory and public conformance pack (W12-T01, W12-T07)

Evidence record for the W12-T01 sentence inventory and the W12-T07 conformance pack. It
qualifies nothing by itself. Runs are under
`/home/ubuntu/jope-rescue-20260921/regression-remediation-20260926/claude-w0-w12/runs/`.

## 1. W12-T01: the sentence inventory is generated from the verified claims (branch `rr2/integration-c`, 2026-09-26)

**New capability; there is no failing reproduction.** The claim schema, its binding rules and
the stage/kind separation already existed (commits `9e31ad12e3d7`, `c658d14891d3`). The gap was
that no sentence inventory was generated from `claims.json`.

- `jope-records claim-inventory [--claims PATH] [--write]` (module `claim_inventory`) first
  verifies the claims record with `claims::verify`. It then renders
  `docs/internal/core-ip/claims/INVENTORY.md`. Each claim sentence appears with every binding
  the schema requires:
  - subject, metric, units, denominator, comparison and baseline;
  - input scope, source commit, observation time and uncertainty;
  - its evidence nodes with their kinds, and its limitations.
- Claims are grouped as publishable (`active`) or withdrawn (revoked or superseded), with a
  stage-by-status count table.
- Without `--write` it refuses a stale committed view ("INVENTORY.md is stale; regenerate with
  --write"), as `task-status` and `progress-view` do. The fast lane (`deploy/ci/fast-lane.sh`
  and its `coverage.json` job command list) runs the check next to `verify-claims`.
- Acceptance ("altered scope/source/subject/denominator cannot inherit"):
  - The inventory is re-rendered from the record's fields alone.
  - A claim whose scope or source no longer matches its evidence is refused by the receiving
    verifier, so nothing renders.
  - A changed denominator renders as a different sentence and never as the old one.
- Tests, `jope-records` `tests/records/claim_inventory.rs` (4, in the existing `records` target):
  - `the_committed_inventory_is_generated_from_the_verified_claims`
  - `every_binding_is_rendered_with_the_sentence` (includes pipe escaping and deterministic
    bytes)
  - `changed_bindings_do_not_inherit_and_withdrawn_claims_are_not_publishable`
  - `a_stale_inventory_is_refused_and_rewritten_only_on_request`
- **Open.** The inventory holds the 2 committed claims. The monograph's capability and KPI
  sentences become inventory entries only once each is recorded as a claim with its evidence.
  Until then an unrecorded sentence has no inventory entry and is not publishable under this
  rule. Recording them is content work for the claim owners.
- **Runs.**
  - Post `w12c-t01-post-20260926T212704Z`: `jope-records` 88/88; `verify-claims`, `claim-inventory`, `task-status` and `progress-view` exit 0.
  - Pre-commit `w12c-t01-c-20260926T212726Z`: fence suites and `gp::fit_contract_tests` 85/85; workspace all-targets check, coverage (0 unassigned), verify-test-inventory, fmt, diff-check and `parity-matrix.sh --no-build` exit 0.

## 2. W12-T07: the public synthetic conformance pack and the private-canary scan (branch `rr2/integration-c`, 2026-09-26)

**New capability; there is no failing reproduction.** The receiving rule already existed:
evidence of synthetic identity cannot back a hosted, live, measurement, promotion or real-corpus
claim (`c658d14891d3`). The fixture pack and the scan did not exist.

- **Pack.** `docs/internal/core-ip/conformance/` has 25 files: 12 fixtures of the
  claim-to-evidence boundary plus `index.json`.
  - There is one fixture per refusal class, and each class is refused with the validator's own
    message:
    - synthetic identity as live-host, real-corpus, measurement or decision evidence;
    - a live claim on fixture evidence;
    - changed evidence bytes;
    - an evidence path outside the repository;
    - claim scope drifting from its evidence;
    - revoked evidence still cited by an active claim;
    - an incomplete binding;
    - an unknown evidence kind.
  - It also has an accepted fixture-stage control.
  - `jope-records conformance` (module `conformance`) generates the pack deterministically.
    Without `--write` it requires the committed bytes to equal the generator's, so the public
    replay is reproducible and a stray, changed or missing file is refused. `--write` never
    deletes a file.
- **Synthetic identity.** Every pack file carries `synthetic_identity:
  jope.core-ip-conformance/v1` and lives in the conformance tree. The claims verifier therefore
  treats each fixture as synthetic by both location and bytes.
- **Same receiving validator.** `replay` runs every fixture through `claims::verify`, the
  verifier of the committed `claims.json`. It requires the recorded outcome: acceptance, or a
  refusal of the rule class whose message contains the recorded text. A refusal for another
  reason fails.
- **Private-canary absence scan.** `scan` refuses any pack file without the synthetic identity,
  or with a private canary:
  - AWS access-key and secret-name shapes, private-key blocks and GitHub tokens;
  - private home paths, the custody roots (`jope-rescue`, `jope-evidence`) and the private
    hosts;
  - `s3://` URIs.
  The fast lane runs `conformance` (generator check, scan and replay) next to
  `claim-inventory`.
- **Tests** in `jope-records` `tests/records/conformance.rs` (4, in the existing `records`
  target):
  - `the_committed_pack_is_generated_and_replays_through_the_claims_validator`: 1 accepted,
    11 refused, 11 distinct classes;
  - `a_fixture_with_another_outcome_than_recorded_fails_the_replay`: both directions, plus the
    drift check;
  - `a_refusal_for_another_reason_fails_the_replay`;
  - `private_canaries_and_unmarked_files_are_refused`: 6 canary classes, an unmarked stray
    file, and `--write` not deleting it.
- **Scope and what stays open.**
  - The pack covers the claims boundary only. The dispositions, fitted-package,
    proposal-ledger and lineage boundaries keep their synthetic cases in their own crates'
    tests. A dispositions fixture would have to bind the live findings register. It would go
    stale with every register change, so it was not built.
  - Synthetic identity keeps the pack from establishing any real-corpus, hardware, live,
    historical or operational claim.
  - Replay of the private evidence package is W12-T06 and is not part of this pack.
- **Runs.**
  - Post `w12c-t07-post-20260926T213004Z`: `jope-records` 92/92.
  - The fast-lane commands `verify-claims`, `claim-inventory`, `task-status`, `progress-view` and `conformance` exit 0. `conformance` scanned 25 files, and of its 12 fixtures 1 was accepted and 11 refused.
  - The hygiene home-path grep over `crates/*/{src,tests}` found nothing.
  - Pre-commit `w12c-t07-c-20260926T213031Z`: fence suites and `gp::fit_contract_tests` 85/85; workspace all-targets check, coverage (0 unassigned), verify-test-inventory, fmt, diff-check and `parity-matrix.sh --no-build` exit 0. `gitleaks detect --no-git` found nothing in `crates/jope-records`, the conformance and claims trees, or the other directories this branch changed.
