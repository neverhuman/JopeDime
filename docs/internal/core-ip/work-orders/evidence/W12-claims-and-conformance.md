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
