# Jankurai vibe finding dispositions

## Scoped false positives

These are source-specific adjudications for `HLT-001-DEAD-MARKER`. Each
exception names one exact file in `agent/audit-policy.toml`; no directory or
broad glob is excluded. Recheck each disposition by 2027-02-01 and renew only
with fresh source and test evidence.

### Preregistration sentinel

- **Owner:** Rust maintainers (`rust` in `agent/owner-map.json`).
- **Scope:** `crates/jope-learning/src/clover/round_qual.rs:108`.
- **Reason:** this string is an accepted input sentinel that the preregistration
  validator rejects case-insensitively. Removing it would accept an unfilled
  contract value and change runtime behavior; it is not an unfinished code path.
- **Evidence:** `is_placeholder_sentinel` and its callers implement the
  validation contract. `crates/jope-learning/tests/selfplay_round_qual.rs`
  includes the `sample.c158_sentinel_todo` refusal case. The exact-file policy
  entry is also listed in the P5 R2 exception register.

### Frozen compatibility oracle

- **Owner:** Operations maintainers (`ops` in `agent/owner-map.json`).
- **Scope:** `deploy/dl-trainonly/lane/src/dl_training_v1/tests_p6/test_encodings.py`
  at lines 1107, 1122, 1124, and 1127.
- **Reason:** `_old_fetch_memberships` is retained as an exact pre-refactor
  compatibility oracle; its docstring records source hash
  `c454e1d2...`. The work-list name `todo` is ordinary local state; changing
  it would only disguise the valid term and does not improve the oracle.
- **Evidence:** `MapAuditRegression` compares the preserved implementation
  against the current map audit; its tests use synthetic membership fixtures.
  The exact-file exception does not disable those tests or change the oracle.

### Preregistered H72 control

- **Owner:** Research maintainers (`research` in `agent/owner-map.json`).
- **Scope:** `research/flywheel_selfplay/h72_w8_control_prereg/w8_control.py`
  at lines 199, 200, 201, 209, and 211.
- **Reason:** the local name describes jobs not yet present in the results
  ledger. The file is part of a preregistered study control, so its source is
  preserved for study-owner reproducibility review. No training result or fit
  budget is changed by this disposition.
- **Evidence:** the control’s cap and continuation behavior are covered by
  `test_w8_control.py`; `PREREG.md` binds each shard to the generated plan hash
  and records the required source revision with run outputs. The study prereg
  says no fit has run. The exact-file exception is limited to this control
  script and leaves the cap, work list, worker, and fit logic unchanged.

### Retired family diagnostic

- **Owner:** Operations maintainers (`ops` in `agent/owner-map.json`).
- **Scope:** `deploy/dl-trainonly/lane/src/dl_training_v1/dltrain/holdout.py:596`.
- **Reason:** “old family_id -> new family_id” accurately describes the
  required remapping from a retired identifier to its replacement. This is
  diagnostic vocabulary, not stale or unfinished behavior.
- **Evidence:** `validate_registry` separately verifies that retired IDs are
  not live families and that each replacement resolves to a known family.
  The exception is limited to this exact file; runtime validation remains
  covered by the existing P3 tests.

Reassess the exact policy exceptions at the review date or sooner if the input
contract, compatibility oracle, diagnostic, or study owner changes.
