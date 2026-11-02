# Jankurai vibe findings: scoped language allowances

The five `dead_language_allow` records in `agent/audit-policy.toml` cover only
the exact term and file listed below. They keep those validated source terms
out of the vibe cap while leaving the files in the scan, so other findings in
these files remain visible. Each allowance expires 2027-02-01 and must be
reassessed against fresh source and test evidence.

## Preregistration sentinel

- **Scope:** `crates/jope-learning/src/clover/round_qual.rs`, term `todo`.
- **Reason:** `TODO` is an explicit input value rejected by the
  `is_placeholder_sentinel` contract, not an unfinished code path.
- **Evidence:** `crates/jope-learning/tests/selfplay_round_qual.rs` checks the
  `sample.c158_sentinel_todo` refusal case.

## Frozen compatibility oracle

- **Scope:**
  `deploy/dl-trainonly/lane/src/dl_training_v1/tests_p6/test_encodings.py`,
  term `todo` at lines 1107, 1122, 1124, and 1127.
- **Reason:** the local list tracks remaining comparisons in the frozen
  `_old_fetch_memberships` compatibility oracle. Renaming it would disguise a
  valid work-list term without improving the oracle.
- **Evidence:** the copied implementation retains source hash
  `c454e1d2...`; `MapAuditRegression` compares it with the current map audit.
  The test remains enabled.

## Preregistered H72 control

- **Scope:**
  `research/flywheel_selfplay/h72_w8_control_prereg/w8_control.py`, term `todo`
  at lines 199, 200, 201, 209, and 211.
- **Reason:** the local name describes jobs absent from the results ledger.
  The file is part of a preregistered control, and its source remains unchanged
  for study-owner reproducibility review.
- **Evidence:** no fit has run. `test_w8_control.py` covers the control, while
  `PREREG.md` binds each shard to the generated plan hash and records the
  required source revision with run outputs. The allowance does not change the
  cap, work queue, worker, or fit logic.

## Retired family diagnostic

- **Scope:**
  `deploy/dl-trainonly/lane/src/dl_training_v1/dltrain/holdout.py`, term `old`
  at line 596.
- **Reason:** “old family_id -> new family_id” accurately describes remapping
  a retired identifier to its replacement.
- **Evidence:** `validate_registry` checks that retired IDs are not live
  families and that each replacement resolves to a known family. Existing P3
  tests continue to cover runtime validation.

## Late overlay refusal sidecar

- **Scope:** `deploy/dl-flywheel/pipelines/native_result_overlay_v1/build.py`,
  term `best effort` at lines 29 and 82.
- **Reason:** after the original deadline is crossed, the builder attempts a
  create-only refusal sidecar for custody and raises. That sidecar is
  diagnostic; it is not a positive result. If its write fails, the exception
  still propagates and the builder has not produced a caller acceptance
  envelope. Consumers require that separate envelope before admission, so the
  best-effort write cannot admit a late or failed packet.
- **Evidence:** `tests/test_durability.py` exercises refusal when publication
  crosses the original wall or monotonic bound, checks that refused packets
  remain inadmissible, and verifies no retry. The pipeline README documents
  that only the caller writes the post-return acceptance envelope.
