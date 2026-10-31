# Q1 CPU preparation — not a qualification report

**Status: CPU_PREPARED / Q1 NOT RUN.** This file records only driver readiness and a synthetic CPU protocol dry-run. It contains no empirical Q1 effect estimate and makes no guidance-default decision.

## One-page preparation design note

**Approach.** Reuse the merged QLT-01 Q1 driver at `deploy/qualification/q1/harness/q1.py`; do not fork the plan, request/response contract, or append-only JSONL receipt writer. Validate the existing closed schemas using its synthetic unit suite, then execute its `heuristic_smoke` orchestration with synthetic rows and a fake backend. Keep the full `qualification` command gated on the exact owner signature, frozen plan/backend/cohort identities, and its persistent single-use ledger.

**Alternatives rejected.** The catalog smoke command was not run because it reads real FIT payloads and its source catalog records unknown license status. No qualification command was run because the preregistration is unsigned and no owner authorization or qualified production backend is present. Synthetic receipt-analysis fixtures are not substituted for Q1 results.

**Risks and limits.** The synthetic backend cannot establish real evaluation counts, backend isolation, parity, quality, or subgroup behavior. QLT-01 still records the owner allowed-signers digest as `PENDING`; its status also says the owner signature and independent COMPLEX reviews remain outstanding. T122 requires the actual COMPLEX study to be run by the stronger-model screen; this Luna preparation does not close QLT-04.

**Spec clauses.** Spec §13 keeps DOPE frozen and requires the later authorized engine boundary; §14 defines Q1 as the default-selection gate and OD7 subgroup margin. This prep changes neither protocol nor default.

## Existing driver and receipt schema

The committed QLT-01 driver supports `smoke`, `run`, `approval-payload`, and `analyze`. It writes `started` and `finished` JSONL events with plan, source, pin, dataset, seed, policy, budget, and request-digest identities. The `jope.q1-request/v1` request carries the frozen identity and (smoke only) payload/cut digest. The `jope.q1-response/v1` response carries status, charged/completed counts, RMSE, completed fold-score count, failure inventory, champion program, and adapter identity. `analyze` accepts only a complete, unique, identity-matched qualification panel and requires verified authorization; smoke receipts cannot produce a Q1 decision.

## CPU synthetic dry-run

`python3 deploy/qualification/q1/harness/q1.py --help` — PASS; no execution started.

`PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s deploy/qualification/q1/harness -p 'test_*.py' -v` — PASS, 12/12. Tests use synthetic plans/receipts, a temporary signing fixture, and mocked backend calls; they do not execute a trained or sealed model.

An end-to-end `heuristic_smoke` driver exercise used three pinned **metadata entries only**, 16 synthetic rows, and a temporary fake backend marked `synthetic-no-model`. The driver emitted 12 append-only receipt events for six synthetic requests; all six responses passed schema and request-identity checks. Receipt SHA-256: `843c79eb9e4abf8c1e43d6e9e5197eda690fa02fb093af9637e157ba862073e1`. No dataset payload was opened. The run's synthetic loss values are plumbing fixtures and are not Q1 measurements. External reproducibility files: `/home/ubuntu/jope_plan/evidence/QLT-04-cpu-prep/` (`synthetic-smoke.py`, plan, JSONL receipts, summary, test log, and help output).

## Required before Q1 can be reported

A stronger-model owner must first obtain and verify the OWNER authorization and exact frozen qualification plan/backend/cohort, then execute the signed qualification protocol under `qualification.lock` within its CPU-only run limits. Preserve all start/finish receipts and hashes, failures, compute, effect estimate, interval, subgroup counts/results, and the separately approved guidance-default decision. Until those gates are satisfied, Q1 remains **NOT RUN / UNKNOWN** and no default change is supported.
