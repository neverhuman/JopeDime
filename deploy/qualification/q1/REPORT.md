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

## Current CPU admission audit — 2026-10-31, owner t0210

The `overnight23_t0210` CPU-only assignment refreshed execution readiness at
source `d0bfa72a6c8270653b277ebd87025089ee3ddeeb`. It did not execute Q1.
The historical smoke receipt above remains a separate observation.

| Requirement | Observed state at this source |
|---|---|
| OWNER authorization | Preregistration still says `UNSIGNED / NOT AUTHORIZED FOR QUALIFICATION`; allowed-signers digest is `PENDING`. No study signature was supplied or created. |
| Product backend | `harness/production_backend.py` is absent. The reference native adapter remains smoke-only; no production backend was executed or qualified. |
| Actual plan, cohort and candidate proofs | Not assessed by this audit. No real cohort metadata, payloads, model, M2/M3 receipts or persistent study ledger were opened. |
| Q1 results and default decision | Not run: effect, interval, subgroup results, study failures and study compute remain unknown. No guidance-default decision is recorded here. |

On local xbabe2, nice 10 and serial execution, the existing harness suite passed
**12 tests, 0 failures, 0 skips**. Its temporary signing key is a software fixture,
not an OWNER key or study authorization. Three additional CLI controls used only
invented metadata from the existing unit fixture and a trapped, never-executed
backend:

- `run` refused with exit 1 and `OWNER_SIGNATURE_REQUIRED: preregistration is unsigned`.
- `analyze` refused with the same exit and reason before reading a nonexistent
  receipts file.
- `approval-payload` returned exit 0 and the expected document/plan/backend digest
  binding. Those bytes are an unsigned approval message, not a signature.

The controls produced **0 backend calls**, no execution receipts, no analysis
report and no ledger directory or consumption marker. They opened no actual
cohort or dataset payload and performed no native fit, optimizer update or GPU
work. Fixture result arithmetic is not an empirical effect estimate; these
checks establish only the current software admission boundary.

Command, exit, timing, fixture and input identities are retained in
`/home/ubuntu/jope_plan/evidence/QLT-04/obj02-t0210/admission-verification.json`.
The source-file SHA-256 pins are:

- Preregistration: `067c123307ed6ad5ab5a699fbf31334ecb453af052290f25378f53609087c964`.
- Driver: `40d4b153de279bb66cb689e4d61c610164598c4bf533705173f055c91372eecf`.
- Harness tests: `07c06c01091b160ad6e2697c0cad050b51916b7f799d5a7489abb9cc3ff31628`.

QLT-04 remains open. A later qualification must supply its real admitted plan,
backend and cohort evidence plus OWNER authorization; this CPU audit consumes
no study budget and grants no restart, qualification or default change.

## Owner-granted CPU wrapper dry run — 2026-10-31

The CPU-only wrapper now pins `deploy/qualification/q1/inputs-manifest.json` to
the published QLT-02 candidate metadata. The manifest contains only 2,606 dataset
IDs, canonical dataset SHA-256 values and S3 manifest SHA-256 values, plus source
cohort/catalog hashes; it includes no rows, object keys, family/shape metadata or
payload bytes. It is explicitly
`qualification_eligible=false`: the candidate cohort is not an owner-approved Q1
roster. No sealed or holdout payload was opened.

`fifo_launch.py dry-run` checked the candidate-manifest and frozen-DOPE hashes, then
created `/home/ubuntu/jope_plan/evidence/QLT-04/luna-qlt02-t0955/decision-record-candidate-metadata-final.json`
against source commit `d88176a6810c23c977789baf035bff5875344701`. The manifest SHA-256 is
`32e0abeba07e19c71e7e1b32c6959df14791e47dabbaf381e5983a0236c62e32`; the record SHA-256 is
`6822375ce98c51662366953069b84c884adbbc2fa2ea1a5a6aed8e658fcb05fa`.
The record says `q1_status=NOT_RUN`, `decision=UNKNOWN_NOT_RUN`; effects, intervals,
subgroups, failure inventory and Q1 compute remain unknown. It records no lock,
FIFO ticket, backend, payload or default change. This validates CPU plumbing only;
QLT-04 remains NOT_RUN/UNKNOWN and the existing report tables remain unfilled.
