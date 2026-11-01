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
created `/home/ubuntu/jope_plan/evidence/QLT-04/luna-qlt02-t0955/decision-record-source-52379.json`
against source commit `52379e482af435c198464c5696baa93c401a1ee5`. The manifest SHA-256 is
`32e0abeba07e19c71e7e1b32c6959df14791e47dabbaf381e5983a0236c62e32`; the record SHA-256 is
`1e68153d685d3d19c1ca18c21db149199a926b84a076a7399cd5cc3fd47f64e4`.
The record says `q1_status=NOT_RUN`, `decision=UNKNOWN_NOT_RUN`; effects, intervals,
subgroups, failure inventory and Q1 compute remain unknown. It records no lock,
FIFO ticket, backend, payload or default change. This validates CPU plumbing only;
QLT-04 remains NOT_RUN/UNKNOWN and the existing report tables remain unfilled.

## Owner-delegated key and remaining blocker — 2026-10-31, Jepson GO 10:32 MDT

Jepson approved, for Q1 only, at 10:32 MDT on 2026-10-31: (1) Q1 may use the sealed
meta-test holdout only through its pinned evaluator; (2) a bot-held ed25519 key may
sign as principal `jepson`; (3) an `--admin` merge of the Q1 PR(s). This section
records item (2) as required.

The earlier sections of this report are dated observations. Where they describe
the allowed-signers digest as `PENDING`, or pin preregistration SHA-256
`067c1233…`, they describe the state before this change.

**The Q1 signing key is an owner-delegated agent key, not a key Jepson holds
personally.**

- It was generated on xbabe2 by the `qlt04_sign_claude` session at 16:37Z.
- The private key stays outside Git, but any agent process running as the shared
  `ubuntu` user can read it, and so can sign Q1 as `jepson`. The owner accepted this
  risk.
- The public half is committed as `owner/allowed_signers`, restricted to namespace
  `jopedime-q1`, with fingerprint
  `SHA256:2KNNiZS7c+YbwsAMt2x0Q9gIdRCd2Yi6CR3k/d1N6LY` and no expiry. Its file
  SHA-256 is `6bc05cef59ee45376fe77fcaf27074edac2f7bd390faac3f193c466175313235`,
  which replaces `PENDING`.
- The amended preregistration's SHA-256 is
  `f2a234dda18171623053e233a1e6a486df5215ea977097ea3f9d6b81a6f219b5`.
- The canonical ledger root, `/home/ubuntu/jope_plan/ns/qualification/q1_ledger`,
  exists and is empty. Signatures will go in
  `/home/ubuntu/jope_plan/ns/qualification/q1_approval/<q1.digest(plan)>.sig`.

**No qualification plan is signed and Q1 is still NOT RUN.** A signature has to bind
`q1.py approval-payload` for a final plan, and that plan needs the SHA-256 of a
qualified `production_backend.py`. That backend cannot be built honestly on current
main:

- The discovery loop's learned-guidance seam is the `Admission` trait in
  `crates/jope-service/src/discover/loop.rs`. Its `admit` receives only the proposal
  strings, the explore flag and the RNG.
- The learned path needs more than that: `ServingAdmissionSlate::prepare` takes
  measured reveals, program features (prog-412 and proj-16) and landmark/token
  conditioning, and none of these reach `admit`.
- No caller outside `crates/jope-service/src/ranker/` uses the slate, `select_costs`
  or adapter costs. `resolve_unqualified`, which refuses `Guidance::Learned`, is
  referenced only by its own tests.
- The RNK-10 CPU policy primitives are merged (#755, #800). The composition card,
  RNK-10-F01, is blocked: it needs an owner grant for the per-round context hand-off,
  qualified RNK-07/08 evidence and OD13. RNK-12 M2 parity and the M3 quality
  receipts are not admitted.
- QLT-01-F01, the card that owns this backend, forbids service edits and says to
  stop and ask for a COMPLEX or OWNER ruling when no suitable engine interface
  exists.

Running Q1 on an improvised learned path would consume the sealed cohort for that
bundle permanently, because the ledger is single-use, and would produce an
unqualified default decision.

A materialized roster is also still needed. Owner ruling t2300 defines the sealed
group by reference as all P3 `sealed_excluded` families (293 production + 470
catalog). The plan also needs a cut identity per dataset, and that identity depends
on the evaluator protocol the backend freezes. The published `cohort.json` and
`inputs-manifest.json` are the non-sealed candidate set
(`qualification_eligible=false`), not the Q1 cohort.

## R19 pre-run subgroup disposition — 2026-11-01

Owner ruling [R19](PREREG-AMENDMENT-R19.md) permits the existing roster's four
distinct wide families for this run. Q1 remains **NOT RUN / UNKNOWN**. No empirical
effect, interval or subgroup verdict is supplied by this amendment.

| Subgroup | Frozen family count and reporting status | Effect / interval |
|---|---|---|
| wide | 4/5 families, below preregistered minimum, descriptive only, no subgroup verdict | UNKNOWN — Q1 NOT RUN |

Small, large and narrow retain their original inferential requirements. Any later
overall decision must identify R19; it cannot claim a wide subgroup verdict.

## Current T1115 deduplicated plan — 2026-11-01

**Q1 NOT RUN / INCOMPLETE / signable:false.** The latest owner direction requires
five distinct wide families and supersedes the historical R19 exception for this
new plan. [T1115](PREREG-AMENDMENT-T1115.md) selects 97 frozen owner families from
the unchanged153-record source packet by the existing minimum-SHA rule, excluding
56 same-family variants. The four old wide records contain two RECS members of
one owner family; the selected97 therefore contain only three wide families.

| Subgroup | Selected distinct families | Required | Q1 result |
|---|---:|---:|---|
| small | 70 | 5 | UNKNOWN — Q1 NOT RUN |
| large | 27 | 5 | UNKNOWN — Q1 NOT RUN |
| narrow | 94 | 5 | UNKNOWN — Q1 NOT RUN |
| wide | 3 | 5 | INCOMPLETE — two families short; no result |

The full pinned4750-row catalog census supplies no additional wide member bound
to these97 owner families. No row is duplicated or family reclassified to fill
the gap. Primary estimates/intervals/subgroup verdicts, failures and study compute
remain unknown. Actual evaluator/runtime/closure/signature gates still apply.

## R20 pre-outcome census update (2026-11-01 19:35 UTC)

Q1 remains **NOT_RUN** while the exact source-bound evaluator/runtime and exposure
qualification are prepared. R20 admits all **3 eligible distinct wide families**,
bound to the exhaustive [QLT02 audit](WIDE-GAP-R20.md) and
[pre-outcome amendment](PREREG-AMENDMENT-R20.md). Full selection: 97 unique owner
families (small70/large27/narrow94/wide3), 56 same-family variants excluded before
outcomes. No new dataset/family/cut or training reassignment. All four subgroup
checks retain the -0.02 margin and 0.05/4 lower quantile. Wide has only three
families, limiting precision and coverage; this is not the original five-family
design. No scientific PASS/FAIL, signed approval or execution is claimed here.
