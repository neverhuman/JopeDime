# W5 provenance refresh — t0855, 2026-10-31

**FAIL / W5 HELD.** Fresh S3 version/SHA checks pass, but parent-child and original-row closure still cannot be established. No GPU-queue preparation plan is filed because the owner's PASS condition is unmet. This is missing evidence, not a finding of contamination or an assertion that custody contains no valid pairs.

The unchanged [original provenance gate](../verify.py) exits **1**. Its actual verdict and an exact offline replay are in [verdict.json](verdict.json) and [evidence.json](evidence.json). The unchanged H102 relation checker separately reports missing lineage and original-row provenance for **504 children / 254 contexts / 123 families** ([relations.json](relations.json)). Its historical `next_action=W9_CPU_PREPARATION` field is preserved as checker output; no W9 work or budget reset was performed.

## Gaps checked

| Requirement | Fresh evidence | Disposition |
| --- | --- | --- |
| Exact-version real S3 input integrity | Three pinned 806-byte retention counters and five previously admitted metadata objects: exact VersionId, byte length and SHA256 all match | PASS for these eight objects only |
| Source export availability | Complete source census: same83 keys/sizes as the earlier inventory. W5 output namespace12 objects; w5_provenance namespace17 objects. All are prior audit artifacts; no new separately identified clean export | No new closure artifact found in this bounded search |
| Sealed-group authority | Owner t2300 explicitly defines the sealed group as all P3 sealed_excluded families (293 production +470 catalog), by reference | Definition RESOLVED; do not repeat “no authority supplied” |
| W5 membership exclusion | No field-allowlisted, computable identity projection bound to that owner reference and matching alias registry was supplied | UNKNOWN global overlap; no protected roster or data opened |
| Exact parent/operator/child join | Fresh v3 manifest key census still has pool-level parent hashes only; no per-child parent/operator link | Missing for504 children; no AST/chronology inference |
| Original training and strat300 rows | No original-row ledger or child/source-row/key/version/SHA relation in the admitted inputs | Missing for504 children; row closure UNKNOWN |

The authority definition closes a documentation gap, not the content join. The QLT-02-F02 status explicitly closed that separate task **by reference** and claimed no overlap computation. It does not establish W5 exclusion. An earlier H117 “unadopted proposal” comparison is bounded evidence; it cannot substitute for the owner-defined P3 identity projection.

The 83-object listing and the two audit-prefix listings establish only their enumerated scope. No claim is made that the missing artifacts cannot exist elsewhere. Known mixed phase-7/v2 report bodies, retained outcome files, pool arrays, tables, model/checkpoint files and sealed/holdout payloads remain unopened. A successful hash readback does not establish completion, evaluation identity, collection policy, current role/alias closure or unbiased pair coverage. These obligations in the original gate remain in force.

## Concrete closure handoff

The data custodian must provide an independently admitted, field-allowlisted training-only projection with:

1. An identity-only exclusion roster tied to the P3 owner reference and family-alias registry, with immutable byte and authority receipts. Never grant permission to inspect protected data from this request.
2. For each of504 candidate children, the exact context, parent program, operator and child ID; independently verified pool membership and immutable lineage-event source. Whole-pool hashes are insufficient.
3. Exact original-training row IDs and strat300 row IDs at the same context/table/cut; source key, VersionId, SHA256 and verified byte receipt; completion/evaluator/metric/gate/vocabulary and unrounded outcome identities. Retention counters are not pair coverage.

If the historical producer never recorded these links, use a separately authorized lineage-recording regeneration. Do not synthesize historical parenthood or row provenance. Rerun the unchanged relations and row/pair checks on independently authenticated bytes, then file a W5 CPU prep/admission plan only after all obligations pass. No new execution budget is granted here.

## Verification and custody

55 synthetic tests and28 subtests pass, zero failed/skipped. Four deterministic metadata shards completed on xbabe2/3 and xbabe4 Linux Docker; xbabe1 was refused before worker launch because its154,466,062,336 free bytes were below the inherited150GiB guard. Its single shard was reassigned to xbabe2; no successful shard was repeated. Eligible workers used nice10 and passed<=85% load checks; no live-loop interaction, job interruption or cleanup occurred. Exact CPU receipts are in evidence.json; S3/SSH/test overhead is unmeasured. Zero model/native fits, optimizer updates or GPU seconds; DOPE remains frozen at c51a56d0 without loading it. Elo/U4/val_reg_inner are null; no scientific readout or HWM candidate.

[custody.json](custody.json) identifies the create-only archive/manifest and exact-version restored member verification. It includes all source pins, input receipts, listings, unchanged checker sources, commands, preflights, shard results and gate/test logs. Checkpoints, datasets and receipts were not deleted. Source and prior evidence are preserved; this dated result does not rewrite historical receipts.
