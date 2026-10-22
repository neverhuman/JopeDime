# Four-model benchmark delivery

This checklist tracks the three serial deliverables requested on 2026-10-22.
Implementation, fixture proof, hosted CI, live measurement and merge are separate
states. The predecessor preparation is historical evidence, not a completed run.

| Deliverable | Pull request | Implementation | Fixture proof | Live measurement | Merge |
| --- | --- | --- | --- | --- | --- |
| Frozen Gemini integration | Pending | Implemented, unmerged | Targeted Cargo and compiler proof passed; full gate pending | Not part of this PR | Pending |
| Execution, fitting closure, timing, export and accounting tools | Pending | Pending | Pending | Both manifests pending | Pending |
| Recomputed results, figures, recovery and paper | Pending | Pending | Pending | Pending | Pending |

The prerequisite [PR #405](https://github.com/neverhuman/JopeDime/pull/405) merged
as `bdf9441aaa17f87dcb37053d33b185ba2217bf17`. Both 11-job CI runs passed, as did
the local required gate with a documented isolated view of the exact committed
custody files. Its reviewed tree matches remote main and `main-guard` passed.
Its final source audit reported 54/85. The receiving-main audit in run
`39744007127`, attempt 1, reproduced that failed quality gate, verified all
5,956 captured source files and successfully retained and published its evidence.

## Frozen integration

- [x] Preserve the original source and four-annotation successor, supplied design,
  telemetry, correction and recovery identities.
- [x] Verify the original's four compiler errors and the successor using
  `python3 tools/qualify-gemini.py --output OWNED_FRESH_EVIDENCE_DIRECTORY`.
- [ ] Qualify both selectors, regression validation, state independence, missing
  values, malformed shapes, feature-only prediction and held-out-target isolation.
- [x] Refresh maintained inventories through their owning tools. The regenerated
  Rust list adds eleven integration fixtures and four previously unlisted tests
  already present on receiving main; it removes none.
- [ ] Complete local gates, exact-head manual CI, review, merge and receiving-main proof.

The original hash is
`ddddc2557195830ee7fdb156f14802c63da22c171a31b15c232e45635b74b88c`;
the corrected hash is
`b0aa42d24d0b067e0597f6f538b14cab75dca8b976731668c894d80684d4992d`.
The [entrant review](../../crates/jope-automl/submissions/gemini-argon-4/REVIEW.md)
retains its internal validation concerns. They do not authorize algorithm edits.

## Campaigns and publication

- [ ] Pin the new 576-cell Gemini manifest to merged source and binary identities.
- [ ] Independently restore and rescore the 1,728 historical fits without refitting.
- [ ] Qualify corrected fitting-only projection, resource enforcement, cache,
  timing, exports and accounting with synthetic inputs.
- [ ] Record fresh, separate CPU and GPU allocations before live admission.
- [ ] Reconcile 576 Gemini cells and retain every terminal or unknown outcome.
- [ ] Qualify the native CUDA assets without fallback or new real-dataset searches.
- [ ] Reconcile exactly six requests: seed 0 cold and seed 41 warm on each of
  `230_machine_cpu`, `auto_insurance_price` and `197_cpu_act`, with 96 opportunities.
- [ ] Recompute four-way statistics, source counts, yields and figures independently.
- [ ] Publish compact evidence and exact-version recovery pointers; finish PRs 2 and 3.

The six requests permit at most 2,880 fold fits and six winner fits. There are no
scientific wall-time cutoffs. Cancellation or lease loss remains infrastructure
censoring. No baseline refits, authoring API calls, neural training or serving
promotion are authorized by this work.

Current measured completion is unchanged: 1,728 historical fits, zero new Gemini
fits and zero complete code-production requests. CPU/GPU grants and corrected
runtime/input qualification are pending. Unavailable measurements remain explicit.

The [preparation recovery pointer](../../crates/jope-automl/submissions/gemini-argon-4/preparation-recovery.json)
identifies the earlier independently restored archive. Its prepared manifests,
source qualifications and figures are historical; completed campaigns require new
manifests and a new independently restored archive.
