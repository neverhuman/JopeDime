# JopeDime v1.0.0 — release notes draft

**DRAFT / NOT RELEASED.** Final qualification receipts and OWNER release approval
are pending. This document does not assert that the candidate has passed every
release gate. Candidate runtime source:
`ccaf14b210b3b55a350218e60f6b49ef76524cf3`.

**Q1 guidance default — OWNER-RECORDED DECISION REQUIRED:**
`[PENDING: QLT-04 result/decision receipt, its SHA-256, OWNER record and the final
guidance:auto default. Replace this block from those records before publication.]`

The separately recorded R21 scope decision keeps learned guidance disabled as a
v1 default and requires the default-off admission/determinism proof. This does
not substitute for QLT-04's Q1 result or its OWNER-recorded release decision.
No learned-quality, Elo improvement or successful Q1 outcome is claimed here.

M3 operator mix has a one-time waiver for v1.0.0 under
`M3-WAIVER-V1.0.0-T1955`. The recorded merged-proposer head-replacement deviations
are **+2.30015625 percentage points overall**, **+0.05208333 points non-exploring**
and **+2.79084158 points exploring**. The raw verdict remains **FAIL**; the separate
release disposition is **WAIVED_V1_0_0**, never PASS. Final measurements and hashes
must be recorded in the qualification receipt before release.

This accepts a small search-mix bias toward head replacement in the exploring
stratum. It makes no claim about improved neural quality. The waiver applies
only when every other operator remains within ±2.0 points and head replacement
within 3.0 points in all three views. Other M3 clauses, including numerical CV,
and every other release gate remain required. The ±2.0-point criterion itself
is unchanged for future releases. The waiver **expires at v1.0.1**;
**REL-07-F01** blocks v1.0.1 until the exploring-stratum gap is fixed. The older
D5 edge-case disposition is separately tracked by QLT-04-F04.

Known limits and qualification boundaries:

- Windows native-host qualification is waived for v1.0; do not describe it as a
  qualified shipped platform. Linux x86-64 and macOS arm64 still need their
  actual source-bound D1 receipts.
- CUDA/GPU and Metal qualification are not established by this CPU candidate.
  No GPU or training work is part of this release-receipt task.
- D2's owner-scoped size verdict and recorded latency measurements must remain
  separate. Do not infer a startup or first-model speed guarantee from a size PASS.
- The full candidate uses the `internal-restricted` package scope. Its envelope
  signature does not grant unrestricted redistribution of embedded assets.
- Draft qualification blocker: fresh Darwin worker CV recorded 58/1,024 parity
  failures outside the operator-mix waiver. Preserve that failure and its scope;
  do not relabel it PASS or treat a Linux result as Darwin qualification. Replace
  this draft blocker only with the actual resolved source/runtime disposition.

The final qualification receipt will identify the tested signed bytes, commands,
inputs, outputs, hashes and machines. Envelope signatures and component proofs
alone do not establish whole-release qualification.
