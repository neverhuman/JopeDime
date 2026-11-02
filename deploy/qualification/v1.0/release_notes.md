# JopeDime v1.0.0 — release notes draft

**DRAFT / NOT RELEASED / NOT QUALIFIED.** The current evidence packet is not a
final qualification receipt, signer input, owner release approval, or aggregate
release signature. It retains separate evidence observations and does not
combine them into a final gate run. Candidate runtime source is RC4
`ccaf14b210b3b55a350218e60f6b49ef76524cf3` (tree
`2e4324f0bb387ece970664eac193a7ef6a8eb05c`), package scope
`internal-restricted`.

## Current evidence and remaining gates

The prior RC4 controller snapshot recorded **25 PASS, 2 FAIL, and 1 BLOCKED**.
It is preserved as a historical, exact-candidate attempt only; its passing rows
are not promoted or combined with later receipts. A fresh standalone M2 provider
receipt is PASS for the exact RC4 Linux artifact, but the final controller must
execute M2 again using its own fresh parity-report path. The current draft keeps
that standalone result separate from the earlier controller snapshot.

M3's raw operator-mix verdict remains **FAIL**. For v1.0.0 only, the separate
owner ruling `M3-WAIVER-V1.0.0-T1955` permits disposition
**WAIVED_V1_0_0**, subject to its recorded guard. The measured
native-minus-reference `edit_replace_head` deltas are **+2.30015625 pp overall**,
**+0.05208333 pp non-exploring**, and **+2.79084158 pp exploring**. The raw
verdict is not changed to PASS. Every other operator must remain within ±2.0 pp
in all three views, and the head-replacement residual must remain at or below
3.0 pp in all views. The ±2.0 pp tolerance is unchanged for future releases;
the waiver expires at v1.0.1. REL-07-F01 blocks v1.0.1 until the exploring
stratum gap is fixed.

A5 is **PENDING**: the latest three-client capture was incomplete, all clients
were excluded, and no accepted provider manifest or immutable
`candidate-inputs-v5` binding exists. The final same-controller run has not
completed. The separate QLT-04 Q1 result, its SHA-256, and an owner-recorded
guidance default decision also remain pending. No owner release approval or aggregate signature
is recorded; `release_qualified` remains false.

## Candidate artifacts

The RC4 full Linux x86-64, full Darwin arm64, and fused-cpu Linux x86-64
artifacts have matching source-bound signature and seal verification receipts.
Those package receipts establish artifact custody and identity; by themselves
they do not establish release qualification. Full artifact SHA-256 values and
byte counts are recorded in `qualification_receipt.json`.

## Release boundaries

- Windows native-host qualification is waived for v1.0.0 and is not described
  as a qualified shipped platform. Linux x86-64 and macOS arm64 require
  candidate-bound D1 results in the completed final run.
- CUDA/GPU and Metal qualification are not established by this CPU candidate.
  This draft task used no GPU or training job.
- D2's size verdict and latency measurements remain separate; no startup or
  first-model speed guarantee is inferred from a size PASS.
- The package scope is `internal-restricted`; envelope signatures do not grant
  unrestricted redistribution of embedded assets.
- A prior Darwin worker-CV report recorded 58/1,024 parity failures. This draft
  does not supersede or relabel that report; its disposition must remain
  explicit in the final candidate-bound evidence.
- No learned-quality, Elo improvement, or successful Q1 outcome is claimed.
