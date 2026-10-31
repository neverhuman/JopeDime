# Wave delivery record — owner t2335

Dated publication audit, 2026-10-31. Source inventory: `a535e8a6d9d434b2aea9a6f25a6deef54da18011`.
This is a status and scheduling record, not a new numerical experiment.

## Current scheduling instruction — t2335

After H57, the first **W1 or fixed-D2 signed CPU pre-review PASS** takes the
qualification lock. D2 remains HOLD until its repair passes; the old D2-first
wait is superseded. H75 has no new GPU slot in this pass. ASTRA/OPUS CPU shard
order is W1/W3/W4/W6. Tests route to eligible xbabe1 or local xbabe2 JMAX<=8,
not xbabe3/xbabe4. Scientific gates, budgets and deadlines remain unchanged.

[H78 W1 artifact audit](../h78_w1_artifact_identity/README.md) is a completed
read-only preparation shard: four files present with matching hashes/ID order,
only baseline report absent at the recorded snapshot. NOT_READY is not a
scientific W1 verdict or a signed launch review. Source and controls are in its
verified S3 manifest. The metadata audit does not resolve producer lineage or
outstanding W1 source-review findings.

## Earlier scheduling instruction — t2320 (superseded)

`~/nudges/owner_t2320.md`, SHA256 `a2860ba9164e4ac5728df679a14ccd7f0f1c36c1637a83918ec6fed6b0edea54`, explicitly orders
**H57 → Fused D2 → W1** under the existing qualification lock, followed by
**W3 → W4 → W6 → W5 (provenance hold) → W7 → W9**.
The user delivered this pass after `owner_lift_t2250.md`; it supersedes that
pass's conflicting H75-before-W1 schedule. Keep the prior evidence immutable.
H75 CPU runner/support work continues; no H75 GPU placement is specified here.
No ongoing job is stopped, no lock taken and no experiment launched by this
publication audit. E has no new GPU slot. Data closure, budgets and deadline
are not reset by reordering.

## Completed work already published

| Work | Durable result / publication | Interpretation |
| --- | --- | --- |
| W2 | [H62 summary](../h62_slate_consistency/SUMMARY.md), [merged PR799](https://github.com/neverhuman/JopeDime/pull/799) | Diagnostic complete; 24/800=3% violates the written2% tolerance. Corrected result, not invariance or a trained-policy gain. |
| W8 | [H63 side-by-side summary](../h63_training_resplits/SUMMARY.md), [initial PR804](https://github.com/neverhuman/JopeDime/pull/804), [equal-cost PR822](https://github.com/neverhuman/JopeDime/pull/822), all merged | Original-cut sensitivity and equal-cost control both published. Control's primary interval touches zero: INCONCLUSIVE. Secondary Elo is not a full-wave admission. |
| W1-related H73 | [family-balance diagnostic](../h73_family_balanced_sampling/README.md), [merged PR830](https://github.com/neverhuman/JopeDime/pull/830) | CPU concentration diagnostic, no optimizer update or completed W1 comparison. |
| W4 cached shard | [H75 W4 shard](../h75_w4_cached_baseline_shard/README.md), [merged PR821](https://github.com/neverhuman/JopeDime/pull/821) | Completed descriptive random/hash baseline screening; no adaptive W4 primary verdict. This H75 is distinct from closed-loop H75. |
| W6 roster | [H72 W6 metadata](../h72_w6_train_roster_prep/README.md), [merged PR823](https://github.com/neverhuman/JopeDime/pull/823) | Field-allowlisted roster/folds prepared. Resolves H66's older missing-roster note; not W6 outcomes or a fitted model. |

All rows already have their H folders and publication history. Do not reserve
a new H number or duplicate a PR for the same artifact. S3 pointers in those
folders remain the evidence authority; this pass inspected source/receipts and
GitHub lifecycle metadata, not dataset or model payloads. New independent
results may extend an existing study under an explicit claim; a distinct wave
requires checking current claims and main for the next free H identifier.

## Remaining wave status and next publishable artifact

| Wave | Existing preparation | What is still needed before a result claim |
| --- | --- | --- |
| W1 | [H70 gate component](../h70_w1_prelaunch_gate_audit/README.md); ARGON owns the external runner | Completed admitted run and result receipts. The latest T2315 source review still records launch issues; a calculator or prior shard is not the W1 verdict. |
| W3 | [H65 preparation](../h65_w3_continuous_top_resolved/PUBLICATION.md), [H74 readout](../h74_w3_readout_gate/README.md); [PR809 merged](https://github.com/neverhuman/JopeDime/pull/809) | Actual W1 configuration, H59 disposition, clean completed-status outcome export and executed matched cells. |
| W4 | [H67 preparation](../h67_w4_active_reveal_2plus2/README.md) | Admitted adaptive-versus-baseline experiment with matched four-total-observation budget and full analysis; cached screening alone is insufficient. |
| W6 | [H66 design](../h66_w6_cross_family_transfer/README.md) plus H72 roster above | W1 model outputs and admitted outcome history, then actual fitted comparison. Do not repeat the resolved roster blocker. |
| W5 | [H71 preparation](../h71_w5_parent_child_critic/README.md) | Exact same-context parent/child/operator and original-row provenance; hold remains. |
| W7 | [H69 design](../h69_w7_informative_acquisition_qd_replay/README.md) | Qualified acquisition/replay inputs and executed common-learner comparison. |
| W9 | [H68 preparation](../h68_w9_causal_closeout/README.md) | Executed causal contrasts and matched replay; no duplicate spend of the H75/W9 shared reservation. |
| H75 closed-loop | [merged preregistration](../h75_closed_loop_evolution/PREREG.md), [PR841](https://github.com/neverhuman/JopeDime/pull/841) | Runner integration, input/execution manifest, completed experiment and explicit current GPU placement. GOV03 scorer and OBJ02 feedback are CPU components, not lift results. |

## Result publication contract

As each wave finishes, ASTRA/OPUS publish a compact summary and machine-readable
metrics in its claimed H folder. Include the literal verdict, exact source and
input identities, all intended/completed/failed/unfinished cells, actual
optimizer changes where applicable, family Elo and raw utility with their
registered endpoints, cumulative cost ledger, and immutable S3 manifest URI,
version and downloaded-byte SHA256. Missing metrics are null with reasons.

Review the prespecified denominator, controls, row/family closure and primary
gate before declaring success. H62's corrected denominator and H63's
inconclusive primary remain explicit. Ship a ready PR with targeted checks
and exact-head handoff to mergers; do not wait for a cosmetic complete-wave
story or convert preparation into PASS. Preserve foreign work and existing
receipts. This publication pass performs zero native fits and zero GPU work.
