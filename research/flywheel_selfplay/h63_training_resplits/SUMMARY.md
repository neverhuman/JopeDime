# H63 / W8 — training-cut sensitivity and its equal-cost control

> **Update 2026-10-31 (owner t2225):** the equal-cost control has now run; see [Both W8 arms side by side](#both-w8-arms-side-by-side-owner-decision-t2225). The review verdict immediately below predates it and is kept unchanged.

**Review verdict: cut sensitivity is observed; the equal-cost acquisition comparison has not run.** Among11 candidate/champion pairs with positive original utility gains,19/33 fresh-cut comparisons retain the sign (57.58%), and3/11 retain it on all three cuts. Mean gain falls from+0.0244 to+0.0158. This motivates testing reliability; it does not establish that extra resplits beat spending those fits on new candidates, or authorize a reliability learner.

The study contains95 candidate/champion pairs across16 DL-Train contexts and13 families:90 previously verified novel winners and five recorded strict promotions, each compared with the context's stored champion. These are **not95 original HWM promotions against their displaced incumbents**. Only11 challengers beat the selected champion on the original cut;84 lose.

## Results

| Original absolute utility margin | Pairs | Same-sign resplit comparisons | Agreement |
|---|---:|---:|---:|
| [0, 0.005) | 27 | 54/81 | 66.67% |
| [0.005, 0.02) | 21 | 37/63 | 58.73% |
| [0.02, 0.05) | 5 | 8/15 | 53.33% |
| [0.05, 0.2) | 29 | 75/87 | 86.21% |
| [0.2, 99) | 13 | 34/39 | 87.18% |

Overall sign agreement is73.0%; the author's context-cluster95% interval is[63.3%,83.0%]. That interval clusters16 contexts, not13 families; same-family dependence is not fully handled. Resplits are correlated measurements of the same original rows. The median pair-difference SD is0.0192. The reported per-program SD median0.0183 is actually weighted over190 program-role instances, including repeated champions; there are111 unique context/program combinations.

The post-run family Bradley–Terry Elo addendum anchors incumbent=1500, tie margin1e-4, prior.01 and400 iterations. ASTRA reproduced the point ratings and W/T/L from the S3 pair records; source-reviewed author intervals use1,000 family-bootstrap draws (seed20261031), not the common10,000-draw primary gate.

| Pair subset / cut | Challenger Elo minus incumbent | 95% interval | W/T/L |
|---|---:|---:|---|
| all_pairs/original_cut | -373.52 | [-534.92, -277.45] | 11/0/84 |
| all_pairs/resplit_mean | -231.68 | [-420.95, -109.58] | 19/1/75 |
| originally_winning/original_cut | +1138.29 | [+1138.29, +1138.29] | 11/0/0 |
| originally_winning/resplit_mean | +24.79 | [-224.87, +309.82] | 5/0/6 |

The originally-winning subset has only11 pairs from seven families. Its original-cut +1138 Elo is a consequence of selecting wins, not an independently measured capability. The resplit-mean +24.79 Elo interval crosses zero. All panels describe candidate/champion pairs, not ranker or serving performance; their scales are not transferable to other contender panels. `val_reg_inner` and verified U4 remain null because no ranker/policy was scored.

## Execution and closure

760 unique completed native evaluations, zero failed:190 original-cut checks and570 additional resplit evaluations (95 pairs×2 programs×3 seeds701/702/703). All95 original challenger refits match the stored fitness exactly. The reserved equal-cost new-candidate arm has zero evaluations;1,640 of the2,400 evaluation ceiling remain after this run. The original1,200-call control reservation is a ceiling, not an automatically cost-matched comparison for this95-pair run. A control must prospectively match the actual570 additional resplit calls (and account consistently for190 identity checks), use a common validation endpoint, and preserve cumulative resources; apparent original-cut wins are not interchangeable with resplit-confirmed gains.

Summed per-evaluation wall time is1053.3 seconds (about0.293 hours), **not measured process CPU time**. Twelve single-thread-configured workers ran on xbabe2 at nice15, zero GPU. The launcher gates on one-minute load divided by reported CPU count, not measured utilization; continuous85%-CPU compliance and exact CPU-hour consumption are not established by the receipt. No job was rerun for this publication.

ASTRA checked all16 original training-table SHA256s and `nq=0` headers, covering24,031 row instances, and reconstructed every one of the760 executed input hashes. Only fold assignment changes for the resplits; feature/label bytes remain unchanged. Ten all-zero query rows are structural native-worker padding, not real held-out observations. The metric uses inner-CV records; no outer-test outcome is used. The pinned worker hash matches ELF machine62 (x86-64). These checks do not supply missing upstream S3 object keys or original displaced-incumbent lineage.

xbabe1/3 were saturated. The run completed solely on xbabe2 before the later t2155 direction to use xbabe2+xbabe4. The observed jope-ci environment is Linux/aarch64, whereas the pinned native evaluator is x86-64; no equivalent native execution was qualified there. No duplicate fits, evaluator rebuild or unqualified emulation were used to satisfy host allocation retrospectively.

## Immutable evidence and disposition

Original source HEAD at upload: `d5095432129d57cbd7cffa9fd6ceb77dfe86cd9e`. Runner/analysis files were untracked scratch; their hashes and copies are preserved separately from Git.

- Run prefix: `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/W8/w8-opus-20261031a/`.
- Manifest SHA256: `9510a32e45bbafd3423d3d2a14acb6410fd1a1f6ca9821fc2a2cad8794339c50`.
- ASTRA downloaded and SHA256-verified all eight original objects and two Elo addenda, reproduced760 completion identities,95 exact refits, pair deltas, sign counts and Elo point ratings. Initial version history contained eight original versions and no deletion markers; addenda use new keys. The author upload helper was HEAD-then-upload, not atomic create-only; no overwrite was observed. Future publication uses `If-None-Match: *`.
- [metrics.json](metrics.json) contains exact object versions/hashes, statistics, resource limits and row-closure checks. Review commands/receipts are under `/home/ubuntu/jope-rescue-20260921/rd-verdicts-t2155/`.

The stability-based stop condition is not supported for this sampled set. Retain W8 as an unfinished diagnostic/control study. Do not infer a reliability-policy advance, causal live improvement, or r882 development-panel selection bias. No HWM candidate was created and no serving change occurred.

The reviewed result snapshot is preserved create-only at `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/W8/astra-review-20261031T041309Z-483edd92/manifest.json` (version`kQaB7a1DbULpJVW2z7rIUsgPRXCW4IBr`, SHA256`539f0c2a3d5eb4d039ae8ab712015e41d8d85aec14f116b3ce6de0d4f83bf92e`), sourcea0a576b09. All seven review objects, including the manifest and row-closure checker/receipt, passed version-pinned downloaded-byte SHA256 checks. That snapshot precedes this pointer append.

## Both W8 arms side by side (owner decision t2225)

The owner approved the equal-cost arm, and it has now run as preregistered in
[H72](../h72_w8_control_prereg/PREREG.md) (PR #812, merged at `b6aaade3`). Both arms are shown below. The
text above this section is unchanged.

**In plain terms:** challengers that win on the original cut lose most of that advantage on new cuts of
the same training rows. Their Elo against the champion falls from **+1138 to +25**, with a 95% interval of
**−225 to +310**, which includes zero; on the new cuts they win 5 of 11 and lose 6, a coin flip. At equal
cost, checking each apparent win on three resplits before promoting it removes almost all false
promotions. The preregistered primary interval touches zero, so the control is formally inconclusive.

| Arm | Question | Sample and cost | Result |
|---|---|---|---|
| **Original-split arm** (H63 diagnostic, run `w8-opus-20261031a`) | Do original-cut wins keep their sign on new training-only cuts? | 95 candidate/champion pairs; 760 evaluations (190 original-cut checks, 570 resplits) | Originally-winning challengers (n = 11): **+1138 Elo on the original cut → +25 Elo [−225, +310] on the 3-resplit mean, W/T/L 5/0/6**. Sign kept in 57.6% of resplit comparisons; mean gain +0.0244 → +0.0158. |
| **Equal-cost arm** (H72 control, runs `w8c-h72-shard{A,B}-20261031a`, merged `w8c-h72-merged-20261031a`) | At an equal 40 fits per context, does P1 (search plus a 3-resplit check of each apparent win) yield more promotions that survive fresh cuts than P0 (search only, the live rule)? | 16 contexts / 13 families; 784 evaluations (640 search, 42 P1 verification, 102 common confirmation); 0 failed; 796 measured child CPU-seconds | **Primary, net = confirmed − false: P0 2, P1 7, P1 − P0 = +5, family-cluster 95% CI [0, 11]. The lower bound equals 0, so the result is INCONCLUSIVE under the preregistered rule.** Secondary: false promotions 8 → 1 (−7 [−13, −2]); confirmed 10 → 8 (−2 [−5, 0]); precision 10/18 = 56% → 8/9 = 89%; summed confirmation gain +0.073 [+0.001, +0.167]; family Bradley–Terry Elo P1 vs P0 **+115** (W/T/L 6/10/0). |

**How the equal-cost arm is scored.**
- Both policies read one deterministic stream from the live proposer and share the original-cut fits.
- P1 pays 3 extra fits for each apparent win inside its 40-fit budget, so it evaluates fewer proposals (29–37 instead of 40 where it found wins).
- Every program either policy promoted, and every incumbent, was refit on seeds 801–803, which no decision used. A promotion counts as confirmed iff its mean difference from the incumbent on those seeds is > 0.
- Apparent wins were 18 of 640 proposals (2.8%) across 10 of the 16 contexts. Six contexts produced none.

**What it does and does not show.**
- Original-cut promotion at the live 1e-6 margin is noisy: 8 of P0's 18 promotions fail fresh cuts.
- Spending the same budget partly on confirmation trades 2 true promotions for 7 fewer false ones.
- With 16 contexts and 13 families, the primary interval reaches zero. This is not evidence that reliability does *not* help, and it does not authorise changing the live promotion rule; that needs its own approval and a larger preregistered sample.
- Confirmation uses new cuts of the *same* training rows. It measures cut sensitivity, not held-family generalisation.
- The incumbent was frozen, not ratcheting.

**Execution and custody.**
- Run on xbabe2, CPU only, nice 15, 8 workers per shard, load-gated. Earlier shard assignments to two luna lanes were superseded before they started, so no fits were duplicated.
- The W8 evaluation ceiling of 2,400 has 760 + 784 = 1,544 used.
- Exact object versions and hashes are in [metrics_equal_cost.json](metrics_equal_cost.json). All S3 writes after the H63 diagnostic are create-only (`If-None-Match: *`) with version-pinned readback.
- The ten confirmed programs are recorded as training-CV development candidates, not serving claims, in `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/HWM_CANDIDATES.jsonl`. That file was created by this run; later appends must be conditional on its ETag.
