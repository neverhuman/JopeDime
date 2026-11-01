# RNK-10 R10: train-only landmark causality candidate

State: preregistered before candidate training. This is a single CPU research candidate, not a production release or qualification claim.

## Frozen source and initialization

- Full-cohort trainer source commit: `3b4f349fd4691db1672ad9243d9cc056b7e055f0`, recorded by `flywheel-full-20261030/run-20261030/OPERATIONS.md` on xbabe1. That source tree is read-only for this work.
- Round-882 checkpoint: `sha256:0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64`.
- Signed starting bundle: watcher-verified R882 package SHA-256 `7523af9700e21325a69f9f71b8b60eb4897d46f3948546e4024010b1ce129a60`, bundle digest `057e50a9a741d30d541ed8af8339cec84a5157af621100cb70857052ef14dacc`, JOPE weights SHA-256 `ff3f9ea5d76fa6c13cd74d40a9effbf7e9c891d98746dc542d851726b9e3138c`, DOPE weights SHA-256 `0f5e98fa29cf9c45bfaac5a3cf001b13a0de37d7b6e154c582d7ec689b21a84d`, release catalog SHA-256 `3af56a817bfae39b70f0b2f60668fe88fae178650ee596e1ba7a180acba1e0c6`, and signed message SHA-256 `45e21d576cec91b5171bb8e279ef24fef761456fa916e4de1ed8ed2b8bd6eba5`. The canonical signed bundle is `/home/ubuntu/jope-data/RNK10-t0140-t0250-rnk11/verified-r882`; the detached signature receipt is `/home/ubuntu/jope_plan/signing/receipts/release-r882-20261101T094716Z.json`.
- R882 preregistered manifest SHA-256 `f3ea7cfaeda35a3e768bfc21827a84debc48adb3791fd5e613471b28b1b7fdba`; selected-roster SHA-256 `981960ad628f5029f8c54552c7b1cb1d440072e7e9dcffecdea52c6a3493a0f1`; source-manifest identity `3eb4a62acea512f528500b043299a5968e1c9b1d1b1759bbf77a426b9fa1ecc3`.
- DOPE is frozen at `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`.
- Only the 29,211 non-anchor adapter parameters may update. Model and CPU fused-source hashes are verified against the R882 provenance packet before execution.
- Supporting training helpers are read from immutable source commit `3b4f...`: `automatic_promotion.py` SHA-256 `37e50568fd097944cf7125de340a19da3dbf27f7be80f47e28fb93b6cc24d16d` and `device_pool_cache.py` SHA-256 `cdcb1d672a643716ffba3f2c20aa75d5ae0b19766db616ccd6f6ad65a9adbd76`.
- Landmark preprocessing inputs are pinned before execution: `JopeDimeMicro/python/hpc/landmark_workspace.py` SHA-256 `834b3a13db9a257f2b8f3cb892cbdd5571092a41b0e5257826b7d9d3cf61f91a`, `python/research/levels/building_block_landmarks.py` SHA-256 `fc2cc82ec4c654a512f592211e21a940a2c7a5baf3e002fe5a1e7649fdfb9186`, `python/research/levels/descriptors.py` SHA-256 `18cfefdddca262616dadd4841119abfd2b073f9e2ee28fb58c31cad09dd1b928`, and `bin/libjope_hpc.so` SHA-256 `f37f3f4d8ed9954a8ceb23baca08f8b08a79234ad3afd7f90638d68242484901`.

The full-run sidecar states that a per-round run manifest for round 882 is absent; the exact training commit basis is the `OPERATIONS.md` record above. No missing run manifest is fabricated.

## Fixed cohort and treatment

- Use only the 1,219 approved `DL-Train` regression contexts and their train-only two-fold JTT1 tables and candidate pools. The roster contains 205 families. Input preparation filters to the `train` role before copying or opening payloads.
- Do not open DL-Val, sealed meta-test, outer rows, or holdout payloads during candidate training.
- For each context, the baseline uses its generated landmark tokens. The paired ablation zeros token slots `[0, 1, 7, 9]` and retains every other token and mask bit, matching the unchanged M5 baseline-versus-ablation control.
- The auxiliary target pair is selected deterministically among gate-passing unrevealed candidates (indices 4 onward): the lowest and highest train-only fitness, with source-order tie breaking. No selection or hyperparameter tuning follows candidate outcomes.

## Objective and optimizer

The original train regression objective is retained exactly: smooth-L1 predicted cost against the gated fitness targets (`2.0` for gate failures), plus `0.1 * KL` between predicted and target softmax distributions at temperature `0.2`.

Exactly one auxiliary term is added. For each context, let `base_margin = cost_base[worst] - cost_base[best]` and `ablated_margin = cost_ablate[worst] - cost_ablate[best]`. The auxiliary loss is `relu(0.04208 - (base_margin - ablated_margin))`, averaged over contexts. The preregistered coefficient is `0.1`. `0.04208` is fixed at one tenth of the existing F12 cost scale `0.4208`. The term directs train-label ranking margin through landmark features relative to the ablation while leaving M5's acceptance predicate unchanged.

Initialize from the signed R882 weights; use source slate size 192 and source `training_indices` selection; AdamW (`lr=1e-4`, `weight_decay=1e-4`), global gradient clipping at `1.0`, seed `20261101`, and exactly two optimizer updates, the full-cohort operation's existing maximum. Each update traverses the fixed 1,219-context train roster in microbatches of 8 with gradient accumulation; all contexts receive equal weight. Keep train mode enabled; replay the baseline RNG state for the paired counterfactual forward, then restore the post-baseline state so paired dropout masks match without changing random consumption. Run on xbabe2 at nice 19, at most 8 PyTorch/BLAS threads and 4 landmark-workspace threads, with CPU use kept at or below 85%. No GPU and no flywheel-loop process or source mutation.

## Acceptance and handoff

The code regression must prove the auxiliary loss is zero after its margin is met, positive below the margin, and supplies a nonzero gradient to both baseline and ablated cost paths. The candidate run must bind the exact source, input roster, checkpoint, DOPE identity, optimizer updates, resource limits, and output hash. The candidate is not admitted based on training loss.

After training, rnk02 evaluates the candidate against R882 on the unchanged dev-fold Elo procedure and performs the unchanged M5 baseline-versus-ablation control on the signed-R882 replay. Report M5 as failed unless the original predicate passes; preserve the existing val_reg utility and Elo method. This one candidate does not trigger another A/B study or any default/serving/qualification promotion.

## Operational recovery note (before persisted candidate rerun)

On 2026-11-01, an initial process completed both preregistered optimizer updates on xbabe2 but failed at Safetensors serialization because the model state dict contains tied aliases for frozen anchor parameters. It wrote no candidate file or receipt, and no M5/Elo or validation outcome was read. The emitted training diagnostics were: step 1 regression `3.6877517332494794`, hinge `0.04109427094508429`, gradient norm about `0.454`; step 2 regression `3.6873105067127154`, hinge `0.04106810149890269`, gradient norm about `0.457`. These values are retained only to document the failed export attempt. Before the next process, the exporter is changed to clone each state tensor into independent CPU storage, with a focused regression. The persistence recovery repeats the same source, signed initialization, cohort, seed, optimizer and exactly two updates; it introduces no parameter or outcome driven changes. Only a successfully written candidate and receipt will be handed to rnk02.
