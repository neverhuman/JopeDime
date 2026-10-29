## Regression investigation — 2026-10-29 16:35Z nudge

Paired CPU replay of checkpoints 38, 40, 41 and 60 on identical dev inputs
reproduces the recorded utilities exactly: 2.267816, 2.267816, 1.147988,
and 1.142684. Every non-adapter tensor is bit-identical across these checkpoints;
DOPE and the evaluator did not change. This is a real adapter ranking regression.
The eighth context changed from program 1d7c9ff6… (utility10.614736) to
0046fb9c… (1.656109). The nudge's afded78… names a different context's program.
The top-two predicted-cost gap narrowed to0.004075 at round40 then crossed by
round41; this is not merely changed reporting.

Best-dev rollback restores model AND AdamW state, keeps an immutable checkpoint
pointer with comparison identity/hash, and logs each decision in
adapter_admissions.jsonl. Strictly worse proposals are saved under rejected/ and
are not used for GP selection or the next round. Equal dev scores may advance
training; this avoids freezing all learning during long argmin plateaus. The
selected best dev cohort is repeatedly observed and is not a sealed-quality test.

The round-13 resume path was read once at process startup, not every round.
Optimizer steps progressed206→210→212→250 at rounds38/40/41/60; LR stayed1e-4.
Two updates at batch8192 mean16,384 stochastic presentations per round, over only
16 unique training contexts. Learning is occurring (41 tensors change each round),
but the regression/KL training loss is poorly aligned with dev argmin utility.
Flat noisy batch losses do not justify increasing LR or steps after this collapse.
Keep the existing two-step/LR budget for now; log update L2, gradient norms,
optimizer step counters and effective LR to support a controlled later change.

Wall-time audit: mean wall14–35 was70.60s (training23.39s, other47.22s);
rounds37–60 averaged79.10s (training31.10s, other48.00s). Summed native-worker
wall actually stayed roughly62.22→61.17s. Most drift is therefore in training,
not more GP evaluator work. Historical receipts do not separate slate assembly,
GPU work and CPU headroom waits, so a specific cause cannot be inferred yet.
New receipts time these phases separately. Old133–211s numbers are GP-only and
are not an equivalent full-round speed comparison.

DOPE remains c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814;
Phase2 kernels and sealed/outer rows remain outside the loop.

### Live guarded-round proof

Round69's proposed utility1.147988 was rejected. Accepted utility was restored to
2.267816. CPU readback proves BOTH the saved model and AdamW state in round69.pt
are bit-identical to round40.pt; rejected/round69.pt preserves the different
proposal. Candidate step212 rolled back to incumbent step210. Future rejected
rounds may intentionally repeat those counters under this guard; that is logged
rollback, not reloading round13 each iteration.

Round69 phase timings: headroom0.50s, slate assembly0.67s, forward/input/targets
28.35s, backward+optimizer0.18s, dev gate0.17s, GP40.96s; total70.92s. The new
measurement localizes training overhead to forward/input/target handling. It does
not separate CPU input packing/H2D from GPU forward kernels, so attributing the
historical drift specifically to host contention remains unproven.

### Repeated-rollback correction

The round69 single-rollback proof was correct, but later step counters exposed
optimizer-state aliasing: load_state_dict reused the cached AdamW tensors, and
subsequent opt.step calls mutated the saved snapshot. Later pre-fix admission
rows describe the intended rollback; exact optimizer restoration after the first
was not established. Model weights/dev utility remained protected. A new repeated
rejection test reproduces the failure (step2 instead of1 on the second rejection).
The fix deep-copies optimizer state on EVERY restore and makes the persisted best
pointer authoritative over equal-score logical checkpoint copies at restart.
All old checkpoints/receipts are preserved; do not treat them as corrected proof.


Repeated-rollback verification now passes in live rounds78 AND79: both saved
model states and complete AdamW states are bit-identical to round40, both raw
rejected optimizers reached212, and both accepted-state checkpoints retained210.
Retained dev utility remains2.267816 and DOPE is unchanged. The focused suite now
has24 passing tests, including the must-fail-before-fix repeated-rejection case.
The loop is running under the corrected guard; no learning-rate/step budget or
other owner's job was changed.


## 17:05Z fixed-point escape — first live result

Round88 is the first alternative proposal: seed29477181, one optimizer
update at LR2.5e-05, training order rotation88.
Its dev mean is2.267816067, different from the repeated
1.147987758 rejection; it was accepted as a tie to the protected best. Parameter
SHA-256 is fd27311bffda8a2a757182d72337393a1383691aaef8a0667a7d35efebd4085c; update L2 is0.00760390.
DOPE checksum remains c51a56d0… and only the adapter trains.

GP stagnation mode generated16 candidates/context, rotated parents, discarded
pool duplicates, and selected two model-ranked plus two seeded exploration
candidates for the same max4 native evaluations/context. Actual native evaluations
64, gate passes62, new strict promotions1.
GP used the retained/admitted incumbent and ran independently of adapter rejection.
Round88 wall was58.39s. The maximum two-update/64-native-evaluation
budget, original CPU/GPU/storage limits and Phase2/sealed-data exclusions persist.
Gentle accepted LRs persist; later rejected proposals get another bounded reduction
and a new round seed. This protects dev utility while permitting real parameter
progress rather than repeating the rejected full-sized move.
