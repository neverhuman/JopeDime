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
