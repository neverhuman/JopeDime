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

Round89 also accepted a distinct parameter proposal (b24e9c46bcca6b181e2cae32535927464b2b40b19010164a04d15efdb0dd30b2) at retained LR2.5e-5, with two updates and protected dev2.267816. It produced one more strict GP promotion: cumulative promotions since the escape fix are2 across rounds88–89. The loop has progressed beyond round40; best-dev pointer now references round89.


## 17:40Z margin diagnosis and projected-optimizer fix

CPU replay at best95 measures cost(A=1d7c9ff6…)=-2.8880329132 and
cost(B=0046fb9c…)=-2.8879978657, so B−A=+3.5047531e-5. Rejected112 has
B−A=−5.0067902e-6 at LR1e-6. This supports a knife edge relative to the
actual optimizer direction, not the claim that EVERY mathematical move flips it.

Raw argmin and the dev cohort remain unchanged. The new optimizer projects only
violated low-margin incumbent rankings (global activation threshold1e-3, half of
the pinned initial margin retained). Two contexts qualify; there is no hardcoded
context8 branch. At most two small projection passes are part of each of the
original max2 optimizer steps; GP retains its64-evaluation cap and runs even if
projection fails. LR is no longer shrunk when these constraints are active.

This is explicit development-feature prediction-stability tuning: reference
choices come from the anchor model, not held-out candidate utility gradient
targets. The diagnostic is therefore not independent validation; outer/sealed
families remain excluded. No score overriding, selection hysteresis or metric
change is used. DOPE remains bit-identical and only the adapter moves.

First live result129: raw context8 margin before projection−2.7656555e-5,
after+1.8358231e-5 (pinned floor1.7404556e-5); dev2.267816067 retained, optimizer
219→220 accepted, parameter update L2=.0004647599. The correction itself was
only.0000112713 L2, leaving substantial movement in other parameter directions.
The selected context8 program retains utility10.61473568. No rollback to95 was
needed; the accepted adapter actually advanced. Full live proof follows in the
round receipts and committed projected-margin evidence.


Independent CPU raw-argmin replay of accepted checkpoints129/130 confirms the
original context8 winner and utility10.61473568 with mean2.267816067. Optimizer
steps advance219→220→222 and accepted model movement is.00046476 then.00092217
L2; no checkpoint rollback was used. The complete non-adapter state and DOPE
remain identical to95. Both live steps satisfy the pinned margin constraints.
Thirty focused tests pass, including repeated projection that preserves the
ranking while continuing progress along an unconstrained parameter direction.


2026-10-29 18:35 nudge — allocation slowdown and exact device cache.
Read-only live profile156–158: 2,642,384 minor faults, 35.31s kernel CPU,
zero major faults/read bytes/process swap over127.55s. Legacy training rebuilt
2.592GB feature slates per step, plus temporary copies and CPU→GPU transfer.
DevicePoolCache now retains context rows and gathers identical slate indices
on-device; promotions/replacements invalidate the affected context. No score,
target, RNG or selection changes. Startup real-data fields and predicted costs
are bit-identical (maximum cost delta0). Own worker resumed completed161 only.
Rounds162/163/164: total57.54/59.20/63.77s; training2.110/2.001/1.996s;
packing.070/.060/.056s. Immediately before: rounds160/161 total99.88/103.24s,
training38.05/37.14s; r154 training56.48s. Input/forward now.60–.62s,
CPU index transfers25.17MB versus5.184GB avoided feature slates per round.
All three rounds:2 optimizer steps,41 changed tensors,positive update norm,
64 native evaluations,raw dev2.267816067203105,DOPE c51a56d0 unchanged.
Post profile (includes startup)127.58s:286,999 minor faults,2.17s kernel CPU,
14 major faults/2.87MB reads,zero process swap; peakRSS in durable receipt.
Current training overhead is flat across these three observed rounds; GP wall
variation remains. Oct23–24 baseline183.26/133.06/211.24s covered GP-only;
current totals include training+GP, so no matched speedup is asserted.
35 focused tests pass (five cache tests), ruff/check/format,diff,paper-trace pass.
Loop continues with unchanged campaign budgets; no Phase2 or xbabe0 access.


1935 nudge — GP promotion drought audit and independent exploration trigger.
Audited187–205:128 proposals/round,125–128 survive pool dedup,64 native CV
evaluations/round. All1216 completed:1211 worse than current incumbent,one
tie within1e-6,four CV failures. No eligible strict improvement was suppressed.
Per-context best utility/incumbent/reasons and candidate-level classifications
are retained in plateau-audit/round187-205.json; compact full context tables
are committed in gp_plateau_audit.md/json. Incumbents reconstructed from
passing original pools plus chronological journal; selected IDs matched scores.
Pool dedup does not retain rejected candidates, so repeated evaluations remain
a search-efficiency limitation. Training device cache is outside native GP
admission, and invalidation after185/186 is visible in next-round transfers.

Native GP dominates200/201 spikes:100.56/91.11s of103.09/96.43s total.
Same-context/same-DSL matched wall times rose1.61x/1.73x (20/23 matches).
Round201 CPU slate assembly.67→2.45s,device packing.055→.279s; model/targets
only.54→.62s. Consistent with temporary shared-host scheduling/CPU contention;
no historical host samples identify a particular consumer. Recovered afterward.

Reproduced policy liveness defect: accepted adapter ties always reset the only
GP exploration trigger (which watched rejections), even with18+ GP rounds
without promotions. Fix separates GP progress from adapter admission. Before
new outcomes, preregistered18-round threshold,3 exploration/3 cooldown rounds,
using existing16-proposal generator/two ranked+two seeded random choices.
Only actually completed policy rounds advance phase; historical rounds do not
invent exploration. Promotion/unknown/missing/zero-evaluation history resets
the drought. Evaluation cap4/context,64/round and max2 updates unchanged.
No DOPE,raw-dev,LR,Phase2,sealed/outer or other-job change.48 focused tests
pass,including reproduced failure,burst/cooldown,reset and installation clocks.
Installation uses one own-worker completed-checkpoint handoff; live proof to
follow without promising promotions or independent quality improvement.


Live GP-drought fix qualified on source2547632664f196d676ad1bbbc6c0c61baa174b6a.
Own boundary resume210→211 under existing campaign lock. Startup cached/legacy
fields and costs bit-identical; preregistration hash retained in source manifest.
211/212 explore at policy phases0/1:256 proposals,64 native evaluations each,
wall54.57/63.25s.211 has no promotion;212 admits one strict train-CV improvement
in context0411fbc1:utility.18569920995→.18586409938 (+.00016488943),from
seeded-exploration slot2 (zero-based),program0b0d5658b3da5ed63b89bf0d4e8b22c7582e1ca526ecbd6adda28f2c5f269def.
No confidence interval or independent validation is available for this gain.
213 resets drought to0 and returns to128 proposals/ranked selection,64 evaluations,
wall60.41s. Its one device-pool transfer confirms invalidation after promotion.
All three:2 optimizer steps,41 changed tensors,29,211 trainable parameters,
positive adapter movement,rawdev2.267816067203105 and DOPE c51a56d0 unchanged.
The existing adapter-rejection escape remains separate; the3-round cooldown
applies to the new drought trigger.48 focused tests/ruff/format/paper-trace pass;
full hosted CI is unqualified (prior run has failed jobs),not a merge gate pass.
Live worker continues PID441927,log continuation-gp-drought.log.
