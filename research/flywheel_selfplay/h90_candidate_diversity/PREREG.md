# H90: candidate diversity from fixed training recipes

Registered before H90 timing/training/outcomes under
`h90_cand_diversity_t0130.md`, including both 01:33/01:34 MDT addenda.
Conditional development experiment on an offline r882 COPY; no live-loop,
unseen-family generalization or serving claim.

## Prior evidence and specific differences

[H46](../../../docs/research/studies/dope-neural-ranking-20261021/h46/README.md)
already tested warmup-cosine and family-balanced loss without a qualifying win.
It started from r785 and combined some recipes with longer training/EMA and
validation-selected trajectories. H90 does not present those ideas as new:
FBGA changes sampling instead of H46's loss weights; WCR has a round-11 restart.
All H90 arms have matched fresh CONTROL and fixed-final evaluation.

[H49](../../../docs/research/studies/dope-neural-ranking-20261021/h49/README.md)
found distinct control adapters with identical dev utility (effective seed
replicates one). [H51](../../../docs/research/studies/dope-neural-ranking-20261021/h51/README.md)
also tried balanced minibatches and combined clipped/smoothed rank recipes,
admitted no paired candidate, and documented step-like utility, seed dependence
and checkpoint exposure to inner folds. H90's utility-mean smoothing differs
from that softened-rank/SWA recipe; balancing/smoothing are not untried ideas.
New comparisons use paired seeds27/28/29, r882, H82 CONTROL and explicit
selection-overlap/adapter-distance diagnostics. Reused dev remains exploratory.

## Frozen science and inputs

Reuse H82/H76/H59/H55 loaders/scoring: real-S3 H50 packet, 1,219 DL-Train
contexts/205 families, 91 dev contexts/19 disjoint families; seed-4900 whole-family
lineage. All train contexts are used, including historical fold B; r882 is not
an unexposed inner-fold baseline. No sealed/holdout/outer payload reads.

- Manifest SHA: `99a53569fcaf59f9309e69329c830ae1684adff382da3ae22e0b2645c4ebc4d4`.
- Selected roster: `981960ad628f5029f8c54552c7b1cb1d440072e7e9dcffecdea52c6a3493a0f1`.
- Fused source: `b9f1468a61aa4dc888c8f2931f9c759f7d4f057793a77ccb05eed6300fb4f9ab`.
- Every cell starts from checkpoint COPY
  `0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64`.
- Only29,211 non-anchor adapter parameters train. Check inventory and frozen
  DOPE `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`
  before/after every cell. Never access the live xbabe1 run directory.

CONTROL is exactly H82 CONTROL:20 rounds,2 AdamW updates/round, batch16,
slate192, lr1e-4, wd1e-4, final-gradient clip1, H59 CONTROL
SmoothL1(beta1)+0.1KL(temperature0.2). Reuse actual H76 keyed planner/optimizer.
Every arm has40 optimizer/decay applications and640 context draws. No early
stopping, dev selection, replacement seed or best checkpoint.

## Arms fixed before timing

- **CONTROL:** unchanged above.
- **FBGA:** SHA-sorted contexts, same keyed seed/round/step stream, weight
  1/(train family size) per context, giving equal total family sampling mass.
  Two microbatches8 per update: each loss weighted1/2 and backpropagated before
  the next forward; zero once, clip once, AdamW once. No loss reweighting.
- **LS:** after existing H59 gate/utility clipping, blend each sampled context's
  192 targets toward its own mean, eps0.1. Cost notation t'=0.9t+0.1mean(t).
  Both unchanged SmoothL1 and KL formulas use t'; no extra entropy/probability
  mixture. CONTROL/LS/WCR share identical context and candidate streams.
- **WCR:** both updates in each round use its LR. Round1=1e-4, round2=2e-4;
  rounds3..10: 1e-5+(2e-4-1e-5)*(1+cos(pi*(round-2)/8))/2.
  Restart round11 at2e-4; rounds11..20:
  1e-5+(2e-4-1e-5)*(1+cos(pi*(round-11)/9))/2.
  Rounds10/20 end at1e-5. The owner specifies warmup rounds1..2; no second
  warmup is added at the restart.

Seeds **27,28,29** paired across arms. First WCR/27 timing runs training plus
train-only scoring, discards scores/weights and opens no dev payload. Then12
outcome cells: seed-major, CONTROL/FBGA/LS/WCR. No post-timing recipe changes.

## Readouts and pseudoreplication

Keep terminal unrevealed-only selection4..P-1, utility clip[-2,4], failed gate=-2.
Each CONTROL additionally scores its original r882 adapter on the same dev
packet, then restores/verifies candidate weights. Primary: each seed's
equal-family mean gain vs same-seed CONTROL, then median of3. Bootstrap10,000
common19-family cluster draws with RNG **90000**, keeping complete paired seed
vectors. Two-sided95% percentile CI; one-sided LCB95=5th percentile. Admit only
LCB95>.001546 strictly AND median>=0; ties fail. All12 complete unique cells
required; missing/void is unknown, never zero. Three-comparison Bonferroni LCB
is diagnostic and does not change the gate.

Report H82 family-grouped Elo vs CONTROL1500 and fresh r8821500, per seed and
seed median with shared-family bootstrap95% intervals. Explicit strings retain
infinite Elo endpoints. No AUC. Report train/dev gaps.

For every arm, group exactly identical dev context-utility vectors, family
vectors and scalar equal-family utilities across seeds; report groups and
adapter hashes. Identical utilities flag pseudoreplication, not independent
confirmation. Never replace such seeds or count families three times. These
are conditional family intervals, not seed-noise intervals; the flag does not
silently modify the fixed admission gate.

Descriptive diversity: stable ascending-cost top4 unrevealed candidates per
context. Compare their program-SHA multisets against same-seed CONTROL's fresh
r882 reference: sum of minimum multiplicities/4, mean over91 dev contexts.
This handles duplicate program identities consistently. Report per-seed and
arm mean/median overlap plus raw Euclidean adapter L2 distance from initial
r882. Diversity never changes admission.

## Execution and hard H80 exclusion

CPU source/tests PR and merge precede execution; CPU dry-run validates only
metadata/train identities/plans,0 dev payload reads and0 updates.
**No H90 GPU ticket until explicit H80 closeout evidence AND no H80 ticket in
`~/jope_plan/qualification.queue/`.** Empty queue between H80 cells is not
closure. Enforce before FIFO acquisition; CPU preparation/waiting outside the
queue may proceed. Then use qualification_lock_fifo.py, wave_name H90, a fresh
FIFO-tail ticket per attempt, prompt release,>=30s yield and legacy backoff.
Respect older authorized tickets, especially H82.

Reuse H82 guards: nice19, OMP/MKL4, BLAS1, RSS<6GiB, MemAvailable>=16GiB,
CPU<=85%, launch disk<94.5%, writer<95% and>=1GiB free. Failed launch guards
WAIT; delete nothing. No xbabe0/3, live-loop or foreign-job changes.
Only host-load aborts before the durable full-scoring boundary retry, at most
2 additional attempts, same seed/checkpoint/stream. Other errors, uncertain
phase or scoring started refuse retry. Charge every attempt/void/setup/score
against one non-resettable **2,700s GPU-cell-wall cap**,10s shutdown reserve;
unknown in-progress launches forbid resume. Timing gate: prior charged voids
+successful timing wall*13*1.25<=2700s. Keep H82's cumulative3h resource/lock-wait
cap after admission; H80-exclusion waiting creates no ticket or GPU charge.

Closeout: compact README/metrics, one overnight-hwave-index row and claim/
launch/closeout receipts. Full artifacts outside Git. If an arm clearly admits,
flag the flywheel owner by nudge; never modify the live loop.
