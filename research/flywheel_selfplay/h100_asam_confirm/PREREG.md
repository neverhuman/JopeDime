# H100: ASAM confirmation and drift control

Frozen before H100 timing, optimizer updates or outcome reads. Owner: sol_v1_hwv.
This is conditional evidence on the reused H50 development packet; no serving promotion.
H82 is complete (#978), with no admitted arm. Its ASAM gains could reflect less drift
from r882 rather than an additional benefit from flatness-aware updates.

## Fixed intervention and inputs

Start every cell from an offline COPY of r882, SHA-256
`0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64`.
Freeze DOPE `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`;
train exactly 29,211 non-anchor adapter parameters. No live-loop, sealed, holdout,
outer-test or meta-test payloads. Use the real-S3 H50 packet only: all 1,219
DL-Train contexts / 205 families, with terminal unrevealed-only scoring on the
91 dev contexts / 19 disjoint families. Retain H82's manifest, roster, fused-model,
sampler and evaluator pins. The historical seed-4900 folds are lineage only;
no fold subset replaces the full training roster.

Fresh paired seeds: **39, 40, 41**. All arms use H82's 20 rounds, two updates per
round, batch 16, slate 192, H59 CONTROL loss, AdamW weight decay 1e-4 and gradient
clip 1. The final adapter is fixed; no round selection or early stopping.

- CONTROL: H82 CONTROL, constant learning rate 1e-4.
- ASAM05: H82 ASAM exactly, rho 0.5, learning rate 1e-4.
- ASAM10: same ASAM update with rho 1.0, learning rate 1e-4.
- DRIFT_MATCH: CONTROL with constant learning rate **8.465252628604803e-5**.

ASAM uses T=abs(weight)+0.01 for parameters whose name contains `weight`, T=1
otherwise, perturbation rho*T^2*g/(||Tg||+1e-16), matched stochastic state for
both passes, exact parameter restoration, clipping of the second gradient and
one AdamW step. CONTROL/DRIFT_MATCH use 40 backwards; ASAM arms use 80.
Every terminal cell also scores the untouched r882 adapter and then restores
its candidate. Same-seed reference utilities must agree across arms.

## Historical drift calibration, fixed before fresh seeds

H82's summary has no adapter-L2 field. Derive it on CPU from the six hash-bound
CONTROL/ASAM snapshots at seeds 24–26 and the pinned initial r882 state, using
all 29,211 adapter scalars and float64 Euclidean distance. Mean distances are
CONTROL **0.4485506635796817**, ASAM **0.3797094683930329**. The one-shot ratio
is **0.8465252628604802**; multiply CONTROL's learning rate by that ratio.
`calibration.json` retains each distance, snapshot hash and H82 summary hash.
No H100 pilot, outcome, dev payload, optimizer update or tuning enters this fit.
The ratio is a first-order calibration, not a guarantee of matching fresh-seed
drift. Measure the achieved distance and never retune the learning rate.

## Primary, multiplicity and harm gates

Require all twelve unique paired cells, unchanged source/input/checkpoint and
parameter inventories, and exactly 19 common dev families. Missing, duplicate,
void, unknown or nonfinite outcomes block admission; never replace them by zero.

For each candidate, compute its per-seed mean family utility gain versus fresh
same-seed CONTROL, then the median over three seeds. Use 10,000 common family-
cluster bootstrap draws, RNG 100000, retaining all seed vectors together. Report
two-sided CI95 and one-sided fifth-percentile LCB95. The raw gate is strictly
LCB95 > 0.001546 and median >= 0. Ties fail.

Apply Holm step-down across ASAM05, ASAM10 and DRIFT_MATCH at family-wise alpha
0.05. The preregistered one-sided percentile-bootstrap tail area at the margin is
p=(1 + number of bootstrap gains <= 0.001546)/10001. Sort by p, break ties by arm
name; require p strictly below 0.05/(3-rank), stopping at the first failure.
Report raw tail areas, adjusted values and decisions. These are approximate
family-bootstrap tests on this exposed panel, not independent qualification.

For the harm gate, average each family's candidate-minus-r882 utility over the
three seeds. A family loses if this mean is strictly below -0.01. Require the
fraction of losing families <= 0.10 (at most one of 19); also report each seed's
harm fraction descriptively. An arm admits only if raw, Holm and harm gates pass.

## Descriptive and interpretive readouts

Report family-grouped Elo versus CONTROL=1500 and r882=1500, per seed and median,
with paired family-bootstrap CIs, train/val gaps, and all adapter L2 distances.
Top-four overlap is intersection size / 4, using only indices 4..P-1, ascending
prediction with stable index ties; average contexts within families, then families.
Compare each final candidate with its restored r882 reference. Cross-arm overlap
uses saved terminal indices only after the whole panel completes.

Operationalize “DRIFT_MATCH ~ ASAM05” prospectively: the actual mean L2 ratio
DRIFT_MATCH/ASAM05 lies in [0.90,1.10], and the paired family-bootstrap 90% CI for
median ASAM05-minus-DRIFT_MATCH utility lies wholly inside [-0.001546,+0.001546].
If both hold, interpret H82's gain as consistent with drift control, without an
established additional flatness benefit. If either fails, that explanation is
unresolved; failure to reject a difference alone is not equivalence. Neither
pattern proves a unique causal mechanism. No adaptive follow-up is authorized.

## Execution and budgets

CPU preregistration/runner/tests/dry run and PR merge precede any H100 ticket or
CUDA initialization. Dry run opens only pinned source/roster metadata, never dev
payloads. A waiter stays outside FIFO until H93, H97 and H99 have merged terminal
metrics with closed ledgers and no live coordinator/ticket; an inter-cell gap
is insufficient. Then use qualification_lock_fifo.py, wave_name=H100, the actual
~/jope_plan/qualification.lock, a new tail ticket per cell and release each time.

Nice 19, OMP/MKL 4, BLAS 1; load <=85/102 twice 30 seconds apart before acquisition
(up to 30 minutes), RSS <6 GiB, MemAvailable >=16 GiB, launch disk <94.5% and writer
disk <95%. No other job or screen is paused or killed. Stage into an isolated
scratch/<cell>-attempt1 directory; an existing scratch copy must match r882 and
may be reused read-only. Never chmod or overwrite the source checkpoint.

Hard cumulative GPU-cell wall cap: **2700 seconds**, including setup, one discarded
ASAM10/seed39 train-only timing cell, scoring, failures and shutdown. Timing
projection must fit with 25% headroom. Separate lock/resource wait allowance is
3 hours after precursor closure. No cell retries are authorized for H100; a failed
attempt stops the wave with unknown missing outcomes and no admission. Never
reset a ledger or deadline. Frozen source and this preregistration are recorded
in every receipt. No outcome values are inspected before all twelve cells finish.
