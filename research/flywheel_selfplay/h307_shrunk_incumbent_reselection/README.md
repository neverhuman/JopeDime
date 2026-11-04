# H307: robust offline incumbent reselection

**RESELECT_NOT_ADVANCED. r3175 stands.** The frozen non-S0 choice is S1
ARGMAX_FAMILY_MEDIAN. It selects r3175 on all three primary splits, so its
median EVAL gain versus r3175 is **0.000000, 95% CI [0, 0]**, with LCB95=0 on
all three splits and zero wins versus S0. Family-level MM Bradley–Terry Elo,
anchored to actual r3175=1500, is **1500 [1500, 1500]**. No serving, HWM,
checkpoint, training or flywheel change follows this analysis.

## Hypothesis and frozen setup

Can offline robust ranking of all recorded candidates identify an incumbent
that generalizes better across dev families than the exposed full-dev maximum?
[PREREG.md](PREREG.md) was pushed as `0cbc429eb7f2e8092fe6146f7ea01633fb613def` before any new EVAL computation.
The tested analysis source was pushed as `ef3a5a97436d515a76f393a2afcc0fcf0dbbc98a` before the reselection run;
its exact file hashes match the execution lock and resource receipt.

The immutable H289 closure contains **1,500 evaluated rounds r3175–r4675**,
excluding unevaluated r4331, **91 dev contexts and 19 families**. No earlier
unrecorded candidates are implied by “all history.” The input is recorded
utility, not a model, table or fresh evaluation. See exact S3 VersionIds and
SHA256s in [input_receipt.json](input_receipt.json) and [manifest.json](manifest.json).
Evaluation identity is `e79a7715c1531ccb8c4055ab96971101cd4034ec3a71d83bca6d25b0e604ca83`.
DOPE `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`
is untouched. Optimizer updates=0, forwards=0, GPU use=0, fresh-pool reads=0.

Regression ran first and passed: r3175 context mean
**1.1389572403748425**; all 1,500 journal utility and evaluator-identity checks;
H304 R0 seed30401 in both family-equal and context-mean selection modes.
Reduced statistics match within 2.22e-16 (declared tolerance 1e-12), including
zero promotions, final r3175, SELECT/EVAL levels, precision and shrinkage.
See [regression_receipt.json](regression_receipt.json).

SELECT/EVAL splits match H304 SHA256 parity seeds30401/30402/30403:
10/9,12/7 and6/13 families. Sensitivity uses H304's exact RNG304150 permutation
stream for 200 unique9/10 balanced splits. Each candidate is ranked using SELECT
families only; EVAL cannot enter candidate scores. Five preregistered rules:

- S0: largest mean family utility.
- S1: largest median family utility.
- S2: largest one-sided q05 of 1,000 whole-family bootstrap means.
- S3: empirical-Bayes moment shrinkage of SELECT means toward their grand mean,
  using family-bootstrap variance and a nonnegative between-candidate variance.
- S4: smoothed SELECT mean over recorded rounds within numeric±5; select the
  center checkpoint, not an ensemble.

Scores within 1e-4 of the global maximum tie; earliest round wins. Three primary
pick locks and200 sensitivity pick locks were written before their EVAL outputs.
S1 was chosen by the seed30401 EVAL tie-break, then locked before30402/30403 EVAL.
No choice was revised after sensitivity. LCB95 is q05; CI95 is[q025,q975].
Per-seed reporting uses 10,000 whole-EVAL-family draws, RNG304150. Median intervals
share 19-family bootstrap draws across overlapping EVAL halves. Picks stay fixed.

## Primary outcomes

Columns list seeds30401/30402/30403. Utility is family-equal; Elo is a family-match
comparison against actual r3175, not the r882 slot0 reference used by H303/H306.

| Rule | Picked rounds | EVAL gains vs r3175 | Median gain [95% CI] | Median Elo |
|---|---|---|---|---|
| S0 | 3175/3175/3175 | +0.000000/+0.000000/+0.000000 | +0.000000 [+0.000000, +0.000000] | 1500.00 |
| S1 | 3175/3175/3175 | +0.000000/+0.000000/+0.000000 | +0.000000 [+0.000000, +0.000000] | 1500.00 |
| S2 | 3175/3182/3175 | +0.000000/-0.016731/+0.000000 | +0.000000 [+0.000000, +0.000000] | 1500.00 |
| S3 | 3175/3175/3175 | +0.000000/+0.000000/+0.000000 | +0.000000 [+0.000000, +0.000000] | 1500.00 |
| S4 | 3175/3215/3212 | +0.000000/-0.017133/-0.003184 | -0.003184 [-0.010348, +0.000000] | 698.27 |

All rules pick r3175 on30401. S2's alternative on30402 loses0.016731 utility;
S4's alternatives lose0.017133 and0.003184 on30402/30403. S2's median zero and
zero-width median CI reflect two exactly unchanged seeds, and do not erase the
third seed's loss. S4 median gain CI is[-0.010348,0], and median Elo CI is
[494.71,1500]. Full per-seed gains, paired differences from S0, family deltas,
W/T/L, SELECT-minus-EVAL shrinkage and Elo intervals are in [metrics.json](metrics.json).
Ties never count as wins. Elo uses0.01 symmetric decisive-count smoothing;
all ties give1500. This differs from H304's context-count Elo/0.5 smoothing.

The preregistered “generalization-adjusted plateau” is S1's **median EVAL
family-equal level 1.0732784011, CI[0.6012168035,1.7965618713]**. r3175 has exactly
the same EVAL level and interval. Its full19-family equal-weight level is
**1.1547293570**, while its full91-context mean is**1.1389572404**. These use
different subsets/weighting; their difference is not a measured generalization gap.
The wide level interval records between-family heterogeneity.

## 200-split sensitivity and stability

These descriptive splits do not change the frozen primary decision.
W/T/L uses a strict1e-4 utility threshold, not raw numerical sign.

| Rule | Mean gain vs r3175 | Mean gain vs S0 | Splits W/T/L vs r3175 | Pick r3175 | Unique picks | Mean SELECT-minus-EVAL gain |
|---|---|---|---|---|---|---|
| S0 | -0.002387 | +0.000000 | 0/125/75 | 125/200 | 3 | +0.003608 |
| S1 | -0.000561 | +0.001826 | 0/185/15 | 167/200 | 3 | +0.000488 |
| S2 | -0.002605 | -0.000218 | 0/121/79 | 111/200 | 5 | +0.003633 |
| S3 | +0.000000 | +0.002387 | 0/200/0 | 200/200 | 1 | +0.000000 |
| S4 | -0.005658 | -0.003271 | 6/38/156 | 7/200 | 10 | -0.000992 |

S0 picks3175/3182/3381 in125/14/61 splits; all 75 non-tied changes lose versus
r3175 on EVAL. S1 retains r3175 in167 splits and has only 15 strict losses,
versus 75 for S0; its +0.001826 mean advantage over S0 is avoided harm, with
no strict EVAL wins against r3175. The complete pick histogram and per-split
outputs are preserved in S3. S4 is least stable (10 different rounds) and has
156 strict losses against r3175, despite six positive sensitivity splits.

**S3 degenerates to retention.** Its estimated between-candidate variance tau²
is zero on all primary splits; all shrinkage factors are zero. All200 sensitivity
picks are r3175. Its apparent +0.002387 mean gain against S0 comes from retaining
the incumbent, not discovering a stronger checkpoint. Floating residuals below
2e-17 in its sensitivity utility differences are arithmetic roundoff and count
as ties. This is a limitation of the preregistered raw-level EB estimator with
large family heterogeneity and correlated candidate histories; it is not evidence
that shrinkage in general cannot work.

## Decision, limits and next step

The fixed S1 fails all three gates: median gain is not>+.001, LCB95 is not>0 on
any primary seed, and S1 does not beat S0 on any primary seed. **r3175 stands.**
Do not promote sensitivity-only gains versus a weaker selection rule as an
incumbent improvement. This also qualifies the H304 interpretation: stricter
selection can avoid harmful promotions without creating a better checkpoint.

These are 19 historically exposed dev families. The candidates and r3175 were
already generated/selected with full dev91 feedback. Seed30401 EVAL also chooses
the non-S0 rule; it is not independent confirmation. Split seeds are not training
seeds, and the fixed-pick bootstrap does not correct selection bias. No external
unseen-family generalization, training improvement or breakthrough is established.

Next step: keep the incumbent. Any further shrinkage proposal should first
preregister a paired residual/noise model, then test a fresh shadow candidate
stream with genuinely separated selection and evaluation families. That experiment
requires its own authorization and budget; H307 launches none.

## Verification and custody

**14 focused tests passed**, including global-max tie behavior, R0 chronology,
SELECT isolation, exact H304 split stream, numerical-round smoothing, analytic
shrinkage, brute-force family bootstrap, paired zero deltas, MM Elo, shared-family
resampling and all strict decision gates. Ruff and compilation checks pass.
Source identity, all 203 pick locks, rule-choice ordering, decision recomputation,
and metrics size(42718bytes) were verified.
Full repository/hosted CI is **NOT_RUN under the owner CI freeze**, not passed.

The guarded analysis ran on xbabe2 cores 60–63, nice 19, one numeric thread;
aggregate peak RSS110,096,384bytes, observed peak hostCPU
21.6%, no guard violation. It finished at
2026-11-04T07:11:21.346700+00:00, before the fixed08:45Z compute deadline. Guard caps are RSS 4GiB,
hostCPU 85%, MemAvailable minimum 20GiB, scratch 2GB. H306 cores 120–127 are excluded.
All data, hash, archive and scientific work uses this guard; it signals only its
own child process group. No other host/job or live run was touched.

Create-only S3 prefix:
`s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H307/astra-20261104T070300Z/`.
The evidence tar contains source, all pick locks, primary selection scores, full sensitivity output,
execution/resource logs and audit receipts; original inputs remain recoverable
from their pinned versions. Exact-version GET, archive SHA and every member SHA
are verified before scratch removal. [s3_manifest.json](s3_manifest.json) pins
the manifest; [manifest.json](manifest.json) pins the evidence archive. Final
publication and scratch-removal receipts are retained in the create-only
`lifecycle.tar.gz` at the same prefix and in local custody
`/home/ubuntu/jope-rescue-20260921/h307-astra`.

## Reproduction

Use a fresh external evidence directory with these code files in its `source/`
subdirectory. Dependencies are numpy, boto3, psutil; ruff is for local checking.
The resource guard is deliberately pinned to this host, cores, scratch and
absolute campaign deadline. It refuses late execution: a later replay needs a
new resource authorization, not a silent deadline reset. Under that authorization,
run `fetch_inputs.py`, `run_h307.py regression`, then `run_h307.py run`, using
`--scratch /home/ubuntu/jope-data/h307-astra`; the fetch also takes `--receipt`
pointing to `source/input_receipt.json`. Run tests with
`python3 -m unittest discover -s source -p test_h307.py -v`.
Launch each phase through `resource_guard.py` under `taskset -c60-63 nice -n19`
and numeric-thread environment variables set to 1. Existing outputs are never
silently overwritten. Nothing imports peer scientific evaluators.
