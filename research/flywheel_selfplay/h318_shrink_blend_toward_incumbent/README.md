# H318 — shrinkage toward the incumbent at the hard-pick score interface

**BLEND_NOT_ADVANCED — shadow TRAIN-only study.**
Paired drop-top10 median **-0.012701995**, whole-family bootstrap 95% CI
**[-0.026546196257095183, -0.002728627626544254]**, one-sided 95% LCB **-0.024310320**.
BASE pooled gain **+0.048214800**; BLEND **+0.000000000**;
median harmed-family rates BASE **0.092683**, BLEND
**0.000000**. Criteria: `{'paired_drop10_positive': False, 'gain_retention_80percent': False, 'harmed_nonincrease': True}`.
Failed clauses: `['paired_drop10_positive', 'gain_retention_80percent']`. No dev91 score, r3175 TRAIN comparison,
serving change or neural optimizer update occurred.

## Hypothesis and executable scope

H312 gains concentrate in roughly 10/19 development families. H318 tests whether
shrinking the ENS_VOTE2 output toward r882 slot0 reduces concentration and harm,
using only nested family CV within 1219 DL-Train contexts/205 families.

The pinned selector emits a **candidate slot**, not continuous sample predictions:
H303 `ens_picks`, H287 `ensemble.vote_consensus_pick` at commit 213b30ad52. The
preregistered executable representation gives slot0 action score e0 and the ENS
pick action score ej. BLEND chooses argmax((1-alpha)e0+alpha*ej); ties select 0.
Grid alpha{0,.25,.5,.75,1} therefore has **two distinct policies**: alpha<=.5
incumbent, alpha>.5 ENS. It does not smoothly interpolate program predictions.
No scalar fitness/utility interpolation was substituted. BASE alpha 1 exactly
preserves ENS_VOTE2. This degeneracy was disclosed before fitting, not discovered
from favorable outcomes; the result is limited to this hard-pick score interface.

For each of 36 outerfold/root tasks, BASE selects unchanged H303 TAU/k/agreement
from 18 configs using its inner OOF family-mean objective and conservative ties.
The same fitted members and chosen BASE configuration supply BLEND. Alpha uses
that configuration's 4-fold inner OOF predictions, family context means, then
**median family gain - .5*harmed-family rate** (harm<-.01). Objective ties within
1e-4 choose smaller alpha. Alpha histogram: `{'0': 36, '0.25': 0, '0.5': 0, '0.75': 0, '1': 0}`. No joint
retuning of BASE configurations for BLEND. No outer family enters fitting or
alpha selection; each alpha is frozen before its outer score.

The ENS policy's inner alpha objective ranges from
**-0.094275235** to
**-0.006578947**; it is positive in
**0/36** cells. The incumbent has
objective zero. ENS inner family medians range from
+0.000000000 to
+0.012021240, while harmed rates range from
0.013158 to
0.195946.
These [inner diagnostics](alpha_diagnostics.json) describe the fixed selector's
behavior; they add no endpoint or post-hoc decision rule. In particular, a positive
mean gain does not imply a positive median-minus-harm objective.

## Preregistration and exact baseline replay

[PREREG.md](PREREG.md) commit **d6ff6e97f71db8bdd40f50a7800dc95f60e8ffe5** preceded all H318 fits/alpha
selection. Fit code **66c8b4b49b5183fbdbe594abe62b9a22e6e9db71** and score/statistics code
**9b2b77f832120f1f840d0de02aff8d7a3ce8a09d** preceded their execution. Parent modules are byte-pinned to
H310 **819a45aa5e30e662ab511cd9962b85b2b8c1e44a** and unchanged H303/H302/H287
code: [vendor pins](vendor_pins.json). Model and pick archives use committed
H303/H308 exact VersionIds/SHA256s: [input receipt](input_receipt.json).
No raw archive member is trusted before whole-object hash verification.
[TRAIN input provenance](train_input_provenance.json) records selected/H50/H58
metadata and the TRAIN-only r882 cost hash. Original H50/H58/H276 local input S3
VersionIds are not established here; actual allowed TRAIN ranges/packets are
hashed per task. Historical whole-cache hashes are metadata declarations, not
new full-file verification that would read forbidden DL-Val rows.

The independent regression process replayed all 36 frozen outer refits and matched
every fold/root and full-seed pick hash. Exact partition medians:
P0 **+.0643163334285782**, P1 **+.04821480030677507**, P2 **+.04360142642936269**.
[Regression receipt](regression_receipt.json) retains all 9 seed gains and hashes.
It is historical TRAIN regression; alpha fitting receives only a PASS/hash gate,
not historical outer outcomes. Fresh BASE picks were checked again after locked
scoring and match every archived pick. Independently, all 36 fresh BASE choice
tables and final model hashes match archived locks exactly.

P0 uses H302 outer family salt; P1/P2 H308 outer-v1/v2 salts; all4 folds,
roots 30001–30003 and original H300 member seed derivation. DOPE c51a56d0 frozen.
r3175 is **NOT_MEASURED** on TRAIN. **reference = r882 slot0 (DL-Train outer)**.

## Input boundary and operational proof

Workers physically load only their own outer-TRAIN pools, per-context r882 cost
members and exact allowed byte ranges from mixed H58 uncompressed NPZ caches.
Full cids/offset metadata establishes the TRAIN prefix; no DL-Val feature bytes
are decoded. The loader also excludes the held-out outer-TRAIN-fold rows, rather
than prebuilding all 1219 packets before fitting. Each packet and byte range is
hashed and bound to allowed context IDs. A denied outer pool/range test and a
poisoned interleaved-cache fixture demonstrate the boundary. Fit packet hashes
match the regression loader's corresponding TRAIN packets exactly.

[Lock receipt](lock_receipt.json) verifies 36 fsynced locks and 1080 final LightGBM
model files before outer access. **outer_test_rows_read_before_lock=0** and
**outer_test_packets_built_before_lock=0** for every task. A separate review
recomputes each alpha objective from inner-OOF picks and permitted inner utility
members: [lock review](lock_review.json). Scoring performs 36 ensemble evaluations
(three members each), once per partition/fold/root, derives both arms from the
same outputs, and writes raw utilities/picks. [Score receipt](score_receipt.json)
binds the panel. All further summaries use the saved panel; no repeated model
scoring. Historical regression is separately identified, not counted as H318's
locked outer evaluation.

## Results and uncertainty

Primary gain is mean of within-family utility differences, equal weight to each
of 205 families; pooled statistic is median across 9 partition/root cells. Literal
median family gains are also reported. Raw utility is -fitness on gate pass,
otherwise -2. W/T/L tolerance 1e-4, ties never wins; harm < -.01. Drop-top1/5/10
removes the largest family gains. Elo uses two-arm Bradley–Terry MM 400 iterations,
pseudocount .01; ties omitted. Nine dependent seed/partition cells are not nine
independent populations. The bootstrap conditions on these fitted, selected
policies; it does not rerun training or capture all model-selection uncertainty.

Whole-family bootstrap 10000 draws RNG 318180, SAME sampled family IDs across all
arms/cells. Primary paired drop10 subtracts arm-specific trimmed means, then
takes the nine-cell median, re-sorting within every bootstrap draw. It is not
the same as trimming paired differences: secondary median **-0.061781157**,
CI **[-0.08888650584650007, -0.039596287518699175]**, LCB **-0.083915643** is reported without decision authority.
Two-sided 95% percentile CIs; decision uses one-sided 95% lower .05 quantile.

### BASE

| Partition/root | Family mean gain | Family median gain | 95%CI | W/T/L | Harmed | Invalid | Drop1/5/10 | Elo | alpha by fold0..3 |
|---|---:|---:|---|---|---:|---:|---|---:|---|
| P0/30001 | +0.041289223 | +0.000000000 | [+0.021449404,+0.064843265] | 88/93/24 | 13 | 13 | +0.034482399/+0.022905348/+0.013173557 | 1725.66 | [1, 1, 1, 1] |
| P0/30002 | +0.064316333 | +0.001588084 | [+0.015787805,+0.133264962] | 108/67/30 | 18 | 14 | +0.037100103/+0.018969761/+0.008632837 | 1722.48 | [1, 1, 1, 1] |
| P0/30003 | +0.072257657 | +0.006984224 | [+0.023311559,+0.141246888] | 122/50/33 | 19 | 11 | +0.045081165/+0.027513773/+0.016620543 | 1727.10 | [1, 1, 1, 1] |
| P1/30001 | +0.048214800 | +0.002999623 | [+0.018359072,+0.093907141] | 112/57/36 | 23 | 17 | +0.030657068/+0.018310873/+0.010197530 | 1697.13 | [1, 1, 1, 1] |
| P1/30002 | +0.048657946 | +0.003624258 | [+0.021990392,+0.085088304] | 116/51/38 | 21 | 15 | +0.035309425/+0.023339580/+0.014063352 | 1693.84 | [1, 1, 1, 1] |
| P1/30003 | +0.038294238 | +0.003253142 | [+0.016447707,+0.061183116] | 112/64/29 | 17 | 16 | +0.033332512/+0.020954292/+0.012701995 | 1734.68 | [1, 1, 1, 1] |
| P2/30001 | +0.043601426 | +0.006326451 | [+0.017179150,+0.072809594] | 122/48/35 | 22 | 16 | +0.037279881/+0.021223015/+0.010678475 | 1716.88 | [1, 1, 1, 1] |
| P2/30002 | +0.031839978 | +0.003624258 | [+0.010170148,+0.054570365] | 112/65/28 | 16 | 17 | +0.026501550/+0.014887443/+0.008773683 | 1740.78 | [1, 1, 1, 1] |
| P2/30003 | +0.054720630 | +0.008144684 | [+0.033341130,+0.079765865] | 129/41/35 | 22 | 15 | +0.048595320/+0.033993371/+0.023605281 | 1726.57 | [1, 1, 1, 1] |

Pooled median gain **+0.048214800**, CI **[0.02515357681667722, 0.07618777930194197]**; median Elo
**1725.66**, median harmed **19.0**.

### BLEND

| Partition/root | Family mean gain | Family median gain | 95%CI | W/T/L | Harmed | Invalid | Drop1/5/10 | Elo | alpha by fold0..3 |
|---|---:|---:|---|---|---:|---:|---|---:|---|
| P0/30001 | +0.000000000 | +0.000000000 | [+0.000000000,+0.000000000] | 0/205/0 | 0 | 17 | +0.000000000/+0.000000000/+0.000000000 | 1500.00 | [0, 0, 0, 0] |
| P0/30002 | +0.000000000 | +0.000000000 | [+0.000000000,+0.000000000] | 0/205/0 | 0 | 17 | +0.000000000/+0.000000000/+0.000000000 | 1500.00 | [0, 0, 0, 0] |
| P0/30003 | +0.000000000 | +0.000000000 | [+0.000000000,+0.000000000] | 0/205/0 | 0 | 17 | +0.000000000/+0.000000000/+0.000000000 | 1500.00 | [0, 0, 0, 0] |
| P1/30001 | +0.000000000 | +0.000000000 | [+0.000000000,+0.000000000] | 0/205/0 | 0 | 17 | +0.000000000/+0.000000000/+0.000000000 | 1500.00 | [0, 0, 0, 0] |
| P1/30002 | +0.000000000 | +0.000000000 | [+0.000000000,+0.000000000] | 0/205/0 | 0 | 17 | +0.000000000/+0.000000000/+0.000000000 | 1500.00 | [0, 0, 0, 0] |
| P1/30003 | +0.000000000 | +0.000000000 | [+0.000000000,+0.000000000] | 0/205/0 | 0 | 17 | +0.000000000/+0.000000000/+0.000000000 | 1500.00 | [0, 0, 0, 0] |
| P2/30001 | +0.000000000 | +0.000000000 | [+0.000000000,+0.000000000] | 0/205/0 | 0 | 17 | +0.000000000/+0.000000000/+0.000000000 | 1500.00 | [0, 0, 0, 0] |
| P2/30002 | +0.000000000 | +0.000000000 | [+0.000000000,+0.000000000] | 0/205/0 | 0 | 17 | +0.000000000/+0.000000000/+0.000000000 | 1500.00 | [0, 0, 0, 0] |
| P2/30003 | +0.000000000 | +0.000000000 | [+0.000000000,+0.000000000] | 0/205/0 | 0 | 17 | +0.000000000/+0.000000000/+0.000000000 | 1500.00 | [0, 0, 0, 0] |

Pooled median gain **+0.000000000**, CI **[0.0, 0.0]**; median Elo
**1500.00**, median harmed **0.0**.
[metrics.json](metrics.json) (20,612bytes) preserves per-partition CIs,
all 36 alpha choices, invalid/baseline counts, hashes, primary and secondary tests.
Every evaluated BLEND output is an actual candidate's utility, never a convex
combination of observed outcomes.

## Verification, custody, limits and next step

20 focused tests PASS; source compilation/ruff and exact-head diff/scope/link/index
checks. Same-agent independent pre-score model/choice/input-closure review and
post-score utility/count/bootstrap/decision review PASS; [numerical receipt](numerical_review.json).
Bootstrap review uses independent partial-selection trimming and Python fsum
checks. This is same-agent review, not external approval. Full repository/hosted
CI **NOT_RUN** under the owner freeze; no PR/main/fastlane action.

xbabe2 CPU only, nice 19, one numeric thread, cores 13/15/38/57/59/125, disjoint active H315/
H316 at launch; no H317 CPU unit then. systemd MemoryMax 6 GiB and aggregate RSS 6 GiB,
scratch 2 GB, host CPU <=85%, MemAvailable >=20 GiB. Peak sampled RSS through analysis
3,739,275,264bytes, host CPU 49.1%; no guard violation.
Absolute compute 14:15Z / close 14:30Z Nov 4 never reset. [Resource receipt](resource_receipt.json)
retains a metadata-only correction: copied early launch/summary affinity fields
named prior-wave cores; actual taskset, runtime assertion and eight live process
readbacks prove the six declared cores / nice 19. Original receipts retained;
future receipt fields corrected, with no running-fit or scientific change.
Dependencies and source versions: [pins](dependency_pins.json).

36 create-only model archives were exact-version GET/SHA/member verified before
own inner model files were reclaimed; [model archive index](model_archive_index.json)
binds all14184 members. Final evidence archive,
source/metrics manifest and lifecycle receipts are under
`s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H318/astra-20261104T112505Z/`.
[Manifest](manifest.json) and [S3 receipt](s3_manifest.json) identify exact versions.
Every current scratch file receives a verified recovery mapping before deletion;
foreign files/processes/checkouts remain untouched. The executable layout is the
custody archive: source/, vendor/, receipts (restore these at the custody root),
and outputs (restore under the declared H318 data root). Published code is an
exact run artifact, not a standalone checkout CLI. Numerical replay needs only
the archived panel/locks and pinned package versions; full model refitting also
needs the separately retained original TRAIN inputs whose S3 versions are not
established here. No new replay or compute budget is implied. Local custody:
`/home/ubuntu/jope-rescue-20260921/h318-astra`.

Cross-critique: H314 CLEAN is a metadata rule result, not semantic independence.
H315 changes margin gating; H316 changes objective/normalization. H318 changes the
hard-pick output rule, so these are distinct mechanisms. Discrete-policy results
cannot substantiate continuous Polyak/James-Stein prediction smoothing. Reused
TRAIN families and adaptive study choice remain limitations despite outer nesting.
Dropping 10/205 TRAIN families removes 4.88%, versus 10/19 dev families 52.63%; the
positive TRAIN vs negative development drop10 contrast cannot isolate cohort or
incumbent effects. H315 also observed lower harm despite failing its advance gate;
its title should not be read as proof of unchanged harm. No matched-fraction
post-hoc decision was added to H318.

[H319 source cross-critique](crosscritique_h319.md): its negative nine-member
bagging result does not rule out all selector methods. A metadata-only replay
finds original-family copies crossing internal train/validation folds in all
324 members and 1,296 splits (7–27 original families per split), because bootstrap
copies were relabelled as separate families before fold assignment. This concerns
internal validation independence, not proven outer-family leakage. H319 also
prebuilt outer packets before access guards; zero guarded accesses is distinct
from zero physical reads. H318's physical input boundary and protocol are unchanged.
Do not advance this score-blend policy to dev91. Any continuous program-prediction blend needs a separately specified evaluator and authorization; it was not measured here.
