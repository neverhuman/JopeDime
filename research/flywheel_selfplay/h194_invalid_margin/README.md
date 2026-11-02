# H194: invalid-margin hinge, FINAL A3 NO_ADVANCE

**NO_ADVANCE.** The complete final attempt rejects IMH on pooled B and retains r882 at step0.
Raw 40-step continuations underperform r882 on A. IMH does not establish a gain over CTRL
at this fixed budget. No model or live selector is promoted; no fourth attempt is authorized.

The hypothesis is that a FIT-calibrated invalid-candidate margin improves r882 continuation
while retaining its active revealed-history kernel. [Preregistration](PREREG.md) and merged
[#1601](https://github.com/neverhuman/JopeDime/pull/1601), source
`9686429f7e89fb34da3e797afbb2855c93e02fb1`, fix CTRL, primary IMH (Holm1), and the count-matched
true-valid shuffled placebo. Each arm gets40updates in each of5folds and3seeds:45trajectories,
1800updates. Seeds194101–194103 have independent fold seeds194141–194143; bootstrap194150
uses10000paired family draws. The real S3 regression TRAIN corpus has1219contexts/205families;
FIT/B/A have123/41/41families. DL-Val91contexts/19families is descriptive. Binary is deferred.

Only47adapter tensors/29211parameters train. DOPE and all remaining parameters stay frozen.
IMH adds lambda0.1 times the mean-context hinge on min-invalid minus min-valid predicted cost.
Each cell freezes its margin from the FIT-only median gap of the two cheapest valid original-r882
full-pool costs before training. Eight shared FIT contexts and24candidates (4reveals+20sampled)
are used per update. Placebo subsets use true-valid candidates, a count matching true-invalid
candidates, and a disjoint remaining-valid comparison; zero-contribution contexts stay in the mean.

## Registered decision and diagnostics

Pooled-B IMH mean gain is **-0.031365693**,95%CI **[-0.056811203,-0.006978891]**,
Holm1 p **0.995100490**,0/3positive seeds: rejected, selected step0. The global B panel was
fsynced/hash-locked after all45B trajectories and before every candidate A/Val evaluation.
The sole IMH test uses the original H177 mean functional with raw p/Holm1. Its native four-slot
Holm4 diagnostics are separately replayed; this new m1 test does not inherit H178 m4 calibration.

The selected A policy is r882: gain0.000000,CI[0,0],Holm1 p1.0,0/3positive,Elo1500/lift0.
The following are unconditional40-step A diagnostics, with equal-family utilities and median
across seeds; Elo uses400-iteration MM,prior0.01,tie1e-4,r882 anchor1500,and paired-family CIs.

| Arm | Median A gain vs r882 | 95%CI | Positive seeds | Median Elo | Elo lift and95%CI |
|---|---:|---|---:|---:|---|
| CTRL | -0.032910 | [-0.059008, -0.005019] | 0/3 | 1484.088 | -15.912 [-25.605816, -4.857410] |
| IMH | -0.032958 | [-0.058798, -0.004918] | 0/3 | 1484.681 | -15.319 [-25.345614, -4.532585] |
| IMH_SHUF | -0.032909 | [-0.057807, -0.004637] | 0/3 | 1483.975 | -16.025 [-25.690238, -4.745010] |

Raw IMH per-seed A gains are[-0.032957565,-0.025668338,-0.033203064],median-0.032957565,
95%CI[-0.058798184,-0.004917884],Holm1 p0.991000900. IMH-minus-CTRL median is+0.000070902,
CI[-0.000127313,+0.000302372]; this does not establish a hinge benefit. These conditional
cross-fit results include historical TRAIN exposure and do not establish a representation ceiling.

True-invalid A picks per1219contexts are r882[17,17,17],CTRL[23,21,25],IMH[23,21,25],
placebo[23,21,25]. Stored labels were replayed from the true gate; valid picks at utility-2 were0.
FIT margins span0.018951535–0.024692357 (median0.021648407). IMH hinge is positive in97/600steps;
placebo228/600. Both have2524/4800active context slates, with0infeasible placebo slates.

DL-Val averages utilities over five fold models per seed, representing uniform model choice.
It performs no score ensemble or Val selection. Its descriptive IMH median equal-family gain
is-0.032177384,CI[-0.069419350,-0.003076624],0/3positive; median Elo1494.519/lift-5.481,
CI[-44.126,+44.075]. Equal-family utility is1.121437893 versus r8821.153615277.
This weighting differs from the native91 context-mean baseline1.1377941898164525.
True-invalid Val counts per455model/context decisions are IMH[2,2,2],CTRL/placebo[2,3,2],r882[0,0,0].

## Execution and verification

| Accounting | A1 | A2 | FINAL A3 |
|---|---:|---:|---:|
| Optimizer calls completed |1080|1246|1800|
| Backward calls started/completed |1080/1080|1247/1246|1800/1800|
| Complete40-update trajectories |27/45|31/45|45/45|
| Complete seed/fold cells |9/15|10/15|15/15|
| Candidate A/Val panels |0/0|0/0|45/45|
| Pooled-B selection |Not reached|Not reached|Locked; IMH rejected|
| Elapsed minutes |88.69|106.80|169.22|
| Termination |Shared disk93.8016%|Shared sustained load131.8555>108.8|Complete; no guard triggered|

A1 [closeout#1627](https://github.com/neverhuman/JopeDime/pull/1627) and A2
[closeout#1674](https://github.com/neverhuman/JopeDime/pull/1674) merged before their authorized
fresh repeats. Neither supplied a B selection or candidate A/Val outcome. A3 used the exact
original source/seeds/folds/budgets and original r882, with no A1/A2 trajectory or selector reuse.
The two infrastructure repeats therefore did not choose among candidate A/Val results.
There were4126completed optimizer calls and4127started/4126completed backwards across all attempts.
A2's pre-model binding fault (zero constructions/scores/updates) remains preserved. A2 also has
an independent zero-forward stored-state closure for31adapters. [Reattempt record](REATTEMPT.md)
records all authority and the external host-only admission change.

A3 ran CPU-only,nice19,math threads1,on actual xbabe1. All1336input hashes matched; four small
files totaling14386305bytes were staged. MemAvailable>=12GiB,free disk>=40GB,aggregate<=3threads,
RSS<6GB,output<500MB,CPU<=85%,load108.8/three independent5s windows,and4h wall were guarded.
Peak aggregate RSS was3.426GB; scientific output209131388bytes. No foreign jobs or protected
flywheel/H199/H200/H201 files were changed or signaled; no GPU or xbabe0/3 was used.

Native91 utility and float32 candidate-cost bytes matched before TRAIN outcome decoding.
DOPE composite`c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`
matched before/after, the whole model restored, and protected reads were0. Independent numerical
replay verified181score shards/44413choices,all1800updates/14400FITslates/15margin locks,
all native B diagnostics,192statistic fields and60Elo fields. Native replay verified all45states,
45first-step objectives and exact cost bytes for9trained adapters/72contexts. Post-run audits
made zero optimizer calls. A sorted-mask-versus-RNG-draw-order audit comparison was corrected;
the exact sampled sets, RNG stream, data and training remained unchanged.

[Compact metrics](metrics.json) retain all utility/CI/Elo scopes and true-invalid counts.
[Custody](custody.json) records actual-version S3GET and every payload hash/decode:
A3 has404payloads,184NPZ and47Torchfiles. Seven historical Git heads and36registered source/parent
hashes restored independently, with the c859 prerequisite declared. An initial package's active
resource-log mismatch was preserved and diagnosed; all its scientific/source members matched,
and the fresh final archive passes every member check. Final reviewed/merged Git custody is
recorded externally before own-lane cleanup. Publication follows the owner-local22controls/Ruff/
five-record gate and exact-head honest author COMMENT/squash flow; hosted CI and independent
GitHub approval are not claimed.

Retain r882. H194 ends with this final attempt; further optimizer runs are outside this authority.
