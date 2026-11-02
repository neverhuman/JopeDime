# H202: current-top-five inverted-pair fine-tuning

Owner: gpu_tokbench_codex. Authority: `~/nudges/hw_t1505_h202_gputokbenchcodex.md`.
One guarded CPU attempt, no pilot outcome analysis, budget adaptation or resume.
No live policy/model change or adoption. A CANDIDATE needs a separate confirmation.

## Hypothesis and prior evidence

H1: pairwise logistic training on the current adapter's top-five ranking mistakes,
with family-balanced context sampling, improves full-pool native-protocol utility
on held-out TRAIN families. HNEG40 is compared with r882, a matched ordinary-loss
control and a within-context shuffled-utility placebo.

The assignment's broad novelty claim needs correction: H58 already used hard-negative
pool admission/harvesting and listwise/margin objectives. H202 tests this specific
online top-five inversion loss from the fixed r882 adapter. H201's original
qualification is INCONCLUSIVE after native/archive parity failed; its descriptive
results are compatible with selection noise, rather than establishing its cause.

## Fixed inputs and native endpoint

Use an immutable offline copy of r882, SHA256
`0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64`.
Hash-pin native source, model assets, context cache, roster, real-S3 catalog,
reference utilities and all own source in `manifest.json`. Pool hashes are bound
by the selected roster. Real catalog origin must be `real` for every dataset.
No sealed, holdout or meta-test payload/roster is read.

Before TRAIN outcome pool loading, backward or optimizer construction, evaluate
r882 on the fixed 91 DL-Val contexts. Every utility and the context mean
1.1377941898164525 must be bit-exact against the pinned reference. The actual
DL-Val roster has 19 families. Use the unchanged H197 Engine/runtime, called one
context at a time, matching the live utility routine's batch shape. Candidate
choice is first argmin over the complete pool; true utility is negative original
fitness if valid, otherwise -2.0. No masking of revealed candidates is introduced.

The native/cached full cost vectors must agree on this regression to validate
the frozen-anchor cache. The old H170 batch-eight raw costs are an additional
reported diagnostic, not another requested utility gate. All held-family and
report-only DL-Val predictions use the same batch-one cached H197 forward and
fixed original first-four reveals. Frozen anchors are computed once per approved
context; this does not change the formula, labels, ranking or gradients.

## Folds, initialization and context sampling

Seeds202101–202103, paired with independent family partitions202141–202143. Permute
sorted 205 TRAIN family IDs with each partition seed and split into five groups
of41. For outer k, A=group k, B=group(k+1)%5 and FIT=the other three groups:
123 FIT /41 B /41 A. B lies within the four non-A TRAIN groups and is excluded
from gradients. Each family supplies B once and A once per seed. Store the fresh
fold manifest before execution; all three partitions must differ.

All45 trajectories (3arms x3seeds x5outerfolds) start from identical r882 adapter
weights, with a fresh optimizer per trajectory. For each step, draw four FIT
families uniformly with replacement, then one context uniformly within each
drawn family. Repeats are allowed; each draw gives each family probability1/123.
Use SHA256-derived stream(seed,outer,step,H202-family4-v1). The complete forty-step
context plan is shared across arms within each seed/fold; shuffled labels use
separate deterministic streams and cannot perturb that plan.

## Objectives and optimization

All arms make a full-pool forward for each of four contexts, preserve the same
original four reveals, and accumulate loss/4 gradients before one optimizer call.
Only the 47 tensors /29,211 parameters outside `adapter_10tok.anchor` within
`adapter_10tok` may receive gradients or enter the optimizer. DOPE and all other
tensors stay frozen/eval; assert the gradient mask after every micro-backward.

* **HNEG40:** stable ascending cost order (original index breaks exact cost ties),
  take K=5. For every positions i<j where utility(j)>utility(i), include the inverted
  pair (better=j,worse=i). Minimize mean softplus(cost_better-cost_worse), temperature1,
  equally weighting included pairs. Utility comparison uses original float64 pool
  labels. True-utility ties are excluded. Predicted ties follow the native stable
  index ordering. No-inversion contexts return differentiable zero, not a substituted
  pair or a changed sampling rule. Restriction/top-k selection is detached from gradients.
* **CTRL40:** identical context sampling, full-pool forward, reveals and accumulation,
  using the pinned ordinary flywheel SmoothL1 +0.1 KL loss over that full pool.
  H202 changes the number/sampling of contexts and full-pool objective domain as
  explicitly assigned; it reuses H197's loss/optimizer recipe.
* **HNEG_SHUF:** HNEG40, but permute full-pool utilities within that context with
  stream(seed,outer,cid,H202-within-context-label-shuffle-v1). Keep this permutation
  fixed for repeated visits in that seed/fold. Candidate ranking still uses the
  placebo model's own current predictions. No family/context labels are exchanged.

Forty updates per arm/cell, lr1e-4, AdamW weight decay1e-4, betas(.9,.999), default
epsilon1e-8 and global accumulated-gradient clip1.0. This is the actual H197 CTRL
optimizer, correcting the assignment's shorthand “Adam.” No schedule, dropout,
extra loss term or calibration. Total1,800 optimizer calls /7,200 context-backwards.
Record zero-inversion contexts, nonzero-gradient steps, pair counts and checkpoint
changes so regularization-only movement is not misreported as ranking learning.

## B-only common step and durable lock

Save adapter snapshots and B full-pool scores at steps10,20,30,40; include fixed
r882 as step0. Fill each arm/step's pooled B panel over all205 families for each
seed. Once all45 forty-step trajectories and all B snapshots finish, choose **one
common step per arm** from0/10/20/30/40 maximizing seed-averaged equal-family B
gain over r882. Earliest exact ties win, including zero. There is no A or DL-Val
selection and no additional unrequested B admission threshold.

Write all B/state hashes and selected steps into an exclusive lock, fsync its
contents and parent directory, then verify its SHA before every candidate A/Val
cell. No candidate A or Val scoring precedes this lock. Fixed r882 comparator
predictions may be cached earlier. Training still completes40 updates even if
the selected step is0. Step0 confirmation copies the identical fixed reference.

## Primary statistics and verdict

Pooled A has1,219 contexts /205 families, each family A-held once per seed.
Average context utilities within family; use the median of three seed-specific
equal-family gains. Jointly resample the same family indices across all seeds
(boot seed202150,10,000 draws), reduce **median across seed axis0** for every draw,
and report the95% interval, each seed, median and one-sided Monte Carlo p.
Holm m=1 leaves primary HNEG40-vs-r882 p unchanged; other contrasts are reported.

CANDIDATE iff HNEG40 median gain>.001+1e-12, lower95% bound>0, all3 seed gains>0,
Holm p<.05, and harm<=10%. Harm is the fraction of205 family deltas below zero
after averaging the three seeds; also report per-seed family/context harm.
HNEG40 must strictly beat both selected CTRL40 and HNEG_SHUF: paired median
difference>1e-12, as the point-comparison mechanism in H197. Report their full CIs
without adding unrequested significance gates. The placebo must not clear the
same substantive r882 gain/CI/three-seed/harm gate. Otherwise NO_ADVANCE. Ties
and zero confidence endpoints are rejected; no live adoption follows either verdict.

Report equal-family utility, all paired contrasts and exact -2 picks per seed.
Compute context-pair wins/ties (1e-4 band), then family means and Bradley-Terry MM
(400 iterations, symmetric prior.01, r882=1500), separately per seed. Report median
Elo and paired-family median-bootstrap95% interval, using the same family draws.

DL-Val is descriptive only. Score the selected model for each of five outer folds
per seed:91 unique contexts /455 model-context pairs. Utility is the expected
utility of a uniformly drawn fold model; Elo uses paired same-fold context wins,
not score averaging into an unqualified ensemble. Bootstrap19 families. Exact -2
counts are out of455 pairs per seed. DL-Val never affects selection or verdict.

## Artifacts, guards and limitations

Retain every full training forward, B/A/Val score, trace, forty-step count, snapshot,
fold/source binding and lock. Lossless XOR of float32 cost bits against immutable
r882 costs reduces storage; roundtrip tests and independent decoding verify exact
bytes. This is an artifact encoding, not numerical quantization. No checkpoint or
npz enters Git. Full payloads receive versioned S3 custody/readback verification.

Actual attempt on xbabe1 only: nice19, math1, aggregate<=3 threads, RSS<5GB,
MemAvailable>=12GiB, free>=40GB, output<300MB, load1<=108.8, hostCPU<=85%, wall<=4h.
The host whitelist rejects xbabe0/xbabe3 before scientific reads; CPU-only xbabe2
is used for source/data-free verification. One exclusive attempt claim, no retry or
resume. Stop only owned worker groups; keep partial counts/unknown in-flight calls
as INCOMPLETE on a stop. No GPU, live flywheel writes or foreign process actions.

A and B reuse the same205 families across cyclic roles, rather than supplying new
independent-family confirmation. The selected B step may be0; an all-zero selected
readout does not prove a representation ceiling. Historical DL-Val reuse and CPU
execution do not qualify the live serving backend. Prereg merge follows18 data-free
controls/Ruff/five records/exact-head same-agent review, without remote CI wait.
Closeout publishes the truthful verdict, compact metrics<100KB/index, independent
verification and recovery custody, merges and removes only the owned lanes.
