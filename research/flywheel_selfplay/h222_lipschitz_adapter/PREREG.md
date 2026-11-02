# H222 preregistration: Lipschitz smoothing of the adapter

Owner: luna_v1_res08. Authority: hw_t1300_res08_h222.md. Registration precedes
model construction, smoke and training. CPU xbabe2 only, frozen r882 COPY;
DOPE composite c51a56d0 verified before and after every fit. No live loop,
GPU, xbabe0, sealed, holdout or meta-test rows. Input identities are in manifest.json.

The hypothesis is that restricting sharp context-dependent adapter directions
improves family-grouped DL-Val utility. CTRL retains the H144/H140 r882 training
recipe: AdamW lr1e-4, WD1e-4, 40 updates, sixteen contexts and192 candidates,
top48 plus144 sampled candidates, smooth-L1 plus0.1KL at T0.2, clip-norm1,
eval mode/no dropout. The H144 sampler consumes eight draws and retains two per
round for20 rounds. All arms share its exact context/candidate stream. Training
uses mathematical attention in all arms to support paired second derivatives.

SN applies differentiable spectral normalization to every trainable adapter
linear transform, including each MHA input projection. The frozen anchor is
not a trainable adapter layer and remains unchanged. Initialize each pair of
power vectors with one declared iteration, then perform exactly one detached
power iteration per optimizer update. Forward/evaluation does not advance vectors.
This is an approximate layer spectral constraint, not a proof of a global network
Lipschitz bound. Record layer names and exact power-step counts. SN changes the
initial function; record training changes relative to that arm's initial function.

JAC adds lambda times the squared Jacobian norm of unrevealed adapter total-cost
outputs with respect to active10x32 context tokens. Use one deterministic
Rademacher vector per context/update; compute its VJP with create_graph=True,
sum squared input gradients and divide by the number of unrevealed outputs,
then average contexts. Candidate/anchor inputs otherwise stay fixed. Lambdas
are exactly0.001 and0.01; no surrogate, grid extension or Val tuning.

Seeds222101–222103 use distinct fresh five-fold TRAIN-family partitions seeded
222141–222143. For each seed and lambda, train five40-update fits excluding one
whole TRAIN-family fold, then score that fold. Maximize the resulting equal-family
OOF utility over205 TRAIN families; an exact tie chooses0.001. These30 inner fits
use only TRAIN1219/205 families. Then fit CTRL, SN and selected JAC once per seed
on all TRAIN, nine more40-update fits. Fresh folds choose JAC lambda only; final
fits use all TRAIN. There is no outer-test or new-family generalization claim.

All three lambda choices and all nine final states must be hashed and locked
before any candidate DL-Val scoring. Frozen-r882 parity is checked first:
91/91 exact context utilities and context mean1.1377941898164525. Native scoring
uses the first cost argmin; utility is negative native fitness, or-2 for a failed
gate. These historical real-S3 pools are not newly generated native-CV labels.

For every arm, report paired utility deltas vs CTRL and r882, per-seed signs,
family harm rate (any strictly negative paired family gain), and descriptive
family Bradley–Terry Elo anchored at r882=1500 with prior0.01. Exact Elo ties
receive half wins. Primary gains average within each of19 Val families, weight
families equally, then take the median of the three seed means. Also report
context-mean utilities separately; the1.1377942 reference is a context mean.

Use10,000 shared family-cluster bootstrap draws, seed222150, resampling19
families with replacement jointly across seeds and contrasts. Report two-sided
percentile95% intervals. CANDIDATE requires strictly positive gains and lower
CI bounds vs BOTH CTRL and r882 for the median and individually all3seeds,
using fresh registered folds. Exact ties fail. Otherwise NO_ADVANCE; any missing
seed/family/fit/panel is INCOMPLETE. No AUC or speed verdict, promotion or hidden
checkpoint selection. No extra multiplicity or harm threshold is added.

Prereg README/manifest/code/tests are reviewed and squash-merged before smoke.
One TRAIN smoke performs two actual updates for each arm (six total), proves
changed parameters/forward output, finite JAC second derivatives, exact SN step
counts and frozen DOPE. Only its exact-source PASS permits the single full run.
The full inventory is39 fits/1560 updates; combined cap1566 including smoke.
No scientific retry or deadline reset is allowed.

Independent supervision uses nice19, four-core affinity, single-thread numerical
libraries, CUDA hidden, aggregate worker+supervisor RSS strictly below4GiB,
load5<=60, disk free>=300GB decimal, MemAvailable>=16GiB and hostCPU<=85%.
It signals only its own verified process session. Compute stops absolutely at
17:00Z on2026-11-02, reserving30minutes for closeout before17:30Z. Missing work
at the deadline closes INCOMPLETE, not zero. Preserve per-update intent/completion
journals, gradients, changed parameter/forward hashes, partial adapter states,
OOF/Val costs and panels, read audit, resource samples and actual exit status.
Keep large artifacts outside Git and publish create-only versioned custody with
exact-version byte/member replay. Small README/metrics/index closeout PR follows.

The execution capsule reads the exact merged Git blobs, not mutable checkout
HEAD. This avoids another lane while the host already exceeds its lane limit;
canonical indexes, RC4/REL07 lanes and other owners' source/jobs remain unchanged.
