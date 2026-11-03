# H245: fresh nested lambda + zgap gate (zero optimizer updates)

Owner luna_v1_res08; authority `~/nudges/hw_t0505_res08_h245.md`; claim
`~/jope_plan/ns/receipts/H245_claim.txt`. Preregistration must merge before 06:15Z
on 2026-11-03 and before H245 fitting or outcome assessment. Compute stops at
10:00Z and closeout merges by 10:45Z. A context resume never resets these times.

Hypothesis: FIT-only joint choice of lambda in {0.125, 0.25, 0.5} and an
outcome-free zgap threshold lifts the confirmed H217 gated recipe beyond the
DL-Val +0.002 margin. H218 was incomplete; it supplies the fixed recipe, not a
positive outcome. H223/H224 closed NO_ADVANCE. This is one fresh wave and one
45-fit run, with no extra smoke fits or scientific retry.

The seven blend/gate/selection functions are AST-identical to merged H218
(comments/docstrings excluded); shared family/Elo/legacy plateau primitives
come from exact merged H195/H217 bytes. Immutable parent modules and assets
are SHA256-pinned; shared checkout HEAD and files are not runtime source.

Use 1,219 real-S3 DL-Train contexts / 205 families and DL-Val91 / 19 disjoint
families, the pinned H50 roster/cache and a private read-only r882 COPY. Exclude
has_meta_test and all sealed/holdout/live data. Every dataset joins the catalog
as origin=real. Native91 must first match all 91 utilities and mean
1.1377941898164525 exactly, before TRAIN fitting. DOPE c51a56d0 stays frozen;
all neural parameters require no gradients and there are zero optimizer updates.

Seeds 245101–245103; one family partition per seed, fold seeds 245141–245143;
sample245140, utility bootstrap245150, Elo245151, legacy plateau245152,
zgap permutation245160. Each partition is the H124 sorted-family shuffle and
five-way split. A=k, B=(k+1)%5, FIT=other three. Inner three-fold family OOF
predictions calibrate cost and select lambda/tau on FIT only. HistGBM model,
192 sampled programs per FIT context, loss/feature definition/calibration,
max_iter300, learning_rate.08, max_leaf_nodes63, l2_regularization1 remain
unchanged. Fifteen unique exclusion models per seed (5 outer +10 inner) give
45 fits total, shared across all lambda values and placebo policies.

For each lambda, tau candidates are closed (-inf), switching zgap q10/q25/q50/
q75/q90 and open (+inf). Maximize FIT equal-family native gain subject to FIT
harm <=5% (family mean delta <−0.01). Ties choose smaller tau, then smaller
lambda; absence of positive safe gain keeps closed r882. zgap uses cost only,
first argmins, and r882 standard deviation. All 15 cell lambda/tau choices
are fsync-locked in stage1_lock.json before any trained A or DL-Val assessment.
Native91 baseline and FIT outcomes are the explicit pre-lock exceptions;
rotating cross-fit raw pools are loaded for fitting but each cell's A/B labels
are excluded from its fit/calibration/selection. Every policy is assessed once.

Arms: R882; GATED_NEST sole primary (Holm m=1); GATED_025 fixed-lambda
replication anchor with paired contrast; GATED_NEST_SHUF report-only placebo.
To implement the owner's same-gate-rate requirement, SHUF copies primary
lambda/tau and permutes zgap across the same context set: FIT, each A fold and
each DL-Val cell have exactly matched open counts (including ties). It never
retunes lambda/tau from placebo outcomes. This report-only control refinement
does not change GATED_NEST or GATED_025. Ungated .125/.25/.5 and calibrated
GBM costs are report-only, with no added fits.

Native utility is full-pool first-argmin negative fitness when valid and −2
otherwise. Higher is better. Report pooled-A equal-family gains, shared-family
10,000-draw bootstrap CI95 and median across seeds. The sole primary DL-Val
estimand retains H217/H218 context weighting: average five outer policies per
context within each seed, then median the three context-mean gains. Shared
19-family bootstrap draws pool all contexts in each sampled family, then
median across seeds. Report one-sided p against zero; Holm m=1 equals this p.
ADVANCE requires DL-Val median gain >.002 (+1e-12 numerical tie tolerance),
LCB95>0, Holm p<.05, at least2/3 positive seeds, and pooled-A harm<=10% using
the retained H217/H218 definition (205 families, three-seed mean family delta
<−.01). Also disclose DL-Val harm. All failed/tied gates yield NO_ADVANCE;
an unfinished run yields CLOSED_INCOMPLETE with missing outcomes null.

Report native Elo anchored at r882=1500 (400-step MM, prior.01, 1,000 paired
family draws), GATED_NEST vs GATED_025, lambda/tau counts, gate rates, harm and
context utility vs r3175's historical plateau 1.1389572. Elo against r3175 is
null unless paired outcomes already exist in the frozen input closure; the
scalar mean alone cannot supply it. No extra plateau model evaluation, AUC,
speed verdict or serving promotion is authorized.

All H203–H233 registered fold manifests are inventoried. Context identifiers
and unprefixed family hashes are normalized to pinned family identities.
Twenty-six comparable waves have maximum mean best-match Jaccard
0.19621375708679545 <=.20. The five remaining waves have no comparable
five-fold manifest; this is recorded explicitly, not passing coverage.

Runtime: xbabe2 only, CPU40–43, nice19, OMP/HistGBM4 and other math pools1;
torch intra/inter-op1, no GPU. RAM admission>=30GiB/load5<=100, free>=300GB;
load>108 cooperatively pauses / <=100 resumes. RAM<20 pauses, >=25 sustained
30s resumes. Hard stop only RAM<12 sustained30s or <6 instantaneous, RSS>=4GiB,
disk<300GB or fixed10:00Z deadline. Safe points are between HistGBM iterations,
feature/prediction contexts and stages; loss-stopping callback inputs/return
and RNG state remain unchanged. No SIGSTOP/SIGCONT or other-job controls.
Completed fits are checkpointed for custody; no restart is authorized here.

Small metadata/source PRs use targeted data-free pytest, Ruff/format, JSON/
schema/diff and actual jope-records static coverage validation, exact-head
same-agent COMMENT review and owner-authorized guarded admin squash. Hosted
full CI is NOT_RUN/NOT_A_GATE. No mutation of PR2071. Bulk output and parent
source capsule go create-only to learning-use2 experiments/rd_breakthrough_
20261030/H245/<run_id>/ with VersionId and exact-version byte/member replay.
Closeout updates README, metrics<100KB, both wave-index rows and appends
FINDINGS.md; claim is released with durable receipts. Raw roots are retained.
