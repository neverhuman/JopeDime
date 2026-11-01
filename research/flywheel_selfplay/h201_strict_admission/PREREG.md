# H201: strict admission and the r3175 selection audit

Owner: gpu_tokbench_codex. Authority: `~/nudges/hw_t1356_gputokbenchcodex.md`.
Preregistered before native candidate evaluation or outcome analysis. One guarded
attempt, zero optimizer construction, zero backward calls, zero parameter updates.
No live flywheel writes or adoption. This is a retrospective development-cohort audit.

## Inputs and regression gate

Three read-only checkpoint copies: r882, r3100 and r3175. Pin every copy, source,
asset and numeric input in `manifest.json`. The r3175 file must have SHA-256
`1d6ea900ebed8d2a05c3b3f37576adc15aadd2b53c585bfd5cb71a103e9ff97b`.
Only approved real-S3 DL-Val pools are opened; no TRAIN outcome pools, sealed,
holdout or meta-test inputs. The existing context feature cache is read-only.

First evaluate r882 through the identical Engine/runtime used by merged H197,
also cross-checking the H199/H139 first-argmin endpoint. Before any candidate
forward or history analysis, all 91 utilities and all native candidate cost bytes
must equal the pinned r882 reference; context mean must be exactly
1.1377941898164525. DOPE c51a56d0 and every non-adapter tensor must match r882
for each candidate. All parameters are frozen during all inference. Restore r882
at closure and verify its whole-state hash. Candidate r882/r3100/r3175 archived
per-context utilities must exactly match the fresh native vectors; mismatch stops
the attempt and yields INCOMPLETE, never fabricated equivalence or a retry.

**Order correction:** the pinned H197/H199 91-context cohort has **19**, not 20,
families. Preserve those identities, bootstrap 19 clusters, and use disjoint 9/10
family split-halves. No fabricated twentieth cluster or outcome-dependent regrouping.

The live history is frozen at round 3192: 2,418 complete admission lines, rounds
775–3192. Initially its final round journal was in flight; preserve that original
snapshot and finish only that same frontier before pinning. All 2,418 approved
DL-Val vectors are now captured with original journal hashes. No future rounds
enter this study. The initial incumbent vector is `initial_dev_utility` in round
775 (r774). Preserve both the raw scalar history and the compact dev-only projection.
Validate every archived candidate mean, previous-incumbent mean and evaluation
identity against the raw history. Do not read the earlier r774 checkpoint or run.

## Primary hypotheses and inference

H1: r3175's equal-family utility gain over r882 has lower 95% bound <=0, and the
observed historical maximum is compatible with the prespecified no-gain null
(Monte Carlo max-test p>=.05). H2: sequential strict-margin admission (SMA),
replayed on the captured history, rejects every actual tie (at least four exist)
and at least three actual improvements (at least four exist). These are hypotheses,
not conclusions forced by the implementation.

For r3175-r882, r3175-r3100 and r3100-r882, compute per-family mean paired utility
differences, then a paired family bootstrap (10,000 draws for each seed 201101,
201102, 201103). Report every seed's CI, the observed gain and median seedwise CI
endpoints. Bootstrap seeds are Monte Carlo repetitions, not independent models or
cohorts. Point estimates are identical across these seeds. Report context means
separately from the primary equal-family means. Report context and family harm
fractions (strictly lower utility) and exact -2 counts.

Compute three-policy Bradley-Terry Elo from context-pair wins/ties (tolerance1e-4)
before equal-family averaging, reference r882=1500, MM400 iterations, symmetric
prior .01. Use10,000 paired-family draws per bootstrap seed, derived by
SHA256(201150:elo:seed), and report all seedwise CIs plus their median endpoints.

## Winner's-curse null and family split

Primary selection history ends at the target round3175 (N=2,401); later captured
rows are used only for the full-history descriptive admission replay. Form the
candidate-by-family matrix of gain over r882. Center **each candidate** by its
own mean family gain to impose zero empirical effect. Draw shared family weights
with seeds201101–201103; this preserves cross-candidate dependence. For each draw,
take max(0, gain of every candidate). Report observed uncentered max, expected null
max, null-max95% interval and `(1+#null>=observed)/(10001)`, median over seeds.
This is a specified empirical null conditional on the adaptive history; it is not
an identified causal distribution of truly independent candidates.

For each fold seed201141–201143, permute sorted family identities and select the
best equal-family candidate on the first9 families; score its paired gain over
r882 on the remaining10. Include r882 as the first option and use stable chronological
tie-breaking thereafter. Report selected round, A/B family identities, B gain,
10k family CI using SHA256(201150:split-B:fold_seed), and median B gain/CI endpoints.
Allthree held-B point gains must be strictly positive for CANDIDATE_REAL. These
are retrospective checks: both halves already influenced the live history, so
they do not constitute prospective or sealed-test validation.

## SMA replay

Use one fixed paired-family weight matrix (seed201150,10k draws). Start from the
actual prehistory r774 vector and replay every captured candidate in order.
Admit iff candidate-minus-current-SMA-incumbent equal-family mean>0 and its
two-sided95% interval lower endpoint>0. Exact ties reject. Rejected proposals
leave the SMA incumbent unchanged. This is a genuine counterfactual chain.
Separately screen candidates against the historically observed incumbent and
label that diagnostic distinctly; it must not replace the sequential result.
Tabulate accept/reject by original action, keep every round decision in raw custody,
and evaluate H2 against the sequential decisions. SMA is descriptive and no live
policy is changed. If H2 holds, write a proposal for a separately reviewed owner
switch; repeated-testing calibration remains a prospective design requirement.

## Verdict, resources and lifecycle

WINNERS_CURSE iff H1 holds. CANDIDATE_REAL only if r3175-r882 lower95% bound>0,
allthree selection-half B gains>0 and r3175 Elo lower95% bound>1500. Otherwise
INCONCLUSIVE. Missing coverage or failed native/provenance regression cannot pass.
WINNERS_CURSE means evidence compatible with this selection-noise null, not proof
that the true effect is zero. Never recommend adopting r3175 from this reused cohort.

Host gate accepts only xbabe1 or xbabe2 before scientific input reads; tests must
reject xbabe0/xbabe3 without opening any input. Prefer xbabe1. Nice19, affinity<=2
cores, math threads1, MemAvailable>=12GiB, aggregate RSS<4GB, hostCPU<=85%, wall<=90min,
attempt output<200MB, free>=40GB. On xbabe2 also stop at disk used>=93.6% (before
the shared93.8% stop). Stop only owned process groups; no GPU or foreign signals.
One exclusive attempt claim; no resume or retry. Guard stop preserves INCOMPLETE.

Data-free controls, Ruff/format and five jope-records gates precede exact-head
same-agent review and squash merge, with no remoteCI wait under the owner order.
Run only merged source. Closeout publishes README, metrics<100KB, H201 index row,
data/source custody and independent numerical verification; self-squash merge,
verify actual merged source, restore recovery evidence, remove only owned lanes,
then append the final claim/verdict. No checkpoints or numeric npz enter Git.
