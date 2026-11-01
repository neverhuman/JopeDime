# H194 invalid-candidate margin (2026-11-01)

Owner: gpu_fused_redteam_codex. Authority: `~/nudges/hwave_t1003_h194_redteamcodex.md`.
Hypothesis: discouraging low predicted costs on invalid candidates during r882 continuation
reduces invalid picks and improves equal-family regression utility with the active kernel intact.
No candidate outcomes or optimizer calls preceded this registration.

## Frozen experiment

R882 is the zero-update reference. CTRL, IMH (sole primary, Holm m=1), and IMH_SHUF
(descriptive placebo) each start from the same private copy of r882, retain its complete
inference and revealed-history kernel, and take exactly40 updates per seed/fold. AdamW
lr1e-4, weight_decay1e-4, betas(.9,.999), clip L2=1; effective dropout zero/eval mode.
Only47 adapter tensors /29,211 parameters train. DOPE composite c51a56d0 is checked before,
after and through the inherited per-update freeze checks. No live checkpoint write/adoption.
The source-hash-pinned H146/H165/H182 runtime keeps full native inference, including kernel.

All three arms share the fixed H156 slate plan:8 distinct uniform FIT contexts per update,
24 candidates/context (four lowest-fitness revealed candidates,20 sampled from remainder).
Common plan seed=H177.stream_seed(seed,outer,"fit-123","h156-batches-v1"). Live loss is
unchanged SmoothL1 (clipped valid fitness, invalid target2) plus .1 KL at temperature.2.
IMH adds .1 * mean_context relu(m-(min_invalid_cost-min_valid_cost)) on that training slate.
For m, run original r882 on complete FIT candidate pools, with native first-four storage-order
reveals, before any updates in that cell. Each context with>=2 true-valid candidates supplies
the absolute float64 difference between its two cheapest float32 valid predictions. m is the
median across these eligible FIT contexts; freeze/fsync its IDs, gap values and source cost
vectors before training. No eligible context aborts; m=0 is permitted without a floor.
No-invalid or no-valid slate contributes0, still included in the batch mean denominator.

IMH_SHUF draws WITHOUT replacement exactly n_invalid candidates from true-valid slate
candidates as pseudo-invalid; comparator is the disjoint remaining true-valid set. Original
invalid candidates participate in live loss, never in these two pseudo groups. A count-match
with a nonempty comparator is feasible only when0<n_invalid<n_valid. Otherwise placebo
hinge is0 and its infeasible count is logged; the primary IMH is still applied. This diagnostic
has reduced coverage when invalid candidates outnumber valid candidates. Separate per-step
NumPy stream=H177.stream_seed(seed,outer,step,"pseudo-invalid-v1"), independent of Torch
and shared FIT plans; save every pseudo index and all candidate indices. No labels are changed.

## Roles and selection

Real S3 regression DL-Train1219 contexts/205 families, DL-Val91/19 descriptive;276 distinct
real datasets, no family overlap. Pinned catalog metadata is actually retrieved by VersionId;
numeric input lineage inherits H165 independent S3 restoration and every pool is rehashed.
First replay native91 utility and ALL float32 cost bytes exactly (mean1.1377941898164525)
before any TRAIN outcome pool is opened. Metadata and unlabeled frozen feature cache are
permitted earlier. Sealed/meta-test/outer protected files are denied by inherited audit hook.

Seeds194101,194102,194103 use INDEPENDENT family-fold seeds194141,194142,194143,
respectively (full manifest committed). For each seed: five cyclic folds, A=j, B=j+1 modulo5,
FIT=remaining123 families. No A or B labels enter that cell's fitting or m. All families remain;
no timing-based fold/seed/update reductions. 45 trajectories,1800 actual optimizer calls.

Only one trained candidate checkpoint is eligible: IMH step40. Complete ALL45 B evaluations,
then pool B1219/205 per seed and choose original step0 or IMH40. This is the owner-prescribed
m1 variant of the unchanged H177 mean-functional rule: equal-family mean gain>.001+1e-12,
paired-family95% mean CI lower>0, and raw one-sided bootstrap p<.05 (Holm1). H177's native
four-slot API receives CTRL/IMH/IMH_SHUF/R882 diagnostic slots; its m4 decisions are retained
but DO NOT determine the H194 m1 selection. CTRL and placebo can never be advanced.
Fsync pooled_b_selection_lock.json and bind its SHA before ANY candidate A or Val scoring.
Reference FIT/B costs are allowed before the lock; no candidate A forwards occur then.
After lock score each fixed40 arm unconditionally on its A and descriptive Val, even if IMH
was rejected. Report both raw fixed40 panels and the B-selected IMH policy (R882 if rejected).
The single global B selection is not an independent outer test: A/B families recur in other
cells' FIT roles and r882 has historical TRAIN exposure. This is a conditional continuation
screen, with new family partitions, not a fresh-data generalization or serving qualification.

## Reporting and strict ruling

Boot194150,10000 shared paired-family draws. Within family average contexts; per replicate
average families, then median across three seeds. CI resamples the SAME family indices across
all seeds/arms and takes median over seed axis0; primary p=(1+count(bootstrap median<=0))/10001,
Holm m1. Also report seed-average H177-functional companions, per-seed gains and positive count.
ADVANCE only if B admitted AND selected A median gain>.001+1e-12 AND A LCB95>0 AND primary
Holm p<.05. Strict ties fail. No additional all-three-positive gate; report positive count.
Elo is H148/H177 equal-family context wins, tie1e-4,400 MM iterations, prior.01, R882=1500;
report Elo and lift vs1500 with paired-family intervals. No AUC. Descriptive Val averages
utilities over the five fold models per seed (uniform random model expectation), never cost
ensembles; Val never controls selection. H178 m4 calibration is inherited context only and
does not empirically calibrate this new owner-prescribed m1 branch. Any ADVANCE is nomination
only, requires separate qualification; serving_promotion=false unconditionally.

## Execution, custody and delivery

Single attempt, CPU xbabe2, nice19, math libs1 thread, supervisor+worker<=4 threads,
aggregate RSS<6GB,4h hard cap from monitor launch; output<500MB. Prelaunch three independent
5s checks: user-visible disk<93.8%, free>=40GB, MemAvailable>=16GiB, hostCPU<=85%,
1minload<=.85*online cores. Runtime load stop requires three independent5s over-limit
windows; CPU/memory/disk/thread/RSS caps are immediate. Only own child process group can
receive termination. No foreign pause/kill, no xbabe0/3, no GPU initialization.
If xbabe2 prelaunch fails, owner explicitly permits xbabe1; do not move after updates or reset
the single-attempt budget. This implementation is xbabe2-only; any needed host rebind must
be a recorded pre-science operational change with host-specific disk guard, not silent fallback.
Persist backward/optimizer started/completed counters every call and terminal failure receipt;
no automatic retry/resume. Save states, raw costs/choices/utilities, role/margin/plan/trace and
all panels outside Git. Versioned S3 custody must be actual GET/hash/member-decode verified,
including source and restored Git history, before owned-lane cleanup. Keep payload<500MB;
large artifacts never in Git. Small prereg PR with local pytest/Ruff/five jope-records gates,
head-bound honest self-review, squash/admin merge under owner authority (no remoteCI wait),
then merged-source verified launch; closeout README/compact metrics/index/custody PR and cleanup.
