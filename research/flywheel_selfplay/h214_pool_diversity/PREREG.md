# H214 — candidate-pool diversity at fixed evaluated width

Owner: luna_v1_res08. Authority: `~/nudges/hw_t0200_res08.md`, ENG H-wave loop.
Claim: `~/jope_plan/ns/receipts/H214_claim.txt`. Registration precedes any H214
utility readout. This is offline research with zero optimizer updates.

## Hypothesis and frozen state

At equal evaluated width16, a diverse native proposal pool improves the frozen
r882 selector's DL-Val91 utility over native proposal order. Use an owned copy of
round-882.pt, SHA256
`0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64`.
DOPE composite
`c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`
must match before and after; checkpoint bytes must also match before and after.
All parameters have requires_grad=false, model training=false, inference uses
no_grad, CUDA is disabled and must remain uninitialized. No live-loop directory
may be read or written. There is no optimization, adapter continuation or promotion.

## Arms and exact width

Each arm scores EXACTLY16 unique programs, including the same designated parent
in slot0 and15 proposals. The parent is the first revealed program in the
context's pinned H50 pool, chosen by stored order, never by H214 utility.
The other three revealed anchors condition the unchanged r882 predict_costs
call but are not additional candidates in the arm's argmin.

- NATIVE16: call the pinned H212 proposer with n_proposals=16 and keep the first15
  unique nonparent proposals in native order, after stable program_sha256 dedup.
- FPS16: call that proposer with n_proposals=64. Start with the parent, then
  greedily add15 candidates maximizing minimum squared distance to selected
  programs in native proj16 features. L2-normalize each16-coordinate vector;
  leave a zero vector zero. Exact distance ties take the earliest native
  generator position. Do not use costs, fitness, labels, family utility or an
  outcome-dependent feature transform during pool selection.
- TEMP16: request16 native proposals at prespecified temperature T=2.0
  (native reference T=1.0), then apply NATIVE16's parent-first ordering/dedup.
  The hook must change native proposal exploration before generation. Softmax
  over r882 scores, resampling an already generated pool, seed substitution and
  changing neural weights are prohibited substitutes.

H212 N16 actually presents4 anchors plus16 proposals, and its pinned proposer
has no temperature argument. Thus this registration does NOT claim the H214
16-total control is byte-for-byte H212 N16. The explicit H214 fixed evaluated
width governs. This discrepancy and the unavailable temperature hook were sent
to the coordinator before registration. Launch requires resolution recorded
before outcomes; a changed design needs a prospective preregistration amendment.

If fewer than16 unique programs survive, refuse the panel: no refill, reduced
width, dropped arm, adaptive temperature or extra attempt. Native outputs must
have the requested generation length and matching DSL/program SHA256 identities.
Preserve each arm's exact ordered identities. Evaluate only the selected16;
unselected FPS proposals receive features only, never Rust fitness. Deduplicate
identical Rust evaluations across arms without changing their logical16-wide
panels. Oracle-best-in-pool means the best utility among the scored16 only.

## Cohort and source closure

Reuse H212's content-pinned H50/H58 real-S3 regression packet:1,219 TRAIN
contexts/205 families and DL-Val91/19 families, disjoint by family. H214's
primary endpoint is DL-Val91, explicitly authorized by the H214 nudge; H212's
TRAIN primary endpoint is not silently inherited. TRAIN is used for the frozen
retrieval bank and the prerequisite smoke only. For a DL-Val target, the entire
TRAIN bank is family-disjoint; exclude the target family explicitly regardless.
The parent is target-local and its identity comes from stored program order.

Retain H212 manifest pins for selected.json, source_manifest.json, context
features, program/projection caches, pool identities, champion bank, model code,
r882 and baseline reference. No generic corpus search or alternate cohort is
allowed. Parent DSL resolution must use that pinned packet and refuse missing
identities, never search the live journal. Read only the TRAIN and authorized
DL-Val contexts, and only their train-only JTT1/CV envelopes. Every Rust call
uses run_outer_test=false and the same pinned evaluate_gp_batch_rust evaluator,
fitness and CV gate. No sealed, holdout, outer-test or meta-test payload is read.

First reproduce r882's pinned91-context native reference exactly (91/91,
mean1.1377941898164525) as an identity check; a mismatch refuses the launch.
The baseline's native complete-pool utility is also the descriptive r882
comparison. Arm selection always uses the native first np.argmin rule over
the registered16 costs: ties select the first slot, parent first. A completed
gate-passing evaluator result has utility -fitness; a completed CV-gate failure
has utility -2.0. Missing results, execution errors, timeouts or mismatched
identities mean INCOMPLETE, never a numerical zero or utility failure.

## Randomness, endpoints and decision

Fresh proposal seeds214101,214102,214103. Per-context native stream seed is
the low31 bits of the little-endian first8 bytes of SHA256 of
`H214:propose:<seed>:<context_sha256>`. Pair the same seed, parent, target
context and TRAIN retrieval bank across arms. The proposer is width-dependent;
do not assume the16 and64 outputs are nested prefixes.

For each seed and treatment, average paired treatment-minus-NATIVE16 utility
within each of the19 DL-Val families, then average those family means equally.
The point estimate is the median of the three seed-level equal-family means.
With NumPy default_rng(214150), perform10,000 shared family-cluster bootstrap
draws: sample19 families with replacement, use the same family indices across
all seeds and both treatments, recompute each seed mean and its median.
Report percentile95% CI [2.5%,97.5%]; LCB95 is its lower endpoint.

The two registered hypotheses are FPS16 and TEMP16 versus NATIVE16. For each,
the one-sided bootstrap margin p is
`(1 + count(bootstrap median <= 0.001546)) / 10001`; apply Holm m=2.
An arm advances only when median gain>0, LCB95>0.001546 AND Holm-adjusted
p<0.05. Every inequality is strict: ties REJECT. Report each arm and the overall
verdict; overall ADVANCE requires at least one qualifying arm, otherwise
NO_ADVANCE. Any absent seed, family, arm or required capability is INCOMPLETE;
do not reduce the multiplicity family.

Report family-grouped Bradley–Terry Elo using the existing H124 Elo helper,
tie_margin=1e-4, prior=0.01, n_iters=400. Provide separate anchor views for
r882=1500 and NATIVE16=1500, per seed and median over seeds. Elo is descriptive,
never an advance gate. Secondary diagnostics: unique programs, mean pairwise
Euclidean distance on normalized proj16 and oracle-best-of-scored16 utility.
No AUC or speed/tokens-per-second efficacy verdicts.

## Admission, resource budget and closeout

Reuse the fixed H212 runner/proposer interface only after its end-to-end smoke
fix is merged. At the exact merged H214 launch head, rerun the real one-TRAIN-
context smoke across H212 widths16/32/64 through proposal, Rust evaluation,
pool recording and analysis-row writing. Bind its PASS receipt to that full
head. A stale receipt, synthetic-only smoke, missing native temperature hook,
unreviewed temperature source pin or unresolved width interpretation refuses
launch before creating an attempt root or reading H214 utility. No study
attempt is consumed by data-free tests or prelaunch refusal.

One attempt only, fresh external root `~/jope-data/h214/attempt-001`, CPU on
xbabe1, nice19, at most4 numeric/evaluator threads and CUDA_VISIBLE_DEVICES="".
Fetch the merged source into the owned registered xbabe1 worktree first.
Two-hour wall cap from attempt launch, including pauses and all study work;
resume never resets it. Instantly stop at disk used>93.8%, free<40GB decimal,
MemAvailable<16GiB or aggregate process-tree RSS>=6GB decimal. Stop immediately
when load5>0.85*online cores; stop when either load1>0.85*cores or host CPU>85%
persists continuously300s. Record observed resource samples. Never use xbabe0,
xbabe3, GPU or the live learning loop.

Prereg PR open by2026-11-02T10:30Z. The owner authorizes squash merge after
targeted local tests; hosted checks are waived, not reported as passing.
Launch from the merged head only after all admission gates pass.
At closeout, write README with hypothesis/setup/outcome/key metrics/verdict/next
step, metrics.json<100KB, and one H-wave index row; open and merge a small
closeout PR. Keep large arrays, inputs, outputs and runtime receipts outside Git.
Post host/PID/run root/launch time/deadline to H214_claim.txt.
