# H155: pairwise label smoothing and TRAIN-fold temperature calibration

Prospective preregistration. No H155 optimizer update, model scoring or calibration has run.

Owner: gpu_tokbench_codex; authority: /home/ubuntu/nudges/gpu_tokbench_codex_lift_t2215.md.
Lift is the only outcome criterion. All D13/D14/tokbench speed work is parked.

## Fixed design

Start every cell from a byte-exact owned copy of r882, SHA256
0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64.
Only the 29,211 adapter parameters in 47 tensors train. DOPE is frozen,
excluded from optimizer groups, kept in eval mode and checked before/after
every update against c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814.
All other state remains bitwise unchanged. No writes to r882/r782, pinned
checkpoints, or the xbabe1 live flywheel/PID2297929.

Seeds: 155101, 155102, 155103. Five equal-sized family groups made by
NumPy default_rng(155140).permutation(sorted(TRAIN families)), array_split(5).
Group0 is inner evaluation; group1 is temperature calibration; groups2-4
fit the adapter. These family sets are disjoint. All 205 TRAIN families
were previously exposed to r882; inner evaluation is conditional continuation
evidence, not an untouched generalization test.

Five arms for each seed: CTRL(eps0,T1), LS05(eps0.05,T1),
LS10(eps0.1,T1), LS05_TEMP(eps0.05,learned T), LS10_TEMP(eps0.1,learned T).
Each has 40 AdamW updates, lr1e-4, weight_decay1e-4, gradient clip1.
Use paired 16-context/192-candidate H153 batch plans, replacing its registered
seed list with H155's. Keep H153's smooth-L1 plus 0.1 listwise-KL objective,
and add 0.1 pairwise binary cross entropy. For unequal target-cost pairs
i<j, logit=(pred_j-pred_i)/T and hard label=(target_i<target_j).
Smoothing maps hard labels to eps+(1-2eps)*label. Exact ties are excluded.
Targets clip native cost to [-4,2], with failed gates assigned cost2.

Fit one positive scalar temperature per smoothing level, before adapter
updates, from frozen r882 predictions on calibration families only. Use 16
evenly spaced original candidate positions per calibration context and all
unequal target pairs. Optimize log(T) from0 with Adam lr0.05 for200 steps,
clamped to [log(0.05),log(20)], against the same smoothed pairwise loss.
Temperature is then frozen during adapter fitting. Positive temperature alone
cannot change an argmin; any selection lift comes from changed training
gradients, not posthoc rescaling. Do not fit T on group0 or DL-Val.

## Data and baseline

Use only the approved, real-S3-derived H50 selected.json/source_manifest.json
and candidate pools plus H58 context_features.npz, with row/payload SHA checks.
Metadata contains 1219 DL-Train contexts/205 families and91 DL-Val contexts/
19 families. Both are derived exclusively from source TRAIN/inner data;
DL-Val is an exposed development panel, not sealed/outer test.
Require unique context IDs, no TRAIN/Val family overlap, regression tasks,
source_split in {train,inner}, no has_meta_test, and no protected payload paths.
No synthetic training datasets; no sealed/holdout/meta-test payloads.

First scored action: reproduce all91 r882 native CPU utility values exactly
and the context mean1.1377941898164525 from the pinned reference
30853cad78d860870338316dfb5a4e869e46dc5c22d42fe4ab9d5f24f31c373a.
If parity fails: INCOMPLETE, zero training updates.
Report all5x3 cells versus r882 and CTRL on inner group0, the91-context
development panel, and the full1219-context/205-family TRAIN panel.
Full TRAIN scores are descriptive and exposed; no selection on either
reported panel, no adaptive tuning or retries. Evaluate native argmin utility,
failed gate=-2, never reuse foreign experiment outcome cells.

## Inference and execution

Equal-family utility deltas, per-seed and median across3 seeds; Elo anchored
at1500 for r882 from equal-family win/tie/loss rate (ties half, Jeffreys
0.5 correction). Shared10000 family-cluster bootstrap draws, seed155141,
95% percentile intervals. Holm correction across the4 treatment-vs-CTRL
comparisons on group0; report comparisons to r882 too. A promising arm
requires positive median family gain and positive95% lower bound versus
both CTRL and r882 on group0, Holm p<0.05 and at least2/3 positive seeds.
This is exploratory, not serving promotion or sealed-test qualification.

Execute only after this preregistration is squash-merged and the executor
source/input/fold manifest is pinned and reviewed. One run, no outcome-driven
amendments, no speed gates. CPU nice19, <=4 threads, hostCPU<=85%.
GPU only with canonical flock-n /home/ubuntu/jope_plan/qualification.lock;
if busy prep/run CPU. Stop if free disk<40GB or used>=95%; protect others'
jobs and never touch xbabe0 or admiralty.
