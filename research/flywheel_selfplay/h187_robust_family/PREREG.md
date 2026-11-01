# H187: robust per-context loss and family-balanced fresh adapters

Owner gpu_tokbench_codex; authority `~/nudges/hwave_t0730_h187_tokcodex.md`.
Preregistered before H187 model training, timing pilot, candidate scoring or outcomes.
H180/H184 motivate this study; their cohorts are reused, so this is a fresh-fold
comparison on the same real-data bank, not independent new-dataset confirmation.

## Hypothesis and fixed design

The live objective may over-weight difficult contexts and large families. Test two
single changes against the H180 fresh-init control. Reference R882. Three candidate
arms, Holm m=3 vs R882: FRESH160, FRESH160_RNORM, FRESH160_FBAL. All 160 updates
unless the prespecified timing-only rule selects 80 for EVERY arm. Names retain
FRESH160 for identity; actual updates are always reported. No snapshot search.

Use seeds187101–187103, fold_seed187140 and boot_seed187141. Five fresh cyclic
3/1/1 family splits: sorted205 TRAIN families, seeded permutation into5 groups;
A=outer, B=next group, FIT=other123 families. Every family gets B once and A once.
All3 seeds/all5 folds mandatory. FIT/B/A disjoint within each model; the same205
families rotate through roles across models. Pooled A/B therefore correlate and
must not be described as independent population replications.

Import the merged H180 Engine, native forward and native fresh adapter constructor.
For each (seed,outer), seed the constructor with H177 stream_seed(seed,outer,
"fresh-native-init-v1"). Clone the SAME47 tensors/29,211 parameters across all3
arms. Fifteen independent constructor initializations, paired across arms. Eval
mode, no dropout. Only the adapter trains; DOPE and every non-adapter tensor frozen.
AdamW lr1e-4, weight_decay1e-4, betas(.9,.999), ordinary gradient clip L2=1.

Control and RNORM share the exact H180 batch plan: eight uniformly sampled distinct
FIT contexts, four lowest-fitness candidates as reveals plus20 random unrevealed
candidates per context. Shared seed H177 stream_seed(seed,outer,"fit-123",
"h156-batches-v1"); same optimizer step RNG across paired arms.

## Precisely defined treatments

Control/FRESH160 and FBAL use the UNCHANGED H180 live objective: per-context smooth
L1 cost loss + .1 KL(softmax(-target/.2) || softmax(-prediction/.2)), averaged over
contexts. Cost target = fitness clipped[-4,2] on gate pass, otherwise2. Failed
candidate utility in evaluation remains exactly-2; evaluation labels are not clipped.

RNORM preserves the live per-context loss L_c. On the SAME24-candidate FIT batch,
compute frozen anchor-only loss L_anchor,c with the same target and detach it.
Scale s_c=max(L_anchor,c,.001); z_c=L_c/s_c (no centering). Objective = mean rho_2(z),
where rho_2(z)=z²/2 for |z|<=2, otherwise2(|z|-1). Thus Huber influence d rho/dz
is clipped to[-2,2], and tail gradients remain nonzero. This bounds influence in
normalized LOSS units; it does not promise a bound of2 on total parameter gradients.
The floor.001 is fixed before outcomes. Anchor scales use FIT labels only and no
candidate prediction. Log scale floor usage, z/influence maxima and clipped counts.

FBAL changes ONLY context sampling: weight context i by1/(number of FIT contexts
in its family), normalize to sum1, then sample8 contexts WITH replacement. Each
family has exactly1/123 probability PER draw, including families with one context;
duplicated contexts in a minibatch are allowed. This intentional replacement rule
avoids claiming exact balance from unequal-probability sampling without replacement.
Use the same H180 four+twenty candidate mechanism conditional on sampled contexts;
FBAL's context/slate RNG path differs because sampling is the treatment. Control
and RNORM plans must match byte-for-byte; FBAL weights/role closure are persisted.

## Baseline, timing and stage order

Native91 DL-Val r882 utility bytes and full candidate-cost bytes must match the
restored gold first, mean exactly1.1377941898164525, before TRAIN pool decoding or
any optimizer call. Feature-cache bytes may be hashed/read for baseline construction;
TRAIN true outcomes are gated. The common r882 reference on TRAIN is computed once
before treatment; it is not candidate A confirmation.

Timing pilot is the first80-update FIT-only prefix of the first FRESH160 control
cell. Retain its optimizer/weights, so it is part of either final80 or160 updates,
not a discarded/retried trajectory. No B/A candidate scores until timing is locked.
Project7200 updates × median pilot time (excluding warmup step1) × fixed1.25 overhead
factor +1200s evaluation reserve. If <=3h use160; otherwise all45 cells use80 (3600
updates). If the80 projection still exceeds3h, stop INCOMPLETE. Never fewer seeds,
folds or arms. Candidate budget is immutable before the first B read. No loss/gain
pilot adaptation or new attempts after terminal interruption.

Stage1: train45 FIT-only trajectories. Score ONLY the final prespecified step on
that model's B41. Save all adapter states, FIT IDs, plans, losses and candidate
cost vectors/choices/true utilities. No candidate A forward until ALL pooled B panels
(3 seeds×1219 contexts×3 arms) complete and a selection lock is fsynced.

Directly import H177_DECISION_RULE/pooled_b_select and StageOrderGuard from the
hash-pinned merged H177 source, with committed H178 receipt/source hash. Map
R882→CTRL and FRESH160/RNORM/FBAL→ACC4/ACC16/OC15. Invoke the imported function
unchanged and use its built-in THREE-treatment branch (`admitted_treatments3`,
`holm_p_treatments3`). No rewritten selector math. It averages the3 seeds within
each family BEFORE10000 paired family bootstrap draws, then requires gain>
.001+1e-12, LCB95>0 and Holm-m3 p<.05. Ties reject. The dummy CTRL is exactly R882.
Native slot metadata says step40; H187 records the actual80/160 final update budget
separately and binds StageOrderGuard's expected steps to it before B. Do not mistake
native slot metadata for H187's actual trained step. H178 calibrates the four-arm
mechanism; this m3 branch is owner-prescribed, not a new empirical m3 calibration.
Persist the raw imported decisions, mapped m3 decisions, source/receipt hashes,
actual step and lock hash; refuse A if the lock is missing/corrupt/changed.

Stage2: restore each locked model and score its disjoint A41 plus baseline-only
DL-Val91. Pool A exact-once across folds. DL-Val is DESCRIPTIVE: retain5 fold models
per seed; summarize their455 model/context pairs with19 equal-weight families.
This is the expected outcome of a uniform random fold model, not a score-averaged
ensemble. Report its unique91 contexts and -2 counts out of455 pairs explicitly.

## Reporting and verdict

Per arm: per-seed gains, median and10000-draw paired family95% CI, seed-averaged
H177 functional/CI, Holm-m3 p, exact-2 context count and equal-family Elo vs r8821500
with10000-draw paired family CI. Use common family resamples across seeds and arms;
MEDIAN reduces seed axis0, leaving10000 bootstrap draws. Elo uses context-pairwise
wins/ties (1e-4) BEFORE family averaging; imported H148/H177 MM400 iterations,
pseudo-win prior.01. Diagnostic RNORM−FRESH160 and FBAL−FRESH160 contrasts are paired
and do not affect admission. Report B/A/Val separately; raw scores/states under S3.

ADVANCE for an arm ONLY if admitted on pooled B AND pooled A median gain>0,
3/3 positive seeds AND median-family-bootstrap LCB95>0. Else NO_ADVANCE. Report
unchosen/unconstrained outcomes too. H178 source-bound receipt must verify inherited
machinery. Do NOT adopt or edit the live flywheel; flag any ADVANCE to the
@flywheel_gp_fused owner for a separate decision. A/B correlation remains explicit.

## Resources and release

CPU xbabe2 nice19/math1; supervisor+worker <=4 threads, aggregate RSS<6GB. Terminal
stop CPU>85%, MemAvailable<16GiB, user-visible disk>=93.8% OR free<40GB, load>85% of
host cores, or3h wall. Only29,211 adapter parameters admitted to optimizer. DOPE
c51a56d0 and non-adapter SHA checked before/after/every update. Durable started and
completed optimizer/backward counts, SIGTERM/SIGINT failure receipt; no automatic
retry. Restore r882 at end to verify source weights while retaining every trained
adapter checkpoint. Real S3 TRAIN only; no sealed/holdout/meta-test, no GPU or
xbabe0/3, live flywheel read-only. Report utility/Elo, no AUC/throughput metric.

Flow: claim, fresh own worktree, prereg+data-free tests, rd_argon cross-review<=15min,
targeted pytest/Ruff/five jope-records gates, owner self squash-merge, ONE guarded
run, independent recount/native checks, README+metrics<100KB+own index/S3 closeout,
self-merge, preserve/independently restore owned source/evidence, remove own lane.
