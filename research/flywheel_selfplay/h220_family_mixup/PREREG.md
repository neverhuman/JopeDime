# H220 — within-family mixup

Prospective preregistration under `hw_t0500_astra_h220.md`; no H220 GPU outcomes.
Merge this source before any native regression, smoke or training. This study
reuses [H219](../h219_fbal_accum_run/PREREG.md)'s tested execution harness.

## Hypothesis and fixed design

Within-family convex mixing of adapter inputs and targets may reduce the harm
of training from r882. Arms are unchanged R882, CONTROL, and MIX. The optional
cross-family placebo is omitted prospectively. CONTROL and MIX use the same
H219 context-uniform sampling, candidate subsets, AdamW and training budget.
No arm, alpha, probability, threshold or seed is chosen from H220 outcomes.

Seeds 220101–220103 each use a distinct whole-family partition (fold seeds
220141–220143). Sorted families are permuted then assigned round-robin to five
folds, as in H219. Rotate all five held-out folds: 164 FIT and 41 held-out
families per cell, paired OOF coverage of 205 families /1,219 real-S3 DL-Train
contexts. This gives 30 fits, 40 updates per fit, four 16-context microbatches
per update (effective batch64): 1,200 scientific optimizer updates. A separate
four-update CONTROL smoke, stream220001/outer0, has no held-out read or science
score and counts within the same attempt. It is not used for a speed verdict.
[fold_manifest.json](fold_manifest.json) pins context identities;
[fold_jaccard.json](fold_jaccard.json) records all overlaps against H210/H217/
H218/H219 partitions. This reused development cohort is not sealed qualification.

Each cell stages its own SHA-verified, read-only r882 checkpoint copy and resets
the model to independent clones of exactly those r882 adapter tensors. There
are 47 adapter tensors /29,211 parameters, 41 active gradients and six inactive.
No random initialization. DOPE composite
`c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`
and every nonadapter tensor remain frozen and are checked before/after cells.
AdamW LR1e-4, weight decay1e-4, betas(.9,.999), gradient clip1; eval/dropout off.
H109's sampler, LR, accumulation and checkpoint staging are called directly;
only its seed-admission whitelist is extended. RNG seed is outer seed×10+fold.
Each context slate contains four best-fitness FIT reveals and 20 random
remaining candidates. Primary slates and sampling are paired across arms.

## Exact mixup semantics

For each MIX primary context draw, independently select mixing with probability
0.5. Select the partner uniformly among distinct FIT contexts from the same
family. Singleton families are identity rows. Draw lambda from Beta(0.4,0.4).
RNG derives from H220/seed/fold/update/microbatch, independently of primary
candidate-subset RNG. Both contexts must be in the current FIT closure.

Each partner has its own 24-candidate slate with four best-fitness reveals and
20 random remaining candidates. Pair slots by position (four reveal ranks,
then random slots); this does not assert semantic identity between programs.
The shared lambda convexly mixes program features, continuous context tokens,
masks (Boolean source masks are converted to float), priors, scale, revealed raw fitness and gate weights, and separately
computed frozen anchor costs/priors. The discrete reveal indices are identical
0–3 on both sides and remain fixed. The frozen anchor is evaluated on each
unmixed context; it is not trained or evaluated on synthetic programs.
The adapter sees the mixed continuous inputs. Soft masks follow its existing
attention-mask threshold behavior. Inference always uses original full-pool
inputs without mixup.

Targets are transformed on each side first: successful fitness clipped [-4,2],
failed gate cost2. Mix those cost targets with the same lambda. Apply the H219
SmoothL1 plus 0.1 KL loss (temperature.2) against the mixed target. This avoids
converting a convex gate weight into a Boolean outcome. Original dataset labels
and pool files are never changed. Identity rows preserve the original slate,
inputs and effective loss targets. Per-fit receipts report attempted rows,
actual distinct partners and singleton rows. No cached label from another
family, fold or native91 baseline can enter a training batch.

## Regression, closure and ordering

The owner explicitly requires native91 and thereby changes H219's prohibition
on DL-Val baseline reads. Before any adapter update, unchanged r882 must match
all91 pinned reference utility values bit-exact, with mean
1.1377941898164525. This is regression only: no trained arm is evaluated on
DL-Val. Training and scientific readout use DL-Train only. The feature reader
restricts decoded rows to the declared TRAIN plus native91 roster, and native91
predictor entries are cleared before the treatment engine is constructed.
No sealed, meta-test, outer-test or holdout payload is read.

Cached static r882 scores must match native scores bit-exact on all1,219 TRAIN
contexts before training. Every cell must show actual40 optimizer steps,
nonzero gradients, changed adapter hashes and changed FIT forward scores.
All30 terminal models/proofs are saved, fsynced and S3-recovered under a global
lock before any held-out utility read. No B selector, early stopping, checkpoint
search or fallback. First argmin over the original full candidate pool defines
selection; utility is -fitness for a passed gate, otherwise -2.

## Readout and verdict

Primary MIX vs R882: equal context weight within each family, equal family
weight within each seed, then median of the three seed gains. Bootstrap10,000
paired family draws (seed220150), sharing each draw across all arms/seeds.
Report two-sided95% CI and one-sided95% LCB. There is one primary, so Holm m=1.
MIX vs CONTROL is secondary/diagnostic; CONTROL vs R882 is reported in full.
Family-grouped Elo (seed220151) anchors r882=1500. No AUC, speed metric or U4.

To preserve H219's strict decision standard, CANDIDATE requires LCB95>.001,
Holm p<.05, harm<=10%, and three positive seed gains. Tails count bootstrap
values<=.001+1e-12; LCB must exceed .001+1e-12. Harm counts families whose
seed-average paired utility gain is below -.01. Ties reject. Failure of any
gate is NO_ADVANCE. This margin/harm/seed choice is fixed before outcomes;
H220's owner specified the primary but delegated routine choices. No serving
promotion follows this development study.

## Execution and evidence

Use xbabe2 GPU0 under the shared FIFO qualification.lock, numeric1/nice19.
Retain H219's RAM20GiB, free200GB, RSS<6GB, output<2GB and load85%-of-affinity
limits, including separate300s overload clocks for load1/hostCPU; load5 and
admission guards act immediately. Record swap-free and host resources; a full
swap device alone is not a new scientific stop rule. One attempt, fixed90min
including import/setup, native91, smoke, full run and worker finalization.
Latest launch14:00Z; hard finish no later than15:30Z, 2026-11-02. The supervisor
claim is create-only and its absolute deadline cannot reset. Failure preserves
partial evidence and closes INCOMPLETE; no retry.

[manifest.json](manifest.json) pins own source, parent helpers, versions and
inputs; [provenance.json](provenance.json) records real-S3 lineage. Evidence
writes use new keys with create-only PUT, exact VersionId GET and SHA256 checks.
Large artifacts stay outside Git. Closeout records README/metrics and both wave
indexes, verifies recovery, and uses the owner-authorized targeted-check squash
lifecycle. No scientific code changes after outcomes.
