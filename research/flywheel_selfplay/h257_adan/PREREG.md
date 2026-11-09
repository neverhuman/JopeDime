# H257 — Adan adapter continuation

Owner rd_astra [ASTRA], assignment `~/nudges/hw_t1335_astra_h257.md`.
This protocol and all executable source must be pushed to **origin/v1/next** before Native91,
smoke, pool reads or training. Prepublication checks use data-free fixtures and cohort metadata only.
No main/fastlane merge or PR2071 mutation. No source or hyperparameter tuning after outcomes.

## Hypothesis and frozen arms

Adan's smoothed gradient differences may improve transfer under the same 40-gradient-evaluation
budget. Every arm starts from a separate restoration of the read-only r882 copy, trains only its
47 adapter tensors / 29,211 coordinates, and keeps DOPE `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`
and every non-adapter parameter frozen.

- CONTROL: exact H170 AdamW lr1e-4, betas(0.9,0.999), wd1e-4, eps1e-8.
- ADAN, primary: lr1e-4, betas(0.98,0.92,0.99), wd0.02, eps1e-8.
- ADAN_HALF, secondary: identical Adan with lr5e-5.

Adan follows the official single-tensor, non-fused **proximal decay** variant (`no_prox=False`).
For the globally clipped gradient g, delta=g-g_previous; the first delta is zero. Three zero-initialized
EMAs track g, delta and (g+beta2*delta)^2 with beta1, beta2 and beta3 respectively. All are bias-corrected
at group step t. Subtract lr*(m_hat+beta2*d_hat)/(sqrt(v_hat)+eps), then divide the parameter by
1+lr*wd. Parameters with None gradient have no update, decay or state change; the group step advances.
The current clipped gradient initializes previous on first participation. No schedule, restart,
internal clipping, gradient amplification or extra objective.
[Official implementation](https://github.com/sail-sg/Adan/blob/2c65beaf400775753b155da4082c3819fea689b9/adan.py)
is pinned in the manifest and tested on data-free gradients; no optimizer dependency is added.
Adan changes both the update rule and the specified weight decay, so this is a fixed recipe comparison,
not an isolated causal test of gradient-difference smoothing.

Every arm uses the inherited H146/H170 CONTROL objective: mean smooth-L1 on raw costs plus
0.1 ListNet KL with temperature0.2; valid costs clamped[-4,2], invalid-gate targets+2.
Evaluation keeps original utilities: valid=-native fitness, invalid=-2. Global norm clip1 after
backward precedes every optimizer step. All arms use identical 8-context ×24-candidate minibatches:
top-fitness four reveals and 20 sampled remaining candidates, original H156/H170 data order,
stream_seed(seed,step). No family-balanced sampling or z-normalization from H250.

## Fixed design and assessment

Reuse the merged H247/H251 CONTROL machinery, with H250's pinned Git source loading, raw-cost
custody, post-Native91 three-update smoke, own-protocol read allowlist and directory-fsync fixes.
Preserve the same engine, loss, batch planner, selector, immutable DOPE guard and canonical Elo.
Only optimizer arms and the stricter 3/3-positive decision differ scientifically.

- Seeds257101–257103, one family partition seed257140, bootstrap seed257150.
- DL-Train1,219 contexts /205 families; partition5×41 families, shared across optimizer seeds.
- 15 cells per arm. Each has123 optimization FIT,41 SELECT,41 ASSESS families.
- DL-Val91 /19 families is separate from every TRAIN family; no sealed/meta-test/holdout reads.
- Snapshots0/10/20/30/40. SELECT chooses earliest best equal-family mean, numerical ties1e-12.
  Accept only SELECT gain>.001+1e-12 AND paired-family bootstrap LCB95>0; otherwise step0 r882.
- All three arm selections **within each cell** are fsynced, then the directory is fsynced before
  any trained-policy ASSESS/DL-Val reads for that cell. This is not a global 15-cell lock.
- Each unique selected/best/step40 policy is assessed once per cell/arm/panel; step0 reuses baseline.
  Raw SELECT/ASSESS/DL-Val costs, choices, gates, utilities, snapshots and update traces are retained.
- Each DL-Val seed is a five-cell **utility average**, not a prediction ensemble or single checkpoint.
  TRAIN strict OOF uses the cell's held-out ASSESS predictions.
- Native91 first: all91 r882 utilities and mean1.1377941898164525 must match exactly, plus native/cached
  forward bits. Only afterward open TRAIN pools and perform one smoke update per arm on the first
  original FIT minibatch, check changed parameters/finite gradients/frozen DOPE, restore exact r882.
- Primary budget15×3×40=1,800 gradient evaluations/updates, plus three smoke updates. Fresh optimizer
  state for every arm/cell. Single process attempt, no restart/resume or budget/deadline reset.

## Decision and reporting

Primary statistic: median across3 seeds of DL-Val equal-family utility gain versus r882.
Paired family bootstrap10,000 draws; CI95 of the seed-median gain. One-sided p=(1+count(boot<=0))/10001;
Holm m=2 across ADAN and ADAN_HALF, within-wave only. ADVANCE requires **all**:
median>**+0.002+1e-12**, LCB95>0, Holm p<.05, and **3/3 seeds positive**. This freezes the inherited
strict +.002 margin where the brief says strict margin and retains Holm's inherited .05 threshold.
Wave advances if either registered arm passes; primary and secondary verdicts are reported separately.
Ties are NO_ADVANCE. CONTROL comparisons, step40 and unconstrained SELECT-best are descriptive only.

Report all strict/unconstrained/step40 DL-Val and TRAIN OOF panels, per-seed gains, family CIs,
harm families(loss>1e-4), parameter L2 drift and r882-anchored Elo1500 (canonical400-iteration BT,
prior.01,1000 family bootstrap draws). Context27 is retained and separately reported. The historical
plateau1.1389572 and archived-only r3175 Elo are descriptive; the pinned historical vector is
postprocessing-only, H226 CPU parity remains failed, no new r3175 inference. No AUC or speed verdict.

Repeated step0 fallbacks can force median/CI to zero; that is not equivalence. Folds overlap and seeds
share data; bootstrap conditions on fixed selected policies and the already exposed development panel.
It does not represent full training/selection uncertainty or correct the campaign-wide adaptive search.
Fixed candidate-pool continuation does not establish a live GP-generation mechanism. No serving promotion.

## Resources, provenance and publication

xbabe2 cores52–55, nice19, one numeric thread, aggregate<=4, aggregate RSS<4GiB,
free disk>=300GB. H239-A1: admit RAM>=30GiB/load5<=100; cooperative pause RAM<20GiB or load5>108;
resume RAM>=25GiB sustained30s and load5<=100; hard stop RAM<12GiB sustained30s or<6GiB instantly.
Pause only our worker at cooperative boundaries; no foreign signals or jobs. CPU only, no other host,
GPU, live flywheel, cores40–51/57–60, PR2071 or fastlane access/mutation.

Prereg push deadline14:15Z, absolute compute stop16:00Z, closeout16:45Z on2026-11-03. If incomplete,
report CLOSED INCOMPLETE with actual updates and unknown efficacy, never zero-fill incomplete results.

Real S3 lineage and all276 dataset IDs are pinned in provenance.json and checked against authoritative
catalog v246; all rows with has_meta_test=true and protected contexts are excluded. Dataset labels remain
intact. Local derivative recovery sources/versions and original historical-version gaps remain explicit.

Run targeted pytest, Ruff, JSON, source/fold/schedule bindings, optimizer-reference parity and diff-check
before exact-head same-agent review and non-force v1/next push. No schema, records-pin, rootREADME-glance,
CI config or global registration edits; full hosted CI NOT_RUN under the explicit PR2071 split.
Closeout contains README, small metrics.json, manifest and own index rows on v1/next; no new per-wave PR.
Archive all durable evidence create-only under `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H257/astra-t1335/`,
verify exact-VersionId recovery/member hashes, then reclaim redundant local bulk. No objects in Git.
