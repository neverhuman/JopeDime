# H244 preregistration — weight-perturbation prediction smoothing

Authority: `hw_t0440_astra_h244.md`, assigned to [ASTRA]. Freeze before outcomes. Zero optimizer updates, no fitting of weights. Test whether averaging predicted costs under small adapter perturbations reduces argmin instability and improves DL-Val91 regression utility. WSINGLE separates smoothing from a single perturbation; it is report-only.

## Fixed construction

From an offline r882 copy, for each of 47 adapter parameter tensors (29,211 coordinates), including scalars, biases, norm gains and 3-D embeddings, set `theta_k = theta + sigma * rms64(theta_t) * N(0,I)`. The two fixed adapter buffers and all other state, including DOPE and anchor, remain unchanged. Zero-RMS tensors stay unchanged. The exact tensor order, shape, parameter count and float64 RMS values are frozen in [perturbations.json](perturbations.json).

K=8 perturbed members plus r882 as member 0, each with equal weight 1/9. Average **predicted costs**, in float64, then take the lowest-index argmin. This is not parameter averaging or voting. Sigma grid `{0, .01, .03, .1}` ordered from smallest to largest. Sigma 0 uses the cached native r882 result exactly, avoiding floating-point averaging artifacts. Each perturbation is computed in float64 then cast back to the tensor dtype. No renormalization or clipping.

Seeds244101/244102/244103, eight member seeds per study seed: `study_seed*100 + k`, k=1..8. NumPy PCG64/default_rng standard_normal in the explicit tensor order. A seed/member uses the same Gaussian direction for all three nonzero sigmas and all folds, and identical weights on TRAIN and DL-Val. All 24 member seeds are explicit in perturbations.json. Each scoring batch restores r882 after its eight members. No autograd or optimizer is used.

## Selection and evaluation order

1. Verify immutable input/source hashes and real-S3 provenance; exclude `has_meta_test=true`, sealed/holdout/meta-test and live-loop data. The inherited corpus is 1,219 TRAIN contexts/205 families and 91 DL-Val contexts/19 disjoint families.
2. Native91 must reproduce all 91 r882 utilities exactly and mean **1.1377941898164525**. Prove cached/native cost bit identity before proceeding.
3. Score the three smoothed sigma policies for each of three seeds on TRAIN only; obtain WSINGLE from the same member-1 prediction. Frozen family folds244141/244142/244143 form five outer cells per seed. FIT consists of four folds (164 families); ASSESS is the remaining 41 families for descriptive TRAIN OOF. No ASSESS or DL-Val data enters a cell's selector.
4. WSMOOTH_SELECT chooses maximum equal-family FIT mean among baseline/.01/.03/.1; numerical ties (1e-12) prefer baseline then smallest sigma. Accept only if FIT gain > .001 + 1e-12 and family-bootstrap LCB95 > 0; otherwise r882. WSINGLE uses member 1 at that **selected** sigma, with no separate tuning and r882 on fallback. It also reports its contrast with WSMOOTH_SELECT.
5. Write all 15 selections exclusively, fsync the file, write a hash-bound exclusive lock receipt, and fsync the directory **before** any post-Native91 DL-Val scoring. Recheck the selection hash before every DL-Val policy. Each fixed seed/sigma policy is scored once; baseline is reused from Native91. Save mean and member-1 cost vectors plus choices/utilities for replay. Never exclude context #27 (zero-based selected.json dev order; its exact context/family identities are in fold_manifest.json); report its baseline, fixed and selected utility/choice/gate outcomes separately.
6. For each seed, average the utilities of its five FIT-selected policies per DL-Val context, matching the H242 harness. This evaluates a uniform mixture of five selection decisions, not a new ensemble of their predictions. TRAIN OOF uses each context's held-out cell. Neither cells nor repeated r882 fallbacks count as independent replications.

## Decision and reporting

Primary is WSMOOTH_SELECT vs r882 on DL-Val. Equal-family paired utility, median of the three seed gains, 10,000 paired family bootstrap draws with the same resampled families across seeds (fixed seed244150). CI95 is the percentile interval of median gain. One-sided p=(1 + count(bootstrap median <=0))/10001; Holm m=1 for the sole primary. This conditional bootstrap is inherited, not a fresh untouched-test significance claim.

ADVANCE requires median gain > .002 + 1e-12, family LCB95 >0, Holm p<.05 and at least2/3 seed gains >0. Ties reject. WSINGLE never advances the wave. Report all seeds, family CI95, family-weighted Bradley–Terry Elo (r882=1500, H148 MM400/prior.01, 1,000 family bootstrap draws), harm families per seed (loss >1e-4), TRAIN OOF, selection counts and identical-policy dependence. Report context-mean utility relative to historical plateau1.1389572 (r3175) descriptively; that value is not a family-weighted comparison identity. No AUC or speed verdict.

DL-Val is an exposed development panel used historically for r882 selection. This wave's exclusion rules do not certify historical r882 training. No live promotion follows an ADVANCE verdict.

## Resources and custody

One run on xbabe2 CPU48–51, nice19, one math thread, aggregate threads<=4, RSS<4GiB, free disk>=300GB. Admission RAM>=30GiB/load5<=100. Cooperative pause between batches/members/pool loads when RAM<20GiB or load5>108; memory recovery requires >=25GiB continuously30s, load recovery<=100. Hard stop only RAM<12GiB continuously30s or<6GiB instantly, RSS/disk/thread violation, or absolute07:30Z. Only our new worker process group is controlled. No process restart or reroll. No GPU/xbabe0/xbabe3/PR2071 or other-job control.

Prereg merged before05:30Z and before outcomes; compute stops Nov3 07:30Z; closeout before08:00Z. Otherwise CLOSED INCOMPLETE with unknown efficacy. H239 A1 guard is retained, while H244's single-run rule disallows process resumes. Partial scoring files and resource receipts retained. Source/inputs/predictions/selection lock/full metrics stored under `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H244/astra-t0440/` with exact VersionId, SHA256 and recovered member verification. Small metrics/manifest and both indexes in Git. Full remote CI is NOT_RUN under the explicit owner local-check/admin-squash exception; local targeted tests/Ruff/records/pins/JSON/diff and same-agent exact-head review remain required.
