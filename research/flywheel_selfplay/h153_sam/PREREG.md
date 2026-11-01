# H153 — SAM / ASAM r882 adapter continuation

Status: prospective preregistration; no scored outcome or optimizer update has run. Exploratory research on the repeatedly exposed 91-context DL-Val panel; no sealed-test qualification or serving authorization.

## Hypothesis and design

Sharpness-aware continuation may improve the r882 adapter’s DL-Val utility by reducing sharp-minimum overfit. Copy H149’s r882 setup, model, frozen DOPE, loss, sampler, fold, train-only readout, and per-cell 40-update budget. The only intended treatment change is the optimizer step. Arms: CTRL (AdamW lr=1e-4, wd=1e-4), SAM (rho=0.05), and adaptive SAM (ASAM rho=0.5, eta=0.01). All use gradient clipping at 1.0 and the identical paired batches and seeds. SAM and ASAM use two training forward/backward passes per update; CTRL uses one. Nine cells, 360 updates.

Seeds 153101–153103; five-fold DL-Train family assignment uses seed 153140. Fold 0 is excluded from fit; remaining four folds train the adapter. The full 1219-context/205-family DL-Train panel is secondary and explicitly TRAIN-exposed/descriptive only. No validation examples enter training. No sealed, holdout, or Q1 data.

## Regression and manipulation checks

The first scored operation must reproduce all 91 native r882 DL-Val utilities exactly, including exact mean 1.1377941898164525. Any discrepancy yields INCOMPLETE and zero optimizer updates. The checkpoint is an owned copy at `/home/ubuntu/jope-data/h153/inputs/round-882.pt`, pinned to SHA256 `0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64`; reference SHA256 `30853cad78d860870338316dfb5a4e869e46dc5c22d42fe4ab9d5f24f31c373a`.

Report adapter L2 drift and a fixed radius-0.05 training sharpness proxy per arm. For either SAM arm, if median drift is within the inclusive 5% CTRL band and its complete 3x91 utility panel is bit-identical to CTRL, label DOSE_INVALID; do not call it a valid null. Require the 29,211-parameter/47-tensor adapter inventory and bitwise frozen-state/DOPE checks.

## Outcomes and inference

Primary outcome: native argmin utility, paired SAM/ASAM versus CTRL across 91 DL-Val contexts, averaged within each of 19 families, then median across three seeds. Also report each arm versus the exact r882 baseline and Elo with r882=1500 (no AUC). Use 10,000 shared family-cluster bootstrap draws, seed 153141; percentile 95% CI and one-sided tail p; Holm step-down over exactly two contrasts. Ties reject. Report per-seed gains, invalid picks, top-three positive-family share, Elo and CI. Advance only if dose is valid, median gain exceeds +0.001, CI lower bound >0, at least 2/3 seed gains >0, concentration <=0.80, and Holm-adjusted p<0.05. A BREAKTHROUGH is separately flagged if any SAM arm has median gain versus r882 >0 and CI lower bound >0.

## Execution safeguards and flow

CPU only on xbabe2, nice 19, at most four OS threads, RSS <6e9 bytes, scratch <3GB, free disk >=40GB, and filesystem use <95%; stop if load exceeds 85 or MemAvailable falls below 16GiB. Every threshold breach is terminal INCOMPLETE; never pause and resume a scored attempt. No GPU. Assert frozen DOPE SHA256 `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`. Input access is restricted to H149’s pinned manifest and owned outputs. No large Git objects.

This preregistration and its tests/index row require self-review, read-only cross-review by @rd_argon or a free super_merge reviewer, and squash merge by a super_merger. Only then: zero-evaluation smoke, followed by one run. No retry or cached-cell reuse. Closeout will publish README, compact metrics, and index update in a small PR.
