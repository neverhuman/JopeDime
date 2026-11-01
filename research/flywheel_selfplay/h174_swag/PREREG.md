# H174 — SWAG posterior predictive averaging on r882 continuation

Owner `luna_ns_hbaseline`; reserved by `ENG:hwave-loop`. Authority: `/home/ubuntu/nudges/hwave_t0253_h174_hbaseline.md`. Seeds 174101–174103, fold seed 174140, bootstrap seed 174141. This file fixes the analysis, fit scope, and candidate rules before any TRAIN outcome is opened or optimizer update is made. The frozen DL-Val r882 91-context baseline is the required pre-run integrity gate; it cannot choose an arm, set a hyperparameter, or train the adapter.

## Question and estimands

After one fixed 40-update AdamW continuation from the read-only r882 copy, does a late-trajectory SWA point or SWAG Bayesian model average improve the strict inner-family selector relative to r882? SWA is the point at the arithmetic mean of the 16 late FIT snapshots. BMA05 and BMA10 average predicted **cost vectors** over 16 deterministic SWAG posterior draws at covariance multipliers 0.5 and 1.0, then choose the argmin of those averaged costs per context. This is Bayesian prediction averaging; BMA does not use the single averaged-weight model.

All effects are conditional continuation on the historically exposed r882 checkpoint and TRAIN cohort. No claim is made that prior r882 fitting excluded the current cross-fit families. The treatment trajectory and its SWAG moments, however, use only the current outer fold's 123 FIT families. No B, A, or Val labels are used to fit snapshots, posterior moments, or posterior samples.

## Data and cross-fit

Reuse the H148/H156/H170 family-grouped five-fold cross-fit and paired H156 training batches: 1,219 real-S3 TRAIN contexts in 205 families, with each outer cell using 123 FIT families, 41 selector-B families, and 41 assessment-A families. For outer fold `j`, A is fold `j`, B is `(j+1) mod 5`, and FIT is the other three folds. Each seed assesses each TRAIN context once. The historical 91-context/19-family DL-Val panel is secondary only. All 276 dataset IDs must join `origin=real` in the pinned S3 catalog and the inherited exact numeric closure must verify. No synthetic study-training data, sealed, holdout, meta-test, serving, or live-loop input is permitted.

Before any TRAIN outcome pool is read and before any update, evaluate the frozen native r882 model on all 91 DL-Val contexts. Require exact per-context utility identity and mean `1.1377941898164525`; also require the frozen-anchor cached forward to match native cost bits at step zero. If that gate fails, stop as incomplete with zero optimizer updates. Only 47 adapter tensors / 29,211 parameters may change on a private read-only r882 copy. DOPE is frozen at SHA `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`; verify it and every non-adapter tensor before and after updates/evaluations.

## Training and snapshots

Each outer-fold/seed cell starts from the same r882 adapter, fresh native AdamW state (lr `1e-4`, weight decay `1e-4`, beta1 `.9`, beta2 `.999`, default epsilon), global gradient clipping L2 `1`, H156 loss, and 40 paired 8-context/24-candidate batches. Batches are generated only from FIT context IDs with the seed fixed by this preregistration. There is one trajectory per cell: 15 trajectories and 600 total optimizer updates.

Store adapter-only float32 snapshots at updates `10,12,14,...,40` (16 snapshots). SWAG statistics use these FIT-only snapshots: the posterior mean and population diagonal variance are computed over all 16 snapshots; the rank-10 deviation matrix is the last 10 snapshots centered on that 16-snapshot mean. For each cell, use a sample seed derived from its seed (same standardized Gaussian draws pair BMA05 and BMA10). For draw `s`, with diagonal variance `v`, rank deviations `D`, independent standard normals `z_d,z_r`, and scale `c`, sample

`theta_s = mean + sqrt(c/2) * sqrt(v) * z_d + sqrt(c/(2*(10-1))) * D.T @ z_r`.

The scale multiplies covariance. Generate 16 posterior samples per scale in float32 for the 29,211 adapter coordinates only. The code must check the exact snapshot schedule and explicitly reject any trajectory ID outside FIT or overlapping B/A/Val.

## Candidates and selector rules

The candidate set is `CTRL`, `SWA`, `BMA05`, and `BMA10`.

- `CTRL`: compare r882 and trajectory snapshots 10/20/30/40 on B. Choose the earliest maximum equal-family mean; numerical ties within `1e-12` go to the earlier choice. Select a nonzero step only if gain is greater than `0.001 + 1e-12` and its 10,000-draw paired family-bootstrap LCB95 is above zero; otherwise select r882. Do not try a second-best step after rejection.
- `SWA`, `BMA05`, `BMA10`: each is binary versus r882 on B. Admit the candidate only if equal-family gain is greater than `0.001 + 1e-12` and a 1,000-draw paired family-bootstrap LCB95 is above zero; ties reject. Otherwise use r882. Decisions and the complete raw per-context B candidate panels must be fsynced before any A or candidate-Val predictions are made.

For BMA, average all 16 sample cost vectors first and then take per-context `argmin`; do not average sampled utilities or choices. Retain all 16 sample utility panels as a descriptive posterior-spread diagnostic, not as independent observations. Retain raw B context IDs, costs/choices/utilities, the locked decision and bootstrap draw seed so a red-team replay can reproduce the selection. No A or Val result may change a decision.

## Outcomes and summaries

Primary: equal-family OOF utility gain versus r882 on A, per seed; report the median of the three seed effects and a 10,000-draw family-bootstrap 95% interval using the preregistered H148 aggregation; Holm correction across SWA/BMA05/BMA10. Report canonical H148 Bradley–Terry Elo against r882=1500, including the 91-context DL-Val secondary utility/Elo. Do not report AUC, speed metrics, or a verdict.

Descriptive only: unconstrained inner-B gain for each arm, per-context argmin flips relative to r882, diagonal-variance summaries, and posterior-sample utility spread. Disclose exact-identical selector choices and utility panels across seeds; identical panels are not independent replications. Include all raw B panels and all 15 cell receipts outside Git in the evidence archive.

## Resources, audit, and lifecycle

CPU-only on local xbabe2, process nice 19, Torch/math threads one, aggregate supervisor+worker thread count at most four, aggregate RSS below 6,000,000,000 bytes. Stop before launch or immediately on load above 85% of 102 online cores, MemAvailable below 16 GiB, free disk below 40 GB, used fraction at least 95%, RSS at least 6 GB, or aggregate threads above four. Swap is full. One exclusive attempt; no restart or cell reuse after a partial attempt. No foreign process signals, no xbabe0/xbabe3, no GPU or qualification lock.

Lifecycle: preregistration PR with runner and data-free tests, exact-head review by `gpu_fused_redteam_codex` capped at 15 minutes, targeted local tests/Ruff/all five jope-records verifiers, and merge only after the repository's required exact-head gates pass. Then one monitored run and a separate small closeout PR with README, metrics, custody, and one index row. Large run objects and S3 custody remain outside Git. No serving promotion.
