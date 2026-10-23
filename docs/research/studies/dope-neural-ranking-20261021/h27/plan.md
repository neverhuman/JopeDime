# H27: fitted GP performance fingerprints

Prospective source-preparation plan, 2026-10-22. No H27 numerical input, model, prediction, training curve or ranking result has been produced. Necessary new TRAIN-only probe fits are authorized in principle; an exact source/input/resource freeze and execution grant are still required.

## Question and current support

Can a small, fixed basket of diverse genetic-program predictors provide useful task descriptors for selecting the best recorded GP candidate? Compare one fixed internal heldout fold with the mean of five folds of the same TRAIN-internal partition. These are new **feature measurements**, not replacement target labels.

Preparation identified the existing 850-context, 162-component TRAIN support as a route that can proceed independently of new-cohort admission work. Prior receipt metadata reports 733–1,959 permitted TRAIN rows and 3–100 features per context, with no row subsampling on these selected cases. All 850 declared cache files are present by metadata checks. Current array bytes, contents and row membership have **not** been newly read or reverified. A capped or otherwise projected input must never be silently described as the entire GP training matrix.

The archived source-specific S1 two-fold ranking targets remain unchanged. The new five-fold probe partition and the neural component cross-fit partition are separate contracts. Existing CAL/DEV tasks, sealed outer reservations and known reserved families remain outside learning and selection. This historically exposed support permits an internal mechanism experiment; it is not untouched external validation.

## New information, with existing controls retained

The common 240 descriptors already contain A0 statistics and landmarks, Dope input descriptors and model metadata. A0 includes ridge and 1-NN cross-validated scores and linear residual summaries. Existing experiments also used program–state bilinear interactions and neural function/Jacobian summaries. H27 therefore tests **incremental diverse GP performance responses**, rather than claiming the first landmarks, first interaction model or new pairwise ranking loss.

The proposed eight cheap programs cover raw/robust/rank-view ridge, standardized elastic net, two local smoothing scales and two shallow boosting depths. Their exact syntax is prepared, but parser identities, fitted behavior, binary integrity and measured cost are pending. No whole evolutionary campaign, finalist search, CV7 stage, baseline refit or outer-test evaluation is included.

## Feature and target-scaling contract

Every context uses the same fixed five-way probe partition. Fold0-only is a view of the same captured five-fold results; it is never refitted. Both alternatives have eight score coordinates and approximately the same 80% fit fraction for each individual fit.

Use stable source-bound positions in the qualified teacher TRAIN view, bound to the exact context/input/prefix identity. These positions are not proven acquisition CSV row identifiers. Hash the typed identity in a fixed namespace, sort and assign balanced folds; reordering storage must preserve each position's fold. Known grouping/time restrictions remain authoritative. Equal-valued rows retain their distinct positions and multiplicity. Unknown grouping limits must be explicit.

For each probe fold, use only its fitting rows to compute target mean and population standard deviation; use scale1 when that standard deviation is zero. Fit and predict in those normalized target units. Feature transforms also fit only on those fitting rows. This prospective normalization addresses absolute prediction clamps in the existing kernel; the clamps remain active and exact numerical affine invariance is not assumed.

The proposed feature is

`q = 0.5 log((mean((prediction_z − heldout_z)^2) + 1e-12) / (mean(heldout_z^2) + 1e-12))`.

The normalized constant baseline is zero. Lower q is better. Onefold uses q for fold0; fivefold uses the arithmetic mean of the same five q values. Freeze this formula before controls and fits; pooled RMSE, mean RMSE and archived constrained fitness are different quantities. No dispersion coordinates enter the primary eight-dimensional vector.

Schedule every probe/fold for every admitted context and retain all successes, failures, costs and unattempted counts. No winner-only capture, successful-probe subset or performance-based context replacement. Unexpected fit or nonfinite failure makes the feature phase incomplete and refuses neural training under the initial proposed policy. Any different failure-value policy needs a prospective amendment.

## Matched state–probe experiment

Plan a 2×3 factorial: verified permitted Dope neural state versus zero-state, crossed with zero, onefold and fivefold GP fingerprints. Every arm retains the common240 and candidate412 descriptors. Zero-state does not mean all Dope information is absent. The latest source/checkpoint/input identity must be qualified before reusing any state encoding.

Use the same padded dimensions, capacity, candidate interactions, initialization policy, seeds and ordered schedules throughout. Ablated state/fingerprint blocks must be literal zeros **after** fitted scaling. Task-level fingerprints must interact with candidate descriptors: adding only a task bias cannot change within-query ranks. Synthetic controls must demonstrate that cancellation and verify the actual interaction path.

Two primary contrasts are fivefold-minus-zero and fivefold-minus-onefold **within real Dope state**. Zero-state arms and state–probe interaction contrasts are predeclared secondary mechanisms. Three fixed seeds are retained; model/update counts, objective and checkpoint recipe still require the scientific freeze. No post-result seed, checkpoint or score-ensemble selection.

## Evaluation and shortcut checks

Primary endpoint is selected recorded **raw constrained fitness**, which is minimized. Average contexts within each component, weight components equally, and average paired component differences across the three seeds. For two primary contrasts, the proposed Bonferroni familywise97.5% procedure uses a central98.75% interval per contrast and 100,000 paired component bootstrap draws. A favorable established direction requires the adjusted interval's **upper bound below zero**, together with all declared support criteria. An interval including zero or positive harm does not establish lift.

Choose gradient-heldout monitoring components from each neural TRAIN partition before fits. Exclude their ranking labels from gradients, scaler fitting and recipe selection; use fresh matched ranking models. A heldout task's own permitted GP outer-TRAIN X/y may generate its decision-time probe descriptors. It does not authorize using that task's archived ranking outcomes to fit or select the ranker. Evaluate final component-OOF ranking endpoints once after all models close. Preserve actual TRAIN and heldout-monitor curves separately; lower surrogate loss alone is not selection success.

Report selected gate failures, full-slate candidate coverage, fitness regret and secondary CV metrics. Preserve candidate failures and cost severity. Inference receives features, never target-derived candidate masks.

Qualify exact canonical probe/candidate overlap before outcomes. Prefer disjoint support if it can be fixed without performance selection; otherwise retain full slates and predeclare a nonprobe-support endpoint. Direct best-probe selection is a required comparator only where that exact probe has an archived target candidate. Else its recorded-fitness endpoint is unavailable, with honest coverage counts. New probe scores must not be substituted for missing archived targets. Benefits concentrated on exact probe candidates are paid decision-time knowledge, not broad deep generalization.

## Qualification and planned cost

The source exposes public parse→fit→predict APIs that fit supplied TRAIN transforms and heads. A research-owned offline wrapper, complete dependency/features/compiler checks, bounded input validation, private model serialization/replay and independent source review remain pending. The whole campaign command is unsuitable. No product source, CI, runner, serving or existing study files are changed by this plan.

An identity-selected 16-context, distinct-component panel would require640 probe fits if all eight×five execute. Reuse those exact results in a later full build:850×8×5 =34,000 fits total, not34,640. Fold0-only deployment would require6,800 fits. These are planned counts, not observed runtime or completed work. Probe features are shared across the six neural conditions; their generation count is not multiplied by the factorial. Charge fitting, scoring, storage, replay and inference cost explicitly.

There is no compute deadline or forecast cutoff. CPU/RAM/swap/tasks, host headroom, disk, ownership, input-role and reservation guards remain. Root source/resource qualification and a separate numerical grant precede any actual fit. Current counts are zero builds, GP fits, neural fits, optimizer updates, numerical-array reads and API/provider calls.

See [hypothesis.json](hypothesis.json) for the declared contrasts and [source-receipt.json](source-receipt.json) for source-only bindings and limits.

## Primary references

[Pfahringer, Bensusan and Giraud-Carrier (ICML2000)](https://ml.cms.waikato.ac.nz/publications/2000/landmarking_icml2000.pdf) motivate cheap learners with diverse biases as task descriptors and distinguish heldout performance from training performance. This supports the mechanism, not a lift claim here.

[Leite, Brazdil and Vanschoren (PlanLearn2012)](https://ceur-ws.org/Vol-950/planlearn2012_submission_5.pdf) study relative performance and active testing. H27 freezes its initial basket; adaptive probe selection would be a separate costed experiment. Official [leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html) supports fitting transformations on fitting rows and applying them unchanged to heldout rows.
