# H170 — one-cycle learning rate and inverse momentum

Owner gpu_tokbench_codex; reserved ENG:hwave-loop. Authority: ~/nudges/hwave_t0120_h170_tokbench_codex.md. This protocol, discrete schedules, source and inputs are fixed before any H170 policy forward, real-data optimizer update or outcome readout. Hypothesis: a one-cycle LR/momentum policy improves r882 continuation at a fixed 40-update budget. Concept references: [Smith and Topin](https://arxiv.org/abs/1708.07120) and [Smith](https://arxiv.org/abs/1803.09820). This study measures regression utility and Elo; no speed metrics or serving change.

## Fixed optimizer settings and discrete endpoints

All four arms start from an identical read-only private r882 copy, fresh native PyTorch AdamW state, weight_decay=1e-4, beta2=.999, global gradient clip L2=1, and the inherited H146/H156 loss/batches. Exactly 47 trainable adapter tensors / 29,211 parameters; all DOPE and non-adapter state stays bitwise frozen. Native AdamW changes beta1 through the parameter group, including its ordinary current-beta bias correction; there is no custom product-of-beta correction.

Applied update t is an integer 1..40. For t<=12, q=(t-1)/11, lr=1e-5+(peak-1e-5)q and beta1=.95-.10q. Thus update1 uses the initial value and update12 uses the peak. For t=13..40, q=(t-12)/28, c=(1+cos(pi*q))/2, lr=1e-6+(peak-1e-6)c and beta1=.95-.10c. Update13 is below the peak, and update40 uses lr 1e-6/beta1 .95. No duplicated peak or scheduler.step off-by-one. Settings are applied before every native optimizer step and actual group lr/betas are journaled.

| Arm | LR | beta1 |
| --- | --- | --- |
| CTRL | constant 1e-4 | .9 |
| OC3 | above cycle, peak 3e-4 | .95 -> .85 -> .95 |
| OC15 | same endpoints/shape, peak 1.5e-4 | .95 -> .85 -> .95 |
| CTRL_MEANMATCH | constant .0001481125 | .9 |

The discrete OC3 sum is .0059245, mean .0001481125; CTRL_MEANMATCH has the same 40-step LR sum. OC15 mean is .0000749875. The full applied table is committed in schedules.json and verified against the function before construction. The matched control tests the complete cyclic LR/momentum policy at matched mean LR, rather than isolating LR shape from the momentum change; it does not promise identical AdamW trajectories or exact multiplicative decay products.

The six historically inactive adapter tensors may have p.grad=None. None contributes zero to diagnostic norms, finite checks and whole-adapter drift; clip_grad_norm_ already ignores it. Preserve native AdamW's None behavior for inactive parameters, rather than adding synthetic gradients/weight decay updates. Record actual None tensor names/counts/elements, including any deviation from the expected six. Drift at step40 is float64 L2 over all 29,211 coordinates, including unchanged inactive tensors.

## Data, frozen-state and native gates

Use the admitted 1,219 regression TRAIN contexts / 205 families and historical 91-context DL-Val / 19 disjoint families. All 276 IDs must join origin=real in the actual pinned S3 catalog; hash the authoritative local catalog and independently rehash 1,319 numeric input members against the versioned recovery closure. No synthetic study-training datasets, sealed/holdout/meta-test payloads or live flywheel/serving files. Original H50/H58 derivative S3 versions are not established; exact numeric recovery is retained in the versioned H144 archive and runtime sources are separately pinned. Retained prepared real-S3-derived inputs are used, not fresh raw-dataset re-ingestion.

Before any TRAIN outcome-pool opens or update, the first policy readout must match all 91 native r882 reference utilities exactly, including mean 1.1377941898164525. Frozen-anchor cached forward must also match every native cost bit at step0. Use the proven native adapter_10tok.predict_costs API from H165, not an unverified wrapper. Pin private checkpoint SHA0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64 and DOPE composite c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814. Check DOPE and every non-adapter byte before and after each update/evaluation. Constructor/optimizer smoke performs zero policy forwards, scored contexts and updates.

## Cross-fit, selection and outcomes

Seeds 170101–170103; family-fold seed170140; bootstrap170141. Permute sorted 205 families into five groups of 41. Assessment A, selector B=(A+1)%5, FIT other 123 families. Each cell fits only FIT, then uses B to select among snapshots 0/10/20/30/40. Assessment A and DL-Val do not fit, select or calibrate. Each TRAIN context is assessed once per seed. H156 sampler: 8 FIT contexts per update, 24 candidates each, four lowest raw-fitness reveals plus 20 sampled without replacement. Paired batches across all arms. Native full-pool storage-order first-four evaluation reveals stay unchanged; the inherited training/evaluation support/reveal mismatch is not a treatment in H170.

Pick the earliest maximum equal-family B mean, treating differences<=1e-12 as ties. Bootstrap that single best step versus r882 with 10,000 paired family draws. Accept only gain > .001+1e-12 AND 95% LCB > 0; otherwise step 0. Do not try a second-best eligible step after rejection. Persist/fsync choices, identifiers, plan/schedule SHA and drift before any trained-policy A/Val evaluation. Descriptive unconstrained best uses this same B-best step including 0, without eligibility gates; lock it before A/Val and never use it to advance. Retain every selector panel, including negative/flat snapshots. Four arms x 3 seeds x 5 folds x 40 updates = 60 trajectories / 2,400 updates.

Primary: equal-family OOF utility gain per seed, then median of the three scalar seed gains. Use 10,000 shared family-bootstrap draws, median of resampled per-seed means in each draw, linear percentile 95% CI. One-sided bootstrap tails include ties; Holm across OC3/OC15/CTRL_MEANMATCH versus r882. CTRL is a descriptive reference; paired treatment-vs-CTRL contrasts are descriptive. Advance only if median > .001+1e-12, CI LCB > 0 and Holm p<.05; otherwise NO_ADVANCE. Canonical H148 Bradley–Terry MM Elo: context-pairwise wins averaged equally by family, tie margin 1e-4, pseudo-win prior .01, 400 iterations, r882 = 1500; median seed Elo and 1,000 shared-family draws for Elo CI.

Secondary: each seed's 91-context panel averages its five selected policies' context utilities before 19-family inference; do not count 455 independent observations. Report the same primary/unconstrained summaries and step 40 drift per arm/seed/fold plus median/range over 15 trajectories.

Pseudoreplication checks are mandatory disclosures: exact utility-vector groups on a matched ordered cohort across seeds/arms for strict/unconstrained OOF and DL-Val, per-outer/per-snapshot selector vectors, step40 adapter SHA groups, and selected-adapter SHA groups. Normalize signed zero in utility identity, reject nonfinite vectors, and distinguish vector equality from equality of means. Report if all arms' selectors equal CTRL. Step0/fallback duplication is expected; distinct parameter hashes do not establish independent prediction replication.

Interpretation is conditional continuation on historically exposed TRAIN/r882 and a repeatedly exposed DL-Val panel. The family bootstrap conditions on the trained/selected policies and does not retrain during each draw; it is not calibrated full uncertainty for overlapping fitted policies. A fallback panel cannot establish that every nonzero trajectory harms or that the selector has calibrated power; the planted +.005 fixture tests only mechanical reachability.

## Resources, launch and custody

CPU only xbabe2, nice 19, one math/Torch thread, aggregate supervisor+worker tree<=4 threads, RSS<6,000,000,000B. Terminal stop if load1>0.85 times host online CPU count (SC_NPROCESSORS_ONLN, independent of task affinity), host CPU>85%, MemAvailable<16GiB, free disk<40GB or user-visible used>=95%. Samples every .5s; 12-hour administrative limit. A guard/fault yields INCOMPLETE with saved diagnostics, never fabricated zeros. One exclusive output and one attempt marker; no automatic scientific retries or cell reuse.

External launch binds manifest/PREREG/fold SHAs, positive real PR/review IDs, merged source HEAD and exact reviewed source verification. Require that binding before construction; read-only allowlist includes exactly launch.json and deliberately rereads it after the access hook. Native model workers have no API credential action or S3 uploads. No GPU jobs/qualification lock,xbabe0/xbabe3,foreign signals,live-loop writes or serving promotion.

Flow: read-only cross-review by gpu_fused_redteam_codex with <=15-minute wait, targeted local checks, author squash/admin merge as explicitly authorized, merged-byte check, then one guarded wave. S3 create-only/versioned byte-readback stores raw panels, snapshots, update/selection/drift/identity/resource receipts and exact source closure outside Git; independent labels/choices/statistics/drift/schedule and fresh native spot replay before small-results closeout. Add/update only H170's index row, preserve every foreign row, squash-merge and verify actual merged files. Preserve/independently restore unique lane work before git worktree remove. H-wave lanes are cap-exempt under the owner's lines35/73 decision.
