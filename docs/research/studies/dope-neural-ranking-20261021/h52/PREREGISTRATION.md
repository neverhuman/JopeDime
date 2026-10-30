# H52 preregistration draft — objective alignment + anchored updates

Drafted 2026-10-30T14:58:48Z; amended 2026-10-30T15:30Z. **Do not launch H52 training until H51 is fully closed.** This is a planning artifact for later transfer into a dedicated H52 repository branch after H51's one closeout PR. The CPU-only diagnostics below may run while H51 holds the GPU; they do not launch H52 training.

## Question

H51 tests whether larger AdamW steps move away from the discrete argmin utility plateau. H52 tests whether the training objective better matches that utility and whether anchoring can preserve r882's selected programs while allowing useful movement.

### Inner-CV baseline leakage and paired comparison amendment

r882 and r785 were trained in the live loop using all 1,219 DL-Train contexts, including the families later assigned to each held-out inner fold. Therefore candidate-minus-r882 fold gains are descriptive continuity metrics, not independent held-out estimates; the base has already fit the evaluation families. Every H52 arm must also be compared per family with a regression-loss CONTROL trained on the same r882 initialization, fold, seed, update count, learning rate, and minibatch order. Keep both paired-control and r882-relative vectors and summaries. The live-sized CONTROL remains k=2/lr=1e-4; run matched control references at k=16/lr=1e-4 and k=16/lr=3e-4 for the corresponding arms. These are baselines, not new objective arms. A CONTROL-vs-CONTROL different-seed sanity comparison should have family median 0 and bootstrap CI [0,0] under the step-like utility regime; report the observed result rather than assuming it.

## Frozen source and scope

- Base: read-only offline `~/jope-data/h51/ckpt/round-882.pt`, SHA256 `0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64`; verify round 882, 29,211 adapter parameters, and DOPE SHA256 `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814` before every arm.
- Inputs: `~/jope-data/h50/input`, manifest identity `3eb4a62acea512f528500b043299a5968e1c9b1d1b1759bbf77a426b9fa1ecc3`, file SHA256 `99a53569fcaf59f9309e69329c830ae1684adff382da3ae22e0b2645c4ebc4d4`. Use eligible real S3 regression DL-Train only, 1,219 contexts / 205 families; five fixed whole-family folds, seed 4900. Fresh seeds 12, 13, 14.
- DOPE remains frozen. No DL-Val terminal reads before admission; no sealed, holdout, outer-test, live-loop, xbabe0, or xbabe3 reads. No checkpoint/large files in Git. Preserve run outputs outside Git under `~/jope-data/h52/`.

## Step 0: CPU diagnostic (before arms)

On r882 and the five held-out DL-Train folds, use full-pool predictions to record:

1. Fraction of held-out contexts whose predicted argmin is gate-failed (`gate=false`, realized utility `-2.0`).
2. Mean oracle utility gap split into (a) selected gate failures and (b) selected passing but suboptimal programs. Define the per-context passing oracle as the minimum fitness among gate-passing programs; utility is `-fitness` for a passing selection and `-2.0` for a failed selection. Report counts and means per group, plus no-passing-program contexts separately.
3. For each CONTROL seed 12–14, fraction of the 205 families with at least one held-out context whose selected program SHA flips relative to r882; report per-seed values and the pooled family-cluster count.

Commit only a small JSON receipt containing source hashes, fold identity, counts, means, and summary fractions. This diagnostic reads only DL-Train payloads and does not score or select on DL-Val.

## Arms and fixed training schedule

All arms start from r882, evaluate the same five inner folds, use family-balanced minibatches, and train only the adapter. A completed arm/seed produces 205 per-family gains versus the same-fold r882 baseline. Use AdamW weight decay `1e-4`; keep its state persistent for each fold.

1. **CONTROL** — `k=2`, `lr=1e-4`, the live-loop-sized update budget. Family-balanced batches with seed-specific order.
2. **ANCHOR** — `k=16`, `lr=3e-4`, lambda `(1, 10)`. Objective is H48 base loss plus `lambda * (0.5 / 29,211) * ||theta - theta_r882||_2^2`, with the penalty computed over trainable adapter parameters only. Save per-step drift L2.
3. **FOCUS** — `k=16`, `lr=1e-4`, tau `(0.1, 0.5)`. Per-context program weights are `softmax(-true_cost/tau)`, where `true_cost=fitness` for gate-pass and `true_cost=2.0` for gate-fail. Standardize prediction and target within context using target median/MAD (MAD floor 0.1), then compute weighted Huber loss. Two-step linear warmup followed by cosine decay across the 16 updates.
4. **GATE** — same update count, learning rate, warmup, and cosine schedule as FOCUS. Add a per-context hinge term `relu(0.5 + pred(best_passing) - pred(gate_failed))` for every gate-failed program, using lower-is-better predicted cost. Contexts with no passing program are reported separately and contribute no gate-pair hinge.
5. **NULL** — exact FOCUS recipe, but permute target vectors only among contexts in the same family using a seeded, recorded permutation. Singleton families remain unchanged and their count is disclosed.

Keep seeds 12–14 genuinely distinct: record adapter-weight SHA256s and whether all three per-family inner utility vectors differ. Identical utilities indicate step-like behavior, not seed agreement. Report both family-median and family-mean gains; the mean is secondary and never admits a candidate.

## Admission and terminal scoring

For each arm/configuration, form paired per-family gains against its matched CONTROL. Pool the 615 paired family gains (205 family IDs × 3 seeds). Bootstrap 10,000 times by sampling 205 family IDs with replacement and keeping all three seed rows for each sampled family. Admit only if pooled paired family median is strictly `>0.001` and the cluster-bootstrap 95% lower bound is strictly `>0.001`; ties fail. Also report per-seed paired bounds and candidate-minus-r882 continuity results. A paired admission only authorizes the single terminal DL-Val read for that selected candidate; the terminal result determines whether it beats r882 and r785. Do not claim an r882 win from inner-CV evidence. Report Elo versus both r785 and r882; Elo/AUC are not the admission metric. With no paired admission, terminal reads are zero and Elo is N/A.

## Compute guards and execution order

Serialize arms. Hold `~/jope_plan/qualification.lock` for GPU use. MemAvailable >=20 GiB, CPU <=85%, GPU free >=20 GiB, process RSS <6 GiB; `nice -n 15`, OMP/MKL <=8 and OpenBLAS=1. Because xbabe2 swap is full, stop new seeds if any guard fails. CPU fallback may be used only after the GPU lock has been busy >20 minutes, with <=8 Torch threads and the same RAM/CPU/RSS guards; report device strata separately. Never interrupt another lock holder. No full `just required`; use targeted checks (py_compile and H52 audit scripts) only.

## Reporting and stop rule

Record source identities, device/guards, per-step update L2, per-family candidate summaries, point median, clustered bootstrap LB/UB, `n_families`, admission, terminal-read status, and Elo. No additional arm is authorized by this protocol. Conclusions are limited to these seeds, settings, and eligible regression data.

## CPU-only H52 diagnostics allowed before H51 closes

Run Step 0 on CPU with nice 15, OMP/MKL <=8, and no GPU lock contention: report the gate-failed argmin fraction, nonnegative oracle utility shortfall split by selected gate failure versus suboptimal passing choice (and no-passing contexts separately), and family-level CONTROL argmin flip fractions using the completed H51 CONTROL seeds 9–11. This reads DL-Train only. Separately, if an offline copy already exists, bootstrap 91-context DL-Val utility differences between r882 and the last 50 live rejected candidates to quantify incumbent selection bias. Use only offline copied artifacts, never inspect or open the live xbabe1 run directory, and do not train or select a candidate in this diagnostic. If no offline copies exist, record unavailability rather than searching live state.

Selection-side diagnostic availability check (2026-10-30): searched only offline H50 closeout/run custody and H51 run custody for copied rejected-candidate checkpoints or a last-50 candidate package. None was present; H50 custody has aggregate JSON/run receipts only, and H51 custody contains its own offline experiment artifacts. The requested last-50 bootstrap is therefore unavailable and is skipped. No xbabe1/live run directory was searched or opened.

Step 0 execution receipt (2026-10-30T15:57Z): CPU-only diagnostic completed from offline r882 and H51 CONTROL seeds9–11. Receipt: `~/jope-data/h52/runs/step0-diagnostic.json` (manifest `3eb4a62…`, checkpoint `0084479a…`, DOPE `c51a56d0…`, 1,219 train contexts). r882 argmin gate failures: 17/1,219 (0.013946); passing argmins 1,202; no-passing contexts 0. Mean oracle utility shortfall: 2.925913 for selected gate failures and 0.448226 for suboptimal passing selections. CONTROL argmin flips in 39/205 families (seed9, 0.190244), 38/205 (seed10, 0.185366), and 41/205 (seed11, 0.200000). Validation not scored; excluded payloads opened 0; outer rows delivered 0. CPU run used at most 8 threads, nice 15, no GPU lock or GPU access.

H52 implementation amendment: set Python/NumPy/Torch RNG seeds from the arm seed before model setup so matched CONTROL/objective runs use the same sampled training slates as well as identical family-context batch seeds. Candidate output must persist r882-relative and paired CONTROL family vectors; only the paired pooled median/LB95 gates admission. H52 controls at k16/lr1e-4 and k16/lr3e-4 provide matched references for FOCUS/GATE/NULL and ANCHOR, respectively. The r882-relative comparison is continuity reporting; the terminal read alone determines improvement over r882/r785.

## Authorized execution amendment — 2026-10-30T16:25Z

H51 PR #547 is closed and squash-merged at `8d9f2fa2a05008b544e2aa8a925b67dc0e5a7055`. H52 is authorized to begin with seeds 12, 13, and 14. The source lane is based on that exact H51 merge commit as instructed; the host's `origin/main` advanced after the nudge was written. The existing merged H51 lane is reused because the repository is over the eight-lane limit and the required `lanes sweep --apply` found no removable lanes.

Run arms serially, releasing `qualification.lock` between cells, in this order:

1. Matched regression CONTROL references, all seeds: `(k=2, lr=1e-4)`, `(k=16, lr=1e-4)`, `(k=16, lr=3e-4)`.
2. FOCUS `tau=0.1` and `tau=0.5`, seeds 12–14.
3. GATE, seeds 12–14.
4. ANCHOR `lambda=1` and `lambda=10`, seeds 12–14.
5. NULL, seeds 12–14.

Keep the admission rule unchanged. Report every seed and the pooled 615-row result for each objective configuration. There is no best-of-seed selection: if a configuration meets the pooled admission rule, the one authorized terminal DL-Val read uses seed 12 as the preregistered representative, regardless of the individual seed ranking. Otherwise terminal reads are zero and Elo is N/A.

The median gate may have low power when family utility changes are sparse: H51 saw 94.1–96.1% exact-zero paired family deltas for DATA100 and about 19% of families with any CONTROL argmin flip. Keep median and LB95 admission as preregistered, but for every seed/cell and pooled configuration also report the diagnostic-only nonzero-family fraction, positive/negative/zero family counts, two-sided exact sign-test p-value, and family-cluster-bootstrap 95% CI for the paired family mean. These diagnostics never alter admission.

The paired CONTROL removes the shared r882/fold offset but does not remove the fact that both arms start from r882 trained on the held-out families. State this residual leakage in the final README. Propose H53 as a fold-clean baseline, retraining each fold's adapter from the pre-r882 initialization on out-of-fold families only; do not run H53 here.

For matched-slates evidence, persist a SHA256 for the ordered context IDs and sampled program indices at every fold/step. After CONTROL and the first FOCUS `tau=0.1` cell for seed 12 complete, compare their per-step slate hashes and require exact equality before starting the next cell. NumPy and Torch RNGs are seeded before model construction and slate sampling. Report the CONTROL seed 12/13 different-seed paired sanity result as observed; do not assume it is zero.

## H52 launch record — 2026-10-30T16:25Z

The preregistration and implementation were committed before the first H52 training cell. The copied Step 0 receipt is `step0-diagnostic.json` (1,037 bytes); its source checkpoint, DOPE, and input-manifest identities match the values above. No validation payloads or excluded rows were read for Step 0. Full run outputs and model snapshots stay outside Git under `~/jope-data/h52/runs/`.
