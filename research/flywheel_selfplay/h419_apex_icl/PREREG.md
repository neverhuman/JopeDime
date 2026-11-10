# H419 PREREG — APEX-ICL: our own in-context tabular regressor

**Status.** Part A. Published on `origin/v1/next` before any H419 prediction is computed on a real table. The two
pretraining runs named in §3 were started about 30 minutes before publication; they read no data file (every task is
drawn from a seeded generator), so no outcome on any cohort exists. Part B (§5) is published before the first
real-table prediction.

- **Owner:** `cl_apex` (Claude Fable 5, xbabe2 session 947340e7). Claim: `ns/receipts/H419_claim.txt`.
- **Authority:** owner order and answers, 2026-11-10: every learner must be ours (no external pretrained model);
  DOPE `c51a56d0` stays frozen and a new trainable encoder may sit beside it; the fresh `FIT-ICL` families may join
  training; `CONFIRM` is never trained on; run on the full compute of xbabe0–3.
- **Hard constraints:**
  - No dev91, DL-Val, DL-Holdout or sealed byte is read.
  - Training inputs are the synthetic prior and, from part B on, `FIT-ICL` tables only. The 205 DL-Train families and
    the APEX `DEV`, `FIT-SEL` and `CONFIRM` cohorts are never training inputs, so their stack readouts stay leak-free.
  - No external weights. No stored entrant is rerun. Commits go to `v1/next` only.

## 1. Question
Do the predictions of an in-context regressor trained from scratch earn stack weight beside the GP-era programs and
improve the stored `h416.SXB` recipe? A standalone score is reported but is not the target.

## 2. Model (`icl_model.py`)
One forward pass takes a whole task: context rows with targets and query rows without.

| Part | ICL-S | ICL-M |
| --- | --- | --- |
| Column embedder: 24 learned queries attend over one column's context cells; emits a scale and shift for that column's cell embedding | width 96, 2 blocks | width 128, 2 blocks |
| Row encoder: attention across the features of a row, 4 CLS tokens, random feature-identity codes | width 96, 2 layers | width 128, 3 layers |
| Trunk: context rows self-attend; query rows attend to context rows only | width 256, 8 layers, 4 heads | width 384, 12 layers, 6 heads |
| Head | 1,024 bins | 1,024 bins |
| Parameters | 7,558,624 | 23,659,392 |

- **Output.** 1,024 fixed bins on the context-standardised target (sinh spacing over ±40), two-hot cross-entropy,
  mean readout.
- **Normalisation.** Inside `forward`, from context rows only: features by median and IQR (standard deviation when
  the IQR is 0), clipped at ±8, with a missing indicator; target by mean and standard deviation.
- **Shapes.** One (rows, features, context size) per batch: no padding and no mask anywhere.
- **Self-test.** Perturbing one query row changes that row's output only (maximum change elsewhere below 1e-4).

Differences from the design review in the plan: LayerNorm and standard attention replace RMSNorm and QK-norm; the
feature identity is a random code, not a rotary position; no learning-rate sweep and no Muon sprint were run, so the
GPUs went straight to the long runs.

## 3. Training (`icl_prior.py`, `icl_train.py`)
- **Prior (generated on the GPU):** 35% random MLP causal graphs, 25% Gaussian-process draws by random Fourier
  features (RBF and Matérn-like spectra, sparse relevance, sums, products, linear and periodic parts), 20% random
  oblivious forests, 20% a GP draw combined with a forest draw. Then target warps (35% of tasks), noise (30% of tasks
  noise-free; homo- or heteroscedastic; normal, Student-t or Laplace), feature quantisation, monotone feature warps,
  missing cells, and random affine maps. Features: 50% 3–12, 30% 13–32, 20% 33–100.
- **Recipe:** AdamW (betas 0.9 / 0.98, weight decay 0.01 on matrices), clip 1.0, EMA 0.9995, bf16 autocast, warmup 2%
  of the run, cosine decay over the last 15%. Rows per task grow with the run: 384, then 640 at 25%, 1,024 at 50%,
  1,536 at 85%. Context size is log-uniform from 96 rows.

| Run | Host | Size | Hours | Peak LR | Code stage |
| --- | --- | --- | --- | --- | --- |
| `ICL_S_r2` | xbabe3 RTX 3090 | S | 24 | 3e-4 | `r2` |
| `ICL_M_r2` | xbabe1 RTX 4090 | M | 48 | 2e-4 | `r2` (row-token cap 40,000) |

Each run logs the training loss and, every 20 minutes, the EMA model's loss and u on fixed held-out prior tasks next
to a ridge reference. Checkpoints: latest every 20 minutes, named every 2 hours. Both GPUs hold
`~/jope_plan/qualification.lock`.

- **Engineering rule (not a cohort outcome):** if a run's held-out prior u is not above the ridge reference after
  3 hours, I stop it, fix the prior or the model, and restart under a new run name. Restarts are listed in the
  closeout.

## 4. Tests (fixed)
ICL predictions enter the stored H416 stack as extra candidate columns: 5 out-of-fold passes on the
`deploy_h415.kfold_labels` folds, 5 fold-context AUDIT passes averaged, and one all-SEARCH pass, with the unchanged
`BCAR.mix` weights rule over programs plus ICL columns. AUDIT targets never reach the GPU process.

- **Stage 1, stored matrices (no worker run):** the 201 P201b DL-Train contexts of H416. `SXB + ICL` versus stored
  `SXB`, `h335_stats.primary_test` (family-mean gain clipped at ±0.5 above 0, sign-flip p < 0.025 one-sided,
  leave-one-family-out minimum above 0, median ≥ 0). With zero ICL columns the code must reproduce stored `SXB`
  to 1e-9.
- **Stage 2, APEX-DEV (115 families):** the same comparison on H417's `DEV` matrices. This is the primary test.
- **Reported:** share of contexts where an ICL column has weight; standalone u of the 8-member ICL ensemble against
  `rank_ridge`, `gpr_std` and `hgb2_l15_b5`; results by table width.

**Verdicts.** `ICL_CONFIRMED` if stage 2 passes. `ICL_TRAIN_ONLY` if only stage 1 passes. `ICL_NULL` otherwise.

**Kill rules.** Hour 36: stop and rework if ICL-S is below `rank_ridge` standalone on `FIT-SEL`, or has zero stack
weight in more than 90% of the stored contexts. Hour 72: drop the standalone ambition unless ICL-M beats
`hgb2_l15_b5` standalone on `FIT-SEL` or the stack win fraction over `SXB` is at least 0.52.

## 5. Part B (before the first real-table prediction)
Inference and column code (`icl_infer.py`, `icl_columns.py`), the 8-member ensemble (feature orders × target
transforms), the handling of tables wider than 100 columns, the checkpoint choice rule, the sha256 of every checkpoint
used, and, if a real-derived phase is run, its sampler over `FIT-ICL` with the refusal test for other families.

## 6. Integrity
- Run configs record the code sha256, the torch version and the GPU. Checkpoints used for scored columns are named
  with sha256 in part B and archived.
- Column files are create-once with sha256; the stack readout reads stored columns only.

## 7. Files
| File | Role |
| --- | --- |
| `icl_model.py` | model, normalisation, loss, mean readout |
| `icl_prior.py` | synthetic task prior |
| `icl_train.py` | trainer, held-out prior evaluation, self-test |
| `icl_stress.py` | one training step at the largest shapes (memory and kernel limits) |
