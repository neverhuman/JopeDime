# H419 PREREG addendum 1 — column injection, real-table tasks and a real-data monitor

**Status.** Published on `origin/v1/next` before the runs it describes start. No H419 prediction on a test cohort
(stored P201b matrices, APEX-DEV, CONFIRM) exists. One two-minute smoke run (`SMOKE_S_r3a`, discarded) exercised the
new code on the 3090 before publication.

**Reason.** Owner instruction, 2026-11-10 10:45Z: use synthetic data injection into columns to maximise
generalisation during the deep-learning training, and track the metrics that matter.

## 1. Column injection (`icl_inject.inject_columns`)
H317's view regime 3 ("appended synthetic columns"), generated on the GPU. Each injected column is one of seven kinds:
Gaussian or uniform distractor; Student-t3 or Cauchy distractor; noisy collinear duplicate
`a·z_a + (1−a)·z_b + 0.15·e`; bilinear `z_a·z_b` or quadratic `z_a² − 1`; monotone warp (`sign(x)·ln(1+|x|)` or normal
scores of ranks); quantile step (`x > q25` or `x > q75`); random Fourier `sin` / `cos`. The target never depends on an
injected column. Columns are shuffled after injection.
- **Synthetic tasks:** in half of them, 20–60% of the feature slots are injected columns.
- **Real-table tasks:** injected columns fill every feature slot the column crop leaves open.

## 2. Real-table tasks (`icl_inject.RealBank`)
- **Source:** the FIT-ICL cohort only. `icl_bank.py fit` builds the bank inside the sandbox with the frozen readers
  and refuses any other cohort; the trainer refuses a bank that names another cohort. Bank: 419 tables (21 of the
  440 are too small), sha256 `f6e283ce…`, at most 4,096 rows and 100 feature columns per table.
- **Task:** a row crop of one table, a column crop in random order, injected columns, and one of three targets:
  the table's own target (50%), another real column (25%), or a synthetic teacher function of the real features
  (25%). Real targets get 1.5–3% Huberised noise in 30% of tasks (H317 regime 4).
- **Share:** real-table batches ramp from 0 to 40% of batches over the first hour. Synthetic tasks stay ≥ 60%.
- **Consequence:** FIT-ICL families are in-sample for this model. No ICL stack readout is made on them. DL-Train,
  FIT-SEL, DEV and CONFIRM tables are never training inputs, as in part A.

## 3. Runs
The part-A runs `ICL_S_r2` and `ICL_M_r2` (synthetic only) are stopped after about 1.5 hours. Their last checkpoints
initialise the new runs (raw and EMA weights; fresh optimizer and schedule; checkpoint sha256 in each run config).

| Run | Host | Size | Hours | Peak LR | Starts from |
| --- | --- | --- | --- | --- | --- |
| `ICL_S_r3` | xbabe3 RTX 3090 | S | 22 | 3e-4 | `ICL_S_r2/ckpt_last.pt` |
| `ICL_M_r3` | xbabe1 RTX 4090 | M | 46 | 2e-4 | `ICL_M_r2/ckpt_last.pt` |

## 4. Real-data monitor (`icl_inject.RealVal`)
Every 20 minutes the EMA model is scored on the 106 core FIT-SEL contexts, **SEARCH rows only**: query = fold 0 of
`deploy_h415.kfold_labels`, context = folds 1–4, one context per forward pass. No AUDIT row leaves the sandbox
(`icl_bank.py val`, sha256 `8e517921…`). Logged: family-mean and median u, a ridge reference on the same rows, the
win rate against it, and the same on tables of at most 100 columns. Also logged: u on held-out synthetic tasks with
injected columns.
- **Use:** engineering decisions and checkpoint choice. FIT-SEL is a design cohort; DEV and CONFIRM are not touched.
- **On-track rule:** by hour 12 of a run the win rate against ridge on the monitor is at least 0.55 and the median u
  is above its value at the start of the run. Otherwise I stop the run and rework the prior.
- **Reference at the start (S after 1.3 h of synthetic-only training):** mean u 0.78 against 1.06 for ridge, win rate
  0.46, median u 0.59. Ridge is exact on some near-deterministic tables, so the mean is not the target; the win rate
  and the median are.

## 5. Unchanged
Model, output head, tests, verdicts and kill rules of part A. Part B (inference and stack columns) is still to come
before the first prediction on a test cohort.

## 6. Files added
`icl_inject.py` (injection, real-table tasks, monitor), `icl_bank.py` (bank builders, sandbox only); `icl_train.py`
gains `--init`, `--real-bank`, `--real-frac`, `--inject-p`, `--val-real`.
