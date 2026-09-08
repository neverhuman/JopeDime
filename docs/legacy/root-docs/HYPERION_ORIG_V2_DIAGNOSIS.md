# Latest-model Hyperion under old `.vs.orig` semantics

Date: 2026-08-09 UTC  
Scope: the frozen 20-dataset tabular-regression cohort

## Diagnosis

The old and current artifacts are not two evaluations of the same inference
procedure.

As established in the [old-method forensic receipt](HYPERION_OLD_METHOD_FORENSICS.md),
the installed `.vs.orig` files are a deterministic best-evidence
reconstruction of the January procedure, not a claim that its lost random
state and original bytes were recovered.

The reconstructed old `.vs.orig` lineage does all of the following before the
CatBoost split is scored:

1. removes the target column;
2. concatenates train-feature rows followed by test-feature rows;
3. rank-normalizes every feature over that joint matrix to `(0, 100]`;
4. runs 100 shuffled sweeps, with seed `42 + 777*sweep`;
5. tiles at windows 3/6/9/12 from origins `(0,0)` and `(1,1)`, using sampled
   row/column padding;
6. retains raw `h_real` logits; and
7. averages both shifts, all four scales, and all 100 sweeps before splitting
   the rows back into train and test cells.

This is transductive at the feature level. Test targets never enter Hyperion,
but test feature values affect ranks and encodings. It is appropriate for
reproducing the historical lift experiment, not for claiming a sealed
inductive result.

The current `hyperion_vs` path is materially different. Its CLI default is
three sweeps with windows 4/7/11/15. Sweep zero is the identity; subsequent
sweeps shuffle. It fits 1/99-percentile input bounds on train, encodes train
and test as separate matrices, uses one neutral-padded tiling origin, averages
logits, applies sigmoid, and finally converts every training confidence column
to `(rank + 0.5) / n`. Test confidence is mapped through the corresponding
train-score empirical distribution.

That last transform is especially important. For all 20 inspected `train3.vs`
artifacts, every column sorts to the same deterministic rank grid (maximum
observed deviation `3.0e-8`); column means are effectively `0.5`. The learned
model can change row ordering, but all confidence magnitude and calibration
information is discarded. This explains why a newer cell model can fail to
translate into a better appended CatBoost feature.

## Pinned latest model

Jain's governed default identifies `hyperion_v5_arm_b` as its validated
better-routing model:

- weights SHA-256:
  `88b9e8345e44d3ea609a8286215618a762f14e607fcdada6d27447b1f74668be`;
- manifest SHA-256:
  `ebb745f6a0aa6398b1c3d3c9e26585e0536326b65e2083401391203fa37d21ef`;
- Jain source head:
  `a029dbe5aa7a4da7b6f5aa1c5e93f73e3e207bc6`;
- exact PyTorch architecture source SHA-256:
  `31e095d483013689307da7a191ed54b0e3c1ffafdf66d9779c4fd467a3fd180a`.

The safetensors state has 119 tensors. It fully supplies the row/column
encoder, `cell_mlp`, and `h_real` path used for v2. The only absent model keys
are the unused `h_row` and `h_col` heads, so unlike the historical partial-load
checkpoint, no randomly initialized parameter affects the v2 cell output.

## Frozen v2 procedure

[`reconstruct_latest_hyperion_old_semantics.py`](../scripts/reconstruct_latest_hyperion_old_semantics.py)
combines the old artifact semantics with the pinned latest cell model. The
recipe was fixed before looking at the v2 CatBoost result:

- joint train-then-test features, with labels excluded and non-finite features
  filled with zero;
- old GPU rank-to-100 normalization;
- the latest model's required 1/99 robust uint8 transform and `[-3, 3]` model
  scale;
- old global seed 42, 100-sweep permutation schedule, 3/6/9/12 windows,
  two shifted origins, sampled padding, CUDA AMP, and batch size 4096;
- raw `h_real` logits, with no sigmoid and no percentile postprocessing;
- the old all-100 mean collapse; and
- exclusive `train3.vs.orig.v2` / `test3.vs.orig.v2` creation.

The generation receipt binds the 40 input/output files, full sweep tensors,
collapsed cells, model and source hashes, CUDA environment, dataset order, and
all inference parameters. Installation preflights every source hash and shape,
uses exclusive destination creation, re-hashes every copy, and rolls back files
created by that invocation if any copy fails.

## CatBoost protocol

The matched quick benchmark uses CatBoost regression with 1,000 iterations,
seed 42, and the existing train/test split. For each dataset it fits three
models against the same target and evaluator:

1. raw features;
2. raw features plus old `.vs.orig`; and
3. raw features plus latest-model `.vs.orig.v2`.

The target is always the final CSV column and is excluded from all model
features. Test labels are read only to calculate RMSE and R² after fitting.
