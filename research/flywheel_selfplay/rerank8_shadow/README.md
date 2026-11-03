# RERANK8_CR_VETO read-only shadow

This is preparation for shadow inference on completed flywheel rounds. It has no
promotion hook, trainer, optimizer, DOPE invocation, or writer-control operation.
The writer checkout, PID 1402163, its run output and custody are read-only inputs.
Run with nice 10, OMP_NUM_THREADS=4, OPENBLAS_NUM_THREADS=1, MKL_NUM_THREADS=1,
PYTHONDONTWRITEBYTECODE=1 and CUDA_VISIBLE_DEVICES=''.

`prepare` creates an immutable, hash-bound packet of H273's 91 Dev contexts,
r882's stable top-eight pool candidates, and 568 outcome-blind features. It
checks the H273 context/prior cache hashes, original pool hashes, disjoint
TRAIN/Dev families, and exact pool-array equality with the live Dev corpus.
Scoring utilities are separate from feature inputs. Only DL-Val records with
approved `train`/`inner` source folds are admitted; no sealed/outer/meta-test data.
The three feature/selection functions are exact AST copies of H273 source
29fdbd8be15e2d2aa4ddf7b33e96f637f11dbd85 (runner SHA
24606a84b2402c59eebb40ebed6e09ee4b8a51f8d43c27779ec3b25a8b0b15f2).

`watch` reads each subsequent completed round and compares its **incumbent** picks,
not its rejected adapter candidate, with each of the 45 frozen H273 veto cells.
It records all cell picks, the median of family-mean utility deltas, and
family-equal two-contender Elo lift anchored at 1500, with paired family-bootstrap
95% intervals (10,000 draws, seed 2030). Utility tie tolerance is 1e-9, and a
symmetric 1e-3 pseudocount matches the two-contender H273 MM limit. It does not
choose a winning cell or use speed as an acceptance criterion. These repeated
Dev observations are not independent test or serving-promotion evidence.

## Current prerequisite failure

H273/attempt-001's version-pinned S3 inventory includes metrics, utility panels
and fit-selection receipts, **no serialized LightGBM rankers or veto classifiers**.
The study runner retained those objects in process memory, then exited without
exporting them. Receipts, hyperparameters and outcome panels are not frozen
models. The shadow currently emits `BLOCKED_MISSING_FROZEN_MODEL`, null picks and
metrics, and zero scored contexts. It never trains a replacement, derives picks
from outcomes, or silently substitutes r882 as the shadow pick.

A model owner must export the original H273 frozen trees (or separately obtain
approval and provenance for a reproducible reconstruction). Supply a JSON
manifest with schema `jope.h273.frozen-shadow-models/v1`, H273_source_sha, arm
RERANK8_CR_VETO, fit_split DL-Train, dl_val_reads_before_lock 0,
fit_selections_lock_sha256, training_family_ids, and exactly the 45 `cells` from
fold_seed 273140..273142 × seed 273101..273103 × assess_fold 0..4. Each cell
contains selected_n_estimators, strict_margin and selected_tau from its pinned
fit-lock receipt, plus `ranker` and `veto` objects containing file, sha256,
S3_key and S3_VersionId. Only hash-bound LightGBM text models with 568 features
are loaded; pickle and fitting APIs are absent. Every future scored readout
remains shadow-only. **H276 PASS plus the watcher's decision is still required
for any adoption; this scorer cannot promote anything.**

The observer stops after six hours, 40 completed rounds, writer exit/identity
change, or the 99 MB output ceiling. It does not delete receipts to make room.
Generated packets, models and shadow logs belong outside Git in a separate shadow
directory. No GitHub CI or PR to main is part of this lane.
