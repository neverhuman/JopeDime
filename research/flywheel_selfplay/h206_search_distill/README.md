# H206 native-search / student learning

**CPU interfaces implemented; scientific run NOT STARTED; execution qualification PENDING.**

The source now includes:

- contracts.py: original TRAIN ObjectRefs, exact-version body/tensor hashes,
  immutable per-cell roles and transitive protected-family closure.
- native.py: real inherited GP proposal generation, source/binary/environment
  bound --fit-s1 subprocess calls, full-identity outcome cache, and create-only
  S3 objects with exact-version GET verification.
- student.py: fixed width128 residual adapter, a frozen r882 anchor, live
  inherited DOPE/SVD tokens, batch32 gradient accumulation and actual per-round
  parameter/gradient/forward-change proofs.
- executor.py: charged initialization, persisted native teacher consumption,
  three linked expert rounds, verified model+optimizer resume, terminal-only
  B scoring, durable cost panels and cumulative cost reservations.

The initial architecture/initialization and TRAIN-only S1 bridge amendment are
in [PREREG.md](PREREG.md). All unresolved model/data/source/runtime bindings and
their owners are listed in [MISSING_PINS.md](MISSING_PINS.md).

Data-free integration tests execute the original inherited token methods (asset
constructor omitted), controlled native subprocesses, real AdamW updates and
S3 protocol fixtures. Three fixture rounds perform9 updates/288 examples and
require DOPE+residual changes in EACH round. A prior successful round cannot
hide a later no-op optimizer. These are source integration proofs, not real
r882/native-label/learning results.

A separate bounded CPU check uses the real SHA-pinned S1 binary on generated
data. The initial undersized fixture was refused and retained. At admitted
shape, scientific CV fields are exact under a change of TRAIN-derived query
rows; raw timing fields naturally differ. No numerical tolerance, real dataset,
model payload or GPU was used. Receipts and costs remain outside Git under
evidence/H206/executor-20261102/native-protocol/.

Run launch.sh --plan to inspect the queued metadata. --execute remains refused.
The CPU round interface also refuses CUDA models: the GPU phase controller and
full study/qualification bindings have not been supplied. No verdict, positive
lift, completed scientific endpoint or serving promotion is claimed.

## Recovered custody and reproducible source checks

[bindings.json](bindings.json) binds the actual r882 checkpoint/assets, native
binary, native91 reference and canonical-ledger registration. Every S3 identity
has an exact-version GET/SHA receipt outside Git under
`/home/ubuntu/jope_plan/evidence/H206/pins-20261102T0835Z/`.
The ledger is UNRECONCILED and blocks reservations; it does not grant a new
budget. [MISSING_PINS.md](MISSING_PINS.md) identifies remaining artifacts/owners.

Run these from the repository root. The integration deliberately requires CUDA
hidden, one numeric thread per library, and an owned pytest directory outside
the checkout and `/tmp`. Pytest owns the selected scratch directory.

```bash
mkdir -p /home/ubuntu/jope_plan/evidence/H206
H206_TEST_TMP=$(mktemp -d /home/ubuntu/jope_plan/evidence/H206/pytest-review-XXXXXX)
nice -n 19 env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=python/src:. python3 -B -m pytest -q -p no:cacheprovider --basetemp="$H206_TEST_TMP" research/flywheel_selfplay/h206_search_distill/test_h206.py
nice -n 19 ruff check --config python/pyproject.toml research/flywheel_selfplay/h206_search_distill
nice -n 19 ruff format --check --config python/pyproject.toml research/flywheel_selfplay/h206_search_distill
```

The explicit Python Ruff configuration uses the repository line-length/style;
running Ruff with its unrelated root defaults is a different check.
