# H206 native-search / student learning

**Model/study source implemented; scientific run NOT STARTED; actual CUDA/data qualification PENDING.**

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
The CPU round interface still refuses CUDA models. A separate admission-gated
GPU phase controller/study now exists; real scientific qualification is pending. No verdict, positive
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


## Actual factory and phased study implementation

model.py invokes the original hash-pinned FusedJopeNetwork.__init__, reads
the four exact-version parameter assets, strictly loads r882 and checks actual
full/anchor/DOPE tensor-map hashes. The CPU construction probe matched all three
maps and exposed exactly556,809 intended trainable DOPE+residual parameters,
including112,641 residual parameters. It made zero optimizer steps and zero
scored/candidate/GPU forwards. Final source-bound receipts are recorded outside
Git; construction does not establish native91 or CUDA qualification.

qualify_historical_native91 uses the approved historical H50 91-DL-Val roster,
prepared pools and tokens. native91_inputs.json pins actual exact-version GET
copies of those approved artifacts and the original a5109aff full-cost reference.
The source calls the original frozen r882 scorer one context at a time and
requires every full float32 vector, ordered pick, utility and exact mean to
match. Failed vectors are persisted before stopping. This is the mandated
baseline compatibility gate, never H206 FIT data admission or new Rust-CV
label acquisition. No historical outer/sealed table is opened.

study.py implements the fixed60 cells, three rounds, separate GPU proposal /
CPU native-teacher / GPU600-update phases, terminal B pooling, the private
unchanged H177 selector with1800-step metadata and H178 calibration, a durable
checkpoint-bound B lock, all raw A panels and selected-arm A diagnostics.
There is no A-driven reselection or sealed read. supervisor.py admits only
actual merged, clean source with real source/study-bound model/native91/input/
queue receipts and reconciled original accounting. It owns a single canonical
attempt, shared GPU lease, original deadlines, per-round6h and per-fit300s caps,
actual child/numeric/resource checks and terminal signals. GPU processes exit
between phases; no GPU is held by the native-label worker.

Every update's loss, gradient sums, intended before/after parameter and state
hashes, forward-change hashes and actual update/example count are fsynced
locally and exact-VersionId persisted remotely before the next update.
Completed/partial checkpoints and terminal failures retain available proofs;
S3 outages do not erase the local proof or partial model. Native scratch/cache
duplicates are reclaimed only after a deduplicated create-only S3 archive is
independently restored and every member hash checked. No foreign file or
unverified cache entry is eligible.

The conditional full-study entrypoint is:

~~~bash
nice -n 19 env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 -B -m research.flywheel_selfplay.h206_search_distill.study --study /path/to/approved-study.json --task-sha256 EXACT_CONFIG_SHA256
~~~

That command is a protocol entrypoint, not a ready launch configuration.
A missing genuine qualification, wrong merged source, unknown canonical budget,
different population, occupied GPU or resource breach refuses/terminates the
entire attempt. The existing CPU run_round CUDA refusal remains unchanged.
Actual CUDA execution and scientific study qualification remain NOT_RUN.

Actual CPU qualification at code15f365007f7c978af262184e108edc05cc09779d
passed original constructor/state/trainability and all91 exact full baseline
vectors/picks/utilities (mean1.1377941898164525). Vector records have independent
exact-version S3 readback. The original scorer makes lazy imports during forward;
factory.runtime preserves the verified private source environment through each
qualification/model phase and restores ambient modules afterward. No scientific
optimizer update or GPU was used. Receipt details are in MISSING_PINS.md;
actual CUDA qualification, H205 input admission and accounting remain held.
