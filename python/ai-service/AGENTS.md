# Python AI/data service boundary

This directory records the reference profile for the repository's existing Python AI/data surface. The
implementation currently remains under `python/src/jope/`; do not move files here as part of policy cleanup.

## Owns

- Python model, training, evaluation, dataset feature and compatibility code under `python/src/jope/`.
- Python unit and contract tests under `python/tests/`.
- The dated Python exception and migration plan in [`../../PYTHON_EXCEPTIONS.md`](../../PYTHON_EXCEPTIONS.md) and [`../../PY_TO_RUST_PLAN.md`](../../PY_TO_RUST_PLAN.md).

## Forbidden

- Durable product truth, authorization decisions, workflow state, release authority, or database ownership.
- Changes to frozen vocabularies, emitted bytes, scientific fixtures or custody snapshots without their owning proof and identity review.
- GPU work in this service profile unless a separate owner-authorized lane explicitly permits it.

## Proof lane

Run `cd python && python3 -m pytest -q tests/test_training_input_closure.py tests/test_policy_training_contract.py`,
then `python3 -m py_compile` on changed Python files. The repository-wide Python lane is `just python-check`.
