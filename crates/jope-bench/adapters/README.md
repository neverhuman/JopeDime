# External estimator adapter boundary

These Python adapters bridge the benchmark's typed request to the installed
XGBoost, LightGBM, CatBoost and scikit-learn estimators. They fit preprocessing
and models on the supplied training rows and labels. Prediction rows contain
features only. They do not own product admission, hidden evaluation labels,
benchmark scoring, dataset registries or shared database mutation.

## Input and allocation limits

The panel-v2 CLI accepts at most 128 MiB of JSON request bytes. It rejects duplicate
object fields, nonfinite JSON numbers and unknown contract fields. Raw feature
cells and target labels retain their existing string/null schema; category strings
such as `NA`, `null` and the empty string remain distinct from a missing value.

A request has at most 4096 columns and one million rows in each split. The combined
float64 training/prediction feature matrices must fit 16,777,216 cells (128 MiB).
The prediction probability matrix has the same separate cell ceiling. Checks run
before model construction or dense allocation; one-hot expansion is included in
the feature budget. These bounds do not bound estimator-internal memory, native
library threads or fit time. The benchmark caller must still apply its declared
process and resource limits.

## Private fitted artifacts remain a trusted-code boundary

`jope.panel-fitted-python/v1` retains the complete estimator and preprocessing in
Python pickle so that reload reproduces the fitted predictor. The CLI verifies
reload only for bytes it has just created itself, then writes the artifact and
result with mode 0600 and exclusive creation. It refuses existing output paths.

**Pickle can execute code. SHA-256 checks integrity, not authenticity.** The internal
replay helper must receive its digest from trusted custody and the caller must trust
the artifact's producer. Supplying an arbitrary pickle together with its own
hash does not make it safe. There is no public/untrusted artifact import endpoint,
and this function is not a deserialization sandbox. The closed same-invocation
flow is recorded in `docs/release/jankurai-pysqlpickle-exceptions-20261102.md`.
That exception does not cover arbitrary artifacts or make untrusted pickle safe.

Removing this trusted-code requirement needs an explicit format migration: native
model formats for all four engines, complete preprocessing/state/version schemas,
and the twelve engine/task replay equivalence cases. An allowlist of package name
prefixes does not establish safe object construction. Until that migration is
qualified, do not import arbitrary fitted artifacts into this process.

## Verification

Run `python -m pytest crates/jope-bench/adapters/test_panel.py -q` in the pinned
benchmark environment. The full suite requires all four external estimators and
exercises all twelve regression/binary/multiclass fit/reload combinations on six
artificial training rows. Missing required engines are failures in that lane.

For a lightweight boundary check with NumPy/pytest only, use
`python -m pytest crates/jope-bench/adapters/test_panel.py -q -k 'not all_external_engines'`.
That deliberate subset does not qualify the four real engines. Tests cover invalid
and valid input, refusal before construction/allocation, exact-limit acceptance,
private immutable output, train-only preprocessing and SHA-bound reload. CLI errors
fail with status 1; missing dependencies use status 69. The final exception handler
reports a failed adapter call; it does not retry with another estimator or report
success after failure.
