# Recovery and learning gate

Run the native status writer first, then the release-gate report:

```text
cargo run --release --bin jope-selfplay-status -- --root /home/ubuntu/jope-flywheel/test_exact_gen0 status
python3 scripts/jope_recovery_report.py --root /home/ubuntu/jope-flywheel/test_exact_gen0 --strict
```

The report is healthy only when controller, ingestion, queue, supply, trainer,
promotion, watchdog, and freshness gates are all present and current. Queue
health requires useful nonzero depth; a fresh error document is not healthy.
The report contains one explicit row for each task with the accepted pointer,
latest shadow, corpus watermark, queue freshness, promotion eligibility,
blocking gate, and last verified learning delta.

Checkpoint evidence is published immutably under
`ops/learning-reports/<checkpoint>.v1.json` with schema
`JopeLearningReport.v1`. Use `scripts/jope_learning_report.py` to publish a
receipt; incomplete replay, family overlap, invalid proposals, or missing
sealed-panel evidence produces `inconclusive`, never an improvement claim.

An accepted checkpoint is a learning claim only when its sealed evidence shows
family-disjoint matched performance improvement with bounded latency and
failure regressions. Corpus growth, completed training, a larger observation
watermark, or test-adaptive evidence alone cannot mark a checkpoint smarter.
The per-checkpoint learning curve records proposal validity, strict-win rate,
champion rate, GP-relative lift, p95 latency, failure rate, and promotion
disposition; unavailable measurements remain `null` and the disposition is
`inconclusive`.

Until the gate is green for a sustained 24-hour window, the current accepted
checkpoint remains the production parent. Binary and regression plateau
directives must address proposal validity and GP-relative performance; learned
risk ranking waits for family-disjoint calibration, and repeated same-recipe
rejection requires a recipe-change directive.

Foundation plumbing already present in this checkout—four-task protocols,
training watermark, alternating shadow lanes, immutable replay/evidence,
promotion/rollback controls, and native orchestration—is recorded as completed
infrastructure, not as proof of continual improvement.
