# REL-07 T2 no-GP-leak provider

## Approach

Bind the provider to the exact full release candidate SHA and its embedded source SHA. Require a clean checkout at that same source commit, build the local appliance executable needed by the bounded integration fixture, then run exact `jope-service/tests/no_gp_leak.rs` cases. The bounded real-run case injects the four internal DSL/operator/lineage/round sentinels and scans HTTP, SSE, MCP, CLI, terminal, errors, export manifests and logs. Separate exact tests exercise the synthetic projection and every surface in the scanner. Finally rerun the bounded product test with the intentional HTTP leak switch; count that control as passing only when the test fails on the expected HTTP DSL sentinel.

The coordinator command is `python3 -B deploy/qualification/v1.0/providers/t2.py` with the standard five `JOPE_REL07_*` bindings. Results bind candidate/source hashes and count the four positive Rust test cases; the expected-failure negative control is reported separately and accepted only for the exact expected sentinel.

## Alternatives rejected

Do not scan only fabricated output as gate evidence, accept an arbitrary command failure as the leak negative control, or treat a skipped/zero Rust test as PASS. Do not read sealed/holdout data or use the test dataset as a training/qualification input; the existing bounded test uses its committed synthetic fixture only.

## Risks and limits

The test-only failpoints feature exercises current source pinned to the candidate; it is not a proof that the signed executable itself was launched. The provider refuses any source drift and verifies the complete candidate bytes, but candidate-bound release execution still requires a real full candidate. The TUI placeholder is not selected because the bounded full-run case itself captures and scans the PTY transcript. No other §14 gate or release qualification is closed.

## Acceptance

Run data-free controls with `nice -n 10 env CARGO_BUILD_JOBS=8 CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only uv run --project python --locked --no-sync pytest -q deploy/qualification/v1.0/test_provider_t2`.
