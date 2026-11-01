# REL-07 S1 surface-equivalence provider

## Approach

Run the existing exact acceptance test `all_product_surfaces_share_the_bounded_run_projection` from the `jope-appliance` `e2e` integration target. The test creates a bounded `RunSpec` from the repository's committed public TRAIN fixture, exercises batch CLI, terminal, MCP, loopback HTTP and in-process calls, and asserts the champion, scores and deterministic event projection match. It also covers same-key retries and independent same-spec runs. Its test harness counts as one exact assertion case; unrelated e2e tests are excluded by the selector.

Before the assertion, the provider verifies the supplied non-development full candidate through `jope-dist inspect` and `verify`, requires the embedded source pin to match the declared 40-character source SHA, and requires the local checkout to be clean at that exact commit. It checks the artifact hash and source/checkout identity again after the test. The successful child result is written exclusively to a new external path with the exact S1 ID, source SHA, candidate SHA-256 and one passing test count.

Run through `release-gates` with:

```sh
python3 -B deploy/qualification/v1.0/providers/s1.py
```

The coordinator supplies the five `JOPE_REL07_*` variables documented in `GATE-RUNNER.md`. The provider removes external INT-01 CSV, golden-packet, executable and qualification-home overrides from the test child environment, so this assertion uses the committed fixture and lane-local target only. It sets incremental compilation off, debug info to line tables, and at most four Cargo jobs.

## Alternatives rejected

A generic successful Cargo command is insufficient: it could select no tests, a different test, or one of the synthetic comparator controls. Reimplementing `SurfaceCapture` comparisons in Python would not exercise the product transports. Re-running the whole e2e target would include unrelated checks and blur the S1 coverage boundary. No gate-specific candidate or gate result is synthesized from provider unit tests.

## Risks and limits

This provider binds the source checkout used by the acceptance test to the full candidate's embedded source pin; it does not launch the signed candidate binary itself. The exact existing acceptance test uses one bounded committed TRAIN fixture and does not qualify every dataset, mode, platform, named client or release gate. The local provider-control tests use synthetic bytes and mocked child processes only. A real S1 gate result exists only after `release-gates` invokes this provider against the supplied full candidate; current implementation and unit-test passes are not release qualification.

## Acceptance

Data-free contract tests:

```sh
nice -n 10 env CARGO_BUILD_JOBS=4 CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only pytest -q deploy/qualification/v1.0/test_provider_s1
```

The exact source acceptance command used by the provider is:

```sh
nice -n 10 env CARGO_BUILD_JOBS=4 CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only cargo test --locked -p jope-appliance --test e2e --features engine,mcp all_product_surfaces_share_the_bounded_run_projection -- --exact --test-threads=1
```

Neither local command supplies or qualifies signed candidate bytes.
