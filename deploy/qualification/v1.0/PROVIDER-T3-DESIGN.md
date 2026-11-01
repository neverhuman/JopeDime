# REL-07 T3 provider: telemetry observer neutrality

T3 checks that telemetry on/off does not alter search output. Reasoning-mode
neutrality is not tested: the reasoning setting is parsed but is not routed to
a production run or observer, so the former config-only test was removed to
prevent a vacuous pass. T3 remains incomplete for reasoning neutrality until
that setting reaches a real run consumer and an observable effect is asserted.
Telemetry timing tests report measurements only; current owner direction
forbids speed verdicts.
`providers/t3.py` binds
each execution to the coordinator's candidate artifact and source SHA, verifies
the complete release artifact, and refuses a dirty or differently pinned
checkout. It runs three exact `jope-service` unit tests: telemetry on/off
parity, sensor-tick measurement, and fixed-work measurement. The benchmark uses `--nocapture`,
so the provider validates libtest's interleaved case output with
`common.exact_case_passed` alongside exact test counts. Each required test must
execute exactly once, pass, and not be ignored or measured before a
candidate-bound result is written outside the checkout.

The provider is a release-gate adapter; its fixture tests use synthetic
candidate bytes and mocked child commands and are not release qualification.
Actual qualification requires the coordinator's candidate artifact and a
clean checkout at its declared source SHA. It uses CPU-only local Rust tests;
it does not access sealed data or require a GPU.

Run the adapter contract tests with:

```sh
python3 -B deploy/qualification/v1.0/test_provider_t3 -v
```
