# REL-07 T3 provider: observer neutrality

T3 proves that observer settings do not alter search output and that telemetry
sampling stays within its fixed-work overhead budget. `providers/t3.py` binds
each execution to the coordinator's candidate artifact and source SHA, verifies
the complete release artifact, and refuses a dirty or differently pinned
checkout. It runs four exact `jope-service` unit tests: reasoning off/detailed
and telemetry on/off parity, the existing telemetry parity test, sensor-tick
cost, and the fixed-work benchmark. The benchmark itself enforces overhead
below 2 percent. Each required test must execute exactly once, pass, and not be
ignored or measured before a candidate-bound result is written outside the
checkout.

The provider is a release-gate adapter; its fixture tests use synthetic
candidate bytes and mocked child commands and are not release qualification.
Actual qualification requires the coordinator's candidate artifact and a
clean checkout at its declared source SHA. It uses CPU-only local Rust tests;
it does not access sealed data or require a GPU.

Run the adapter contract tests with:

```sh
python3 -B deploy/qualification/v1.0/test_provider_t3 -v
```
