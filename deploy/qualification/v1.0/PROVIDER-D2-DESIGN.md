# REL-07 D2 fused-cpu size provider

The provider qualifies only the signed candidate's executable size budget. It requires the coordinator's exact D2, artifact SHA-256 and source SHA bindings; independently requires a clean checkout at the candidate's embedded source SHA; and accepts only a verified, non-development `fused-cpu` package. Candidate paths must be regular files, not symlinks, and the source and candidate hashes are rechecked around measurement.

The sole measurement command is:

```sh
cargo run --locked -p jope-dist -- size-report --file "$JOPE_REL07_ARTIFACT" \
  --budget deploy/appliance/size-budgets.toml
```

PASS requires both `base_executable_bytes` and `final_executable_bytes` budget checks from the maintained report and a final executable no larger than 40,000,000 bytes. The provider records three size assertions in the standard `jope.rel07.gate-execution/v1` result. A size-report budget failure is retained as D2 FAIL when its JSON and exit status agree. Latency and startup timings are not executed, parsed or used in this result; the owner ruling makes D2 a size-only verdict. REL-06 startup and first-model measurements remain informational and separate.

Controls use synthetic candidates and mocked `jope-dist` output. They verify binding, candidate type/profile/source, both budget checks, the absolute 40 MB limit, malformed and duplicate report rows, failure accounting, and exclusive external result creation. These tests do not qualify a release. Actual D2 gate execution requires the signed candidate and the release-gate runner; no such candidate is claimed here.

Run the data-free controls with:

```sh
nice -n 10 env CARGO_BUILD_JOBS=4 CARGO_INCREMENTAL=0 \
  CARGO_PROFILE_DEV_DEBUG=line-tables-only pytest -v \
  deploy/qualification/v1.0/test_provider_d2
```
