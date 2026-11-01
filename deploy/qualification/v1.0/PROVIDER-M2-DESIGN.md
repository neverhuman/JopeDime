# REL-07 M2 CPU ranker parity provider

M2 uses the existing ignored `ranker::cuda::fixture::real_full_forward_cpu_fixture_replay` test as its one required CPU assertion. That exact Rust test instantiates the frozen CPU forward graph, consumes the hash-pinned RES-07 TRAIN packet and frozen RES-06 tensors, and compares `v256`, `col`, DOPE 629, `lm48`, tokens, mask, and costs at the §5.8 tolerances over 50 contexts and 100 serving-shaped slates. It also checks admitted-set identity on decisive slates and reports tie-band slates separately. The test does not build the CUDA feature, initialize a GPU, alter the frozen weights, fit a model, or run an evaluator.

The provider requires the standard five `JOPE_REL07_*` bindings and three explicit parity inputs: `JOPE_GOLDEN_DIR`, `JOPE_GOLDEN_INDEX_SHA256`, and `JOPE_PARITY_BUNDLE_DIR`; it writes the Rust replay report under the new external `JOPE_PARITY_REPORT_DIR`. Because the current CPU gate runner forwards only its documented environment allowlist, an explicit command manifest must pass `--golden-dir`, `--golden-index-sha256`, `--parity-bundle-dir`, and `--parity-report-dir`; these flags bind the same environment inputs. Automatic provider discovery without the fixture bindings fails closed. Only the owner-accepted RES-07 R5 packet index `ddaa6cfc558f0a7e56fe73f657266d00d6c821739a602afc701f87fa1c4f5af1` is accepted. The Rust fixture loader hashes every consumed file against that index and validates the frozen bundle tensor hash. The report is checked for exact source/index identity, 50 contexts, 100 slates, all required stages, zero out-of-band admission mismatches, and zero tolerance misses before the standard child result can say PASS.

Before and after the assertion, the provider requires the exact clean source checkout, a regular non-symlink candidate with unchanged bytes, and new report/result paths outside the source tree. It verifies the full non-development candidate with `jope-dist inspect` and `verify`, matching the embedded source SHA to the tested checkout. Missing inputs, wrong identities, changed candidate/source, zero/skipped/extra cases, malformed reports, missing stages, stage mismatches, or report/result collisions refuse PASS.

The child inherits and validates the runner's `CARGO_BUILD_JOBS` setting (1–4; default 4), without adding a Cargo `-j` override. Data-free controls run with:

```sh
nice -n 10 python3 -B -m pytest -v deploy/qualification/v1.0/test_provider_m2
```

These controls mock candidate verification and the Rust child; they prove provider refusal and result-writing behavior only. The actual M2 numerical assertion runs only when `release-gates` supplies the exact signed candidate and its approved external TRAIN fixtures. This provider does not qualify CUDA, Q1, a release, or learned-serving promotion.
