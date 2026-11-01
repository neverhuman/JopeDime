# REL-07 CPU providers: M1 and M4

## Approach

Add the first CPU-only providers as direct adapters to the existing acceptance suites. Both first verify the supplied candidate with the maintained `jope-dist inspect` and `jope-dist verify` commands, require its embedded full-profile source pin to equal the runner's frozen source SHA, and rehash the candidate after testing. M1 runs the pinned fused-bundle exporter suite, which rebuilds the bundle under different hash seeds and working directories and exercises the complete tensor/freeze guard refusals. M4 runs the existing Rust signed-bundle corruption controls for wrong hashes, invalid headers, non-finite tensors, shape errors and size caps. Each writes the existing `jope.rel07.gate-execution/v1` child result with positive test counts only when the targeted suite has no failures, skips or exclusions.

The command argv for the coordinator is `python3 -B deploy/qualification/v1.0/providers/m1.py` or `python3 -B deploy/qualification/v1.0/providers/m4.py`. The coordinator supplies the standard five `JOPE_REL07_*` identity/result variables and retains command logs and results.

## Alternatives rejected

Do not turn a source formatter, generic harness test, empty command, or fixture-only pass into a gate result. Do not copy or alter the signed candidate to create corruption cases; M4 uses the existing signed fixture-key unit tests for adversarial bundle inputs. Do not let either provider emit PASS from test output unless the candidate envelope verifies and pins the exact tested source.

## Risks and limits

These CPU provider commands are candidate-bound to the verified package source identity but their test cases use the repository's pinned exporter custody and Rust loader fixtures. They do not qualify CUDA, native platforms, other §14 gates, signer approval, or release qualification. A real candidate is still needed to execute the providers through `release-gates`; no signed candidate or all-28 receipt is present at implementation time. Later batches need their own gate-specific command/input review and stay under this same fail-closed result contract.

## Acceptance

The focused new test command is:

```sh
nice -n 10 env CARGO_BUILD_JOBS=8 CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only python3 -B deploy/qualification/v1.0/test_gate_providers_m1_m4 -v
```

M1/M4 runtime assertions are run only by the provider executables against an owner-approved candidate whose signed full-profile manifest pins the frozen source SHA. Local synthetic tests exercise contract handling only and are not gate execution evidence.


## Exclusive result publication

M1/M4 now apply the same new, external, absolute non-symlink result-path
contract as the other gate providers. The shared suite helper checks the path
before executing a child and again before publication, then opens the final
result exclusively. A receipt created while the suite runs is preserved and
causes refusal. The helper no longer writes or replaces a predictable `.new`
file, so prior incomplete evidence at that path is also retained. A failed or
interrupted write can leave a partial result; it cannot qualify because the
coordinator requires successful child exit and valid complete JSON.

This changes evidence publication only. Candidate checks, M1 exporter/freeze
assertions, M4 integrity selectors, actual test counts and the six-field result
schema are unchanged. Controls cover prior and concurrent receipts, a concurrent
symlink, unsafe result locations, retained temporary evidence, successful M1/M4
results and a failing child. Fixture results are not release qualification.
