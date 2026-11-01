# REL-07 A1 transport-equivalence provider

The provider binds a full non-development candidate to the exact clean source, then requires four named Rust cases to execute and pass once each. Three maintained `jope-client` contract cases compare all 17 registry operations, request bodies and idempotent replays over in-process and loopback HTTP transports; exercise resumable SSE; and compare byte-identical replies from the real service router. Other crate tests are filtered outside this gate's coverage.

The fourth case is a new, explicitly selected no-engine integration test, `a1_lite_registry_and_candidate_background_contract`. It compiles the shared request vectors without the engine feature, verifies methods, targets and bodies for all 17 operations, and talks through the lite HTTP client to an actual background engine started from the supplied candidate. It checks system information and invalid registration, verifies the binary's source pin, and stops only its own child. It requires no dataset or fit. The full 17-operation response/replay comparison remains the maintained mock-router contract suite; the real-router and lite daemon cases are bounded runtime checks, not complete positive discovery/model workflows (those belong to S1).

Run `python3 -B deploy/qualification/v1.0/providers/a1.py` with the five standard `JOPE_REL07_*` bindings in [GATE-RUNNER.md](GATE-RUNNER.md), under the controller's nice10/CPU admission. The provider uses shared `jope-dist inspect` and `verify` checks, checks source cleanliness and byte identity between commands, refuses symlinks/non-files and existing/in-checkout results, and writes the existing candidate-bound child schema only after all four exact named cases pass. Zero tests, an ignored/measured/skipped case, duplicate summaries/success lines, a failed child or identity drift cannot write PASS. A required external-candidate test is ignored by ordinary `cargo test`; the provider explicitly selects it with `--ignored`, and never credits that default omission as coverage.

The lite case receives the candidate path/source through the restricted child environment. Engine/fake feature flags are absent from its Cargo command. Inherited endpoint, injected Rust cfg flags and store failpoint settings are removed; CPU jobs are four, math threads one and CUDA disabled. Private short runtime directories live below `~/.local/state/jope-a1/`, outside `/tmp` and the source tree. Daemon tokens stay in the private temporary store and are not logged. No signing, sealed data, dataset download, GPU work or speed verdict is involved.

Data-free controller acceptance:

```sh
nice -n 10 python3 -B -m pytest -q -p no:cacheprovider deploy/qualification/v1.0/test_provider_a1
```

The unit suite simulates inspector/verifier and Rust outputs; it is not release execution evidence. Local source tests may separately use an explicitly labeled development fixture binary to check the new lite path; such a run does not satisfy the provider's full-release envelope admission. A real A1 result requires all four commands against the actual full candidate's pinned source and supplied bytes. No other release gate or platform is qualified by this implementation.

A full-feature-only wrapper was rejected because it would omit the lite graph. A zero-test lite invocation of the existing engine-gated contract target was rejected because it executes no assertion. Reimplementing the client/router or silently relaxing candidate admission was unnecessary. No production API, worker/model protocol, dependency, Cargo lockfile or common-runner change is needed.
