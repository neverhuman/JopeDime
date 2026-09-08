**Audit evidence — 8 September 2026**

See [the full report](/home/ubuntu/jope-dime/docs/ALGORITHM_KERNEL_AUDIT_2026-09-08.md). These are focused audit reproductions, not a production test suite or implementation changes.

| Files | Purpose |
|---|---|
| `syntax-probe.rs`, `.log` | Canonicalization, scalar behavior and caught expression panic |
| `admission-probe.rs`, `.log` | Public validation/fitting versus legacy emitter support |
| `evolution-probe.rs`, `head-mutation-probe.rs`, `evolution-probe.log` | Deterministic synthetic mutation/crossover probes; head probe mirrors repair and placement checking, not the full campaign |
| `codegen-probe.rs` | Fits an out-of-range-column symbolic model and writes emitted C++ to its first command-line argument |
| `codegen-model.cpp` | Generated C++ plus a minimal two-column inference harness |
| `codegen-asan.log` | AddressSanitizer heap-buffer-overflow from that harness |
| `legacy-freeze-test.log` | Bound parity and legacy freeze results, including the pre-existing pinned source-hash failure |
| `source-sha256.json` | Digests captured for inspected source files; the checkout was already dirty |

The Rust probes were compiled with Rust edition 2021 against the root workspace's freshly built `feat-math` rlib, with `target/debug/deps` as the dependency search path. Build it with `rtk cargo build -p feat-math`. For example, using the build artifact present during the audit:

```bash
rtk proxy rustc --edition=2021 docs/audits/algorithm-kernel-2026-09-08/syntax-probe.rs --extern feat_math=target/debug/deps/libfeat_math-47b92535736aa9b2.rlib -L dependency=target/debug/deps -o /tmp/gp-syntax-recheck
rtk proxy /tmp/gp-syntax-recheck
```

Use the rlib produced by the current build if its filename differs. The syntax probe deliberately catches a panic; Rust may still print the panic hook message to stderr.

To repeat the memory check on the preserved generated artifact:

```bash
rtk proxy g++ -std=c++17 -O1 -ffp-contract=off -fno-fast-math -fsanitize=address,undefined docs/audits/algorithm-kernel-2026-09-08/codegen-model.cpp -o /tmp/gp-codegen-asan-recheck
rtk proxy /tmp/gp-codegen-asan-recheck
```

The sanitizer invocation is expected to fail with a bounds violation. To verify a future emitter fix, regenerate the C++ with `codegen-probe.rs` and append the small `main` harness from the preserved artifact before compiling; recompiling the old artifact does not test a new emitter.

Probe counts describe the supplied fixtures and seed sweeps, not measured production failure rates. No dataset campaigns were run.
