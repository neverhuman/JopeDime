# GenZone exact-path Go target disposition

The full report `rerun/j_27137b5fb.json` reports HLT-000 fingerprint
`sha256:06bb2e0afe56865b87a6ba80b72f3405dd2d13371be33abb0e490e3a0cbe6f71` at
`crates/feat-math/src/go/model.go` because the source suffix is `.go`.

This file is the adapter template for the explicit optional Go export target
documented in `docs/go.md`. `crates/feat-math/src/go.rs` substitutes the bounded
header list and emits the result at artifact path `model.go`; the generated Go
library runs the existing Rust numerical implementation compiled to WebAssembly
through the pure-Go wazero runtime. The repository's primary numerical core
remains Rust. This supported export is a narrow target exception, not permission
to add Go elsewhere in product code.

Only `crates/feat-math/src/go/model.go` is excluded from Jankurai scoring in
`agent/audit-policy.toml`. Its source bytes and emitted artifact path remain
unchanged. Go/WASM golden checks, Rust tests, and other security scans remain in
place. Remove this exact-path disposition if the Go target is retired or the
template stops serving that target; review any other finding separately.
