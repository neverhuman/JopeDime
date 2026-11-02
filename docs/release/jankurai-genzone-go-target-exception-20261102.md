# GenZone Go target template relocation

The full report `rerun/j_27137b5fb.json` reports HLT-000 fingerprint
`sha256:06bb2e0afe56865b87a6ba80b72f3405dd2d13371be33abb0e490e3a0cbe6f71` at
`crates/feat-math/src/go/model.go` because its source suffix is `.go`.

This file is the adapter template for the explicit optional Go export target
documented in `docs/go.md`. `crates/feat-math/src/go.rs` substitutes the bounded
header list and emits the result at artifact path `model.go`; the generated Go
library runs the existing Rust numerical implementation compiled to WebAssembly
through the pure-Go wazero runtime. The repository's primary numerical core
remains Rust. This supported export is a narrow target exception, not permission
to add Go elsewhere in product code.

The template now lives at `crates/feat-math/templates/go/model.go.template`, outside
the product runtime source tree. `include_str!` still emits the same bytes at artifact
path `model.go`; only the repository's template path changed. The ineffective
exact-path score exclusion has been removed. Go/WASM golden checks, Rust tests, and
other security scans remain in place.
