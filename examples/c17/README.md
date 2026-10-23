# C17 consumer example

From the repository root, run `bash examples/c17/run.sh`, or
`CC=clang bash examples/c17/run.sh`. Requires Rust, Cargo and GCC or Clang on Linux.
The script exports a small complete training recipe, builds its Rust-backed native library,
compiles the strict C17 caller and prints two predictions. Its fresh output directory retains
all sources and the manifest. This is a synthetic integration example, not a benchmark.

`main.c` demonstrates fit, prediction, status handling and model release. See the
[C17 guide](../../docs/c17-export.md) for the ABI, coverage, build dependencies and limitations.
