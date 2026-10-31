# `numpy-pcg64`

This crate is an internal Rust port of the integer-sampling subset of NumPy
1.26.4 used by JopeDime's frozen sampling recipes. It reproduces the integer
seed path through `SeedSequence`, `PCG64`, and `Generator`, including
`integers`, `choice`, `shuffle`, and `permutation`. It is not a general NumPy
random API: floating-point distributions and weighted `choice` are outside its
scope.

## Recipes and documents

`recipes` implements the repository's foundation and target sampling recipes;
`document` renders their `sampling.json` documents in the frozen byte layout.
These helpers retain the seed, schema, version, and row-order rules from the
Python exporters. Use the committed sampling goldens as the compatibility
oracle. The tests do not need Python or NumPy at build time.

The crate deliberately pins its behavior to NumPy **1.26.4**. Do not change a
sampling result or its document format as a cleanup; a compatibility change
needs updated source evidence and reviewed golden fixtures.

## Checks

From the workspace root:

```sh
cargo test --locked -p numpy-pcg64
cargo clippy --locked --all-targets -p numpy-pcg64 -- -D warnings
```

The `vendored_subset_matches_its_sources_when_available` test checks external
source evidence when `NUMPY_PCG64_JOPE_DIME_GIT` and/or
`NUMPY_PCG64_ARCHIVE_ROOT` are configured. Unavailable source paths are reported
as `SKIP provenance`; set `NUMPY_PCG64_REQUIRE_SOURCES=1` to fail when a required
git source cannot be read. The committed Rust output and sampling goldens run as
ordinary tests.
