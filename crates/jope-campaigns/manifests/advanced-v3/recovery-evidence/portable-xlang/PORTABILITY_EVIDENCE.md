# Cross-language portability proof — portable math kernel

**Result: Rust, C++, and JavaScript produce BYTE-IDENTICAL output** for the portable
`exp`/`ln` kernel across an 8352-point grid (subnormals, ±0, near-1, boundaries, overflow,
underflow, ±inf, NaN). All three output files hash to a single sha256.

This proves the central thesis of the multi-language codegen program: the hand-rolled
`portable_math` kernel (fdlibm/musl algorithms over IEEE-exact ops + bit manipulation)
computes identical bits on every backend — which native `libm` CANNOT (JS `Math.exp` and
device/CUDA libm diverge from glibc). Bit-manipulation crosses languages faithfully:
Rust `to_bits`/`from_bits` ≡ C++ `memcpy` ≡ JS `Float64Array`/`BigUint64Array`, and JS's
32-bit-only bitwise limitation is handled with `BigInt` for all 64-bit ops.

## Files (this dir)
- `ref.rs`  — Rust reference (exact copy of `vendor/feat-math/src/portable_math.rs` exp/ln).
- `ref.cpp` — C++ port (`double`, `memcpy` bit-casts, `-ffp-contract=off`).
- `ref.js`  — JS port (plain-number f64 arithmetic; `BigInt` bit ops; NO native `Math.*`).

## Reproduce
```sh
rustc -O ref.rs -o ref_rs && ./ref_rs > out_rs.txt
g++ -O3 -ffp-contract=off -std=c++17 ref.cpp -o ref_cpp && ./ref_cpp > out_cpp.txt
node ref.js > out_js.txt
sha256sum out_rs.txt out_cpp.txt out_js.txt   # -> one distinct hash
```

## Scope + next
- Proven: `exp`, `ln` (the two foundations — exercise bit-manip, `scalbn`/2^k reconstruction,
  polynomial eval, all branch + special-value routing). The other kernel functions
  (`log1p`/`expm1`/`tanh`/`pow`/`sin`/`cos`/`atan`/`atan2`) use the SAME technique and will be
  validated identically by the generalized golden parity oracle at integration time.
- NaN output payloads are canonicalized to a `NaN` token (payload bits are not IEEE-specified
  across languages); every finite/inf value is compared bit-exact.

## Coordination note
Done entirely in scratch — no edits to the frozen emitter boundary, the (currently
mid-refactor / non-compiling) feat-math crate, or any contested `advanced_v3/*` path. This is
the portability evidence for the Search-V3 review gate on the imported portable-math WIP.
Formalize into `vendor/feat-math/tests/` as a `(lang,build,run)` parity oracle once the crate
is green and the integration gate is lifted.

## P-1A root audit (2026-07-22)

- `full_ref.rs` now uses a repository-relative path, declares its f64 grids explicitly, and
  compiles with the stock Rust compiler. Two audited runs produced the same 1,325,310 bytes,
  SHA-256 `cb88fe62a5c6d6a2950b4ec14bbc0733ac9ac8527934c821e592d117c108e276`;
  the immutable task receipt remains the authority for this observation.
- The full Rust grid now includes the minimum subnormal, maximum subnormal, minimum normal,
  finite extrema, infinities, NaN, signed zero, and two-dimensional `pow`/`atan2` cases.
- There is still no full-helper Python/C++/JavaScript port. `ref.py`, `ref.cpp`, and `ref.js`
  remain exp/ln-only recovery evidence, so this file does not authorize P-1B or any emitted
  portable backend.
- The legacy Python/Rust emitter now fails closed when `MathKernel::Portable` is selected until
  every transcendental call is routed. `PlatformLibm` remains the byte-identical default.
