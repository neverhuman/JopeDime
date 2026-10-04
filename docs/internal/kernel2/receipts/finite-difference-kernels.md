# Causal integer difference kernels — version 2 / wire 59

## Scope

The first B08 slice provides guarded scalar-column kernels, a causal group-local
cursor with at most `m` previous rows per group, and standalone Rust/Python/C++17
helper twins. It does not admit an executable program stage, model, fitted
predictor, full training recipe or search profile. Wire 59 stays reserved and
not generatable. Full state serialization, recipe continuation and EEC admission
are subsequent B08 requirements, not established by helper parity.

Acceptance: `cargo test -p feat-math --locked --test finite_difference_v2`.
Nine nonignored regressions cover all four orders, polynomial exactness,
boundaries, width limits, causal group/cut isolation, held-out poison, transactional
refusal and 2,940 shared helper cases. Product PR and full main each add nine
tests. Raw command receipts, inventories, source identities and tool versions:
`/home/ubuntu/jope-evidence/kernel2/bt/B08_CAUSAL_FINITE_DIFFERENCE/`.

## Arithmetic and state contract

Only version 2 / wire 59, order 1–4 and finite fixed `h>0` are accepted. The
integer backward stencils are `[1,-1]`, `[1,-2,1]`, `[1,-3,3,-1]`,
`[1,-4,6,-4,1]` in newest-first mathematical notation. Implementation consumes
oldest-first rows, taking successive `next-previous` differences and dividing
by `h` at each level. It never forms an overflowing or underflowing `h^m`.
If finite subtraction overflows, the level retries `next/h-previous/h`.
Arithmetic order is fixed; contraction and fast math are disabled in C++.
No intermediate clipping changes the stencil. Nonfinite input cells become
positive zero with `NONFINITE_INPUT`. A nonfinite final result becomes positive
zero with `NONFINITE_SANITIZED`; finite output projects to ±1e12 with
`MAGNITUDE_CLAMPED`. Valid negative zero is preserved.

Before projection, with normal finite intermediates, the absolute forward error
is bounded by `gamma(2m) * sum_k(C(m,k)*abs(x_k)) / h^m`, where
`gamma(n)=n*u/(1-n*u)` and `u=2^-53`. Cancellation may make relative error large;
this bound excludes underflow and sanitized paths. This is a frozen floating
point evaluation order, not a claim of correctly rounded exact-real stencils.

The first `m` observations of each group return positive zeros and the
stage-local `BOUNDARY_HISTORY` bit 17. Numerical bits retain their scalar meanings;
bit 17 belongs to this versioned stage result only. The scalar `FaultSet` catalog
and decoder remain unchanged. Boundary results also record nonfinite cells in
the causal window.

State owns a B01 SampleTime/Causal/Stateless cursor. It rejects IID tasks,
invalid ordered identities, invalid group keys, nonfinite/backward timestamps,
empty/ragged/wrong-width rows, and excess declared input. Refusal leaves the
cursor and history unchanged. Stable timestamp ties retain input order; gaps do
not change fixed `h`. Interleaved groups carry separately across the same cut's
train/held-out boundary. A new instance resets the cut. Labels and fit callbacks
never enter this stateless transform. `finish` checks the declared row count.

## Per-operation definition of done

| Required check | This slice |
|---|---|
| Shortest existing-tree comparison | Held for complete stage admission |
| At least six-dataset utility | Held; no utility claim |
| Runtime/interpreted/compiled and emitted twins | Kernel runtime + standalone helper execution only; full routes held |
| Guarded kernel | Defined width, parameter, input and output guards |
| Defined NaN/Inf/zero/fault semantics | Explicit contract and shared bit corpus |
| Arity bound | Order 1–4; at most five observations, width 1–512 |
| Bounded Experimental prior | Generation remains closed |
| Equal-budget promotion G-A…G-E | Held; no Baseline admission |
| Token/bytes delta | Standalone helper sizes retained externally; full recipe delta held |
| Capability lifecycle | Reservation unchanged, no admission rung promoted |
| Train-only fit | No fit; causal held-out continuation only |
| Deterministic width | Preserve admitted positive width |
| Saturation fault | Explicit finite output projection |
| Removal from generation without removing parsing | Generation remains disabled |
| Append-only IDs/helpers | Wire 59 only; legacy IDs/catalog/helper bytes unchanged |

V2 additions: existing EEC regressions run unchanged; new-node creation/change/
removal, complete Rust/Python/C++17 recipes and fitted C++ remain held. Generated
helper text must pass the IP scrub. No new crate dependency. Unknown version/ID,
empty row and old compatibility gates remain required.

## Verification discipline

Fresh baseline on remote main `b036848460f3a2887fcdc6120db97152a3c01490`:
PR 2099/29, main 4181/66, Python 231/2 deselected, historical golden 16/16,
fitted sanitizer and C++20 18/18 each. Named skips and all 100 protected paths,
including 38 identity inputs, are pinned. Expected after counts: PR 2108/29,
main 4190/66. All exact-head after results and hosted attempts remain separate
evidence; this source receipt does not assert that a queued gate has passed.
`just required` has the owner-accepted held Clippy floor on unchanged main;
actual exit 101 is not PASS. Zero new workspace/owned findings and unchanged
held paths remain mandatory. The copied B07 completion assertion failure and
its external harness-only correction are preserved; raw baseline suites were
not restarted.
