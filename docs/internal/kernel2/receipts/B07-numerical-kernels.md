# B07 numerical kernels — helper prerequisite

Status: source plus focused local numerical/helper proof. This slice remains
outside program/model loading, full train-predict recipes, fitted prediction,
shipping profile generation and serving promotion. It does not complete B07.

Base: `0d030295024b4c7b83c001c993b00da82a35a937`, after B06 #169;
main-guard 37128184320 attempt 1 SUCCESS. Sole lane `bt-implementation`,
branch `kernel2/bt07-numerics`. External custody:
`/home/ubuntu/jope-evidence/kernel2/bt/B07_NUMERICAL_POLICY`.

## Numerical contract

Separate `kernel::numerical_policy` APIs implement the opt-in `v2` kernel
identity `feat-math.fdlibm-neumaier-lassq.v2`. No caller default selects them.
The historical Func13/asinh/symlog and v1 row reductions stay unchanged.
No Func, Var or wire ID is added. `EMITTER_VERSION` remains `2.2.0`;
no evaluator-identity input or existing frozen fixture is edited/regenerated.

- Asinh uses tiny-input identity, log1p/central, middle and huge logarithmic
  branches; signed zero is preserved.
- Sum/l1/mean retain error-free partials. Sum/l1 preserve the authorized ordered
  Neumaier unrepresentable-prefix sanitizer, then use the same bounded expansion
  as mean. The threshold is the exact sum/l1 bound or bound times count.
  Mean always scales by a power
  of two at least twice the row count and accumulates an error-free FastTwoSum
  expansion, retaining every nonzero low component. It restores those components
  and the original inputs' scaling losses into a second expansion. An explicit
  magnitude-order normalization precedes the initial leading-component guard:
  cancellation can leave the first expansion weakly overlapping. A conservative
  early guard avoids overflowing restoration; an exact expansion comparison
  against the integer threshold `1e12 * row.len()` determines the saturation
  fault. The leading component is divided once to produce the bounded mean.
- L2 accumulates exact positive dyadic squares into 86 base-2^26 digits in
  units of 2^-2148. A three-term 26/27-bit significand split needs only u64
  arithmetic, including the square of the minimum subnormal. Inputs above the
  output limit clamp before multiplication. Exact integer comparison with
  (1e12)^2 precedes output rounding; the result is normalized by a power of two
  before one square root and rescaling. Tiny positive tails retain their fault.
- Nonfinite inputs produce +0 with NONFINITE_INPUT. Empty sum/l1/l2 are
  +0 identities; empty mean raises EMPTY_ROW_REDUCTION. Finite outputs above
  ±1e12 raise MAGNITUDE_CLAMPED; an unrepresentable compensated sum/l1 uses
  +0 with NONFINITE_SANITIZED. These guards and arithmetic order are shared
  by the emitted twins.

The mathematical asinh branches follow the [Netlib fdlibm reference](https://netlib.org/fdlibm/s_asinh.c);
L2 uses an exact dyadic accumulator followed by power-of-two normalization.
The kernel identity is introduced by this PR; no previously shipped v2 kernel
identity or v1 artifact changes.
References stay in documentation outside emitted text.

## Mean error bound and overflow argument

Assume IEEE binary64 round-to-nearest, gradual underflow, and no contraction,
reassociation or fast math, as used by the certification commands. Let `n` be
the integer row width, `S` its exact real sum, `h` the leading restored expansion
component, `u = 2^-53`, and `eta = 2^-1074`.

Ordered FastTwoSum preserves each addition as an exact high/low pair; retaining
all low components avoids a single compensation accumulator losing nested
cancellation. This primitive follows [Ogita, Rump and Oishi, Algorithm 1.1](https://www.tuhh.de/ti3/paper/rump/OgRuOi05.pdf).
The argument below concerns this helper's composition, not a claim that it is
one of that paper's complete summation algorithms.

Scaling by `q >= 2*n` bounds every prefix's absolute sum by approximately half
`f64::MAX`, with ample margin even when converting a very large integer width
rounds. Power-of-two scaling is exact except at underflow; the separately
retained `x - (x/q)*q` restores that loss exactly. If the scaled leading
component exceeds twice the scaled output threshold, its expansion tail and
scaling losses cannot bring the true mean inside the threshold, so it clamps
before restoration. Otherwise, restored components are bounded by the small
output threshold times the width, far below overflow for a 64-bit `usize`.
The final threshold comparison uses 13-bit integer chunks: each `chunk*1e12`
is exactly representable, and power-of-two place values preserve that exactness.
This avoids a rounded threshold or rounded mean inventing a magnitude fault.

The bound below applies to the explicitly normalized scaled and restored
expansions, not the potentially weakly overlapping first pass. That distinction
was exposed by the fourth independent review and its failing initial-tail
invariant remains in immutable custody. The nonoverlapping expansion's omitted tail satisfies
`|S-h| <= u/(1-u) * |h|`. For an exactly represented count, before output
projection the absolute mean error is bounded by
`|S-h|/n + u*|h/n| + eta/2`. Including a potentially rounded integer-to-float
count gives the conservative bound `gamma4*|S/n| + eta`, where
`gamma4 = 4*u/(1-4*u)`. Projection onto the certified finite output interval
cannot increase error when the exact mean lies inside it. Exact cancellation
returns positive zero; saturation faults use the exact expansion comparison.
This is an analytic roundoff bound, not a universal correctly-rounded promise
or a waiver of the unchanged corpus limits: normal results must still be within
two ULP, exact zeros must match bits, subnormals must be within one `eta`, and
faults must match exactly. The independent reviewer must verify the expansion,
scaling/underflow, threshold and bound argument across the entire helper family.

## Exact-square L2 bound and integer safety

Each finite input at or under 1e12 has a 53-bit significand. Splitting it into
26-bit low and 27-bit high pieces gives exact square terms `low^2`, `2*low*high`
and `high^2`, each under 2^54. Adding them at their exact bit offsets uses
26-bit digit carries with intermediates under 2^52. The accumulator unit is
2^-2148, so no positive square is discarded through floating underflow.
Every accepted prefix is at most 1e24; before the next exact comparison it is
at most 2e24. Its top bit is at most 2228 relative to that unit, and 86*26=2236
bits leave room for all carries. Positive terms permit immediate saturation
without depending on later row order. Faults compare exact integers, including
[1e12, minimum-subnormal] and [6e11,8e11, minimum-subnormal].

The top 53 bits plus round/sticky bits round the normalized exact square sum
once to binary64, ties-to-even, in [1,4]. On the certified host with correctly
rounded square root, its square root and exact power-of-two rescaling have at
most the square-sum rounding error halved plus one sqrt rounding error: a
conservative normal relative bound `gamma2 = 2*u/(1-2*u)`. Gradual-underflow
rescaling adds at most eta/2 absolute error. The unchanged two-ULP normal,
one-eta subnormal and exact fault corpus gates still apply; no universally
correctly rounded norm or cross-host libm claim is made. Final projection to
1e12 cannot create a fault after the exact comparison proved the input inside.

## Focused proof and acceptance

`cargo test -p feat-math --locked --test numerical_policy_v2` executes fifteen
nonignored tests. Tests-first missing-module failure is retained, followed by
five initial passes. The expanded run retained its two IP-scrub failures:
a reference URL was included in emitted Rust. Moving the reference into
non-emitted module documentation corrected both without changing the gate.
The initial corrected focused run passed 9/9. Independent native review
5401784989 then found finite extreme means at non-power-of-two widths
returning zero/NONFINITE_SANITIZED. A new regression preserves that failure;
the first retry fixed that case without changing the finite unscaled path.
Fresh review 5402083050 found that division by a non-power-of-two row count
introduced a false residual in an exact cancellation row. The second correction
uses power-of-two scaling and retains tiny-input scaling loss. Its regression
covers both signs, exact zero, a fractional residual, and normal/subnormal
residuals. The failing numerical repro and earlier test setup errors remain
in separate immutable custody logs; setup errors are not numerical proofs.
Third native review 5402426906 exposed nested cancellation within the compensation
accumulator itself. The whole-class correction replaces the v2 mean path with
the scaled retained expansion above, including finite unscaled rows. Sum/l1,
L2, asinh, public guards and fault assignments retain their prior behavior.
`correction-03/focused-red-02` preserves the numerical failure on the unchanged
third-review head; `focused-red` preserves a separate test-count setup error.

The additive v2 corpus pins 1,380 asinh inputs and 72 row inputs with an
independent mpmath 1.3.0 / 180-decimal-digit oracle (generator and corpus
hashes in custody). Normal oracle results are within two ULP; exact zero
and one-min-subnormal absolute bounds are checked separately. Guarded
nonfinite/unrepresentable inputs and saturation have explicit expected faults.

All 72 existing row cases retain their four reduction oracles. The 94 mean-only
cases include the complete third reviewer's adversarial family, using exact
binary-rational Fraction arithmetic and a single rounded conversion. The
previous eight mean-only cases remain unchanged. Exact zero and small residuals,
both signs, repeated extremes and nested three-scale cancellation are covered.
The original 64 row cases are preserved. Eight independent-oracle rows add
positive and negative f64::MAX at widths 3, 6, 7 and 9; the runtime regression
covers both signs at every width from 2 through 32.

The property-style regression adds 11,136 deterministic probes across every
width 1 through 128, both signs and three orders. Cases include repeated
positive/negative MAX, paired and mixed-sign extremes, nested overflow and
finite-prefix cancellation, normal/subnormal residuals, and exact/adjacent
clamp boundaries. Analytic case expectations are checked again by an independent
Fraction audit of the actual generated matrix inputs, including exact faults.

Rust runtime and helper twins execute 63,610 value/fault observations in
each of nine drivers: rustc -O, Python 3.12.3, g++ and clang++ C++17 O2/O3
and C++20 O3, plus g++ C++17 ASan/UBSan. C++ always disables contraction
and fast math. Empty rows and unknown helper reduction names are exercised.
The same Rust source body serves runtime and generated Rust; independent
oracle tests supplement that common-source parity. These are helper drivers,
not full training recipes. No new dependency, tolerance/allowlist, lint
suppression or gate weakening is introduced.
The fourth independent native review 5402865547 passed all previous mean
repros and 19,713 broader mean observations but found nested finite sum loss,
L2 error above two ULP, and missed exact clamp faults in sum/l1/L2. Three new
nonignored regressions reproduce those failures at the reviewed head, then
cover nested sum signs/orders/residuals, repeated small squares at every width
1..128 and count seams through 16,385, and exact/inside clamp tails including
minimum subnormals. Another 4,226 row families in three orders contribute
50,712 new four-reduction driver observations. All original asinh/row/mean
corpus objects and old metadata remain unchanged. The complete independent
69,416-case oracle is replayed separately in ten drivers; writer replay is not
a GitHub approval. Full proof and the earlier failed scrub check are retained
under correction-04; no tolerance or fault exception is added.

The next independent review must sweep all available v2 reductions
(sum/l1/mean/l2), asinh, fault thresholds, overflow/cancellation and underflow.
A variance API is absent from this helper prerequisite; full program/reduction
admission, including any variance integration, remains an explicit held gap.

## Per-op definition of done — remaining admission

| Requirement | Evidence/state in this slice |
| --- | --- |
| Shortest existing-tree comparison | Frozen overflowing asinh and cancelling row-sum comparison; broader utility held |
| Six-dataset utility and equal-budget G-A…G-E | Held; no Baseline/promotion claim |
| Runtime/interpreted/compiled/emitted twins | Numerical helpers only; full program twins held |
| Guarded kernel and defined NaN/Inf/zero/faults | Focused boundary/value/fault tests |
| Arity/bounded Experimental prior | No new opcode or generation admission; program gate held |
| Token/byte delta and capability lifecycle | New helper artifacts only; old identities untouched |
| Train-only fit and deterministic width | No fitting or width transformation added; future recipe gate |
| Saturation fault | Huge L2 and finite-bound corpus |
| Removal from generation while keeping parsing | Generation remains unadmitted; policy envelope gate held |
| Append-only IDs/helpers | Separate module and corpus; no existing IDs changed |
| EEC create/change/remove/lossless codec | Held for versioned program/profile integration |
| Rust/Python/C++17 training recipe and fitted C++ | Held; helper proof is supplementary |
| IP scrub and no new feat-math dependency | Focused IP scrub passes; Cargo files untouched |
| Version-envelope/negative input/frozen v1 | Negative/helper/frozen v1 proof; policy envelope remains held |

Full standing after-gates, frozen-input/skip-name/Clippy ratchets, read-only
lane audit, exact-head automatic plus full manual CI, independent native
approval and guarded rebase merge remain publication gates. Continue B07
program/model/recipe/generation/EEC work before advancing the card to B08.
