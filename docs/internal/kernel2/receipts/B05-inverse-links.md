# B05 inverse-link kernels — numerical prerequisite slice

The approved probit contract is Acklam's rational approximation with relative
error at most 1.2e-9 and no erfc refinement. This owner decision overrides the
plan's generic two-ULP target for probit. A refined higher-precision variant may
be a later versioned card. Historical Blom and its 1e-12-clamped helper remain
byte-identical; the new kernel projects finite probabilities to [2^-53,1-2^-53].
The owner also froze the certified central logit form at at most1ULP against
the independent high-precision corpus, explicitly overriding the plan's
literal-formula requirement. For q in[.25,.75], d=(q-.5)*2 and the result is
log1p(d)-log1p(-d); tails use log(q)-log1p(-q). All three helpers use the same
branch and operation order. Existing literal/clamped paths remain unchanged.
Nonfinite inputs yield positive zero with NonfiniteInput (Domain). Finite
projection reports ProbabilityClamped (Saturation), appended at fault bit16;
all bits, names and severities0-15 remain unchanged.

`kernel::inverse_links::InverseLink` is separate from legacy `math_expr::Func`.
Its ID adapter refuses versions other than2 and IDs other than19/20. The B04
rows remain reservation-only/nongeneratable. Existing parsers, program/model
codecs, search profiles, seeded streams and recipe helpers are unchanged.

Acceptance:

```sh
cargo test --locked -p feat-math --test inverse_links_v2
cargo test --locked -p feat-math --test fault_trace
cargo clippy --locked -p feat-math --all-targets -- -D warnings
```

The new90-digit mpmath1.3.0 oracle fixture has1128 inputs. It preserves binary64
inputs and correctly rounded reference bits, covering both branch thresholds
and adjacent bits, endpoints and adjacent bits, a probability grid, powers of
two, signed zero, extreme finite values, subnormals, infinities and NaN payloads.
The generator is outside the checkout; it adds no product dependency and does
not regenerate any existing fixture. A separate10,001-point grid checks order.

`emit_helpers` returns numerical helper twins, each reporting value and fault
bits. Tests compile/run Rust and C++17/C++20 with g++/clang++ at O2/O3 and run
Python. All2256 value/fault rows per backend are compared exactly on the same host, with C++
contraction disabled; this does not certify cross-host PlatformLibm parity.
These helpers are not complete train-and-predict recipes or fitted artifacts.

## Current tests-first state

Missing-API red proof and failed literal-logit oracle logs are retained. The
literal formula first failed2ULP atq=.444; an independent1128-case diagnostic
found53 failures, maximum41ULP atq=.499. The owner then approved the certified
central form. All seven focused tests now pass, including the new1ULP oracle,
probit reference bound, guard/refusal, dense-grid and helper value/fault parity.
Logit qualification counts binary64 representable steps with matching signs
and exact positive zero, rather than using a tolerance at binade boundaries.
Probit uses the approved raw rational approximation throughout, with no erfc.
The unchanged historical/compatibility golden suite passes16/16; feat-math
Clippy passes. Full after-gates, lane audit and exact-head CI/review are pending.
The initial frozen candidate failed hygiene on an undeclared test target and
audit on a generic Language enum collision. The corrected source adds only
the required static target mapping and uses InverseLinkLanguage; failed
evidence is retained outside the checkout, with no gate weakening.
The next qualification exposed four jope-records failures because the new
coverage target also needs a maintained regression classification. The
correction adds only its gp/export classification; all existing rows and
baseline triage stay unchanged. Full suites must qualify the corrected head.

## Per-op definition of done and remaining B05 gates

| Requirement | This slice / remaining work |
| --- | --- |
| Shortest existing-tree comparison | Existing logit compositions and Blom helper identified; expression-node comparison pending |
| At least6-dataset utility | Pending; no utility or lift claim |
| Runtime / interpreted / compiled / emitted twins | Numerical runtime and compiled helper twins; AST and full recipes pending |
| Guarded kernel | Finite projection and nonfinite fallback covered |
| NaN / Inf / zero / faults | Boundary corpus and exact fault-bit twin tests |
| Arity | Unary typed numerical API; parser/codec arity integration pending |
| Bounded Experimental prior | Generation disabled; future prior must be bounded and explicit |
| Equal-budget G-A..G-E before Baseline | Pending; no promotion |
| Token / byte delta | Full AST/artifact measurement pending |
| Lifecycle | Reservation-only, nongeneratable; no Baseline claim |
| Train-only fit | Stateless kernel; eventual training wrappers must retain fold closure |
| Deterministic width | Unary scalar; row/width integration pending |
| Saturation fault | Appended ProbabilityClamped bit16 |
| Disable generation without removing parsing | Existing profiles unchanged; versioned ID adapter retains two IDs |
| Append-only IDs/helpers | Separate v2 functions19/20 and new helper names; old helpers unchanged |
| EEC Rust/Python/C++ recipes + fitted | Pending expression/search/full artifact integration |
| IP scrub | New helper source checks shipped-text lint |
| No new feat-math dependency | Cargo manifests/lock unchanged |
| Envelope / freeze / empty-row negatives | Typed ID version refusal now; full program/model/row tests pending |

This is the first coherent numerical prerequisite of B05, not B05 completion.
Do not activate19/20 in shipping profiles until versioned expression/model
integration, full recipes/fitted execution and EEC witnesses all qualify.
Durable execution/qualification/CI/review receipt:
`/home/ubuntu/jope-evidence/kernel2/bt/B05/implementation-01/receipt.md`.
