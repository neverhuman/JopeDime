# B07 portable stable exponential numerical prerequisite

## Scope and contract

This is a small remaining prerequisite of B07, following the B06-specified
Func21 `expm1` / Func22 `log1mexp` contract. These are version-two standalone
numerical helpers. Existing scalar-program whole-AST refusal of Func21/22 stays
in place; this PR does not admit models, fitting, complete recipes, search,
generation, EEC, utility promotion or B08.

Nonfinite input returns +0 with NONFINITE_INPUT. Finite expm1 arguments project
to [-30,30] with EXP_ARG_CLAMPED; output projects to +/-1e12 with
MAGNITUDE_CLAMPED. Log1mexp requires x<0, including negative subnormals; x>=0
returns +0 with LOG_DOMAIN. The threshold bits are bfe62e42fefa39ef: below it,
portable log1p(-exp(x)); otherwise portable ln(-expm1(x)). Valid finite tails
retain -0. Unknown version/function IDs refuse before input access.

Existing portable_math and its Python/C++ arithmetic bodies are reused without
edits or new libm calls. Helper emission embeds the immutable exp/log-family
prefix at compile time; no fixture file is read at runtime. New standalone
helpers return both value and fault bits. Old emitter entry points never select
them. Frozen/default v1 and numerical-policy kernels remain byte-identical.

## Verification

Acceptance: `cargo test -p feat-math --locked --test stable_exponential_v2`.
Tests-first missing-module E0432 is preserved as an API-absence RED, explicitly
not a runtime regression. An initial implementation used unsupported FaultSet
bitwise operators; that compilation attempt is retained and corrected using the
existing union API without modifying fault types.

Eight new, nonignored tests cover closed versioned IDs/legacy admission,
nonfinite/signed-zero guards, distinct projection faults, branch/subnormal/tail
behavior, both independent numerical references, dense monotonic grids and
compiled helper parity. The fixture has3048 cases from1524 distinct bit inputs.
Python decimal450 and900-digit independent oracles converge to identical rounded
bits and fault masks in every case. Normal reference bound is2ULP; zero/subnormal
absolute bound is one minimum-subnormal step, with guard/fault/sign contracts
checked exactly. For x<-1000, -ln(1-exp(x)) has magnitude<2*exp(-1000)<2^-1075,
so its correctly rounded binary64 value is -0. No B05 exception or old allowlist
is inherited. Exact same-host output/fault parity covers Rust -O, Python, g++ and
clang++ C++17 O2/O3 and C++20 O3 with contraction disabled/no fast math, plus
g++ C++17 O2 ASan/UBSan.

Full fresh baseline on4eb555be: PR2083/29, main4163/66, Python231/2, golden16/16,
fitted sanitizer18/18 and C++2018/18, strict existing O3 plus added O2 emitters.
Expected after totals PR2091/29 and main4171/66, exactly8 additions; old selected,
skipped and excluded names must remain unchanged. After gates/audit/hosted CI
and fresh independent native approval are separate pending lifecycle gates.
Actual `just required`101 is the owner-accepted81-location held design floor,
NOTPASS; normalized workspace Clippy83occurrences/59unique is a zero-new ratchet,
owned feat-math lint must pass. Protected93 paths and38 evaluator identity
inputs, old fixture/template bytes and prior numerical-policy core/twins must
remain unchanged. No regeneration.

Durable evidence: `/home/ubuntu/jope-evidence/kernel2/bt/B07_STABLE_EXPONENTIAL_KERNELS`.
Oracle generation: `generate-oracle.py` there; its SHA-256 is embedded in the
new fixture. Complete commands, environment, times and exits are preserved.
