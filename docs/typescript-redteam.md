# TypeScript red-team qualification

This review continues PR #412. The original implementation was source-only: its
hosted jobs had not executed. The follow-up adds executable coverage and keeps the
PR draft until its required checks pass.

## Reproduced findings and repairs

| Finding | Reproduction and repair |
| --- | --- |
| Sparse scoring targets produced `NaN` | `score([[0],[1]], new Array(2))` passed `Array.some` validation. Iteration now rejects holes and non-number/nonfinite labels in both fitting and scoring. |
| Sparse string rows could become missing CSV cells | `Array.some` and `Array.map` skipped holes. Every raw cell is now validated by iteration before CSV construction. |
| Disposal traps left the facade alive | Disposal now invalidates in `finally`; a throwing kernel cannot be called again through the same facade. Fault injection covers this case. |
| A multiline title escaped the inherited Rust documentation heading | A CR/LF/Unicode-separator fixture failed Wasm compilation. TypeScript export now folds title whitespace to a single line before calling the existing emitter, and executes the corrected model. Numerical Rust/Python templates remain unchanged. |
| A minimum execution count allowed missing operators | Qualification now enumerates the real publication registry and records each executed/refused identity. Missing records, unequal module executions and zero tests fail. |
| TypeScript qualification did not block the main CI aggregate | The normal CI workflow now calls the Node matrix and requires its complete result. Workflow/tool changes select the gate. |
| The pipeline executable-fixture target did not compile | Its exhaustive target match omitted TypeScript. The fixture now executes the verified TypeScript/Wasm bundle in both module modes; qualification also checks all targets of feat-math, jope-cli and jope-pipeline. |
| Browser support had no actual-browser test | Every census fixture is now required in Chromium, Firefox and WebKit, in a page and module worker. |
| Explicit scalar/row-vector APIs lacked the target | `typescript::emit_scalar` and `typescript::emit_row_vector` reuse the existing validated Rust recipes. |

The numerical Rust/Python templates are unchanged. Fitted TypeScript intentionally
inherits the bound Rust emitter's refusals; fitted portfolio is one such refusal.
A configured registry target can describe broader intent than the currently
admitted fitted state, so both successful execution and inherited refusal are
recorded rather than calling a refusal a positive prediction test.

## Coverage interpretation

The facade test substitutes only the emitter's header marker before strict
compilation. Recipe/fitted bindings run the same compiled template body in isolated
VM contexts. ABI doubles inject missing exports, incorrect versions, imports,
allocation refusal, memory growth and traps, and force both byte orders. V8 line,
branch and function coverage each have a 100% threshold. Seven top-level Node
contracts also include negative qualification-receipt checks. These metrics do not
measure numerical Rust code. Real Rust-compiled Wasm tests provide the separate
numerical evidence.

Registry tests cover every Rust-supported legacy recipe spelling, every declared
`rust_bound` entry, and fitted legacy states. Scalar tests discover every decodable
function ID and combine both numerical policies with both carriers. The complete
existing golden suite retains its interpreter/Python/native-Rust comparisons.
TypeScript additionally runs their generated Wasm in strict CommonJS and ESM.
Arithmetic tolerance remains `1e-8 * (1 + abs(reference))`, with finite results
required. It is not a universal bit-parity claim.

## Evidence and acceptance

Run `bash deploy/ci/typescript/qualify.sh` after installing the locked compiler and
Rust Wasm target, then install the pinned Playwright engines and run
`node deploy/ci/typescript/browser.mjs`. Evidence is retained in
`target/typescript-evidence/`: toolchains, source/run/attempt, each Rust suite,
V8 coverage, census identities and browser/worker cases. The script checks nonzero
execution, complete census accounting and the actual covered facade filename.

The reusable CI workflow tests Node 24.19.0 and 26.1.0. The Node 26 leg requires all
three browser engines. Workflow permissions are read-only, checkout credentials are
not persisted, actions and dependencies are pinned, npm lifecycle scripts are
disabled, and fixture serving binds only loopback within the generated corpus.
The existing aggregate gates and numerical templates are retained.

Local results and hosted status are recorded in the PR's head-bound review receipt.
Missing or blocked browser downloads, queued jobs and unexecuted suites remain
unqualified. No merge or deployment is part of this review.
