# JopeDime Release Specs — Engineering Review and Scorecard

**Reviewed:** October 29, 2026
**Prepared for:** Jepson Taylor / VEOX
**Documents reviewed**

| Label | File | Size | Self-described scope |
| --- | --- | --- | --- |
| **Spec A** | `docs/product/drafts/spec-a-enterprise-mvp.md` | ~450 lines | "Enterprise MVP", "Brilliantly Simple, Enterprise-Solid" |
| **Spec B** | `docs/product/drafts/spec-b-production.md` | ~1,700 lines | Full production release specification |

**Method.** Both specs were read in full and scored against an 11-criterion weighted rubric (100 points). Their factual claims were checked against fresh clones of both repositories, the resolved `cargo tree` of the `jopedime` appliance, and the current MCP specification. Where a claim could not be checked (for example, private host-custody files), that is stated.

**Repositories as verified**

| Item | Value | Note |
| --- | --- | --- |
| JopeDime `main` | `58f2bfb093a8ea30034a0e5a0297bc3e66054e35` (Oct 29, 18:07 MDT) | Both specs audited `c1318a7b`. Three telemetry commits (#454–#456) have landed since; `collectors.rs` is unchanged, so the CPU finding still holds. |
| DOPE `main` | `5c2e8df8e424c31087a4d8284231f17157319399` | Matches both specs. |
| `dope-core` pin used by the build | `15e12aa3dc33851655580f6aa7c9c66353143f23` (tag `dope-engine-v1.1.0`) | `crates/jope-encoders/Cargo.toml:40`. Some docs still mention `25d1785c` (unmerged branch pin); the Cargo pin is authoritative. |
| New requirement since both specs were written | API-first: monorepo CLI, a later lightweight CLI talking to an API (localhost first, cloud later), still one downloadable binary | Neither spec was written for this. Criterion 4 scores against it explicitly. |

---

## 1. Verdict

| | Spec A (Enterprise MVP) | Spec B (Production Release) |
| --- | --- | --- |
| **Total** | **63 / 100** | **71 / 100** |
| Grade | C+ | B− |
| One-line verdict | The right *shape* for an MVP, but several load-bearing claims are wrong or unverified, and it forbids the daemon/API topology you now need. | The most *correct* document, with excellent evidence discipline and UX/agent depth, but it is a program plan for a mature product, not an MVP. |
| Best thing in it | Five-digest identity model, six-state run machine, 10-tool MCP catalog, 6 vertical PRs with exit gates. | Evidence boundaries, external-closure honesty, claim levels for reasoning, metric definitions, terminal editor details, snapshot/watermark replay protocol. |
| Worst thing in it | Over-claims: TypeScript export, binary/multiclass ranker qualification, "redb already in the closure", an in-binary "AST/bytecode verifier". Rejects `serve`. | 12+ identities, 12 run states, 25 MCP tools, 3 deployment modes, 20 work packages, ~50 release tests. The MVP is buried. |
| Use it for | Skeleton and economy of the final spec. | Correctness, guardrails and test intent of the final spec. |

Neither spec is ready to hand to implementation agents. Both missed the true critical path (Section 4): the h40b ranker's serving inputs depend on host-custody files that exist in neither repository, and the appliance already links a pure-Rust GP engine that both plans route around.

---

## 2. Rubric

Weights reflect the stated priorities: correct DOPE/fused-model integration, a single 100% Rust binary, world-class human and agent experience, and "fight over-engineering; reward brilliance in simplicity without loss of function."

| # | Criterion | Weight | What a full score requires |
| --- | --- | --- | --- |
| 1 | Repository grounding and factual accuracy | 10 | Every claim about current code is true at the audited revision; unverified items are labeled; no invented capabilities. |
| 2 | Core ML integration (DOPE pin, fused serving contract, landmarks, flywheel boundary) | 15 | The exact trained graph, its real inputs, its training distribution, the frozen-DOPE invariant, train/serve parity and promotion rules are specified correctly. |
| 3 | Architecture simplicity (anti-over-engineering) | 15 | Fewest concepts that deliver full function; each mechanism justified by a real failure mode; reuse of existing code over new code. |
| 4 | Deployment topology fit (single binary + API + thin client, localhost → cloud) | 10 | One binary; a versioned API that the TUI/CLI/MCP all use; works in-process, as a local service and later remotely, without a rewrite. |
| 5 | Human CLI/TUI experience | 10 | A first-run path, inline transcript, backslash commands with completion, editing under load, color/accessibility, exit and interrupt semantics. |
| 6 | Agent experience (MCP and machine CLI) | 10 | Small typed tool catalog, idempotency, cursors, long-poll or subscriptions, stable errors, no TTY scraping, named-client setup. |
| 7 | Run lifecycle, persistence and recovery | 8 | Durable states, pause/continue/cancel/restart, crash consistency, idempotent retries, realistic determinism. |
| 8 | Telemetry honesty and GP abstraction in reasoning | 7 | Correct units and denominators, null-not-zero, no fabricated token rates, redaction by construction, evidence-bound claims. |
| 9 | Exports and target languages | 5 | Faithful fitted predictors including preprocessing and calibration, honest coverage, constraints before search, atomic multi-language bundles. |
| 10 | Qualification and release gates | 5 | Crisp, testable pass criteria bound to the exact shipped artifact; covers the real failure modes without padding. |
| 11 | Executability (work plan, sequencing, owner decisions) | 5 | Ordered, mergeable increments that respect the repo's one-PR-at-a-time rule; explicit decisions with defaults; earliest usable product. |
| | **Total** | **100** | |

---

## 3. Scores and feedback by criterion

| # | Criterion | Wt | A | B |
| --- | --- | --- | --- | --- |
| 1 | Grounding and accuracy | 10 | 5 | 8 |
| 2 | Core ML integration | 15 | 8 | 11 |
| 3 | Simplicity | 15 | 12 | 5 |
| 4 | Topology fit | 10 | 2 | 5 |
| 5 | Human TUI | 10 | 7 | 9 |
| 6 | Agent experience | 10 | 7 | 8 |
| 7 | Lifecycle and recovery | 8 | 6 | 7 |
| 8 | Telemetry and reasoning | 7 | 6 | 7 |
| 9 | Exports and languages | 5 | 2 | 4 |
| 10 | Qualification | 5 | 4 | 4 |
| 11 | Executability | 5 | 4 | 3 |
| | **Total** | **100** | **63** | **71** |

### 3.1 Repository grounding and factual accuracy — A 5/10, B 8/10

**Both** correctly identified the ten core findings F01–F10. All ten were re-verified (Section 4.1).

**Spec A** inherits those findings but adds claims that are false or unsupported:
- "`redb` is already in the workspace closure." It is in `Cargo.lock` only through `jain-smartcluster-daemon`; it is **not** in the appliance's dependency tree. Adding it is a new appliance dependency (still a good choice, but the justification is wrong).
- The bundle manifest example lists `qualified_tasks: ["regression","binary","multiclass"]` and `trained_objectives` including `logloss`/`brier`. The h40b loop standardizes `y` and trains on regression contexts; nothing in the repo qualifies binary or multiclass learned ranking.
- "Removing `onig_sys` makes `fused-cpu` 100% pure Rust" — directionally right, but it does not identify where `onig_sys` comes from (vendored `candle-core` 0.11 → `tokenizers` with the `onig` feature), so the fix is not actionable as written.
- Four "faithful" export targets including TypeScript. Today GP math-pipeline objects export **JavaScript only**; Python and Rust return "not implemented" (`model_object.rs:376–386`). No TypeScript emitter exists.
- "Embeds weights via aligned read-only section mapping" is presented as design fact without evidence it is needed; the DOPE MLC-v3 bundle is 1.78 MB.

**Spec B** is consistently careful: it separates source pins from weight digests, states what it did not execute, and names the external `JopeDimeMicro` and `cpu_exact_fused.py` dependencies. It loses points for two misses that change the plan: it does not notice that GP math pipelines export only to JavaScript, and it treats a mask-aware DOPE checkpoint purely as future work even though `docs/optional-landmarks.md` records an owner "binding mission addendum" requiring one. That conflict needs an owner decision, not silence.

### 3.2 Core ML integration — A 8/15, B 11/15

**Spec A strengths.** It commits to the h40b candidate-cost ranker, insists on golden parity fixtures, states that the model ranks rather than generates tokens, and gives a clean score-comparability invariant (landmark updates change what is proposed, never how it is scored). The surrogate-mismatch guard is a good idea.

**Spec A gaps.**
- It never states the ranker's real inputs. `adapter_10tok.predict_costs` takes `revealed_raw_f`, `revealed_gate` and `revealed_indices` (measured fitness of up to four slate members), `prog_412` and `proj_16` per candidate, `mm_tokens_10x32` and `mm_mask_10`, plus fixed `scale=0.4208` and zeroed `ctx_feat`/`base_prior_cost`/`sos_prior` (`run_fused_closed_loop.py:351–365`). This is an **in-context** ranker. A serving loop that does not supply revealed measured fitness is serving a different model.
- The progressive design starts every run at `mask=[false;23]`. h40b was trained only on full ("high") landmark extraction (`fit_transform_and_extract_bundle_high`, `use_landmarks=True`), so revision 0 is out of distribution. The spec presents this as the default happy path.
- No mention of the candidate featurizer (DSL → 412 features), the champion bank, or the proposal operators the head was trained against. Those are part of the serving contract.

**Spec B strengths.** Explicit architecture IDs, external closure resolution ("do not reconstruct the missing external network from call signatures"), structured mask-dropout training, `surrogate_target_recipe`, selection-bias-aware flywheel records, causal-wiring *and* usefulness tests, and a promotion ratchet with paired analysis.

**Spec B gaps.** It mentions the "candidate-feature recipe" but does not identify that the featurizer lives in a host-custody file (`/home/ubuntu/jope-evidence/.../rank_prepare.py`) or that the evaluator is a SHA-pinned external binary (`bootstrap.s1main`). It does not mention the revealed-fitness inputs or the champion bank. It does not notice that `dope-core` already provides the MLC-v3 encoder in Rust, which shrinks the port to the downstream pieces only.

### 3.3 Architecture simplicity — A 12/15, B 5/15

**Spec A** is the better MVP instinct by a wide margin. Five digests instead of twelve-plus identities; six run states; ten MCP tools; one in-process controller; `redb` instead of a custom journal; same-executable workers; six vertical PRs. The "cycle-safe crate topology" note is useful. It loses points for complexity in the wrong places (two MCP handshakes on day one, four export languages, a lock-free progressive watch channel for a head that was not trained on partial masks) and for simplifying away something the product needs (a local service).

**Spec B** contains very little that is wrong, but it does not choose. Twelve independent identities, a twelve-state lifecycle with `created/validating/ready/…/interrupted`, three deployment modes, twenty-five MCP tools, a resource-lease system for independent MCP children, a Tasks-extension mapping, modern and legacy MCP adapters, five release profiles, five platform rows, BF16/quantized policy and a plugin-ABI discussion. Each is defensible in isolation; together they hide the MVP and multiply review and test cost. Rewarding "brilliance in simplicity" means the score has to be low here.

### 3.4 Deployment topology fit — A 2/10, B 5/10

Scored against today's requirement (single binary, API-first, localhost now, cloud later, a lightweight CLI later). Both specs predate it, so this criterion measures how much rework each would need.

**Spec A** explicitly forbids a background service ("Do not … spawn background OS daemons (`serve --local`)"), exposes MCP over stdio only, and has no network API. The run controller is reachable only from inside the process. Adopting A would require redesigning every client surface.

**Spec B** has a "managed local service mode" (`jopedime serve --local`, `attach`, owner-restricted socket/pipe), Streamable HTTP for MCP, and an authorization section for remote HTTP. That is a real head start. But the API is not the primary contract; clients talk to "the command service" through several transports (in-process, local socket/named pipe, MCP child, HTTP), and there is no thin-client concept or a single client library that switches between local and remote.

Neither spec noticed that the existing `jopedime api` is JSON-RPC over **stdin/stdout**, sharing one dispatcher with MCP (`service.rs:1–6`). That "one dispatcher, several framings" pattern is the right seed for the API-first design.

### 3.5 Human TUI — A 7/10, B 9/10

**Spec A:** correct choice of Ratatui `Viewport::Inline`, scrollback preservation, a compact status bar, `NO_COLOR`/`TERM=dumb`, panic and signal restoration, and good Ctrl-C semantics (clear draft → pause → cancel). Missing: a first-run flow (what happens after `\open` when no target is given), target picking, path completion, history, multi-line input, a color system, what a transcript card looks like, and detach/attach.

**Spec B:** the strongest UX section of the two — information hierarchy, a semantic palette with fallbacks, Windows drive paths not being parsed as commands, bracketed paste, history privacy, live editing under event load, a PTY test matrix, and an honest stance on natural-language input. It loses a point for size (29 commands) and for leaving the first-run golden path implicit.

### 3.6 Agent experience — A 7/10, B 8/10

**Spec A:** ten well-chosen tools, idempotency keys, `expected_revision`, cursor replay, `structuredContent`, stdout purity. Missing: a cheap "wait until something changes" primitive (agents will otherwise poll in tight loops), dataset upload for a non-local server, a stable error schema, and named-client setup.

**Spec B:** excellent protocol detail — the snapshot/watermark/subscription race protocol, idempotency scoped to principal and workspace, protocol vs. tool errors, decimal strings for 64-bit sequences, named-client acceptance scenarios. Twenty-five tools is too many for an agent to choose among well; several are thin variants (`run_pause`, `run_resume`, `run_cancel`, `run_set_effort`) that A correctly folds into `run_control`.

Note on facts: MCP `2026-07-28` is real and stateless (per-request `_meta`, `server/discover`, `subscriptions/listen`, Tasks extension, no SSE resumability). Spec B calls `server/discover` mandatory; public summaries describe it as optional. Both specs correctly push durable replay into application-level cursors.

### 3.7 Lifecycle and recovery — A 6/8, B 7/8

**Spec A:** six states, automatic interrupted → `Paused` recovery under a file lock, artifact-before-commit publication order, worker kill on timeout/OOM. Gate R01 ("resumed run finishes with the exact same champion") is unattainable as written while budgets are wall-clock and workers run in parallel; determinism needs an evaluation-count budget and ordered result application.

**Spec B:** the deepest treatment — execution epochs and attempt fencing, accepted/applied/completed command stages, prediction-stream resume, cancellation quiescence. Twelve states and three revision counters are more than an MVP needs.

### 3.8 Telemetry and reasoning — A 6/7, B 7/7

Both fix the CPU denominator correctly (verified: `collectors.rs:150` divides process ticks by per-core ticks, then clamps to 100 while its comment says "100% means all node CPUs busy"). Both refuse to invent a neural token rate for a ranker and both require redaction by construction. **Spec B** adds the claim-level table (hypothesis → active → measured → selected → deployment-ready), sentinel-injection tests and precise metric definitions; it earns full marks. **Spec A** is concise and correct but thinner on how GP edits become plain-language explanations.

### 3.9 Exports and languages — A 2/5, B 4/5

**Spec A** promises Rust, Python, JavaScript and TypeScript as "faithful" v1 targets and an in-binary "AST/bytecode verifier" that evaluates generated source on 32 rows. An in-binary check can only re-evaluate the intermediate representation; it proves nothing about the emitted Python or JavaScript text. Real parity requires running the target toolchains in CI. It also does not account for the current coverage gaps (math pipelines JS-only; calibration, feature ops, target mapping and most missing-aware heads refused by `fitted_portable_unsupported`).

**Spec B** is honest that coverage is uneven, separates algorithm source / fitted predictor / model object, defines `require-all` vs `best-effort`, retained-training-data disclosure and atomic bundles. It misses that fitted GP math pipelines export only to JavaScript (the Python and Rust recipe emitters exist; the fitted-state glue does not), which is the biggest single export task.

### 3.10 Qualification — A 4/5, B 4/5

**Spec A**'s 28 gates are crisp and mostly measurable; a few are mis-specified (R01 determinism, E02 claiming TypeScript, Q01 without a defined baseline or margin). **Spec B**'s ~50 tests cover more real failure modes, with the right principle ("skipped and zero tests cannot count as passed"), but volume dilutes priority and several pass criteria are qualitative.

### 3.11 Executability — A 4/5, B 3/5

**Spec A**'s six vertical PRs respect the repository's one-implementation-PR-at-a-time rule (`AGENTS.md:136–138`) and each ends in named gates. Two problems: PR 1 cannot start until the external h40b closure is in the repo (not scheduled anywhere), and PR 2 includes Q01, a model-quality study, inside a code PR.

**Spec B**'s twenty work packages have excellent handoff requirements, but twenty serial review cycles is slow, and the first usable product arrives only at milestone M3. Its decision table with defaults is strong.

---

## 4. What both specs missed

### 4.1 Re-verification of the drafts' findings

| ID | Finding (from drafts) | Status at `58f2bfb0` | Evidence |
| --- | --- | --- | --- |
| F01 | Research h40b and Rust policy are different graphs | Confirmed | `run_fused_closed_loop.py:237` (`FusedJopeNetwork(adapter_variant="h40b")`); `jope_policy.rs` |
| F02 | Landmark sidecars do not condition search | Confirmed | `landmark_encoding.rs:1` — "the search receives its original candidates and seeds" |
| F03 | Legacy `full` profile blocked | Confirmed | `models.lock.json`: `full` → `internal-restricted`, `blocked` (prohibited asset; neural adapters not compiled) |
| F04 | Narration exposes GP vocabulary | Confirmed (partially) | `render/v3.rs:57` uses "generations" |
| F05 | Three separate MCP dispatchers | Confirmed | `service.rs` (`jopedime-regression`), `model.rs:476` (`jopedime-models`, no init lifecycle enforced), `narration/mcp.rs` (`jope-search-commentary`) |
| F06 | CPU process % uses per-core denominator and clamps | Confirmed, unchanged after #454–#456 | `collectors.rs:149–155` |
| F07 | Custom fitness not uniform across search modes | Not re-verified in code; consistent with `docs/regression-objectives.md` | — |
| F08 | GPU presence ≠ all-GPU path | Confirmed by docs | `docs/optional-landmarks.md` ("Decoder and reward execution remain CPU-only") |
| F09 | Freeze guard skips missing normalizer buffers | Confirmed | `dope_freeze_guard.py:183–193` (`if buf_name in net_buffers:`) |
| F10 | Source export drops/refuses calibration | Confirmed, and broader (see N6) | `fitted_portable.rs:55–62`; `model_object.rs:350` |

### 4.2 New findings that change the plan

| ID | Finding | Evidence | Consequence |
| --- | --- | --- | --- |
| **N1** | **The h40b serving path depends on files outside both repositories.** The fused network loads from `/home/ubuntu/JopeDimeMicro`; `cpu_exact_fused.py` is SHA-pinned (`b9f1468a…`) and passed in by path; the DSL→412 featurizer requires host custody `/home/ubuntu/jope-evidence/dope-lift-20261021/corrected-study/rank_prepare.py` plus `program-feature-names.json`; the CV evaluator is a SHA-pinned binary `bootstrap.s1main` (`e3be9386…`) from a receipts directory; the champion bank is `champions.json`. | `run_fused_closed_loop.py:36–40, 222–237`; `rust_evaluator.py:49–51, 190–200` | This is the true P0. No Rust port, parity test or bundle can be produced until these are committed (or vendored into a pinned private repo) and reproducible from a clean checkout. Neither spec schedules it. |
| **N2** | **h40b is an in-context ranker.** Its inputs include the measured fitness of four revealed programs. In 192-wide training slates, `training_indices` puts the best 48 first, so the revealed four are the top four; at proposal time the loop ranks `4 + n` candidates, prepending the pool's first four entries in storage order. Normally 8 proposals are ranked and 4 evaluated; after a promotion drought, 16 are generated and 2 of the 4 picks are random. Candidates are measured with full-training-table CV (`full_train_cv`). | `run_fused_closed_loop.py:178, 320–365, 900–920`; `automatic_promotion.py:160–170`; `dev_incumbent.py:165–182` | The faithful serving loop ranks `4 + 8` (or `4 + 16`) candidates and evaluates 4. Revealing already-measured programs costs no extra evaluations; which four to reveal needs an owner decision after the custody work. |
| **N3** | **The product already links a pure-Rust GP engine.** `feat-math::evolve` is a generational GP over the same `(pipe X …)` DSL with param-jitter/insert/drop/swap/head-swap mutations, splice crossover, honest k-fold CV, rayon parallelism and a memo cache. The appliance depends on `feat-math` (with `default-features = false` to exclude the C++ runtime). | `feat-math/src/evolve.rs:1–9`; `jope-runtime/Cargo.toml:17` | Discovery should be built on `feat-math`, not on `jope-pipeline`. Both drafts' PR plans edit `jope-pipeline/src/run.rs`. |
| **N4** | **`jope-pipeline` is not appliance-safe.** It is not in the appliance's dependency graph and launches `docker`, `systemctl`, `sudo`, `/bin/sh` and `git`. | `execution.rs:1423, 1774, 2241`; `supervisor.rs:129–131`; `run.rs:712` | Do not link it into the product. |
| **N5** | **h40b was trained only on full landmarks;** the mask-aware DOPE candidate is unqualified; the repo records an owner addendum requiring one. | `run_fused_closed_loop.py:270–283`; `docs/optional-landmarks.md:73–77, 145` | v1 serves `full` landmarks only (bounded FIT subsample). Progressive and `none` stay behind the same flag, refused until a mask-aware head qualifies. Owner decision required on which DOPE checkpoint is the v1 pin. |
| **N6** | **Export coverage is narrower than either draft assumed, but the gap is glue, not emitters.** `feat-math` already emits Python, Rust, JavaScript, C++, Go and a Rust-backed TypeScript bridge, yet fitted GP math-pipeline objects export only JavaScript because only `fitted_js.rs` glue exists. Native portable export refuses feature ops, calibration, target mapping, row order and most missing-aware heads. Fitted pipelines also predict through an `f32` cast that maps non-finite values to `0.0`. | `model_object.rs:376–386`; `fitted_portable.rs:48–62`; `feat-math/src/transpile.rs:53`; `jope-runtime/src/math_pipeline.rs:202–207` | Python and Rust fitted-state glue plus calibration and an `f64` path are the largest export items. TypeScript should ship as a `.d.ts` over the JS module, labeled as such. |
| **N7** | **Native C in the CPU appliance closure comes from vendored `candle-core` 0.11**, which depends on `tokenizers` with `onig` (→ `onig_sys`, C) and `esaxx-rs`, used only by `candle-core/src/quantized/tokenizer.rs`. | `cargo tree -p jope-appliance --features dope-encoding -i tokenizers`; `vendor/security/candle-core-0.11.0/Cargo.toml:273–276` | One-line fix in the already-vendored manifest: make `tokenizers` optional and off. |
| **N8** | **The DOPE encoder half is already in Rust.** `dope_core::mlc_v3::MlcV3Model` provides `([B,256],[B,64,17]) → [B,629]` on CPU and CUDA. Only `lm48` projection, stage-1B tokens, `adapter_10tok` and the featurizer need porting. The DOPE MLC-v3 bundle is 1.78 MB. | DOPE `docs/jopedime-integration.md:236–262, 279–283`; `deploy/appliance/dope-bundles/dope-mlc-v3.json` | The Rust port is smaller than both drafts implied, and embedding weights in the binary is trivial. |
| **N9** | **No network API exists.** `jopedime api` is JSON-RPC over stdin/stdout through the MCP dispatcher. The appliance has no `tokio` or `axum`; `axum` exists in the workspace only via `jope-observatory`. | `cli.rs:77–80`; `service.rs:1–6`; appliance `cargo tree` | API-first is new work, but small: one router, reused by every client. |
| **N10** | **`jope-telemetry` pulls `arrow` and `parquet` unconditionally,** and its CPU collector is Linux `/proc` only. | `jope-telemetry/Cargo.toml`; `collectors.rs:185–242` | Make the Parquet sink an optional feature for the appliance; have workers self-report CPU time and peak RSS so no cross-process sampling is needed on macOS/Windows. |
| **N11** | **The crate cycle disappears with a new top-level crate.** `starforge` (jope-encoders) depends on `jope-runtime`. Putting inference in a new `jope-service` crate that depends on both removes the cycle without a trait-injection workaround. | `crates/jope-encoders/Cargo.toml` | Simpler than either draft's resolution. |
| **N12** | **The repo's coding standard gate already exists.** `justfile:28–29` installs and runs Jankurai. | `justfile` | Release gates should require no Jankurai regression alongside `just required`. |

---

## 5. What to keep, change and drop

| Topic | Keep from A | Keep from B | Final decision |
| --- | --- | --- | --- |
| Identities | Five digests | — | Four content digests (`Bundle`, `Problem`, `Conditioning`, `Model`) plus `RunId` + `revision` |
| Run states | Six states | Execution fencing, accepted vs completed commands | Six states; interrupted folds into `paused{interrupted}`; attempt IDs fence late results |
| Storage | `redb` + content-addressed artifacts, publish-before-commit | Never rewrite config silently; no auto-delete of user data | Same, with a user-level store, `JOPEDIME_HOME` override, idempotency committed atomically with each effect, and events published only after commit |
| Process model | Same-executable `__worker` | Parent-death cleanup tests | Same; the parent samples worker CPU and RSS and enforces limits; workers memory-map a dataset matrix parsed once |
| Deployment | — | Local service mode | **API-first**: one HTTP+JSON+SSE API; the terminal and MCP share one auto-started background engine (idle exit); batch commands run the same router in-process when no engine is running; remote later; lite client = same crate without the `engine` feature |
| Ranker | h40b, golden parity, surrogate guard | External closure, architecture IDs, promotion ratchet | Faithful h40b loop (`4 + 8` slates, 4 evaluated, exploration when stalled); full landmarks only in v1; featurizer and bank are bundle assets |
| Search engine | — | — | **Build on `feat-math::evolve`**; never link `jope-pipeline` |
| MCP | 10 tools, `run_control` | Replay protocol, error contract, named-client tests | 12 tools including `run_status(wait_seconds, after_event)` long-poll; MCP is a client of the API; HTTP MCP deferred |
| Languages | Constrain search to the intersection | Honest coverage, `require-all` | Rust, Python, JavaScript (+ `.d.ts`); parity proven in CI with real toolchains and decision-flip rates; no in-binary "verifier" |
| TUI | Inline viewport, Ctrl-C ladder | Palette, editor behavior, PTY matrix | Combined, plus a defined first-run path; runs live in the background engine, so closing the terminal never stops them |
| Tests | 28 crisp gates | Test intent and anti-gaming rules | 28 release gates plus one capability gate (learned guidance), each with a numeric or binary pass criterion |
| Plan | 6 vertical PRs | Handoff template, decisions with defaults | 1 research custody track + 8 PRs; a usable native-discovery product after PR 2 |
| Drop | "AST/bytecode verifier"; progressive watch channel in v1; TS as a native emitter | 12 states; 25 tools; 3 modes; resource leases; Tasks mapping in v1; plugin ABI; BF16 policy | — |

---

## 6. Sources

Repositories: [neverhuman/jopedime](https://github.com/neverhuman/jopedime) at `58f2bfb0`; [neverhuman/dope](https://github.com/neverhuman/dope) at `5c2e8df8`.

External: [Composio — MCP 2026-07-28 update](https://composio.dev/content/mcp-2026-07-28-update-statelessness-apps-auth); [GitLab work item on MCP 2026-07-28 support](https://gitlab.com/gitlab-org/gitlab/-/work_items/627825); [rmcp 3.1.0 README](https://docs.rs/crate/rmcp/3.1.0/source/README.md); [Claude Code MCP docs](https://code.claude.com/docs/en/mcp).
