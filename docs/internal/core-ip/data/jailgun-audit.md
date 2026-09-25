# JopeDime / Hyperion Core IP Pipeline — Jailgun Battletest

**Run:** `run-6206c0d6-52ea-4031-94b8-f3794c1f3573` · **Date:** 2026-09-24 22:31:08 MT · **Tabs recovered:** 8/8 (all via ChatGPT download-chip / tar salvage; jailgun `done-no-tar`)  
**Source audit:** `CORE_IP_PIPELINE_DEEP_AUDIT.md` (757 lines) · **Stance:** adversarial + opportunity scan + ML roster / zero-Python audit  
**Authority:** Prefer packet §refs + READ-ONLY JopeDime code on xbabe2; mark UNVERIFIED where runtime not executed. No product commits.

---

## EXECUTIVE BRIEF (priority-ranked — decision-maker)

### 1. Most critical cracks / holes
1. **Encoder / context provenance can bypass “pure” and OOF narratives** — held-out-influenced scalers/targets/contexts can contaminate proposals while `heldout_labels_read=false` stays true; nested split scopes lack executable membership. (§§3.1, 5, 7.1–7.2; shards 1,3,5,7)
2. **Hyperion product/shape conflation + TrainFitted insufficiency** — p=128/512 cell blocks can collide with embed/signature widths; train-only ranks ≠ whole-path holdout isolation. (§§1, 4.2–4.3, 5.1; shard 2)
3. **Recipe held-out score ≠ fitted-predictor claim; emit hashes ≠ authorized origin** — verification can pass a wrong or substituted artifact; adaptive/oracle scores can inherit an inductive seal via emit. (§§8.3, 9.2, 14, 17; shards 6–7)
4. **Teaching-guard arithmetic can admit dishonest fitness** — 4/5 fold wins hide a catastrophic fold; alternate teacher ingress / cache reuse / test-adaptive ancestry can launder eligibility. (§§7.1–7.4; shards 5,7)
5. **Neural \(\mathcal{L}\) / runtime-unit / task-metric identity collisions** — multi-term loss, Huber log1p vs “seconds”, and G1/G9 observation metrics can disagree without failing listed gates. (§§6.2–6.5, 11; shards 4,7)

### 2. Highest-value breakthroughs
1. **Capability-bound encoder + counterfactual noninterference harness** — mutate forbidden labels/features; require identical training-side digests; bind `jope.encoder_receipt/v1` into pure/admission. (shards 1,3,5,7)
2. **Typed Hyperion / context ABI firewall** — `CellSurface` vs `RowEmbedding128` vs `DatasetSignature512` vs `TeachingContext2048` with fit-scope envelopes; reject same-width swaps. (shards 2–3)
3. **Proof-carrying emits** — separate integrity / execution-parity / evaluation / promotion decisions; typed `ArtifactSubject` so recipe scores cannot become fitted claims. (shard 6)
4. **Executable claim dependency graph** — consumer-side gate that requires mutually consistent receipts; unknown ≠ true. (shard 7)
5. **Unified evaluation ledger + KPI denominators** — close G4–G8/G10 with attempted-job universe, not just successful scores. (§11; shard 7 + executor)

### 3. Top proposed work items / kickoffs (opportunity scan)
1. **P0 claim/admission blockers (1–2 weeks):** encoder receipts + teaching-admission contract + Hyperion product firewall — stop sealed inductive claims without lineage.
2. **P0 emit evidence module:** recipe-vs-refit subject split + executable parity replay before any promotion narrative.
3. **KPI nightly `evaluations.jsonl` → `kpi_daily`:** admit rate, pure hygiene, router include, benchmark_valid fraction with failed/attempted denominators.
4. **Native-only train lane freeze:** quarantine Python training controller behind explicit experimental profile; document demotion of XGB/LGBM/CatBoost/RF as bench-only.
5. **Teaching-corpus `assembly_sha256` backfill** for historical rows (closes §16.1 UNVERIFIED).
6. **Inductivity challenge suite** (JointCompatibility quarantine label + CI reject for unlabeled production artifacts).

### 4. Most important ML library / model-roster gaps
1. **Not “dozens of models running in training.”** Governed Rust `feat-math` `HEADS` = **11** (`ridge`, `ridge_gcv`, `elasticnet`, `histgbm`, `ensemble`, `histgbm2`, `krr`, `gpr`, `ksmooth`, `stack`, `portfolio`) + related `FittedHead` variants (hurdle2, GprKernel). Same-cut baselines: **10** single-head methods. (xbabe2 `full_rust.rs` / `heads.rs` / `methods.json` — executor-verified)
2. **XGBoost / LightGBM / CatBoost / sklearn RF are Python subprocess bench adapters only** (`jope-bench/adapters/*_adapter.py`) — README external-engine comparison path; **not** in `HEADS` / `fit_head` training search. Claiming them as “included when training a dataset” is a **roster gap**.
3. **`jope-models` is a weight/bundle fetch/publish crate**, not an ML trainer registry — do not conflate with algorithm roster.
4. **Task estate ≠ trainer coverage:** catalog has regression/binary/multi_label/timeseries counts (§9), but target-encoder seam is **binary|regression only** (§3.3) — multi_label/timeseries lack a sealed end-to-end native path in the packet.
5. **C++17 emit is head-gated** (`supports_cpp17`); deep audit: GPR/HistGBM typically Python+Rust emit only — “ported to Rust/C++ with zero Python” is **not** true for the full head surface.

### 5. Python-dependency violations (block zero-Python goal)
1. **`jope-cli` train still execs pinned Python training controller** (`TrainCommand::PythonWorker` / `python_command`) — active training path dependency. (xbabe2 `crates/jope-cli/src/train.rs`)
2. **Entire `python/src/jope/` tree** (torch/numpy trainers, encoders, target_maker, policy, DDP) remains first-class.
3. **Bench adapters require `python3` + XGBoost/LightGBM/CatBoost/sklearn** for external-engine comparisons.
4. **`deploy/kernel-v1/source/build.rs` discovers LibTorch via `python3 -c import torch`.**
5. **No `pyo3` in crate Cargo.toml** (good for FFI), but **emit_py** still produces Python artifacts; zero-Python runtime/goal is **not met**. Flag: treat any “all models Rust/C++ only” release claim as **FAIL** until the above are removed or formally demoted with fail-closed gates.

---

## DETAILED APPENDIX

### A. Provenance
| Tab | Title (first line) | Lines | SHA-256 (content) | Recovery |
|---:|---|---:|---|---|
| 1 | <!-- from-tar tab=1 url=https://chatgpt.com/c/6ab5f4ac-5788-83ea-8796-39bfb4f6abc5 sha256=6a0e7dacabcc7deeb961801e4dc33a | 248 | `f9078364a5039d2c…` | tar chip salvage |
| 2 | <!-- from-tar tab=2 url=https://chatgpt.com/c/6ab5f4e2-ac4c-83ea-a262-d16b45c27a68 sha256=ccfe45c85223100d1f70d3ec097531 | 253 | `72615aaf73336782…` | tar chip salvage |
| 3 | <!-- from-tar tab=3 url=https://chatgpt.com/c/6ab5f519-f0ac-83ea-9b9c-5ba00b4fd106 sha256=651252f817a59327b7125fa8bbba91 | 214 | `4cba3a299be9a78a…` | tar chip salvage |
| 4 | <!-- from-tar tab=4 url=https://chatgpt.com/c/6ab5f54f-ceb8-83ea-a203-ccc81a2ff048 sha256=5b4ea7176e15bcb73929eaf80d26e6 | 225 | `a54bb46e817eb0d9…` | tar chip salvage |
| 5 | # Shard 5 — GP targets, eligibility, fitness, and the neural-proposal bridge | 226 | `22fe4512bece4a14…` | tar chip salvage |
| 6 | # Shard 6 of 8 — Code emission, verification, and hash-receipt adversarial audit | 216 | `1cddb9d365909958…` | tar chip salvage |
| 7 | # Shard 7 of 8 — Failure modes, verification gates, KPIs, and evidence map | 254 | `bdd7e30b9fd91e10…` | tar chip salvage |
| 8 | # Shard 8 of 8 — Cross-family adversarial hardening report | 254 | `21080ba0e366ac75…` | tar chip salvage |

**Jailgun:** config `jailgun.jopedime-core-ip.toml`, prompt `jopedime-core-ip-battletest.txt` (full audit inlined; source_archive off), model=`auto`, tabs=8, CDP=9225, profile=`/workspace/jailgun-runtime/chrome-profile`.  
**Watcher:** thinking_failed=0. All tabs finished `done-no-tar`; recovered via CDP download chips → `downloads/jopedime-core-ip/recovered/shard-N.tar.gz`.

### B. Equal-weight coverage map
| Layer | Where stressed |
|---|---|
| Fitness | Teaching-guard OOF, in-search RMSE, reward percentiles, Elo/KPI conflation — shards 1,4,5,7 |
| Objectives | Multi-term \(\mathcal{L}\), DPO, Huber/runtime units, task_contract vs neural L — shards 4,5,6,7 |
| DL design | Policy/heads, context 2048, proposal purity, preference pairs — shards 3,4,5 |
| All algorithm families | Encoders/Starlight/target, Hyperion cell+emb, context, DL, GP v1/v3, emit Py/Rust/C++ — shards 1–8 |
| Opportunity scan | Kickoffs in brief §3; per-shard breakthroughs B* |
| ML roster / Python | Brief §§4–5; executor notes + xbabe2 READ-ONLY probes |

### C. Per-shard pointers
Shard markdown: `/workspace/xbabe2_kicker-docs/CORE_IP_PIPELINE_JAILGUN_BATTLETEST_shards/shard-01.md` … `shard-08.md`  
Working copies: `/workspace/jailgun-runtime/jopedime-core-ip-results/shards/`  
Executor ML notes: `/workspace/jailgun-runtime/jopedime-core-ip-results/EXECUTOR_VERIFIED_NOTES.md`

### D. Explicit non-claims retained
Neural \(\mathcal{L}\downarrow\) ⇏ GP search ↑; no Rust↔Python Hyperion byte identity; prison/test-adaptive not benchmark-valid; no single all-time GP-tree total; no product commits performed.

### E. Shard focus index
1 Encoders/Starlight/target · 2 Hyperion · 3 Context 2048 · 4 DL/objectives · 5 GP/fitness · 6 Emit/verify · 7 Gates/KPIs · 8 Moonshot/attack surface

---
*End of CORE_IP_PIPELINE_JAILGUN_BATTLETEST.md*
