# CURSOR_GITHUB_PORT_PLAN

**Standalone JOPE-Dime on GitHub: engineering spec, port, S3 data, CI, and concerns.**

| Field | Value |
|---|---|
| Document | `CURSOR_GITHUB_PORT_PLAN.md` (this spec; copy into the checkout at execution start) |
| Target GitHub | https://github.com/veox-ai/JopeDime (private, empty, created 2026-09-18) |
| Source checkout | `/home/ubuntu/jain-split/jope-dime` (symlink `/home/ubuntu/jope-dime`) |
| Current origin | `https://git.neverhuman.org/git/veox/jope-dime.git` @ `c9513e8288` (`main`) |
| Date | 2026-09-18 |
| Author | Cursor / Grok 4.6 planning pass |

This document is the port plan. After approval it is written to `/home/ubuntu/jain-split/jope-dime/CURSOR_GITHUB_PORT_PLAN.md` and then executed. It is **not** a `git push --mirror` of the current tree.

**Prior records this plan is aware of and does not contradict:**

- `/home/ubuntu/jain-split/jope-dime/CODEX_SESSION_REPORT_2026-09-15.md` — 11,903 winner identities, recipe-vs-generated-CSV distinction, accepted step-60000 union, corpus snapshot-000444, two evaluator binaries, asset-model hashes.
- `/home/ubuntu/jope-migration-evidence/claude/CLAUDE-SESSION-RECORD-20260915.md` — node freeze and decommission green light, forge 403 on `veox/jope-dime` and `veox/jain-datasets`, hosted LFS limits, single-copy warning on xbabe2, `jain-split` registration still missing.
- Codex did **not** leave a `CODEX_GITHUB_PORT_PLAN.md`. Family-registration packets still point at `veox/jope-dime` on neverhuman, **not** `veox-ai/JopeDime`.

---

## 1. Mission

Stand JOPE-Dime up as a **self-contained product repository** on GitHub so work no longer depends on `~/jain-split/` sibling wiring, host-absolute paths, or the 8×A100 node tree.

The GitHub repo **must** contain, as source and tests:

1. **Hyperion** — the native tabular encoder (v5-armB geometry, 512-d pooled component).
2. **Every encoder needed to turn a dataset cut into a policy context** — Hyperion + Foundation + Target.
3. **The encodings → GP deep model** — the native transformer policy that ranks/decodes genomes from the 2048-d context.
4. **All technology that turns a GP genome into valid code** — Python, Rust, C++17, JavaScript/ES2020, plus C++20 compile verification of the C++17 artifact. Fitted C++17 runtime included.
5. **How to download training data.** Training corpora **do not** live in git. They live in **S3**, organized and de-duplicated by SHA-256.
6. **At most 10 in-repo test datasets**, chosen where JOPE/ASPFM was actually competitive.
7. **World-class layout, GitHub Actions CI, and tests that run on a clean clone.**

### 1.1 Non-goals

- Integrating this repo into `jain-split-ops/repos.manifest.toml` or Jankurai family CI. That integration is what this port exits.
- Pushing the 2.17 GiB all-refs bundle, recovery refs, agent worktrees, or 90 GB accepted-checkpoint graph to GitHub.
- Claiming the 8×A100 flywheel’s neural policy beat generic GP. The preserved science says it did not, on held-out seed comparisons. SOTArena dominance is the **GP/feat-math campaign** (ASPFM / `jope-dime`), not the neural proposer.
- Re-opening sealed `meta_test` for learning.
- Redistributing withheld ZTS, STS clinical, or other restricted tables.
- Shipping CUDA SDKs, LibTorch trees, AutoGluon pickles, or `aws.env` / `gh.env`.

---

## 2. Current state (facts the port must respect)

### 2.1 This checkout is a node workspace, not a product repo

| Fact | Number / path |
|---|---|
| Git-tracked files | **118,719** |
| Of those under `parallel/recovery` | **88,998** |
| `parallel/kernel-work` | **14,361** duplicated source snapshots |
| `parallel/releases` | **7,458** |
| Live product Rust | `parallel/src/` (**107** tracked files) + root `src/` (**149**) + `vendor/` (**91**) |
| `.github/workflows` | **none** |
| Root `LICENSE` | **missing** (Cargo.toml says Apache-2.0) |
| `aws.env` / `gh.env` | untracked in the working tree; **not** in `.gitignore` |
| Golden42 `train.csv`/`test.csv` | **already in git** under `sotarena/data/tabular-regression-v1/` |
| GitHub target | private, **empty**, `diskUsage: 0` |
| `jain-split` required_repos | 31 jain-* crates; **jope-dime is not one of them** |

`/home/ubuntu/jope-dime` is a symlink to `/home/ubuntu/jain-split/jope-dime`. That alias already caused a real miss during migration (Claude session record §0 / §4 #48). The standalone repo must not depend on either spelling.

### 2.2 Two overlapping stacks

| Stack | Crate | What it actually is |
|---|---|---|
| Campaign / control plane | root `jope-dime-regressor` | Frozen Golden42/46, Binary42, Encoder V2 identity, ratchet, self-play services, Hyperion *gate* |
| Production DL→GP flywheel | `parallel/` `jope-parallel` | Prepare cuts, encode, train policy, sample GP, fit, export code |

`parallel/Cargo.toml` vendors feat-math from **`parallel/assets/source/jope-dime/vendor/feat-math`** (a nested snapshot of this repo), **not** the live root `vendor/feat-math`. They can drift. The port uses **one** feat-math.

### 2.3 The product pipeline that must work

```
backing CSV (content-addressed)
        │  prepare.rs  (recipes only; no stored cut CSVs)
        ▼
prepared cut (train rows + train-only preprocessing)
        │
        ├─ Hyperion native  ── 128-d rows ── pool mean/std/p10/p90 ──► 512-d   hyperion.f32
        ├─ Foundation TS    ── pooled 2048-d hiddens, PCA+std     ──► 1536-d   (from foundation_raw.f32 2048)
        └─ Target TS        ────────────────────────────────────────► 512-d   target.f32
                │
                ▼  contexts.rs assemble, cohort transforms fit on meta_train only
        2048-d policy context  =  1536 dataset  +  512 target
                │
                ▼  neural.rs Policy  (width 512, 6 layers, 4 heads, FFN 2048)
        grammar-masked decode  →  feat-math genome s-expression
                │
                ▼  gp.rs fit on prepared view (CPU evaluator)
        champion event (identity includes evaluator + vocab hashes)
                │
                ▼  champion_export / feat-math emitters
        predict.py  predict.rs  predict.cpp (C++17)  predict.js (ES2020)
        + fitted C++17 runtime when BoundModel supports it
        + CI also compiles predict.cpp as C++20
```

Encoder V2 (`src/encoder_v2_*.rs`, 2048-float mean/std/p10/p90 layout) is the **self-play Foundation identity layer**, not the flywheel. Keep its contracts if tests still bind them; do not run two production encoders.

### 2.4 What SOTArena actually measures

Public SOTArena (`/home/ubuntu/sotarena`, 1,790 datasets): family **ASPFM** matches `jope-*` and `aspfm*`. Adaptive Elo **#1 global 1894.2**. Strict benchmark-valid Elo has ASPFM on **regression** only; binary/multiclass/timeseries champion rows are largely **test-adaptive**.

Confidential Golden42/45 verifier inside this tree: method **`jope-dime`**, protocol `aspfm-generated`, `benchmark_valid=true`. **11/42 strict wins**, Elo ~2259, rank 1.

**Do not mix those.** CI that “proves we beat TabPFN” on Binary42 / TimeSeries42 / multiclass union43 would launder test-adaptive evidence. The 10 in-repo fixtures are labeled with their protocol.

### 2.5 Weights are pinned and mostly absent here

| Artifact | Spec SHA-256 prefix | In this checkout? |
|---|---|---|
| `assets/models/hyperion-v5-armB/model.safetensors` | `88b9e834…` | **No** (mirror class `assets`, 918 files / ~0.94 GiB) |
| `assets/foundation-export-v1/foundation.pt` | `87bae6a2…` | **No** |
| `assets/target-export-cuda-v1/target.pt` | `14fbc78d…` | **No** |
| `assets/target-export-v1/target.pt` | CPU export | **Yes, 75 MiB** |
| Policy checkpoint step-60000 `8ac80c26…` | union 90.26 GiB | gitignored recovery tree |

Production inference is Rust (`tch` CModule + candle Hyperion). Python exists to **export** TorchScript, not to serve.

### 2.6 Data is not in S3 today

There is **no** `s3://` training layout in jope-dime or jain-datasets. Current homes:

- `https://data.neverhuman.org/<task>/<hash8>/{train,test}.csv` (SOTArena fetch)
- `/home/ubuntu/remote_super/quant/<hash8>/`
- node mirror `jain-datasets/.runtime/node-archive-mirror-20260914/` (~193 GB)
- Golden42 CSVs in git
- hyper `/home/ubuntu/hyper/data/tabular/` (~5.6 GB, OpenML/PMLB/sklearn fetchers)
- TabArena Lite OpenML suite 457

`jain-datasets` is a **bridge CLI**, not a published payload. Hosted `veox/jain-datasets` was 403 / never created.

Prepared GP tables are **regenerated**: Codex offline replay verified **9,783 cuts / 19,566 train+val tables** from recipes + backing hashes. Storing those CSVs is the redundancy this port deletes.

---

## 3. Key decisions

| # | Decision | Rationale |
|---|---|---|
| D1 | **New orphan `main` on GitHub.** Do not `git push --mirror`. Keep neverhuman as `archive` remote only. | 2.17 GiB bundle, 50 refs, recovery branches, agent worktrees. GitHub is a product repo, not a node dump. |
| D2 | **One Cargo workspace**, crate `jope` + libraries. Dual root/`parallel` workspaces go away. | Two feat-math copies and two toolchains is the confusion we are leaving. |
| D3 | **S3 is the object store for training data and model weights.** Git holds catalogs + SHA-256 + fetch code. Git LFS is not used for corpora. | Owner request. GitHub rejects raw blobs ≥100 MiB; hosted forge LFS policy does not apply here. |
| D4 | **At most 10 tiny datasets in git**, all hash-pinned, all licensed for this use. Golden42’s 42 CSVs leave git. | Owner cap. Current git already violates it. |
| D5 | **Production encode/infer is Rust.** Python is export/conversion only. | Matches `parallel` today (`build.rs` never discovers PyTorch via Python). |
| D6 | **Default build is CPU, no LibTorch, no CUDA, no SmartCluster.** Features: `neural`, `hyperion-cuda`, `cluster`. | `just required` must pass on `ubuntu-latest`. |
| D7 | **jain-smartcluster is optional.** Default does not git-depend on `git.neverhuman.org`. | That URL 403’d this host; it is the jain-split coupling. |
| D8 | **GP emit stays C++17** (`-std=c++17 -ffp-contract=off`). CI also builds the same file as **C++20**. No second numeric dialect. | Bit-stable emit already exists. `parallel/native/runtime.cpp` is the C++20 LibTorch bridge, not genome codegen. |
| D9 | **Do not vendor the entire `hyperion-jain` workspace** (catboost/xgboost/web/sagemaker). Extract Hyperion encode path only. | That nested workspace is Jain 7.0.1, not JOPE. |
| D10 | **SOTArena CI uses frozen published scores + train-only GP smoke.** Test-adaptive campaigns stay labeled and off the required gate. | Honest science. |
| D11 | **Content-addressed S3 layout, one object per SHA-256.** Names are indexes. | Quant already has ~104 duplicate-byte files; encodings/prepared/quant/sotarena copies of the same CSV are the main waste. |
| D12 | **Workspace toolchain 1.97.1** (current files). **MSRV 1.90** on the CPU evaluator. Kernel-v1 docker (rustc 1.95 / glibc 2.35) is a separate nightly job. | Do not pretend one rustc ever built every historical evaluator. |
| D13 | **No jain-split Jankurai/proof CI.** GitHub Actions tests JOPE. | Owner: jain-split integration is slowing us down. |
| D14 | **Secrets live in GitHub Actions + AWS IAM, never files.** `aws.env` / `gh.env` are gitignored; the IAM key in `aws.env` should be **rotated** because it was sitting untracked in the working tree. | `.gitignore` does not currently list them. |
| D15 | **Accepted60000, corpus444, 11,903 winners stay off GitHub.** Catalog + S3 (+ existing xbabe2 mirror) until a dedicated data bucket is green. | Codex/Claude preservation already hashed those bytes. Git cannot hold them. |

---

## 4. Target repository layout

```
JopeDime/                          # github.com/veox-ai/JopeDime
  README.md
  LICENSE                          # Apache-2.0
  THIRD_PARTY_NOTICES.md
  rust-toolchain.toml              # 1.97.1
  Cargo.toml                       # workspace
  Cargo.lock
  deny.toml
  .gitignore                       # env, target, recovery, datasets bulk
  .github/workflows/ci.yml
  .github/workflows/models.yml     # optional, weights from S3
  Justfile
  crates/
    feat-math/                     # single copy (from vendor/feat-math)
    jope-core/                     # hashes, identities, errors, paths
    jope-data/                     # catalog, S3, prepare, regenerate views
    jope-encode/                   # hyperion + foundation + target
    jope-policy/                   # neural.rs / proposals / grammar
    jope-search/                   # gp.rs evolution + fit
    jope-export/                   # champion_export + transpile
    jope-cli/                      # `jope` binary
  native/
    hyperion-host/                 # extracted from parallel/native
    runtime.cpp                    # C++20 LibTorch/NCCL, feature neural
  python/
    export_foundation.py
    export_target.py
    pyproject.toml                 # export extras only
  assets/
    grammar/                       # featmath-jain-v1.json
    sampling/                      # frozen sampling recipes (JSON only)
  catalogs/
    datasets.json                  # every training set: sha, fetch, license
    models.json                    # encoder/policy objects in S3
    redundancies.json              # SHA collisions / dropped copies
  tests/
    fixtures/datasets/             # THE 10
    gp/
    emit/
    prepare/
    encode/                        # skipped unless JOPE_MODELS_DIR set
    policy/
  ops/ci/
    required.sh
    emit-parity.sh
    path-audit.sh                  # fail on /home/ubuntu
    s3-smoke.sh
  docs/
    ARCHITECTURE.md
    DATA.md
    MODELS.md
    CODEGEN.md
    CI.md
```

No `parallel/recovery`, no `kernel-work`, no `AGENT_CHAT`, no nested `sotarena/private/jope-dime`, no `AutogluonModels`, no paper `target-*` build dirs.

### 4.1 Crate responsibilities

| Crate | Owns | Must not own |
|---|---|---|
| `feat-math` | Genome DSL, fit/predict, emitters, portable math, fitted C++17 | IO, S3, GPU |
| `jope-core` | SHA-256 helpers, champion identity struct, relative-path rules | Host paths |
| `jope-data` | Catalog schema, S3 get/put, `prepare::load_view`, split regeneration | Model weights |
| `jope-encode` | Hyperion host, TorchScript foundation/target, 2048-d assemble | GP search |
| `jope-policy` | Transformer, grammar mask, propose-ranked | Dataset fetch |
| `jope-search` | GP population, fit_view, evaluator receipt | Neural |
| `jope-export` | `champion_export`, language parity, C++17/C++20 compile | Training |
| `jope-cli` | `jope fetch`, `encode`, `search`, `export`, `verify-inputs` | Hidden defaults to `/home/ubuntu` |

Public CLI (minimum):

```text
jope fetch --catalog catalogs/datasets.json --id <hash16>   # S3 or upstream
jope prepare --dataset <id> --cut N
jope encode --prepared <dir> --cut N                        # needs models
jope search --prepared <dir> --cut N --seconds T
jope export --champion <event.json> --out <dir>
jope verify-inputs                                          # fixtures + pins
```

---

## 5. Models that must ship (as downloadable objects)

### 5.1 Encoding models (required)

Context assembly in `parallel/src/contexts.rs` binds three components per cut:

| Component | File | Dim | Runtime |
|---|---|---|---|
| Hyperion | `hyperion.f32` from `hyperion-v5-armB/model.safetensors` | 512 (pooled 128-d rows × 4 stats) | native candle/CUDA host (`parallel/native/hyperion-host`, `hyperion_v7` loader with v5-armB geometry) |
| Foundation | `foundation_raw.f32` then projected | 2048 raw → 1536 in context | TorchScript `foundation.pt` via `tch::CModule` |
| Target | `target.f32` | 512 | TorchScript `target.pt` |

Source that must be **in git** (code only):

- Hyperion: `parallel/assets/source/hyperion-jain/crates/feat-core/src/{hyperion.rs,hyperion_v7/**,encode/**}` extracted into `crates/jope-encode` / `native/hyperion-host`. Drop catboost/xgboost/feat-web/sagemaker members.
- Foundation export: `parallel/tools/export_foundation.py` + the Starlight expert load path from `fable_god` **needed to rebuild `foundation.pt`**, not to run encode.
- Target export: `parallel/assets/source/target_maker/` + `parallel/tools/export_target.py`.

Weights that must be **in S3**, pinned in `catalogs/models.json`:

```
s3://<bucket>/models/encoders/hyperion-v5-armB/<sha256>/model.safetensors
s3://<bucket>/models/encoders/foundation-export-v1/<sha256>/foundation.pt
s3://<bucket>/models/encoders/target-export-v1/<sha256>/target.pt
s3://<bucket>/models/encoders/target-export-cuda-v1/<sha256>/target.pt
```

CPU CI does not download these. `tests/encode` is `ignore` unless `JOPE_MODELS_DIR` or `JOPE_S3_BUCKET` is set.

### 5.2 Encodings → GP model (required)

| Piece | Path today | Notes |
|---|---|---|
| Policy | `parallel/src/neural.rs` | 1536+512 context MLP → width 512, 6×4-head, FFN 2048, `dense_pairwise_v2` |
| Train | `parallel/src/training.rs` | 8-GPU NCCL; **not** required for CPU product |
| Propose | `parallel/src/proposals.rs` | grammar-masked decode → `GenomeIr` |
| Grammar | `parallel/assets/grammar/featmath-jain-v1.json` | **in git** |
| Serving checkpoint | accepted step-60000 `8ac80c26…` | **S3 only**, plus a tiny randomly-init fixture for CPU unit tests |

Honest README line: neural seeding was **not** shown to beat generic seeds on held-out families (seed-quality experiment: larger-pool RMSE **worse**). The policy still ships because the owner required the encodings→GP model. CI checks **forward + decode + parse**, not a leaderboard claim.

### 5.3 Hyperion training

Training of Hyperion itself lives outside this repo (Hyperion Entanglement / veox-bootstrap). This repo is **inference, pooling, and export**. Do not copy `/home/ubuntu/hyperion` training dumps.

---

## 6. GP → valid code

Keep `vendor/feat-math` as `crates/feat-math`. Product surface is `crates/jope-export` wrapping `parallel/examples/champion_export.rs`.

| Language | API | Status | Port action |
|---|---|---|---|
| Python 3 | `transpile::emit` / `emit_py` | Full (emitter 2.1) | Required CI |
| Rust | `emit_rs` | Full | Required CI |
| C++17 | `emit_cpp` + `emit_fitted_cpp17` | Fail-closed; narrower op set | Required CI; compile `g++ -std=c++17 -O2 -ffp-contract=off -fno-fast-math` |
| JavaScript ES2020 | `emit_js` | Fail-closed | Required CI (Node) |
| C++20 | **no separate emitter** | `runtime.cpp` is LibTorch | **Compile the C++17 artifact with `-std=c++20`** and require identical object behavior on the fixture; do not invent a C++20 dialect |
| Fitted BoundModel | `emit_bound` Py/Rs, fitted C++17 runtime | Multiclass probability matrix rejected | Keep fail-closed; test the supported heads |

Parity contract (already in champion_export): native refit matches observation `semantic_sha256` and RMSE; emitted Python on the same rows agrees elementwise at `1e-9`. **Zero LLM.**

Portable math (`MathKernel::Portable`) stays behind a flag; default remains host libm with the certified op set.

---

## 7. Datasets and S3

### 7.1 Identity model (do not collapse)

A scientific dataset identity is **not** a display name. Champion identity (from `champions.rs` / Codex report §4):

```
dataset, family, partition, cut,
prepared_sha256, training_sha256, validation_sha256,
evaluator_sha256, vocabulary_sha256
```

Object store identity is **SHA-256 of bytes**. Short id is first 16 hex. Paths are provenance.

### 7.2 What goes in S3 vs git vs regenerate

| Class | Git | S3 | Regenerated |
|---|---|---|---|
| Catalog JSON (ids, licenses, urls, hashes) | yes | copy | no |
| 10 CI fixtures (train+test CSV) | **yes** | optional replica | no |
| Golden42/46, Binary42, TabArena, TS42, quant bulk | no | **yes** | no |
| Prepared recipes (`prepared.json`, preprocessing, row maps) | small recipes yes | bulk yes | no |
| Generated cut train/val CSVs | no | **no** | yes (`load_view`) |
| Encodings `*.f32` / `*.vs` | no | optional cache | yes |
| Sealed meta_test winners | no | `s3://…/sealed/` isolated | never used for train |
| Restricted (full ZTS, STS, NOMAD extras) | no | `s3://…/restricted/` or omit | no |

### 7.3 Proposed S3 layout

Bucket name is an open question; default **`veox-jope-dime`** (one bucket, prefixes). Versioning on. Block public access. SSE-S3 or SSE-KMS.

```
s3://veox-jope-dime/
  catalogs/
    datasets.json
    models.json
    redundancies.json
  objects/sha256/<aa>/<aabbcc…>/     # one blob per hash; content-type + metadata
  indexes/
    sotarena/binary/<id16> → {train_sha, test_sha, task, license}
    sotarena/regression/…
    sotarena/multiclass/…
    sotarena/timeseries/…
    golden42/
    golden46/
    binary42/
    tabarena-lite/
    timeseries42/
    multiclass-union272/
    quant/regression/                 # 2009 metadata
    quant/binary/                     # 307
    quant/multi_label/                # 70
    quant/timeseries/                 # 125
    hyper-tabular/                    # 5.61 GB unique-ish
    prepared/
    backing/
  models/
    encoders/…
    policy/step-60000/<sha>/
  sealed/meta_test/                   # 873 identities; IAM deny train roles
  restricted/                         # ZTS parent, etc.; separate IAM
```

`jope fetch` always pulls **objects/sha256/…** and verifies. Indexes are JSON.

### 7.4 Families to ingest (training, not git)

| Family | Count | Source today | Fetch method to keep |
|---|---:|---|---|
| SOTArena downloadable | 1,695 of 1,790 | `data.neverhuman.org` | Port `sotarena fetch` to S3 after copy |
| Golden42 | 42 | git + quant | Hash-verify into S3; **delete from git** |
| Golden46 extras | 3 | encrypted LFS | restricted prefix or skip |
| Binary42 | 42 | quant / missing sotarena path | OpenML/UCI + frozen hashes |
| TabArena Lite | 51 | OpenML suite **457** | `download_tabarena_lite.py` |
| TimeSeries42 | ~42 real | `remote_super/quant/timeseries` + HF `autogluon/fev_datasets` | HF + hash |
| Multiclass union | 272 runnable / 70 A100 | OpenML + synth | hash catalog |
| A100 quant regression | 2009 | node-mirror + `remote_super/quant` | copy unique SHA |
| A100 quant binary | 307 | same | same |
| Hyper tabular | 6,707 files / 5.61 GB | `/home/ubuntu/hyper/data/tabular` | `dl_run_all.py` replay against catalog |
| Expansion backing v1–v10 | various | hyper `real_world_additions`, quant `expansion_v*_raw` | SHA union |
| Prepared views | thousands of recipes | `parallel/data/prepared/` + mirror | JSON to S3; no generated CSVs |

95 SOTArena datasets are metadata-only (skipped by fetch). Record them as `undownloadable`.

### 7.5 Redundancies to collapse

1. **Same 16-hex CSV** in git Golden42, `remote_super/quant/<id>/`, flywheel encodings, node-mirror quant, USB. Keep one SHA.
2. **Hyper `openml-<name>-<id>-target-<col>.csv` vs quant 16-hex vs SOTArena 16-hex.** Join on content hash.
3. **Prepared views** duplicated across `views`, `views-v2`, `views-expanded-20`, `views-extra-20`, mirror `expansion-views-v1..v8`. Keep the recipe that winner identities bind.
4. **Encodings** in flywheel, `parallel/data/encodings`, node-mirror `contexts/`. Do not upload all three; encodings are derived.
5. **Quant census** 13,023 files vs 12,919 distinct hashes (~104 byte-duplicates).
6. **Nested copies of this repo:** `parallel/assets/source/jope-dime/`, `sotarena/private/jope-dime/`, `raw223-test-adaptive/source/`. **None** of those trees go to GitHub.
7. **kernel-work** = 14k copies of source for evaluator lineage. Keep **one** frozen kernel-v1 tree for the product-build job, not 30.
8. **Reports** under `remote_super/quant/<id>/reports/` (180+ JSON per dataset) are not training data. Skip or a separate `s3://…/reports/` prefix.

Ingest job: walk known roots, hash, write `redundancies.json` `{sha256, bytes, locations[]}`, upload each sha once.

### 7.6 License / redistribution

| Item | Rule |
|---|---|
| Full ZTS | Withheld. Not in S3 public prefix. |
| ZTS_small / two NOMAD | Encrypted Golden45 bundle; `restricted/` or omit from the public catalog |
| Golden42 | Per-row `source_license` in registry; S3 private bucket is still a redistribution — keep license in catalog; skip rows that forbid it |
| Binary42 “STS AFib mortality” | Restricted even if a CSV exists locally |
| GPL Rdatasets (peanuts, glucose, etc.) | Apache-2.0 repo can **depend** on GPL data files as data, not link them as code; document. Prefer MIT/CC0/PMLB for the 10 fixtures |
| `data.neverhuman.org` | Current CDN; S3 becomes canonical after ingest; keep URLs as `upstream` |

### 7.7 The 10 in-repo test datasets

Selection rules used:

- JOPE/ASPFM **actually competitive** (strict Golden42 win and/or large public margin).
- Small (tens of KB).
- Diverse tasks.
- Prefer **benchmark_valid** regression; label adaptive tasks honestly.
- Skip leaky-target / identifier-target / simulated-name screens from `docs/paper/data/showcase_cases.csv` (Parkinsons, Bike Sharing Day, food-safety `observation`, `pwLinear` name filter).

| # | id | task | why competitive | size (approx) | protocol | local bytes today |
|---|---|---|---|---|---|---|
| 1 | `693890aae9dc860b` | regression | Golden45 **strict win**. RMSE 0.00338 vs TabICL 0.00386 / TabPFN 0.00434 / AutoGluon 0.00848 | 202/41 rows, 1 feat | `aspfm-generated`, **benchmark_valid** | `sotarena/data/tabular-regression-v1/` |
| 2 | `047e62b52a3f4ede` | regression | Strict win. chatfield_4 vs TabPFN 0.0534 / TabICL 0.0538 | 195/40, 12 feat | valid | same |
| 3 | `79e600bc5fcd5da7` | regression | Strict win. peanuts vs TabFM 0.1153 | 180/36, 3 feat | valid | same (GPL agridat — document) |
| 4 | `b5beff4a53fa12c0` | regression | Strict win. UCI credit-approval vs TabPFN 0.01693 | 575/115, 15 feat | valid | same |
| 5 | `17a4542ee4fb9e44` | regression | Strict win vs AutoGluon 0.106 / TabICL 0.107 | 157/32, 9 feat | valid | same (registry license **null** — confirm or swap for `019de5d76a718163` CC0 seaborn flights, rank 3 smoke) |
| 6 | `c32fa09b48fb2f27` | regression | Strict win cmc, 1000/473 | ~1.5k rows | valid | same |
| 7 | `e73becccb4c234d3` | binary | Native bupa (not threshold-derived). AUC 0.798 vs GPT-*sol 0.760 | 287/58, ~21 KB | **adaptive** | `remote_super/quant/binary/` |
| 8 | `8b46a264c7f95580` | binary | Paper-selected eligible strict-best **saheart**. AUC 0.824 vs poly2_logreg 0.794; 385/77, 9 feat, **7.3+36.6 KB** | tiny | adaptive public | `sotarena/data/binary/…/dataset.json` + neverhuman URL |
| 9 | `7d35db2038691458` | timeseries | MASE 0.277 vs TimesFM 0.517 (−46%). **~2.5 KB** | 697/28 | **test-adaptive** | quant/timeseries |
| 10 | `5c77c48322624f7f` | multiclass | vertebra-column macro-F1 0.815 vs AutoGluon-3600 0.791 | 258/52, ~23 KB | **test-adaptive** | quant/multi_label |

**CI use of the 10:**

- All 10: parse, schema, hash, `jope prepare` smoke where recipes exist.
- Regression 1–6: **train-only** GP fit smoke + emit parity on a frozen genome (do not evolve against test).
- 7–10: fixture parse + optional replay of **published** score JSON; never a “we beat AutoGluon” required gate.

**Rejected despite looking strong:** Parkinsons `21fbfe8b53dbcd60` (leaky), Bike Sharing Day (leaky + rank 17), food-safety (identifier target), porto-seguro seeds (synthetic-style), Binary42 ILPD as primary (explicit `test_adaptive`), ZTS/NOMAD (restricted).

If license-null on #5 is a problem, replace with `019de5d76a718163` (CC0 flights) as the smoke set and keep 5 strict wins.

Layout:

```
tests/fixtures/datasets/
  regression/693890aae9dc860b/{train.csv,test.csv,dataset.json}
  …
  MANIFEST.json   # sha256 per file, task, protocol, license, sotarena url
```

Total well under 2 MB.

---

## 8. CI and testing

GitHub Actions on `ubuntu-latest`. No xbabe2 hostname locks. No `xbabe2-local-ui-ci.sh`.

### 8.1 Required (PR gate, ~15–25 min)

1. `ops/ci/path-audit.sh` — fail if `/home/ubuntu`, `git.neverhuman.org` (except documented archive URL), or `158.101.` appear in tracked source.
2. `cargo fmt --all -- --check`
3. `cargo clippy --workspace --all-targets --locked -- -D warnings` (first PR may start as warn-only if the extracted tree is dirty; become `-D warnings` before calling CI green).
4. `cargo test --workspace --locked` with `OMP/RAYON/OPENBLAS/MKL_NUM_THREADS=1`
5. Emit parity: Python + Rust + C++17 + C++20 compile + JS on the fixture genome (`ops/ci/emit-parity.sh`)
6. `jope verify-inputs` on the 10 fixtures
7. `gitleaks` (or equivalent) so `aws.env` cannot land

### 8.2 Optional / nightly

| Job | When | What |
|---|---|---|
| `s3-catalog` | nightly | `jope fetch --dry-run` against catalogs; no GPU |
| `models` | workflow_dispatch + nightly if secrets | download encoder weights, CPU Hyperion smoke |
| `kernel-v1` | nightly | docker glibc≤2.36 CPU evaluator; needs a GHCR copy of `jeryu-builder` **or** a public rust:bookworm image with the frozen source |
| `neural-cuda` | self-hosted GPU **later** | `cargo test --features neural`; do not block PRs |

### 8.3 Tests that must exist in the new tree (mapped from today)

| New test | From | Gate |
|---|---|---|
| feat-math portable + fitted C++ | `vendor/feat-math/tests/` | required |
| prepare regenerate hash | `prepare.rs` / Codex replay idea, 1 fixture cut | required |
| champion identity parse | `champions.rs` | required |
| grammar decode → GenomeIr | `grammar.rs` | required |
| export 4 languages | `champion_export.rs` | required |
| EncoderSpec pin check | `encoding.rs` (hashes only, no weights) | required |
| Hyperion CPU forward | feat-core hyperion tests, tiny tensor | nightly/models |
| Policy CPU init + one decode | `neural.rs` without real checkpoint | optional feature |
| Absolute path audit | `scripts/absolute_path_audit.py` | required |

Do **not** import the Hyperion Python campaign tests that SSH to xbabes or read `/home/ubuntu/remote_super/quant`.

---

## 9. GitHub bootstrap (how the empty repo becomes real)

Verified: `gh repo view veox-ai/JopeDime` → private, empty, `main` unset.

**Do not** `git remote add origin` on the current 118k-file checkout and push.

Procedure:

1. Write this document to `CURSOR_GITHUB_PORT_PLAN.md` in the current checkout (documentation only; current origin stays neverhuman).
2. Create a **new worktree or `/tmp/JopeDime-export`** (not inside `jain-split` as a sibling that family tools will scrape).
3. `git init -b main`, Apache-2.0 LICENSE, README, toolchain, empty workspace, CI that builds a stub `jope --version`.
4. `git remote add origin https://github.com/veox-ai/JopeDime.git` and first push.
5. Add neverhuman as `archive` on the **old** checkout only: `git remote add archive https://git.neverhuman.org/git/veox/jope-dime.git`.
6. Subsequent PRs copy extracted crates into the new repo. Prefer **history-less file copy + pin of source commit SHA in `docs/ORIGIN.md`** (`c9513e8288` plus asset hashes).

Branch protection on GitHub: require `ci` on `main`, no force-push, no 100 MiB blobs.

`.gitignore` must include at minimum:

```
target/
**/*.env
aws.env
gh.env
*.passphrase
AutogluonModels/
lightning_logs/
parallel/recovery/
parallel/kernel-work/
parallel/kernel-work-evidence/
runs/
```

---

## 10. Extraction map (what to copy, from where)

### 10.1 Copy (source)

| Destination | Source |
|---|---|
| `crates/feat-math` | `vendor/feat-math` (live, not the nested snapshot) |
| `crates/jope-search` + `jope-policy` + `jope-encode` | `parallel/src/{gp,neural,proposals,grammar,encoding,components,contexts,prepare,champions,hyperion}.rs` and deps, split along crate boundaries |
| `native/` | `parallel/native/` |
| `crates/jope-export` | `parallel/examples/champion_export.rs` + feat-math transpile |
| `python/export_*.py` | `parallel/tools/export_{foundation,target}.py` |
| `python/target_maker` | `parallel/assets/source/target_maker` |
| Hyperion encode | extract from `parallel/assets/source/hyperion-jain/crates/feat-core` |
| Grammar JSON | `parallel/assets/grammar/` |
| Sampling JSON | `parallel/assets/sampling-reviewed4-v1/{foundation,target}-spec.json` (not rank logs) |
| Catalog seeds | `sotarena/registry/`, `manifests/panel.txt`, SOTArena `dataset.json` hashes |
| Absolute-path audit | `scripts/absolute_path_audit.py` rewritten to scan the new tree |
| Kernel-v1 **one** source | `parallel/releases/kernel-v1-8aa0affbf…/source` if the nightly job is kept |

### 10.2 Do not copy

`parallel/recovery/`, `parallel/kernel-work/`, `parallel/kernel-work-evidence/`, most of `parallel/releases/`, `AGENT_CHAT*`, `PR_AGENT_CHAT.md`, `REALTIME_AGENT_CHAT.md`, `AutogluonModels/`, `lightning_logs/`, `feedback/`, `docs/paper/internal/`, `docs/paper/tools/target-*`, `raw223-test-adaptive/source`, `sotarena/private/`, `sotarena/data/tabular-regression-v1/**/*.csv` (except the 10), CUDA SDK, LibTorch, `aws.env`, `gh.env`, `NEW_REPO.md`, systemd units that SSH to `158.101.20.226`.

### 10.3 Rewrite while copying

- Every `/home/ubuntu/...` default → `JOPE_ROOT` / XDG cache / S3 URI.
- `jain-smartcluster` git dep → optional path or omit.
- `JOPE_CPU_EVALUATOR` must be overridable; tests ship a fixture binary built from feat-math.
- Registry default `/home/ubuntu/remote_super/quant/regression` → `catalogs/datasets.json`.
- Foundation expert `sys.path.insert(0, "/home/ubuntu/fable_god")` → package-relative (export-only).
- Dual `rust-toolchain.toml` → one at repo root.

---

## 11. Concerns and risks

| ID | Concern | Mitigation |
|---|---|---|
| C1 | **Neural policy is not a held-out winner.** Seed-quality: neural RMSE worse than generic. Flywheel F4/F9/F11: no independent improvement; step-69000 rejected. | README + CI claims match evidence. Ship the model; do not score it as SOTA. |
| C2 | **ASPFM SOTA ≠ flywheel.** Mixing them in marketing or CI is a scientific error. | Separate `docs/BENCHMARK.md` vs `docs/LEARNING.md`. |
| C3 | **Missing weights in the checkout.** Hyperion/foundation/cuda-target not on this tree. | Ingest from mirror class `assets` (Claude gate: 8/8 hash match) into S3 before encode CI. |
| C4 | **Single-copy data on xbabe2.** Claude §1.1: only git bundles exist on two disks. | S3 ingest is also the backup. Do it before any disk cleanup. |
| C5 | **GitHub 100 MiB blob limit.** `target.pt` is 75 MiB (ok as git, still better in S3). Hyperion safetensors size unknown here; treat as S3. | No raw ≥100 MiB. |
| C6 | **Forge 403 / SmartCluster git dep.** | D7: optional feature. |
| C7 | **C++17 emit coverage < Rust/Python.** Unsupported ops fail closed. | Tests assert fail-closed; do not widen silently. |
| C8 | **Toolchain zoo** (1.90 / 1.95 docker / 1.96 hyperion / 1.97.1). | D12. Extracted Hyperion uses workspace 1.97.1; if it will not compile, pin a `crates/jope-encode` exception in the plan’s first encode PR, do not nest a second workspace casually. |
| C9 | **GPL data in an Apache repo.** | Data vs code; catalog licenses; prefer MIT/CC0 in fixtures. |
| C10 | **Test-adaptive leakage** if CI evolves on test. | Train-only GP for required; published scores for adaptive fixtures. |
| C11 | **Encoder V2 vs flywheel dual encode.** | Flywheel 3-encoder path is production. V2 is contracts-only unless a test still requires the Python TabPFN child. |
| C12 | **Nested feat-math drift.** | One copy. Diff snapshot vs live at extraction; take live + note SHA. |
| C13 | **`aws.env` in the working tree.** | Gitignore + rotate IAM key. Never commit. GitHub secret `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / `JOPE_S3_BUCKET`. |
| C14 | **LibTorch 2.13 vs comments saying 2.7.** | Pin **2.13.0** (tch 0.26.0) in MODELS.md. |
| C15 | **8-GPU NCCL training will not run on GitHub-hosted runners.** | CPU propose/decode tests; GPU job later on a runner you control. Do not reuse retired `actions.runner.*` on xbabes without an explicit owner decision. |
| C16 | **Paper / IP.** `docs/paper` is private working tree. | Omit from GitHub unless owner asks. |
| C17 | **Sealed 873 meta_test winners.** | Isolated S3 prefix; IAM deny to train roles; never in git. |
| C18 | **Family registration still describes `neverhuman/jope-dime`.** | Ignore it. This GitHub org/repo is a new identity. |
| C19 | **Hyperion Python export still pulls TabPFN/Starlight.** | Keep as `python/` extra; production encode uses exported `.pt`. |
| C20 | **Prepared regeneration is the source of truth.** Agents previously filed false missing-CSV bugs (Codex §9.2). | Tests regenerate; S3 does not store cut CSVs. |

---

## 12. Alignment with the 2026-09-15 session records

What those sessions **preserved** and this port **must not lose**, but also **must not dump into GitHub**:

| Preserved item | Port home |
|---|---|
| 11,903 GP winners (11,030 train/val + 873 sealed) | S3 `winners/` + `sealed/`; not git |
| Evaluator binaries `7ebc8476…` and `17e9b156…` | Optional `s3://…/evaluators/`; CI builds from source |
| accepted60000 union 224,317 files / 90.26 GB | stay on xbabe2 + S3 later; **not** git |
| corpus snapshot-000444 | S3 later; not git |
| Asset models 8/8 hash match | S3 `models/` first-class |
| Offline replay of 9,783 cuts | becomes `jope-data` test (1–10 cuts, not 9,783 in CI) |
| Recipe vs generated CSV | D11 / §7.2 |

What those sessions **left open** that this port closes:

- Forge publication 403 → **GitHub `veox-ai/JopeDime`** as the code home.
- `jain-datasets` never hosted → **S3 catalog**, not a second git-LFS science repo, unless the owner later wants both.
- Absolute paths / `jope-dime` symlink miss → path audit + env.
- Dirty Codex tree vs Claude temp-index commits → orphan GitHub `main` from a curated export, pin `c9513e8288` as origin snapshot.

---

## 13. Execution phases / PR plan

PRs are against **the new GitHub repo**, except P0 which only writes the plan file in the current checkout.

### P0 — Record the plan (this checkout)

- **Title:** `docs: add CURSOR_GITHUB_PORT_PLAN.md`
- **Files:** `CURSOR_GITHUB_PORT_PLAN.md` (this document)
- **Deps:** none
- **Verify:** file exists; no secrets in it.

### P1 — Empty GitHub product skeleton

- **Title:** `chore: bootstrap veox-ai/JopeDime workspace and CI`
- **Files:** LICENSE, README, rust-toolchain, workspace Cargo.toml, `crates/jope-cli` stub, Justfile, `.github/workflows/ci.yml`, `.gitignore`, `ops/ci/{required,path-audit}.sh`
- **Deps:** P0 conceptually; new repo
- **Verify:** GitHub Actions green on `jope --version`; gitleaks clean.

### P2 — feat-math + emit parity

- **Title:** `feat: vendor feat-math and four-language emit CI`
- **Files:** `crates/feat-math`, `crates/jope-export` (minimal), `ops/ci/emit-parity.sh`, portable fixtures
- **Deps:** P1
- **Verify:** Python/Rust/C++17/C++20/JS parity on the certified genome; unsupported ops fail closed.

### P3 — data catalog + S3 client + 10 fixtures

- **Title:** `feat: dataset catalog, S3 fetch, ten CI fixtures`
- **Files:** `crates/jope-data`, `catalogs/datasets.json` (fixtures + schema), `tests/fixtures/datasets/**`, `jope fetch`
- **Deps:** P1
- **Verify:** `jope verify-inputs`; hashes match; no Golden42 bulk CSVs.

### P4 — prepare + regenerate views

- **Title:** `feat: prepared-view loader with hash-checked regeneration`
- **Files:** `prepare.rs` extraction, one fixture recipe
- **Deps:** P3
- **Verify:** regenerated training SHA matches catalog (Codex replay contract, small).

### P5 — GP search + champion identity

- **Title:** `feat: native GP fit/search on prepared views`
- **Files:** `crates/jope-search`, grammar JSON, CPU evaluator built from feat-math
- **Deps:** P2, P4
- **Verify:** train-only fit on fixtures 1–6; identity struct tests.

### P6 — champion export wired to CLI

- **Title:** `feat: jope export GP genomes to Python/Rust/C++17/JS`
- **Files:** `champion_export` as library + `jope export`
- **Deps:** P2, P5
- **Verify:** export dir + `parity.json` on a frozen champion fixture (synthetic if no live event).

### P7 — S3 ingest of training corpora (ops PR, may be a runbook not a code PR)

- **Title:** `ops: content-addressed S3 ingest and redundancy census`
- **Files:** ingest script, `catalogs/redundancies.json`, IAM notes in DATA.md
- **Deps:** P3; AWS bucket exists
- **Verify:** unique SHA count; Golden42 42×2 files present by hash; no generated cut CSVs uploaded.

### P8 — Hyperion + foundation + target encode (CPU)

- **Title:** `feat: three-encoder context assembly`
- **Files:** `crates/jope-encode`, extracted Hyperion, export scripts, `catalogs/models.json`
- **Deps:** P4; weights in S3
- **Verify:** hash pins; CPU encode smoke when `JOPE_MODELS_DIR` set; required CI still skips weights.

### P9 — encodings → GP policy (CPU feature)

- **Title:** `feat: policy decode from 2048-d context`
- **Files:** `crates/jope-policy`, grammar mask, tiny fixture weights
- **Deps:** P8
- **Verify:** decode → parseable genome; no accuracy claim.

### P10 — CLI unification + README

- **Title:** `feat: jope fetch/prepare/encode/search/export end-to-end on fixtures`
- **Files:** `crates/jope-cli`, README, ARCHITECTURE.md
- **Deps:** P5–P9 (encode/policy steps skip if no models)
- **Verify:** documented happy path on fixture #1 without `/home/ubuntu`.

### P11 — hardening

- **Title:** `chore: cargo-deny, clippy -D warnings, secret scan, path audit ratchet`
- **Files:** `deny.toml`, CI
- **Deps:** P10
- **Verify:** required gate is the full list in §8.1.

SmartCluster, 8-GPU train, kernel-v1 docker, and neural-cuda are **post-P11** and optional.

---

## 14. Success criteria

The port is done when all of the following are true:

1. `https://github.com/veox-ai/JopeDime` has a green `ci` workflow on `main` from a **clean clone** on `ubuntu-latest`.
2. The clone has **no** `/home/ubuntu` defaults, **no** training corpus beyond the 10 fixtures, **no** `aws.env`.
3. `jope export` emits Python, Rust, C++17, and JS; C++20 compilation of the C++17 file passes on the fixture.
4. `catalogs/datasets.json` lists the training families in §7.4 with SHA-256 and a working fetch path (S3 and/or upstream).
5. `catalogs/models.json` pins Hyperion, foundation, and target hashes; `jope fetch --models` retrieves them from S3.
6. A reviewer can run train-only GP + emit on at least one Golden42 strict-win fixture without the rest of `jain-split`.
7. Redundancy census exists and S3 object count is **unique SHA**, not “every historical copy.”
8. README states, in plain language, that SOTArena ASPFM wins are the GP campaign, and the neural policy is shipped without a held-out win claim.

---

## 15. Open questions (defaults in parentheses)

1. **S3 bucket name and AWS account?** (Create `veox-jope-dime` in the account whose keys currently sit in untracked `aws.env`, then rotate those keys.)
2. **Keep GitHub private?** (Yes, until a public release is an explicit owner ask.)
3. **Include `docs/paper`?** (No.)
4. **GPU self-hosted runner in this port?** (No. CPU GitHub Actions only; GPU is post-P11.)
5. **Fixture #5 license-null `17a4542ee4fb9e44` vs CC0 flights `019de5d76a718163`?** (Prefer confirming license; fallback to flights.)
6. **Keep frozen kernel-v1 docker job?** (Nightly if a public builder image is available; else drop until then.)
7. **Also stand up `jain-datasets` later?** (Not required. S3 + this catalog replaces it for JOPE.)

Owner answers override D1–D15.

---

## 16. Immediate next step after approval

1. Copy this file to `/home/ubuntu/jain-split/jope-dime/CURSOR_GITHUB_PORT_PLAN.md`.
2. Add `*.env` / `aws.env` / `gh.env` to `.gitignore` in the **old** checkout so a stray `git add` cannot publish keys.
3. Execute P1 against the empty GitHub repo.

No training data, recovery trees, or node systemd units move in that step.
