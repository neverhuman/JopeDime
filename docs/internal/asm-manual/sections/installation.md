# Installation, preparation, assets, workers and doctor

## Scope and command surface

### Product, checkout and CLI evidence

Algorithm Synthesis Model (ASM) means **“learns to generate and optimize executable algorithms from data.”** The product is the private `veox-ai/JopeDime` repository. On this host its canonical checkout is `/home/ubuntu/JopeDime`; the similarly named research repository at `~/jain-split/jope-dime` is a different upstream.

This chapter is pinned to source `68f56404c3e199e06413bfd645d3fe0803d7b96f`, the reviewed PR83 head. PR83 merged it on 2026-09-23 at 16:34:47 UTC as main `a75e543d2166…` with the same Git tree (`176b75bc36d7…`); as of 2026-09-23 22:00 UTC no later change had merged to main. Sixteen actual version/help invocations against its immutable CLI completed successfully. The executable reports `jope 0.1.0`; its SHA256 is `eab2d5b1b05629020aab3d540d51eb24ba0db18ff0917da188f89a34b6dc5e2b`. The machine-readable evidence is `agent-maintainer-manual/cli-help-68.json` under the ASM custody root (`C` in [§](#physical-layout-and-identities)); the `68` in its name abbreviates source `68f56404`. A version string alone cannot distinguish later builds; use the full source, executable digest and build receipt.

The source inventory, a help page, an executed fixture, exact-head CI and a live operational qualification establish different things. This chapter records interfaces and operational requirements. It does not turn an unexecuted command example into measured capability. The maintainer-private examples were executed at the same source: all fourteen commands passed. Their separate receipt and scope limits appear in [§](inference.md#measured-execution-of-the-private-examples) and `data/example-acceptance/`.

### Command surfaces and the stopped integration order

| Surface | Present in the immutable 68f56404 CLI | Remaining distinction |
|---|---|---|
| `doctor` | Yes | Static capability inventory; it does not inspect every runtime dependency or qualify a live campaign. |
| `benchmark verify-manifest`, `verify-inputs`, `methods`, `plan`, `run`, `report` | Yes | Planning performs zero fits. Run/report require exact admitted resources, inputs and journal identities. |
| `optimize capabilities`, `proof-plan`, `plan`, `run`, `report` | Yes, execution feature-gated | A compiled ASM feature is separate from qualified encoding, genuine neural-origin fits and successful evolution. |
| `fit`, `emit`, `search`, `pipeline run` | Yes | Legacy numerical regression/recipe paths; their outputs are not complete frozen-panel fitted inference exports. |
| `encode`, `champions`, `train-step`, `route` | Yes | `train-step` trains a local centroid router, not the bounded neural policy campaign. |
| `datasets`, `models` | Yes | Existing catalog/model tools; a fetchable file is not automatically admitted as an ASM input. |
| `train prepare-inputs`, `plan`, `run`, `resume`, `export`, `verify` | Pending combined training overlay | Includes native admission, actual updates/reload/continuation checks and experimental exports. See [§](training.md#the-bounded-training-campaign-chain). |
| `demo` and private CLI installer | Pending product overlay | Includes generic fitted demos and authenticated paired ASM demos; actual speed and parity must be measured. |
| `benchmark export-winner`, `qualify-export` | Pending winner add-on | Preserves the selected native state and ranking; backend qualification needs actual supervised execution. |
| `motifs mine`, `validate` under the umbrella CLI | Not present at 68f56404 | Separate preparation exists; production eligible-State reading, mine command and umbrella integration were identified as remaining work. |

The stopped integration order is combined **training64** (including regression consumption, accounting and coverage), then **product28**, then unchanged **warm-target** and **winner-export** add-ons. A later two-file regression proof-gate patch (`0e634858…` under `agent-training-proof-gate`, stopped on 2026-09-23 at 19:30 UTC) comes after them: generate the training, product and add-on replays before applying it, because the downstream guards pin every training postimage. The whole queue waits until the manual's own publication lifecycle closes ([§](maintenance.md#updating-the-book)). The add-on refresh is pinned under `agent-delivery-refresh-68f56404`; two strict replays produced identical nine-file postimages, with 95 canonical guards. These are custody patches, not installed behavior. Never apply the training coverage/accounting constituents a second time or copy an older loose installer postimage over its corrected form.

## Physical layout and identities

Keep source, bulk inputs and evidence in separate locations. The examples below use these names; resolve them from actual receipts before execution:

```sh
R=/home/ubuntu/JopeDime
C=/home/ubuntu/jope-rescue-20260921/asm-integration-20260923
D=/home/ubuntu/.cache/jope/asm-private-30-v1
M="$R/benchmarks/panel.json"
# S: full reviewed source SHA; J: physical immutable CLI path.
# A: new durable private acceptance directory, outside Git and /tmp.
```

`JOPE_HOME` chooses the source/assets root; `JOPE_STATE` chooses mutable local state; `JOPE_REGISTRY` chooses the dataset registry. Those roots are resolved at call time. Pass explicit paths when producing reproducible evidence. Do not repurpose `HOME` or globally override `CARGO_TARGET_DIR`. A managed installation symlink is convenient for interactive use; a campaign pins the physical executable path and checksum so changing the symlink cannot silently replace its worker.

There are two different hashes in the CLI contracts. An artifact SHA256 hashes exact file bytes. A campaign's `plan_sha256` is its validated semantic identity, printed by planning. Do not substitute `sha256sum plan.json` for that semantic ID. Worker/build/source inventories have their own schemas and pins; preserve each rather than renaming one receipt into another type.

Campaign journals require absolute durable paths outside Git working trees and `/tmp`. Private directories are restricted by the front doors. Do not publish entire run directories: they can include raw development features, labels, recipes, private context, model state and provenance.

## Build and install

### Development build of the CLI

The current development build path builds the Rust CLI from the existing checkout:

```sh
cd "$R"
CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only \
  cargo build --locked --release -p jope-cli --features asm --bin jope
```

Allocate build/test resources through the repository's existing coordination before executing it. Use one reused `target/` per claimed checkout. The CLI's `asm` feature exposes the private Linux execution path; a default-feature build may show the command while reporting ASM execution unavailable. The ordinary CLI should not require the monolithic engine, Torch or CUDA. Separate context and policy workers and their authenticated model inputs are still necessary for a live neural run.

### Private installer after integration

After the pending installer is integrated, prefer its measured source-build and checksum-install front doors:

```sh
python3 "$R/deploy/install-private.py" source \
  --checkout "$R" --source-sha "$S" --output "$A/cli-build" --jobs 4
python3 "$R/deploy/install-private.py" binary \
  --binary "$PRIVATE_BINARY" --sha256 "$BINARY_SHA256" \
  --source-sha "$S" --prefix "$A/install"
```

Read the binary path and digest from the actual source-build receipt. The source command checks the checkout before and after compilation, uses the same `asm` selection for dependency inventory and build, rejects unwanted engine/Torch/CUDA dependencies, and selects the executable from Cargo artifact messages. Its receipt includes the toolchain, dependency graph, features, source identity, compiler output, executable digest and measured cold duration. Compilation failures retain failure receipts.

The binary installer requires Linux x86_64, Python 3.10 or later and `readelf`. It validates the supplied checksum before executing the binary, checks architecture/glibc/dynamic dependencies, probes version and command help, and activates a digest-addressed private binary through a managed symlink. It does not download models or datasets. A caller-supplied checksum proves byte identity; retain independent source/review/build provenance as well. Cold build time, cold installation time, data preparation and warm demonstration time are separate measurements.

The installer's `jope.private-cli-source-build/v1` receipt does not replace a `jope.asm-cargo-build/v1` worker receipt or `jope.asm-worker-build-inventory/v1`. Build and retain the context/policy workers, source manifest, accepted assets and native context qualification through their owning workflow. The native canonicalizer used by experimental proposal admission must match the final source/build and physical executable bytes.

## Prepare the panel and external runtimes

### Prepare and verify the panel

The approved panel contains ten regression, ten binary and ten multiclass datasets. Its exact manifest SHA256 is `ba7dace4cd535687002cc95538dbc7f7aaf77ec973177c7159b3c6d22471eee0`. Reformatting the JSON changes its bytes and is refused even when its apparent meaning is unchanged. Do not replace a failed dataset with an easier fixture.

```sh
uv run --project "$R/python" --extra benchmark python "$R/benchmarks/prepare.py" \
  --manifest "$M" --output "$D"
"$J" benchmark verify-manifest --manifest "$M"
"$J" benchmark verify-inputs --manifest "$M" --datasets "$D"
"$J" benchmark methods
```

Preparation must preserve actual frozen row assignments, original labels, category strings, nulls and groups. `verify-inputs` verifies development bytes and schema, not just file existence. The benchmark does not open the sealed evaluation partition. An operational proof of one regression family retains the original thirty-dataset manifest but selects a separately declared proof scope; it does not modify the official panel.

### Native roster and external comparator runtimes

The native roster has 17 regression recipes, five binary recipes and four multiclass recipes. Aliases do not create extra screening slots. The second GPR configuration is distinct because it has a distinct fitted implementation. Every task also includes four external adapters, each run at library defaults with seed 0 and one thread: XGBoost, LightGBM, CatBoost and a scikit-learn random forest. Native recipes are fixed before the dataset screen and the external adapters keep those defaults, so the campaign performs no per-dataset hyperparameter search for any non-ASM comparator (a head's own internal fitting, such as `ridge_gcv`'s generalized cross-validation, is part of its fixed recipe; [§](benchmark.md#minimum-complete-future-campaign-report)), while each ASM arm searches within its declared allowance. The highest-rated methods on the historical benchmark-valid ladder, such as TabFM Ensemble, are not panel comparators ([§](gp-targets.md#sotarena-published-and-corrected-ladders)); panel wins against this roster would therefore not show that ASM beats those methods. Sources: `crates/jope-bench/src/panel/campaign_cli.rs:67`; `crates/jope-bench/adapters/sklearn_rf_adapter.py:2`; `crates/jope-bench/src/panel/selection_exposure.rs:91`; `crates/feat-math/src/heads.rs:65-67`. External library versions, entry points, environment and package bytes must be authenticated before admission.

```sh
python3 "$R/deploy/prepare-panel-runtime.py" \
  --binary "$J" --native-command benchmark \
  --build-receipt "$CLI_BUILD_RECEIPT" --source-sha "$S" \
  --python "$EXTERNAL_PYTHON" --site-packages "$EXTERNAL_SITE_PACKAGES" \
  --adapters "$R/crates/jope-bench/adapters" --output "$A/runtimes.json"
"$J" benchmark plan --manifest "$M" --datasets "$D" --source-sha "$S" \
  --runtimes "$A/runtimes.json" --output "$A/base.json"
```

The preparation helper checks actual dependencies/constructors and pins the runtime. It is not a completed fit comparison. Retain unavailable-dependency reasons rather than dropping methods. Prior inventories observed XGBoost 3.4.1, LightGBM 4.7.0, CatBoost 1.2.10 and scikit-learn 1.7.2 on this host; these historical metadata observations do not authenticate a later environment. A new package installation invalidates old package-byte inventory. Keep the external CPU environment separate from the dedicated CUDA training environment.

Omitting `--runtimes` from `benchmark plan` produces a verified Rust screening inventory with zero fits; it is not the complete admitted ASM/comparator campaign. The runtime-aware plan is the base for `optimize plan`.

## Stored model assets and the active encoder distinction

### Bundle manifests and declared asset sizes

The metadata-only S3 audit on 2026-09-23 fetched the three bundle manifests below and pinned their returned bytes, ETags and lengths. No model bodies were downloaded for this chapter. These are declared object sizes from the retrieved manifests; they do not measure resident memory, compute cost or a new numerical qualification.

| Bundle asset | Bytes | Role in this chapter |
|---|---:|---|
| `policy-step-60000/policy.safetensors` | 111,199,088 | Accepted policy export identity, consumed only with its checkpoint binding |
| `policy-step-60000/context_transforms.safetensors` | 8,413,808 | Accepted fixed context assembly transforms |
| `jope-prime-gen31-v2/model-bundle/hyperion_v5_arm_b.safetensors` | 6,650,692 | Exact Hyperion weights pinned by the current CPU context producer |
| `jope-prime-gen31-v2/starforge/starlight_core_regressor.safetensors` | 233,183,532 | Foundation component; manifest describes512 hidden features pooled into2048 |
| `jope-prime-gen31-v2/model-bundle/target_encoder_fused_v1…safetensors` | 78,615,176 | Target component; manifest describes normalized64+64+384 fused representation |
| `hyperion-v7/model.safetensors` | 246,456,988 | Separate stored bundle, not the accepted context producer's pinned weights |
| `jope-prime-gen31-v2/model-bundle/jope_policy.safetensors` | 108,846,812 | Separate gen31 policy artifact; not step-60000 |

The three current model bundle manifests are `hyperion-v7`, `jope-prime-gen31-v2`, and `policy-step-60000`. Their existence and object sizes do not establish new training, inference parity or serving promotion. [§](status.md#the-identity-tuple) gives their roles among the product identities, and [§](maintenance.md#release-and-installation-are-independent-from-model-promotion) gives the acceptance boundaries. Historical lineage-A material in learning storage cannot replace its missing canonical training-state chain.

### Encoder identity pitfalls and provenance declarations

The storage name `hyperion-v7` must not be confused with the CPU implementation's `V7Model` Rust type. The actual current producer pins weight hash `88b9e834…`, which matches the v5 arm-B asset, with row width 128. The separate stored v7 manifest describes dimension512, eight heads,12 layers and six bands. Replacing the pinned asset based on its newer-looking name changes the encoder identity and invalidates compatibility.

Two identity pitfalls matter when maintaining this boundary. Hyperion `load_settings` reports a legacy 16-hex identifier, not a SHA-256 digest. The producer therefore authenticates manifest bytes before and after the path-based settings load, authenticates weights before and after `V7Model::load_cpu`, and requires embedding width 128. A legacy identifier must never be compared with or substituted for a cryptographic admission pin. Separately, ordinary `jope encode` resolves `HYPERION_V7_WEIGHTS` as a **bundle directory**, else `$JOPE_HOME/models/bundles/hyperion-v7` (relative to the working directory when unset). The default stored bundle is 512-dimensional and differs from the pinned v5-arm-B producer; both manifests use the loader string `version: v7`, so that string cannot establish checkpoint identity. `jope encode` is not the qualified 2048-value ASM context interface. Current documentation drift remains at `docs/models.md:105`, which supplies a file instead of the expected directory, and the default-v5 comment in `crates/hyperion/src/hyperion.rs:28`. Use exact asset hashes and the qualified context-worker flow.

The retrieved manifests also contain internal/redistribution/provenance declarations that vary by artifact. They are retained evidence about the bundle metadata, not proof that every object is approved for every distribution. The active private run's exact release/input inventory remains the authority for which files it used; S3 presence alone does not establish serving eligibility. Sources: `crates/jope-policy/src/context_producer.rs:41`; custody `metadata-receipts/veox-jopedime-models-use2--bundles--policy-step-60000--manifest.json`, `…--hyperion-v7--manifest.json`, and `…--jope-prime-gen31-v2--manifest.json`.

## Bounded workers: the context worker and the policy producer

### Context worker contract

In the bounded ASM flow, encoder weights are loaded inside `jope-context-worker` (`jope-policy`, feature `context-producer`). `--version` prints `jope-context-worker/1`. Its exact eight named arguments are:

```text
jope-context-worker --mode produce|qualify-case --input ABSOLUTE_PATH --input-sha256 HEX64 --assets ABSOLUTE_ENCODER_FILES_JSON --assets-sha256 HEX64 --source-sha40 HEX40 --producer-sha256 HEX64 --output ABSOLUTE_PATH
```

Extra arguments such as `--dataset-root` are refused. The child requires OMP, OPENBLAS, MKL, BLIS, NUMEXPR and RAYON thread limits of 1, `VECLIB_MAXIMUM_THREADS=1` and `CUDA_VISIBLE_DEVICES=-1`; it authenticates its own executable and the component closure before model loading. The pinned bundle manifest begins `5e7884ec…`; each inventoried file/notice is verified with a 512 MiB per-file limit. Outputs are create-new, mode 0600 and fsynced. Stdout records `jope-context-worker/1 <output_sha256>`. A failed `qualify-case` comparison retains `<stem>.unqualified.json`, exits nonzero and does not create the requested successful output.

For current `jope optimize`, input derives from `dataset.cut(0)` fitting rows and is retained at `<journal>/asm-context-inputs/<input_sha256>.json`. Admitted output is `<journal>/asm-contexts/<neural-arm-id>/context.f32` plus `bundle.json`; the vector contains 2,048 little-endian float32 values. Invoke this through the supervised planner/controller contract, not as an unsupervised shortcut.

### Policy producer contract

`jope-policy-producer` is a separate pinned bounded job. Its `asm.private-policy-producer/v1` request binds worker source, prepared/training/context hashes, cut, accepted checkpoint state, policy/vocabulary hashes, three RNG seeds and sampling shape. It accepts only the pinned step-60000 weights (`4b4b5a40…44f1e`) and state, rounds 1–8, batch 1–128 and a finite positive temperature. Context input is exactly 8,192 bytes, finite and clipped within [-32,32]. It requires either a qualified CPU context bundle or an authenticated original `parallel.encoding.v1` receipt with the original LibTorch lineage, train-side partition and `final_test_targets_used:false`. Frozen transforms are pinned to `1b779c6f…6e22`.

Its output must be an absolute physical-parent path outside `/tmp`, created new with mode 0600. The final stdout JSON declares `numerical_backend:candle_cpu_float32` and `native_cuda_parity_qualified:false`. Reading the envelope proves consistency, not execution: consumption also requires the completed pinned worker/output receipts. Replacing accepted weights, transforms or vocabulary is not a free configuration substitution.

The producer shares the 30-second worker cap and original 120-second neural allowance; the accepted proof used a 3 GiB worker allocation. The pre-cache proof at source `55e9ed688faf` was killed at 30.004805335 seconds, before neural fits. With cached decoding, later measured producer wall times were 20.832133058 seconds at source `3fd7d5b2ebb6`, 20.651920435 at `c5d11ffca92c` and 20.486968041 at `68f56404c3e1`. These leave about 9.5 seconds of per-job headroom in the last measurement, not a portability guarantee. Sampling-shape or host changes need another bounded proof, and the fixture-only decoder-equivalence limits of [§](architecture.md#component-dimensions-and-assembly) still apply.

Sources: `crates/jope-policy/src/context_producer/worker.rs:28`, `:188`, `:255`; `crates/jope-policy/src/producer.rs:16`, `:91`; `crates/jope-policy/src/bin/jope-policy-producer.rs:13`, `:78`, `:121`, `:172`; `crates/jope-bench/src/panel/optimize.rs:454`, `:705`; retained `proposal-execution.json` receipts.

## Doctor checks and troubleshooting

### Reading identities from the immutable CLI

These metadata commands do not train or load a checkpoint:

```sh
R=/home/ubuntu/JopeDime
C=/home/ubuntu/jope-rescue-20260921/asm-integration-20260923
J="$C/neural-integration/live-build-68f56404c3e199e06413bfd645d3fe0803d7b96f/bin/jope"
"$J" --version
"$J" doctor
"$J" optimize capabilities
git -C "$R" rev-parse HEAD
sha256sum "$J" "$R/Cargo.toml" "$R/python/pyproject.toml" "$R/benchmarks/panel.json"
```

Do not interpret `qualified: false` in capability metadata as a missing command, or compile-time availability as completed runtime qualification. The measured executable SHA is `eab2d5b1b05629020aab3d540d51eb24ba0db18ff0917da188f89a34b6dc5e2b`; a different executable needs its own receipt.

### Troubleshooting and safe recovery

| Symptom | Meaning and next action |
|---|---|
| `compiled: false` or ASM feature unavailable | Use the reviewed Linux CLI built with `asm`; then separately qualify its model inputs and workers. |
| Running binary differs from pinned native runtime | Resolve the installed physical binary, verify its hash and prepare a new pre-execution inventory/plan; do not alter a running plan. |
| Frozen manifest or development checksum differs | Stop admission, preserve both files and find the preparation/source mismatch. Do not regenerate folds or substitute datasets to make a check pass. |
| Missing external dependency/compiler | Retain unavailable status and inventory. Install/qualify before opening a new authorized experiment; never drop the method or replace a winner. |
| Context/reference/worker identity mismatch | Rebuild or requalify through the owning source-bound workflow; a copied success flag is insufficient. |
| Deadline expired | Recover/report the original window. A repeated command cannot renew it. Any new experiment needs its own explicit scope and resource declaration. |
| Started job lacks a finished receipt | Treat execution cost/status as unknown; let the shared pool perform terminal recovery/drainage. Do not blindly retry the child or delete its artifacts. |
| No complete five-fold winner | Keep incomplete/failed confirmations visible. Do not select a different recipe merely because it has an emitter. |
| Warm result exceeds 60 seconds | Preserve fitted artifacts, comparisons and the measured miss of the 60-second goal. Do not relabel elapsed time or omit preparation performed inside the warm invocation. |
| S3 access stalls at instance metadata | On the host, select the configured credential-process profile such as `AWS_PROFILE=veox-jepson-prod`; never source the bare-token credential note files. Runners have no AWS credentials. |
| No eligible training families/no-signal admission | Preserve refusal; obtain real eligible data under the declared protocol before GPU training. Do not use development/test gain as a substitute for the train-side OOF admission rule. |
