# Private CLI installation and measured demo

The private Linux CLI installation path accepts a local binary and a separately retained
SHA256. It checks the digest before executing the binary, checks ELF architecture and glibc,
rejects a linked Torch/CUDA runtime, then probes `--version`, `doctor`, all four installed command help pages and the read-only
`optimize capabilities` contract. The compiled ASM feature is required; numerical qualification
and available model inputs remain separate from that source capability. It keeps each
binary under its digest in a private installation directory and records an immutable receipt.
Activation changes a managed `bin/jope` symlink; previous binaries and receipts remain available.
A checksum establishes byte identity. Retain the reviewed source and build receipt separately.

Prerequisites are Linux x86_64, Python 3.10+, and `readelf` from binutils. Supply the actual
paths and full digests from the private build receipt:

```sh
python3 deploy/install-private.py binary \
  --binary "$PRIVATE_BINARY" --sha256 "$BINARY_SHA256" \
  --source-sha "$SOURCE_SHA" --prefix "$HOME/.local/share/jope-private"
"$HOME/.local/share/jope-private/bin/jope" doctor
```

The script never downloads a binary, model or dataset, obtains credentials, starts training,
or changes host services. Its receipt records cold installation time; it makes no warm-speed
claim. A missing platform dependency fails before activation.

## Build the CLI from an existing source checkout

Use the repository's reviewed lifecycle and an existing clean checkout at the full source SHA.
The helper rejects a changed checkout before and after compilation, inventories normal/build
dependencies with the exact same `asm` feature selection as the build. It refuses a CLI
dependency on `jope-engine`, `jope-parallel`, Torch or CUDA packages.
It builds only the CLI package and selects the actual executable from Cargo's artifact messages.
Rust and the repository's ordinary CLI build prerequisites are required; Torch is not required.

```sh
python3 deploy/install-private.py source \
  --checkout "$JOPE_CHECKOUT" --source-sha "$SOURCE_SHA" \
  --output "$PRIVATE_CUSTODY/cli-build-UNIQUE" --jobs 4
```

This invokes `cargo build --locked --release -p jope-cli --features asm --bin jope` and reuses the checkout's
existing target directory. It defaults `CARGO_INCREMENTAL=0` and
`CARGO_PROFILE_DEV_DEBUG=line-tables-only` when unset. It neither creates a checkout nor sets
a global `CARGO_TARGET_DIR`. The retained build receipt records the source, toolchain,
feature selection, exact dependency command/graph digest, binary digest, command and cold build time. Install that retained binary
with its recorded digest using the first command.

This CLI-only build does not build or install the separate policy/context worker executables,
checkpoints, accepted transforms or reference corpus. Optimize dependency admission requires
those separately authenticated inputs and a worker-build inventory from their actual build
receipts. An installed `optimize` command is not a claim of a prepared or qualified live loop.

## Prepare real development inputs

Use the existing frozen manifest and preparation workflow. Bulk input materialization belongs
outside the source checkout. Preparation/download time is separate from the warm demo:

```sh
uv run --project python --extra benchmark python benchmarks/prepare.py \
  --manifest benchmarks/panel.json --output "$PREPARED_PANEL"
jope benchmark verify-inputs --manifest benchmarks/panel.json --datasets "$PREPARED_PANEL"
```

Preparation preserves the fixed row assignments, labels, categories, nulls and groups. A real
demo reads only one dataset's verified development partition and fold zero. It never opens
sealed rows. No synthetic input replaces a failed real dataset.

## Measure one fitted demonstration

`jope demo` uses the same bounded Linux worker pool and systemd launcher as the panel
controller. The host needs the already qualified cgroup-v2/systemd capability and an available
exclusive allocation. The CLI handles supervisor details and cancellation; it does not stop
an active CI allocation. Missing capability is an unavailable run, not successful coverage.

Select a real dataset ID from the unchanged panel:

```sh
jope demo --dataset "$DATASET_ID" --manifest benchmarks/panel.json \
  --datasets "$PREPARED_PANEL" --source-sha "$SOURCE_SHA" \
  --output "$PRIVATE_CUSTODY/demo-real-UNIQUE"
```

Or explicitly select a deterministic synthetic fixture:

```sh
jope demo --synthetic multiclass --source-sha "$SOURCE_SHA" \
  --output "$PRIVATE_CUSTODY/demo-synthetic-UNIQUE" --export python
```

The default recipe is ridge for regression, logit ridge for binary, and multiclass ridge for
multiclass; `--method` selects another exact registered task-compatible recipe. This command
currently demonstrates one generic fitted recipe. It records `neural_consumed: false` and does
not establish neural search or thirty-dataset panel results.

Each generic run reserves one CPU and one fit worker, 2 GiB total, no swap and no GPU. The shared policy reserves 128 MiB for the controller and caps its fit at 1920 MiB. The fit is limited to
30 seconds; the total allowance is 60 seconds including five seconds reserved for reporting.
Input verification is timed inside the supervised allowance. Held-out labels stay in the
controller. The command saves the complete private fitted state, reloads it from disk, and
requires unchanged predictions before scoring. It compares against the fitting rows' mean or
class-prior predictor. Scores are RMSE, binary AUROC or multiclass matrix log loss.

The output directory retains the private plan, worker execution journal, `fitted.json`,
`predictions.json`, `baseline.json`, `demo-result.json` and `invocation-receipt.json`.
The invocation receipt includes actual warm elapsed time and an `under_60_seconds` result;
the target is measured on each run. Installation and downloads are excluded and reported
separately. Failed attempts retain failure/execution evidence and never become zero-cost wins.
A fresh demo requires a new directory; it does not overwrite or reset an earlier run.

Optional `--export python|rust|cpp17|cpp20|javascript` emits only allowlisted fitted inference
files and metadata. Unsupported states remain explicit. Emission is marked
`emitted-unqualified` until that exact bundle passes its compile/replay qualification; the
native fitted artifact remains the result regardless of backend coverage. Private recipes
and discovery provenance are not included in public inference files.

For a measured three-task synthetic acceptance after the shared worker host is qualified:

```sh
python3 deploy/ci/qualify-demo.py \
  --binary "$PRIVATE_BINARY" --sha256 "$BINARY_SHA256" --source-sha "$SOURCE_SHA" \
  --output "$PRIVATE_CUSTODY/demo-qualification-UNIQUE" --require-under-60
```

This performs exactly three real fits, one explicit synthetic fixture per task, and checks the
retained fitted hashes, reload evidence, predictions and both finite comparison scores. It
retains all three outcomes even if one fails. Run it only when exclusive full-CPU CI is absent.
No result from this harness is a real-panel result or evidence of GPU training.

## Warm authenticated ASM invocation

The optional ASM demo uses the same qualified context producer, policy worker, 64 neural plus
16 exploration slots, 80 generic slots, training-side selection and bounded execution journal
as `optimize`. It does not install those dependencies or substitute a generic-only success.
The current accepted origin-proof scope supports one frozen regression dataset. The generic
three-task synthetic demonstration remains a separate mode.

First install the reviewed CLI with the `asm` feature. Prepare its benchmark runtime inventory
using the **installed immutable executable's canonical path and SHA**, not a prior build path.
Keep the separately built `jope-policy-producer` and `jope-context-worker` plus the exact
`jope.asm-worker-build-inventory/v1`, build receipts, source manifest, accepted model inputs and
eight-case numerical qualification. A CLI installer receipt does not authenticate worker
executables. `jope optimize capabilities` checks compiled support and always reports
`qualified: false`; the original qualified execution records decide numerical admission.

Prepare an ordinary authenticated base benchmark plan and configuration with the existing
`benchmark plan`/`optimize proof-plan` front doors. Supply a fresh immutable resource JSON with:

```json
{"workers":20,"cpus":20,"memory_bytes":68719476736,"swap_bytes":0,"gpu_tasks":0,"threads_per_worker":1,"total_seconds":60,"reporting_seconds":5,"per_fit_seconds":30}
```

Pin the exact resource-file SHA256. The 64 GiB service allocation is shared: the fixed memory
policy reserves 4 GiB for the controller and caps each fit at 3 GiB. CPU count stays twenty.
Preregister an explicit warm-only configuration with equal arm allowances that together fit
within the total minus reporting reserve. Equal 20-second arms leave 15 seconds for setup
and both refits mathematically, but this is an **unqualified sizing example**, not a measured
successful configuration. In the retained source3fd7 proof, policy production started about
22.1 seconds into the original execution window and took 20.8 seconds; the first fit started
about 50.4 seconds into that same window. Those measurements do not qualify the example or a complete
60-second demonstration. A changed implementation needs a fresh complete measurement.
The panel/proof default remains 120 seconds per arm. The warm route refuses a configuration
whose two full allowances cannot fit, and never silently shortens it.
Actual original arm starts, declared deadlines, available milliseconds and global truncation
are retained in `paired_budget`. Both arms must receive their entire declared allowance;
missing or clipped time makes `matched_comparison: false` and the warm demo fails, even if
both refits happened to finish. Equal clipping is also a failure. No candidate counts or
failure outcomes are reduced to meet the time target.

```sh
jope optimize proof-plan \
  --base-plan "$BASE_PLAN" --base-plan-sha256 "$BASE_PLAN_SHA256" \
  --configuration "$ASM_CONFIGURATION" --configuration-sha256 "$ASM_CONFIGURATION_SHA256" \
  --dataset "$REGRESSION_DATASET_ID" \
  --resources "$WARM_RESOURCES" --resources-sha256 "$WARM_RESOURCES_SHA256" \
  --output "$WARM_ASM_PLAN"

jope demo --asm-plan "$WARM_ASM_PLAN" --asm-plan-sha256 "$WARM_ASM_PLAN_ID" \
  --manifest benchmarks/panel.json --datasets "$PREPARED_PANEL" \
  --output "$PRIVATE_CUSTODY/warm-asm-UNIQUE" --export python --export rust
```

`WARM_ASM_PLAN_ID` is the semantic `plan_sha256` returned by `proof-plan`; resource and runtime
artifact pins are file-byte SHA256 values. The warm timer begins before input/configuration
verification and includes supervisor launch, context production, proposals, real fits,
selection, both outer refits, artifact reload, predictions, scoring and requested export.
Prior installation, dataset preparation and the retained eight-case qualification are cold
preparation and remain separately measured. The supervisor retains its original execution
window; report/reload work gets no extra fit or reporting allowance.

Success requires two full equal preregistered arm allowances, authenticated genuine
neural-origin fits and bounded refinement, both exact selected outer refits, saved complete private fitted artifacts, unchanged reload predictions and finite
controller comparisons. The result retains `asm-fitted.json`, `generic-fitted.json`, their
predictions, `baseline.json`, the complete original `origin-proof.json`, `asm-demo-result.json`
and `invocation-receipt.json`. Selection uses only train-side OOF; the controller reports ASM
versus generic and train-mean development losses without replacing a winner. Optional exports
remain emitted-unqualified or explicitly unsupported; fitted state is preserved either way.

A timed-out policy, missing qualification, absent refit or incomplete refinement fit produces no
successful neural demonstration. Partial actual consumption stays visible in the original
journal. `under_60_seconds` is true only after a successful full invocation measures below 60
seconds; the source implementation alone establishes no speed claim. A fresh demonstration
uses a new output directory outside Git checkouts and `/tmp`. It cannot reuse an expired
origin-proof journal. This one-dataset operational demonstration is not a thirty-dataset panel
result, a held-out quality claim or training proof.

The cold source build receipt also retains checksummed dependency inventory, Cargo metadata,
artifact messages and compiler diagnostics. Failed compilation writes a private failure
receipt with the actual nonzero exit and elapsed cold duration. The checksum-pinned binary
installer still establishes caller-pinned byte identity, not a signature or independent source
attestation.
