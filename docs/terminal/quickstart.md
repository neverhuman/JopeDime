# Run CSV fitting and source streaming today

These commands describe the implementation at
`45490f2ce33075a355e08844351db197001beb90`. The animated README is a **design
preview**; live progress, ETA, `--ui`, `--show-code`, a TUI and cloud/MCP job
commands are proposed in the [workplan](../plans/terminal-experience-20261021.md).

## 1. Build a development appliance

Run from the repository root on Linux x86-64, glibc 2.36 or newer, with the
repository's pinned Rust toolchain, Bash, Git, `jq` and `readelf` (binutils).
Cargo needs access to its dependencies on the first build. Allow enough space
for the workspace build; the small final executable does not imply a small build.

```sh
git clone https://github.com/neverhuman/jopedime.git
cd jopedime
export CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only
bash deploy/appliance/build.sh --profile fixture --cargo-profile dist-ci \
  --out target/appliance/terminal-demo
APP="$PWD/target/appliance/terminal-demo/jopedime"
"$APP" version --json
"$APP" doctor
"$APP" methods --task regression
```

The build's output directory must be new. Reuse the executable on subsequent
runs. This fixture package trusts a **development key** and is for local
evaluation. No production `jopedime` download has been published at this source
revision. Production signing and the native-only profile are documented in
[appliance deployment](../../deploy/appliance/README.md); do not present the
development package as a production release. An ordinary `cargo build` builds
the research `jope` command, not this packaged appliance.

## 2. Supply train.csv and test.csv

`train.csv` needs a header and a target column; `test.csv` supplies the feature
columns. `--input test.csv` means prediction input, not a scored test set. Select
models using training folds only. This small numeric example is deliberately
regression, with target `y`:

```sh
DEMO_DIR=$(mktemp -d "$HOME/jopedime-terminal-demo.XXXXXX")
printf 'DEMO_DIR=%s\n' "$DEMO_DIR"
cat > "$DEMO_DIR/train.csv" <<'CSV'
x1,x2,y
0,0,0.1
1,0,2.0
2,0,4.1
3,0,6.0
0,1,3.0
1,1,5.1
2,1,7.0
3,1,9.1
0,2,6.1
1,2,8.0
2,2,10.1
3,2,12.0
0,3,9.0
1,3,11.1
2,3,13.0
3,3,15.1
CSV
cat > "$DEMO_DIR/test.csv" <<'CSV'
x1,x2
0.5,1
1.5,2
2.5,3
CSV

"$APP" fit --train "$DEMO_DIR/train.csv" --target y --task regression \
  --method auto --candidates ridge,histgbm,pls --folds 3 --threads 2 \
  --output "$DEMO_DIR/model.jope" --receipt "$DEMO_DIR/fit.json" \
  --json > "$DEMO_DIR/fit.stdout.json"
"$APP" predict --model "$DEMO_DIR/model.jope" --input "$DEMO_DIR/test.csv" \
  --output "$DEMO_DIR/predictions.csv" --threads 2 \
  --receipt "$DEMO_DIR/predict.json"
cat "$DEMO_DIR/predictions.csv"
```

For your own data, change the two paths, target and task. `--task` accepts
`auto`, `regression`, `binary`, or `multiclass`. `--method ridge` selects one
method directly; `--candidates` applies only to `--method auto`.

If you need predictions without retaining a model:

```sh
"$APP" fit-predict --train "$DEMO_DIR/train.csv" --target y --task regression \
  --input "$DEMO_DIR/test.csv" --output "$DEMO_DIR/one-shot.csv" --threads 2 \
  --receipt "$DEMO_DIR/one-shot.json"
```

Useful current controls: `--device cpu`, `--threads N` (`0` chooses available
parallelism), `--max-input-mib`, `--max-rows`, and `--time-limit-seconds`.
The time limit stops starting selection folds; an in-flight fold and the final
refit can finish afterward. It is not a hard wall-clock deadline. The current
human output is a final summary on stderr; `--json` writes the final receipt on
stdout. Keep receipt, model, predictions and input paths distinct. Exit statuses
are 0 success, 2 input, 3 integrity/compatibility, 4 backend, 5 budget, 6 numerical,
and 7 output failure. See [`cli.rs`](../../crates/jope-appliance/src/cli.rs).

## 3. Stream every generated program occurrence

This is a separate **program-search workflow**, not the native model-fitting
command above. It emits training recipes rather than saving `model.jope`.
The existing collector and WebSocket need Linux with `rustc`, `python3`, `g++`,
`node`, GNU `timeout` and `prlimit` on PATH for compiler checks.

Start the collector in terminal A:

```sh
cargo run --release --locked -p jope-observatory --bin gp-source-stream -- \
  --journal-dir "$HOME/jope-stream/journals" \
  --state-dir "$HOME/jope-stream/state"
```

In terminal B, from the repository root, set the existing numeric training CSV
path and a new, empty search output directory. In a new shell, recover the
printed `DEMO_DIR` path from step 2 or replace `TRAIN_CSV` directly:

```sh
TRAIN_CSV=/absolute/path/train.csv
SEARCH_DIR="$HOME/jope-stream/search-$(date -u +%Y%m%dT%H%M%SZ)"
JOPE_GP_SOURCE_DIR="$HOME/jope-stream/journals" \
  cargo run --release --locked --bin jope -- pipeline invent \
  --train "$TRAIN_CSV" --target y --work-dir "$SEARCH_DIR" \
  --budget evaluations=100,seconds=120 --threads 2 --json
```

Connect a WebSocket client to
`ws://127.0.0.1:9876/api/v1/gp/source?after=0`. Complete Rust, Python, C++17
and JavaScript source rows arrive in `source` events; `checked` events follow.
Resume at `after=<last processed sequence>`. Metrics are available with
`curl http://127.0.0.1:9876/api/v1/gp/metrics`. This server binds to loopback;
use an SSH tunnel for a remote host. It checks compilation without executing
the generated programs. C++17/JavaScript coverage refusals stay visible.

Capture adds I/O and can reduce the number of evaluations completed under a
time budget. Journals are outside the checkout; keep them for replay. There is
currently no bundled terminal source viewer. See the full
[stream protocol, limits and client examples](../program-source-stream.md).

## 4. Understand the preview

[terminal-preview.gif](terminal-preview.gif) and its
[generator](render_preview.py) illustrate proposed progress and source views.
All counters, rates, scores and times in the animation are synthetic. The
appliance scene and generated-source scene are different runs. Native fitting must
not advertise generated source when its chosen method has no source artifact.
No model-quality or performance result is established by the animation.

To regenerate with Python 3, Pillow and DejaVu Sans Mono installed:

```sh
python3 docs/terminal/render_preview.py
```

Override `--font` and `--bold-font` if the distro uses different font paths.
The renderer performs no network requests, training or generated-code execution.
