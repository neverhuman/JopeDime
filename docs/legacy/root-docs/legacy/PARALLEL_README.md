# JOPE parallel

Rust GP teaching and neural policy training on the eight-A100 node. Start the live dashboard from the source machine:

```bash
rtk proxy /home/ubuntu/jope-dime/parallel/tools/tui-remote.sh
```

Use **1–5** or **← →** to switch views. **↑ ↓ / Page Up / Page Down** scroll the GP workers. **q** detaches; background services keep running. A 160 × 55 terminal shows the full hardware and learning views; resizing is supported down to 48 × 15.

The Overview reads durable context manifests, live GP receipts, neural steps and checkpoint receipts, and `runs/cycle-progress.json`. It shows progress through encoding, GP teaching, policy training, neural-seeded GP, and retraining. Completed blocks remain visible as saved receipts; stale active workers are marked. Hardware samples refresh once per second and the interface renders four times per second.

## Work and resource assignment

- Encoding and neural blocks use all eight GPUs. Hyperion's latest concurrent 3,500-cut block measured 82–85% mean utilization on each GPU while all 104 GP workers remained active; host CPU usage averaged 88.6%, with zero failures. The earlier exclusive encoding block measured about 93%. Target and foundation encoders finish much faster; these utilization measurements apply specifically to Hyperion.
- GP currently uses 104 bounded workers with one CPU thread each. Training and the target/foundation encoders use 16 CPUs, leaving four host CPUs reserved. Hyperion's exclusive preprocessing phase can use 120 CPUs. The controller pauses admission for stale hardware telemetry or insufficient memory/disk headroom.
- The current prepared workload is **10,992 distinct cuts from 78 regression datasets and 72 source families**: 7,000 original cuts, 3,692 new grouped cuts after skipping 308 duplicate memberships, and 300 validation cuts. The registry contains 108 reviewed datasets in total; 30 are withheld from this preparation pipeline. The broader approximately 2,330 regression candidates remain provisional.
- All **10,992 full 2,048-dimensional contexts are saved and hash-verified**. `data/encodings/contexts-assembly-v2-all.json` points to their immutable files; it does not duplicate vectors. The final 300 Hyperion components passed recovery parity using a smaller native batch cap. The TUI reads the 10,992-cut denominator from immutable job manifests. The v2 projection uses only 55 meta-training families, with equal family mass; all source assignments are checked against registry v4. Original v1 vectors and checkpoints remain unchanged.
- Two earlier qualification cycles completed DL → GP → DL: 396 GP comparisons and 132 fitted predictors. The second cycle retrained on 641,151 scored program examples. At matched evaluation counts, neural seeds had 1.56% higher RMSE than generic seeds across three validation families; a performance advantage is not established. Reports are saved under each cycle’s `report-v1/`. An injected rank failure resumed to byte-identical parameters against an uninterrupted eight-GPU run. Results remain qualification work with adaptive cut targets.
- The requested final 24-hour pilot has not started. Complete provenance/transfer reconciliation and final evaluation/reporting remain necessary. A separate 64-family qualification selection is frozen at 48 training / 8 validation / 8 test families; its use of the 55-family projection is explicitly recorded. Additional random cuts increase teaching variety without increasing independent-family counts.
- The frozen 64-family cohort contains 8,799 context pointers. Its admitted starting teaching corpus contains **779,158 programs across 1,620 cut groups**, covering all 48 training and eight validation families. Meta-test labels are excluded from neural supervision. `runs/parity/cohort64-initial-corpus.json` binds the verified payloads.
- The managed **24-hour qualification campaign started at 2026-09-08 04:10:03 UTC**, with a fixed deadline of 2026-09-09 04:10:03 UTC. `jope-parallel-campaign.service` runs repeated cycles on fixed cut anchors; `jope-parallel-reports.service` publishes matched-budget comparisons after each completed cycle. The v44 release passed actual eight-GPU rank-failure recovery with identical parameters and two-block auxiliary optimizer continuation. This campaign is separate from the full independent-evaluation pilot above.
- Its first complete cycle finished at **2026-09-08 05:01:15 UTC**: 336 searches, 112 verified fitted predictors, and a second training block on **1,114,934 scored programs**. The next cycle started automatically. At matched evaluation counts, neural seeding had 0.42% higher validation RMSE than generic seeding across eight families; an advantage remains unestablished. The immutable results are in `runs/regression-campaign64-qualification-v1/cycle-0000/report-v2/`; the original `report-v1/` is preserved. The second report adds the proposal-input scope qualification below, with all previous measurements unchanged.
- The live v44 trainer completed its first 3,000-step block in 199 seconds versus 258 seconds for the recorded control, approximately 30% more training examples per second. Mean GPU utilization during that block was 63.2%; utilization is not continuously at the 80% target. GP fitting is currently the cycle bottleneck. Auxiliary learning stops when its fixed validation panel plateaus, retaining its best checkpoint. See `runs/parity/training-performance-v44/` for loading, communication, checkpoint, and hardware measurements.
- A separate v49 candidate passed exact parameter parity through 500 steps and injected eight-rank recovery. Its matched 500-step optimizer loop was 5.8% faster than v44, with 60.7% mean GPU utilization in that short test. It remains a qualified candidate; the running campaign keeps its pinned v44 executable. See `runs/parity/ablation-pairwise-v49-performance/`.

## Storage and evaluation contract

Originals stay immutable. `data/prepared/views-expanded-20/` and `views-extra-20/` store source pointers, hashes, exact row indices, deterministic seeds, and training-only preprocessing recipes; workers materialize numeric views in memory. No cut CSV copies are retained. Repeated cuts do not create additional independent source families.

Each cut's held-out partition is deliberately optimized by GP to produce teaching targets. These scores are labeled adaptive and are not independent estimates of generalization. Encodings use only that cut's training rows. Original final tests remain sealed until a selection receipt authorizes scoring. Saved encodings bind the training rows, preprocessing, encoder identities, and vector hashes. The combined context projection/normalization is fitted on meta-training families only.

The self-play upgrade keeps a durable best-found program for each exact cut and evaluator identity under `runs/champions/`. Both comparison arms use one frozen incumbent snapshot. Programs inherited from an older evaluator are refitted before receiving new scores. New searches evaluate retained teachers before neural candidates, protecting the incumbent from the search time limit. Neural sampling includes the strongest teacher in each eight-program comparison group. These mechanisms preserve and learn the best program found under the recorded budget; they do not establish a global optimum.

The policy scores a fixed candidate pool using its learned reward heads and saved training context. Its output is a relative quality prediction, not a calibrated RMSE estimate. The saved pool's input scope depends on the protocol described below. While GP evaluates frozen proposals, a separate eight-GPU auxiliary run continues learning in bounded blocks. A fixed family-balanced validation panel selects its checkpoint; stalled or worse learning does not replace the selected model. The next training block incorporates newly scored GP programs.

The v44 proposal pool excludes duplicates of retained teachers before ranking. Consequently, its `best-predicted.json` describes the best program within that filtered pool; the pool can depend on teachers selected using the cut's held-out targets. Raw neural conditioning and scoring a fixed pool use the training context, but an end-to-end prediction independent of those teacher identities is not established for v44. `runs/acceptance/proposal-training-input-scope-v44.json` records this limitation without changing previous artifacts.

The separate v53 runtime fixes this with `pure-ranking.json` and `pure-best-predicted.json`, followed by explicit GP population assembly and overlap receipts. Its eight-GPU acceptance test removed 224 retained-teacher entries: all 112 pure rankings and predictions remained byte-identical while 99 GP populations changed. All populations remained unique and preserved the 64/10/6 composition. Resume preserved saved pools; old v44 outputs were rejected before GPU work. See `runs/parity/pure-proposals-v53/report.md`. The live campaign retains its pinned v44 runtime; the separate proposal library now uses v53. Updated reports distinguish these protocols.

An opportunistic proposal library under `runs/gp-reviewed3/neural-breadth-v44/` reuses the saved encodings and frozen checkpoint between learning blocks. Its first bounded run saved 3,204 pools, including 3,071 in complete chunks and 133 in a preserved partial chunk, then released the GPUs when the campaign reached its admission cutoff. Saved seed pools become teaching examples only after GP evaluates them. `runs/proposal-library.json` reports generation; GP admission is recorded separately. This run averaged about 19% GPU utilization and 89% CPU utilization, so proposal throughput remains limited by catalog loading and generation overhead.

All 3,071 complete pools were subsequently admitted through a protected boundary before the next cycle's GP admission. The GPU scheduling lock was held for 3.765 seconds during verification and queue insertion; no GPU computation occurred in that interval. The next cycle's 336 comparison jobs precede pending library jobs. The 133 partial pools remain unadmitted. `runs/proposal-library-admission-v44-boundary1/complete.json` binds the transaction, and the TUI counts matching GP progress separately from saved seeds.

The separate v53 library under `runs/gp-reviewed3/pure-neural-breadth-v53/` saved **3,933 pure prediction pools / 251,712 ranked program slots** across nine complete chunks, using the next cycle's frozen checkpoint and existing encodings. It released all eight GPUs when the main GP comparisons reached the 90% cutoff. Its first chunk independently verified all 446 pools, their eight rank completions, and every prediction/assembly hash chain. Teacher overlaps are recorded separately from within-policy duplicates. The full 254-second interval averaged 16.7–17.5% utilization per GPU and 89.0% host CPU utilization, with a maximum GPU temperature of 42°C and no hardware errors. This library has **zero GP admissions**; these saved predictions are not additional evaluated teachers. `runs/pure-proposal-library.json` has its own dashboard row, separate from the original library's 3,071 admitted jobs. No encodings were recomputed and no cut tables were copied.

## Isolated seed comparison and binary qualification

The September 8 experiment freezes the selected cycle-0010 checkpoint and its **2,994,560-record** corpus under `runs/seed-quality-experiment-v1/`. It reconstructs 11 distinct checkpoint-selection panels across 23 ancestry states before choosing 16 unused decision cuts from the eight validation families. All six rankings precede outcome access and retained-teacher assembly. Pools contain 1,198–1,215 unique programs, including 1,024 exact meta-training champions. Decoder-free scoring matched full-forward reward/runtime outputs exactly on all 16 reference pools, with unchanged ordering.

The primary result is negative: larger-pool best-of-eight RMSE is **9.08% higher than current selection** (3 family wins / 5 losses) and **8.61% higher than generic seeds** (5 wins / 3 losses). All 744 distinct primary candidate attempts completed: 722 successful fits and 22 30-second timeouts. Failed primary candidates are noncompetitive; secondary best-of-16/64 results require successful completion of every candidate. The initial stricter sensitivity report and measurement-policy clarification are preserved. Production seeding remains unchanged. `report.json`, `downstream-report.json`, and `meeting.md` contain the final coverage, costs, and downstream results.

All 48 downstream searches completed. Larger-pool RMSE was 0.18% higher than current and 0.007% higher than generic, so the no-regression gate also failed. Fourteen of its 16 winners were retained teachers; one evolved and one direct selected program won. The direct selected winner improved both controls on one cut, without improving the aggregate result. Sixteen searches resumed after the production handoff; their native clocks include 480 seconds of conservative interrupted-work allowances in total.

Adult, German credit, Ionosphere, and Sonar now have **80 verified binary contexts and 240 raw components**. The preferred separate index is `runs/binary-qualification-v1/deterministic-v2/index.json`; the original batched index remains preserved at `runs/binary-qualification-v1/contexts/index.json`. Originals were staged once; cuts store references and indices, with training-only preprocessing. Cross-task aliases are frozen, and Sonar's unresolved acquisition/entity grouping remains explicit. The existing projection is reused for binary qualification; final tests remain unopened. Successful initial and expansion runs took 49.3 and 39.4 seconds; isolated dependency setup failures and their costs are recorded separately.

Independent recomputation of the initial 20 binary views reproduced target and foundation components byte for byte. Timing-dependent Hyperion batching introduced small differences; that failed exact check is preserved in `reproducibility/verification.json`. Serial dispatch of the same frozen encoder then passed **20 independent Hyperion/context replay pairs with zero error** and published all 80 contexts separately. It reused target/foundation components and the frozen projection. All 19 completed stages skipped on restart, with 204 artifact hashes and modification times unchanged and no GPU work. Serial qualification/publication took 128.5 seconds. See [the binary workflow](tools/BINARY_QUALIFICATION.md) for `--serial-hyperion`, frozen launch policy, semantic environment constraints, and qualification limits.

Run from the target node. The manifest retains the frozen v56 evaluator and neural checkpoint; the v62 comparison launcher forwards termination directly to its controller:

```bash
rtk proxy python3 tools/propose_ranked.py --root /home/ubuntu/jope-dime/parallel --manifest runs/seed-quality-experiment-v1/manifest.json --binary bin/jope-parallel-neural-v56-experiment
rtk proxy bin/jope-parallel-cpu-v62-experiment --root /home/ubuntu/jope-dime/parallel compare-seeds --manifest runs/seed-quality-experiment-v1/manifest.json --workers 16 --prefix 64 --seconds 3600 --downstream
rtk proxy python3 tools/seed_meeting_report.py --root /home/ubuntu/jope-dime/parallel
```

`propose-ranked --manifest --rank --world` is the native per-rank interface; the Python wrapper owns the eight-GPU lease, cancellation, and resume. `assemble-ranked --manifest` checks all rankings before consulting retained teachers. `compare-seeds` writes experiment results and never records production champions. The scheduler shares one 16-CPU allocation (0–7 and 62–69) across comparison and binary work and yields to the campaign. The TUI launcher uses its separate v62 executable and shows the selected checkpoint, current corpus, export/packing phases, seed quality, binary coverage, and resource wait reasons.

## Operations

The node is `ubuntu@158.101.20.226`; its workspace is `/home/ubuntu/jope-dime/parallel`. `jope-parallel-monitor.service` collects telemetry, `jope-parallel-gp.service` grows the teaching corpus, and the oneshot `jope-parallel-encodings.service` / `jope-parallel-contexts.service` run frozen encoders. A completed oneshot service can be inactive without indicating failure.

```bash
rtk proxy ssh -i /home/ubuntu/.ssh/lambda_cloud_a100_20260907_ed25519 ubuntu@158.101.20.226 'systemctl --no-pager status jope-parallel-monitor jope-parallel-gp'
rtk proxy ssh -i /home/ubuntu/.ssh/lambda_cloud_a100_20260907_ed25519 ubuntu@158.101.20.226 'journalctl -u jope-parallel-gp -n 30 --no-pager'
```

`runs/hardware.json`, `runs/gp-service.json`, `runs/neural-service.json`, and `runs/cycle-progress.json` are the dashboard's live interfaces. Complete checkpoint receipts point at an atomically published model, optimizer, corpus, vocabulary, sampler, and RNG identity. `runs/telemetry/` preserves hardware history. `runs/parity/` contains numerical and live terminal verification receipts.

`enqueue --jobs PATH --contexts PATH` verifies the exact saved context and cut identities before atomically adding GP work. It keeps the queue intact on a mismatch and interleaves cuts across datasets. Queue receipts bind both manifests. `apply-contexts` reuses an existing frozen assembly; `union-contexts` combines only index references.

The neural executable's `cycle --corpus PATH --output PATH --steps 300 --cuts 2 --qualification` command runs fresh training, eight-GPU grammar-masked proposals, matched generic/shuffled-context GP comparisons, a new teaching-corpus snapshot, and continued training. Both paths must be absolute. It publishes `runs/cycle-progress.json` throughout and resumes from complete receipts. Use the existing managed campaign when one is active rather than launching another controller against the same workspace. `pilot-gate` separately checks full pilot eligibility; running a qualification cycle does not start the 24-hour pilot.

Do not overwrite an executable used by active GP workers. Deploy each new executable to a new versioned filename, update its owning service deliberately, and preserve evaluator hashes. The TUI uses its own executable and can be replaced without stopping GP.

## Build and checks

The standalone workspace requires Rust 1.90 or newer. The default build is CPU-only and needs no LibTorch or CUDA; `just required` (rustfmt check plus `cargo test --locked --lib`) runs this configuration. The build never runs Python.

The `neural` feature uses `tch =0.26.0` (its `torch-sys` expects LibTorch 2.13.0). You must name the SDK explicitly:

- `LIBTORCH`: the LibTorch root, containing `include/` and `lib/`.
- `LIBTORCH_CXX11_ABI`: `0` or `1`, matching how that LibTorch was built.

If either variable is missing, `build.rs` stops with an error. It does not fall back to discovering PyTorch through Python. Do not set `LIBTORCH_USE_PYTORCH`, because it makes `torch-sys` call Python.

The CUDA and NCCL locations come from `CUDA_HOME`, `JOPE_NCCL_INCLUDE`, `JOPE_CUDA_LIBRARY_PATH`, `JOPE_CUDART_SONAME` and `JOPE_NCCL_SONAME`.

Native Hyperion has its own Cargo workspace and pinned source/weight manifests. The frozen models were exported once by Python conversion scripts. Production inference and training run in Rust.

```bash
rtk proxy env CARGO_BUILD_JOBS=4 cargo test --test dashboard
rtk proxy env LIBTORCH=/opt/libtorch LIBTORCH_CXX11_ABI=1 CXX=g++ CARGO_BUILD_JOBS=8 cargo build --release --features neural
```

`jope-build-receipt --binary PATH --source-dir DIR --output PATH [--lineage NAME] [--rustc-vv FILE] [--max-glibc 2.36] [--forbid-gpu-libraries]` writes a `jope.product_evaluator_build.v1` receipt. The receipt records:

- the git commit and tree of the source directory;
- the SHA-256 of `Cargo.toml` and `Cargo.lock`;
- the output of `rustc -vV`;
- the binary's SHA-256 and size;
- the highest `GLIBC_*` version the binary requires, read from its ELF version-requirement section;
- its `NEEDED` libraries;
- whether any of those libraries is libcuda, libtorch or libc10.

`just kernel-v1-product-build OUT` (`../ops/ci/kernel-v1-product-build.sh`) builds the product evaluator from `releases/kernel-v1-8aa0affbf…/source`. It is separate from `required`. The script runs in three stages:

1. **Fetch.** On the host, `cargo fetch --locked` downloads the pinned crates into `OUT/cargo-home`. This is the only stage that uses the network. To reuse an existing registry, set `KERNEL_V1_SEED_CARGO_HOME`.
2. **Build.** The build runs in `jeryu-builder:rust1.95-glibc2.35` with `docker run --network none`. The container mounts the source read-only and copies it into `OUT/run-*/work`. It then runs `cargo build --release --offline --locked --bin jope-parallel`.
3. **Receipt.** The script writes the receipt to `OUT/run-*/receipt.json`. The run fails if the binary requires anything newer than `GLIBC_2.36` (Debian bookworm) or links CUDA or LibTorch.

The bounded `campaign --spec PATH` controller chains resumable qualification cycles with an immutable admission/deadline schedule and family-coverage checks. The source includes `systemd/jope-parallel-campaign.service`; it must point at a verified corpus, complete saved contexts, and the qualified executable before launch. It does not authorize independent final-test scoring.

Run these commands in `parallel/` on the target node. Full runtime, data, and asset transfers use SHA-256 manifests and resumable disjoint streams. Do not transfer environments, build directories, credentials, or copied cut tables.

## Recovery console (native, live) — `tui-recovery.sh`

`rtk proxy /home/ubuntu/jope-dime/parallel/tools/tui-recovery.sh` attaches **read-only** to the live native recovery console
(tmux session `jope-recovery` on the console unit's private socket) after verifying that exactly one renderer pane is
alive, that it belongs to `jope-eight-a100-console.service`, and that its frame is fresh. Verbs: `check` (verify only, prints
pane/unit/frame age), `console` (default; detach with `C-b d`, keys are not delivered so `q` is harmless), `plots` (runs the
reviewed `flywheel_plots.py` in your own ssh tty after a sha256 match; it never creates a window in the supervised session),
`dashboard` (attaches to the native dashboard once its `live-v1` receipt says passed). Flags: `--interactive` (keys delivered;
`q` closes the renderer and the supervisor restarts it in ~5 s), `--plain-ssh`. `tui-remote.sh` (the Sept-9 observer +
plots window) is unchanged and root-owned.

## Migration to xbabe2 and the jain-split lane (2026-09-14)

The node keeps no git; xbabe2 is the durable mirror and the only publisher. `tools/node_mirror_pull.py` (interim Python one-shot)
and the Rust `jain-datasets mirror pull` (crate `~/jain-split/jain-datasets/tools`, binary pinned under
`/home/ubuntu/jope-node-mirror/bin/`, driven by `tools/node-mirror.sh fast|bulk` and the `jope-node-mirror-{fast,bulk}` user
timers) mirror one class at a time with a node census, `rsync --checksum`, local re-hash, drift refusal, atomic install, node
re-hash and an immutable receipt under `/home/ubuntu/jope-node-mirror/receipts/mirror/`. Classes: `champions`, `champion-git`,
`receipts`, `weights`, `dataset` (`/home/ubuntu/quant`), `prepared` (splits), `contexts` and `teaching-corpus` (encodings),
`results` (per-generation GP results: seeds, ranking, best-predicted, search, progress, quality reports), `observations`
(the search ledgers), `node-source`, `node-release`, `node-home`, `lane-store` (the node's dataset-lane Store),
`recovery-20260908`, `corpus`, `assets`, `kernel-work`; `runs-census` and `recovery-run-census` index the run roots that are
too large to copy without an owner decision (sha256 index only, `census_only: true` in the receipt).

Mirrored classes enter the lane Store `~/jain-split/jain-datasets/.staging/a100-store` as captures — one per unit (prepared
view, quant dataset, GP cut result, champion identity, node-only root) — with `jain-datasets capture-mirror` (native) or the
interim `tools/lane_capture.py` (which also imports the >5 MB xbabe2 binaries listed in
`recovery/eight-a100-20260912/inventory/xbabe2-untracked-binaries-20260914.json`); receipts under `receipts/lane/`. The Store
publishes to `veox/jain-datasets` `updates/a100` once the forge grant exists. Code publishes with `tools/code_publish.sh
[remote] [extra branches...]` (receipted; `wip/xbabe2-working-tree-20260914` = the dirty working tree as found,
`archive/research-untracked-20260914` = the research job trees). `~/jain-split/jope-dime` is a symlink to this checkout
(the family forbids copied repo folders).
