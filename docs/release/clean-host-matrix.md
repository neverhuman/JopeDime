# Clean-host matrix (REL-04)

This records executed platform slices. Gate D1 requires a signed candidate to complete
discovery, Python/Rust/JavaScript source exports and prediction without network access
or installed Python, rustc or Node, plus the OS-library link audit. A source build and
an empty toolchain `PATH` establish a pre-stage only. No platform is qualified by this table.

## Linux x86-64, 2026-10-31

| Matrix row | Source / input | Actual result |
| --- | --- | --- |
| xbabe1, Ubuntu 24.04.2 LTS, kernel 6.8.0-134-generic, x86-64 CPU; fresh worktree | `origin/main` `a51ad108f59d6b7f10cf1e22e247919885f83e65` (after required `b77186f09`); default-feature `jope-appliance` build, Cargo/rustc 1.97.1, `--locked`, eight jobs | Build and isolated installation completed; unsigned development executable, `package: null` |
| Network/toolchain pre-stage | Owned network namespace, loopback only, no default route; isolated home and empty toolchain `PATH` | Network checks passed; Python, python3, rustc and Node unavailable through application `PATH`. Host filesystem/toolchain removal was **not** proved |
| Real training-only input | Catalog `b4c8807a75fc190f`, 25 training rows, 13 columns, target `target`; native training SHA256 `782acd17eed2bfc7acfa4438be076fb63026763e6578661cfc284c2f3f9a21e1` | Header-only preparation preserves every native training payload byte; three prediction inputs use training features with the target omitted. No test/holdout/sealed payload opened |
| Method discovery / dataset registration | Same installed executable and owned engine home | Passed; dataset ID `ds_153f24a64081035aaa97541e` |
| Run discovery | Required full path, heuristic guidance, CPU, one evaluation / 180-second budget | **Failed, CLI exit 2:** `Operation run_start is not implemented in this build.` No run ID or completed discovery |
| Python / Rust / JavaScript run exports; prediction | Must follow the successful run | **Not run** after the actual discovery refusal; prediction row count unknown. No local-fit substitution |
| Link audit, `fused-cpu` | Same installed executable | Passed: `ld-linux-x86-64.so.2`, `libc.so.6`, `libgcc_s.so.1`, `libm.so.6` |
| Linux D1 | Signed release candidate + complete runtime path + toolchain-free filesystem | **Not qualified:** unsigned candidate, missing `run_start`, and filesystem isolation not established |
| macOS arm64 / Windows x86-64 | Native signed candidates and native execution receipts | **Pending; not executed by this Linux slice** |

Binary SHA256: `13f7367e523fe7b2323da1abac6b1ff41d051185214e85e8927d86ac7f16c79b`.
Prepared training CSV SHA256: `153f24a64081035aaa97541e08bf0eb037a14dcb61eca355bafaa23ab62cbf7f`.
Prediction CSV SHA256: `6644d7d992a26b0cf64cdcabe34064a0373f0e5e06bf8ef91aacf9ea0766fdd2`.
The catalog license is unknown; these training rows remain in private custody, outside Git.

Raw build/artifact messages, install receipt, exact invocations/exits, CLI outputs,
receipt and regression controls are retained on xbabe1 and mirrored to xbabe2 under
`~/jope_plan/evidence/REL04-LINUX-sol_v1_mcp01-t1112/`; handoff is
`~/jope_plan/master/status/REL04-LINUX-sol_v1_mcp01-t1112.md`.
The source commit identifies the built product; `harness_sha256` in the receipt separately
binds the executing shell script, and the handoff binds its final branch commit.

## Reproduce the Linux pre-stage

Use a fresh worktree at the selected main commit. Set `REL04_CUSTODY` to a fresh private
directory outside the worktree and `/tmp`. Supply an approved, header-bearing training CSV
and a prediction CSV containing only permitted features; `--csv` fixtures require one
physical line per CSV record and a final newline. The runtime home is fresh for each run.
No application traffic leaves its namespace. Fetch/build dependencies before isolation.

```bash
mkdir -p "$REL04_CUSTODY/install/bin"
env CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only CARGO_BUILD_JOBS=8 \
  nice -n 10 cargo build --locked -p jope-appliance --bin jopedime -j 8 \
  --message-format=json-render-diagnostics > "$REL04_CUSTODY/build.jsonl" \
  2> "$REL04_CUSTODY/build.stderr"
built_binary=$(jq -r 'select(.reason == "compiler-artifact" and .target.name == "jopedime") |
  .executable // empty' "$REL04_CUSTODY/build.jsonl")
install -m 0755 "$built_binary" "$REL04_CUSTODY/install/bin/jopedime"

# A root-owned output parent is necessary when sudo maps namespace root to host root.
sudo -n install -d -m 0700 "$REL04_CUSTODY/runtime"
sudo -n env GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=safe.directory \
  "GIT_CONFIG_VALUE_0=$PWD" nice -n 10 bash deploy/appliance/test-clean-host.sh \
  "$REL04_CUSTODY/install/bin/jopedime" "$REL04_CUSTODY/runtime/linux" --prestage \
  --require-golden-path --csv "$TRAIN_CSV" "$TARGET_COLUMN" "$PREDICT_CSV"
```

Respect host load ≤85%; one reused Cargo target, no global `CARGO_TARGET_DIR`, eight build
jobs here (owner ceiling 40). The executed `nice -n 10` commands inherited niceness 9 and
therefore ran at effective niceness 19. No GPU or hosted CI is needed for this slice.

`clean-host.json` preserves the original command failure and completed stages after runtime
setup. Namespace/preflight failures before work-directory creation remain nonzero stderr
failures. `--require-golden-path` forbids the older dataset-refusal local-fit fallback;
without it, that legacy smoke remains explicitly unqualified. A passing pre-stage still
leaves D1 unqualified. Export hashes alone never certify export execution or prediction.

## Controller regression controls

```bash
bash deploy/appliance/test-clean-host-prestage-controls.sh
```

Five fixtures check a refused run, partial exports, a prediction row-count mismatch,
strict refusal of local-fit substitution, and successful forwarding of supplied CSV/target
inputs. The original helper loses failure receipts; the corrected helper retains the
nonzero exit and per-stage state. These controls replace the namespace/link scanner and
CLI with explicit mocks, exercise no actual fitting, and cannot qualify D1. Actual Linux
evidence above uses the real binary and namespace/link scanner.
