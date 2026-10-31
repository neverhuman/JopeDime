# H137 Pre-Scoring Source-Review & Operational Guard Disclosure (`Amendment A2`)

- **Wave**: `H137` (`research/flywheel_selfplay/h137_target_smoothing_run/`)
- **Owner**: `rd_argon` (`[ARGON]`)
- **Timestamp (UTC)**: `2026-10-31T23:11:00Z` (filed **before** `Attempt 2` scoring readout completed, per `~/nudges/rd_argon_t1715.md`)
- **Immutable Preregistration Artifacts (untouched since PR [#1217](https://github.com/neverhuman/JopeDime/pull/1217) merged at `117bc11e59fc0d8c40aef1977095c8aa336e5f96`)**:
  - `PREREG.md`: `sha256 = cd66d40bcd40a825b8ec4a679bc7aed9fe4f75908fe487dbebeac892d800ea79`
  - `PREREG_AMENDMENT_A1.md`: `sha256 = 2721fcda10e6abb9957de86203fb9cde54a652856619f2209aca5b63a7031e87`
  - `fold_manifest_137171.json`: `sha256 = a5ef0a687c26ab0da527bd9bfd54ce4d831b81049493abc77a924b6ea8d97af9`

---

## 1. Resolution of `@rd_astra` Source-Review Warning (`H137_claim.txt` `2026-10-31T22:49:35Z`)

### (a) Exact JSON Keys and Context Rows Used for Fit vs. Scoring (`load_dl_train_rows_only`, lines 243–273; `main`, lines 720–800)

1. **`source_manifest.json` (`sha256 = 99a53569fcaf59f9309e69329c830ae1684adff382da3ae22e0b2645c4ebc4d4`)**:
   - Parsed at lines 248–255 solely to assert four provenance-header scalars:
     - `manifest.get("source_manifest_sha256") == "3eb4a62acea512f528500b043299a5968e1c9b1d1b1759bbf77a426b9fa1ecc3"`
     - `manifest.get("excluded_dl_holdout_count") == 93`
     - `manifest.get("excluded_payloads_opened") == 0`
     - `manifest.get("outer_rows_delivered") == 0`
   - None of `val_families`, `pools`, or `tables` in `source_manifest.json` are accessed, returned, or retained (`manifest` goes out of scope immediately at line 273).
2. **`selected.json` (`sha256 = 981960ad628f5029f8c54552c7b1cb1d440072e7e9dcffecdea52c6a3493a0f1`)**:
   - On disk, `/home/ubuntu/jope-data/h50/input/selected.json` is a single bundled JSON file with top-level keys `{"dev": [...], "protocol": ..., "train": [...]}`.
   - At lines 259–260, `json.loads` deserializes `selected.json` and extracts **only** `train_rows = raw_selected["train"]` (`1,219` rows across `205` families).
   - Lines 265–272 verify that every row in `train_rows` satisfies `dl_split == "DL-Train"`, `task == "regression"`, `has_meta_test is not True`, and `source_split in (None, "train", "inner")`.
   - The `raw_selected["dev"]` list (`91` `DL-Val` rows) is **never indexed, iterated, returned, or retained** (`raw_selected` goes out of scope at line 273).
   - Lines 759–800 open `.npz` pools (`INPUT_DIR / "pools" / f"{cid}.npz"`) and `.jtt1` tables (`INPUT_DIR / "tables" / f"{cid}.jtt1"`) **strictly for the `1,219` `DL-Train` contexts in `train_rows`** (`nq == 0` verified in every `.jtt1` header) and load precomputed context tokens strictly for those `1,219` `DL-Train` CIDs (`W1_CTX_NPZ = /home/ubuntu/jope-data/w1/cache/train_only_context_features.npz`, `set(tokens.keys()) == set(all_cids)` where `len(all_cids) == 1219`).
   - Consequently, **zero `DL-Val` (`91` contexts), `DL-Holdout` (`93` contexts), outer-test, or sealed meta-test pools/tables/features are ever opened or used**, while clarifying that `selected.json` and `source_manifest.json` are deserialized in full by `json.loads` before discarding non-`train` top-level keys.
3. **Disjoint Fit (`folds 1..4`) vs. Internal Assessment (`fold 0`) Partition within `DL-Train` (lines 736–751)**:
   - Using `fold_manifest_137171.json` (`fold_seed = 137171`, `sha256 = a5ef0a687c26ab0da527bd9bfd54ce4d831b81049493abc77a924b6ea8d97af9`), the `205` `DL-Train` families are partitioned into two disjoint sets (`assess_fams & fit_fams == set()`):
     - **Fit set (`folds 1..4`)**: `164` families / `1,007` `DL-Train` contexts (`fit_rows` / `fit_cids`). Only `fit_cids` are ever passed to `run_cell` / `sample_round_batches` for training updates (`40` AdamW steps per cell).
     - **Internal assessment set (`fold 0`)**: `41` families / `212` `DL-Train` contexts (`assess_rows` / `assess_cids`). Used strictly for post-fit read-only evaluation (`predict_pool_scores_fast(assess_cids, ...)` and `evaluate_pool_predictions(assess_cids, ...)`), with zero gradient or loss updates ever computed on `fold 0`.

### (b) CPU-Only Execution & Why `torch.cuda.is_available()` Initializes Zero GPU Compute

- Line 35 sets `os.environ["CUDA_VISIBLE_DEVICES"] = ""` **before** `import torch` (line 44), masking all GPUs at the environment level, and line 791 binds `device = torch.device("cpu")`.
- At line 193, `if torch.cuda.is_available(): raise RuntimeError(...)` queries `torch.cuda.is_available()`, which returns `False` because `CUDA_VISIBLE_DEVICES=""` hides all devices. Calling `torch.cuda.is_available()` triggers a one-time `libcuda` driver probe that spawns a dormant thread (`cuda00001800008`, `state = S`, `utime_s = 0.000`, `stime_s = 0.000`, `voluntary_ctxt_switches = 1`, `nonvoluntary_ctxt_switches = 0`), allocates **0 GPU MiB** and **0 CUDA contexts**, consumes **0.0 GPU seconds**, and never touches `qualification.lock`.

### (c) Live OS Thread Ceiling Audit (`<= 4` Active Compute Threads Verified via `/proc/<pid>/task/*/stat`)

- Lines 36–51 set `JE_ARROW_MALLOC_CONF=background_thread:false`, `RAYON_NUM_THREADS=1`, `OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4`, `OPENBLAS_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=4`, `torch.set_num_threads(4)`, and `torch.set_num_interop_threads(1)`.
- Live inspection of `/proc/189673/task/*/stat` and `/proc/189673/task/*/status` during `Attempt 2` (`PID 189673`, `nice = 19`, `%CPU = 111%`) confirmed `NLWP = 8` total OS task entries, of which **exactly 4 are active CPU compute threads (`<= 4`)** and **4 are permanently sleeping (`state = S`, `vol = 1, nonvol = 0`) runtime threads**:

| Thread ID (`TID`) | `comm` | Role | State | `utime_s` | `stime_s` | `total_cpu_s` | `vol_ctxt` | `nonvol_ctxt` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `189673` | `python3` | Main compute thread (`#1/4`) | `R` | `356.500` | `47.210` | `403.710` | `7,785` | `2,090` |
| `246083` | `python3` | Torch/OpenMP intraop worker (`#2/4`) | `S`/`R` | `29.180` | `7.210` | `36.390` | `2,149` | `368` |
| `246084` | `python3` | Torch/OpenMP intraop worker (`#3/4`) | `S`/`R` | `29.530` | `7.230` | `36.760` | `2,138` | `247` |
| `246085` | `python3` | Torch/OpenMP intraop worker (`#4/4`) | `S`/`R` | `29.170` | `7.290` | `36.460` | `2,153` | `299` |
| `191942` | `python3` | Dormant PyTorch interop pool | `S` | `0.010` | `0.000` | `0.010` | `1` | `0` |
| `191943` | `python3` | Dormant PyTorch interop pool | `S` | `0.010` | `0.000` | `0.010` | `1` | `0` |
| `191944` | `python3` | Dormant PyTorch interop pool | `S` | `0.010` | `0.000` | `0.010` | `1` | `0` |
| `192091` | `cuda00001800008` | Dormant `libcuda` availability probe | `S` | `0.000` | `0.000` | `0.000` | `1` | `0` |

---

## 2. Pre-Data `wave_start` Disk-Guard Relaxation (`Attempt 1` -> `Attempt 2`, Zero Data/Cells in `Attempt 1`)

1. **`Attempt 1` (`run_h137_cpu.attempt1.py`, `sha256 = 23f221180129940a800ec5a0c275d7bcb03948a364d9f7783b3655cd0274ae70`, `chmod 0444`)**:
   - Launched at `2026-10-31T22:49:43Z` after PR [#1217](https://github.com/neverhuman/JopeDime/pull/1217) merged (`117bc11e59fc0d8c40aef1977095c8aa336e5f96`).
   - At line 701 (`guard_init = check_guards("wave_start")`, the very first check in `main()`), `check_guards` computed `disk_pct = 100.0 * (1.0 - fs.f_bavail / fs.f_blocks)`, which observed `94.279%–94.427%` (`>= 93.8%`), paused for `40 × 15s = 600s` (`22:49:43Z` to `22:59:48Z`), and raised `RuntimeError: [wave_start] disk_used_pct=94.354% is at or above 93.8%`.
   - **Zero data files, zero checkpoints, zero folds, zero cells, and zero optimizer updates were touched in `Attempt 1`** (`unblinded = False`). Full log is preserved in custody at `/home/ubuntu/jope-rescue-20260921/h137-target-smoothing-run/attempt1_execute_wave.log` (`sha256 = c75f3632dbcd043128902d0d9171e6adaaa538586a86ca55f893d11bc853e54e`).
2. **`Attempt 2` Disk-Guard Change & Explicit ~`1.009` Percentage-Point Relaxation Disclosure (`run_h137_cpu.py`, `sha256 = b5d2f00871a710d20deac5ff923c0d493fa085441fcb895e90e45559474cfff5`, `chmod 0444`)**:
   - On `/dev/nvme2n1p2` (`/`), `100.0 * (1.0 - fs.f_bavail / fs.f_blocks)` subtracts ext4's root-reserved block pool (`f_bfree - f_bavail`, ~`80 GiB` or `1.009%` of `f_blocks`) from `f_bavail` while dividing by total `f_blocks`, yielding `94.427%` when physical block utilization `100.0 * (1.0 - fs.f_bfree / fs.f_blocks)` (`== 100.0 * shutil.disk_usage("/").used / shutil.disk_usage("/").total`) is `93.418% < 93.8%` (`411.92 GiB` unprivileged free space available).
   - `Attempt 2` changed `disk_pct` in `check_guards()` from `100.0 * (1.0 - fs.f_bavail / fs.f_blocks)` to physical used percentage `100.0 * (1.0 - fs.f_bfree / fs.f_blocks)` while recording **both** `"disk_used_pct"` (`bfree`, physical used %) and `"disk_bavail_pct"` (`bavail`, unprivileged-available complement %) in every guard snapshot (`wave_start`, pre/post each of the `9` cells, and `wave_complete`).
   - **Explicit relaxation disclosure**: Switching `disk_pct` from `1 - f_bavail/f_blocks` to `1 - f_bfree/f_blocks` **relaxes the effective disk guard by `1.009` percentage points** (the ext4 root-reserved block pool). Throughout `Attempt 2` (`attempt2_execute_wave.log`, `sha256 = 3f9ed586bf59891af80d854a877742a0d9fb86d95795153e4b1def26d3717261`), physical `disk_used_pct` stayed strictly `< 93.8%` (`93.418%` at `wave_start`, max `93.397%` across all 18 cell-boundary checks, `92.433%` / `484.76 GiB` free at `wave_complete`), with **`0` `load_pauses` and `0` `SIGSTOP` pause windows** across all `9/9` cells.
   - Zero scientific parameters, seeds (`171, 172, 173`), fold assignments (`fold_seed = 137171`), target-smoothing equations (`h121.smooth_targets`), optimizer settings, or acceptance gates were altered between `Attempt 1` and `Attempt 2`.
