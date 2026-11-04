# H317 Stage V: mounts, staging and commands

Status: **plan only. Nothing here has been run.** Every command below touches DL-Val. It may run only at the step it
belongs to, after that step's precondition holds. Driver: `stage_v_h317.py` (PREREG §2.6, §7 A1).

## 0. Phases, hosts and sandboxes

| Step | Host | Sandbox | DL-Val visible? | Writes |
| --- | --- | --- | --- | --- |
| S0 `write-lock` | xbabe2 | `sandbox_h317.sh` (TRAIN view) | no | `$DATA/stage_v/LOCK.json` (create-once) |
| S1 receipt | xbabe2 host (parent) | - | no | `~/jope_plan/ns/receipts/H317_LOCK_sha256.txt` (create-once), comms post |
| S2 `unlock` | xbabe2 | `sandbox_h317_stagev.sh unlock` (TRAIN view + receipt ro) | **no** | `$DATA/stage_v/STAGE_V_STARTED` (O_EXCL), `UNLOCK_RECEIPT.json` |
| S3 build the V root | xbabe2 host (parent) | - | links and copies only | `$SV/` (below) |
| S4 `run --phase prep` | xbabe2 | `sandbox_h317_stagev.sh v` (V view at RES01) | yes (91 tables + pools, R882 inputs) | `$SV/run/*`, `$SV/views-*` |
| S5 stage to xbabe0 | xbabe2 to xbabe0 | rsync + `sha256sum -c` | 91 tables + 91 SEARCH view sets | `$STAGE_V/` on xbabe0 |
| S6 `run --phase audit` | xbabe0 | `sandbox_h317_docker.sh --stage-v` (+ the mounts in §3) | 91 tables (no pools, no R882 reference) | `$STAGE_V/stagev-root/run/audit_units.*.jsonl`, `gp/` |
| S7 pull back + `run --phase stats` | xbabe2 | `sandbox_h317_stagev.sh v` | records only | `$SV/run/metrics.json`, `panel_stage_v.json` |

Abbreviations:
- `SRC=/home/ubuntu/jope_plan/evidence/H317/source/research/flywheel_selfplay/h317_island10_multiscale_curriculum`
- `DATA=/home/ubuntu/jope-data/h317-cl`
- `SV=/home/ubuntu/jope-data/h317-stagev`, the Stage V root on xbabe2 (`H317_STAGEV_ROOT`). It sits outside `$DATA`, so
  no Gate H0 sandbox ever binds it.
- `RECEIPT=/home/ubuntu/jope_plan/ns/receipts/H317_LOCK_sha256.txt`
- `STAGE_V=/home/ubuntu/h317-stage-v`, the xbabe0 Stage V tree (`sandbox_h317_docker.sh` default).

Why prep runs on xbabe2:
- It needs the full encoder stack. That is the MICRO_PINS files, `python/hpc/landmark_workspace.py`,
  `bin/libjope_hpc.so` and `res-01/input/cpu_exact_fused.py` (b9f1468a…).
- It also needs the DL-Val pools and the R882 inputs.
- Running prep here keeps the outcome-bearing inputs (pools, `best_dev_checkpoint.json`, r882 costs) and
  `cpu_exact_fused.py` on xbabe2. xbabe0 receives the LOCKed DOPE/encoder files only so `resume_stage_v` can hash
  them; it never runs the encoder.
- Before encoding any DL-Val table, `prep` re-encodes 2 TRAIN pilot SEARCH view sets and their full-table view 0. They
  must be bit-identical to the stored xbabe1/xbabe3 encodings. If they are not, prep stops before any outcome read, and
  one relaunch is allowed.

## 1. Must happen BEFORE `write-lock` (S0)

`write-lock` hashes every top-level `*.py`, `*.sh`, `*.json` and `PREREG.md` in `$SRC`. That includes
`sandbox_h317.sh`, `sandbox_h317_stagev.sh`, `sandbox_h317_docker.sh`, `stage_xbabe0.sh` and `stage_v_h317.py`.
Any later edit to one of them breaks `unlock` and `resume_stage_v`. So these must be final first.

**Every Stage V host needs every LOCK file at its exact LOCK path.** `common_h317.resume_stage_v()` re-hashes the whole
`code_sha256` in every phase, with no remapping and no tolerance for absent files:
- H317 source at `$SRC`;
- the worker and the DOPE/encoder files under `/home/ubuntu/JopeDimeMicro`;
- the LOCK models under `$DATA/neural/lock`.

The xbabe2 sandboxes see all of these.

**The LOCK `inputs` group (red team MEDIUM-3)** is hashed at the paths the code reads, so on each host the
staged copies mounted there are what gets hashed:
- `/home/ubuntu/jope-data/res-01/input/{champions,winners}.json` (the canonical RES01 mount);
- `$DATA/neural/views_c32_train1219.npz` (eb042276…);
- `$DATA/stage_v/train_inputs_manifest.json`.

`fold_manifest.json` is in the source group. In addition, every phase checks every TRAIN table at RES01 against the
manifest, and checks `RES01/selected.json` against the pins allowed for the phase:
- prep requires the original 981960ad…;
- audit and stats accept the redacted 7d64774a… or the original.

`selected.json` cannot be in `code_sha256` because unlock and prep see different copies; `common_h317` pins it to
those two shas anyway.

1. **`sandbox_h317_docker.sh --stage-v` additions** (GP agent; confirmed). Without them, `resume_stage_v` cannot run
   in the container:
   - `-v $RECEIPT:$RECEIPT:ro`. The in-container process re-reads the receipt; `common_h317.LOCK_RECEIPT` is a fixed
     path.
   - `-v $STAGE_V/stagev-root:/home/ubuntu/jope-data/h317-stagev:rw` with
     `-e H317_STAGEV_ROOT=/home/ubuntu/jope-data/h317-stagev`.
   - `-v $STAGE_V/stagev-root/views-search:/home/ubuntu/jope-data/h317-stagev/views-search:ro`.
   - `-v $DATA/stage_v:$DATA/stage_v:ro`, holding LOCK.json, STAGE_V_STARTED and UNLOCK_RECEIPT.json.
   - `-v $DATA/neural/lock:$DATA/neural/lock:ro`, the LOCK models.
   - Keep the default guard log (`$DATA/guard_refusals.log`, rw in the container). Do not point `H317_GUARD_LOG`
     under `h317-stagev/`: the guard refuses that tree before the unlock, so a pre-unlock refusal would not be logged.
   - `$STAGE/micro` must also hold the LOCKed DOPE/encoder files that the GP micro subset lacks:
     - `assets/models/dope_mlc_v3_multitask_best.pt` (a36510b9…);
     - `assets/projections/frozen_stage1_stage2_inference_params_v3.pt` (136b2381…, 102 MB);
     - `python/model.py` (0aa76360…);
     - `python/hpc/landmark_workspace.py`;
     - `bin/libjope_hpc.so`.

     `rust_evaluator.py`, `phase12_multimodal_model.py` and `program-feature-names.json` are already staged. The audit
     never runs the encoder; these files are present only so `resume_stage_v` can hash them. Their shas are in
     `LOCK.json` `code_sha256`.
   - `$STAGE_V/dlval-view` should hold the TRAIN view plus the **91 DL-Val tables only**. The audit phase reads no pool,
     and the redacted `selected.json` (7d64774a…) is enough there.
2. The xbabe2 Stage V sandbox is `sandbox_h317_stagev.sh` (new; `sandbox_h317.sh` is unchanged). Modes:
   - `unlock`: the TRAIN layout plus the receipt read-only. It refuses without a matching receipt, and refuses if
     STAGE_V_STARTED already exists.
   - `v` (prep / stats): the V view at RES01, `$SV` rw with `inputs/` and the V view read-only, `$DATA/stage_v`
     read-only, the receipt read-only, and `H317_STAGEV_ROOT`. It refuses unless the marker names the receipt's LOCK
     and the V view holds 1,310 tables.
   - `--print` prints the bwrap argv.

   The `v` layout was probed on a synthetic tree:
   - RES01 shows the V view, read-only;
   - the receipt is readable;
   - `$DATA/stage_v`, `inputs/` and the V view are read-only, while `$DATA` and `run/` are writable;
   - `/home/ubuntu/jope_plan/evidence/H258` and `~/jope-evidence` are not visible, `benchmark_data` is empty, and there
     is no network.
3. The lock-fit model dir is pulled to `$DATA/neural/lock` (sha-verified), so the unlock sandbox can see it.
   `write-lock` refuses a model dir outside `$DATA`.
4. The xbabe0 code stage (`stage_xbabe0.sh` code group) is refreshed from the final `$SRC`.
   `enter_stage_v("audit")` rechecks every source hash in the container.

## 2. Files added for Stage V (exhaustive)

### xbabe2: `$DATA/stage_v/`

| File | Created by | Notes |
| --- | --- | --- |
| `LOCK.json` | S0 `write-lock` | create-once, fsynced. Contains `code_groups` (source / worker / dope / lock_model), `code_sha256`, locked configs, seeds, pipelines and decision rules |
| `STAGE_V_STARTED` | S2 `common_h317.unlock_stage_v()` | O_EXCL, 0444, `"<pid> <LOCK sha>"`. One-shot |
| `UNLOCK_RECEIPT.json` | S2 | create-once (host, pid, utc, LOCK sha, guard report) |
| `train_inputs_manifest.json` | S0 `write-lock` (before LOCK.json) | create-once: sha256 of the 1,219 TRAIN tables at RES01 + the selected.json pins allowed per phase; hashed in the LOCK `inputs` group |

### xbabe2: `$SV` (built in S3 only after `STAGE_V_STARTED` exists)

| Path | Source | Pin |
| --- | --- | --- |
| `res01-v-view/tables/*.jtt1` (1,219 TRAIN) | hard links from `$DATA/res01-train-view/tables` | `selected.json` train `train_only_jtt1_sha256` |
| `res01-v-view/pools/*.npz` (1,219 TRAIN) | hard links from `$DATA/res01-train-view/pools` | `pool_sha256` |
| `res01-v-view/{champions,winners,source_manifest}.json` | hard links from the TRAIN view | as staged for Gate H0 |
| `res01-v-view/tables/<91 dev cids>.jtt1` | hard links from `/home/ubuntu/jope-data/res-01/input/tables` | dev `train_only_jtt1_sha256` (checked in prep) |
| `res01-v-view/pools/<91 dev cids>.npz` | hard links from `/home/ubuntu/jope-data/res-01/input/pools` | dev `pool_sha256` (checked in prep) |
| `res01-v-view/selected.json` | hard link to `/home/ubuntu/jope-data/res-01/input/selected.json` (ORIGINAL) | `981960ad…` (prep refuses the redacted copy) |
| `res01-v-view/cpu_exact_fused.py` | hard link from `/home/ubuntu/jope-data/res-01/input/` | `b9f1468a…` (checked by `DopeViewEncoder`) |
| `inputs/best_dev_checkpoint.json` | copy (0444) from `/home/ubuntu/jope_plan/evidence/H258/argon-t1405/inputs/r882/` | `30853cad78d8…` |
| `inputs/r882-secondary-costs.npz` | copy (0444) from `/home/ubuntu/jope_plan/evidence/H258/argon-t1405/inputs/` | `0e812aa7ea4c…` (the H258 runner's DL-Val `total_cost` source) |
| `inputs/target_cids.txt` | the 91 dev cids (identifiers, from the redacted `selected.json`) | - |
| `views-search/<cid>.npz` (91) | prep: `encode_context(…, "search")`, canonical recipe h317-views-v1 | sha in `run/targets_manifest.json` |
| `views-full0/<cid>.npz` (91) | prep: full-table view 0 (float32) for V-native | sha in `run/targets_manifest.json` |
| `run/` | prep / audit / stats | see below |

`run/` contents:

| File | When | Holds outcomes? |
| --- | --- | --- |
| `PREP_STARTED` | create-once at the first prep | no |
| `RELAUNCH_USED` | create-once, the single pre-outcome relaunch | no |
| `inputs_verified.json` | sha256 of all DL-Val inputs, bytes only | no |
| `dedupe.json` | row-fingerprint dedupe | no |
| `encoder_parity.json` | encoder host parity | no |
| `targets_manifest.json` | 91 targets: family, dataset, table / view shas, nt, width | no |
| `FIRST_OUTCOME_READ` | create-once, fsynced immediately before the regression reads pools and the reference | no |
| `native.json` | regression + TOP4_PICK + V-native | **yes**, stays on xbabe2 |
| `STOPPED_REGRESSION_FAIL` | create-once on REGRESSION_FAIL; every later phase refuses | yes |
| `PREP_DONE.json` | create-once, with the sha256 of the files above | - |
| `audit_units.s<k>of<n>.jsonl` | xbabe0, append-only | yes |
| `AUDIT_STARTED.s<k>of<n>` / `AUDIT_DONE.s<k>of<n>.json` | xbabe0 | - |
| `events.jsonl` (+ `events.xbabe0.jsonl` after pull-back) | every phase | - |
| `guard_refusals.log` | every phase | - |
| `metrics.json` (< 50 KB, compact) / `panel_stage_v.json` | stats | yes |

### xbabe0 (S5, only after `PREP_DONE.json`)

| xbabe0 path | From xbabe2 | Mounted in the container as |
| --- | --- | --- |
| `$STAGE_V/dlval-view/` = hard links of `$STAGE/res01-train-view/*` + `tables/<91>.jtt1` | `$SV/res01-v-view/tables/<91>.jtt1` | `/home/ubuntu/jope-data/res-01/input` (ro) |
| `$STAGE_V/stagev-root/views-search/<91>.npz` | `$SV/views-search/` | `/home/ubuntu/jope-data/h317-stagev/views-search` (ro) |
| `$STAGE_V/stagev-root/run/{PREP_DONE.json,targets_manifest.json}` | `$SV/run/` | `/home/ubuntu/jope-data/h317-stagev/run` (rw) |
| `$DATA/stage_v/{LOCK.json,STAGE_V_STARTED,UNLOCK_RECEIPT.json,train_inputs_manifest.json}` | `$DATA/stage_v/` | same path (ro) |
| `$DATA/neural/lock/**` | `$DATA/neural/lock/` | same path (ro) |
| `$RECEIPT` | `$RECEIPT` | same path (ro) |
| `$STAGE/micro/{assets/models/dope_mlc_v3_multitask_best.pt, assets/projections/frozen_stage1_stage2_inference_params_v3.pt, python/model.py, python/hpc/landmark_workspace.py, bin/libjope_hpc.so}` | `/home/ubuntu/JopeDimeMicro/…` (sha-checked against LOCK `code_sha256`) | `/home/ubuntu/JopeDimeMicro/…` (ro); needed only so `resume_stage_v` can hash every LOCK file |

Not staged to xbabe0: DL-Val pools, the original `selected.json`, `best_dev_checkpoint.json`, the r882 costs,
`native.json`, `dedupe.json` and `views-full0`. The GP memo, envs and slots go to
`$STAGE_V/stagev-root/gp/` (rw). Memo files are keyed per context, so no state crosses contexts.

## 3. Commands (DO NOT RUN before each step's precondition)

### S0 write-lock (xbabe2, TRAIN sandbox; after every Gate H0 decision and the lock-fit pull)

```sh
cd $SRC && taskset -c 44,45 nice -n 19 ./sandbox_h317.sh python3 stage_v_h317.py write-lock \
  --h01 $DATA/neural/decisions/H0.1.json --h02 $DATA/neural/decisions/H0.2.json \
  --h03 $DATA/neural/decisions/H0.3.json --gp-decisions $DATA/gp/<run>/decisions.json \
  --lock-model-dir $DATA/neural/lock
# prints the LOCK sha256 on its last line; stage_v_runs=false => verdict NO_ADOPTION_STAGE_V_SKIPPED, stop here
```

### S1 receipt (parent, create-once)

```sh
( set -o noclobber; echo "<sha256>  $DATA/stage_v/LOCK.json" > $RECEIPT ) && chmod 0444 $RECEIPT   # then post on comms
```

### S2 unlock (xbabe2). `sandbox_h317_stagev.sh unlock` is `sandbox_h317.sh` plus the receipt; no DL-Val is visible

```sh
cd $SRC && taskset -c 44,45 nice -n 19 ./sandbox_h317_stagev.sh unlock python3 stage_v_h317.py unlock
# exit 0 => STAGE_V_STARTED exists; 3 => NO_ADOPTION_STAGE_V_SKIPPED or a sandbox refusal; a 2nd unlock is refused
```

### S3 build the V root (xbabe2 host shell; links and copies only, nothing is decoded)

```sh
test -s $DATA/stage_v/STAGE_V_STARTED || exit 3
mkdir -p $SV/res01-v-view/tables $SV/res01-v-view/pools $SV/inputs $SV/run
ln $DATA/res01-train-view/tables/*.jtt1 $SV/res01-v-view/tables/
ln $DATA/res01-train-view/pools/*.npz $SV/res01-v-view/pools/
ln $DATA/res01-train-view/{champions.json,winners.json,source_manifest.json} $SV/res01-v-view/
jq -r '.dev[].context_sha256' $DATA/res01-train-view/selected.json > $SV/inputs/target_cids.txt   # 91 lines
while read -r c; do
  ln /home/ubuntu/jope-data/res-01/input/tables/$c.jtt1 $SV/res01-v-view/tables/
  ln /home/ubuntu/jope-data/res-01/input/pools/$c.npz  $SV/res01-v-view/pools/
done < $SV/inputs/target_cids.txt
ln /home/ubuntu/jope-data/res-01/input/selected.json     $SV/res01-v-view/selected.json
ln /home/ubuntu/jope-data/res-01/input/cpu_exact_fused.py $SV/res01-v-view/cpu_exact_fused.py
install -m 0444 /home/ubuntu/jope_plan/evidence/H258/argon-t1405/inputs/r882/best_dev_checkpoint.json $SV/inputs/
install -m 0444 /home/ubuntu/jope_plan/evidence/H258/argon-t1405/inputs/r882-secondary-costs.npz $SV/inputs/
sha256sum $SV/res01-v-view/selected.json $SV/res01-v-view/cpu_exact_fused.py $SV/inputs/*.json $SV/inputs/*.npz
# expect 981960ad… b9f1468a… 30853cad78d8… 0e812aa7ea4c…; tables 1310, pools 1310
```

### S4 prep (xbabe2, `sandbox_h317_stagev.sh v`, 2 cores)

```sh
cd $SRC && taskset -c 44,45 nice -n 19 ./sandbox_h317_stagev.sh v \
  python3 stage_v_h317.py run --phase prep --threads 2
# exit 4 = REGRESSION_FAIL (stage stops; run --phase stats records the verdict).
# Failure before FIRST_OUTCOME_READ: rerun ONCE with --relaunch-before-outcome (disclosed).
# Failure after it: --resume (completes never-recorded steps only).
```

In this mode `$DATA/stage_v` is read-only. Prep writes only under `$SV`, plus the production guard log in `$DATA`.

### S5 stage to xbabe0 (xbabe2, only after `$SV/run/PREP_DONE.json`)

```sh
X0=xbabe0; L=$SV/run/x0_stage
sed 's|.*|tables/&.jtt1|' $SV/inputs/target_cids.txt > $L.tables.files
(cd $SV/res01-v-view && sha256sum $(cat $L.tables.files)) > $L.tables.sha256
sed 's|.*|views-search/&.npz|' $SV/inputs/target_cids.txt > $L.views.files
(cd $SV && sha256sum $(cat $L.views.files) run/PREP_DONE.json run/targets_manifest.json) > $L.views.sha256
ssh $X0 "mkdir -p $STAGE_V/dlval-view $STAGE_V/stagev-root/run $STAGE_V/stagev-root/gp $DATA/stage_v $DATA/neural/lock \
  /home/ubuntu/jope_plan/ns/receipts && cp -al /home/ubuntu/h317-stage/res01-train-view/. $STAGE_V/dlval-view/"
rsync -a --chmod=Fa-w --files-from=$L.tables.files $SV/res01-v-view/ $X0:$STAGE_V/dlval-view/
rsync -a --chmod=Fa-w --files-from=$L.views.files $SV/ $X0:$STAGE_V/stagev-root/
rsync -a --chmod=Fa-w $SV/run/PREP_DONE.json $SV/run/targets_manifest.json $X0:$STAGE_V/stagev-root/run/
rsync -a --chmod=Fa-w $DATA/stage_v/{LOCK.json,STAGE_V_STARTED,UNLOCK_RECEIPT.json,train_inputs_manifest.json} \
  $X0:$DATA/stage_v/
rsync -a --chmod=Fa-w $DATA/neural/lock/ $X0:$DATA/neural/lock/
rsync -a --chmod=Fa-w $RECEIPT $X0:$RECEIPT
scp -q $L.tables.sha256 $L.views.sha256 $X0:$STAGE_V/
ssh $X0 "cd $STAGE_V/dlval-view && sha256sum --quiet -c $STAGE_V/x0_stage.tables.sha256 && \
  cd $STAGE_V/stagev-root && sha256sum --quiet -c $STAGE_V/x0_stage.views.sha256 && \
  test \$(ls $STAGE_V/dlval-view/tables | wc -l) = 1310 && sha256sum $DATA/stage_v/LOCK.json $RECEIPT"
```

### S6 audit (xbabe0; needs the §1 docker additions)

```sh
H317_STAGE_V=$STAGE_V H317_CPUSET=16-63,80-127 ./sandbox_h317_docker.sh --stage-v \
  python3 stage_v_h317.py run --phase audit --cpus 16-57,80-121 --jobs 14 --shard 0/1
# 14 context jobs x 6 worker cores = 84 Rust workers; context order = largest nt*width first.
# GIL alternative: 7 processes `--shard k/7 --jobs 2` on 12 disjoint cores each.
# After an interruption: the same command + --resume (recorded units never rerun).
```

### S7 pull back + stats (xbabe2)

```sh
rsync -a xbabe0:$STAGE_V/stagev-root/run/{audit_units.*.jsonl,AUDIT_STARTED.*,AUDIT_DONE.*} $SV/run/
rsync -a xbabe0:$STAGE_V/stagev-root/run/events.jsonl $SV/run/events.xbabe0.jsonl
(sha256sum on both ends for every pulled file)
cd $SRC && taskset -c 44,45 nice -n 19 ./sandbox_h317_stagev.sh v python3 stage_v_h317.py run --phase stats
```

## 4. Wall-time estimate

| Phase | Host | Wall |
| --- | --- | --- |
| S2 unlock | xbabe2 | < 10 s: hashing ~110 MB of code, DOPE and model files |
| S4 prep | xbabe2, 2 cores | ≈ 2–3 min. Input shas ≈ 0.4 GB (≈ 2 s); dedupe 91 × 1,219 tables (measured 13–17 s on a TRAIN proxy in the sandbox); DOPE init + parity ≈ 10 s; 91 × (32 SEARCH views + full view 0) ≈ 35 s (0.31 s/context measured for 32 views); regression ≈ 1 s; V-native 91 × 2 seeds × 1,152 candidates ≤ 1 min |
| S6 audit | xbabe0, 84–88 workers | see below |
| S7 stats | xbabe2 | < 1 min (10,000 bootstrap / sign-flip draws per contrast; 2 Elo panels) |

S6 audit, step by step:
- Inputs: 91 × 2 × ≈ 270 live SEARCH requests = 49,140.
- Within-context memo sharing saves ≈ 15%. H317_LOCKED and EVOA_MATCHED share the seed set; NN8_PORTFOLIO reuses
  NN8_PICK. That leaves ≈ 42k physical SEARCH evals × 1.41 s ≈ 59k CPU-s.
- AUDIT evals: ≈ 15 per context-seed at ≈ 2× cost, ≈ 7.6k CPU-s.
- Total ≈ 67k CPU-s, about 13 min on 84 workers if perfectly packed.
- Generations are 6-eval barriers with a heavy tail (p95 5.6 s), so packing efficiency is ≈ 35–45%.
- **Expected ≈ 35–60 min. Planning bound 1.5 h**, which also covers host-CPU > 85% pauses.
- Tail: the largest context's 2-seed chain is ≈ 100 rounds × 4–12 s ≈ 7–20 min. LPT ordering starts it first.

## 5. Readout notes (parent rulings 2026-11-04)

- **Dedupe is descriptive, never a gate.** Both modes are reported. `raw_sorted` (exact raw values, sorted within each
  row) is the interpretive one. Per-table standardization (`std`) is blind to row-subset duplication: on a TRAIN proxy
  it found 0 overlap even inside families that share 60–75% of rows.
- **TOP4_PICK is descriptive.** H258's 1.252000 is the median over its 3 seeds of the family-equal mean. TOP4_PICK
  reads no RNG, so the Stage V family-equal mean is compared with it directly (`abs_diff_family_vs_h258`).
- **Pinned worker:** `bootstrap.s1main` e3be9386… The "e2be…" in the brief was a typo.

## 6. Retention

Archive to S3: `metrics.json`, `panel_stage_v.json`, `native.json`, `dedupe.json`, `encoder_parity.json`,
`targets_manifest.json`, `audit_units.*.jsonl`, `events*.jsonl`, `PREP_DONE.json`, LOCK.json and the receipts.
DL-Val tables, pools and views stay local. Close them out under the PREREG_TEMPLATE retention receipt; no deletion
without owner confirmation.
