# H335 Stage V: hosts, mounts, staging and commands (runbook)

Status: **plan only; nothing here has been run.** Every step from S2 on touches DL-Val and may run only once its
precondition holds. Driver: `stage_v_h335.py` (PREREG §4–§5); it reuses the frozen H317 driver `stage_v_h317.py` by
import (H317's `stage_v_mounts.md` is the long form of this note; differences are in §4).

`SRC` = this directory (the H335 source path). `CODE=/home/ubuntu/h335-stage/code` = the frozen read-only copy
(`H335_CODE`): every step runs from it, and the sandboxes bind it at `SRC`. `DATA=/home/ubuntu/jope-data/h335-cl`.
`SV=/home/ubuntu/jope-data/h335-stagev` is the Stage V root on xbabe2 (outside `$DATA`; the guard denies it before
the unlock). `RECEIPT=/home/ubuntu/jope_plan/ns/receipts/H335_LOCK_sha256.txt`. On xbabe0, `STAGE_V=/home/ubuntu/h335-stage-v`.

## 1. Steps
| Step | Host | How | DL-Val visible? | Writes |
| --- | --- | --- | --- | --- |
| S0 write-lock | xbabe2 | `sandbox_h335.sh` (TRAIN view) | no | `$DATA/stage_v/{LOCK.json, train_inputs_manifest.json}` |
| S1 receipt | xbabe2 (parent) | shell | no | `$RECEIPT` (create-once, 0444) + AGENT_CHAT |
| S1b stage + verify LOCK files on xbabe0 | xbabe2 → xbabe0 | `stage_v_xbabe0.sh lock` | no | `$CODE`, `$DATA/bin`, `$DATA/stage_v`, `$RECEIPT` on xbabe0 |
| S2 unlock | xbabe2 | `sandbox_h335_stagev.sh unlock` | **no** | `STAGE_V_STARTED`, `UNLOCK_RECEIPT.json` |
| S3 build `$SV` | xbabe2 (parent) | `build_stagev_view.sh` (links / copies only) | links | `$SV/{res01-v-view, inputs, run}` |
| S4 prep | xbabe2 | `run_prep_v.sh` unit | 91 tables + pools, R882 inputs | `$SV/{run, views-*, llm-envs}` |
| S5 stage | xbabe2 → xbabe0 | `stage_v_xbabe0.sh prep` (after `PREP_DONE.json`; the LLM step may still run) | 91 tables + views | `$STAGE_V` |
| S6 audit | xbabe0 | `run_audit_v.sh` unit (docker `--stage-v`) | 91 tables, no pools | `$STAGE_V/stagev-root/{gp, run}` |
| S7 pull back | xbabe2 ← xbabe0 | `stage_v_xbabe0.sh pullback` | records | `$SV/{gp/stages, run/...}` |
| S8 stats | xbabe2 | `sandbox_h335_stagev.sh v` | records | `$SV/run/{FIRST_OUTCOME_READ, metrics_v.json, panel_stage_v.json}` |
| S9 seal xbabe0 | xbabe2 → xbabe0 | `stage_v_xbabe0.sh first-outcome` | – | `FIRST_OUTCOME_READ` on xbabe0 |

## 2. What the LOCK verifies (unlock and every `resume_stage_v`, on every host)
- **`code_sha256`** (runtime code only):
  - `LOCK_CODE_FILES` = h335_core.py, h335_lab.py, h335_stats.py, h335_frontier.py, frontier_audit_h335.py,
    stage_v_h335.py, gp_grammar_h317_learners_v1.py, sandbox_h335.sh, sandbox_h335_stagev.sh, sandbox_h335_docker.sh;
  - the 18 frozen H317 pins;
  - the Micro featurizer files `/home/ubuntu/JopeDimeMicro/{python/rust_evaluator.py, python/rank_prepare.py,
    assets/projections/program-feature-names.json}`;
  - `$DATA/bin/{bootstrap.s1main.superset-2a848c8f8085, frontier_audit_h335}`, the 10 `$DATA/neural/runs/H0.1-A/**/model.pt`,
    `$DATA/neural/views_c32_train1219.npz`, RES01 `champions.json` / `winners.json` and `train_inputs_manifest.json`.
- **`docs_sha256`** (recorded, NOT verified): every other file in `CODE` (PREREG.md, *.md, test_*.py, run_*.sh, stage_*.sh,
  h335_replay.py, h335_encode.py, *.rs), so they may still change.
- **`prep_sha256`** (checked by prep only): the xbabe2 encoder stack.
- write-lock also requires every Phase T run dir's `config.json` `code_sha256` (H317 pins + h335_core / h335_lab /
  h335_stats / learners grammar) to equal the current files.

## 3. Commands (DO NOT RUN before each step's precondition)
```sh
cd $CODE   # every step below runs from the frozen copy
# S0 (xbabe2) -> prints the LOCK sha256 last; refuses a hard stop or a Phase T code mismatch
H335_CODE=$CODE ./sandbox_h335.sh env H335_GROUP=BASE python3 stage_v_h335.py write-lock \
  --decision $DATA/decision_T.json --phase-t-run-dirs $DATA/runs/T_xbabe2,$DATA/runs/T_xbabe0
# S1 (parent)
( set -o noclobber; echo "<sha256>  $DATA/stage_v/LOCK.json" > $RECEIPT ) && chmod 0444 $RECEIPT
# S1b (xbabe2; LOCK files + the S6 script run_audit_v.sh (sha from docs_sha256) staged and checked at their xbabe0
#      host paths, LOCK files also inside the docker sandbox; nothing unlocks)
./stage_v_xbabe0.sh lock
H335_CODE=$CODE ./sandbox_h335.sh env H335_GROUP=BASE python3 stage_v_h335.py verify-lock-files --prep   # xbabe2 too
# S2 (xbabe2): exit 0 => STAGE_V_STARTED; a second unlock is refused
H335_CODE=$CODE ./sandbox_h335_stagev.sh unlock python3 stage_v_h335.py unlock
# S3 (xbabe2 host shell, after S2; hard links / read-only copies only, nothing decoded; fatal checks; idempotent)
./build_stagev_view.sh
# S4 (xbabe2; cpus 38-53,102-117 (the freed Phase T set; 54-63 are now h344-run's), nice 19; prints counts only; exit 4 = REGRESSION_FAIL)
systemd-run --user --unit h335-v-prep -p MemoryMax=32G -p MemorySwapMax=0 -p CPUQuota=2000% $CODE/run_prep_v.sh
#   after an interruption: the same command with a new unit name + --resume (any number of times before FIRST_OUTCOME_READ)
# S5 (xbabe2, after $SV/run/PREP_DONE.json); ends with a dry run of the exact S6 container layout
#    (sandbox_h335_docker.sh --stage-v ... verify-lock-files)
./stage_v_xbabe0.sh prep
# S6 (xbabe0; prints counts only; after an interruption: a new unit name + --resume)
ssh xbabe0 systemd-run --user --unit h335-v-audit -p MemoryMax=100G -p MemorySwapMax=0 $CODE/run_audit_v.sh
# S7 (xbabe2, after the audit unit ended and the prep unit's LLM step wrote LLM_DONE.json)
./stage_v_xbabe0.sh pullback
# S8 (xbabe2) refuses (records stay sealed) while any of the 364 units / 364 LLM records is missing;
#    --final-incomplete (INCONCLUSIVE) only once every planned shard has AUDIT_DONE and run/LLM_DONE.json exists,
#    or after 2026-11-05T07:30:00Z
H335_CODE=$CODE taskset -c 38-53,102-117 nice -n 19 ./sandbox_h335_stagev.sh v python3 stage_v_h335.py stats
# S9 (xbabe2)
./stage_v_xbabe0.sh first-outcome
```

## 4. Differences from H317 Stage V
- `FIRST_OUTCOME_READ` is created by `stats` (PREREG §4.4) after every record has passed its identity check. It binds
  the sha of every record file and blocks any later audit unit or prep. Prep resumes any number of times before it;
  the regression record (`native.json`) is never recomputed, and a recorded FAIL stops everything.
- No V-native / TOP4_PICK. The LLM regressors are part of prep (`run/llm_audit.jsonl`, after `PREP_DONE.json`).
  `h335_frontier.py --phase V` is not used for Stage V.
- Units are (context, seed, group) records from `h335_lab.run_units`, run one unit at a time per job slot, with
  H317's `MAX_UNIT_ATTEMPTS` = 2, counted over durable culprit lines only (`unit_attempts.*.jsonl`): a unit is capped
  only after it raised twice in-process; units in flight when a process was killed are never capped (resume with a
  lower `--jobs`; at the 07:30Z stop an incomplete run reads INCONCLUSIVE). An integrity error writes `STOPPED_INTEGRITY`
  (audit, prep and stats refuse) and cancels the shard's pending units. Locks, markers and `AUDIT_DONE.*.<attempt>.json`
  are per (group, shard).
- The LOCK is `jope.h335.lock/v1` with absolute keys, runtime code only (`docs_sha256` is not verified). The DOPE files
  are in the prep-only group.

## 5. Retention
Archive (create-only): LOCK.json, receipts, `run/{metrics_v.json, panel_stage_v.json, native.json, dedupe.json,
encoder_parity.json, targets_manifest.json, llm_envelopes.json, llm_audit.jsonl, PREP_DONE.json, LLM_DONE.json,
FIRST_OUTCOME_READ, AUDIT_*, unit_attempts.*, events*.jsonl}` and `gp/stages/`. DL-Val tables, pools, views and
envelopes stay local. No deletion without owner approval.
