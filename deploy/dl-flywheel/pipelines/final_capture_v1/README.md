# final_capture_v1 (W10 r2): terminal continuous-generation capture

Schema `jope.continuous-generation-final-capture/v2` (r1 wrote `/v1`). This is source and private CPU fixtures only. It
is a proposal for independent review by the live owner (Codex /root), who keeps the live final capture. Nothing here
has touched the live campaign beyond the read-only reads listed under "Live facts".

## r2: the `runtime_limit_stopped_idle` terminal class (owner ruling R17)

v1 did not end through its own closure. systemd's `RuntimeMaxSec` stopped it at 00:29:05Z (`Result=timeout`,
`ExecMainStatus=15`, `NRestarts=0`). At that point `training/progress.json` still said
`waiting_for_mandatory_gp_capacity`; it was last written at 00:26:03.947Z, before the in-loop deadline of 00:26:06.38Z.
No `status.json` or `result.json` exists. R17 (receipt `rulings/owner-askuserquestion-receipt-R17-v1.json`, sha256
`2a1b0a5e…b534`) accepts this end as a clean stopped terminal state, but only once a read-only capture proves four
things. r2 adds exactly that class. The classification is closed:

| class | progress.state | requires |
| --- | --- | --- |
| `clean_terminal` (r1, unchanged) | `global_deadline_reached`, `maximum_updates_reached` | r1 checks. Passing `owner_ruling` or `live_walk` refuses. |
| `runtime_limit_stopped_idle` | `IDLE_WAITING`: `waiting_for_mandatory_gp_capacity`, `waiting_for_output_capacity`, `waiting_for_unique_batch` | every r1 check, plus the gates below |
| anything else (`running`, `failed`, unknown) | | refused: `non-terminal or failed progress state refused: <state>` |

`IDLE_WAITING` lists every waiting state of the pinned runner (`prepared/continuous_runner.py`, sha `10cbf4ea…`, the
same bytes as the live execution root's runner). The runner publishes each of these before it reserves anything in
that loop iteration. `waiting_for_unique_batch` follows a `reserve()` that returned `Waiting` and wrote no batch row.
A cold-recovery waiting state can still sit on a certified reservation, so the planner gate below still applies.

The R17 gates, in check order:
1. **Unit stopped (R17 proof 1).** The r1 owner check (inactive or failed, `MainPID 0`, cgroup absent or empty, read
   before and after) must pass. The readback must also show `Result == 'timeout'`; `exit-code`, `signal`,
   `core-dump`, `oom-kill`, `watchdog`, `success` and every other value refuse. `NRestarts` is now part of the closed
   readback, and it must be `0`.
2. **Ruling.** `owner_ruling` must be bytes whose sha256 equals the pinned R17 receipt sha. The capture records
   `owner_ruling: {id: 'R17', receipt_sha256}`. The reader refuses an R17-class document without that binding.
3. **In-loop deadline.** `window.json` must join `config.json`:
   - `config_sha256` equals the campaign sha;
   - `deadline_unix == config.global_deadline_unix`;
   - `closure_reserve_seconds` is finite and ≥ 0.

   The cutoff is `window.deadline_unix − config.closure_reserve_seconds`, which is the runner's own in-loop deadline
   with nothing else subtracted. It is computed exactly (`Fraction`, floored to ns). It must already have passed.
4. **No write after the deadline (R17 proof 2).** `walk(root)` makes a bounded, lstat-only, non-following walk of the
   private copy. It records `{path, kind, bytes, mtime_ns}` for every entry ('.' is the root), covering directories as
   well, because a create, delete or rename shows up only in the directory's mtime. The walk:
   - admits only regular files and directories, so any symlink (escaping the root or not, never followed) or special
     file refuses;
   - is bounded at `WALK_MAX_ENTRIES` (1,000,000) entries and `WALK_MAX_BYTES` (256 MiB) of canonical records;
   - requires every `mtime_ns ≤ cutoff`.

   Its closed summary must equal `live_walk()`, the same walk of the live owned root. That comparison runs before the
   first input read, and the live walk is repeated after the last read. This binds the private copy to the live root
   (paths, kinds, sizes, mtimes), so the proof is about the live root and not just the copy. The records are sealed as
   chained `walk-NNNNNN.jsonl` parts, and `load_capture` re-verifies their order, digest and mtimes.
5. **Planner (R17 proof 3).** r1 already runs `verify()` on the private planner backup and checks 0 unresolved and 0
   non-committed batches, contiguous commits, the proof/ACK joins and the newest rolling slot. On top of that, an
   explicit R17 closure requires the last committed absolute step == `progress.optimizer_steps` == the rolling
   manifest step == the newest rolling slot. It also requires that commit's `checkpoint_after` == the rolling
   manifest parameter sha == the planner cursor == `progress.checkpoint_sha256`.
6. **Absent, never reconstructed.** `status.json` (campaign root) and `training/result.json` must not exist. The
   document records `result_json: 'absent', status_json: 'absent'`.
7. **CPU restore (R17 proof 4).** r1's throwaway CPU restore of the final rolling slot must reproduce the sha. It is
   recorded under `terminal.runtime_limit.r17_proofs.cpu_restore`.

`terminal.runtime_limit` holds `deadline`, `walk` (summary + part chain) and `r17_proofs`, which has four entries:
`unit_stopped`, `no_write_after_in_loop_deadline`, `planner` and `cpu_restore`.

### r2 integration changes for the live capture

1. `owner_readback()` must also return `NRestarts` (an int, from `systemctl show -p NRestarts`).
2. Copy the whole campaign root, preserving metadata, after the readback passes: every file and directory, with
   `mtime_ns` preserved and symlinks not followed (for example `cp -a`). A copy that does not preserve mtimes refuses,
   because its mtimes fall after the cutoff, and a partial copy fails the live-walk equality.
3. Pass the following:
   - `owner_ruling=<bytes of rulings/owner-askuserquestion-receipt-R17-v1.json>`;
   - `live_walk=lambda: C.walk(LIVE_ROOT)[1]`. This does lstat and listing on the live root only; no file is opened.

   For reference, a read-only `find` at 2026-10-10 ~00:55Z found the live root at 52,329 entries (60 dirs, 52,269
   files, depth 4, no symlinks, no `*.pending`). Nothing was newer than 00:26:04Z. After the last commit only
   `training/progress.json` (00:26:03.947Z) and the `training/` directory (00:26:03.984Z) changed.
4. For this class, **never open SQLite in the live root**, including through a live-SQL `outbox_reader`. A read-only
   WAL open can rewrite or create `-shm`, and the live walk would then correctly refuse. Use
   `probe.staged_outbox_reader` on the private copy. It byte-copies the parent outbox from its own location, which is
   outside this root.
5. The recovery CLI publishes `status.json` and `fatal.json` in its **execution root**
   (`/home/ubuntu/JopeDime/.agent/proposals/dl-flywheel-recovery5-20261009/`, outside the campaign root). The capture
   cannot see that directory. A read-only `ls` at ~00:55Z showed neither file there. See the open questions.

Base: PR277 head `745be43fa28685bc4fc464ad00169ea2850e2e73` (`deploy/dl-flywheel` export, BASE_MANIFEST sha
`a9fc94eb…5909`). The package imports the base modules unchanged. `SOURCE_PINS.json` pins 20 base modules, all 10
package Python files and the R17 receipt fixture input (`tests/owner-askuserquestion-receipt-R17-v1.json`, a byte copy
of `rulings/owner-askuserquestion-receipt-R17-v1.json`).

## Why a successor capture

The frozen `pipelines/prepared/reconcile.py:capture` only accepts `running` and `maximum_updates_reached`, and it
hard-codes origin 429. The v1 learner `jopedime-neural-continuous-recovery5-20261009` leaves its loop at the in-loop
deadline (00:26:06Z) with state `global_deadline_reached`. Its `finally` publishes `progress.json` first. While the
`result.json` dict is being built, it runs `outbox.verify()` and then `control()` (which publishes
`optimizer-control.json`). Only after that does it publish `result.json`. `RuntimeMaxSec` (00:29:05Z) can kill the
process anywhere in that sequence, so `result.json` may be missing, and the planner/outbox `-wal` files may hold
frames that were never checkpointed.

## API

```python
from pipelines.final_capture_v1 import capture as C
body, receipt = C.capture(private_campaign_root, new_out_dir, campaign_sha256=sha(config.json), unit=UNIT,
                          owner_readback=callable, outbox_reader=callable,      # part_max defaults to 8 MiB
                          owner_ruling=r17_receipt_bytes, live_walk=lambda: C.walk(LIVE_ROOT)[1])  # R17 class only
body, joins, ancestry = C.load_capture(new_out_dir)   # the only supported reader; ancestry -> BatchStream.inherit
records, summary = C.walk(root)                       # lstat-only; safe on the live root
```

## Output layout

The output is a new directory, published by an atomic rename from a staging directory beside it. If the capture
refuses, it leaves no output and no staging directory. Every file is create-only (`O_EXCL|O_NOFOLLOW`), mode 0444
and fsynced.

| file | content |
| --- | --- |
| `final-capture.json` | Deterministic capture document, at most 1 MiB. It contains no clock value and no private path. |
| `capture-receipt.json` | `capture_sha256`, `captured_at_unix`, the private input and output paths, and a sha256/bytes entry for every file. |
| `planner-backup.sqlite` | The private planner staged as a byte copy (db + `-wal`), then copied by one single-step `sqlite3` online backup (one read transaction). It is converted to a rollback journal and sealed. Every planner read uses it with `immutable=1`. |
| `joins-NNNNNN.jsonl` | One canonical record per committed batch: `absolute_step, local_step, batch_id, epoch, index, fingerprint, proof_sha256, checkpoint_before, parameter_sha256, ack_seq`. |
| `fingerprints-NNNNNN.jsonl`, `exposed-NNNNNN.jsonl` | The lineage fingerprints and exposed context ids, sorted and unique. |
| `walk-NNNNNN.jsonl` (R17 class only) | One `{path, kind, bytes, mtime_ns}` record per campaign-root entry, sorted by path. |
| `active-manifest.json`, `optimizer-control.json`, `checkpoint/{manifest.json,state.safetensors}` | Sealed byte copies. A throwaway CPU `continuous_state.restore` runs on the sealed checkpoint. |

### Chunked parts

Each `*-NNNNNN.jsonl` part is at most `PART_MAX` = 8 MiB, and a record never spans two parts. Each part entry pins
`file, ordinal, sha256, bytes, count` and a link:

`link = digest({schema: 'jope.final-capture-part-chain/v1', kind, ordinal, previous, file, sha256, bytes, count})`

The first `previous` is 64 zeros. The document pins the ordered part list, its `count` and its `head` (the last
link).

`load_capture` refuses in each of these cases:
- a missing, tampered, swapped or renumbered part;
- a truncated list;
- non-canonical lines;
- joins that are out of order or not contiguous, or that break the checkpoint chain;
- unsorted or duplicate fingerprints or ids;
- a reassembled ancestry whose digest differs from `ancestry.ancestry_sha256`.

Before publishing, `capture()` reads its own output back with `load_capture`.

## Refusals

Every refusal raises `ValueError`. The order below is the order of the checks.

1. **Owner.** The readback must be closed, and must show `ActiveState ∈ {inactive, failed}`, `MainPID == 0`, and a
   recorded cgroup that is absent or empty (`procs == []`, `populated is False`). It is read before the first read of
   any input and again after the last. The two readbacks must be identical.
2. **Inputs.** The config sha must match, and the root must be a private copy, never `config.owned_root`. The runtime
   must exactly match `config.runtime_versions` (python/numpy/torch/safetensors). The six verifier modules must be
   byte-identical to `config.owned_sources`.
3. **Terminal markers.** Any of `failure.json`, `failure.json.pending`, `uncertain-physical-state` or
   `closure-errors.jsonl` refuses. `progress.state` must be `global_deadline_reached` or `maximum_updates_reached`
   (`clean_terminal`), or one of `IDLE_WAITING` (`runtime_limit_stopped_idle`, whose gates 1–4 above run here, before
   the planner).
4. **Planner.** `verify()` must pass on the backup with `unresolved_batches == 0`. Any batch that is not committed
   (`reserved`/`needs_reconcile`) is a **HOLD**. The planner config must join the campaign. The inherited ancestry
   must have one batch per absolute step, and `training/ancestry.json` must equal it. The manifest chain must end at
   the active manifest, and epochs must be contiguous.
5. **Every committed batch.** The proof file must exist. Its sha must equal the commit's `evidence.event_sha256`. The
   absolute and local step, the intent and rows key set, the checkpoint chain and the proof checkpoint must all join.
   The pre-optimizer ACK must carry the same batch id, global step and pre-update checkpoint, with a strictly
   increasing head seq. An orphan `batch-(last+1).json` is a **HOLD**.
6. **Outbox.** The summary must be closed. The v1 `batches` and `ack_count` must both equal the committed count. The
   ACK digest must equal `digest([proof.emission_ack in step order])`. Observations must equal the sum of emission
   ids. `numeric_scores_synthesized == 0`.
7. **Checkpoint.** The newest rolling slot must be the last commit (a later slot is a **HOLD**). The rolling manifest
   must equal the final proof checkpoint. The state sha and bytes must match. The key set must be `model.*`, AdamW
   `{step, exp_avg, exp_avg_sq}` and `rng.torch_cpu`, plus `rng.torch_cuda.<i>` where present.
8. **One parameter sha** must hold across the planner, progress and optimizer-control. Control must carry the final
   absolute step and state sha, finite LR groups within `[floor, learning_rate]`, and the plateau
   patience/floor/bad/reductions/best (never reset).
9. **Progress reconciliation.** Steps, lineage epochs, unique contexts, presentations, the fixed panel, the planner
   summary, `resumed_from`, the plateau monitor, outbox batches and observations, and the latest full validation epoch
   must all reconcile.
10. **`result.json` is optional for `clean_terminal`.** If it is absent, checks 1–9 are the closure proof
    (`terminal.closure = proven_without_result_json`). If it is present, it must agree exactly, so a stale or failed
    result refuses. For `runtime_limit_stopped_idle`, `status.json` and `result.json` must both be absent; the R17
    closure runs just before check 7's manifest closure.
11. **Restore.** A throwaway CPU `restore` must reproduce the parameter sha and the optimizer group count. The caller's
    torch RNG is restored afterwards.
12. **Ancestry.** The lineage head must have every generation resolved, with committed batches equal to the absolute
    steps. A fresh dry-run journal must accept `inherit(ancestry)` twice.

## Ancestry for v2.0.0 (exactly what `BatchStream.inherit` accepts)

| key | value |
| --- | --- |
| `planner_ref` | `<config.owned_root>/training/batches.sqlite` |
| `head` | `{schema: 'jope.continuous-lineage-planner-head/v1', committed_batches: L, unresolved_batches: 0, generations: [{planner_ref, head}...]}`. The generations are the v1-inherited parent head followed by v1's own `verify()` head. A lineage head is flattened in the next generation. |
| `checkpoint_sha256` | The final parameter sha. |
| `optimizer_steps` | The absolute step: origin + local steps. |
| `completed_epochs` | The lineage total (ancestral + v1 complete). |
| `fingerprints` | The lineage total: parent ∪ v1's own (`L` = absolute steps). |
| `exposed_context_ids` | The lineage union, sorted. |
| `presentations` | The lineage total. |

## Capacity (measured; receipt `capacity-20261009T231551Z.json`)

The capture has to work at v2's update cap. At that cap the per-batch join records (~40 MB) are well beyond what one
8 MiB part or one 16 MiB report can hold, which is why the joins and lists are split into parts.

The fixture world's real capture measured **483 B per join record** (3-digit steps). It produced a 10,854 B document
and a 212,992 B planner backup.

The rows below run synthetic records with live field widths through the real `write_parts`/`read_parts`. Two of the
counts are deliberately overstated: the v1 row uses 17,773 joins (v1's own count was 17,344 at the last progress
read), and its fingerprint count adds the 429 parent fingerprints a second time.

| scale | joins | join B/record | join parts (max bytes) | join total | fingerprints | fp parts | exposed | document (est.) | `inherit` event (canonical ancestry) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v1 actual | 17,773 | 494.2 | 2 (8,388,409) | 8.78 MB | 18,202 | 1 (1.22 MB) | 7,248 → 1 part (0.49 MB) | 11,075 B | 1.71 MB of 16 MiB |
| v2 cap | 80,000 | 496.6 | 5 (8,388,562) | 39.73 MB | 97,773 | 1 (6.55 MB) | 7,248 → 1 part | 11,738 B | 7.04 MB of 16 MiB |

At 7,248 exposed ids, `BatchStream.inherit` stores the whole ancestry in a single `stream.ancestry` event, which is
bounded by `MAX_EVENT` = 16 MiB. That event bounds the lineage at **≈243,137 fingerprints** (67 B each). v2 at its cap
(97,773) fits, and so would one more 80k generation (≈177,773). The generation after that would not; see the open
questions.

## Integration points Codex supplies

1. **`owner_readback()`** must return exactly
   `{unit, LoadState, ActiveState, SubState, MainPID, Result, NRestarts, ControlGroup, cgroup:{path, exists, procs, populated}}`
   with no time-dependent fields (`NRestarts` is new in r2).
   - The systemd fields come from
     `systemctl [--user] show -p LoadState,ActiveState,SubState,MainPID,Result,NRestarts,ControlGroup <unit>`.
     `MainPID` and `NRestarts` are ints. The v1 unit is a `--user` unit.
   - `cgroup.path` is the cgroup the unit ran in (for example `/sys/fs/cgroup/system.slice/<unit>.service`).
   - If that path exists, fill `procs` from its `cgroup.procs` and `populated` (a bool) from its `cgroup.events`.
     Otherwise use `exists: false, procs: [], populated: null`.
   - `ControlGroup` must be `''` or equal to `cgroup.path`.
   - A `RuntimeMaxSec` kill yields `ActiveState=failed, Result=timeout`. `clean_terminal` accepts it unchanged from r1.
     `runtime_limit_stopped_idle` requires it (r1 open question 1 is answered by R17 for that class).
2. **`outbox_reader()`** must return `{v1: Outbox.verify() + {acks_sha256, ack_count}, parent: Outbox.summary()}`,
   where `acks_sha256 = digest([json.loads(ack) for ack rows sorted by ack.head.seq])`.
   - Consumers keep advancing outcome states. The capture therefore joins only batches, observations and the ACK
     digest (immutable once the learner stops) and records the rest.
   - `probe.staged_outbox_reader` is a reference implementation that reads byte-staged copies of private files. Live
     SQL is the owner's choice.
3. **Exact runtime pins.** The capture refuses unless the runtime equals `config.runtime_versions`. For the live
   campaign that is python 3.12.3, numpy 1.26.4, torch 2.7.0 and safetensors 0.7.0.
   - The xbabe3 fixture venv has **safetensors 0.8.0**. The fixtures pass only because the fixture config records the
     runtime it actually used. Run the live capture in the runner's own venv.
   - The six verifier sources at base 745be43 are byte-identical to the live `config.owned_sources` (checked by sha256).
4. **Private copy.** After the readback passes, copy the campaign root to a private path. For `clean_terminal` that
   means `config.json` and the whole `training/` (`batches.sqlite`, `-wal`, `-shm`, `outbox.sqlite*`, proofs,
   `rolling/`). For `runtime_limit_stopped_idle` it means the **whole root, metadata-preserving** (r2 integration
   change 2). The capture never opens SQLite in that copy; it stages bytes into its own scratch directory.
5. **R17 class only:** `owner_ruling` (the receipt bytes) and `live_walk` (a callable returning
   `C.walk(LIVE_ROOT)[1]`).

## Next-update parity probe (`probe.py`, fixtures and owner review only)

Two independent paths each derive v2's inputs:
- `captured_inputs` reads only the sealed output, through `load_capture`;
- `fresh_resume_inputs` is an oracle with its own SQL over a byte copy of the private planner, plus the rolling slot
  and the control file.

`first_update` then performs the successor's first update for each path: a fresh journal, `inherit`, `restore`, the
carried LR groups, `reserve`, `prepare_microbatches`, one AdamW dropout step and a checkpoint. It works on a new
scratch journal that is then discarded.

The fixture requires the same batch, parameter sha, state bytes and every tensor, including AdamW and `rng.torch_cpu`,
from both paths. It also shows that halved LR groups, or a reseeded dropout RNG, change the update. The original world
and the private copy are byte-identical afterwards.

The probe assumes v2 keeps the v1 runner's epoch argument: `prepare_microbatches(epoch=ancestry.completed_epochs +
batch.epoch - 1)`.

## Fixtures (r2: 32 = r1's 14 adapted mechanically + 18 new; receipts in `w10/receipts-r2`, see `DELIVERY-r2.json`)

r1's own receipt was `final_capture_v1-20261009T231551Z.json` (14/14). r2 adapts r1's fixtures only mechanically:
schema v2, `terminal.class`, `owner_ruling: None`, and `NRestarts=0` in the fixture readback.

The worlds are written by the base runners themselves:
1. `baseline_runner` produces the v1 genesis from fixture parent 429, whose ancestry is labelled synthetic.
2. `reconcile.capture` captures it.
3. `continuous_runner` resumes, adopts a prepared manifest and exits at an injected in-loop deadline.

There are three world kinds:
- `killed`: the unit is killed before `result.json`, and the copy keeps the live `-wal`/`-shm`;
- `clean`;
- `failed`: an authority error raised inside the loop.

| case | fixtures |
| --- | --- |
| (a) | Capture with `result.json` missing, then `inherit()` on a fresh journal. A present `result.json` must join exactly. |
| (b) | Refusals: failed or uncertain state; reserved / needs_reconcile / orphan proof / later slot; parameter-sha disagreement (progress, control, planner); checkpoint byte tamper, raw and rehashed; owner active, deactivating, PID, non-empty cgroup, wrong unit, unclosed readback, or changing during capture; proof bytes, missing proof, ACK digest, dropped ACK, unclosed summary. |
| (c) | Byte-identical determinism across copies. |
| (d) | Parity. |
| (e) | Private-copy-only, pins and create-only output. |
| (f) | Synthetic tensor headers, including CUDA RNG. |
| (g) | Chunk boundaries, tamper, missing, swap, renumber, truncate and re-chained reorder. |
| (h) | v1 and v2 capacity. |

r2 adds a fourth world kind, `idle` (`tests/world.py`). It is the live v1 end, produced by the pinned runner itself:
- the shared unresolved cap (`max_pending_emissions` 11) is reached after the same three updates, so the runner
  publishes `waiting_for_mandatory_gp_capacity` and sleeps without reserving;
- the in-loop deadline fires, and the RuntimeMaxSec kill lands on the closure's first read, so nothing more is
  published;
- `config.json` carries `global_deadline_unix`/`closure_reserve_seconds` and the root carries `window.json`;
- a fixture clock then shifts every mtime uniformly so the newest entry sits 2 s before an in-loop deadline that is
  itself in the past.

Each counterexample tampers a private copy inside `frozen_times()`, which models a live root that reached that state
before the deadline, so exactly the guard under test refuses. The new fixtures (`tests/test_runtime_limit.py`):

| case | fixtures |
| --- | --- |
| live shape | The capture is produced. The document records `class runtime_limit_stopped_idle`, `result_json`/`status_json` `absent` and `owner_ruling {R17, 2a1b0a5e…}`. All four R17 proofs hold. A fresh journal inherits. The bytes are deterministic across copies, and `inactive/timeout` is accepted. An mtime exactly at the cutoff is admitted. |
| (a) | Reservation (`reserved`, `needs_reconcile`) under a waiting state: HOLD. |
| (b) | A file, `-wal`, directory, new file or `config.json` 1 ns after the cutoff. A window that does not join the config, or is missing. A clock at or before the cutoff. |
| (c) | A non-empty cgroup or `MainPID≠0` (r1 guard); `NRestarts` 1 and 3. |
| (d) | `Result` exit-code, signal, core-dump, oom-kill, watchdog, success, start-limit-hit; inactive/exit-code. |
| (e) | `running`, `failed`, `uncertain_optimizer_outcome`, unknown waiting state, typo'd deadline state; `uncertain-physical-state` marker. |
| (f) | Progress one step behind; progress sha of the previous slot; a rolling manifest sha that is not `checkpoint_after`. |
| (g) | The receipt missing, empty, as text, with a trailing newline, with another ruling id, or with another decision. A clean-class capture given R17 inputs. |
| (h) | A symlink escaping the root (dir, file), a symlink inside the root, a fifo. The entry and byte bounds, with the exact bound admitted. |
| (i) | The live walk missing, or the copy has an extra or lost file or a changed mtime. The live walk changes during the capture. |
| (j) | `status.json` present; stale `result.json` present. |
| (k) | Reader: re-chained walk records with an mtime after the cutoff (with and without a re-digest), reversed or dropped. |

Fault injection (`fault_injection.py`, r2) has 31 mutants: r1's 16, with targets and fixture lists adapted
mechanically, plus 15 new (one per new guard, and one edit that empties the idle allow-list). Every mutant makes its
fixtures fail. The unmutated control passes, and the original bytes are re-proven against the pins afterwards. r2
also classifies each failing case:
- `accepted`: the counterexample was captured;
- `refused_otherwise`: a different, later refusal still caught it (defense in depth);
- `error`.

r1's run history: `fault_injection-20261009T231551Z.json` caught 16 of 16. The earlier `…231328Z` run reported
`passed:false` only because its child harness attributed subTest failures to subtest ids.

```sh
JOPE_W10_BASE_ROOT=<base>/deploy/dl-flywheel python -B pipelines/final_capture_v1/run_offline.py --receipt OUT.json
JOPE_W10_BASE_ROOT=... python -B pipelines/final_capture_v1/run_offline.py --measure OUT.json
JOPE_W10_BASE_ROOT=... python -B pipelines/final_capture_v1/fault_injection.py --receipt OUT.json
```

## Live facts (read-only JSON reads at about 22:38Z, before the deadline)

- **Progress:** the state was `waiting_for_mandatory_gp_capacity`. The run was at absolute step 17,773 (17,344
  local), with 40 completed epochs; epoch 40 is open. It had seen 6,982 unique contexts over 284,006 presentations.
- **LR controller:** LR is 1e-6, at the floor, after 9 reductions.
- **Training root:** the rolling slots are 17772 and 17773. There is no `result.json` or `failure.json`. The planner and
  outbox `-wal`/`-shm` files are present.
- **Parameter sha:** `ea078456…f9fb` appears in progress, control and the rolling manifest.

No SQLite file was opened.

r2 additions, all read-only, at about 00:55Z on 2026-10-10:
- `systemctl --user show` of the unit: `Result=timeout`, `MainPID=0`, `NRestarts=0`, `ExecMainCode=2`,
  `ExecMainStatus=15`, `ControlGroup=` (empty), ExecStart `.../dl-flywheel-recovery5-20261009/continuous_recovery_cli.py`.
- The execution root's `continuous_runner.py` has sha `10cbf4ea…`, the same bytes as the pinned base
  `prepared/continuous_runner.py`.
- `find`/`ls` stat listings of the campaign root and execution root, as summarized in the r2 section.

No live file content was read, and no SQLite file was opened.

## Open questions (r2)

1. `NRestarts == 0`, `Result == timeout` and the walk are required only for `runtime_limit_stopped_idle`. Should
   `clean_terminal` (r1) also require `NRestarts == 0`?
2. The recovery CLI publishes `status.json`/`fatal.json` in its execution root, outside the campaign root, so the
   capture checks only `<root>/status.json`. A read-only `ls` at ~00:55Z showed neither file in the execution root.
   Should the capture also take that execution root and record both as absent?
3. `ExecMainCode=2` and `ExecMainStatus=15` (SIGTERM) are not part of the closed readback. Should the R17 class
   require them, refusing a SIGKILL-after-TimeoutStopSec end, or only record them?
4. The R17 receipt sha is pinned in source (`C.R17`). A re-issued receipt needs a source change and a re-pin.
5. The walk refuses every symlink, including ones that stay inside the root, which is stricter than asked. The runner's
   own `bytes_used(root.parent)` already refuses any symlink under the campaign root, and the live root has none.
6. Several new guards have downstream backstops. Under their mutants the counterexample is still refused by a later
   check, recorded as `refused_otherwise` in the fault-injection receipt:
   - `no-write-after-in-loop-deadline` and `r17-sealed-receipt` are re-checked by the reader during the capture's own
     self-readback;
   - `r17-closure` by r1's checkpoint, parameter-sha and progress joins;
   - `status-result-absent` (stale `result.json` case) by r1's result join.
7. r1 questions 2–6 below still stand. r1 question 1 is answered by R17 for the new class.

## Open questions (r1)

1. The brief says to require "inactive". A `RuntimeMaxSec` kill reports `failed/timeout`, so I accept
   `ActiveState ∈ {inactive, failed}` with PID 0 and an empty cgroup. Should `failed` be narrowed to
   `Result=timeout`?
2. `ancestry.head` is a lineage head, not a raw `verify()` head. `inherit()` requires
   `committed_batches == len(fingerprints)`, so it has to cover the whole lineage. Do downstream v2 readers accept
   this?
3. v1's last epoch will most likely be open. A fresh journal cannot inherit the partial-epoch exclusion, so v2 epoch 1
   is a full pass. Its shuffle key equals v1's open-epoch key, but the remaining set and the seen set differ. Is that
   intended?
4. `inherit()`'s 16 MiB event caps the lineage at about 243k fingerprints. This needs a design decision before the
   v2→v3 handoff.
5. A stale `result.json` from an earlier segment refuses (HOLD). None exists now.
6. The fixtures never reach a plateau LR reduction, so LR sensitivity is shown by perturbation. The live controller
   has 9 reductions.
