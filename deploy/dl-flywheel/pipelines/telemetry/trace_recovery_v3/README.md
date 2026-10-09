# Cold telemetry recovery3 — worker handoff

Status: local implementation and 34 new synthetic fixtures passed. Independent source review and actual host qualification remain required. No clock has been frozen, service started, or S3 operation performed by this packet.

## Objective and ownership

Continue custody of already sealed xbabe3 HPC telemetry after recovery2 ended naturally. Start at part607; never replay part153 or any earlier attempted object. Keep the original event writer and every prior archive output immutable. This does not train a model, execute GP, encode data, read training payloads, or claim fresh cloud verification for historical objects.

Only new source/output namespace: `/home/ubuntu/JopeDime/.agent/proposals/dl-flywheel-trace-recovery-20261009-v3`. Only new service: `jopedime-hpc-archive-recovery-20261009-v3` and its one-use preflight unit. Input logs remain `/home/ubuntu/jopedime_probe_exp/ofm-20261008/dl_final/hpc_trace_20261009_v1/logs`. The existing `logs/archive-v1` directory inode is the shared exclusive flock; acquiring this read-only directory lock does not change its files. No foreign unit may be stopped. Previous units must all be physically inactive/failed with PID0 and empty cgroup before and after cold verification.

## Exact prior custody and retained limits

The terminal evidence was read-only captured and verified at 2026-10-09 16:40:41 UTC. It covers original0–152, prior1 154–528, and prior2 529–606. Part153 remains a permanent unknown/absent historical attempt, including its31 event gap. Its earlier HEAD404 is retained; no automatic PUT retry or new reconciliation is authorized here.

| Counter | Value |
|---|---:|
| Verified segments |606|
| Verified objects |1212|
| Charged global PUT attempts |1213 =307 original +750 prior1 +156 prior2|
| New first part |607|
| Remaining PUT attempts |3787|
| Maximum new complete pairs |1893 (parts607–2499)|
| Original maximum PUT attempts |5000|
| Original segment index ceiling |2500 exclusive|
| Original global deadline |1791592146.380285|

One odd PUT allowance remains unused when1893 pairs complete; this is not an invitation to upload a partial extra segment. The new window is one hour and must fit wholly inside the original global deadline. Both wall and monotonic deadlines constrain work. Previous windows and budgets are never changed.

Historical source copies are exact: base `c8a13033c6a93bb97d14d3ccb4de0d2fae01be0928a75f654cfd79f32d835c69`; recovery2 `fe000e7a0bed06f4b339997eaea745a1b0737430da5543f8f61c4d6ce553df22`. These files are provenance and imported verification/publication primitives. Do not run their old main functions again.

## Read first

1. `SOURCE_PACKET.json`, `validation.json`, and independent review once supplied.
2. `terminal-proof.json`, `custody-inputs.json`, `config.template.json`.
3. `trace_recovery_v3.py`: `cold_prefix`, `extend_v2`, `preflight`, `Archive`, `main`.
4. Exact previous `trace_recovery_v2.py` and `trace_archive_v1_c8a13033.py`.
5. `prepare_config.py`, `launch_v3.py`, and `test_trace_recovery_v3.py`.

The template's absolute deployment paths and custody captures are operational evidence, not portable defaults. For repository integration include code, docs, template, validation, source-delta and custody descriptors; exclude multi-MB live captures. Synthetic tests require no live capture, boto3, remote host, training data or network.

## What cold qualification checks

The unchanged v2 verifier physically rereads all original/prior1 pins and raw sealed chain. New code adds every original attempted key to the historical replay guard, then verifies the exact full prior2 inventory: source/config/window/execution marker, previous cold prefix, all attempts/replies/exact-Version readbacks, receipt numbering, cumulative counters, current physical SHA/byte pins, and continuation of the local raw chain through606. Both old owned directories must exactly match their frozen captured file inventories; an added attempt, changed source, missing receipt, unknown nested directory or symlink refuses qualification.

Unexpected partial prior2 tails remain charged and held by the generic layer verifier. This production constructor additionally requires the exact reviewed607/1213/3787 terminal tuple; any changed history stops the run. It never silently advances or recomputes a smaller budget. The first cold verification repeats under the global archive lock immediately before new execution custody is created.

Publication inherits unchanged create-only `IfNoneMatch='*'` and exact VersionId GET/body/hash/length verification. Before every PUT, the attempt is fsynced to a create-only file. Readback errors preserve attempts/replies and stop without retry. An unexpected existing fresh object can only be HEAD/exact-Version GET verified via the old reviewed collision path; past attempted keys are rejected before any PUT. Fresh segment receipts use `jope.hpc-host-archive-recovery-receipt/v3`, link both captures, and publish only after both exact object readbacks and a final deadline check.

## Root execution sequence — not performed here

1. Independently review the frozen packet. The review JSON must contain `state: passed` and the exact `template_sha256`. Root owns coordination and current resource reservation; apparent host idleness is insufficient.
2. Check the old three units and new namespace. Preserve all historical roots. Stage the six configured input files plus `launch_v3.py` create-only, verifying every source/capture SHA and byte count. No existing path may be replaced. Pin the launcher separately using `SOURCE_PACKET.json`.
3. Freeze once in the local packet only after review: `rtk proxy python3 -B prepare_config.py --template-sha <exact-template-sha> --review <review.json> --review-sha <exact-review-sha>`. This creates `config.frozen.json` with one new one-hour window and refuses a second freeze or a window extending past the old global deadline.
4. Stage those exact frozen bytes as new remote `config.json`, then invoke the pinned launcher on xbabe3: `rtk proxy python3 -B /home/ubuntu/JopeDime/.agent/proposals/dl-flywheel-trace-recovery-20261009-v3/launch_v3.py --config-sha <frozen-sha>`.
5. The launcher checks all sources/inputs, empty one-use namespace, closed old units, runs an isolated <=180s no-S3 preflight, checks its exact607/1213 counts and measured timing, then refreshes physical state and resource guards. Only a passing preflight may launch the one-use service with the original frozen remaining time.

Envelope: CPUQuota100%, MemoryMax512MiB, MemorySwapMax0, TasksMax24, Nice19, CUDA empty, BLAS/OMP1, Restart=no, KillMode=control-group. Launch guard retains MemAvailable>=26GiB and load1<100; root must also account for concurrent owned/foreign reserved caps. The worker checks its actual CPU/memory/swap/noGPU cgroup before work. It starts no new pair in the final60seconds.

No automatic rerun on any refusal, timeout, uncertain PUT, missing custody, physical-state mismatch or quota issue. Preserve new failure receipts and ask root to define a separate bounded reconciliation if necessary. A launch timeout never authorizes another launch. Source modifications require a new reviewed version; do not patch this frozen packet in place.

## Validation and interpretation

Command executed locally: `rtk proxy python3 -B -m unittest discover -s work/heartbeat-20261009-1633/trace-recovery3 -p test_trace_recovery_v3.py -q`.

Only34 new tests ran. They cover78-segment prior2 closure, unknown/missing/tampered attempts and receipts, complete three-layer key union, changed inventories, historical replay, pair budgets, monotonic expiry, exact fake-S3 version readback, one-use constructor, global budget/clock retention, and mocked preflight/launcher gates. `ColdJoin` deliberately injects the already-verified prior1 return to isolate the new joining logic; it does not claim to rerun the old verification suite. Actual full raw chain and old verifier execution remain host preflight gates. Test subprocesses and S3 clients are synthetic; no runtime qualification is claimed.

Portable runner must isolate imports for `trace_recovery_v3`, `prepare_config`, `launch_v3` and the exact v2/base modules in a temporary directory. Fixture wall time1791564000 is declared only in tests/runner; production GLOBAL stays unchanged. Keep all34 fixture identities in `validation.json`. The encoded test-output and source-delta JSON retain exact text bytes/hash without tracked `.log`/`.diff` hygiene exceptions.

## Logging, success and final handoff

Keep `preflight-intent/reply`, `launch-intent/receipt`, `execution-started`, `cold-prefix`, `put-attempt-*`, `put-reply-*`, `unknown-put-*`, `verified-object-*`, `receipts/*`, `stage-timing.jsonl`, `progress.json`, and any `fatal.json` in the new root. Never prune incomplete attempts. Every timing event retains UTC nanoseconds, monotonic duration, own/child user/system CPU deltas, config/host/stage and measured bytes; GPU timing is explicitly absent.

Success is a physically closed service with final status and receipts reconciled to all charged attempts, not merely a zero exit. Final handoff must report new/cumulative segments, verified objects, attempted PUT/GET/HEAD counts, unknowns, both deadline pins, exact source/config hashes and preserved gap153. Existing historical object versions are evidenced by saved verified readbacks; current cloud availability is not reasserted without an actual new read.
