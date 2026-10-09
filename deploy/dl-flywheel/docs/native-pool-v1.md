> Repository import note (2026-10-09): the launcher, native_build dependency and21 fixture bodies are byte-exact copies of the independently reviewed source. The template is not a launch permit. The actual12-worker attempt was held by the fresh memory guard before its clock/GP dispatch; this source import does not relabel it as a launch. Exact historical reviews are retained under `pipelines/hpc_pool_v1/provenance/`; their absolute paths are custody references, not test dependencies. All prior587 test identities remain wired; the new pool suite adds21. No earlier suite was rerun for this import.

# Additional native CPU pool v1

Local source proposal only. No remote read, staging, queue claim, native GP execution, S3 PUT, service control or Git/PR write was performed. Root owns deployment after independent review. The RL attachment was read; this package does not implement actor gradients, policy changes or RL files.

`operations_pool.py` derives from frozen native successor revision2 `operations_launch.py`. `native_build.py` is a byte-exact dependency copy. Numeric worker/evaluator/parity code is not edited or repackaged here: the launcher stages all34 worker files byte-for-byte from the reviewed parent's source roster. The caller chooses exactly8,12 or24 **additional** workers. The template requests24; the helper never silently reduces it.

## Safety and ownership

- Current packet2 and its original clock remain unchanged. New workers get new output roots/units and a new explicitly reviewed1,800second window that fits `1791592146.380285`.
- All workers use the same exact existing permanent registry: `/home/ubuntu/JopeDime/.agent/proposals/dl-flywheel-native-global-20261009-v1/registry.sqlite`. It must already exist. Device/inode, canonical config hash, source identities, census and scope are read-only verified. No registry creation, reset, replacement, migration or source-list extension occurs in this helper.
- Parent worker0 is the exact config template. The four parent plan/config identities, input/build identities, guard and all34 worker source hashes/lengths must agree. Old unknown/failed/completed states remain entirely under the unchanged dispatcher/guard. Parent workers are compatible active writers, not legacy units to stop.
- `stage` is create-only and does not start a clock. `freeze` reads local resources first, then fixes its clock. `launch` writes a one-use intent before subprocesses. A failed or uncertain partial dispatch cannot be retried through the same plan.
- Every worker is CPU1, RAM1GiB, swap0, TasksMax32, Nice15, no GPU, BLAS/OpenMP1, Restart=no, control-group kill, maximum128 emissions. The actual native child stops starting work when a600second lease no longer fits; the launcher leaves more than610seconds before any worker launch.
- Up to four CPU-only preflights execute concurrently. All must pass before any actual native worker starts. Physical source/build/context/registry verification is unchanged. New actual scorer starts are sequential and receipt-bound; no automatic retries.
- The launcher physically executes and validates the exact pinned interpreter as Python3.12 before preflights and again before dispatch. The working parent pin is `/usr/bin/python3.12`, SHA `e50d468e8b0adfb05733f5b87b3cff34829c4a8c1aea50c865aa8bdfe4bb150f`. A filename alone does not satisfy this check.

## Resource input and required fresh reads

`manifest.template.json` is deliberately invalid until root fills its physical fields. Do not interpret it as a frozen launch config.

1. Read parent packet2's exact `plan.json` and its referenced `stage.json`; use the latter's actual `manifest` exactref for `parent_manifest`. The template's parent plan SHA is `6c9ba04a65e1eacf420fefeab3f7b8e0c88684c76f0804cba1effb93f4a5d522`.
2. Read only the permanent registry inode/device and the SHA256 of its single exact `config.body` blob. Supply `registry_identity`. Do not hash the changing full SQLite file as a static identity.
3. Supply the current learner's exact systemd unit, `/proc/<MainPID>/cmdline` array, actual declared CPU cap and memory cap (at least8GiB reservation). The learner retains CPU800% exactly and at least8GiB. `other_owned` accepts exact32MiB–32GiB caps and CPU1%–3200%, so small publishers/archivers are included honestly rather than rounded to fictitious1GiB/100% limits. Add every other concurrently authorized owned allocation under `other_allocations`; the four parent scorer units are automatically derived and must not be duplicated. Foreign jobs are not controlled.
4. Select8,12 or24 before freezing. The gate reads actual active unit command lines, cgroup memory/CPU/swap limits and full-cap reservations. A physically closed learner still retains its full reservation; physically closed parent scorers retain their histories but no future memory reservation. A claimed inactive unit must have PID0 and empty cgroups.

Fresh gate: `MemAvailable >=24GiB foreign reserve + sum(full caps for active parent workers and all declared other owned allocations) + worker_count*1GiB`. Full caps intentionally avoid racing a cgroup current-memory sample against MemAvailable and double-crediting freed memory. With four active1GiB parents and an8GiB learner this requires44GiB for8 additional workers,48GiB for12, or60GiB for24. After physical closure of all four parent workers, the corresponding thresholds are40/44/56GiB. No parent service is stopped to obtain that capacity. Require40GiB free output disk and aggregate logical-CPU capacity for active declared CPU quotas plus new workers. No inference about current headroom is made from dated local snapshots. Incomplete owned-allocation inventory is not an honest resource pass; root must supply the full current inventory, including newly authorized RL or encoding jobs where applicable.

## Exact operation sequence

Copy these two runtime files into a new create-only operations directory and verify their final review hashes. Stage the existing source/input paths required by the manifest only when missing through separately approved exact copies; do not rewrite parent source or receipts.

```text
rtk proxy /usr/bin/python3.12 -B <owned>/operations_pool.py --input <manifest.json> --input-sha <SHA>
rtk proxy /usr/bin/python3.12 -B <owned>/operations_pool.py --input <manifest.json> --input-sha <SHA> --stage
rtk proxy /usr/bin/python3.12 -B <owned>/operations_pool.py --input <stage.json> --input-sha <SHA> --freeze
rtk proxy /usr/bin/python3.12 -B <owned>/operations_pool.py --input <plan.json> --input-sha <SHA> --launch
```

Freeze and launch consecutively once the staged packet is reviewed. The final call uses the unchanged full historical physical-closure checks plus compatible active-parent allocation checks. The registry inode/config is checked again after preflights and immediately before every native dispatch. Resource sampling occurs after slow historical closure calls; each dispatch requires a new observation no more than five seconds old. The fixed external runtime is recomputed after these final gates. It never stops any service. Report the generated exact worker commands and full deadline immediately. Launch success remains `native_completion_verified=false`; actual progress/readback is a separate root task.

## Local qualification

From the repository root:

```text
python3 -B deploy/dl-flywheel/run_native_pool_offline.py --output /tmp/jope-native-pool-proof-UNIQUE
```

New fixtures exercise8/12/24 cardinality,24-worker parallel preflight limit/join, exact source/guard/registry inode identity, dry/source database immutability, preservation of an unknown claim and parent deadline, memory/CPU reserves and live-command drift, actual Python version+binary identity, altered preflight census, one-use launch and partial-dispatch interruption. No prior historical suites are rerun and no systemd/SSH/S3/GP commands execute in fixtures.

`pipelines/hpc_pool_v1/provenance/operations-pool-diff.json` retains the exact original operational delta against revision2 as base64 with decoded SHA256 and byte count. The original unified diff contains valid single-space blank context lines; encoding preserves them without introducing trailing-whitespace failures in the source commit. Decode only for inspection and check both declared length and SHA256. The historical `VALIDATION.json` and `PACKET.json` in that directory bind the original proposal; repository `SOURCE_MANIFEST.json` binds the packaged files. Root should obtain independent review of those final bytes before any remote staging or source integration. A new runtime finding requires a new immutable attempt namespace; never reset an old attempt to improve the appearance of progress.

### Shared-host allocation hold (2026-10-09)

Packet8 remains **HELD** pending an owner-bounded shared-host allocation. The CI peer has an idle full-run listener with a48GiB cap plus two QEMU services with9GiB caps each; no shared host lease has been established. These future allocations can expand despite a favorable instantaneous `MemAvailable` sample. The42.625GiB arithmetic for the declared local pool/learner/services alone is therefore insufficient to authorize launch. The fixed24GiB foreign reserve does not certify coverage of those larger peer caps. Do not treat idle peers as released capacity, reduce reserves, stop foreign jobs, or start a clock to force progress. The owner must establish the bounded allocation and a separately reviewed current packet/resource closure before dispatch. The frozen `operations_pool.py` source `d9894721…` is unchanged and does not itself create that cross-owner lease. Historical input reviews remain preserved as bounded local schema checks, not launch permission.
