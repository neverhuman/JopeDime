# Single-job resource guard candidate

`guarded_job.py` is an offline policy/state-machine candidate. It has no host
adapter, controller process, OS writes, signals or launcher. Every plan records
`live_execution_qualified=false` and `actions_executed=false`. Twenty-one focused
tests passed on 2026-10-20; their raw log is retained outside Git. This closes a
policy implementation step, not the live 70% resource admission gate.

The existing research owner alone wrote this packet under SC-DEC-CPU70-0916.
The Admiral retains resource coordination; Root01a155bf retains R4. The shared
fleetcap slice, global services, existing workloads and production defaults are
outside its scope. Do not thaw the shared xbabe2 slice. Do not implement a
live adapter against an ancestor containing protected work.

## Candidate behavior

`CpuPoint` and `cpu_interval` derive host utilization from paired aggregate CPU
counters and owned usage over the same monotonic interval. Actual online CPU
IDs determine the denominator. Online changes, reset/stalled counters, missing
points, inconsistent host/owned measurements and stale/future samples stop the
candidate. The nominal interval is one second, with explicit 0.9–1.25 second
measurement tolerance and at most 0.25 second delivery age. These tolerances
are proposed semantics requiring owner acceptance and measured scheduling proof.
Each next interval must start at the preceding complete point; replaying one
good interval cannot manufacture a safe streak.

`Leaf` binds a named lease SHA256, exact cgroup path/device/inode and nonempty
PID/start-time roster. `ScopeFacts` requires exact unchanged membership, no
child cgroups, descendant confinement, qualified delegation, control/readback
proof and a qualified live parent watchdog. A changed/unknown identity produces
no write plan: its replacement might belong to another owner. The independent
watchdog must contain the original job. Freeze ownership must also match the
lease; foreign/unknown freezes are never released. These supplied facts are
adapter assertions, not cryptographic proof or authority created by this module.

`Guard.observe` uses a 60% operating target, holds at 65%, and records/latches
STOP at an observed 70% breach. The remaining owned quota is the smaller of
eight cores and the 60% target less conservative foreign-load accounting. The
foreign bound retains one core for host/cgroup counter alignment uncertainty.
Less than 0.25 core of remaining budget holds the job. Three consecutive safe
intervals precede a run candidate. RAM, swap, temperature, storage, deadline,
scope, watchdog or measurement faults latch STOP. Load holds reset the safe
streak. A stopped instance cannot silently reset its lease or deadline.

Run plans order quota setting and successful readback before releasing only an
attested own freeze. Active own jobs need quota readback without a thaw action.
Hold/stop plans request only own-leaf freeze and readback. They never contain
ancestor writes, `cgroup.kill` or process-group signals. An adapter must bind
each operation to the pinned leaf and revalidate membership and deadline before
each action; the plan itself does not enforce any action.

Peak utilization, breach sample numbers, exact leaf identity, effective proposed
quota and reason remain in hashed plans. Preserve every observation and refused
operation. A one-second sampled policy cannot promise control over instantaneous
foreign-load spikes. If foreign work crosses 70% independently, pause only our
work and retain/report the actual breach. No sampled PASS replaces continuous
executor/watchdog qualification.

## Next executable owner packet

Read `../NEXT_WAVE.md`, `README.md`, `guarded_job.py`, `test_guarded_job.py`, the
current CPU70 audit and Admiral disposition before another claim. Recheck current
main, PR queue, path owners and original budget. The initial local envelope ends
2026-10-20T10:59:46Z and is not renewed by this source publication.

1. Existing resource owner supplies the exact delegated leaf creation route,
   effective cgroup controllers/limits, all protected-membership exclusions and
   ownership/readback/watchdog mechanism. No privileged/global change is included.
   The guard's pure fields do not establish this grant. Missing delegation stops
   dependent work, with its responsible owner named.
2. This research owner may then claim only `next/linux_leaf_adapter.py` and
   `next/test_linux_leaf_adapter.py`. Implement `read_cpu_point`,
   `read_scope_facts`, `read_health`, `apply_plan` and `run_owned` against that
   exact qualified route. Keep source/policy/task/lease hashes in every receipt.
   Refuse symlink/path replacement and pin directory/file identity, effective
   CPU roster, descendant confinement and monotonic accounting. An idle scope or
   a guessed filename does not prove ownership. Any PID roster change requires
   an explicit owner-bound update while paused, never blanket adoption.
3. The adapter must preserve a start barrier and independent watchdog. The
   payload cannot execute before frozen membership, effective quota, guards and
   watchdog readback pass. Lost controller, stale measurement, failed quota or
   freeze/readback, membership escape and adapter crash must leave only this job
   contained. Readback failure cannot advance to release. A call to freeze is
   not a confirmed frozen state. Bound timing and retain unresolved failures.
4. Add fake-filesystem/backend fixtures for each operation's ordering, replaced
   inode, changed PID start, foreign member, unowned freeze, stale sample,
   write/readback failure and watcher loss. Then the existing resource owner
   qualifies the smallest named own-only live failure exercise under a bounded
   lease. Record exact measured intervals/peaks and behavior without touching
   protected jobs. No training is admitted by fixture-only success.
5. Only after that proof binds to the exact adapter/policy/command may the
   `native-feedback-fresh-gp-v1` source/runtime/data permit proceed. Current
   blockers remain exact-version raw TRAIN data/rights/lineage and measured
   baseline identity, native teaching bridge, CUDA runtime qualification, complete
   sequence microbatch/VRAM reserve and preregistered matched arms/costs. Existing
   native checkpoints and Kumo encodings alone do not resolve raw-data GP rights.

Run this focused source validation from the capsule directory:

```sh
rtk proxy env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  OPENBLAS_NUM_THREADS=2 python3 -m unittest discover -s next -p test_guarded_job.py -v
```

The 21 tests cover quota accounting and safe streaks, load hold/breach latching,
foreign surges, actual 32/128 CPU denominators, counter resets/hotplug/staleness,
replayed intervals, unsafe leaf/PID changes, protected freezes, required guards,
watchdog/readback prerequisites and original deadline bounds. They execute no
payload, cgroup writes or GPU work. Independent exact-head review and CI remain
pending the existing PR387 queue. No new PR, helper, checkout or native runtime
was created for this packet.
