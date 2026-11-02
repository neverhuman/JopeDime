# H209 attempt 1 closeout

H209 is **INCOMPLETE with no verdict**. The registered hypothesis was not tested:
zero optimizer updates and zero trained heads occurred. Native baseline parity
passed, then resource-log reentry terminated the worker during TRAIN native-cost
calculation. No B selection lock or A endpoint scoring was reached.

## Admission and execution

The 06:34:56 UTC admission was refused because user-visible disk usage was
94.0008403%, above 93.8%. It created no attempt claim and consumed no attempt.
After steward work, the unchanged merged supervisor admitted attempt 1 at
07:02:17 UTC on xbabe2 RTX3090 UUID
`GPU-44c63d19-492a-ab46-cc6a-56e12a2ef51b`:

- Disk 93.7489326200%, free 491,094,011,904 bytes; bound 93.8%.
- Load1 39.8027 / 102 effective CPUs; bound 86.7. CPU utilization 17.8%.
- Available RAM 79,872,274,432 bytes; required 16 GiB.
- No existing GPU consumer; qualification flock held by supervisor PID 1402579.
- Worker PID 1404592; measured numerical pool bound one thread.
- A create-only claim fixed the deadline at 08:02:17.460 UTC and one attempt.

The worker reproduced all 91 ordered native float64 utilities and mean
1.1377941898164525 before training. Closeout independently reread the retained
baseline vector and compared it exactly with the pinned reference. No model
was rerun for this verification.

The retained 151 worker and 24 supervisor resource samples show max disk
93.7513130058%, max load1 39.8027, max CPU 28.7%, max RSS 1,409,703,936 bytes,
and minimum available RAM 79,845,183,488 bytes. These are sampled observations,
not proof of every instant between samples. The exception, not a resource-limit
violation, caused the stop. Five OS threads appeared after CUDA initialization;
that is diagnostic under the registered numerical-thread policy, not a breach.

## Failure and interpretation

Exact executed source: `d609dab46dc725925008e750e0813bfe40b44ab4`.
The stack is `runner.py:451 -> check():249 -> check():249`: `SIGALRM` called
the same resource-check function while its buffered writer was active.
The second write raised the retained reentrant-call RuntimeError. The worker
recorded failure after 44.3740273409 seconds with updates=0; supervisor exit code
1 was recorded at 07:03:07.701278 UTC, 50.2410387993 seconds after its claim.

An isolated CPU reproduction extracted the exact `check` AST from this source
and raised SIGALRM from a fake raw writer during the buffered flush. It reproduced
the same reentrant-call exception without loading data, a model or CUDA.
Receipt: `/home/ubuntu/jope-rescue-20260921/h209-astra/closeout-20261102T0703Z/logging-reentry-repro.json`.
This is a diagnostic fixture, not another scientific attempt.

The exact number of TRAIN native-cost contexts completed was not persisted and
remains unknown. No complete native-cost archive, fitted head, B lock or A panel
exists. The final full-model/DOPE state-hash verification after all TRAIN native
costs was not reached; native baseline parity does not substitute for it. Adoption, utility gain, Elo and U4 remain null. The observed baseline
value is not a trained-head result. Do not label this NO_ADVANCE or infer zero
adoption from the absence of measurements.

The supervisor and worker have exited, the GPU returned to 0%/1 MiB with no
compute processes, and a fresh nonblocking qualification-lock probe succeeded.
The attempt claim remains intact; there was no retry, host deviation, deletion
or budget reset. Python, preregistration, folds and provenance remain unchanged.
A future run needs prospective approval/preregistration after repairing signal
reentry (for example, keep signal handlers free of buffered I/O and retain
independent supervisor enforcement) and adding a meaningful regression test.

## Custody

Create-only objects, each recovered with exact VersionId and SHA256:

`s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261101/H209/astra-t2236/closeout-20261102T0703Z/manifest.json`

- Manifest VersionId: `lmJmF39su9zn125Sz8HGbhKhsuU8bcwW`
- Manifest SHA256: `da4e78a2a0e92ca68deaaf972a888216b8ea9aaaee0f4736a3eaaf17afb33e54`
- `runtime-evidence.tar.gz`: all 16 members recovered and individually rehashed,
  including baseline, raw traceback, worker failure and both resource logs.
- `executed-source.tar.gz`: exact H209 source plus the three pinned parent modules.
- `metrics.json`: machine-readable INCOMPLETE state and resource observations.
- Existing worker launch/failure S3 objects were also independently re-read at
  their recorded versions and rehashed.

[manifest.json](manifest.json) carries object version IDs and hashes. Small local
runtime evidence remains under `/home/ubuntu/jope-rescue-20260921/h209-astra/`.
The original source branch and execution manifest are retained. The original
13-fixture/source-review passes are valid for their stated scope but did not
cover asynchronous buffered logging; no broader qualification is inferred.
