# H135 prospective Amendment A4 — xbabe1 with bounded pressure pauses

Authorization: `~/nudges/rd_astra_t1630_h135a4.md`, 2026-10-31. This later owner
direction authorizes **one** prospective A4 continuation and supersedes A3's
no-fourth-attempt disposition for that continuation only. A3 closeout
[PR #1213](https://github.com/neverhuman/JopeDime/pull/1213) merged first at
`3df2979abf861502f69f3fa70e7871952bbf0c06`; all earlier failures remain preserved.

A3 reproduced all 91 r882 utilities exactly, then the external host-load guard
stopped seed 135 at load1 101.30859375 >86.7. Maximum observed worker threads
was one. No completed fit/prediction receipt or blend result exists; internal
fit progress and unflushed seed read counts remain unknown. A3 numerical custody:
`s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H135/astra-a3/manifest.json`,
VersionId `ZEvQ9kPnLHcT_6WV_1lV3sN0qAA6kowL`, SHA256
`e4cb583932bc3c68ac9049b9924060d17b922b223a9a678af8905dc0f643c717`.

## Host and input identity

Select **xbabe1**, as preferred by the owner when the portable input copy is
below 2 GB. The exact selected DL-Train plus DL-Val closure is 1,315 files,
306,360,743 bytes: 1,310 selected pools, selected.json, cpu_exact_fused.py, the two
H58 feature caches, and the authorized offline r882 checkpoint copy. Every copied
file's size and SHA-256 was checked independently. No source_manifest, sealed,
holdout or meta-test metadata/payload is copied or read. The transfer reported a
refused `/home` directory timestamp; all explicit file contents verified afterward.

The three pinned Micro Python sources and four pretrained assets already on
xbabe1 match xbabe2 byte-for-byte. Python3.12.3 and numpy1.26.4, scipy1.16.3,
scikit-learn1.7.2, torch2.7.0, threadpoolctl3.6.0, psutil7.1.0 and pyarrow20.0.0
match. The unscored constructor preflight passed at one thread, peak RSS
912,961,536 bytes. An earlier preflight correctly refused a cold bytecode path
under jope-data before construction; moving execution custody outside jope-data
resolved it without changing the input audit or reading study data.

Use only the claimed sparse worktree
`/home/ubuntu/JopeDime/.agent/worktrees/rd_astra_h135` and external custody
`/home/ubuntu/jope-rescue-20260921/h135-a4-continuation`. The required lane sweep
removed zero lanes. The existing live flywheel worktree/process and its files
are excluded from this claim. Input copies occupy the explicit, previously
absent H50/H58/H134 paths expected by the unchanged runner; no existing file was
overwritten. Do not mutate live flywheel state, use a GPU or acquire qualification.lock.

## Operational policy

The monitor alone changes; the A3 one-thread launcher remains unchanged.
All scientific runner, H124/H129 recipe, source pins, seed135–137 identities,
sampling135000+s, calibration, metric, weight and statistical gates remain
byte-identical. CPU nice19 and the pre-interpreter thread environment remain.

- Load1 above85% of available CPUs (86.7 on this 102-CPU host), or MemAvailable
  below16GiB: SIGSTOP the owned worker process group, preserving its current fit.
- Resume that same process with SIGCONT only after load1 is **strictly below80**
  and MemAvailable is at least16GiB continuously for60seconds. A boundary crossing
  resets the recovery timer. No replacement process or refit is started.
- Before each stage, wait for the same recovery condition if the host is already
  under pressure. Count and record this waiting interval too.
- Threads above4, RSS at least8GB, root usage at least93.8%, or scratch at least2GB
  remain hard aborts, including while paused. The imported helper's stricter6GiB
  peak-RSS check is unchanged.
- A **four-hour total wall deadline** covers all scientific stages, waits and
  pauses. At expiry mark INCOMPLETE and terminate the owned worker, including a
  stopped worker (queued SIGTERM then SIGCONT; bounded cleanup with SIGKILL if
  necessary). There is no deadline reset between seeds or after pressure.
- Record host, start/end wall time, every pressure reason, pause/wait start/end,
  duration, stage, PID and disposition in monitor.json; record resource samples
  throughout. A pressure pause is not an execution retry.

## Execution and interpretation

Central review/merge precedes scientific execution. Verify the actual merged
source and remote input/asset identities, run the synthetic thread preflight,
refresh the bounded seed-use audit, then launch A4 once. Earlier H135 partial
seed135 work is disclosed and not mislabeled unused; no seed substitution.
First repeat the ordered91-context r882 utility vector and mean1.1377941898164525
within1e-6. Any parity failure stops before fitting/blending. Fit the same three
full DL-Train models, excluding validation families, using the unchanged imported
recipes and unsampled-training-candidate calibration. No outcome-driven changes.

The primary remains BLEND_025. PURE_GBM/BLEND_050 remain report-only. Every original
median, family-bootstrap LCB, 3/3-seed, harm, trimmed-gain and independence gate
remains required. Preserve winner-selection and selector-exposed validation
caveats. A pass is CANDIDATE_REVIEW_ONLY, never serving promotion. DOPE remains
`c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`.

On terminal completion preserve numerical-only evidence under
`experiments/rd_breakthrough_20261030/H135/astra-a4/` with create-only writes,
exact-version GET and member hashes before owned scratch cleanup. Report native
utility, family-cluster uncertainty and Elo anchored at1500; unavailable outcomes
stay null. Close out through central review. No automatic fifth attempt.

## Pre-execution receipts

- `/home/ubuntu/jope-rescue-20260921/h135-host-a4/host-selection.json` SHA256 `66482ec5afce9fe8e482dd420e1551aa090c865d8cf0decbc7bcb389af33c210`.
- `/home/ubuntu/jope-rescue-20260921/h135-host-a4/portable-inputs.json` SHA256 `0ab9bee1b76d4e2556386b933af59254bd6c0d2efd8025c1d2647790a5515fb3`.
- `/home/ubuntu/jope-rescue-20260921/h135-host-a4/remote-input-verification.json` SHA256 `805fb3453cb05266d645559e5d913cddd43dbfd1941035bd7b82a6263cfb8a98`.
- `/home/ubuntu/jope-rescue-20260921/h135-host-a4/remote-receipts/constructor-pass.json` SHA256 `7281e52bcf51d9fe9fcd25226132744c850ead488fe1dda0a3fe6f411ce5e5aa`.
