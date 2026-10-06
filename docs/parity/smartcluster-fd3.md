# SmartCluster participant descriptor fixtures

The participant fixture binds `SCQ_CONTROL_FD` to the descriptor it actually owns.
Its outside-job helper duplicates a borrowed socket or regular file with
`F_DUPFD_CLOEXEC`, keeps that duplicate alive through child completion, and clears
close-on-exec only on the child's duplicate. It never overwrites ambient descriptor
3 or changes the caller's descriptor flags. The absent-channel fixture claims
`i32::MAX`, an unallocatable Linux descriptor, and checks `EBADF` before exec.
This is an absence test; it does not claim a particular earlier close history.

On unchanged source `aa48fac53d13c206ae42ea699b36c1169c7b2052`, running the
participant identity test beneath ordinary util-linux `flock` failed because the
wrapper's open regular lock file occupied descriptor 3. Production correctly
refused that file at the control socket's peer-credential check; the test expected
an absent descriptor instead. The same frozen test with `flock --close` passed.
A separate opener confirmed that `--close` leaves the utility parent's exclusive
lock held until its command exits. These observations establish a fixture
precondition defect, not a job-authentication bypass.

The additional inherited-lock fixture passes an actual locked regular file to the
participant, checks the non-socket refusal before any admission or result, and
checks contention with independently opened files before and after the child.
It verifies unchanged parent descriptor flags and successful acquisition after
the parent drops its File. Duplicated file descriptions are not valid contention
controls because they share the original lock.

All original participant checks remain: missing job environment, absent control,
impostor socket peer, genuine managed job reaching the coordinator connection,
wrong dispatch/plan/tenant/service/arguments, stale attempts/fences, non-live jobs,
wrong job response and mismatched pinned daemon. The production verifier,
SmartCluster daemon and runner locks are unchanged.

## Reproduction and verification

Build the real daemon and the test-only host executable in the checkout's reused
target. No GPU inventory, training or network service outside loopback is required.

```sh
export CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only
cargo build --locked -p jain-smartcluster-daemon --bin scqd \
  -p jope-parallel --bins --examples
flock /path/to/owned-evidence/qualification.lock cargo nextest run --locked \
  --profile pr -p jope-parallel --test bounded_smartcluster --test-threads 1
flock --close /path/to/owned-evidence/qualification.lock cargo nextest run \
  --locked --profile pr -p jope-parallel --test bounded_smartcluster --test-threads 1
```

The original PR watchdog remains 240 seconds. Neither wrapper condition is a
retry that replaces a failure. Preserve both pre-fix outcomes, all post-fix
results, the source and binary hashes, exact commands, durations, inventories
and exclusions. Local evidence is kept outside Git under
`/home/ubuntu/jope-evidence/smartcluster-fd3-20261006/`.

These are CPU fixtures with three ranks under one boot on loopback. They do not
qualify three physical hosts, numerical training, GPU execution or deployment.
Hosted CI, independent exact-head approval and owner merge remain separate gates.
