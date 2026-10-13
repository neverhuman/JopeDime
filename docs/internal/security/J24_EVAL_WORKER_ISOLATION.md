# J24 / G38: evaluator child isolation

Gap J24 (gap map G38) implements the isolation requirements in CLOVER review §12.4: generated-program
evaluation must not hold training-control or custody credentials, must not have network access it does
not need, and must run with bounded CPU, memory, process and file resources.

This change confines the two kinds of original child that the fit authority service spawns:

- the **fit child**: `jope registry-service-fit`, which runs `fit_boundary::execute_service`;
- the **query child**: `jope registry-service-query`, which runs `execute_query_worker`.

The implementation is in [`sandbox.rs`](../../../crates/jope-engine/src/eval_registry/service/sandbox.rs).
It is wired into the two spawn sites, `supervise` in
[`service.rs`](../../../crates/jope-engine/src/eval_registry/service.rs) and `predict` in
[`query_worker.rs`](../../../crates/jope-engine/src/eval_registry/service/query_worker.rs).

This is source and local-test evidence only. It makes no claim about qualified throughput, L6 or
learning.

## Threat model

An evaluated program is data that the child interprets: a genome, or a retained model. Isolation is
defence in depth against two things:

- a defect that lets that data steer the child;
- a child that is compromised in some other way.

The child runs as the same Unix user as every agent on the host. Without confinement it could do
anything that user can do:

- read `~/.aws`, `~/.config/gh/hosts.yml`, `~/.ssh`, or another process's `/proc/<pid>/environ`;
- open the authority's SQLite database and custody files;
- connect to S3, GitHub or the user's D-Bus or systemd manager;
- signal or trace training processes;
- inherit the service's environment, including any `AWS_*` or `GH_TOKEN` variables.

## Two layers

| Layer | Who applies it, and when | Controls |
|---|---|---|
| Spawn | The service, between `fork` and `exec` (`Confinement::enter`). The child cannot skip it. | The environment is cleared to an allowlist: `LD_LIBRARY_PATH`, `LD_PRELOAD`, `RUST_BACKTRACE`, `RUST_LIB_BACKTRACE` and `RUST_MIN_STACK`, plus the child's own `JOPE_SERVICE_*`/`JOPE_QUERY_*`, thread-count and policy variables. Every inherited descriptor except the child's channel is marked close-on-exec. rlimits: address space 8 GiB, unchanged from before; CPU = (wall budget + 1 s) × (1 + all contract thread counts); file size = 2 × `max_descriptor_bytes` for fit and 64 MiB for query (`SIGXFSZ` ignored, so an oversized write fails with `EFBIG`); 256 open files; no core dumps. `no_new_privs` is set. A seccomp filter refuses every non-`AF_UNIX` socket, io_uring, `fork`/`vfork`/non-thread `clone` (`clone3` returns `ENOSYS`, so libc falls back to `clone`), `ptrace`, `process_vm_*`, `pidfd_getfd`, `kcmp`, namespaces, mounts, `bpf`, `perf_event_open`, `userfaultfd`, keyrings and handle-based opens. Other architectures and the x32 ABI are refused. |
| Lockdown | The child's own verified code (`sandbox::lockdown`). The fit child applies it after authenticating the service and before requesting start permission. The query child applies it before reporting readiness and receiving inputs. | **Landlock filesystem allowlist.** Read-only: `/usr`, `/lib*`, `/etc/ld.so.cache`, `/proc`, `/sys/devices/system/cpu`, `/sys/fs/cgroup`, `/dev/urandom`, `/dev/null`, the executable and every mapped file. Writable, with no deletion: only `<authority>/worker-output` for the fit child; nothing for the query child. Every TCP bind and connect is refused (ABI ≥ 4). **A second seccomp filter**, synchronised to all threads, refuses every new socket, `socketpair`, `connect`, `bind`, `listen`, `accept`, `execve` and `execveat`, and signals to any other process. |

Landlock also applies its ptrace scoping. After lockdown the child cannot open another process's
`/proc/<pid>/environ`, `maps` or `mem`, including the service's.

### Why the filesystem allowlist is applied by the child

The fit child verifies the service by hashing its parent's `/proc/<pid>/exe`
(`verified_executable`). Landlock forbids a confined process from reading the `/proc` entries of a
less-confined process. A Landlock domain applied at spawn would therefore break that check. The
child applies the allowlist itself once that check has passed. It does so before any fitting or
prediction, and before its descriptor input is used for numerical work. The fit child has parsed the
program text by this point, to compute its fit key.

### Attestation

The service records how many seccomp filters the spawning thread holds. Before it consumes a fit start
token, and before it sends query inputs, it reads `/proc/<child>/status` and requires all three of:

- `NoNewPrivs: 1`;
- `Seccomp: 2`;
- `Seccomp_filters` equal to the recorded count + 2.

The lockdown applies Landlock before it installs the second filter, and it fails closed. The second
filter is therefore present only if the allowlist step also succeeded. If attestation fails, the claim
stays unavailable, the same terminal treatment as any other readiness failure.

### Threads

On this kernel (Linux 6.8) `landlock_restrict_self` binds only the calling thread. Any thread created
after lockdown inherits the domain. Lockdown therefore refuses a child
holding more threads than the service allows:

- production allows one thread. `jope registry-service-fit` and `registry-service-query` start no
  thread between `main` and lockdown;
- test builds allow two, because libtest runs every fixture child on a second thread. The idle libtest
  runner thread is the only thread in a fixture child that Landlock does not bind.

## Configuration and defaults

`ServiceConfig.sandbox: WorkerSandbox`:

- `require_fs_allowlist` defaults to `false`. With the default, the allowlist is enforced whenever the
  kernel offers Landlock. `xbabe2` (Linux 6.8) reports Landlock ABI 4; other hosts and the CI runner
  containers were not checked. On a kernel without Landlock, the service binds and its children skip only the allowlist step; every
  other control still applies.
- With `true`, `Service::bind` refuses such a kernel.

Making `true` the production default is an owner decision. It belongs in `fit_boundary::serve`, which
is outside this lane. Every other control has no switch: the service refuses to bind if
`/proc/thread-self/status` lacks `Seccomp_filters` (older kernels), or on an architecture other than
x86_64 or aarch64.

## Residual limits

- **No network namespace.** This host restricts unprivileged user namespaces through AppArmor
  (`kernel.apparmor_restrict_unprivileged_userns=1`). Network denial is therefore by system call: seccomp, plus Landlock for TCP.
- **Window before lockdown.** Between `exec` and lockdown the child runs only trusted service code.
  During that window it can still create `AF_UNIX` sockets (it connects to the service socket) and
  read files.
- **Visible `/proc` entries.** After lockdown, other processes' `/proc/<pid>/cmdline` and `stat`
  remain readable, as they are to any user.
- **Not covered here.** Sealed-judge row isolation (G23) and the separation of publication
  credentials from evaluation hosts are separate gaps. This change only ensures that an evaluator
  child neither receives nor can read them.

## Verification

Tests: `crates/jope-engine/src/eval_registry/service/sandbox/tests.rs`, plus
`original_fit_child_is_confined_while_fitting` in `service/tests.rs`.

The probe tests spawn a copy of the test binary through the same `Confinement` spawn path. The
service test runs a real service-owned fit child. Each one asserts the following, before and after
lockdown:

- TCP, UDP6 and netlink sockets, `fork`, io_uring, `process_vm_readv` and `unshare` are refused;
- no environment variable outside the allowlist is present;
- the rlimits are bounded;
- `no_new_privs` and seccomp are on;
- after lockdown, UNIX sockets, `socketpair`, `exec` and signalling the parent are refused;
- after lockdown, reading a planted secret, `authority.json`, `registry.sqlite`, `/etc/passwd` and the
  parent's `/proc/<pid>/environ` is refused, and so is creating files outside `worker-output`;
- attestation fails before lockdown and passes after it.

A child holding more threads than allowed is refused. The Landlock-required flag refuses a kernel
without Landlock.

Mutation checks: removing the spawn filter, the Landlock step or the environment clearing each makes
three of these tests fail.

`jope-parallel` is in the deferred test scope (`deploy/ci/rust-test-scope.sh --deferred`). These tests
therefore run in the full CPU lanes, not in `just required`'s `test-pr`. Run them locally with:

```sh
OMP_NUM_THREADS=1 RAYON_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  cargo test -p jope-parallel --lib --locked -- \
  eval_registry gp::fit_contract_tests fit_boundary --test-threads 16
```
