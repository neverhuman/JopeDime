# Self-hosted runner runbook

`.github/workflows/ci.yml`, `main-guard.yml` and `weekly.yml` run on a self-hosted GitHub Actions
runner. This runbook says how to provision one, what each lane is allowed to touch, and what goes
wrong in practice.

It exists because the hosted lanes stopped: `veox-ai` is on the free org plan, its monthly Actions
minutes ran out on 2026-09-21 at 12:01Z, and every job after that failed in 2–4 s with
`runner_id: 0`. Nothing could merge, because a required context that cannot *start* blocks a merge
exactly as much as one that fails.

## Shape

One container per job, and the container exits when the job ends. The runner is registered with a
**JIT config** from `generate-jitconfig`: single-use, self-deregistering, never `config.sh`-
registered, no credential in the image. A systemd timer starts a listener; if GitHub has no queued
job it idles until `RuntimeMaxSec` stops it and the timer starts another.

| Unit | Role |
| --- | --- |
| `jope-runner@<lane>.service` / `.timer` | one listener per lane (`ci`, `ci2`, `ci3`, `ci4`, `gpu`, `kernel`) |
| `jope-runner@.service.d/runtime-max.conf` | raises `RuntimeMaxSec` to 5400 — **see "Idle time is job time"** |
| `jope-runner-prune.service` / `.timer` | reclaims docker and volume space |
| `jope-docker-proxy.service` | the socket proxy the `kernel` lane uses (containers and images only) |

Lane privileges, from `deploy/runner/run.sh`:

| Lane | GPU | docker |
| --- | --- | --- |
| `ci`, `ci2`, `ci3`, `ci4` | none | none — the least privilege of the lanes |
| `gpu` | `--gpus $JOPE_RUNNER_GPU` | none |
| `kernel` | none | socket **proxy** only |

Every lane runs `--user 1001`, `--cap-drop ALL`, `--security-opt no-new-privileges`, `--read-only`
with explicit tmpfs and named volumes.

## Provision a host

```sh
sudo deploy/runner/provision.sh            # volumes, host files, units. Enables nothing.
sudo tee /etc/jope-runner/github-pat       # fine-grained PAT, Administration: write, this repo only
sudo systemctl enable --now jope-runner@ci.timer jope-runner-prune.timer
```

`provision.sh` installs the scripts to `/opt/jope-runner`, the units and the drop-in to
`/etc/systemd/system`, and writes `/var/lib/jope-runner/image.json` recording the image id and the
digest of everything it deployed. Build the image first:

```sh
docker build -f deploy/docker/runner.Dockerfile -t jope-runner:current .
```

**Re-run `provision.sh` after every image rebuild.** The image id is recorded there, and CI
compares the running image against it — a rebuild that provisioning has not seen fails the
provenance step by design.

## Provenance: the lane proves what it is running

`.github/actions/runner-provenance` recomputes, from the checkout, the digest of every file in
`JOPE_RUNNER_DEPLOYED` and compares it with the digest `run.sh` injected; it also hashes the
image's entrypoint from inside the container and compares the running image id with the recorded
one.

This exists because reading a committed script proves nothing about what the host executed. Two
defects shipped in the same pull request that added these lanes — a `work_folder` whose `$lane`
never expanded, and a `cargo-deny` invocation that was a parse error — and both were visible in
the committed files while surviving review. If the step fails, the host is out of date: re-deploy
and re-run, do not weaken the check.

It does **not** bind the rest of the image's contents.

## Things that actually go wrong

**Idle time is job time.** `RuntimeMaxSec` counts from service start, and a listener spends most
of its life idle. At the upstream 1800 s a listener that waited 20 minutes and then picked up a
13-minute job was SIGTERM'd mid-job, surfacing as a run marked failed with steps that have no
conclusion. The drop-in budgets both phases.

**One workspace per lane, not one per host.** `work_folder` must be **absolute** — `entrypoint.sh`
copies the runner to `/home/runner/runner`, so a relative `work` resolves inside the 4 GB
`/home/runner` tmpfs — and **per-lane**, because concurrent lanes sharing one root delete each
other's checkouts (`Unable to create .git/index.lock: File exists`, `Unable to clean or reset the
repository`).

**A refused start is a queued job, not a red X.** `disk-guard.sh` refuses to start a listener below
`JOPE_RUNNER_MIN_FREE_GB`. Nothing then picks the job up, so it sits queued rather than failing
visibly. Check `systemctl status jope-runner@ci.service` and free space before assuming GitHub is
at fault.

**`jope-runner-prune.timer` is not enabled by provisioning** and does nothing until you enable it.
Disk fills quietly until it crosses the guard, at which point jobs stop being picked up — see
above for how that presents.

**The image needs more than the CUDA base ships.** `cmake` (for `aws-lc-sys` via `rustls`),
`libsqlite3-dev` (or `rust-lld: unable to find library -lsqlite3`), and `gh` (or `main-guard`
fails on every push to main) are all required and all easy to omit.

**`/home/runner` must be mounted `exec`.** A tmpfs defaults to `noexec`, and the runner copies
itself there, so it fails with `Permission denied` on `run.sh`.

## Do not

Do not start the `actions-runner*` fleet on any host. Those units belong to a different repository
and are not part of this system.
