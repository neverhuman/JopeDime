# Runner runbook — the xbabe2 self-hosted lanes

What the owner does, once, to bring the `xbabe2-gpu` and `xbabe2-kernel` runners up; what the
tree does on every job; and how to recover. Nothing in the repository registers a runner or holds
a credential: registration is a per-job JIT config requested with a PAT that lives only in a
root-0600 file on the host. Never use `config.sh`, never start the retired `actions-runner*`
GitHub fleet, never mount the PAT file.

## Files

| Path | Purpose |
|---|---|
| `deploy/docker/runner.Dockerfile` | the runner image (CUDA 12.8 devel, actions-runner 2.337.0, rust 1.97.1 from `rust-toolchain.toml`, nextest/deny/audit/sbom/sweep, g++-13, clang-18, node 26.1.0, python 3.12, uv, just, gitleaks, actionlint, syft, AWS CLI v2; every download sha256-pinned; uid 1001; no sudo) |
| `deploy/docker/kernel-builder.Dockerfile` | ubuntu 22.04 (glibc 2.35) + rustc 1.95.0 for the kernel-v1 lane |
| `deploy/runner/provision.sh` | root, once: named volumes, image build, SDK volume from `deploy/receipts/libtorch.json`, units installed |
| `deploy/runner/run.sh <gpu\|kernel>` | one job: disk guard → JIT config → hardened `docker run` |
| `deploy/runner/jope-runner@.service` / `.timer` | per-lane cadence (`gpu`, `kernel`) |
| `deploy/runner/jope-docker-proxy.service` | docker socket proxy (containers + images only) for the kernel lane |
| `deploy/runner/jope-runner-prune.service` / `.timer` | nightly prune (docker 168 h, `cargo sweep --time 7`) |
| `deploy/runner/disk-guard.sh` | refuse under 50 GB free |
| `/etc/jope-runner/github-pat` | **owner-created**, root:root 0600, the fine-grained PAT; never in git |

## 1. Host prerequisites (already true on xbabe2)

* Docker 29 with the `nvidia` runtime (`docker info --format '{{json .Runtimes}}'` shows `nvidia`),
  NVIDIA driver ≥ 560 (595.99.02 receipted), 128 cores, 125 GB RAM, ≥ 200 GB free on the docker
  root (the runner image is 6.1 GB, the SDK volume 3 GB, the shared target volume grows to tens of GB).
* No host Rust, Python or CUDA toolkit is needed by the lanes: everything runs inside the image.

## 2. Provision (root, once per image change)

```bash
sudo deploy/runner/provision.sh            # volumes, images, SDK volume, units; ~15 min (SDK download 2.1 GB)
```

It prints the image ids; compare with `deploy/receipts/runner-image.json` for the commit you built
from. The SDK step downloads LibTorch 2.13.0+cu126 and the five NVIDIA wheels into the `jope-sdk`
volume, verifying each sha256 from the receipt before unpacking, and ends by printing
`2.13.0+cu126` (the `libtorch/build-version` that `deploy/ci/neural-env.sh` checks). Re-run with
`--skip-sdk` after a Dockerfile-only change.

## 3. Register (owner; the only step that touches GitHub)

1. Create a **fine-grained personal access token** scoped to the single repository
   `veox-ai/JopeDime` with exactly one permission: *Administration: Read and write* (needed by
   `POST /repos/{owner}/{repo}/actions/runners/generate-jitconfig`). Expiry ≤ 90 days.
2. Put it in the file the provisioner created, keeping the mode:
   ```bash
   sudo sh -c 'umask 077; printf "%s" "<token>" > /etc/jope-runner/github-pat'
   sudo stat -c '%a %U:%G' /etc/jope-runner/github-pat     # must print: 600 root:root
   ```
3. Start the socket proxy and the lane timers:
   ```bash
   sudo systemctl enable --now jope-docker-proxy.service
   sudo systemctl enable --now jope-runner@gpu.timer jope-runner@kernel.timer jope-runner-prune.timer
   ```
4. Watch the first job: `journalctl -fu jope-runner@gpu.service`. The log shows
   `run.sh: runner xbabe2-gpu-<stamp> id=<n> labels=xbabe2-gpu`, then the listener's
   `Listening for Jobs`. In the repository, *Settings → Actions → Runners* lists the runner while a
   job runs and drops it when the job ends (JIT runners are ephemeral).
5. Trigger `main.yml` with *Run workflow* on `main`, or push. Queued jobs are picked up on the next
   timer tick (≤ 1 min).

Rotation: replace the file's contents; nothing else changes. Revocation: empty the file — `run.sh`
refuses to start (`github-pat is empty`) and the timers keep ticking harmlessly.

## 4. What one job looks like

`jope-runner@gpu.timer` → `run.sh gpu` (root):

1. `disk-guard.sh`: docker root, `/var/lib/jope-runner` and `/tmp` must each have ≥ 50 GB free;
2. read the PAT, `POST …/generate-jitconfig` with `{"name":"xbabe2-gpu-<stamp>","runner_group_id":1,
   "labels":["self-hosted","linux","x64","xbabe2-gpu"],"work_folder":"work"}` (the token is passed
   to curl through a config file on a private fd, never on the command line);
3. `docker run --rm --init --user 1001 --gpus device=0 --cap-drop ALL --security-opt
   no-new-privileges --pids-limit 4096 --memory 96g --cpus 64 --read-only --tmpfs /tmp:size=32g
   --tmpfs /home/runner:size=4g -v jope-work:/home/runner/work -v jope-cargo:/var/cache/jope/cargo
   -v jope-target:/var/cache/jope/target -v jope-uv:/var/cache/jope/uv -v jope-sdk:/opt/jope-sdk:ro
   -v jope-models:/var/lib/jope/models -i jope-runner:current` with the JIT config on **stdin**;
4. the entrypoint copies `/opt/actions-runner` into the tmpfs home and runs
   `run.sh --jitconfig <config>`: one job, then the runner deregisters and the container exits.

The kernel lane is the same without `--gpus` and with the proxy socket mounted at
`/var/run/docker.sock` (`DOCKER_HOST` set); the proxy allows `containers/*` and `images/*` and
denies exec, build, volumes, networks, system, swarm and secrets, so a job can `docker run` the
builder image it finds locally and nothing more. `RuntimeMaxSec=1800` stops an idle listener so a
fresh single-use config is requested by the next tick.

## 5. Volumes

| Volume | Mount | Content | Reset when |
|---|---|---|---|
| `jope-work` | `/home/runner/work` | checkouts | any time (`docker volume rm`; re-run `provision.sh`) |
| `jope-cargo` | `/var/cache/jope/cargo` (`CARGO_HOME`) | registry, git deps | any time |
| `jope-target` | `/var/cache/jope/target` (`CARGO_TARGET_DIR`) | build artifacts; `cargo sweep --time 7` nightly | any time |
| `jope-uv` | `/var/cache/jope/uv` | Python wheels | any time |
| `jope-sdk` | `/opt/jope-sdk` **read-only** | LibTorch + NVIDIA wheels (`deploy/receipts/libtorch.json`) | the receipt changes (`provision.sh` re-provisions when `build-version` differs) |
| `jope-models` | `/var/lib/jope/models` (`JOPE_MODELS_ROOT`) | pinned bundles fetched by `encoder-parity` (WP7) | `models/PINNED` changes |

## 6. Upgrading the image

Bump the version + sha256 pair in `deploy/docker/runner.Dockerfile` (actions-runner from the
release notes' `<!-- BEGIN SHA linux-x64 -->` block; the others from their `checksums.txt` /
`SHA256SUMS` / `.sha256` files; AWS CLI publishes none — compute it from the zip and say so in the
comment), open a PR (`hygiene` lints the Dockerfile's scripts), then on xbabe2:

```bash
sudo deploy/runner/provision.sh --skip-sdk     # rebuilds jope-runner:<sha> and retags :current
```

Running jobs finish on the old image (the container keeps its image); the next tick uses `:current`.
Record the new id/size in `deploy/receipts/runner-image.json` in the same PR.

## 7. Recovery

| Symptom | Cause | Action |
|---|---|---|
| `disk-guard: … refusing to start` | < 50 GB free | `sudo /opt/jope-runner/prune.sh`; if still short, `docker volume rm jope-target` and re-provision |
| `generate-jitconfig failed` | expired/revoked PAT, wrong permission, repo renamed | rotate the token (step 3) |
| jobs queued, runner never appears | timer not enabled, proxy down (kernel lane), docker not running | `systemctl status jope-runner@gpu.timer jope-docker-proxy.service docker` |
| `neural-env: missing /opt/jope-sdk/...` in `neural-cuda` | SDK volume empty | `sudo deploy/runner/provision.sh --skip-image` |
| `LibTorch build-version is '…', receipt expects '…'` | receipt changed, volume not | same |
| listener idles 30 min then stops | no job for that label | expected; each tick requests a new config |

## 8. Security notes

* The PAT never enters a container, an environment variable or a command line; the JIT config
  (single-use, job-scoped) is the only secret the container ever sees, on stdin.
* AWS access is OIDC only (`aws-actions/configure-aws-credentials` → `JopeDimeCiRead`), inside a
  job step; the host has no AWS credentials.
* The container is read-only, unprivileged (uid 1001, no capabilities, no new privileges), has no
  docker socket on the GPU lane and only the filtered proxy on the kernel lane.
* Never start `actions-runner*` GitHub services on this host; that fleet is retired.
