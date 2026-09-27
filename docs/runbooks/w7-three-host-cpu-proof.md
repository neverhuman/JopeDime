# Runbook: W7 three-host CPU lifecycle proof (xbabe1, xbabe2, xbabe3)

This is the operator procedure for W7-T07. It runs one admitted CPU rank on each of three
hosts through each host's SmartCluster daemon, then exercises cancellation, a coordinator
restart and a refused invalid input. The components are `jope-native-loop census`, `plan`,
`preflight`, `agent`, `coordinate`, `cancel`, `status` and `verify`, plus `participant`,
which each host's daemon runs as a trusted service. SSH moves files between hosts; it never
starts, supervises or stops a rank.

**Status: not executed.** No phase has run on xbabe1, xbabe2 or xbabe3. This runbook is the
next, separately admitted step. Do not start it without the admission in §0.

What has run:

- The same components pass a same-host fixture (`bounded_smartcluster`).
- After owner decisions W7-D1 to W7-D6, a local dry run checked §0.3 and §1 to §9 and §13 on
  the build host (xbabe2). It used one rank, loopback addresses, five updates, a scratch
  embedded `scqd` (never the installed daemon) and a throwaway grant key. It covered the
  OpenSSL credentials, `contract author`, `grant sign` and `grant verify`, and the rank's job
  proof (`w7c-runbook-dryrun-20260927T014529Z`).

The dry run is neither the proof nor an admission.

W7-T07 accepts the proof when three distinct host and boot identities have:

- actual execution receipts;
- successful, verified results;
- refused invalid inputs;
- complete job and drain inventories.

A same-host three-process run is a fixture, not this proof.

## 0. Before anything runs

### 0.1 Decided (owner decisions 2026-09-27, W7-D1 to W7-D6)

- **Ranks** are SmartCluster trusted-service jobs of each host's `scqd` (W7-D1).
  - Before any work, each rank proves it is its planned job's leader. It checks the job
    environment, that its control-socket parent is the pinned daemon, and the daemon's live
    record of its attempt, tenant, plan, dispatch, service and arguments.
  - A rank that cannot prove this refuses and writes `outbox/refusal.txt`.
- **TLS identity is pinned leaf certificates** (W7-D4).
  - Each host presents one self-signed Ed25519 leaf, and the plan pins its SHA-256. No CA
    and no chain is accepted.
  - **Rotating a certificate needs a new census and a new plan.** A host with a new
    certificate is refused by every plan that pinned the old one.
  - **The coordinator must be a planned host** (here xbabe1, rank 0's host). Otherwise it has
    no pinned identity and every rank refuses it.
- **Contracts and grants come from repository verbs** (W7-D6). The unsigned contract is
  authored with `contract author`, the owner signs it with `grant sign`, and anyone checks it
  with `grant verify` (§6). Nothing is hand-assembled.

### 0.2 Owner authority (record each in the admission record, §7)

1. **Phase admission** for a 4 CPU / 8 GiB / 15-minute CPU phase on each host, and a CI and
   runner reservation window. The census does not read CI or SmartCluster reservations, so
   the window is kept by hand (§7).
2. **The grant authority key.**
   - It stays in the owner's custody and signs with `grant sign` (§6.2).
   - It is read only from a file that is owner-only (`0600`) and not a symlink; it is never an
     argument or an environment value.
   - Never keep it on xbabe1, xbabe2 or xbabe3.

### 0.3 Host checklist (each host; every line must pass, and its output is kept as evidence)

Run these after §2.3 and before §5. Save the whole transcript as
`$EVIDENCE/host-checklist.txt`.

```bash
CONF=/etc/jain-smartcluster/scqd.toml
# 1. The trusted service jope-bounded-rank maps to exactly the installed binary.
grep -E "^jope-bounded-rank = \"$BIN\"$" "$CONF"
# 2. The operator is a daemon admin uid (rank jobs are submitted by the operator).
grep -E "^admin_uids = \[.*\b$(id -u)\b.*\]" "$CONF"
# 3. Jobs run as their submitting principal: no worker uid override, so a rank's euid is the
#    operator's and its job tenant is exactly unix:<operator uid> (W7-D1 checks this).
! grep -E '^\s*worker_uid\s*=' "$CONF" && echo "tenant: unix:$(id -u)"
# 4. The operator can reach the daemon socket.
scq --socket "$SCQ_SOCKET" --json ping | jq -e '.daemon_version' >/dev/null && echo socket-ok
# 5. One rank per host unless cgroups are delegated. A daemon in safe_single_allocation runs
#    one job at a time, so a plan may put at most one rank on this host (the proof puts one).
scq --socket "$SCQ_SOCKET" --json doctor | jq -r '"mode: \(.mode), cgroup_delegated: \(.capabilities.cgroup_delegated)"'
# 6. The installed binary and the daemon are the pinned ones (§2.2).
sha256sum "$BIN" "$SCQD_EXE"
# 7. The ports are free before the run: 7460 on xbabe1, 7461 on every host.
! ss -ltn | grep -E ':(7460|7461)\b'
```

Check 5 must show `mode: full` with `cgroup_delegated: true`, or else at most one planned rank
on this host. The runbook's plans put exactly one rank on each host.

**Network.** TCP 7460 (the coordinator's rendezvous, on xbabe1) and TCP 7461 (the agent, on
each host) must be reachable between the three hosts. Both are TLS 1.3 with mutual,
plan-pinned certificates.

## 1. Names used below

| Host | Role | Rank |
|---|---|---|
| xbabe1 | coordinator host, and rank 0 | 0 |
| xbabe2 | rank 1 | 1 |
| xbabe3 | rank 2 | 2 |

Set these in every shell on every host. Nothing goes under `/tmp`.

```bash
export W7=$HOME/jope-w7                      # per-host working root (operator-owned)
export BIN=$W7/bin/jope-native-loop          # the one installed worker/controller binary
export STAMP=$(date -u +%Y%m%dT%H%M%SZ)      # set once on xbabe1, then copy the same value to the others
export EVIDENCE=$HOME/jope-evidence/w7-cpu-proof/$STAMP/$(hostname)
export SCQ_SOCKET=/run/jain-smartcluster/scqd.sock
export ADDR=<this host's address, exactly as assigned to a local interface>   # e.g. from `ip -o -4 addr show`
mkdir -p $W7/{bin,install,tls,keys,inputs,census,contracts,plans,runs,state,supervision} $EVIDENCE
chmod 700 $W7/tls $W7/keys
```

`ADDR1`, `ADDR2` and `ADDR3` below are the three hosts' `ADDR` values. `HOST1`, `HOST2` and
`HOST3` are the host names each host's census records (`.host` in its collection receipt, §5).

## 2. Binaries, installation and verification by digest

### 2.1 Build once, from the reviewed commit

On the build checkout (the canonical `~/JopeDime` at the reviewed, merged commit, clean):

```bash
export COMMIT=$(git -C ~/JopeDime rev-parse HEAD)          # 40 hex; recorded in every census
git -C ~/JopeDime status --porcelain                       # must print nothing
export BUILD=$HOME/jope-evidence/w7-cpu-proof/$STAMP/build  # build receipts stay out of the checkout
mkdir -p $BUILD
(cd ~/JopeDime && CARGO_INCREMENTAL=0 cargo build --locked --release -p jope-parallel --bin jope-native-loop)
cp ~/JopeDime/target/release/jope-native-loop $BUILD/
(cd $BUILD && sha256sum jope-native-loop | tee jope-native-loop.sha256)
(cd ~/JopeDime && rustc -vV) > $BUILD/rustc-vV.txt         # the compiler identity every census pins
echo "$COMMIT" > $BUILD/commit.txt
```

### 2.2 Install on each host, then verify

```bash
scp $BUILD/jope-native-loop $BUILD/rustc-vV.txt $BUILD/jope-native-loop.sha256 $BUILD/commit.txt <host>:jope-w7/install/
# on the host:
export COMMIT=$(cat $W7/install/commit.txt)
install -m 0755 $W7/install/jope-native-loop $BIN
sha256sum $BIN                                              # must equal jope-native-loop.sha256
sha256sum $BIN $W7/install/rustc-vV.txt > $EVIDENCE/installed-digests.txt
```

How the installation is verified, and by what:

| What | Pinned by | Checked where |
|---|---|---|
| `jope-native-loop` bytes | `executable_sha256` in each host's census, copied into the plan | The collector hashes `--executable`. Each participant hashes `/proc/self/exe` into its signed hello. The rendezvous refuses the whole group on any difference |
| runtime libraries | `runtime.runtime_sha256` (closure digest of the files named in `installation.json`) | collector and participant, as above |
| compiler identity | `runtime.compiler_sha256` of `rustc-vV.txt` | collector and participant |
| installed policy | `installed_policy_sha256` of `/etc/jain-smartcluster/scqd.toml` | collector and participant |
| host TLS certificate | `certificate_sha256` (SHA-256 of the DER) | Every TLS handshake: coordinator, agents and participants accept only planned digests |
| `scqd` | `daemon.executable_sha256` and `daemon.uid` in the supervision file (§8) | Every daemon request (`native_scq::authenticated_exchange`) |

`scqd` is the host's installed SmartCluster daemon. It is not rebuilt for this proof. Record
its identity on each host:

```bash
SCQD_EXE=<absolute path from the daemon unit's ExecStart>
sha256sum "$SCQD_EXE" | tee $EVIDENCE/scqd.sha256
ps -o uid=,pid=,args= -C scqd | tee $EVIDENCE/scqd-process.txt   # the daemon uid
```

### 2.3 Daemon registration (each host; then run the §0.3 checklist)

Add the following to `/etc/jain-smartcluster/scqd.toml`, then restart the daemon the way the
host runs it. The service path must be exactly `$BIN`, absolute.

```toml
[auth]
admin_uids = [<existing uids>, <operator uid from `id -u`>]

[workers.trusted_services]
jope-bounded-rank = "/home/<operator>/jope-w7/bin/jope-native-loop"
```

Then:

```bash
scq --socket $SCQ_SOCKET --json doctor | tee $EVIDENCE/scq-doctor.json
scq --socket $SCQ_SOCKET --json ls --project w7-cpu-proof | tee $EVIDENCE/jobs-before.json   # expect no job
```

An embedded daemon without delegated cgroups runs one job at a time. That is enough here,
because the proof has one rank per host.

## 3. Credentials (each host)

```bash
# Host TLS identity: a self-signed Ed25519 leaf, presented alone (no chain).
(umask 077 && openssl req -x509 -new -newkey ed25519 -nodes -days 7 -subj "/CN=$(hostname)" \
  -keyout $W7/tls/host.key -out $W7/tls/host.pem)
openssl x509 -in $W7/tls/host.pem -outform DER | sha256sum | tee $EVIDENCE/certificate.sha256

# This host's rank key (PKCS#8 Ed25519), and its public key in the plan's hex form.
(umask 077 && openssl genpkey -algorithm ed25519 -out $W7/keys/rank.key)
openssl pkey -in $W7/keys/rank.key -pubout -outform DER | tail -c 32 | xxd -p -c 32 | tee $EVIDENCE/rank.pub
```

On xbabe1 only, create the coordinator key:

```bash
(umask 077 && openssl genpkey -algorithm ed25519 -out $W7/keys/coordinator.key)
openssl pkey -in $W7/keys/coordinator.key -pubout -outform DER | tail -c 32 | xxd -p -c 32 | tee $EVIDENCE/coordinator.pub
```

Every private key file must be mode 0600, or the tools refuse to load it.

The certificate lives 7 days. The plan pins its digest, so a rotated certificate needs a new
census and a new plan. Private keys never leave their host, and no key is copied into the
evidence.

## 4. Installation file and input closure (each host)

The input closure is one file, byte-identical on every host. Create it on xbabe1, copy it to
the other two, and check it on each:

```bash
# xbabe1:
head -c 1048576 /dev/urandom > $W7/inputs/w7-cpu-proof.bin
sha256sum $W7/inputs/w7-cpu-proof.bin; stat -c %s $W7/inputs/w7-cpu-proof.bin
scp $W7/inputs/w7-cpu-proof.bin xbabe2:jope-w7/inputs/ && scp $W7/inputs/w7-cpu-proof.bin xbabe3:jope-w7/inputs/
# every host (the digest must be identical):
sha256sum $W7/inputs/w7-cpu-proof.bin | tee $EVIDENCE/input.sha256
```

The installation file names exactly what the census collector and every participant on the
host observe:

```bash
LIBS=$(ldd $BIN | awk '$2 == "=>" && $3 ~ /^\// {print $3} $1 ~ /^\// {print $1}' | sort -u | jq -R . | jq -s .)
DAEMON_UID=$(ps -o uid= -C scqd | head -1 | tr -d ' ')
jq -n --arg addr "$ADDR" --arg w7 "$W7" --arg commit "$COMMIT" --argjson libs "$LIBS" \
  --arg socket "$SCQ_SOCKET" --argjson daemon_uid "$DAEMON_UID" \
  --arg scqd_sha "$(cut -d' ' -f1 $EVIDENCE/scqd.sha256)" '{
  schema: "jope.native_host_installation.v1",
  address: $addr,
  certificate: ($w7 + "/tls/host.pem"),
  certificate_key: ($w7 + "/tls/host.key"),
  source_commit: $commit,
  executable: ($w7 + "/bin/jope-native-loop"),
  runtime_libraries: $libs,
  compiler: ($w7 + "/install/rustc-vV.txt"),
  cuda_libraries: [],
  nccl_libraries: [],
  installed_policies: ["/etc/jain-smartcluster/scqd.toml"],
  gpu_inventory: false,
  input_root: $w7,
  smartcluster: {
    daemon: {socket: $socket, uid: $daemon_uid, executable_sha256: $scqd_sha},
    service: "jope-bounded-rank"
  }
}' > $W7/install/installation.json
cp $W7/install/installation.json $EVIDENCE/
```

`smartcluster` pins the daemon each rank must prove it was started by (W7-D1): its socket, its
uid and the digest of its executable. The supervision file (§8) must name the same daemon and
service, or the supervisor refuses to start.

The closure path in the plan is `inputs/w7-cpu-proof.bin`, relative to `input_root`.

## 5. Census (each host), then merge (xbabe1)

Collect once to author the plans (§6), and again within 10 minutes before each `coordinate`
(§9 to §12). The plan pins each host's identity, including its boot, but not a census digest.
The plan's telemetry policy allows 600 000 ms, and `coordinate` journals a fresh preflight of
the census it is given. `collect` never overwrites a file, so each collection gets its own
`CSTAMP`: set it once on xbabe1 and use the same value on all three hosts.

```bash
export CSTAMP=$(date -u +%Y%m%dT%H%M%SZ)
CUDA_VISIBLE_DEVICES="" $BIN census collect \
  --out $W7/census/$(hostname)-$CSTAMP.json \
  --address "$ADDR" \
  --certificate $W7/tls/host.pem \
  --source-commit "$COMMIT" \
  --executable $BIN \
  $(jq -r '.runtime_libraries[] | "--runtime-library " + .' $W7/install/installation.json) \
  --compiler $W7/install/rustc-vV.txt \
  --installed-policy /etc/jain-smartcluster/scqd.toml \
  | tee $EVIDENCE/census-collect-$CSTAMP.json
cp $W7/census/$(hostname)-$CSTAMP.json $EVIDENCE/
```

The receipt (`jope.native_host_census_collection.v1`) records the following. Check each value:

- `census_sha256`;
- `host`, `machine_id`, `boot_id` and `address`;
- `gpu_probe` must be `not_queried_cuda_visible_devices_empty`, with `gpus: 0`;
- `allocation_held: false`.

`CUDA_VISIBLE_DEVICES=""` records a CPU-only census and runs no `nvidia-smi`. The collector
never allocates or reserves anything.

Copy the two other documents to xbabe1, unchanged, then merge. Each document is pinned by the
`census_sha256` its own collector printed, taken from that host's receipt and never
recomputed:

```bash
# xbabe1:
scp "xbabe2:jope-w7/census/$HOST2-$CSTAMP.json" "xbabe3:jope-w7/census/$HOST3-$CSTAMP.json" $W7/census/
$BIN census merge --out $W7/census/merged-$CSTAMP.json --max-age-ms 600000 --clock-skew-ms 2000 \
  --document $W7/census/$HOST1-$CSTAMP.json=<xbabe1 census_sha256> \
  --document $W7/census/$HOST2-$CSTAMP.json=<xbabe2 census_sha256> \
  --document $W7/census/$HOST3-$CSTAMP.json=<xbabe3 census_sha256> \
  | tee $EVIDENCE/census-merge-$CSTAMP.json
```

(The files are named by `$(hostname)`. If that differs from the census `host`, use the
actual file names.)

The merge refuses:

- an altered document;
- a stale or future-dated document;
- a duplicate document, or two that share a host, machine, boot, address or certificate;
- mixed schemas.

## 6. Plans (xbabe1): one per run

Four runs, each its own plan and run name:

| Run | Name | What it shows |
|---|---|---|
| 1 | `w7-cpu-commit-$STAMP` | the whole group commits once |
| 2 | `w7-cpu-cancel-$STAMP` | a cancel drains the whole group |
| 3 | `w7-cpu-restart-$STAMP` | a SIGKILLed coordinator is restarted and reconciles |
| 4 | `w7-cpu-invalid-$STAMP` | a substituted input refuses the whole group |

Work on one run at a time. For each run, set `RUN` first.

### 6.1 Contract (unsigned): `contract author`

Each rank has these ceilings:

- 4000 CPU millicores;
- 8 GiB RAM;
- a 900 000 ms (15-minute) wall;
- no GPU.

The work census stays inside them: 870 000 ms of wall time and 655 000 CPU core-ms per rank.
There are 540 planned updates, one per 1000 ms heartbeat, so about 9 minutes of work. Grants
last 30 minutes, and the original deadline is 1 hour.

The request is small and declarative. `contract author` takes three things from elsewhere:

- every host identity, from the merged census; a host not in the census is refused;
- the closure's digest and size, from the file under `--input-root`;
- every grant body, derived from its participant.

It then checks everything `plan` checks except the signatures. The output is a new file,
never a replacement.

```bash
export RUN=w7-cpu-commit-$STAMP        # then w7-cpu-cancel-…, w7-cpu-restart-…, w7-cpu-invalid-…
jq -n --arg run "$RUN" --arg coord "$HOST1" \
  --arg coord_pub "$(cat $EVIDENCE/coordinator.pub)" --arg auth_pub "<owner grant-authority public key, 64 hex>" \
  --arg h1 "$HOST1" --arg h2 "$HOST2" --arg h3 "$HOST3" \
  --arg k1 "$(cat <xbabe1 rank.pub>)" --arg k2 "$(cat <xbabe2 rank.pub>)" --arg k3 "$(cat <xbabe3 rank.pub>)" '
  def rank($h; $k): {host: $h, public_key: $k, device_uuid: null,
    ceilings: {cpu_millicores: 4000, ram_bytes: 8589934592, gpu_vram_bytes: null, wall_ms: 900000}};
  {schema: "jope.native_bounded_contract_request.v1", run: $run, attempt: 1,
   coordinator_host: $coord, coordinator_public_key: $coord_pub, grant_authority_public_key: $auth_pub,
   window_ms: 3600000, grant_ms: 1800000,
   telemetry: {max_age_ms: 600000, clock_skew_ms: 2000},
   rendezvous: {join_timeout_ms: 120000, max_message_bytes: 65536, heartbeat_timeout_ms: 5000},
   artifact_closure: ["inputs/w7-cpu-proof.bin"],
   participants: [rank($h1; $k1), rank($h2; $k2), rank($h3; $k3)],
   work_per_rank: [
     {kind: "numerical", cpu_core_ms: 600000, gpu_ms: 0, wall_ms: 600000, peak_ram_bytes: 1073741824, peak_vram_bytes: 0},
     {kind: "verification", cpu_core_ms: 10000, gpu_ms: 0, wall_ms: 60000, peak_ram_bytes: 536870912, peak_vram_bytes: 0},
     {kind: "artifact_movement", cpu_core_ms: 10000, gpu_ms: 0, wall_ms: 60000, peak_ram_bytes: 536870912, peak_vram_bytes: 0},
     {kind: "cleanup", cpu_core_ms: 5000, gpu_ms: 0, wall_ms: 30000, peak_ram_bytes: 268435456, peak_vram_bytes: 0},
     {kind: "failure_allowance", cpu_core_ms: 30000, gpu_ms: 0, wall_ms: 120000, peak_ram_bytes: 1073741824, peak_vram_bytes: 0}],
   counters: {committed_updates_at_start: 0, planned_updates: 540}}' > $W7/contracts/$RUN.request.json
$BIN contract author --request $W7/contracts/$RUN.request.json --census $W7/census/merged-$CSTAMP.json \
  --input-root $W7 --out $W7/contracts/$RUN.unsigned.json | tee $EVIDENCE/$RUN-authored.json
```

The printed `review` lists each grant's rank, host, window and body digest. The owner signs
exactly those bodies.

### 6.2 Grants (owner, with the grant-authority key): `grant sign`, `grant verify`

Copy `$RUN.unsigned.json` to the owner's signing machine, which holds the authority key.
There, and only there:

```bash
# The key file must be PKCS#8 Ed25519, mode 0600 and not a symlink; it is never an argument
# value or an environment variable and is never printed.
jope-native-loop grant sign --contract $RUN.unsigned.json --key <grant-authority.key> --out $RUN.json
jope-native-loop grant verify --contract $RUN.json
```

`grant sign` refuses:

- a key that is not the contract's `grant_authority_public_key`;
- a contract that is already signed;
- a grant whose window has already expired.

The signed contract passes Plan's full validation before it is written. Copy `$RUN.json` back
to `$W7/contracts/` on xbabe1 and verify it there as well:

```bash
$BIN grant verify --contract $W7/contracts/$RUN.json | tee $EVIDENCE/$RUN-grants.json
```

`plan_sha256` in that report is the digest `plan` will freeze (§6.3).

### 6.3 Plan, then distribute the frozen plan

```bash
$BIN plan --run-dir $W7/runs/$RUN --contract $W7/contracts/$RUN.json | tee $EVIDENCE/$RUN-plan.json
export PLAN=$(jq -r .plan_sha256 $EVIDENCE/$RUN-plan.json)
cp $W7/runs/$RUN/plan.json $W7/plans/$RUN.json
scp $W7/plans/$RUN.json xbabe2:jope-w7/plans/ && scp $W7/plans/$RUN.json xbabe3:jope-w7/plans/
# every host (each digest must equal $PLAN):
sha256sum $W7/plans/$RUN.json | tee $EVIDENCE/$RUN-plan.sha256
```

`plan` refuses the following before it creates anything:

- invalid, underfunded and legacy contracts;
- an unsigned or wrongly signed grant;
- a grant that outlives the deadline.

Its `review` is the reviewed summary.

## 7. Preflight and admission (xbabe1)

```bash
$BIN preflight --run-dir $W7/runs/$RUN --plan-sha256 $PLAN --census $W7/census/merged-$CSTAMP.json \
  | tee $EVIDENCE/$RUN-preflight.json
```

`preflight` is read-only. It dispatches nothing, holds no allocation (`allocation_held: false`)
and prints a receipt only when there is no refusal reason. Then, on every host, check that
nothing else holds the CPU and RAM the plan needs:

```bash
scq --socket $SCQ_SOCKET --json ps | tee $EVIDENCE/$RUN-jobs-at-admission.json
```

**Admission.** The owner approves exactly `$PLAN`. The approval is based on the plan review,
the preflight receipt and the job listings, and it is recorded as `$EVIDENCE/$RUN-admission.txt`
with these fields:

- the plan digest;
- the merged census digest;
- the approver;
- the UTC time;
- the answers to §0.

Nothing below starts without it.

## 8. Supervision and agents (each host, per run)

```bash
RANK=<0 on xbabe1, 1 on xbabe2, 2 on xbabe3>; HOST=<this host's census host name>
jq -n --arg host "$HOST" --arg w7 "$W7" --arg run "$RUN" --arg rank "$RANK" \
  --arg socket "$SCQ_SOCKET" --argjson daemon_uid <daemon uid from §2.2> \
  --arg scqd_sha "<scqd.sha256 digest>" --arg tenant "unix:$(id -u)" --arg coordinator "$ADDR1:7460" '{
  host: $host,
  daemon: {socket: $socket, uid: $daemon_uid, executable_sha256: $scqd_sha},
  tenant: $tenant,
  project: "w7-cpu-proof",
  service: "jope-bounded-rank",
  plan: ($w7 + "/plans/" + $run + ".json"),
  state: ($w7 + "/state/" + $run),
  coordinator: $coordinator,
  installation: ($w7 + "/install/installation.json"),
  participant_keys: {($rank): ($w7 + "/keys/rank.key")}
}' > $W7/supervision/$RUN.json

setsid nohup $BIN agent --plan $W7/plans/$RUN.json --plan-sha256 $PLAN --host "$HOST" --rank $RANK \
  --listen 0.0.0.0:7461 --certificate $W7/tls/host.pem --certificate-key $W7/tls/host.key \
  --supervision $W7/supervision/$RUN.json \
  > $EVIDENCE/$RUN-agent.out 2> $EVIDENCE/$RUN-agent.err < /dev/null &
echo $! > $EVIDENCE/$RUN-agent.pid
head -1 $EVIDENCE/$RUN-agent.out     # {"agent":"listening","host":…,"address":"0.0.0.0:7461"}
```

The agent serves only its own ranks of this plan. It accepts only the coordinator's planned
certificate, and it refuses to start if its certificate is not this host's planned one.

On xbabe1, write the agent map:

```bash
jq -n --arg a1 "$ADDR1:7461" --arg a2 "$ADDR2:7461" --arg a3 "$ADDR3:7461" \
  '{"0":$a1,"1":$a2,"2":$a3}' > $W7/runs/$RUN-agents.json
```

## 9. Run 1: commit

On xbabe1, with a census collected in the last 10 minutes (otherwise repeat §5 first):

```bash
$BIN coordinate --run-dir $W7/runs/$RUN --plan-sha256 $PLAN --census $W7/census/merged-$CSTAMP.json \
  --listen 0.0.0.0:7460 --certificate $W7/tls/host.pem --certificate-key $W7/tls/host.key \
  --coordinator-key $W7/keys/coordinator.key --heartbeat-ms 1000 --agents $W7/runs/$RUN-agents.json \
  > $EVIDENCE/$RUN-coordinate.json 2> $EVIDENCE/$RUN-coordinate.err
$BIN verify --run-dir $W7/runs/$RUN --plan-sha256 $PLAN | tee $EVIDENCE/$RUN-verify.json
```

Expected results:

- `coordinate.json`: `outcome` is `committed`, `committed_updates_total` is 540, and every rank
  has `disposition` `committed`;
- `verify.json`: `results_verified` is 3.

Each host's `$W7/state/$RUN/<dispatch id>/outbox/` holds `admitted`, `result.bin` (a
`jope.native_cpu_proof_result.v1` document naming that host's `machine_id` and `boot_id`) and
`envelope.json`.

## 10. Run 2: cancellation

Set up the run as in §6 to §8, then start `coordinate` exactly as in §9, in the background
(`&`, with its PID saved). Wait until every host has
`$W7/state/$RUN/*/outbox/admitted`, then run on xbabe1:

```bash
$BIN cancel --run-dir $W7/runs/$RUN --plan-sha256 $PLAN --reason "W7-T07 cancellation drill" \
  | tee $EVIDENCE/$RUN-cancel.json
wait <coordinate pid>
```

Expected results:

- `cancel.json` shows `delivered_to_live_coordinator: true`;
- `coordinate.json` shows `outcome` `cancelled`, `committed_updates_total` 0, and every rank
  `drained`.

## 11. Run 3: coordinator restart

Set up the run as in §6 to §8, and start `coordinate` in the background as in §10. After every
rank is admitted, run on xbabe1:

```bash
kill -KILL <coordinate pid>
$BIN status --run-dir $W7/runs/$RUN | tee $EVIDENCE/$RUN-status-after-kill.json   # outcome null, coordinator_live false
```

Wait at least 5 s (the plan's heartbeat timeout). Every participant loses control and exits
75. Then run exactly the §9 `coordinate` command again, writing
`$EVIDENCE/$RUN-coordinate-restart.json`.

Expected results:

- `outcome` is `drained`, `committed_updates_total` is 0, and every rank has a disposition;
- `original_deadline_unix_ms` is the plan's own, never renewed;
- the journal has `coordinator_restarted`;
- `verify` passes.

## 12. Run 4: a substituted input is refused

Set up the run as in §6 to §8. Before starting `coordinate`, substitute the input on xbabe3
only, keeping the original:

```bash
# xbabe3:
cp -p $W7/inputs/w7-cpu-proof.bin $W7/inputs/w7-cpu-proof.bin.orig
printf 'x' >> $W7/inputs/w7-cpu-proof.bin
sha256sum $W7/inputs/w7-cpu-proof.bin      # must differ from the planned digest
```

Run `coordinate` as in §9. Rank 2's signed hello carries the digest of the input it actually
holds, and the coordinator refuses the whole group as `substituted_input` for rank 2.
Expected: `outcome` is `refused`, no update is committed, the journal records the refusal,
and no rank starts work. Then restore the input and check it:

```bash
# xbabe3:
mv $W7/inputs/w7-cpu-proof.bin.orig $W7/inputs/w7-cpu-proof.bin && sha256sum $W7/inputs/w7-cpu-proof.bin
```

## 13. After each run: inventories and no orphan (each host)

```bash
$BIN status --run-dir $W7/runs/$RUN | tee $EVIDENCE/$RUN-status.json                   # xbabe1
scq --socket $SCQ_SOCKET --json ls --project w7-cpu-proof | tee $EVIDENCE/$RUN-jobs.json  # every job terminal
pgrep -af 'jope-native-loop participant' | tee $EVIDENCE/$RUN-processes.txt              # must be empty
kill $(cat $EVIDENCE/$RUN-agent.pid)                                                     # stop this run's agent
cp -r $W7/state/$RUN $EVIDENCE/$RUN-state                                                # launch records and outboxes
```

On xbabe1, also run `cp -r $W7/runs/$RUN $EVIDENCE/$RUN-run`. That copies the frozen plan,
the journal and the committed results.

## 14. Receipts to retain

Everything goes under `~/jope-evidence/w7-cpu-proof/$STAMP/<host>/`, never in a working tree
and never under `/tmp`. Keep the following, with the UTC time each was taken:

- **Identity:**
  - `installed-digests.txt`, `scqd.sha256`, `scqd-process.txt` and `certificate.sha256`;
  - `input.sha256` and `installation.json`;
  - `uname -a` and `cat /proc/sys/kernel/random/boot_id`, taken before and after the runs.
    A reboot in between voids the proof.
- **Census:** for every collection (`CSTAMP`), each host's census document and
  `census-collect-<CSTAMP>.json`, plus `census-merge-<CSTAMP>.json` and the merged census.
- **Plans:**
  - for each run: the request, `authored.json` (the grant review), the unsigned and signed
    contracts, `grants.json` (the `grant verify` report) and `plan.json`;
  - `<run>-plan.sha256` on every host;
  - the preflight receipt, the job listings at admission and the admission record.
- **Execution:**
  - `coordinate.json` and `.err` (plus `coordinate-restart.json` for run 3);
  - `cancel.json` (run 2) and `status-after-kill.json` (run 3);
  - each agent's `.out` and `.err`;
  - each host's `state/<run>/` directory: `launch.json` and `outbox/{admitted,result.bin,envelope.json,refusal.txt}`.
- **Closure:** each `status`, `verify` and `jobs` output and each `processes.txt`, plus
  `jobs-before.json` from before the first run.
- **The run directories** on xbabe1: the frozen plan, the journal and `results/`.

Do not retain private keys (`host.key`, `rank.key`, `coordinator.key` or the grant-authority
key).

## 15. Not covered here

- **GPU ranks and training** are W7-T08 and W7-T09.
  - SmartCluster allocates a GPU rank's device, and the rank refuses unless that device's index
    translates to its planned UUID (W7-D3).
  - A plan that selects training runs only on a `neural` build of `jope-native-loop` (W7-D2).
  - This CPU proof installs the default build and plans no GPU.
- **Reservations.** The census does not read CI, runner or SmartCluster reservations; §7 checks
  them by hand.
- **Training in this proof.** The CPU phase is a lifecycle proof (`numerical: false`), and its
  plans select no training partition.
