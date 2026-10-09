# Cold telemetry archiver v2 — local review packet

No remote action, cloud request, service lifecycle change or clock freeze was
performed while preparing this packet. Old runtime/source/evidence and its21
fixtures are untouched. This is a new one-use executor, not a restart of v1.

## Verified input and intended continuation

Root's post-exit capture is `prior-capture.json`, exact SHA
`2821889375a2c0b1cd9706cd7e52e1fd008c35545f43802e2119161c30d92bd4`
(2837433B). It records recovery1 inactive/PID0/empty cgroup/resultsuccess/exit0,
with original deadline13:34:39UTC unchanged. Offline `metadata-audit.json`
checks all2636 embedded raw bodies and all750 attempt→reply→verified-object→
receipt joins. The timing JSONL body was omitted by the capture; its exact
physical hash/size remains pinned and is read by actual host preflight.

Recovery1 added375 complete segments,750 PUTs and750 exact-version verified
objects. The full physical archive receipts represent528 complete segments,
1056 objects and16283 events. The reported local chain includes16314 events;
the31-event original gap at part153 remains unresolved and never retried.

Fresh work begins at part529. Conservatively charged historical PUT attempts
are306 original verified +1 original unknown +750 recovery1 =1057. The global
5000 ceiling leaves3943 new attempts. Only complete two-object pairs are begun,
so at most3942 are used; part2500 is never reached. Original global deadline
`1791592146.380285` remains unchanged. A separately frozen3600s v2 window must
fit beneath it. Local trace writers on all hosts remain active and untouched.

## Read-first files and exact pins

- `trace_recovery_v2.py`:
  `fe000e7a0bed06f4b339997eaea745a1b0737430da5543f8f61c4d6ce553df22`.
- `trace_archive_v1_c8a13033.py`: byte-identical prior reviewed primitives,
  checked against `c8a13033c6a93bb97d14d3ccb4de0d2fae01be0928a75f654cfd79f32d835c69`
  **before import**. The old hardcoded constructor/preflight/report are unused.
- `launch_v2.py`:
  `88495db8b75290103e26a69bc0afd382d9e198d29977eb04a68d4436a92de95a`.
- `config.template.json`:
  `5aa2fe2054406ffe05b0ee193afa2efcc35ade93f5be5ce62d08e7b25849d94a`.
- `validation.json`:
  `a85c8b4cbf0e732d07d60eaa0f2cd037b6c89d8dbf4c565e93c6c1986de15ab0`.
- `metadata-audit.json`, `test-result.json`, `test_trace_recovery_v2.py`,
  `tests-log-custody.json`, `prepare_config.py`.

`tests-log-custody.json` preserves the exact historical 2,045-byte test output
as base64, with its original SHA-256
`ca552343a0b81cbf64114ed13626f0a2f34847b871304a599531dded9f27016d`
and the unchanged 13-test qualification identity. Decode `content_base64` and
check `original.bytes` and `original.sha256` to recover `tests.log` byte for byte.
This source representation change did not rerun tests. Historical validation
and independent-review receipts retain their original paths and hashes; the
tracked raw `.log` is replaced by this JSON to satisfy `repo-layout` without
an allowlist exception.

`read_prior.py` is an unused read-only capture helper prepared before root
supplied the authoritative capture. **Do not run another capture merely to
deploy this packet.**

## Verification and stop conditions

`cold_prefix` reopens all exact prior physical pins; validates old config/source/
execution clock; original153 receipt authorities; every later receipt and
durable numbered attempt/reply/verified/unknown object; progress counters;
timing sequence/config/host; and the complete local sealed event chain from
part0 through the last attempted part. It derives fresh start and global
attempt balance from physical evidence. Source-tree files are never edited.

Any unreceipted attempted tail becomes a preserved gap and is skipped wholesale,
even if one or both object readbacks exist. It is never PUT again. This exact
captured run has no new unknown tail; the launcher additionally requires the
expected529/528/1057/3943 result. A different outcome needs a new reviewed
configuration, not a silent dynamic expansion.

The only permissible later resolution of an unknown object is a separately
scoped read-only HEAD followed by exact VersionId GET/hash/size verification.
V2 does not perform that reconciliation and does not assert absence from an
old404 observation. Unknown/local bytes and all old receipts remain in place.

Before each new PUT, the unchanged primitive fsyncs a create-only attempt
record. PUT uses `IfNoneMatch='*'`, retriesdisabled; returned versions receive
exact-VersionId GET and byte/hash checks. Ambiguous failures stop the executor.
New receipt schema is `jope.hpc-host-archive-recovery-receipt/v2`, binding prior
capture, source/config, gap parts and global attempt count. No raw training
data, model weights or arbitrary paths are archived.

The new local receipt namespace is
`/home/ubuntu/JopeDime/.agent/proposals/dl-flywheel-trace-recovery-20261009-v2`.
Fresh immutable S3 segment keys continue under the existing authorized prefix
`experiments/gp-policy-dl/v2-trainonly/paper-evidence/20261009/hpc-trace-v1/xbabe3/`
at previously unattempted part529 and beyond. Objects are the sealed telemetry
JSONL and its seal, two per part; no foreign object is overwritten.

## CPU qualification and limits

13 new tests passed in1.094s on the final source. They exercise physical chain corruption, retired
PID gating, missing intent sequence, version joins, unknown one/two-object
tails, conservative global charging, stale progress, immutable old evidence,
actual fake-client PUT→exactGET closure, pair budget and backward-wall-clock
deadline protection. No old21 fixtures were rerun. All final source ASTs were parsed; qualification timing is included in the
final source tested by the receipt.

Actual host preflight is still required: offline metadata audit does not claim
to have read remote sealed event payloads. Launcher runs the exact new helper
under CPU100%,512MiB,swap0,Tasks24,Nice19,noCUDA,BLAS1,Restartno,KillModecontrol-group,
with preflight<=180s and an external RuntimeMax bounded by the one new deadline.
It requires old service PID0/empty cgroup and new unit notfound; requires fresh
MemAvailable>=26GiB and load1<100. It does not stop any existing service.

New upload/GET/local-seal timing preserves UTCns, monotonicns durations,
process+child CPU deltas, config/host/PID/sequence, bytes/key/version and errors.
Cold-prefix qualification also reports UTCns/monotonic duration/CPU deltas.
GPU execution time remains explicitly null; no GPU work is claimed.

## Root deployment handoff after independent review

1. Claim the new scope in the shared source/handoff protocol. Confirm old v1 is
   physically inactive; its captured bytes must still match.
2. Independent review JSON must include `state: passed` and
   `template_sha256: 5aa2fe2054406ffe05b0ee193afa2efcc35ade93f5be5ce62d08e7b25849d94a`.
3. In this local proposal directory, freeze once:

```sh
rtk proxy python3 -B prepare_config.py --template-sha 5aa2fe2054406ffe05b0ee193afa2efcc35ade93f5be5ce62d08e7b25849d94a --review /absolute/independent-review.json --review-sha EXACT_REVIEW_SHA
```

4. Create-only stage exact `trace_recovery_v2.py`,
   `trace_archive_v1_c8a13033.py`, `launch_v2.py`, `prior-capture.json`, and the
   frozen configuration as `config.json` into the new root above. Compare
   every destination byte hash/size. Preserve existing foreign paths; if this
   new root unexpectedly exists with different bytes, stop.
5. On xbabe3, invoke the staged launcher with the returned frozen configSHA:

```sh
rtk proxy python3 -B /home/ubuntu/JopeDime/.agent/proposals/dl-flywheel-trace-recovery-20261009-v2/launch_v2.py --config-sha EXACT_FROZEN_SHA
```

It saves preflight intent/reply, verifies the full physical chain with no S3
requests, then saves launch intent and starts exactly one
`jopedime-hpc-archive-recovery-20261009-v2` unit. No automatic rerun exists.
Do not use `python -O`; launcher assertions are part of the reviewed gate.

Completion/status evidence is `progress.json`, `execution-started.json`,
`cold-prefix.json`, `stage-timing.jsonl`, create-only `receipts/NNNNN.json`,
attempt/reply/verified-object files, and `fatal.json` on failure. Report actual
verified counts and preserved gaps separately. After a failure, preserve the
new attempt tree and stop; never restart this version or reset its clock.
