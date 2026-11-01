# Q1 evaluator/runtime handoff — T1005 draft

## Design note

Record the existing qualified-input consumer and FIFO protocol against merged
source `b5a1c73de8e3dd074eff1d5c81992056511f7161`, as assigned by ENG T1005.
The [machine-readable draft](evaluator-runtime-handoff.draft.json) pins source
bytes, frozen r782/DOPE and roster identities, and names the missing execution
evidence. This is documentation under spec §§13–14 (Q1), not a new admission
mechanism or a qualified receipt. The final runtime must bind the actual merged
source used for execution; this historical reference does not prequalify a later
binary. Source, executable, adapter, plan-file and canonical-plan hashes have
different meanings and must remain distinct.

Reuse the existing API01 publisher and [FIFO launcher](harness/fifo_launch.py).
An additional launcher, a copied ledger, a reference heuristic backend, or a
source hash substituted for an executable hash would break those bindings.
Missing runtime facts remain null in this draft. Publishing a placeholder at the
consumer's trigger path could start its observer, so the draft has a distinct
filename/status and must never be copied there. Identity-table work stays with
API01; QLT02 and RES03 independently review the exact PR head. No signed schema,
statistical criterion, selection rule, retry budget or Q1 default changes.

## Current record and hash meanings

The draft's `evaluator_sha256`, executable path, host, runtime receipt and identity
closure receipt are **null / NOT QUALIFIED**. No source-bound production evaluator
or passing runtime receipt has been supplied to this handoff. Source pins include
the adapter, request driver, launcher, FIFO helper, engine/worker implementation,
protocol, vocabularies and Cargo lockfile. A source pin is not a binary pin.

`evaluator_sha256` must be SHA-256 of the actual immutable native `jopedime`
executable used by the qualified engine and its worker. Retain Cargo's exact
artifact path, full command, compiler/target/features, source tree, environment,
artifact size and hash. Verify the running engine/worker uses those bytes; a
test executable or reference smoke adapter does not establish this identity.
`backend_sha256` is separately the reviewed executable
[`production_backend.py`](harness/production_backend.py) bytes. The six plan pins
remain bundle, DOPE, evaluator, vocabulary, protocol and training-family manifest.
`sha256(plan file bytes)` is distinct from `q1.digest(plan)` (canonical JSON).

The fixed roster contains 153 cuts with SHA-256
`ab434d145b8cec60399b35c35a1ce9551a881843e8136b1c757e9c93ae62bd9a`.
[R19](PREREG-AMENDMENT-R19.md) alone permits wide minimum four; wide remains
**4/5 families, below preregistered minimum, descriptive only, no subgroup verdict**.
All other statistical and family-closure requirements remain binding. This draft
does not resolve any of the 36 missing authoritative family identities.

## Qualified-input publication

API01's existing host-local consumer is
`~/jope_plan/evidence/QLT-04/api01-t0710/freeze_plan.py`. Its
`QUALIFIED-INPUTS.json` trigger accepts the following factual fields only after
the associated evidence exists:

| Field | Required evidence |
|---|---|
| `source_commit` | Full actual merged source SHA, ancestor of remote main and containing the required Q1 merges |
| `status` | `QUALIFIED_INPUTS_READY`; never use this value for the draft |
| `evaluator_host` | `xbabe1` or `xbabe4`, with actual OS/architecture/runtime recorded |
| `evaluator_sha256` | Actual native executable SHA-256; 64 lowercase hex, not null or a source hash |
| `evaluator_artifact` | Regular, non-symlink immutable local copy under `api01-t0710/evaluator/`, named `jopedime`, `jopedime-r782` or `jopedime-evaluator`; ELF/Mach-O magic and bytes verified |
| `runtime_receipt` | `{path, sha256}` of an actual source-bound passing receipt |
| `identity_closure_receipt` | `{path, sha256}` of actual authoritative closure proof, bound to the final plan digest |

Keep this draft outside that trigger path. The host-local consumer and publisher
are owned by API01; their observed content hashes are retained in the draft for
review. Recheck their actual implementation when producing the final handoff.

The runtime receipt must bind the exact source, executable, adapter and all six
plan pins, fixed r782 bundle and DOPE, three folds, and the frozen dataset/cut
metadata in plan order. Record actual commands, times, host/runtime, resource
caps, input identities, tests collected/executed/failed/skipped and why. Required
observations include integer `learned_forward_count > 0`,
`executed_qualification_checks > 0`, `failed_qualification_checks == 0`, and actual
`charged_admissions == {heuristic: 24, learned: 24}`. These are required observations,
not counts supplied by this draft. Retain nonempty `execution_logs` with existing
regular evidence paths, SHA-256 and exit 0. Qualify runtime using approved
non-sealed development FIT inputs; listing sealed dataset/cut identities is
metadata validation, not permission to open their rows before Q1 admission.

The separate closure proof must have status PASS, the same source,
`qualification_plan_digest` and `training_family_manifest_sha256`, and measured
integer-zero overlap counts for family, dataset ID, full dataset SHA, train-blob
SHA, aliases, split membership and every additional identity domain used. Resolve
the missing identities through API01's evidence table; do not invent authority
flags or drop families to manufacture closure. Existing M2/M3 and actual admitted
cohort/model proofs in the [engine contract](harness/PRODUCTION_BACKEND_CONTRACT.md)
remain required; a handoff document does not make their outstanding gates pass.

## Runtime and signed dispatch

The original driver, physical persistent ledger, shared queue and qualification
lock remain on xbabe2. Heavy evaluation is only on xbabe1/xbabe4. Record the actual
transport from the original driver to that evaluator, endpoint identity and
cleanup behavior, while preserving the adapter's loopback-only daemon-record
authentication, disabled proxies and redirect refusal. No qualified transport is
asserted here. A guessed loopback URL or untested forwarding route is insufficient.
Keep bearer tokens and credentials out of every receipt and log.

After actual qualified readiness, API01 regenerates the exact merged-source plan
and files at most one `kind=q1` request; ENG signs. The separate existing
`api01-t0250/handoff.ready.json` dispatch record must be `QUALIFIED_SIGNED_READY`
and name the registered clean source checkout, exact plan/backend, signature,
allowed signers, watcher receipt, engine home/URL, new run directory, and
`qualification_inputs` references to the source-bound backend receipt and engine
protocol manifest. The installed protocol manifest must bind the plan digest,
source, all pins and every dataset/cut entry. No draft or signature alone grants
runtime qualification. Preserve the original ledger device/inode and candidate
consumption markers; never make a fresh-host ledger or reset a deadline.

## FIFO ticket schema and admission order

The maintained helper generates a **plain file**, not JSON, named
`<epoch_ns>-<pid>-Q1` in `~/jope_plan/qualification.queue`. `epoch_ns` is an integer
nanosecond enqueue time and `pid` is the live local driver PID. Its exact body is:

```text
epoch_ns=<same integer as filename>
pid=<same PID as filename>
wave=Q1
```

The helper creates it with `O_CREAT|O_EXCL` and mode `0644` (subject to umask).
It resolves a filename collision by increasing `epoch_ns`. Live tickets are
ordered by `(epoch_ns, pid, wave, filename)`; Q1 reads other tickets without
pruning them. Only the launcher may create its live ticket, after admission.
Preparation instead emits `jope.q1-fifo-intent/v1`, status
`prepared_not_enqueued`, `active_ticket: null`; this draft reserves no position.

Admission requires the owner signature, qualified backend/cohort/model proofs,
clean exact merged source, frozen DOPE, unconsumed canonical signed ledger and
explicit finite caps. Then the launcher opens the **existing** regular lock with
`O_NOFOLLOW`, enqueues its ticket, observes the 45-second courtesy delay and
30-second polls, and waits for its oldest-live turn and exclusive flock. It checks
the locked device/inode and repeats admission/hashes after acquisition. Wait cap
is at most 10,800 seconds; the whole-job wall cap is separately explicit and at
most `6*N*600` seconds. For 153 cuts, 918 policy runs and 22,032 charged admissions
are planned; 550,800 seconds is timeout arithmetic, **not an ETA or new budget**.

Use current nice19/CPU≤85% limits, eight Rayon threads, one BLAS/OpenMP thread and
no GPU. The driver inherits the exact lock descriptor. Cancellation/timeout must
clean only its owned process group and ticket, retaining logs and the original
ledger. A started-only candidate remains consumed with unknown outcome; there is
no automatic retry. The durable launcher receipt is `jope.q1-launch/v1`, with
actual `ticket`, source, frozen input hashes, planning, caps and final status.
Preparation, queue position, runtime qualification and Q1 results remain separate.
