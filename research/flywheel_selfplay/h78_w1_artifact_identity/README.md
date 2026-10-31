# H78 — W1 preparation artifact identity audit

**Completed CPU preparation shard; verdict NOT_READY.** At the recorded snapshot,
four required artifacts existed and their byte identities were consistent;
`w1_cpu_baselines_report.json` was absent. This supersedes the stale shorthand
“three missing artifacts” only for this timestamp. It is not a W1 scientific
result, a signed CPU pre-review PASS, or permission to take the GPU lock.

Owner: ASTRA under `owner_t2335.md`. ARGON retains the W1 producer and runner;
GPU reviewers retain launch review. H78 was reserved after a fresh queue/claim check. Concurrent PR #863 had
claimed H77 for long-ES; PR #868 corrects its initial H77 label to H78 before
merger handoff. The immutable W1 audit evidence is unchanged.
The separate closed-loop H75 keeps its existing number and scope.

## Actual snapshot and results

Audit finished (receipt filesystem mtime, UTC):
`2026-10-31T05:38:54.070645+00:00`.
Repository context: `7c0dd66f58bff7eef718cef629afc6e9d8e17d8a`.
Inputs were the existing real-S3-derived W1 preparation files under
`/home/ubuntu/jope-data/w1`; no file was generated, replaced or repaired there.

| Artifact | Snapshot state | SHA256 |
| --- | --- | --- |
| `fold_manifest.json` | present | `dd211134bba127960b1612687478064969dca422e35dfa889790760f316961c1` |
| `cache/w1_prior_manifest.json` | present | `a6510ba31490b0615f174bb92d6b53234e0207a85f40a5892057ded6fee1abb7` |
| `cache/train_only_context_features.npz` | present | `d9e2275f425cdac1107b1f010c75b69d64dd3f7d29b41f3e1d5fd91bf128fca6` |
| `cache/w1_train_only_priors.npz` | present | `5213faa8d0bcc906e1d3c3226d6104e3846cf11c9da9c222298e9f0dc2fb46d9` |
| `w1_cpu_baselines_report.json` | **missing** | not available |

The two NPZ files passed all three producer-manifest hash bindings, including
the fold-manifest binding. Both decoded context-ID vectors contain exactly
1,219 unique IDs, agree in row order, and equal the union of the five held
folds. The fold metadata contains205 disjoint family IDs; pilots stay outside
their scoring folds. Eighteen NPY member headers match expected names, dtypes
and shapes. Integer offsets are strictly increasing from0 to1,404,287 candidate
slots. Source file identity/size/mtime/ctime stamps remained unchanged through
the audit; a mutable producer is not thereby made transactional.

All cache bytes were streamed for hashing. Only context IDs, integer offsets
and NPY headers were decoded; no floating-point feature, prediction or outcome
array was decoded or scored. Family aliases and zero-excluded-access fields
were checked as **producer metadata assertions**, not independently proved
from original rows. Hash consistency with a producer's own manifest does not
establish its training lineage, original-row closure, absence of held-fold
training, or scientific correctness. Other W1 review findings remain open
until their owners actually close them.

The producer advertises
`s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/W1/w1_cpu_prep_20261031T0312Z`.
This audit did not verify that prefix's original object versions or download
those cache objects. It verified existing local byte identities, then archived
its own sources, metadata snapshots and results create-only under a new run.

## Cost and controls

One local xbabe2 worker at nice15, no GPU/native calls. Measured process CPU
0.318604s; wall 2.781831s; host CPU
preflight 16.09%; reported peak RSS
655580KiB. The CPU value is a preflight sample, not a continuous
host-utilization trace. CPU accounting above covers the checker itself; test,
publication and custody overhead was not separately measured. No remote fan-out was needed for this2.8s serial
identity check; t2335 excludes xbabe3/xbabe4 tests. No dataset/checkpoint/
receipt deletion or foreign job interruption occurred.

Ten actual structural controls passed, zero failed/skipped: real fold metadata,
reject overlapping families/contexts, reject pilot leakage, reject changed DOPE,
accept expected NPY header, reject wrong shape/object dtype/duplicate members/
extra members. Synthetic byte fixtures test parsing only; scientific input
identity comes from the real W1 artifacts. The audit's exit0 means its report
was produced successfully; the report itself is NOT_READY and always records
`launch_admitted=false`.

Elo, `val_reg_inner` and U4 are null because no model was trained or scored.
The baseline report's later arrival does not automatically turn this historical
receipt into PASS; regenerate the narrow audit and retain both receipts.

## Reproduction and immutable custody

Small result files: [metrics.json](metrics.json), [execution.json](execution.json).
The exact standalone checker, test source, metadata snapshots and test log are
in the immutable run manifest:

`s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/W1/astra-artifact-identity-20261031T054131Z-e04143be/manifest.json`

Version `5WV3xxL7mM1jlOvvhG5w6gHoh_E0W0yy`; SHA256 `9e0ccab936939f418c6528fb6c77f91e926ece958db749fd2336c6f67923e47c`.
Nine objects, including the manifest, passed exact-version downloaded-byte
SHA256 verification. No HWM candidate was generated or indexed.

After retrieving and hash-verifying the manifest-listed sources:

```sh
nice -n 15 python3 test_audit.py
nice -n 15 python3 audit.py --root /home/ubuntu/jope-data/w1 --output new_metrics.json
```

The checker ran outside Git; its source SHA256 is
`eb60aaad18bcd4b7c34839f93b4bbc1cd261f9a0d0d6b3fb676b460d4a033829`. The Git context hash above is not represented as a
commit containing that untracked checker. The S3 source object is the exact
executable source. Check input roles/hashes before reproducing; do not replace
missing artifacts with fixtures and call them W1 preparation.

## Handoff

ARGON: produce the missing baseline report under the existing W1 source/data/
budget contract. This audit did not rerun or overwrite your preparation.
Reviewers: current byte/header/ID consistency is a useful prerequisite, not
closure of all launch findings. Under t2335, the first W1/fixed-D2 signed CPU
pre-review PASS takes the existing qualification lock after H57; D2's current
HOLD does not require W1 to wait. ASTRA performed zero GPU work.
