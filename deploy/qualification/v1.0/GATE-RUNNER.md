# REL-07 gate command harness

`release-gates` executes an explicitly supplied command for each of the 28
release gates. It uses the existing `cpu_gates.py` process/admission controls and
`gates_receipt.py` receipt adapter. It does not supply missing gate implementations,
verify release approval, request a signature or sign anything. Q1 is separate.

For the retained CPU-prestage inputs, run the metadata-only path:

```sh
python3 -B deploy/qualification/v1.0/release-gates \
  --cpu-prestage "$PRESTAGE_RECEIPT" --output "$NEW_EVIDENCE"
```

This executes zero gates, emits 28 NOT_RUN rows and no artifact binding, and
reports `dry_run: true, all_pass: false`. Exit 0 means the dry run succeeded;
its receipt cannot authorize signing. The t1930 inputs are incomplete CPU
diagnostics, not a signed full-product candidate.

## Execute a pinned command manifest

```sh
nice -n 10 python3 -B deploy/qualification/v1.0/release-gates \
  --repo "$CLEAN_PINNED_CHECKOUT" --artifact "$SIGNED_BYTES" \
  --manifest "$GATE_COMMANDS" --output "$NEW_EVIDENCE"
```

When every assigned provider has merged into the pinned source checkout, the
runner can discover the provider modules and build the same 28-command manifest
it consumes above:

```sh
nice -n 10 python3 -B deploy/qualification/v1.0/release-gates \
  --repo "$CLEAN_PINNED_CHECKOUT" --artifact "$SIGNED_BYTES" \
  --providers --provider-inputs "$INPUT_BINDINGS" --output "$NEW_EVIDENCE"
```

For candidate-free inventory checks after provider merges, use dry discovery:

```sh
nice -n 10 python3 -B deploy/qualification/v1.0/release-gates \
  --repo "$CLEAN_SOURCE_CHECKOUT" --providers --dry-discovery
```

This reports the exact source SHA, found provider IDs, missing provider IDs, and
counts. It accepts no artifact or output path, executes no provider, and writes
no gate evidence. A successful inventory report may still be incomplete.

Five providers require explicit external inputs. M5 binds the R21 RNK-10
default-off receipt and its independently computed SHA-256; A5 needs a candidate-bound named-client transcript
manifest and its independently pinned SHA-256. D1 needs the candidate-bound
macOS receipt and its SHA-256. M2 needs its golden directory/index SHA-256,
frozen bundle directory and a new external parity-report directory. M3 needs the
complete full-input manifest and its independently pinned SHA-256;
without it, the bounded source controls cannot qualify the complete M3 criterion.
Supply these through `--provider-inputs "$INPUT_BINDINGS"` with
`--providers` (including `--dry-discovery`). The regular, non-symlink JSON file is:

```json
{
  "schema": "jope.rel07.provider-inputs/v1",
  "gates": {
    "M3": {
      "full_inputs": "/absolute/path/to/complete/M3-inputs.json",
      "full_inputs_sha256": "<replace with the independently pinned full-input manifest SHA-256>"
    },
    "M5": {
      "default_off_receipt": "/absolute/path/to/RNK-10/DEFAULT-OFF.json",
      "default_off_receipt_sha256": "<independently computed lowercase SHA-256>"
    },
    "A5": {
      "manifest": "/absolute/path/to/candidate/transcript-manifest.json",
      "manifest_sha256": "<replace with the independently pinned 64-character lowercase SHA-256>"
    },
    "D1": {
      "macos_receipt": "/absolute/path/to/candidate/macos-receipt.json",
      "macos_receipt_sha256": "<replace with the independently pinned macOS receipt SHA-256>",
      "macos_artifact_sha256": "<replace with the Darwin package SHA-256 from the signed release manifest>"
    },
    "M2": {
      "golden_dir": "/absolute/path/to/qualified/R5/packet",
      "golden_index_sha256": "<replace with the owner-accepted full packet index SHA-256>",
      "parity_bundle_dir": "/absolute/path/to/frozen/bundle",
      "parity_report_dir": "/absolute/path/to/new/external/M2-report"
    }
  }
}
```

The M5 receipt uses `jope.rnk10.default-off/v1` and binds the tested source SHA,
the `PASS` result, SHA-256-pinned shipped config and bundle with heuristic
default and learned guidance disabled, identical
serving-admission hashes with the ranker absent and present-but-disabled, and
identical hashes from two default-path S2 runs. M5 validates those three
non-waived R21 checks and candidate identity; it does not execute the waived
learned-guidance causal replay. The `release-command-scaffold.json` stays
`NOT_EXECUTABLE`; RC2 materialization supplies the candidate and receipt paths
and pins to the generated `jope.rel07.gate-commands/v1` manifest.

Discovery reports `unbound_providers` and `unbound_provider_count` separately from
missing modules. `complete` requires both inventories to be empty. Execution
discovery refuses unbound providers before writing a runnable manifest. It adds
M5's R21 receipt and hash pin; D1's receipt, receipt pin, and platform-artifact pin;
M2's four input/report arguments; M3's full-input manifest/pin arguments; and A5's
two explicit environment bindings to each provider's argv without a shell. The
frozen command-manifest hash covers them. The restricted
child environment still discards inherited provider-input variables. Bindings
cannot replace commands, gate/source/artifact identities, resource settings or
credentials. Unknown fields, relative paths and malformed SHA-256 pins refuse.
Providers retain their full input, byte-hash and qualification checks; configuring
paths does not establish that the evidence exists or passes.
M2's report directory must not already exist; choose a fresh external path for
each execution. A configuration containing only A5/M5 remains accepted as a
partial configuration, but discovery stays incomplete until D1/M2/M3 are bound too.

The maintained runner already writes the watcher-format `gates_receipt.json`.
The older external `bind-rel07.py` adapter consumes validated `gate_results.json`
with its `signed_bytes_sha256` field; it is not a gate executor. A partial or
failed synthetic receipt must be rejected by that adapter. Keep controller
dry-run artifacts and receipts outside signing request directories: synthetic
refusal tests do not authorize signing or qualify a candidate.

Execution discovery requires a clean checkout at `origin/main`, a regular
non-symlink candidate, and one regular `providers/<gate>.py` file for each
release gate.
The generated manifest is retained at
`$NEW_EVIDENCE.gate-commands.json`; its exact bytes are hashed into
`execution.json`. Missing providers refuse before creating gate evidence, so a
partial inventory cannot become a runnable 28-gate manifest. D2 discovery uses
the owner's size-budget policy: artifact size determines its PASS/FAIL status;
speed or latency measurements may be reported as informational data, but never
affect the verdict. An explicitly owner-supplied complete manifest remains
supported through `--manifest`.

The JSON manifest has schema `jope.rel07.gate-commands/v1`, a full 40-character
`source_sha`, the exact file's `artifact_sha256`, and a `gates` array. Supply
exactly one entry for every ID exported by `gates_receipt.RELEASE_GATE_IDS`;
unknown, missing or duplicate IDs are refused before execution. Each entry has
exactly `id`, `argv` (a nonempty array of strings) and `timeout_seconds` (finite,
positive and at most 3600). Commands are executed as argv, without a shell.

The checkout must be clean at the manifest's source commit. The artifact must
be a regular, non-symlink file with the declared hash. The evidence directory
must be new and outside that checkout. Source and artifact identities are checked
between commands and before publishing their results. Drift stops later gates;
unstarted gates remain NOT_RUN. A failed command or invalid child result retains
its actual outcome and allows the remaining commands to run.

Each command receives the restricted CPU environment from `cpu_gates.py`, with
at most four build jobs and its own evidence/scratch directory. CPU/disk admission
is checked before starting each command. Timeouts and interruptions stop only the
invocation's process group. This coordinator does not qualify native platforms
or CUDA; their gate providers still need the actual corresponding evidence.

The additional environment variables are:

| Variable | Value |
| --- | --- |
| `JOPE_REL07_GATE_ID` | Current release gate ID |
| `JOPE_REL07_ARTIFACT` | Absolute path to the supplied bytes |
| `JOPE_REL07_ARTIFACT_SHA256` | Frozen artifact hash |
| `JOPE_REL07_SOURCE_SHA` | Frozen tested source commit |
| `JOPE_REL07_RESULT` | New child JSON result path |

The provider writes a regular, non-symlink result with schema
`jope.rel07.gate-execution/v1`, matching `id`, `source_sha`, `artifact_sha256`,
`status: "PASS"` and a `counts` object. Counts contain all six integer fields:
`collected`, `executed`, `passed`, `failed`, `skipped`, `excluded`. PASS requires
`collected == executed == passed > 0`, with failed/skipped/excluded all zero,
exit 0 and a retained regular command log. A provider's measured assertions or
test cases count as checks; an empty suite never passes. Providers remain
responsible for implementing each spec §14 criterion against these bytes.

The harness retains `execution.json` (commands, source, manifest/runner hashes,
machine, times, exit/timeout outcomes, counts and log/result hashes), each gate's
log/result, and `gate_results.json` for the existing receipt adapter.
`gates_receipt.json` has the watcher's exact 28 IDs and top-level/per-gate
`artifact_sha256`. It contains all PASS only after every gate satisfies the
execution contract. Execution returns 0 only in that case; incomplete or failed
runs return 1, and interruption returns 130. `release_qualified` stays false:
the harness does not establish final release approval.

## Focused acceptance

```sh
python3 -B deploy/qualification/v1.0/test_release_gates -v
```

This manual suite executes harmless fixture children with synthetic bytes,
checks signer-format compatibility and preserves failed/missing/zero/skipped,
wrong-identity, malformed and timeout cases. It checks continuation, admission,
interruption and first/last-gate drift. It is not the product gate campaign or
automatically collected native/CI coverage. Fixtures use the lane's target or
`JOPE_QUALIFICATION_TEST_TMP`, never `/tmp`.


The tracked `common.py` and `m3_full.py` modules are explicitly registered helper
modules, not extra gate IDs. Discovery requires regular non-symlink helpers and
continues to reject unknown `.py` modules. All 28 gate IDs remain mandatory.
See [PROVIDER-DISCOVERY-REPAIR.md](PROVIDER-DISCOVERY-REPAIR.md) for the reproduced
prestage failure and regression scope. Binding M3 inputs never changes its
comparison criterion, historical failures or candidate verification.
