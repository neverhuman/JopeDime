# Jankurai audit remediation — 2026-10-21

This record describes the PR #395 remediation batch and its remaining boundaries.
It does not certify an audit cap as removed. Final audit results, review and CI
belong in the [PR description](https://github.com/neverhuman/JopeDime/pull/395)
and external receipts bound to the audited source; inserting a computed score
here would itself change that source.

## Baselines and dated evidence

The supplied Jankurai 1.7.5 report at
`18b671e90102df16de8b0cd95f93fd4378c911ef` scored **51/100**, with **591 findings
and 21 caps**. A fresh tracked-product baseline at
`98af182c412dc4e8364d6516f477dbf4935b99b6`, covering **5,103 tracked files**,
scored **51/100**, with **594 findings and 21 caps**. These differ in source
and inventory; subtracting the finding counts does not measure this repair's
benefit. Neither baseline passed the gate.

The original [blob verification JSON](../performance/blob-verification-review.json)
retains its source hashes, 93 passing tests and seven-pair warm 16 MiB component
microbenchmark: median 41.46 ms versus 14.24 ms, **2.91×**. It is dated evidence,
not a new measurement of the integrated revision or training throughput. Its
outside-review requirement records the former policy. The current owner policy
permits substantive same-agent review bound to the exact head; historical JSON
has not been rewritten to imply a different past result.

Root subsequently recorded **95 affected dataset/model tests and strict Clippy
passing at `83800605480677a59d1c300689df6f287da6ca91`, before the dependency
upgrade**. Those results do not qualify the new lock graph. Dated helper evidence
comprises **95 CI/security/browser helper controls** (7 security, 6 browser-dependency,
22 rendered-receipt and 60 Gitleaks controls), **28 vendor-custody
controls**, **14 isolated Python adapter boundary tests**, and **5 Vitest tests**.
The Linux browser run passed **6 tests with 16 screenshot/axe/geometry checks**,
using Node **26.1.0** against a provisional dirty tree based on `8380060`; this is
synthetic intercepted observer traffic, not final exact-head CI, a live service
assessment or scientific qualification.

## Implemented mechanisms

[Blob hashing](../../crates/jope-datasets/src/hash.rs) now separates binary
content verification from CSV parsing, bounds retained header bytes, and shares
one content reader. Dedicated admitted read threads avoid the shared Tokio
blocking-pool deadlock when a destination-lock waiter occupies its last worker.
The actual reader retains its permit through receiver cancellation and runtime
shutdown. Blocking storage can still retain capacity; model publication still
materializes a verified snapshot, and streaming download writes remain separate
performance debt. See [the detailed review](../performance/blob-verification.md).

The [security lane](../../tools/security-lane.sh) preserves the optional-feature
graph and makes advisories, bans, licenses and sources blocking. The actual strict
all-feature dependency check passed after fixed direct TLS/root/ruzstd/ZIP
updates, disabling unused postcard defaults, and the bounded upstream patches
below. No advisory ignore or score-floor reduction was added; the `bzip2-1.0.6` license
exception is scoped to `libbz2-rs-sys =0.2.5`. Gitleaks controls now exercise realistic
secret-shaped values as well as integrity hashes. The frontend lock graph is
included in dependency scanning. This dependency-check pass does not stand in
for the entire security lane or final CI.

[Canonical local/CI entry points](../../ops/ci/README.md), path-classifier parity,
concurrency and fail-closed aggregate receipts preserve the existing 11 manual
jobs. The [rendered test suite](../../web/training-dashboard/e2e/observatory.spec.ts)
checks desktop/mobile layouts, keyboard/dialog focus and zero axe violations;
CSS contrast and named scroll regions were repaired. Its
[receipt verifier](../../deploy/ci/rendered-ux-receipt.py) binds source, browser
checkout, workflow event, run and attempt separately, requires nonzero
unskipped/unflaky success, and hashes the report and
complete screenshot/geometry/axe evidence. Browser native dependencies are
pinned and loaded from 107 signed Ubuntu packages into the job's owned writable
evidence area, without a host installation. The pinned immutable tracked-source
audit wrapper records its Git archive manifest and source SHA.

The [Python adapter](../../crates/jope-bench/adapters/README.md) rejects duplicate
JSON fields, nonfinite inputs and excessive dense-input/probability allocations
before model construction. Its pickle reload remains a trusted private benchmark
boundary; SHA-256 integrity does not make attacker-controlled pickle safe.
Rust changes bound corpus copy reads, validate CUDA discovery/loading inputs,
handle poisoned optional probing explicitly, and document concrete native
mapping, descriptor, lifetime and syscall invariants, including expired-deadline
refusal for empty sealed payloads. Source arguments do not establish GPU
execution qualification. A separately bounded emitter component probe retained
all 60 value-bit and 30 fault outputs while stripping debug information; its
84.51% binary-size reduction is not full evaluator qualification.

## Dependency source custody

[PROVENANCE.json](../../vendor/security/PROVENANCE.json) records **20 original
crate archive checksums**, official restoration URLs, and all **797 retained
files (12,750,981 bytes)**. All files remain below 5 MiB. Only 24 historical
`tch/third_party/pytorch/Declarations-*.yaml` offline generator inputs were
omitted; each original path, size and checksum is retained. Restore by fetching
that crate's recorded official archive, verifying its archive checksum, and
extracting the listed paths for checksum comparison. Generated bindings, build
inputs, Rust/C++ sources and licenses remain present.

Twenty normalized dependency manifests replace `paste` with the maintained
`pastey` package under the existing alias, or select ZIP 9.0.5. Of 553 retained
Rust/C++ source files, **549 are byte-identical** to their archives. Four nonmath
ZIP I/O adapters in Candle `npy.rs`/`pickle.rs`, tch `tensor/npy.rs`, and
`torch-sys/build.rs` handle the new fallible ZIP names/options API. Numerical,
tokenizer and generated FFI kernels were retained unchanged. ZIP container bytes
may differ even when names and tensor values agree. Owning rebaseline and the 13 historical numerical fixtures qualify the graph
separately; source preservation alone cannot establish numerical equivalence.

The [vendor guard](../../deploy/ci/verify-vendor-security.py) checks every retained
size/hash, exact inventories and all 20 crate identities, refusing missing/extra
files, unsafe or duplicate paths, nonregular files and directory/file symlinks.
Its 28 controls and actual imported-tree check passed. The trust anchor is the
Git-reviewed provenance; simultaneous malicious replacement of source and its
reviewed pins is outside this checksum check.

## Disposition of all 21 baseline caps

Rows distinguish an implementation or source review from a measured cap change.
No row asserts that a fresh full audit removed its cap.

| Baseline cap | Implemented repair or source evidence | Remaining boundary |
| --- | --- | --- |
| `no-security-lane-on-high-risk-repo` | Canonical blocking security entry point; strict all-feature dependency checks pass locally. | Final full lane and exact-head receipts establish integrated execution. |
| `python-direct-product-truth-or-db-ownership` | Runtime map identifies Rust product ownership and the trusted external Python adapter. | Python control/research ownership remains architectural debt; no mass relabeling. |
| `no-secret-or-dependency-scanning-in-ci` | Gitleaks controls, strict advisories and lock-graph frontend scanning are wired. | Scanner configuration and all declared CI execution still require head-bound evidence. |
| `non-optimal-product-language-found` | Explicit product, adapter and research ownership documented. | No language migration or language-ratio claim. |
| `too-much-python-in-product-surface` | Active adapter input/allocation defects repaired. | Existing Python remains; frozen histories are preserved. |
| `vibe-placeholders-in-product-code` | Current unsafe/native assumptions documented where inspected. | Marker inventory includes unresolved active and dated/frozen material. |
| `fallback-soup-in-product-code` | Optional GPU probe poisoning and unsafe input paths receive explicit handling. | Intentional algorithmic alternatives and other fallback debt remain. |
| `future-hostile-dead-language-in-product-code` | Current runtime navigation added without rewriting historical records. | Dated markers and unfinished capabilities remain visible. |
| `severe-duplication-in-product-code` | Dataset/model duplicate content reader consolidated. | Generated and hash-pinned historical copies remain; no repository-wide deduplication claim. |
| `missing-rendered-ux-qa-lane` | Six real Linux browser tests, 16 visual checks, strict receipt/aggregate validation. | Local provisional-source proof is not final hosted execution or live UX qualification. |
| `false-green-test-risk` | Flagged source is embedded in a content-addressed historical JSON fixture; nonzero receipt checks added for current lanes. | Frozen bytes remain; this contextual finding was not suppressed. |
| `input-boundary-gap` | Ledger SQL uses prepared placeholders; adapter parsing/allocation and corpus-copy boundaries hardened. | Source review is not a claim that all product input paths were audited. |
| `release-readiness-gap` | [Release entry point](../release.md) routes to signed host release and rollback procedures. | Documentation does not publish or qualify a release. |
| `missing-agent-readable-docs` | [Runtime/ownership map](runtime-ownership-20261021.md), canonical entry points and this cap record added. | The architecture target remains dated; operational claims require current receipts. |
| `streaming-runtime-drift` | The flagged `xadd` substring is in `UnixAddr`, a Unix descriptor-transfer test, not Redis streaming. | Existing SCM_RIGHTS controls remain; no detector-evasion rename. |
| `rust-bad-behavior` | Bounded corpus copy, CUDA input validation, native safety arguments and dedicated hash-worker regressions. | Unsafe/unwrap findings outside reviewed paths remain; hardware and new dependency semantics need qualification. |
| `python-bad-behavior` | Strict bounded adapter input and 14 isolated boundary tests. | Trusted pickle and isolated adversarial tests remain; no arbitrary-pickle loading guarantee. |
| `ci-bad-behavior` | Shared entry points, classifier parity, preserved queue concurrency and source/run/attempt-bound evidence. | Final 11-job execution and receipt aggregation remain distinct from helper controls. |
| `git-bad-behavior` | Flagged broad staging/tag actions were reviewed in isolated hostile test repositories or dated recovery prose. | Negative controls are retained; no broad Git exception. |
| `release-bad-behavior` | Mutable tag finding is an intentional release negative test; publication path remains guarded. | No release asset or tag was changed by this remediation. |
| `repo-rot-bad-behavior` | Ownership/navigation clarified; versioned wire contracts and frozen lineage retained. | Numeric/schema version names do not establish dead code; unresolved maintenance debt remains. |

## Reproducing and locating evidence

Use the [CI runbook](../../ops/ci/README.md) for the pinned environment and full
lane commands. Focused read-only custody checks are:

```sh
rtk proxy python3 -B deploy/ci/verify-vendor-security.py --root .
rtk proxy python3 -B deploy/ci/test-vendor-security.py
```

`rtk proxy bash tools/security-lane.sh` runs the complete security lane;
`rtk proxy bash ops/ci/jankurai.sh /path/to/new-report-directory` runs the canonical full audit. A nonzero gate
or missing tool is a failure, and a skipped run provides no pass evidence.
The external evidence root is
`/home/ubuntu/jope-evidence/jankurai-audit-remediation-20261021/`:
`admiral/baseline395-product/` holds the fresh baseline;
`captain-python-ui/dependency-vendor/` holds independent archive restoration;
`captain-python-ui/vendor-guard/` holds custody controls;
`admiral/browser-qualification/application-run/` holds the Linux rendered proof.
The PR should link final source-bound receipts and substantive exact-head review.
