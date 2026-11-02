# Frozen r782 native evaluator entry

Main has a signed frozen-r782 loader and a trusted engine admission API, but
ordinary `jopedime serve` only installs the embedded r882 proposal closure.
The maintained r782 Q1 control therefore cannot yet qualify the real CLI engine.

Add `jopedime research serve-q1 --bundle-dir PATH` behind the existing internal
`research` feature. Authenticate the exact frozen r782 release with the existing
loader before opening an engine home. Then use the existing production engine,
loopback listener, owner-only daemon record, request admission and native worker
from this same executable. Keep ordinary startup and default guidance unchanged.
On orderly shutdown report the measured neural-forward count from the admitted
ranker. The activity report does not grant quality or Q1 launch authority.

Verification: reproduce missing-command refusal on the exact parent; exercise
CLI parsing and refusal before home creation; run the real executable on an
approved nonsealed development FIT dataset with actual source/binary/adapter and
protocol pins. Both policies must charge 24 attempts, preserve three fold scores
per completed attempt, and learned execution must record positive actual forward
count. Retain immutable executables, original requests, durable receipts, logs,
compiler/configuration, data identity and transport checks. Signature, final plan
closure, remaining M2/M3 gates and the original Q1 ledger/lock remain separate.
No source receipt, test host, synthetic pin or unsigned request qualifies Q1.


## D5 evaluator build identity (2026-11-01)

ENG T1440 D5 selects full executable SHA-256
`76721923b663dd2ad2f9fe82c787791e2b97ec32426efa5eab8ec81f146ab0a6`
(107,739,816 bytes, native Darwin arm64), built from source
`2317d4d168b9dc6390916d4874cbe69db734c4c8`. The maintained
[pin](evaluator-pin.json) binds the exact [D3 recipe](evaluator-repro/RECIPE.json)
and both independently produced, byte-identical build receipts:

- [RES03 REPRO.json](evaluator-repro/res03-REPRO.json), SHA-256
  `a60d7790ef9efe381e65adc5dd010f4af0b020dd6696dd8f7c1c948754abfc5b`.
- [MCP01 REPRO.json](evaluator-repro/mcp01-REPRO.json), SHA-256
  `d1f6b96f1c2121e8def0a0e02b9fa6d3a58da46f26ae375b1cb26f0a749eda41`.

The recipe retains Rust/Cargo 1.97.1, the development profile, no default
features, `research` (which enables `engine`), `target-cpu=native`, all three
physical-path remaps, source epoch 1793563220, incremental compilation off and
line-table debug information. Its `/private/tmp/jope-eval-src` paths record the
owner-authorized D3 build location on macOS; they do not authorize other temporary
checkouts or new build directories. No executable bytes were stripped, patched,
normalized or re-signed for comparison. Full reproduction evidence remains under
`~/jope_plan/evidence/QLT-04/repro-t1410/{res03,mcp01}/`.

Qualification plans and explicit draft evaluator arguments reject any other
hash, including the previous `40bc4318...` executable. Drafts without an evaluator
remain unqualified. The new pin establishes **build identity only**. GOV03 must
rerun runtime qualification on these exact new bytes, including identity,
M2/M3 and plan admission, under the existing qualification lock. Old runtime
evidence is not inherited. This change supplies no signature, protected-input
read, Q1 attempt or runtime qualification; the original ledger and budgets remain.
