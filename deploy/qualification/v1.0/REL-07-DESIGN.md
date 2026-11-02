# REL-07 qualification deliverable

**DRAFT — no release qualification or OWNER release approval is claimed.**

Produce the final receipt and release notes for spec §§11 and 14 and work-plan
REL-07. Runtime identity stays frozen at
`ccaf14b210b3b55a350218e60f6b49ef76524cf3`; receipt tooling is the separately
reviewed main implementation, including #1724 and #1737. Later main commits do
not silently move the candidate or transfer old gate results.

Use the maintained `gates_receipt.py --scaffold` assembler with exactly one
receipt from each of the publisher's 28 final gate directories. Supply the full
Linux artifact, explicit RC4 candidate manifest, full runtime source SHA,
`--m3-waiver-ruling` and `--release-version v1.0.0`. The assembler must validate
27 PASS results and the measured M3 FAIL with its separate `WAIVED_V1_0_0`
disposition. Its integer guard, other-clause checks, ruling hash and v1.0.1
expiry remain enforced. D2 binds the same candidate's fused artifact rather
than borrowing the full Linux hash. See [RECEIPT-ASSEMBLY.md](RECEIPT-ASSEMBLY.md).

Generate `qualification_receipt.json` only after that admission succeeds on the
completed publisher run using the versioned provider inputs v2 or later. Preserve
the assembled gate rows, counts, waiver and per-gate artifact hashes verbatim.
Attach provenance from the same run: execution commands and exit statuses,
machines, source/tool identities, input manifests, output receipts/logs and their
SHA-256 pins, plus the separately recorded Q1/default decision. Recheck input
hashes around assembly; a changing run cannot yield the final deliverable.
OWNER release approval and aggregate signing remain separate lifecycle steps.
The existing CPU-prestage schema must not be used to label this final receipt.

Rejected alternatives: projecting a status table into PASS, combining historical
successful subsets, turning M3 into PASS, omitting D2's fused binding, or committing
a partial receipt. Those approaches lose execution, source or artifact identity.

Main risks are a live partial controller run, missing A5 inputs, unavailable
machine evidence, and an unresolved Q1/default decision. Keep drafts explicit;
do not invent a decision, promote a source-only proof or count zero/skipped work
as acceptance. Disclose the M3 waiver and known limits in
[release_notes.md](release_notes.md). No runtime, model, threshold, protocol,
credential or frozen application contract changes are proposed.

Validation: first preserve the partial-run assembler refusal as NOT_QUALIFIED in
external custody. After final gate admission, verify all receipt/provenance pins
and run `just hygiene`, `just lint-paths` and `just audit`, comparing any failure
with unchanged main. The later t0115/t0145 directions authorize a draft
preparation PR now; the final generated receipt still requires the completed
gate condition. Keep the maintained assembler intact because Q1 pins its
reviewed source bytes. ENG owns the gated merge and signer request. No GPU or
training is required for this delivery work.
