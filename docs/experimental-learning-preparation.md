# Experimental learning preparation and comparison

Edition 2 remains a dated publication. These interfaces extend its implementation snapshot;
source support and synthetic CPU proofs do not establish live learning or serving qualification.
The accepted policy is unchanged. No command here grants another experiment window.

## Prepare without starting a clock

Run `python3 deploy/prepare-asm-runtime.py build-workers --source-root REPO
--expect-source FULL_SHA --output CUSTODY/build`. The source must be committed and clean
apart from the coordination ledger. This uses the checkout's existing target, at most four
build jobs, and the host CPU allocation lock. It copies Cargo-reported executables into an
immutable inventory, retaining source hashes, features, compiler messages and the explicit
CLI build-receipt conversion required by panel preparation. Run each command on one shell
line; uppercase operands in this document are paths or identifiers supplied by the operator.

`prepare-context-proof --worker-build CUSTODY/build/summary.json --reference-sources REFERENCES
--encoder-files ENCODER_FILES --resources ORIGINAL_RESOURCES --output CUSTODY/context-proof`
constructs a native proof plan and reports its next command. It does not execute the eight
reference cases or start their clock. The original resource manifest and deadline remain
binding when execution is separately admitted.

`configuration --worker-build CUSTODY/build/summary.json --policy-preparation POLICY_RECEIPT
--asset-preparation ASSET_RECEIPT --output CUSTODY/config` constructs and natively validates
an ASM configuration. Optionally supply `--completed-qualification COMPLETED_RECEIPT` only
for actual retained execution evidence. The validator rechecks all eight executions, source,
worker and asset bindings and recomputes aggregate qualification. Absence of that evidence
produces an explicitly unqualified configuration. The preparation defaults retain the existing
120-second arm, 96 evaluation, four-round, 64-proposal, temperature-one protocol; they are
not an allocation.

After qualified context production, `jope train prepare-request --learning-state LEARNING
--family-registry FAMILY_REGISTRY --vocabulary VOCABULARY --expect-source FULL_SHA
--context-files CONTEXT_FILES --output CUSTODY/corpus` derives comparison and CPU proof
identities from actual admitted records and retained context files. Repeat `--context-files`
for every admitted regression comparison. An optional `--checkpoint CHECKPOINT` binds an
explicit warm start. The command runs the same metadata and evidence checks as
`prepare-inputs`; it performs no fits or optimizer updates.

Its `inputs/worker/inputs.json` is the worker manifest. Its `inputs/receipt.json` is the
controller preparation receipt. Pass both to `jope train plan` using `--inputs` and
`--preparation-receipt`, alongside the existing state, Python, source and learning-store
arguments. Controller proof bundles never enter the projected worker directory. The worker
allowlist binds every corpus, context, recipe, fitting-input, encoder, vocabulary and optional
checkpoint file. It rejects unused contexts and declared-but-unreachable artifacts, undeclared transitive references, cross-split
families, forbidden authority and extra files before model loading. Outer test, sealed
meta-test and outside-selection evaluation are excluded domains. The historical engine
training route is not admitted by this experimental interface.

## Resume exact updates

Campaign workers use strict resume: compatible nonempty optimizer state is required before
updates. Each successful update atomically commits model, moments, scheduler, scaler, RNG,
sampler cursor, counters and loss accumulators. Recovery uses a stopped attempt with the same
plan, parent and original global/category windows. If the controller died before recording
its outcome, recovery requires the native invocation’s authenticated drained-subgroup
observation. A later native invocation can record that observation only after the old unit
is inactive and its owned subgroup is absent. Unmeasured elapsed usage stays unknown;
rejected attempts cannot seed recovery. A retained last-update checkpoint also
covers interruption after the final batch but before epoch/controller completion. Warm start
remains the separate `init_from` operation; it is not optimizer continuation.

## Compare fresh teachers with replay

An existing campaign must already have a completed current parent and original windows.
`jope train compare-plan --state STATE --task regression --fresh-inputs FRESH_INPUTS
--fresh-preparation FRESH_PREPARATION --updates UPDATE_COUNT` freezes one comparison beneath
that campaign. Preparation cannot create a global or category window. A second round needs
its own explicit design; this interface does not replace an existing round.

`jope train compare --state STATE` runs or resumes that frozen comparison through the native
supervisor. Native admission recomputes current fresh-teacher evidence and separately
revalidates replay against intact accepted historical events and retained measurements.
A changed current pointer does not erase an eligible historical replay record. Sealed or
unaccepted records remain forbidden. The same qualified context consumer checks both inputs.

Both siblings start from one immutable parent model **and optimizer**. The explicit corpus
transition permits changed input provenance and sampler only; architecture, objective,
vocabulary and numerical execution settings must agree. Both commit the declared update
count. The mixed branch samples exactly 25% fresh and 75% replay examples with replacement;
the control reads replay examples only, including during worker metadata validation. The count times parent batch size must be divisible
by four. Family roles must remain disjoint. Both retain the parent's development rows;
fresh development rows are not used to train or choose a sibling. A common replay reference
freezes replay reward targets, ranks and routing winners before either branch trains. Its
derived train/development hashes must match the parent’s provenance. Fresh scores on the
same comparison are measured against that frozen incumbent; replay targets do not change.
Independent verification rederives exposure and supervision from retained corpora and
update checkpoints.

Receipts record consumed example identities, unique/duplicate examples and families,
nonpadding token counts, optimizer updates, parent identities and subprocess wall cost.
CPU/GPU active seconds remain explicitly unknown when unmeasured. Per-update checkpoint
I/O is charged to the same category window; its real-model cost is not yet measured. Skipped updates, exhausted
time, missing branches and failed workers produce an inconclusive comparison. Partial work
resumes its branch checkpoint under the same cutoff; neither branch advances the campaign's
parent head. Consumption equality is not evidence of quality improvement. The result makes
no serving promotion and reports quality improvement as unknown.

## Evidence and remaining operational gate

The maintained CPU fixtures exercise all three task heads, actual parameter changes,
optimizer reload, exact interrupted trajectories, excluded-artifact invariance, explicit
corpus transitions, fresh exposure and unchanged deadlines. `deploy/ci/training-proof.py`
requires the named tests to execute; skipped or missing cases fail the gate. Fixture validation
metrics are deliberately synthetic and cannot establish a live model gain.

The operational order remains training, product integration, warm demonstration, winner
export, then mandatory trained-regression consumption. A real regression loop still requires
preregistered eligible data from the required independent families, source-bound context
qualification and bounded trained-proposal-to-fit consumption evidence. Historical full-main
failures remain diagnostics, separately from exact-head CI. Insufficient eligible data or an
exhausted original allocation is **inconclusive**; this continuation does not authorize a new
GPU allowance, a replacement dataset chosen after results, a new deadline or promotion.

An integrity failure is always a refusal. Only an intact input manifest with an unavailable
classification category can use the explicit labelled synthetic fallback, and that fallback
does not count as complete requested real-data coverage.
