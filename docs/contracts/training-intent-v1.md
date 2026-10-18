# Training intent and immutable input closure v1

The I12 preflight consumes the I11 allocated `jope.training-intent/v1` JSON projection. It binds a parent checkpoint, protocol identities, corpus, optimizer epoch, step interval, execution partition and sampler. I11's typed `TrainingIntent` is not present on I12's base commit, so this implementation validates the versioned JSON fields without defining a competing serialized Rust DTO. The integrator must connect this adapter to the accepted I11 type and real caller before treating it as an execution gate.

## Preflight boundary

`training::intent::validate` is pure. It checks the accepted parent and optimizer epoch, initial step, advancing target step, protocol and corpus digests, frozen input corpus identity, and exact parameter/first-moment/second-moment key and shape equality. A mismatch returns an error before the caller can allocate device state, mutate optimizer state, or write journal evidence. The function returns local hashes only; those hashes are not a grant, admission, launch token, committed update, or GPU authorization.

The public `require_original_gpu_grant` hold remains unchanged. I12 fixtures do not issue, emulate, or release GPU grants. A caller must keep the original grant check enforced and call preflight before CUDA; wiring that caller and proving its ordering are coordinator-owned work.

## Frozen input and occurrence closure

`training::inputs::freeze_intent_closure` wraps the existing `jope.training_inputs.v1` snapshot with per-head eligibility proof digests and fresh/replay occurrence IDs. It requires nonempty unique IDs and refuses an ID classified as both fresh and replay. The existing legacy snapshot and its digest remain unchanged. A later accepted outcome is represented by constructing a new closure with a new occurrence set; the prior JSON value and digest remain stable.

I04's accepted role/exposure proof schema and I06's support/behavior identity are not yet present on the I12 base. The proof map is therefore a draft adapter boundary, not validated admission authority. This work does not implement filesystem read/use race protection, launch-time caller integration, optimizer mutation, journal accounting, checkpoint commit, export/reload, or next-actor verification. Those remain unproven until the coordinator integrates and tests the actual route.

## Verification scope

The neural-gated `training_intent` fixture target covers pure positive and negative preflight cases, the existing public grant hold, and immutable later-intent behavior. It is CPU-only fixture evidence. A zero-test default-feature collection is gated out and does not count. The execution target needs coordinator-owned feature, coverage, inventory, and CI registration before integrated acceptance.
