# Search strategy v1

`jope.search.deterministic.v1` defines proposal ordering, duplicate admission, deterministic
resource reservation, evaluation sequencing and stable tie handling in
`feat_math::kernel::strategy`. Product and benchmark code supply a proposer, allocator and
evaluator. The core has no filesystem, store, role, clock, RNG, fitting or driver dependency.

## Independent identities

Every run binds all of these values in its immutable identity:

- strategy ID and version;
- vocabulary profile ID, revision and SHA256;
- evaluator ID and version;
- seed policy and resource cost profile;
- full source commit SHA, caller context ID, exact input SHA256 and event schema version.

The strategy identity is not a field in `SearchProfile` and does not change its serialized bytes,
registry IDs, edit weights or historical profile digests. Changing strategy while holding the
profile fixed changes only the strategy configuration/identity. Changing profile while holding
the strategy fixed changes only the vocabulary identity; the driver's admission callback must
still reject unsupported candidates.

## Execution and traces

The core consumes proposal slots in order. A duplicate proposal emits an event and spends no
reservation. A unique candidate receives a reservation before evaluation; failed outcomes retain
that reservation and remain in the trace. Scores must be finite. Lower normalized scores rank
first and ties keep the earlier admitted candidate. Drivers whose native score is higher-is-better
must declare and apply that normalization in their evaluator version.

The trace omits elapsed wall time. Given the same identities, candidate bytes/order, reservation
results and evaluation outcomes, it emits identical causal events and selection in product and
bench. Parallel drivers must reorder completions by reservation order before recording. A real
deadline can change which operation starts, so this contract does not promise equal traces under
uncontrolled wall-clock deadlines.

## Driver and evidence boundary

The opt-in `kernel2::search::run_cut_shared_strategy` route produces actual kernel2 measurements
and the core trace. The benchmark method `afm.shared-strategy.v1` calls that route and returns the
trace with the method result. Existing product `run_cut`, `afm.lite-race.v1`, and
`afm.lite-race.v1.resampled` keep their original behavior and IDs. A historical row is not
accepted by the shared-strategy evidence validator unless it carries the new method ID and exact
strategy, profile, evaluator and trace identities.

This bounded fixture route uses one declared evaluation unit per unique candidate and a fixed
seed stream. Its trace is a conformance artifact, not live wall-time qualification, optimizer
improvement, actor adoption, durable storage, or serving authority. The current lane used draft
I05 event and I16 reservation interfaces because those dependencies were not accepted at its
base; the caller-supplied source SHA is explicit and the one-unit allocator is local provisional
behavior that must be rebound to accepted I16
cost records during integration.

## Dependency direction

`feat-math::kernel::strategy` depends only on its existing serialization and digest dependencies.
`jope-pipeline` adapts the core to product profiles, cuts and kernel2 evaluation. `jope-bench`
already depends on `jope-pipeline` and adds a new method entrypoint. There is no dependency from
feat-math to either driver and no learning-to-pipeline backedge.
