# W2 — Complete accepted model and encoder semantics

Status: planned. Owner: policy/encoder admission. Depends on W1 release and scope
contracts. Follow [the execution contract](EXECUTION-CONTRACT.md); qualifying a new
architecture or calculation requires a challenger identity, never an accepted-loader
compatibility fallback.

## Starting points and fixed calculation

Inspect `crates/jope-policy/src/model.rs`, `context_producer.rs`,
`context_producer/admission.rs`, `crates/jope-encoders/src/bundle/loading.rs`,
`bundle/integrity.rs`, `foundation.rs`, `target_encoder.rs`, and
`crates/jope-learning/src/encoder_receipt.rs`. The last module is explicitly opt-in
at the planning baseline; W2 producer types alone do not close W3 consumer admission.

Preserve accepted lineage-B step-60000 and its accepted/challenger separation.
Hyperion pools 128 coordinates into mean, population standard deviation, tenth and
ninetieth percentiles: 512 values. Foundation produces target-conditioned 2048 values
then applies the frozen 2048→1024 projection and normalization. Target contributes
512 values from its declared window procedure. Normalize the blocks, concatenate
512+1024+512, and clip to ±32. Preserve the existing `3e-4` threshold on its original
comparison surfaces; new intermediate checks require declared tolerances.

## Ordered tasks

### W2-T01 — Trace accepted assets and consumers

- [ ] Inventory accepted load entrypoints, sidecars, vocabulary/grammar, active heads,
  preprocessing, projection/normalization files, encoder assets and warm caches.
  Resolve actual source operations rather than guessing from tensor names.
- [ ] Separate static release identity from request-specific dataset/scope/RNG/device
  identity. Identify the trusted issuer and expected values at each receiving loader.
- [ ] Acceptance: a complete dependency table links every active operation to validated
  bytes and every inactive tensor to a stated role. Unknown historical ancestry stays unknown.

### W2-T02 — Extend the release-controlled semantic bundle

Depends on T01.

- [ ] Version a manifest binding policy configuration, vocabulary meanings, grammar
  restriction, active heads, encoder assets, transforms, preprocessing and context ABI.
  Define accepted and challenger constructors separately.
- [ ] Authenticate expected identities from the release decision, not the candidate's
  own sidecar. Reject unknown sidecars, same-shape weight edits, vocabulary permutations,
  rejected checkpoints and mismatched cache manifests before inference starts.
- [ ] Acceptance: exact accepted bundle succeeds; each single-dependency substitution
  fails with no accepted inference or cache promotion. Retain loader-entrypoint coverage.

### W2-T03 — Make semantic products opaque

Depends on T02.

- [ ] Introduce distinct product types for cell surfaces, row embeddings (128), dataset
  signatures (512), raw Foundation (2048), projected Foundation (1024), Target (512) and final context (2048).
  Constructors are named authenticated transformations, not public arbitrary-vector tags.
- [ ] Validate external serialized products against producer, transformation, scope,
  layout and exact bytes. A valid width or caller-supplied enum is insufficient authority.
- [ ] Acceptance: compile-fail tests reject same-width internal swaps; serialized
  Target/signature and Foundation/final-context swaps, padding and truncation fail at
  the receiver. Correct producer outputs pass all named transformations.

### W2-T04 — Consume validated bytes without reopening by path

Depends on T02/T03.

- [ ] Trace check-then-load gaps in model bundles and context assets. Keep immutable
  validated buffers or appropriately pinned open handles through decoding/loading.
  Make caches inherit the entire dependency closure and consumer scope.
- [ ] Inject replacement between validation and use; test cold/warm load, cache rebuild
  and resume. An implementation must either use the original bytes or reject the race.
- [ ] Acceptance: same-path replacement never produces inference with unauthenticated
  bytes. Authorized cold and warm paths consume identical tensors; retain loaded-byte hashes.

### W2-T05 — Produce golden intermediate calculations

Depends on T03/T04 and W1 row/window semantics.

- [ ] Freeze small independently calculated fixtures for Hyperion pooling, Foundation
  projection, per-block normalization, Target windows, final concatenation and clipping.
  Include population-versus-sample standard deviation, percentile ordering and row alignment.
- [ ] Retain intermediate vectors, arithmetic conventions, dtype, assets and comparison
  scope; do not derive every expected value through the same implementation under test.
- [ ] Acceptance: block permutations, changed projection, wrong normalization, shifted
  target windows and mismatched row order are detected before qualified context release.

### W2-T06 — Verify complete tensor inventory and roles

Depends on T01/T02.

- [ ] Replay accepted payload/header integrity and all tensor names, shapes and dtypes.
  Reproduce the pinned inventory of 136 F32 tensors and 27,796,494 parameters.
- [ ] Account separately for stored, active, frozen and trainable parameters; explain
  overlap between these categories and which objective/consumer reaches each active tensor.
- [ ] Acceptance: payload bit corruption and same-shape value substitution fail;
  arithmetic totals reconcile to the exact accepted object, not an arbitrary matching header.

### W2-T07 — Qualify producer-to-consumer handoff

Depends on T02–T06; W3 supplies authority-bearing admission integration.

- [ ] Carry release identity, typed transformation chain, input scope and final context
  digest through proposal, teaching and replay. Keep diagnostic unqualified contexts usable
  only in explicitly separate APIs that cannot mint a qualified teacher.
- [ ] Acceptance: correct assembly reaches the intended consumer; wrong consumer/scope,
  stale cache and unknown historical receipts are rejected or quarantined. Retain the
  original native/Candle comparison and its bounded scope.

## Verification and delivery

```sh
cargo test --locked -p jope-policy --features testing,context-producer -- --list
cargo test --locked -p starforge --lib -- --list
cargo run -q -p jope-records -- verify-inventory
```

The tensor verifier needs the exact pinned local asset; absence is a blocked proof,
not permission to download unreviewed replacements. Feature `testing` is fixture-only.
Proposed slices: (A) semantic bundle/loader T01–T02; (B) opaque products and race-free
loading T03–T04; (C) golden calculations and inventory T05–T06; (D) consumer integration
T07 with W3. Deliver semantic manifest, product ABI, intermediate fixtures, payload
inventory and exact tested handoffs. No architecture, sampler or accepted calculation changes.

## Original finding routes

Primary register entries: `S03-G03`, `S03-G04`, `S03-G05`, `S03-G06`, `S03-G07`, `S03-G08`, `S03-G10`, `S03-K02`, `S03-K03`, `S03-K05`, `S03-K06`, `S03-K07`, `S03-K08`, `S04-G01`, `S04-I01`, `S09-G1`, `S09-G2`, `S09-G6`, `S09-G10`, `S09-I1`, `S09-I2`, `S09-I6`, `S09-I10`.

See [the complete crosswalk](FINDINGS-COVERAGE.md) and original clauses. This primary
routing is not a waiver of cross-cutting acceptance or later live integration.
