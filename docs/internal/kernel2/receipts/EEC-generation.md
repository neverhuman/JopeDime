# N4(b) shipped-profile generation witnesses

This Track E prerequisite adds clause (b) to `kernel2_closure` for the exact
profiles returned by `shipped_kernel2_profiles`. It is required before B05
generation. Legacy profiles and revisions one and two retain their existing
closure cases; their old generators are unchanged.

For each shipped profile, the gate uses 100,000 deterministic seeds across input
widths 1, 3, 8, 20 and 600. Each seed creates two independent births, a mutation
and a two-parent crossover using the original profile and weights. Every result
must pass admission and canonical equality: 200,000 births, 100,000 mutations
and 100,000 crossovers per profile, with zero invalid children. Existing E2/E3
per-arm 100k cases remain separate gates.

The inventory comes from the profile's complete `elements()` set. The observer
reads stage/head/carrier roles and scalar AST nodes, including column, constant
and negation policies. Pipeline input `X` cannot stand in for expression `x`.
Every element needs all three observations:

- Creation in a real generated birth.
- Modification or replacement of its node at an expression edit/crossover site,
  or a stage/head parameter edit or head replacement. A whole-program change
  does not establish this observation for unchanged nodes.
- Removal from the whole child, including replacement of a required head.

For each first observation, the test prints the exact profile digest, operator
arm, seed, width and before/donor/after programs, and replays the public wrapper
to obtain the identical child. No narrowed pool or forced weights manufacture
coverage. Missing observations, truncated trial denominators, unexercised arms
and invalid children fail the gate. One nonignored case per shipped profile keeps
the complete trials within the existing PR timeout. Two short negative controls
refuse missing/no-op histories and scalar witnesses inferred from structural
input or unrelated edits; they do not duplicate the full trial loops.

Acceptance:

```sh
cargo test --locked -p feat-math --test kernel2_closure -- --nocapture
cargo clippy --locked -p feat-math --all-targets -- -D warnings
```

This is a test and witness change. It does not change production operators,
profiles, admission, codecs, manifests, models, evaluator identity or emitted
source. Existing clauses (a), (c), (d), declared (e), old revision counts and
strict golden/compatibility gates remain. It does not qualify execution closure,
all C++ recipe shapes, multiple cuts, concurrent/island coordination,
incumbent/evolve integration, policy interplay, equal-budget promotion, B05
generation or release. Those requirements remain separate.

Source, baseline/after inventories, failed tests-first probe, replayable witnesses,
lane audit and exact CI/review receipts are retained outside the checkout under
`jope-evidence/kernel2/bt/TRACK_E_EEC_GENERATION`.
