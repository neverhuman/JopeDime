# M3 R0 diagnostic clock arithmetic

The explicit `r0-clock` probe reconstructs a base seed and advances a source round.
Its u64 multiply/subtract/add operations previously wrapped silently in a release
build for invalid numerical input. Lift that exact formula into one small helper
and return an error if any intermediate operation overflows or underflows.

Valid seeds, rounds, the 104729 multiplier and 16-round cycle are unchanged. The
legacy default branch stays byte-identical. No sampler, RNG, weights, witnesses,
comparison populations, tolerances or release-reference identity changes. The
historical M3 failure remains authoritative; this is input hardening, not parity
or release qualification (REL-07; spec §14).

Rust tests exercise the actual helper with overflow checks disabled: three golden
clock examples, each failure site, and exact u64 maximum boundaries. The old
formula is retained as a private red revision and must fail five refusal tests;
the checked implementation must pass all six. Type-check the complete probe
against locked product dependencies and retain actual source identities. A model
forward, native proposer replay, study or signature is unnecessary for this fix.
