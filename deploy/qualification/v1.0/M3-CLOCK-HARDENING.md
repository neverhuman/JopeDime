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

## Signed BC14 witness binding

The signed RC2 candidate is source `bc14b7f8cfcb6493482c8c2a0add927c7ca1e224`,
which includes the Delete shared-stream RNG correction. Its fresh normal-Cargo
candidate stream is SHA-256 `97e1dacb68420b311705cf8eb9b800025803ae8dc92daa8366c09b01fd82e27d`;
the matching original frozen Python CPU admission is
`051058a1a919c9c64360aa6345d2c1d72b819362c22337f1e1705d2604e86767`.
Both were measured for BC14, with 3,072 repeated generation calls and 3,072
repeated frozen CPU forwards. These current measurements differ from the older
`02a0f381` / `6e06c22f` pair, which remains an optional historical diagnostic.

Bind current measurements to the BC14 tuple. Do not replace `PROPOSER_PINS`,
the original 1,024-record release reference, clock, models or the exact ±2pp
criterion. Fresh generation must still equal the pinned candidate stream.
Admitting these inputs exposes the actual numerical FAIL: non-exploring head
403/1,920 versus 45/192 (-2.447916667pp), and insertion 380/1,920 versus 34/192
(+2.083333333pp). Other numerical clauses pass; this binding fix is not M3 parity
or release qualification.

Data-free RED/GREEN controls accept the actual BC14 tuple, refuse the stale
current tuple, retain original/paired references and preserve the measured
3-pass/1-fail result. Trusted witness provenance and complete replay receipts
are retained under the owner plan's `evidence/REL-07/obj02-M3-t1230/`, with the
fix-forward receipt under `evidence/REL-07/obj02-t1440/`.
