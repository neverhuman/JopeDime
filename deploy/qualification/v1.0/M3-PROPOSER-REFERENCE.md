# M3 source clock and historical reference

The qualification probe must pass the base seed and actual source round to
`Round`. Previously the default used an already round-mixed Python seed and a
process-wide call counter. Correct routing existed but was activated only when
optional paired-R0 diagnostic metadata was present. This made the measurement
clock depend on whether a nonqualifying diagnostic was requested.

Use `source_round_seed - source_round*104729 + context_index` and
`source_round + 16*cycle` for every current measurement. Default invocation and
explicit `r0-clock` use this mapping; `legacy-clock` is an explicit historical
reproduction mode that the release provider never selects. Checked arithmetic
continues to reject overflow and seed underflow.

An invocation manifest can supply `native_clock`, with `clock` equal to
`source-round-104729/v1` and `{path, sha256}` entries `native_candidates` and
`native_admission`. These identify the already-pinned corrected **current native
measurements**, independently of any Python R0 diagnostic. Existing `paired_r0`
inputs can also supply those current witnesses. A legacy-only manifest cannot
silently substitute old admission costs after the clock is corrected; it is
refused. Original cohort, historical admission and native legacy witnesses retain
their exact pins. Fresh native output must equal its current same-forward witness.

Historical qualification remains authoritative: compare against all 1,024
original labelled attempts, overall and in both exploration modes, with the
unchanged exact integer +/-2 percentage-point predicate. Correct-clock native
head replacement remains 411/1,920 versus 45/192: -2.03125 points, **FAIL**.
Legacy native is 399/1,920 (-2.65625 points). Literal frozen Python R0 itself gives
400/1,920 (-2.604167 points) against the same small historical target; its original
1,024-attempt prefix exactly reproduces that target.

The equal-population R0 diagnostic remains optional and cannot replace or improve
that historical verdict. No production sampler, RNG, weights, thresholds or
references change. Seed search and fitting operator frequencies to the old point
target are rejected alternatives. This clock fix does not close numerical M3,
upgrade RC1, or qualify RC2. All four clauses and the new signed bytes still need
actual qualification. Original historical state recovery remains false and
2,688 unlogged generated items remain excluded.
