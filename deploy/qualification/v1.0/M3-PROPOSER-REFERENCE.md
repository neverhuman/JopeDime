# M3 proposer reference and round-clock diagnosis

The historical failure is real: original native head replacement is 399/1,920
non-exploration attempts versus 45/192 historical attempts, a −2.65625 percentage
point difference. The ±2 point criterion remains unchanged and authoritative.

The probe passed an already round-mixed Python seed as `Round.seed` and a global
call counter as `Round.round`. In optional `r0-clock` diagnostics it instead passes
`source_round_seed − source_round*104729 + context_index` and
`source_round + 16*cycle`. Both admission selectors advance the same source clock.
The default legacy reconstruction is unchanged. All production Proposer code,
operator weights, RNG, frozen critic tensors and fixed inputs remain unchanged.

An unmodified frozen Python R0 source self-control (SHA256 `53422343699172b61ed69ed707015558f7cf1a344e5449f30e1eb9d708aef900`)
reproduces every non-exploration family count at the original 1,024-attempt prefix,
but its first 10,000 admitted attempts yield 400/1,920 head replacements: −2.604167
points versus that same historical target. Replacing ChaCha with Python MT in a
separate diagnostic also fails; that prototype is not shipped. Thus changing
production weights or RNG to fit the old point target has no demonstrated basis.

The optional manifest `paired_r0` has `source_sha256`, `clock` equal to
`source-round-104729/v1`, and four `{path, sha256}` witnesses: `native_candidates`,
`native_admission`, `r0_candidates`, `r0_admission`. All are pinned in source;
original cohort, bank, request, vector, admission and native witness pins remain.
Both fixed streams contain 44,544 outputs from 3,072 calls, use the same frozen
critic identity, and admit the first 10,000 attempts without seed search.
Same-population comparisons use the unchanged exact integer ±2 point predicate
for all seven families overall and in both exploration modes.

Actual unchanged Rust Proposer versus literal R0 gives a maximum difference of
0.572917 points, with 1,920 non-exploration and 8,080 exploration attempts per arm.
The corrected-clock native stream still misses the historical head target by
−2.03125 points. Reports retain that FAIL, the original −2.65625 point FAIL, and the
R0 self-control FAIL alongside the paired diagnostic PASS. The diagnostic cannot
qualify a candidate or change the release criterion. Adopting a new reference is
an explicit comparison-identity change requiring substantive protocol review;
this PR does not authorize it. Original historical state recovery remains false,
2,688 unlogged generated items remain excluded, and signed-candidate M3 has not run.

Evidence: `~/jope_plan/evidence/REL-07/hwv-t0307-proposer/` holds the preregistered
plans, immutable raw outputs/admission costs, source self-control, and exact pins.
All measurements use real admitted TRAIN inputs, CPU nice19/math1 and no updates.
