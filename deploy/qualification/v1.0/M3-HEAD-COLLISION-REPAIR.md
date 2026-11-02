# M3 scheduled head collision repair

A head-replacement draw can repeat an accepted candidate, a parent or a memoized
program. Rejecting that draw can lose the scheduled mutation slot even when a
legal, novel preferred head remains. Pinned TRAIN call 130 demonstrates an
intra-batch repeat; independent review also identified the memo/parent paths.

Keep the first weighted draw. If its canonical key is in `seen` or `memo`, scan
the remaining finite preferred-head pool in a uniform random order and accept
the first legal, novel alternative. Preserve language, byte-limit, memo and
uniqueness filtering, and count every attempt. Exhaustion continues the existing
schedule. An invalid or oversized first candidate keeps single-attempt filtering.
The first shared draw and main RNG position stay unchanged. A cloned ChaCha
state uses a distinct stream for each typed slot (current stream plus step + 1)
to shuffle alternatives, preserving conditional pool weights without lookahead
reuse. No round/admission seed, clock offset, quota, weight or model changes.

This intentionally changes conditional collision handling, rather than claiming
Python RNG-stream identity. The policy and the review-driven extension to all
known keys are recorded before measuring the amended policy. Duplicates, unlimited
retries and choosing heads or seeds by observed costs are excluded.

The call-130 regression preserves its row-free parent, landmarks, retrieved pair
and seed/round. It covers repeated draws, a memoized first choice with novel
alternatives, deterministic replay, accounting and exhausted memo alternatives.
A data-free three-head control proves conditional weights, permutation and
shared-RNG invariance; cyclic fallback would overweight the next head.
A separate near-byte-limit parent control requires oversized replacements to be
filtered without aborting the round. The merged shared-Delete RNG correction is
preserved. Existing tests retain language and parent invariants.

Fresh current-source generation and frozen Python admission witnesses must bind
the combined implementation. All original historical reference pins, population,
source-clock mapping and the exact +/-2 percentage-point criterion remain fixed.
All four M3 clauses must be measured at their declared source; earlier-head passes
do not qualify amended code or a new signed candidate. No GPU, training, Elo or
signed-release qualification is claimed by these source controls.

The corrected implementation still fails the fixed historical proposer criterion:
overall heads 1,939/10,000 versus 175/1,024 (+2.30015625 percentage points),
exploring 1,488/8,080 versus 130/832 (+2.790841584), nonexploring 451/1,920
versus 45/192 (+0.052083333). Generation uses 3,072 calls and duplicate checks;
the original frozen Python oracle supplies 3,072 doubled CPU forwards. This is a
recorded qualification failure, not permission to select an earlier passing
policy, alter the reference or change seeds/weights/thresholds. Source tests and
conditional-weight correctness are separate from scientific qualification.
