# Head replacement: reject no-ops before collision recovery

The v1.0.0 collision repair retries a known preferred head inside the same typed
head slot. That is useful when a genuine edit collides with an accepted program,
another parent, or the memo. Applying it when the first draw merely reconstructs
the selected parent also turns a no-op into an additional mutation opportunity.
The frozen R0 sampler rejects that unchanged parent and advances the edit cycle.

For v1.0.1, compare the first draw's canonical program key with the selected
parent's key before enabling collision recovery. A no-op is offered once and
rejected by the existing parent filter. A changed program already in `seen` or
`memo` retains the finite, uniformly shuffled retry order. Canonical parent/memo
exclusion, language and byte limits, accounting, and deterministic replay remain
in force. A collision with a different parent is still a genuine changed edit.

This changes no preferred pool, multiplicity, operator weight, salt, seed, clock,
shared RNG draw, model, reference, label, or admission-order rule. It separates
mutation applicability from novelty recovery rather than fitting a histogram.

A row-free regression reproduces the baseline no-op inflation before the next
insertion arm; it fails on the earlier implementation and passes with this
change. A distinct control requires a real head change colliding with another
parent to retain its retry. Existing memo, intra-batch collision, exhaustion,
conditional-weight, language, byte-limit, and deterministic controls are kept.

The C1 policy was preregistered before mutation and measurement. Fresh normal
production-linked generation, the frozen doubled CPU admission, all three
historical operator views at the unchanged exact +/-2 percentage points, and
all four M3 clauses are required. The v1.0.0 waiver does not apply to v1.0.1;
source controls alone do not establish numerical or release qualification.

The first measured C1 stream (3,072 calls and duplicate checks, 44,544 outputs)
passes all 21 exact operator/view comparisons using the unchanged first 10,000
frozen-model admissions. Head counts are 1,859/10,000 overall, 437/1,920
non-exploring and 1,422/8,080 exploring, versus 175/1,024, 45/192 and 130/832
in R5: +1.50015625pp, -0.677083333pp and +1.974009901pp respectively.
The matched 20 non-exploring and 30 exploring memo replay control returns
640/640 new unique, allowed proposals, with deterministic duplicate replay.
These measured proposer results do not substitute for the required full M3 run
or qualify a new release artifact.
