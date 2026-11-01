# M3 scheduled head collision repair

A nonexploring request can lose a scheduled head replacement when its random
head choice repeats a candidate already accepted in that batch. Call130 from
the pinned TRAIN reconstruction demonstrates the loss: steps1and6 draw the same
histgbm head; the duplicate is rejected and insertion fills slot8. The frozen
Python source draws two distinct heads. ENG lanes_t1145 directs this fix-forward.

Keep the first weighted head draw. Only when it collides with a previously
accepted canonical candidate, scan the rest of the finite preferred-head pool
in its existing cyclic order and accept the first legal, novel alternative.
Every attempted alternative retains language/memo filtering and outcome
accounting. Exhaustion falls through to the next scheduled arm. No extra random
draw, seed offset, head quota, model or operator weight is introduced. The policy
changes conditional collision handling; it is not a claim of byte-identical R0
generation. Initial parent/memo-only or language rejection preserves old behavior.

The regression uses call130's row-free parent, landmarks, retrieved pair and
seed/round. It must fail on unchanged main and pass with two distinct head
outputs after repair. A second scenario excludes all other heads in the memo and
requires bounded completion without bypassing the exclusion. Existing full
proposer tests retain accounting, determinism, language and parent invariants.

Fresh native generation and same-forward admission measurements bind this policy;
all original R0 historical reference pins and the exact +/-2percentage-point
criterion stay unchanged. The four-clause full M3 replay remains authoritative.
Local regression success alone cannot qualify M3 or RC2; any remaining numerical
failure is recorded without sampler/seed/reference fitting. No GPU/training or
signed-candidate qualification is claimed by source tests.

Alternatives rejected: admitting duplicates breaks uniqueness; unrestricted
retry risks nontermination; changing seeds/quotas or choosing fallback heads by
observed costs would fit the measured target.
