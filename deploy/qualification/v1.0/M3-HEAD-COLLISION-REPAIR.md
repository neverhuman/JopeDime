# M3 scheduled head collision repair

A request can lose a scheduled head replacement when its random choice repeats
a candidate already accepted in that batch. Pinned TRAIN call 130 demonstrates
this: steps 1 and 6 draw the same histgbm head; duplicate rejection causes an
insertion to fill slot 8. The frozen Python source draws two distinct heads.
ENG's `lanes_t1145` ruling authorizes this fix-forward.

Keep the first weighted head draw. Only when it collides with an accepted
canonical candidate, scan the remaining finite preferred-head pool in its
existing cyclic order and accept the first legal, novel alternative. Count each
attempt and preserve language, byte-limit, memo and novelty filtering. Exhaustion
continues the existing schedule. Initial parent, memo-only and invalid-candidate
rejections retain the single-attempt behavior. No new random draw, seed offset,
head quota, operator weight or model change is introduced.

This changes conditional collision handling; it does not reproduce Python's RNG
stream. The policy was recorded before measuring its frequency outcome. Admitting
duplicates, unlimited retries, and choosing heads or seeds by observed costs were
rejected. There is one measured policy, with no search over variants.

The call-130 regression preserves the actual row-free parent, landmarks,
retrieved pair and seed/round. It fails on unchanged source and passes with two
distinct head outputs and an unchanged first head. Its memo-exhaustion control
requires bounded completion without admitting measured candidates. A separate
regression covers a valid parent near the DSL byte limit: an oversized replacement
must be filtered rather than abort the round. Existing proposer tests retain
determinism, accounting, parent and language invariants.

Fresh locked-Cargo generation produces 44,544 candidates from 3,072 calls, with
3,072 duplicate-call checks. Frozen CPU admission also passes 3,072 repeated
forward checks. Comparing the first 10,000 admissions to the original 1,024
historical attempts passes the unchanged +/-2 percentage-point criterion overall
and in both exploration strata. Nonexploring head admissions are 446/1,920 versus
45/192 historically (-0.208333 percentage points); the largest difference is
exploring head replacement (+1.875 percentage points).

Only current-source candidate/admission pins are refreshed. Historical reference
pins, population, source-clock mapping and threshold remain unchanged. The full
four-clause M3 replay and a newly signed candidate remain separate qualification
requirements. No GPU, training, Elo measurement or signed-release qualification
is claimed by these source and admission controls.
