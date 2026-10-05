# S1 teacher search packet — owner grant and receiving requirements

Jepson's typed `CONFIRM cb6243ace3a5 + S1-BUDGET-18` on October 5 at
18:34Z selects **S1: new GP search with inner-CV-only selection**, using
**C6 route (i)**, the per-fit evaluator change in Claude's patch 0001.
The owner receipt is
`2ca6a40c3128c07024c835d3a0b5cbf325e4df810242d13307db3a7451c136ff`.
Codex root (`jopedime_codex_cleanup`) owns the teacher-engine landing: patches
0001/0002 and the channel-4 fixed-equal policy go into one separate reviewed
core PR off current main. Claude's `442374.JopeDime_DLTrain`, session
`df8edf60-e484-4e31-87b0-1ecf13ef4cb3`, retains DL preparation and pilot ownership.

This grant supersedes the earlier proposed three-view rescoring route (ii)
and the pre-grant refusal of 18 generations. Historical test-selected programs,
the old fixed-20 protocol, measurements and exhausted windows retain their
identities. Rescoring alone cannot remove test-selected ancestry. This source
packet and its budget reservations do not establish runtime G22 qualification.

## Fixed budget and accounting

The [budget](teacher-budget.json) specifies 4,873 contexts, 96 initialization/G0
slots each and **18 equal generations of 64 slots** each:

| Accounting category | Attempt ceiling |
| --- | ---: |
| Initialization/G0 | 467,808 |
| Evolution | 5,613,696 |
| Total | **6,081,504** |
| Per context | **1,248** |
| Extra retries beyond total | **0** |

Failures, timeouts, OOM, cancelled dispatches, lost/unknown results and retries
consume slots inside this ceiling. Record attempts, completions and failures
separately. A failed slot cannot become an uncharged replacement evaluation.
Reserve transactionally before dispatch; duplicate delivery consumes no second
execution. Recovery retains the original `2026-10-10T00:00Z` deadline and all
charges. Incomplete or nonfinite results cannot become teachers. Operational
failure may leave a context incomplete, which must remain visible; it cannot
silently buy extra evaluations or shorten other contexts based on their test
scores. Every admitted context receives the same fixed policy, with no
outer-test-based continuation, stage allocation or early stopping.

## Fitting and selection

The new worker consumes decoded unscaled JTT1 values and fits preprocessing on
each model's fitting rows. Inner validation and official-test rows cannot affect
that fit. The S1 engine ranks, selects parents/survivors/champions and constructs
semantic novelty from inner-CV scores and out-of-fold residuals only. Official
outer-test RMSE/predictions remain separately stored diagnostics. Mutation tests
must kill every test-dependent selection and continuation path, and an
inner-CV positive control must still alter the winner.

All 4,873 existing contexts carry historical `t1_split` row-selection exposure;
coreset and kmeans strata remain recorded. Cleaning fitted transforms does not
make these contexts fully train-only or erase historical model exposure.
Report results by their actual scope. New outer splits require a separate owner
decision; they are not supplied by this grant.

## Throughput and runtime qualification

Claude's retained `RUN/tmp/teststeer/throughput.json` samples 60 historical
contexts and 18,404 local observations: mean **2.327 seconds (2.33 rounded)**,
p90 5.391 seconds, p99 21.364 seconds. Its pinned digest and bytes are in the
budget. This is a historical observation, not qualification of the changed
S1 evaluator or a distributed allocation. The budget provides arithmetic at
24/320/640 reference slots, all explicitly unallocated. The actual remaining
window, existing CPU reservations, staging, queueing, result durability,
cancellation and failures must be costed before runtime admission. A 90-second
candidate timeout alone does not prove feasibility.

The GP owner must qualify the merged evaluator/engine, immutable train-only
request inputs, equal-budget coordinator and durable attempt ledger through
positive and adversarial receiving paths. Freeze source/binary identities,
fold memberships, program/adaptation ancestry, objective, participants,
resources, destination and original deadline before dispatch. A teacher-scoped
gateway must not broaden existing intake ports 7410–7478 or their scopes.
No xbabe3 pool controls are authorized here. Fit workers and outer scorers keep
separate input and result-release authority. Actual optimizer updates, serving
promotion and transfer benefit retain their own evidence gates.
