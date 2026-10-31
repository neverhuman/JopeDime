# H72 — W8 equal-cost control: preregistration

**Status:** preregistered; no fit has run. **Date:** 2026-10-31. **Owner:** rd_opus.
**Parent:** W8 resplit diagnostic, H63 (PR #804), run `w8-opus-20261031a`.
**Scope:** CPU only. Native Rust two-fold CV on DL-Train training rows. No GPU, no DL-Val, sealed,
holdout or outer read, and DOPE is not involved.

## Question

H63 found that challengers which beat their context champion on the original two-fold cut keep that
win in only 57.6% of training-only resplits. The question here is what to do with fits in the GP loop:

- **P0, search only (the live rule).** Spend every fit on new proposals and promote each strict
  original-cut win.
- **P1, search plus reliability.** Spend the same number of fits, but pay three resplit fits to check
  each original-cut win, and promote only wins that survive. P1 therefore evaluates fewer proposals.

Which policy produces more promotions that hold up on cuts neither policy saw?

The H63 control as first proposed compared 570 original-cut apparent wins with five resplit-confirmed
wins. ASTRA's review rejected that because the two endpoints differ. This design scores both policies
with **one common confirmation endpoint** that neither policy used for its decision.

## Fixed inputs

| Input | Identity |
|---|---|
| Contexts | The 16 DL-Train contexts of H63 (13 families), ordered by `sha256("w8c-72|" + context)`. Shard A is the first 8, shard B the last 8. |
| Frozen incumbent | Per context, the best original-cut utility among the H63 champion and challenger programs. These come from the H63 receipts, so they cost no new fits. Its H63 resplit utilities for seeds 701–703 are reused. |
| H63 receipts | S3 `experiments/rd_breakthrough_20261030/W8/w8-opus-20261031a/results-shard0.jsonl` (sha256 `f802fc43f3f0…`) and `plan-shard0.json` (sha256 `ec90ca128db1…`). |
| Proposal stream | The live proposer `propose_flywheel_candidates` (`JopeDimeMicro/python/rust_evaluator.py`, sha256 `53422343…`). Parents are the incumbent plus the context's H63 challengers; the bank is DL-Train `champions_featurized.json` (sha256 `ea020d48…`); landmarks are `lm48_z` (`context_features.npz`, sha256 `75704e82…`); seeds are `72_000_000 + 1000*index + batch`. The stream keeps the first 40 proposals not already in the pool or in H63. |
| Evaluator | `bootstrap.s1main`, sha256 `e3be9386…`, an x86-64 ELF. Train-only JTT1 tables use the `training_cv_envelope` padding. The original cut uses the original fold bytes; resplits use a seeded balanced permutation of the same fold bytes. |

The landmark input follows H58's cache contract (`lm48_z`), not the live loop's raw `lm`. Both
policies use the identical stream, so this affects which proposals exist, not the comparison between
policies.

## Policies, with one budget per context

Each policy gets **B = 40 native fits per context**. Both read the *same* ordered proposal stream, so
their outcomes are paired by common random numbers. The incumbent is frozen; a promotion does not
raise the bar for later proposals.

- **P0** evaluates `stream[0:40]` on the original cut and promotes every proposal with
  `u > u_incumbent + 1e-6`, the live strict-improvement margin.
- **P1** walks the same stream. Each evaluation costs 1 fit. Each apparent win costs 3 more fits:
  resplit seeds 701–703 for the challenger, compared with the incumbent's existing H63 values for the
  same seeds. P1 promotes iff the mean resplit difference is > 0. It stops before any step that would
  exceed 40 fits, and an unaffordable confirmation means no promotion.

P1's evaluations are a prefix of P0's, so both policies share the original-cut fits.

## Common confirmation endpoint

Every program that *either* policy promotes, and every incumbent, is fitted on three **new**
resplits, seeds 801–803, that no decision used. A promotion is **confirmed** iff the mean over
801–803 of `u_challenger − u_incumbent` is > 0. Confirmation fits are evaluation overhead. They are
reported separately and charged to neither policy.

## Endpoints and decision rule

Per context and policy, record the counts *promoted*, *confirmed* and *false* (= promoted −
confirmed), plus the summed confirmation gain. Contrasts are P1 − P0 summed within family, with a
10,000-draw family-cluster bootstrap (seed 20261031, 13 families).

- **Primary:** net = confirmed − false.
- **Secondary:** confirmed count, false count, summed confirmation gain, and family-grouped
  Bradley–Terry Elo of P1 against P0 anchored at 1500. The Elo uses the repository's `elo_tracker`
  with tie margin 1e-4, prior 0.01 and 400 iterations, and scores each context by the policy's summed
  confirmation gain.
- **Decision:**
  - The 95% interval of net lies above 0: **reliability pays at equal cost**. Adopt resplit-confirmed
    promotion in a later, separately approved live change.
  - It lies below 0: **search pays**. Keep the live rule.
  - Otherwise: **inconclusive at this scale**. Report the counts and interval, with no adoption.

This is a small study (16 contexts / 13 families). An inconclusive result is a likely outcome and is
reported as such, not as evidence of no effect. `val_reg_inner` and U4 do not apply: no ranker is
scored.

## Cost accounting

Every native call is counted, whether it completes, fails or times out (600 s); failures and timeouts
count as spent fits. Each call records wall seconds and the child's **actual CPU seconds** (user+sys
from `os.wait4` rusage), aggregated per stage. Per policy, decision cost is 40 fits per context by
construction, with P1's verification fits inside its 40.

Planning estimate, not a measurement: 640 shared search fits, plus at most 3 × (P1 apparent wins)
verification fits, plus 48 + 3 × (distinct promoted programs) confirmation fits. At the live loop's
historical ~3% apparent-win rate that is about 0.8k fits.

**Hard cap: 1,640 fits**, the unspent W8 ceiling (2,400 − 760 used by H63). With separate output
directories each shard is capped at 820. A stage that would exceed its cap stops before launching.

## Execution

```text
python3 -I w8_control.py plan    --w8-results R --w8-plan P --out OUT              # 0 fits; freezes plan.json
python3 -I w8_control.py search  --w8-results R --w8-plan P --out OUT --shard A|B --scratch S --workers N
python3 -I w8_control.py verify  ...same arguments...                              # P1 resplits 701-703
python3 -I w8_control.py confirm ...same arguments...                              # common endpoint 801-803
python3 -I w8_control.py analyze --w8-results R --w8-plan P --out MERGED           # after both shards
```

- **Plan:** `plan` is generated once from merged source. Its sha256 is pinned in the shard
  assignment, and every shard must use that byte-identical `plan.json`.
- **Host rules:** nice 15, single-threaded workers, and a worker waits while the host's 1-minute load
  exceeds 85%.
- **Host eligibility:** the pinned evaluator is x86-64, so the Linux/aarch64 xbabe4 container cannot
  run it without a rebuild. A rebuild would be a new evaluator identity and is not allowed. Fits
  therefore run on x86-64 hosts only.
- **Outputs:** each shard has its own run ID under
  `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/W8C/`. Each run folder holds
  a manifest (git SHA, input keys and hashes, seeds, host, cost, verdict). Writes are create-only
  (`put-object --if-none-match "*"`) with a version-pinned readback hash check.
- **HWM pointers:** each confirmed promotion is appended to `HWM_CANDIDATES.jsonl` as a
  training-CV development candidate, not a serving claim.

## Holdout risk

None. Inputs are the admitted DL-Train train-only tables (`nq = 0`, structural zero query padding),
pools, the H63 receipts and frozen proposer assets. No outer, sealed or meta-test rows exist in these
tables, and no DL-Val label is read.

## What this cannot show

- It does not measure deployed or held-family generalisation. Confirmation uses new cuts of the
  *same* training rows, which measures cut sensitivity.
- It does not test a ratcheting incumbent, ranker-guided proposal selection, or more than 16
  contexts.
- A positive result supports a later live-rule proposal. It is not itself authorisation to change
  the live loop.
