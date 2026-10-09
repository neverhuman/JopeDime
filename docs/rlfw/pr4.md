# Train-only first-label bootstrap

The new `rlfw.no_winner_bootstrap` CLI defaults to off. An explicitly enabled wave
uses verified original train contexts whose frozen status is `g0_not_ready` or
`no_published_hwm`. Those statuses carry no numeric score. Development, sealed,
excluded, rights-pending and incomplete vector/input descriptors are refused.
Original context, view, dataset, cut, fold and parent identities stay intact.
Fresh authority checks bind those identities and verified vector/input hashes
before selection, proposals, execution and research publication. Boolean and
integer identities remain distinct.

Wave selection uses seed 20261009, at most 100 contexts and one context per native
parent. The frozen bootstrap schema is separate from a development GameSpec; it
reuses the arena's byte-custody, isolated outbox and charging code. It requires a
reserved empty arena queue and yields when priority work arrives. The benchmark
runtime owns that admission lease and refreshes the queue before each execution.

The coverage operator draws the first 32 distinct supported programs, capped at
512 neural draws, and selects five by `P(complete) * P(q > 0)`, two seeded uniform
choices from the remaining slate, and one uniform audit over the whole support.
It records every raw draw and the audit, complete model predictions, the sealed
coverage probabilities, selection and phase timings before evaluation. Explore
propensities are conditional on the logged remaining slate; audit propensity is
1/6640. Exploit selection propensity is unknown and stays NULL. The audit can
repeat a selected program and still consumes its logical slot.

The point q head supplies no probability of q exceeding zero. Live admission
therefore requires a sealed `jope.rlfw-q-positive-calibration/v1` artifact with
`passed=true`, `partition=parent-held-train`, and the identical checkpoint, trie,
normalizer and tokenizer pins. A qualified runtime supplies those calibrated
probabilities; this change does not fit a calibrator or fabricate probabilities
from q's sign. A shortfall is recorded; fewer than seven distinct slate members
cannot fill the mixed eight-slot budget and authorizes no evaluation.

Qualified measured integer q_grid and complete seven-artifact closure are the
only inputs to the research ledger. Its fresh authority callback runs inside the
CAS transaction. Missing closure, role changes, failures and unknown outcomes
produce no numeric label. Nothing enters an optimizer, teacher store or serving
pointer. The yield report counts actual attempted contexts, contexts with at
least one qualified program, and contexts with at least two distinct programs
whose measured micro-q gap exceeds 10000. It reports new parent families relative
to the frozen starting inventory, observed evaluator costs, unresolved costs,
censored slots and unfinished/unattempted contexts. It reports coverage only.

## Acceptance

```sh
python3 deploy/dl-flywheel/pipelines/rlfw/run_rlfw_offline.py --receipt "$B/logs/bootstrap-fixtures.json"
just required
```

Registered synthetic fixtures cover eligibility, native identities, seeded
parent balance, mixed selection, probabilities/propensities, packet replay,
empty-arena admission, first-label CAS, failure NULLs, role changes, missing
closure, strict yield boundaries, the renewed deadline and partial/unknown costs.
Their evaluator,
artifacts and probabilities are synthetic. They establish source behavior;
hosted exact-head CI and independent approval remain required.

## Live runtime requirements

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=deploy/dl-flywheel/pipelines \
"$RLFW_PYTHON" -m rlfw.no_winner_bootstrap --enable-no-winner-bootstrap \
  --spec "$B/runs/bootstrap_wave/spec.json" --runtime-binding "$B/l0/bootstrap_binding.py" \
  --policy-code-root "$B/code_pin" --policy-qualification "$B/runs/cpu_qualified/qualification.json" \
  --device cpu --pool-workers 4
```

Run from the reviewed checkout. `RLFW_PYTHON` must select the qualified
interpreter and exact Torch runtime.
The CLI accepts `--device cpu|cuda`, default CPU. CUDA requires the complete
identical-pin likelihood/head parity certificate. All executions require the
bounded campaign scope, current health/memory guards, same-pin 1000-draw policy
qualification, pinned runtime dependencies and fixed 2026-10-10 05:00 UTC deadline
authorized by D5. The schema refuses any other cutoff, and the oracle refuses
execution at or after that cutoff. A synthetic CLI wave also verifies admission
after the superseded 18:00 UTC deadline.

The sealed `build_bootstrap(spec, device=...)` runtime supplies the complete
catalog, sample/sample_many, score_heads, coverage_probabilities, fresh role/input
authority, qualified evaluator, reserve_empty_arena and arena_queue_empty. No such
qualified live bootstrap binding or calibration is currently installed. L0's
metadata census found 6857 train vectors and input locators, with zero explicit
no-winner status fields; file presence does not qualify prepared inputs or q.
The dated 5461-context estimate is not an eligible roster. Current native parity
STOP remains a hard refusal. No actual bootstrap wave, yield or training gain is
claimed from the fixtures.
