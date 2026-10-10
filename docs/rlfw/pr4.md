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
censored slots, terminal parity slots and unfinished/unattempted contexts. It
reports coverage only.

## Program parity outcomes (R5)

Bootstrap specs (`jope.rlfw-bootstrap-spec/v2`) carry the arena's
`rlfw-kpi/r5-program-parity-v1` version, and the shared oracle and outbox apply
it. A deterministic native/Rust mismatch is the program-level outcome
`parity_unqualified`: the logical slot is charged once, no q, q_opt or q_grid
is recorded, and the slot is never retried. Its permanent program fate and the
attempt's cost are saved before the charge, so a repeat of the same execution
identity reuses that fate without a new native request. The research ledger
records the event as a refusal without a numeric label. The context stays in the
wave, and its other programs remain measurable. The legacy `parity_custody`
retry outcome is refused under R5. `infra_unknown` keeps its bounded retries and
censoring.

## Campaign window (R21)

No deadline is written in the code. Each wave spec binds a
`jope.rlfw-bootstrap-campaign-window/v1` window with `created_unix`,
`deadline_unix`, `window_hours`, `boot_id` and `created_monotonic`. The window
lasts exactly 72 hours, and its deadline must equal creation plus 72 hours.
The CLI requires `--campaign-window`, a benchmark-local manifest whose
window equals the spec's. A missing, malformed or different manifest is refused
before any runtime import. The whole window is checked again immediately before
each guarded step: CLI start, each context before and after its admission
wait, every model dispatch (decode batch, head readout, calibration
probabilities) and every attempt reservation and evaluator request, including
retries within one slot. Each check requires the current wall clock and the
same-boot monotonic clock to lie inside the window; a closing window stops the
next step and keeps the outcomes and costs already recorded. After a reboot the
deadline cannot be verified, so the run refuses and
needs a new owner-approved window. The window starts only when the owner
approves bootstrap waves. The approved launcher writes
`CampaignWindow.opened_now().to_manifest()` once; no CLI path opens, renews or
extends a window.

## Acceptance

```sh
python3 deploy/dl-flywheel/pipelines/rlfw/run_rlfw_offline.py --receipt "$B/logs/bootstrap-fixtures.json"
just required
```

Registered synthetic fixtures cover eligibility, native identities, seeded
parent balance, mixed selection, probabilities/propensities, packet replay,
empty-arena admission, first-label CAS, failure NULLs, role changes, missing
closure, strict yield boundaries, R5 parity outcomes, the injected campaign
window, a scan for literal deadlines in the package source and partial/unknown
costs.
Their evaluator,
artifacts and probabilities are synthetic. They establish source behavior;
hosted exact-head CI and independent approval remain required.

## Live runtime requirements

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=deploy/dl-flywheel/pipelines \
"$RLFW_PYTHON" -m rlfw.no_winner_bootstrap --enable-no-winner-bootstrap \
  --spec "$B/runs/bootstrap_wave/spec.json" --campaign-window "$B/state/bootstrap-campaign-window.json" \
  --runtime-binding "$B/l0/bootstrap_binding.py" \
  --policy-code-root "$B/code_pin" --policy-qualification "$B/runs/cpu_qualified/qualification.json" \
  --device cpu --pool-workers 4
```

Run from the reviewed checkout. `RLFW_PYTHON` must select the qualified
interpreter and exact Torch runtime.
The CLI accepts `--device cpu|cuda`, default CPU. CUDA requires the complete
identical-pin likelihood/head parity certificate. All executions require the
bounded campaign scope, current health/memory guards, same-pin 1000-draw policy
qualification, pinned runtime dependencies and an open injected campaign window.
The oracle refuses execution at or after the window's deadline.

The sealed `build_bootstrap(spec, device=...)` runtime supplies the complete
catalog, sample/sample_many, score_heads, coverage_probabilities, fresh role/input
authority, qualified evaluator, reserve_empty_arena and arena_queue_empty. No such
qualified live bootstrap binding or calibration is currently installed. L0's
metadata census found 6857 train vectors and input locators, with zero explicit
no-winner status fields; file presence does not qualify prepared inputs or q.
The dated 5461-context estimate is not an eligible roster. Current native parity
STOP remains a hard refusal. No actual bootstrap wave, yield or training gain is
claimed from the fixtures.
