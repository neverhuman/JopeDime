# H70 — W1 CPU gate calculator

## Hypothesis

W1 results should never produce an advancement verdict unless every registered
primary contrast clears the frozen family-level gate and the legacy no-harm
check. A small, strict CPU calculator can prevent a summary-only or partial
result from being promoted as `PASS`.

## Preregistered gate

The source is `research/flywheel_selfplay/rd_breakthrough/PROPOSALS.md`,
Common experimental contract and **W1 — Prior replacement and neural
residual**. The calculator requires four paired family-level contrasts:

- U4 residual versus program-only GBM;
- U4 residual versus program-plus-context GBM;
- legacy-selection residual versus program-only GBM;
- legacy-selection residual versus program-plus-context GBM.

Each contrast must provide an already adjusted family-mean lower bound,
median family delta, and count of families losing more than `.01`. It must also
bind the fixed 10,000-draw family bootstrap (seed `20261031`) and state that all
registered contrasts were adjusted together. The common gate is adjusted lower
bound `> +.001`, median delta `>= 0`, and no more than 10% of the full
205-family training roster losing more than `.01`.

The calculator emits `ADVANCE_SEARCH_ONLY` only when both U4 contrasts pass and
the worse of the two legacy lower bounds is `> -.005`. It emits
`ADVANCE_LEGACY` only when both legacy contrasts pass their own common gate.
Otherwise it emits `NO_ADVANCE`. Missing or malformed evidence refuses with a
nonzero exit; partial evidence never becomes a positive verdict.

## Scope and limit

This is an independent result-gate utility in H70's owned folder. It does not
modify or certify the W1 runner, prove input or model provenance, establish
correct lock coverage, or qualify an experiment. Evidence is supplied as a
JSON summary; the tool does not open tables, checkpoints, S3, held-out or sealed
data. Tests use synthetic metric records only. W1 remains held until the
separate source, budget, custody, and lock findings are resolved by its owner.

Run the synthetic CPU checks with:

```bash
nice -n 10 python3 -m unittest research.flywheel_selfplay.h70_w1_prelaunch_gate_audit.gate_checks -v
```

The `PASS` scenarios in unit tests validate gate arithmetic only; they are not
W1 results or experiment evidence. No GPU or `qualification.lock` is used.
