# Generalized, split-robust GP campaign

`jope_generalized_gp` is an additive regression/binary experiment. It does not
change Search V3, the legacy GP bridge, accepted checkpoints, production
pointers, or production-learning inputs.

## Benchmark inclusion

Catalog eligibility is `origin == real`; there is no minimum-row rule. Small
real datasets remain in the benchmark and produce explicit invalid evidence if
the evaluator cannot run them. In particular, a 300-row cutoff must never be
used to shrink the 42-dataset SOTArena regression benchmark or remove current
wins.

Production catalog cardinalities are separately checked against the frozen
17,263/5,030 development and 1,928/569 validation counts. The validation
partition cannot be planned until a content-addressed development-completion
receipt exists.

## Runner commands

```text
jope_generalized_gp validate-freeze \
  --campaign campaign.json \
  --catalog catalog.json \
  --checkpoints checkpoints.json \
  --evaluator evaluator.json \
  --verify-artifact-bytes

jope_generalized_gp launch-gate --evidence launch-gate.json

jope_generalized_gp plan-tranches \
  --catalog catalog.json \
  --campaign-sha256 <sha256> \
  --pilot-comparison-sha256 <sha256> \
  --arm rotating-mean \
  --partition development \
  --output development-tranches.json

jope_generalized_gp dry-run --output /new/output/directory
```

The dry run exercises one regression and one binary dataset through proposal
freeze, exact 64/10/6 generation zero, rotating splits, deterministic
offspring, archive confirmation, CV7 selection, and CAS publication. Its
evaluator scores are simulated and its receipt is always
`benchmark_valid=false`.

## Launch safety

The launch gate requires real U1 evaluator flow, fresh supply evidence, two
protected waves, reviewed SCQ acknowledgement, nonzero free cores, fresh
liveness, and configured storage headroom. A zero-free-core state or a disk
with only 7% free space is rejected. No pilot, development tranche, or
validation audit should be submitted until that gate passes.

