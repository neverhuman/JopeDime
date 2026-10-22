# Frozen entrant review

The supplied design is preserved as author claims. Integration changes only four
type annotations. External comparison folds still supply only fitting-side labels.

Internal OOF concerns are retained as properties of this entrant: preprocessing,
target transformation and feature screening use all supplied fitting rows before
its internal folds. Row-specific target encodings omit each row's own fold; a
base model's training rows may therefore encode labels from its validation fold.
The same validation rows select model hyperparameters and supply stacking scores.
Consequently the internal OOF score is not an unbiased performance estimate.
Independent external comparison folds remain necessary.

The supplied design's universal finite-output, runtime and memory guarantees are
not established by compilation or the small contract fixtures. The supervisor
must enforce one core and eight GiB per process tree and retain failures. No
scientific time ceilings are part of the successor experiment. Supplied telemetry
is estimated and inconsistent with actual source size; it is not inference timing.
