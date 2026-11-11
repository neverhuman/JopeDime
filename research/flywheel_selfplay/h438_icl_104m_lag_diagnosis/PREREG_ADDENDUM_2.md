# H438 PREREG addendum 2 — how far the weights move, from stored checkpoint copies (descriptive; the verdict does not change)

**Status.** Published on `origin/v1/next` after the H438 closeout (`cb1595161f`, verdict `OPTIMISATION_UNSEPARATED`)
and before any checkpoint is opened for this wave. The PREREG said no checkpoint would be opened; this addendum
changes that, for one extra reading. **The verdict and every number of the closeout stay as they are.**

## 1. Why
The closeout left explanation (b) (peak learning rate) as something stored logs cannot read. The order allowed
stored checkpoints. The advisor note A-006 of `agy_apex_w10` (read after the closeout) proposes one checkpoint
number that bears on it without a forward pass: how far the weights move, relative to their size.

## 2. Inputs
The monitor's stored checkpoint copies of hours 12, 18 and 24, each with its sha256 file, for the 24M run
(`M_r3_h012/h018/h024.pt`) and for 104M worker 2 (`L_w2_h012/h018/h024.pt`), under
`~/jope-data/apex-icl/diag/mon/`. Raw weights and EMA weights. CPU, two threads; tensors are subtracted; no model
is built, nothing is evaluated.

## 3. Numbers
Per model and weight set, for 12→18 and 18→24: relative displacement ‖θ_b − θ_a‖ ÷ ‖θ_a‖ over all floating-point
parameters; the optimizer steps in between; the displacement per 10,000 steps, per square root of 10,000 steps and
per training hour; the cosine between the two successive displacements; the same displacement by top-level module.

## 4. Reading (fixed here; a label, not a verdict)
R = (104M displacement per 10,000 steps over 18→24) ÷ (the 24M one), raw weights.
| Label | Condition | What it would suggest |
| --- | --- | --- |
| `MOVES_LESS_PER_STEP` | R < 0.5 | the larger model takes much smaller relative steps: the peak-rate contrast (queue item C) moves ahead of the averaging proxy (B) |
| `SIMILAR_STEP_SIZE` | 0.5 ≤ R ≤ 2 | no sign of a different effective step size; the queue order stays |
| `MOVES_MORE_PER_STEP` | R > 2 | the larger model moves more per step; with the flat readout that would point to noise |

- The cosine tells which normalisation to trust: near 1, movement adds up with steps (per-step comparison is fair);
  near 0, it is a random walk and grows with the square root of steps (then the square-root figure is the fair one).
  Both are reported.

## 5. The other suggestion of A-006 (matching on tokens per parameter): not adopted
At the 104M group's tokens per parameter of hour 24 (about 3,000) the 24M run was at about step 116,000, **before
it switched to the current recipe** (step 138,685): it had not yet been trained on residual tasks, so its residual
readout there is not a control for anything. The closeout already reports tokens per parameter (0.28×) and the two
step matchings.

## 6. Weaknesses
- One pair of runs; the two models differ in peak rate (1.5e-4 against 2.0e-4), size and averaging at once, so R
  mixes all three. A relative displacement is not a learning rate.
- The two intervals hold different step counts (about 43,000 against 96,000), so the per-step figure depends on
  the normalisation; that is why both are given.
- Thresholds 0.5 and 2 are my choices. The label only orders the queue.

## 7. Gate
`gates3_h438.sh`, before the read: this addendum and `disp_h438.py` published and equal to the staged copies; the
six checkpoint copies present with their sha256 files; no displacement result exists; published stage.
