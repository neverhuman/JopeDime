# Native feedback value guide: research result

Branch: `codex/native-feedback-upgrade-20261020`, based on main
`bccbb55d3fff0c03f4e90df48862273338d8c679`. This opt-in research adapter
learns program rankings from measured S3 outcomes while retaining both original
native models. Production defaults and model aliases are unchanged.

The S3 historical replay improved finite-pool selection against both native
checkpoints. It did **not** establish that new feedback beats extra optimization,
or that a live self-improving loop improves fresh GP high-water marks.

## Fixed trial and results

Inputs: 1,136 historical TRAIN contexts, 255 retained parent components and
18,210 completed program outcomes. Actual KumoS 2,051-dimensional encodings
replace the earlier study's hash-derived placeholder features. The archive's
original prospective provenance remains unresolved; this is a new explicitly
historical replay, not a repaired claim about that older study.

Parent-disjoint adapter split: 713 training, 209 unused calibration and 214
assessment contexts. Assessment contains 51 parent components. Both frozen native
models historically saw these contexts as TRAIN; this split withholds feedback
targets from the new adapter, not historical exposure from the native models.

Three ensemble members form one value policy. Initial acquisition reveals two
uniform outcomes per training context. Three waves reveal two more outcomes per
context, then train the adapter. Controls share identical initial weights and
AdamW state, and match the number of updates. Continuation reuses the first two
outcomes; uniform and targeted arms consume additional outcomes. Targeted
acquisition mixes 75% value-plus-uncertainty selection with 25% uniform draws.
Actual adapter updates: 5,760. New GP fits: zero; all outcomes are historical.

Budget four means selecting four programs from each identical frozen candidate
pool. The normalized regret is `(pool_best - selected_best)/(pool_best-pool_worst)`;
constant-quality pools have zero regret. Results average contexts within parents,
then parents equally. Lower is better.

| Policy | Parent-mean normalized regret |
| --- | ---: |
| Native 17,773-update finite-pool trie | 0.089554 |
| Native 429-update finite-pool trie | 0.121599 |
| Native 17,773-update quality head | 0.049531 |
| Native 429-update quality head | 0.068793 |
| Initial feedback adapter, frozen | 0.040239 |
| Adapter continuation, initial outcomes only | 0.040622 |
| Uniform feedback, fixed final checkpoint | 0.025939 |
| Targeted feedback, fixed final checkpoint | 0.021803 |

Uniform feedback reduced regret 47.6% against the stronger native quality head,
71.0% against the 17,773-step finite-pool trie and 78.7% against the 429-step trie.
Those native comparisons are exploratory, pointwise 99% parent bootstrap
intervals, without familywise adjustment.

The preregistered primary contrast, uniform feedback minus continuation, improved
normalized quality by 0.014683, with 99% interval **[-0.001536, 0.034345]**:
**inconclusive**. Targeted minus uniform is also inconclusive: 0.004136,
99% interval [-0.005081, 0.014491]. No favorable secondary endpoint replaces the
primary decision.

Raw incumbent-beating rates did not improve against the strongest frozen native
head. The candidate pools exclude historical incumbent programs, are finite and
already measured, and are not the original 6,640-program policy trie. No claim
about fresh raw-dataset quality, full decoder search, unseen families, GP HWM
promotion or amortized total-compute benefit follows from these results.

The 263-context teacher CE replay reproduced **0.547696302** and **0.956668408**
for the 17,773-step and 429-step models. Their earlier **0.897327233** and
**1.343790745** numbers are composite objectives, not CE. The value adapter
does not change either backbone or its teacher CE. The original diagnostic
expectation mistake remains in `teacher-kpi-replay.json`; the corrected receipt
is `teacher-ce-replay-corrected.json`.

## Implementation and verification

- `feedback.py`: retained-lineage partitioning, TRAIN-only measured target
  ledger, exact finite-trie probability, residual value adapter, ranking
  objectives, acquisition and parent bootstrap.
- `prepare.py`: pinned source/input/checkpoint verification, immutable
  preregistration and frozen feature/control extraction.
- `run.py`: matched initial state and update budgets, three feedback waves,
  safe tensor model/optimizer checkpoints and consumption records.
- `predict.py`: opt-in inference from actual Kumo encoding and program tokens;
  requires explicit checkpoint identities and never reads measured qualities.
- `audit.py`: reproduces acquisition choices using preceding child checkpoints,
  verifies consumed target ancestry and reloads the inference pathway.
- `teacher_replay.py`: original teacher-CE diagnostic with explicit archive field.
- `dltrain/`: unchanged source copied from main and verified against the S3 native
  campaign source. These files are baseline dependencies, not new architecture.

Nine unit tests passed. All nine arm/wave audits passed: actual parameter
changes, acquired TRAIN target consumption and exact next-wave selections.
Reloaded checkpoint inference reproduced the saved assessment selections with
all measured outcomes stripped from inference inputs. Teacher CE matches prior
explicit CE fields within 1e-5 for both native models.

Run with an existing CPU PyTorch/NumPy/safetensors environment:

```sh
rtk proxy env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  OPENBLAS_NUM_THREADS=2 python3 -m unittest discover -s . -p test_feedback.py -v
rtk proxy env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  OPENBLAS_NUM_THREADS=2 python3 prepare.py /absolute/new/artifact-directory
rtk proxy env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  OPENBLAS_NUM_THREADS=2 python3 run.py /absolute/new/artifact-directory
rtk proxy env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  OPENBLAS_NUM_THREADS=2 python3 audit.py /absolute/new/artifact-directory
```

Populate pinned input files before preparation; the script neither provisions
credentials nor downloads. Preregistration and experiment directories refuse
overwrite. Retain any failure and use a declared new revision, never reset a
failed scientific run. Frozen trial source hashes cover the original four
top-level Python files and baseline dependencies. `predict.py`, `audit.py` and
`teacher_replay.py` are post-run verification/delivery additions; they did not
alter the frozen trial or select its checkpoint.

Bulk artifacts live outside Git at
`/Users/bentaylor/Documents/ChatGPT/jope-research-data/native-feedback-upgrade-20261020`
and mirrored custody `/home/ubuntu/jope-evidence/native-feedback-upgrade-20261020`.
The delivery receipt records exact compressed storage, restoration and Git
identities. Do not infer an S3 output publication or receiving-main merge from a
local file. See `NEXT_WAVE.md` for remaining experiment and resource gates.

## Additive follow-up, 2026-10-20 09:32Z check

`next/` contains a durable measured-feedback journal and 17 passing fixture
tests. It charges failed/unresolved logical opportunities across restarts,
retains exact result bytes, and refuses incompatible comparison identities or
unmeasured rewards. Native teaching verification and executor integration remain
pending; its targets cannot admit training or publish HWM changes. See
`next/README.md` for source pins, limits and the next owner's exact work packet.
The completed historical trial and its results remain unchanged. The fleet
resource audit found the current controller does not qualify the requested
70% aggregate CPU ceiling; new GPU/GP training remains held for its owner.
