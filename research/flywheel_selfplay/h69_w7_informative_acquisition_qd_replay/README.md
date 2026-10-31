# H69 — W7 informative acquisition, quality-diversity and replay

**State: CPU design preparation only.** This folder contains the W7 design
surface and an offline plan-printing runner skeleton. It does not read data,
load a model, fit a ranker, score a candidate, or execute a study. No outcome,
Elo, interval, gate result, or advance decision is claimed.

## Authority and pinned contract

The authority is `origin/main` at `1d249852f4080b512910a7303cd97c20d384f4f1`,
file `research/flywheel_selfplay/rd_breakthrough/PROPOSALS.md`, SHA-256
`f0526f20699dea3fe5259d10eee30483ac3c52655249906fd1ce79dd8e193e50`.
The W7 proposal and common experimental contract are copied verbatim below for
this preparation. A future execution must refresh this authority and record any
change before numerical input is read.

## Which prior waves W7 consumes

The proposal's t2058 execution order is W2 and W8 diagnostics first (in
parallel), prepare W1, then W3, W4, W6, W5, W7 and W9. W7 must wait for those
preceding decisions and admission gates. This is a dependency order, not
permission to import every prior wave's rows or labels:

- W1 establishes which prior/baseline policies are eligible. The W7 arm list
  remains the three arms printed verbatim below; no fourth arm is added here.
- W2's slate-consistency decision and W3's continuous-outcome/status decisions
  must be frozen before any W7 candidate scoring or replay construction.
- W4's four-total-evaluation history is distinct from W7's four new candidate
  outcomes after its declared four-anchor history.
- W5 and W6 are preceding waves in the approved execution order. Their reports
  may constrain the future recipe, but this W7 preparation imports no rows or
  labels from them.
- W8's training-only resplit diagnostic informs stability assumptions; resplit
  rows remain separate and are not extra independent families.
- H58 already tests all-completed replay. W7 tests acquisition and data value
  rather than claiming that replaying negatives is novel. W9 is downstream.

No upstream numerical result was read to prepare this folder. Future W7 work
must use the common 1,219 DL-Train contexts / 205-family row closure, original
labels, alias-resolved five-family folds (seed 4900), experiment seeds 701–703,
the frozen r882 continuity control, and its fold-excluded matched control.
Source-verified v3 additions need explicit admission and a dose-matched static
replay arm; score-body exports are not automatic inputs. No W7 materialization,
family assignment, fit, or outcome is authorized by this preparation. The
proposal's row-closure and S3 admission requirements remain prerequisites; this
skeleton does not establish them.

## Common experimental contract: advance gate (verbatim)

> The proposed advance gate is a positive adjusted family-mean lower bound above **+.001**, median family delta ≥0, and at most 10% of families losing more than .01, against the strongest matched control. A **+5 Elo** point gain is a practical target, not a forecast or substitute for raw utility. For search-primary waves also report legacy selection; a search-only gain is called a search gain. Before launch, a frozen fit-only pilot estimates detectable-effect curves and runtime. If the budget cannot identify the intended contrast, report an informative interval or defer the study; low power is not proof of a null. The historical live 1.13779 is a repeatedly selected incumbent value, not an independent target distribution. Rejection counts do not establish a capability ceiling, and no wave is judged solely by exceeding that scalar. No development gate authorizes serving promotion.

## W7 — Informative acquisition, quality-diversity and replay (verbatim)

- **Hypothesis:** spending some search budget on uncertain, structurally diverse candidates produces training data more useful than winner-only or uniform replay.
- **Arms:** r882/current two-greedy+two-random acquisition; matched all-completed uniform replay; and two exploitation + one disagreement/diversity + one uniform slot. Retain a genuine uniform-support slot. A finite descriptor archive (estimator family × AST-size bin) preserves useful alternatives instead of only the global best. Keep archive capacity and replay dose equal; diversity-only versus uncertainty-only is a deferred follow-up, not an unbudgeted grid.
- **Protocol:** seeds 701–703, five family folds, 20 acquisition rounds and 40 terminal updates. Predeclare 100 training-context rounds per arm and seed: three acquisition/replay arms × three seeds × four fits ×100 rounds = 3,600 fits, below the 4,800 ceiling; unused capacity is not automatically spent. Replay all completed outcomes with status masks, candidate sets and known selection probabilities; cap per-family/recent-record exposure. Compare any v3 import to an equal-sized/dose-matched static replay arm after deduplication. Freeze replay and model schedules before observing development outcomes.
- **Metrics / success / kill:** held-family Elo/val_reg_inner, U4, confirmed improvements per fit, effective new-label exposure and the same-data common-learner test. Advance only if the acquired data improve a common frozen-recipe learner over uniform fresh-label acquisition; a fuller archive or higher HWM count alone fails. Avoid inverse-propensity claims where historical support is unknown.
- **Data / budget:** approved training rows and source-verified v3 additions; score-body exports are optional prerequisites, never automatic inputs. ≤2 GPU-h, 4,800 native fits/four CPU-hours.
- **Support:** I-A3, I-O7 and I-AR3/I-AR7. H58 already tests all-completed replay; this wave isolates acquisition and data value rather than repeating “use negatives.”

## Preparation boundary

`runner.py --dry-run --json` validates the fixed design arithmetic and prints a
plan with `execution_admitted=false`. It does not access the filesystem beyond
its own code, inspect the approved S3 prefix, or create a fit, model, data, or
result artifact. Any data-backed execution requires a separate owner-approved
resume, row-closure receipt, source and resource freeze, and qualification-lock
slot. This H69 task performs none of those steps.

Run `bash runner_contract_test.sh` from this directory for the targeted,
offline contract test. It checks the printed design and verifies that omitting
`--dry-run` is refused; it makes no data or model request.
