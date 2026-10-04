# Epsilon-lexicase parent selection (B10)

`EvolveConfig::selection` chooses how the next generation's parents are drawn in `evolve()` and the
lite race. The default, `ParentSelection::Tournament3`, is the selection the search has always run:
no case is captured, no selection stream exists, and every default output (surrogate and race
rankings, fold bits, memo dumps, the frozen `search_runs` and `generator_streams` fixtures) is
unchanged. `ParentSelection::EpsilonLexicase(LexicaseConfig)` is opt-in, engine-internal and
reserves no operator, wire, program or model ID. It is selected only by a caller that sets it; no
shipped profile or default changes.

## Selection core (`kernel/lexicase.rs`, profile `feat-math.parent-selection.epsilon-lexicase.v1`)

- **Cases and errors.** A case is one training row. A candidate's error on it is `|residual|`; a
  non-finite residual is `+inf`, which never passes a filter that a finite error fails.
- **Tolerance rules.** The rule is frozen per call:
  - `MedianAbsoluteDeviation` (default) is semi-dynamic ε-lexicase (La Cava et al. 2019,
    Algorithm 3, `≤` filter). `epsilon_j` is the MAD of case `j`'s finite errors over the whole
    pool, and the filter keeps `error ≤ best-in-current-pool + epsilon_j`. A finite best whose sum
    overflows is held at `f64::MAX`, so infinite errors still fail.
  - `Zero` is plain lexicase.
  - `MinimumVarianceThreshold` is the dynamic minimum variance threshold of Aldeia, de França and
    La Cava (GECCO 2024), with the split enumeration the paper leaves open frozen here:
    - splits fall between consecutive distinct finite errors of the current pool;
    - the objective is `var(left)/|left| + var(right)/|right|` with population variances (Welford
      sums from both ends);
    - the first strict minimum wins, and the left side survives.
- **Down-sampling.** `CaseSampling::DownSampled { per_mille }` (1..=1000) draws one ascending case
  subset per call.
- **Draw order.**
  - The subset comes first, by partial Fisher-Yates.
  - Each event then lazily shuffles the case list, one draw per examined case.
  - A tie draw comes last, and only when more than one candidate remains. A pool of one draws
    nothing.
  - Every bounded draw is the legacy `next_u32() % n`.
- **Pinned behaviour.** A fixed-seed trace pins the draw order. Any change needs a new profile
  identity.
- **Stepwise drawing.** `Selector` draws one event at a time with exactly the draws of `select`.

## Wiring (`evolve.rs`, marked `B10 lexicase: begin/end`)

- **Cases.** They are signed out-of-fold residuals `y[r] - prediction`.
  - **Surrogate search.** Rows are the first 256 rows of the seeded permutation the race uses
    (`permute_indices(n, seed)`), ascending.
  - **Race.** Rows are the rung-0 rows in position order, taken at the rung each candidate is
    ranked by.
  - **Fold isolation.** Every row is predicted only by models fitted on the other folds, so a
    row's own label never reaches its prediction. A test poisons one fold's labels and checks
    that fold's predictions stay bit-identical. No row outside the rows the search was given
    enters.
- **Sibling scorers.** `score_genome_predictions`/`score_genome_cases` and `score_rung_cases` run
  the legacy program, folds, fits, predictions and metric in the same order. Their fold scores
  are tested bit-equal to `score_genome`/`score_rung`.
- **Pool and stream.** The pool is every distinct ranked candidate of the generation. Each parent
  draw is one `Selector` event on the generation's own stream,
  `Pcg32::new(seed, LEXICASE_STREAM + generation)`, so the search stream draws only variation.
  Without lexicase, `pick_parent` is `select_parent` with identical arguments and draws.
- **Receipts.** `Observer::on_selection` receives one `SelectionReceipt` per lexicase generation
  (generation, pool, selection and trace). It is never fired under tournament-3. If a pool could
  not form one case matrix, the generation falls back to tournament-3 and the receipt names the
  reason; the wiring never produces that state, and a test covers the fallback directly.
- **Race scoring.** Under lexicase the race scores every rung on the CPU. The GPU rung-0 backend is
  never called, and exact rung-0 reuse is off with the receipt blocker "exact rung-0 reuse is off
  under lexicase selection: a reused score has no per-row residuals". Time-budgeted lexicase runs
  therefore give up GPU throughput.
- **Fault policy.** Lexicase requires `FaultPolicy::Legacy`. Other policies are refused: the search
  panics, as it does for an invalid fault policy, and
  `evolve_lite_race_with_exact_initial_population` returns the reason.
- **Case loss.** The case loss is the absolute residual, whatever the caller's metric. Receipts
  are for regression; AUC callers are not a B10 claim.

## Identity

`evolve.rs` is an ADR-0004 evaluator-identity input. The card's publication carries an announced
identity-only transition: `JOPE_COMPAT_FREEZE_REGENERATE=identity`, plus
`compat_freeze/transitions/b10_identity_rebaseline.v1.json` and the `historical_v1` identity
chain. Behavioural pins stay strict.

## Ablation

`six_tasks_five_seeds_equal_budget_selection_ablation` is `#[ignore]`d: it runs about six minutes
in a debug build and 24 s in release, with identical floating-point results. It runs six synthetic
regression tasks × five seeds at an equal generation budget. Both arms share the population (16),
generations (5), folds, data and search seed, and each winner's surrogate is refit and scored on
held-out rows.

Over all 30 runs:

| Metric | Lexicase (MAD) | Tournament-3 |
|---|---|---|
| Median held-out R² | 0.898 | 0.869 |
| Mean distinct ranked genomes in the final population | 14.9 | 12.6 |

Paired by run, lexicase is better, equal or worse in 6, 11 and 13 runs, and its mean
distinct-parent fraction is 0.48.

| Task | Lexicase mean R² | Tournament-3 mean R² |
|---|---|---|
| Two-regime | 0.855 | 0.822 |
| Periodic | 0.913 | 0.893 |
| Interaction | equal | equal |
| Heteroscedastic, linear, outliers | at most 0.005 lower | — |

The test asserts the card's two criteria: median R² is no worse, and the surviving population is
more diverse. This is synthetic numerical evidence. Real-campaign W §17 qualification and any
default change remain held, and tournament-3 stays the default.

## Evidence

`/home/ubuntu/jope-evidence/kernel2/bt/B10_LEXICASE/`.
