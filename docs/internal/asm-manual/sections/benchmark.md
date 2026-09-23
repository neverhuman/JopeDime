# Benchmark execution, ranking, Elo and interpretation

## Three evidence collections with different purposes

The frozen ASM panel, the historical same-cut benchmark and the imported SOTArena ladders have different inputs and protocols. Keep their figures and identities separate. A strong historical Elo is not a result for the frozen thirty-dataset ASM panel, which has not yet been run. A rank computed on development labels is not a sealed-test estimate.

| Collection | Inputs and purpose | Current interpretation |
| --- | --- | --- |
| Frozen `asm-private-30-v1` | Ten regression, ten binary and ten multiclass datasets; original frozen row assignments | Current implementation target; one regression operational proof, full panel unrun |
| Historical same-cut v1/v2 | Ten committed regression datasets; all methods share each cut; screening and deepening | Reproducible historical comparisons; v1 and v2 are separate snapshots |
| Historical tabular-regression-v2 | Imported score inventory and corrected benchmark-valid/public-development views | Historical ladder with its own coverage and validity rules; not a current ASM qualification |

The plot source tables retain method, dataset, snapshot, coverage and source hashes. Missing cells stay missing. The book does not manufacture thirty current outcomes from the historical collections. Historical method names are identifiers; the word “jope” in an identifier does not prove the method is this exact product checkpoint or executable.

## Frozen-panel execution contract

### The screen, freeze and confirm procedure

For each dataset, screen every applicable native method, configured external adapter and admitted search arm on development fold zero. Freeze the top five recipe identities and confirm those recipes across all five original development folds. Additionally confirm `asm.neural` if it missed the top five; the current code forces only `asm.neural`, not `asm.generic`, into this extra confirmation (`crates/jope-bench/src/panel/campaign.rs:779-783`). A configuration such as `gpr.rbf_linear` is distinct; a spelling alias does not justify an extra independent fit. The recipe is frozen, while the fitted state for each fold is learned from that fold's permitted training rows.

For each dataset, applicable methods screen on development fold zero. The top five recipes freeze before confirmation across all five development folds. `asm.neural` is also confirmed when it misses that top five. Report every original outcome. Selection can use adaptive development folds; these are not untouched external generalization estimates. Preserve the explicit selection-exposure metadata, and keep sealed evaluation untouched.

### Adaptive development exposure

This screen/freeze/confirm design uses development data adaptively. The selected recipe can already have learned from rows that occur in a later confirmation fold. Fold-zero labels influence screening. The implementation records these intersections; the five-fold summary remains adaptive development evidence. Sealed rows and sealed meta-test families stay outside learning, optimization and this documentation audit. The word “test” in an optimization-visible file does not make it a valid test set.

The frozen protocol screens methods on development fold 0, freezes the top five recipes, and confirms those recipes on the same five development folds; `asm.neural` is additionally confirmed if it missed the top five. The synthesized recipe was learned from the outer fold 0 fitting set, containing development folds 1–4. Thus those later confirmation folds are exposed to recipe learning. Every frozen recipe's fold 0 labels influenced screen selection. The report derives row intersections and records them per recipe/fold; it labels the selected five-fold summary adaptive development and leaves sealed evaluation unopened. This is the chosen protocol, not an unbiased outer generalization estimate. Source: `crates/jope-bench/src/panel/selection_exposure.rs:1`.

The exposure is not symmetric, so the adaptive bias has a direction: it favours `asm.*` over fixed-configuration comparators. A synthesized `asm.*` recipe is recorded with the learning rule `inner_selection_on_cut_zero_training_rows`: it was chosen by a search of up to 1,000 evaluations per arm on the fold 0 fitting rows, which are development folds 1–4. Every `rust.*` and `external.*` recipe is recorded as `fixed_before_current_dataset_screen`. All methods share only the fold 0 screen selection. The five-fold mean therefore compares an optimistic ASM estimate with comparator estimates exposed only through that screen, and the difference can grow as search pressure grows; an apparent panel win can arise from selection alone. Sources: `crates/jope-bench/src/panel/selection_exposure.rs:71`, `:91-92`; `crates/jope-bench/src/panel/asm_campaign.rs:37`; `crates/jope-bench/src/panel/optimize.rs:132-133`. A nested variant, with an independent ASM search inside each outer fold, is not implemented; requiring it before any external claim is a review proposal pending owner decision.

## Running the frozen-panel campaign: plan, execute and report

Prepare the qualified configuration before opening an execution window. It binds the actual context producer, source/worker/model inputs, numerical qualification and policy configuration. The standard comparison preregisters a 64-slot policy pool plus 16 exploration candidates against 80 generic candidates, with equal declared arm allowances. Policy-pool shortfalls carry `neural_rejection_fallback`; only `neural_policy` slots count as genuine checkpoint proposals. Generation-zero neural slots must actually fit; a proposal file or generic-only result cannot demonstrate neural consumption.

That preregistration applies in full only to regression at the pinned source `68f56404`. The accepted policy is regression-only, so binary and multiclass datasets have no neural arm, and their generic arm searches ridge heads only (`logit_ridge` or `multiclass_ridge` with at most four transforms; see [§](architecture.md#experimental-classification-context)):

| Task | Frozen datasets | Neural arm | Generic arm | Non-ASM comparators |
|---|---|---|---|---|
| Regression | 10 | 64 policy slots plus 16 exploration candidates | 80 grammar-uniform candidates | 17 native recipes plus 4 external engines |
| Binary | 10 | Unavailable | 80 `logit_ridge` candidates | 5 native recipes plus 4 external engines |
| Multiclass | 10 | Unavailable | 80 `multiclass_ridge` candidates | 4 native recipes plus 4 external engines |

At this source, a better policy therefore cannot change binary or multiclass results. Sources: `crates/jope-bench/src/panel/optimize.rs:34`, `:35`, `:493`; `crates/jope-pipeline/src/neural_search.rs:28`.

```sh
"$J" optimize plan \
  --base-plan "$A/base.json" --base-plan-sha256 "$BASE_PLAN_ID" \
  --configuration "$ASM_CONFIGURATION" \
  --configuration-sha256 "$ASM_CONFIGURATION_FILE_SHA256" \
  --output "$A/panel.json"

"$J" benchmark run --plan "$A/panel.json" --plan-sha256 "$PANEL_PLAN_ID" \
  --manifest "$M" --datasets "$D" --journal "$A/panel-journal"

"$J" benchmark report --plan "$A/panel.json" --plan-sha256 "$PANEL_PLAN_ID" \
  --manifest "$M" --datasets "$D" --journal "$A/panel-journal"
```

`optimize run` and `optimize report` delegate to the same campaign contracts. Repeating run with the same plan and journal resumes the original authenticated state and absolute deadline. A restart does not grant another two hours. Do not repair journals by editing receipt fields or copying successes from another experiment. Read-only report recovery authenticates retained artifacts and recomputes scores; it does not fit another predictor.

## Metrics and class order

| Task | Primary reported metric | Smaller-is-better search loss | Required prediction shape |
| --- | --- | --- | --- |
| Regression | RMSE | RMSE | One finite real prediction per row |
| Binary classification | AUROC | `1 − AUROC` | Probabilities with original two-class order |
| Multiclass classification | Log loss | Log loss | One probability for every original class on every row |

RMSE is the square root of the mean squared prediction error. It has the target's units, so a raw RMSE on machine CPU cannot be averaged meaningfully with an unrelated dataset's RMSE without an explicitly declared normalization. AUROC measures score ordering between positive and negative examples; the positive class and class order must be fixed by the task contract. A one-class evaluation fold cannot produce a meaningful AUROC. Multiclass log loss averages the negative logarithm of the probability assigned to the true class. Integer class predictions alone cannot supply that probability-matrix metric.

Probability shape, finiteness, class names and order belong to the authenticated prediction boundary. A backend that produces the same numbers in a different class order has failed parity. A backend with only scalar output cannot be advertised as supporting a multiclass winner. Any numerical clipping, normalization, accumulation, tie or tolerance rule must be fixed before scoring and recorded with the evaluator; changing the rule afterward changes comparison identity.

Regression is scored by RMSE. Binary classification uses AUROC, with the frozen second class as positive and loss `1 − AUROC`. Multiclass uses the full probability matrix and original class order for log loss. Predictions must be finite; class order and row count must match; probability rows are validated without silently renormalizing. The pinned log-loss epsilon is `1e-15`, and probability row-sum tolerance is `1e-6`.

Regression uses RMSE; binary uses AUROC with minimization loss `1−AUROC`; multiclass uses probability-matrix log loss with the frozen class ordering. Features retain numeric/categorical type and missing cells; original labels are not converted to manufacture a task. Screen all applicable methods on original fold zero, freeze five recipes, confirm across all five development folds, and confirm `asm.neural` additionally if it missed the five. These optimization-visible folds remain development evidence. Sealed evaluation is separate.

The historical same-cut command has an older contract. It constructs its own hashed cuts, uses regression R², binary AUROC and multiclass macro-F1 for screening, and applies its recorded robust-fitness summary during deepening. Its regression ladder uses RMSE; its classification ladder can maximize AUROC or macro-F1. Those historical definitions must not be silently copied into the frozen ASM panel's multiclass log-loss contract. Source: `docs/same-cut-bench.md` and `crates/jope-bench/src/eval.rs`; current panel code lives under `crates/jope-bench/src/panel/`.

## What Elo means here

### The Bradley–Terry model, tie rule and defaults

The arena implementation fits a regularized Bradley–Terry model on an Elo scale. For two ratings A and B, its modeled pairwise probability is:

`P(A beats B) = 1 / (1 + 10 ** ((rating_B − rating_A) / 400))`.

Each dataset with both eligible scores supplies one equally weighted comparison for that method pair. The observed outcome is 1 for a win, 0 for a loss and 0.5 for a tie. Metric direction is explicit. The relative tie condition is `abs(a − b) <= 1e-12 * max(abs(a), abs(b), 1)`. A missing or ineligible score supplies no invented numerical loss.

The default implementation starts and recenters ratings at 1500, runs at most 250 diagonal-Newton iterations, applies L2 regularization of `1e-4`, clips each iteration's update to ±40 Elo points and stops when the largest update is below `1e-8`. The ±40 setting is an iteration clamp, not an online match-by-match chess update. Source: `crates/jope-arena/src/elo.rs`.

Two source paths preserve historical arithmetic: one accumulates individual sorted dataset/pair matches in the original Python order, and the reporter aggregates each pair's sufficient statistics. They are equivalent in exact arithmetic but can differ slightly in floating-point accumulation. Reproduction must identify the path and snapshot, not merely say “Elo.”

### Interpreting a rating

The rating is relative to the included method universe, evidence eligibility and overlap graph. A high rating on a handful of datasets has different coverage from the same rating on every dataset. Report dataset coverage, full-coverage rank, average rank, wins, pending cells and protocol beside Elo. Adding methods, removing invalid rows or changing overlap can change ratings without any predictor changing. Elo does not measure the magnitude of an RMSE improvement, runtime, memory, statistical certainty or future unseen-data performance. No confidence interval or significance claim is implied by a plotted point.

### How the retained Elo scale is constructed

`crates/jope-arena/src/elo.rs` fits regularized Bradley–Terry ratings. Each dataset contributes one equal-weight comparison for each pair of methods with available scores. The preferred direction is taken from the metric; RMSE comparisons minimize. A tie has outcome 0.5 when `|a−b| ≤ 1e−12 × max(|a|,|b|,1)`; otherwise the better score receives 1 and the other 0. Missing cells create no comparison and must not become invented defeats.

The expected pairwise outcome is `P(A beats B)=1/(1+10^((rating_B−rating_A)/400))`. The committed implementation centers at 1500, uses an L2 regularization coefficient of 1e−4, caps diagonal-Newton iterations at 250, clamps each iteration's rating update to ±40, and stops when the largest update falls below 1e−8. It preserves sorted dataset/method-pair accumulation and the historical numerical summation policy for reproduction. This describes the source algorithm; the audit did not rerun it or infer convergence where the snapshot does not report it.

A rating difference summarizes outcomes within one comparison pool. It does not encode an absolute RMSE difference, probability of unseen-dataset superiority, confidence interval, or universal ranking across tasks. A method observed only on its screened favorable subset needs its coverage displayed beside its rating. The historical `afm.lite-race.v1.resampled` row describes that search procedure; it does not prove the current neural ASM model.

## Minimum complete future campaign report

For every one of the thirty immutable datasets, retain the input and cut identities; applicable, skipped and unavailable methods with reasons; screen outcomes; frozen finalists; all confirmation-fold outcomes; `asm.neural`'s extra confirmation when needed; original and fitted recipe identities; class order; selected loss and exposure labels; resource limits and original deadlines; failures and cancellations; actual fit/predict counts; and retained winner/backend outcomes. Report all losses and gaps, not only wins. A deadline-exhausted dataset remains in the denominator with an explicit incomplete outcome.

The owner's target of ten wins per task category is not yet a countable definition. The estimate that counts, the eligible comparator set, whether `asm.generic` is a comparator or an ASM control, the tie rule and whether an incomplete or unconfirmed dataset can count as a win are not fixed, and no code at merged main `a75e543d2` counts wins against the best non-ASM comparator. A review proposal, pending owner ratification and not adopted, defines PANEL-WINS(asm-panel-v1) as, per task, the number of the ten frozen datasets on which `asm.neural`'s confirmed five-fold mean development loss is strictly below the minimum over all eligible frozen non-ASM comparators (every `rust.*` and `external.*` recipe). Under that proposal, `asm.generic` is reported in its own column as an ASM control, ties and incomplete, failed or unconfirmed datasets count as not won, and the result is reported as (regression, binary, multiclass). The separate win columns are described in [§](learning.md#win-counting-and-a-proposed-acceptance-record). With exactly ten frozen datasets per task, ten wins per category would be a 30/30 sweep. It remains a measured target; the current evidence does not promise that result. The full-panel report, newly trained policy consumption, selected fitted-winner parity and serving promotion remain separate acceptance milestones.

Panel comparators are fixed configurations. Each native `rust.*` recipe keeps its registered configuration, and the four external engines (XGBoost, LightGBM, CatBoost and scikit-learn) run as `external.{engine}.default`; the campaign performs no per-dataset hyperparameter search for any comparator (a head's own internal fitting, such as `ridge_gcv`'s generalized cross-validation, is part of its fixed recipe). Each ASM arm, by contrast, searches for up to 120 seconds and up to 1,000 evaluations. Sources: `crates/jope-bench/src/panel/methods.rs:60-63`, `crates/jope-bench/src/panel/campaign_cli.rs:67-72`, `crates/jope-bench/src/panel/optimize.rs:132-133`. The strongest historical benchmark-valid method, TabFM Ensemble (rank 1, Elo 1934.9), is not a panel comparator; on that historical ladder the `jope-dime-regressor` entry is rank 31 at Elo 1687.0 ([§](gp-targets.md#why-the-two-historical-ladders-differ)). Ten panel wins per category against these default comparators would therefore not show that ASM beats the strongest known tabular methods. Tuned or foundation-model comparator tiers for a future `asm-panel-v2`, never retroactive to v1, are a review proposal pending owner decision.

Classification search is also narrower than its comparators. At merged main `a75e543d2`, binary and multiclass datasets have no neural arm, because the accepted policy is regression-only, and their `asm.generic` search samples only ridge heads (`logit_ridge` for binary, `multiclass_ridge` for multiclass). Their comparators are five binary or four multiclass native recipes plus the four external engines. Sources: `crates/jope-bench/src/panel/optimize.rs:35`, `:492-493`; `crates/jope-bench/src/panel/methods.rs:90-104`; [§](architecture.md#experimental-classification-context).
