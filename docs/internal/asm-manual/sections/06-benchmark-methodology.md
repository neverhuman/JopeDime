# Benchmark protocol, rankings and interpretation

## Three evidence collections with different purposes

The frozen ASM panel, the historical same-cut benchmark and the imported SOTArena ladders have different inputs and protocols. Keep their figures and identities separate. A strong historical Elo is not a result for the current thirty-dataset ASM campaign. A rank computed on development labels is not a sealed-test estimate.

| Collection | Inputs and purpose | Current interpretation |
| --- | --- | --- |
| Frozen `asm-private-30-v1` | Ten regression, ten binary and ten multiclass datasets; original frozen row assignments | Current implementation target; one regression operational proof, full panel unrun |
| Historical same-cut v1/v2 | Ten committed regression datasets; all methods share each cut; screening and deepening | Reproducible historical comparisons; v1 and v2 are separate snapshots |
| Historical tabular-regression-v2 | Imported score inventory and corrected benchmark-valid/public-development views | Historical ladder with its own coverage and validity rules; not a current ASM qualification |

The plot source tables retain method, dataset, snapshot, coverage and source hashes. Missing cells stay missing. The book does not manufacture thirty current outcomes from the historical collections. Historical method names are identifiers; the word “jope” in an identifier does not prove the method is this exact product checkpoint or executable.

## Frozen-panel execution contract

The panel is immutable once comparison begins. Its identity covers dataset metadata, original labels and types, row IDs, group membership, original development folds, sealed membership and prepared data hashes. Do not replace difficult datasets, reshuffle rows, map categories to arbitrary numeric order, drop missing rows or substitute a different target to make a task executable. Missing source subject or time identifiers remain limitations even when duplicate-feature groups are protected.

For each dataset, screen every applicable native method, configured external adapter and admitted search arm on development fold zero. Freeze the top five recipe identities and confirm those recipes across all five original development folds. Additionally confirm ASM if it missed the top five. A configuration such as `gpr.rbf_linear` is distinct; a spelling alias does not justify an extra independent fit. The recipe is frozen, while the fitted state for each fold is learned from that fold's permitted training rows.

ASM compares a 64-slot policy pool plus 16 exploration generation-zero slots with 80 generic slots. The pool retains `neural_policy` versus `neural_rejection_fallback` origins explicitly. Both arms receive the same declared search and fit allowances, but encoding and proposal work consume the neural arm's own clock; fitting time can differ. A producer refusal, duplicate or failed candidate is an outcome to record. It cannot silently become another neural success or a free replacement arm. Actual inner-fold fits determine selection and offspring ancestry. A model's predicted reward never substitutes for measured loss.

This screen/freeze/confirm design uses development data adaptively. The selected recipe can already have learned from rows that occur in a later confirmation fold. Fold-zero labels influence screening. The implementation records these intersections; the five-fold summary remains adaptive development evidence. Sealed rows and sealed meta-test families stay outside learning, optimization and this documentation audit. The word “test” in an optimization-visible file does not make it a valid test set.

## Metrics and class order

| Task | Primary reported metric | Smaller-is-better search loss | Required prediction shape |
| --- | --- | --- | --- |
| Regression | RMSE | RMSE | One finite real prediction per row |
| Binary classification | AUROC | `1 − AUROC` | Probabilities with original two-class order |
| Multiclass classification | Log loss | Log loss | One probability for every original class on every row |

RMSE is the square root of the mean squared prediction error. It has the target's units, so a raw RMSE on machine CPU cannot be averaged meaningfully with an unrelated dataset's RMSE without an explicitly declared normalization. AUROC measures score ordering between positive and negative examples; the positive class and class order must be fixed by the task contract. A one-class evaluation fold cannot produce a meaningful AUROC. Multiclass log loss averages the negative logarithm of the probability assigned to the true class. Integer class predictions alone cannot supply that probability-matrix metric.

Probability shape, finiteness, class names and order belong to the authenticated prediction boundary. A backend that produces the same numbers in a different class order has failed parity. A backend with only scalar output cannot be advertised as supporting a multiclass winner. Any numerical clipping, normalization, accumulation, tie or tolerance rule must be fixed before scoring and recorded with the evaluator; changing the rule afterward changes comparison identity.

The historical same-cut command has an older contract. It constructs its own hashed cuts, uses regression R², binary AUROC and multiclass macro-F1 for screening, and applies its recorded robust-fitness summary during deepening. Its regression ladder uses RMSE; its classification ladder can maximize AUROC or macro-F1. Those historical definitions must not be silently copied into the frozen ASM panel's multiclass log-loss contract. Source: `docs/same-cut-bench.md` and `crates/jope-bench/src/eval.rs`; current panel code lives under `crates/jope-bench/src/panel/`.

## What Elo means here

The arena implementation fits a regularized Bradley–Terry model on an Elo scale. For two ratings A and B, its modeled pairwise probability is:

`P(A beats B) = 1 / (1 + 10 ** ((rating_B − rating_A) / 400))`.

Each dataset with both eligible scores supplies one equally weighted comparison for that method pair. The observed outcome is 1 for a win, 0 for a loss and 0.5 for a tie. Metric direction is explicit. The relative tie condition is `abs(a − b) <= 1e-12 * max(abs(a), abs(b), 1)`. A missing or ineligible score supplies no invented numerical loss.

The default implementation starts and recenters ratings at 1500, runs at most 250 diagonal-Newton iterations, applies L2 regularization of `1e-4`, clips each iteration's update to ±40 Elo points and stops when the largest update is below `1e-8`. The ±40 setting is an iteration clamp, not an online match-by-match chess update. Source: `crates/jope-arena/src/elo.rs`.

Two source paths preserve historical arithmetic: one accumulates individual sorted dataset/pair matches in the original Python order, and the reporter aggregates each pair's sufficient statistics. They are equivalent in exact arithmetic but can differ slightly in floating-point accumulation. Reproduction must identify the path and snapshot, not merely say “Elo.”

The rating is relative to the included method universe, evidence eligibility and overlap graph. A high rating on a handful of datasets has different coverage from the same rating on every dataset. Report dataset coverage, full-coverage rank, average rank, wins, pending cells and protocol beside Elo. Adding methods, removing invalid rows or changing overlap can change ratings without any predictor changing. Elo does not measure the magnitude of an RMSE improvement, runtime, memory, statistical certainty or future unseen-data performance. No confidence interval or significance claim is implied by a plotted point.

## Reading the performance figures

Historical ladders retain their original snapshot names. Benchmark-valid and public-development views are separated. Same-cut screening includes all admitted methods; deepening includes only finalists, so empty cells have a deliberate meaning. A rank heatmap compares methods within a dataset and stage; it is not a plot of absolute loss across datasets. Any chart with only the top methods states that limit, while its CSV preserves the full underlying inventory.

The current operational figure contains only the two development RMSE values from the preselected `230_machine_cpu` proof: neural 166.3181717454848 and generic 155.25909998168862. Lower is better. Generic wins that comparison. It would be misleading to compute a new thirty-dataset Elo from these two values or describe the 832 fitted predictors as 832 independent datasets.

The corpus-size figures are metadata/development statistics, not performance. The language coverage figure counts admitted source recipes under additional state restrictions; it is not a backend success rate. The architecture diagram is explanatory, and the completion matrix is an evidence inventory; neither is an experimental result.

The arm order is fixed: neural first, then generic. Fresh context encoding and policy production run inside the neural arm clock, and each arm is clipped to the original execution cutoff. In the accepted 68 proof the producer gate opened 21.094 seconds after the campaign start and its exit was observed at 41.581 seconds; its measured wall time was 20.486968041 seconds. The neural cutoff then lay 78.613 seconds ahead, before allowing for post-exit verification/setup, so this is remaining arm time rather than guaranteed fit execution time. The neural arm stopped at `original_deadline`. The generic arm began 0.553 seconds after that cutoff, retained its full declared 120-second allowance and stopped at `evaluation_limit`. Equal declared allowances are not equal fit-only time. Sources: `crates/jope-pipeline/src/neural_search.rs:654`, `:701`, `:729`, `:772`; `crates/jope-pipeline/src/execution.rs:592`; the accepted proposal execution and recovered report.

## Resource limits and accounting

The default admitted panel allocation is twenty CPUs/workers, 64 GiB memory, zero swap, one RTX 3090 when GPU work is explicitly allocated, thirty seconds per fit and 120 seconds per search arm. The full panel receives two hours total, including a five-minute reporting reserve. Training receives one GPU-hour total and twenty minutes per category. Do not overlap numerical campaigns with exclusive full-CPU CI. Resume continues the original campaign deadline and resource envelope.

Report elapsed wall time, summed worker wall time, worker CPU time, aggregate supervisor CPU, GPU time if measured, launched commands, successful fits, failures, unique fitted predictors and unique reusable recipes separately. Aggregate supervisor CPU contains worker CPU; adding them double-counts. Parallel worker durations do not equal elapsed wall time. Deduplicated/cache reuse is useful but is not another physical fitted predictor.

The current one-dataset proof completed 964 worker commands: 834 completed and 130 failed. It counted 832 unique fitted predictors and 164 recipes, plus two final refits with 34 development predictions each. Its run took 239.143121970 seconds; independent report recovery took 18.718997325 seconds. These values are measured at the qualified source and protocol, not projected full-panel throughput. The status chapter gives CPU totals and amortized denominators.

Every one of those 130 failed commands exited 1 with a verified stderr message `generated dense width N exceeds 512`. They represent 26 distinct candidate recipes with five failed folds each. Generation zero included 16 of 64 genuine neural-policy candidates failing this way (25%), compared with 4 of 80 generic candidates (5%); the remaining failures came from exploration or offspring. These jobs reached their process gates, but the failed candidates lack complete physical-fit counters, so the corresponding per-origin total remains unknown rather than inferred as zero or five. The 512-column check occurs inside fit execution; a verified prelaunch width rejection remains an optimization opportunity. Sources: `crates/feat-math/src/advanced_v3/feature_block.rs:109`, `crates/feat-math/src/grammar.rs:2300`; retained 130 finished/stderr hashes and report origin counts.

The supplied one-hour/ten-million-token scenario is a contextual scenario only. Its workload, model, hardware, predictor inventory and token/cost traces are unavailable. This proof used approximately 6.64% of an hour for its run, but that fraction is not a matched speedup or token-efficiency result. Dollar cost and token usage remain unmeasured.

## From optimization to reusable knowledge

The pending strict HWM key includes task, dataset hash, cut identity, evaluator, vocabulary and protocol. Scores from different identities cannot overwrite each other or supply training feedback. A recipe must show finite train-side out-of-fold improvement over a trivial baseline and improve on four of five folds before teaching admission. No-signal data should be refused by this gate. An optimization-visible test label is development evidence even if its filename says otherwise.

Typed motifs can reuse eligible program fragments and macros while retaining provenance. The prepared motif library does not yet establish the production eligible-state reader, `motifs mine` integration or operational reuse. A new operator requires compatibility tests and an ablation under the unchanged comparison protocol. Repeatedly selecting the best development result can improve development performance without proving improved sealed generalization.

## Minimum complete future campaign report

For every one of the thirty immutable datasets, retain the input and cut identities; applicable, skipped and unavailable methods with reasons; screen outcomes; frozen finalists; all confirmation-fold outcomes; ASM's extra confirmation when needed; original and fitted recipe identities; class order; selected loss and exposure labels; resource limits and original deadlines; failures and cancellations; actual fit/predict counts; and retained winner/backend outcomes. Report all losses and gaps, not only wins. A deadline-exhausted dataset remains in the denominator with an explicit incomplete outcome.

Ten wins per category is a measured target. The current evidence does not promise that result. The full-panel report, newly trained policy consumption, selected fitted-winner parity and serving promotion remain separate acceptance milestones.
