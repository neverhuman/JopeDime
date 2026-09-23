# Measured performance: the current proof and historical rankings

The current source (reviewed `68f56404`, merged as main `a75e543d2` with the same tree) has a successful operational proof on `230_machine_cpu`: the verified encoder produced context; the neural producer supplied actual generation-zero fits; a fitted descendant followed; two fitted refits each returned 34 development-hold predictions. The neural refit did not beat the generic arm, which had the same declared allowance but more fit time; this is one budget-mismatched sample, not a matched quality comparison. This proof uses the accepted checkpoint, not a newly trained experimental export. As of 2026-09-23, new training, final installed warm demonstrations and the frozen thirty-dataset panel (asm-private-30-v1) had not started, and no current thirty-dataset ranking or Elo exists.

## Reading the performance figures

Historical ladders retain their original snapshot names. Benchmark-valid and public-development views are separated. Same-cut screening includes all admitted methods; deepening includes only finalists, so empty cells have a deliberate meaning. A rank heatmap compares methods within a dataset and stage; it is not a plot of absolute loss across datasets. Any chart with only the top methods states that limit, while its CSV preserves the full underlying inventory.

The current operational figure contains only the two development-hold RMSE values (n = 34) from the preselected `230_machine_cpu` proof: neural 166.3181717454848 and generic 155.25909998168862. Lower is better. Generic has the lower RMSE in this one budget-mismatched, one-dataset sample; it is not a matched quality comparison ([§](#proof-record-and-outcome-figure)). It would be misleading to compute a new thirty-dataset Elo from these two values or describe the 832 fitted predictors as 832 independent datasets.

The corpus-size figures are metadata/development statistics, not performance. The language coverage figure counts admitted source recipes under additional state restrictions; it is not a backend success rate. The architecture diagram is explanatory, and the completion matrix is an evidence inventory; neither is an experimental result.

## Measured current-source operational proof

### Proof record and outcome figure

| Quantity | Measured result |
|---|---|
| Source | `68f56404c3e199e06413bfd645d3fe0803d7b96f` |
| Dataset | `230_machine_cpu`, regression |
| Immutable build wall seconds (preparation; not a cold-install measurement) |170.4488760890672|
| Eight-case native context qualification wall seconds (separate preparation) |29.248627813067287|
| Run wall seconds |239.14312197000254|
| Commands started / finished |964 / 964|
| Completed / failed command outcomes |834 / 130|
| Train OOF / diagnostic / refit-screen commands |960 / 2 / 2|
| Unique complete fitted predictor artifacts |832|
| Unique reusable recipes |164|
| Successful final refits |2, each 34 predictions|
| Neural development-hold RMSE (n = 34) |166.3181717454848|
| Generic development-hold RMSE (n = 34) |155.25909998168862|
| Generic minus neural loss |−11.059071763796169; neural did not win|
| Neural ÷ generic RMSE |1.071 (neural 7.1% higher; `neural_win=false`)|
| Known worker CPU seconds |181.220773|
| Sum of worker wall seconds |202.266164543|
| Exact supervisor aggregate CPU seconds |270.889403|
| Unknown worker CPU / wall counters |0 / 0|
| Teachers admitted from this proof |0|

The two arms had the same declared allowance but not the same work. Per arm, from the recovered report and its audit:

| Quantity | Neural arm | Generic arm |
|---|---|---|
| Declared arm clock | 120 s | 120 s |
| Work inside the arm clock before any fit | Context encoding and policy production; producer exit observed 41.581 s after campaign start, leaving at most 78.613 s to the neural cutoff ([§](07-policy-and-search.md#arm-order-and-clocks)) | No context or proposal production (`proposal: null`) |
| Evaluation cap | 96 slots: 80 generation zero, 8 mutation, 8 crossover (one offspring wave) | 96 slots, same split |
| Stop reason | `original_deadline` | `evaluation_limit` |
| Completed five-fold unique recipes | 77 of 96 | 89 of 96 |
| Generation-zero completions | 62 of 80: `neural_policy` 48 of 64, `random_exploration` 14 of 16 | 76 of 80 `generic` |
| Generation-zero dense-width failures | 16 of 64 `neural_policy` | 4 of 80 `generic` |
| Selected recipe | `ad665e5b…`: generation-1 crossover of two generation-0 `random_exploration` candidates; no `neural_policy` ancestor | `c806c3be…`: generation-1 mutation of a generation-0 `generic` candidate |
| Selection score of the selected recipe (mean of five inner-fold RMSEs on the same 135 cut-0 fitting rows and inner split) | 61.776 | 65.362 |
| Refit and scored rows | Refit on the 135 cut-0 fitting rows; scored on the 34 held rows of development fold 0 | Same rows |
| Development-hold RMSE | 166.3181717454848 | 155.25909998168862 |
| Neural ÷ generic RMSE | 1.071 | — |

Generic has the lower development-hold RMSE in one budget-mismatched, one-dataset sample; this is not a matched quality comparison. Because the neural arm's selected recipe has no `neural_policy` ancestor, its refit score measures the arm as a whole, not a program the policy proposed. Not computed or not recorded for this proof: a mean-predictor (trivial-baseline) RMSE on the same 34 rows, a paired bootstrap interval for the RMSE difference, and a comparison at equal completed evaluations. Numeric parity of the Candle CPU policy port that produced the proposals with the trained tch forward pass is not established, and the port's `cuda` feature has never been compiled (`crates/jope-policy/src/lib.rs:84-95` at `a75e543d2`); the report records `ranking_backend: candle_cpu_float32` and `native_cuda_parity_qualified: false`.

This is the current bounded operational proof, with one preselected regression dataset and equal declared search-arm allowances but unequal fit time. It demonstrates authenticated neural-origin fits and fitted descendants. The complete thirty-dataset ASM panel has not run, so this figure cannot establish its wins, ranking or Elo.

![Current operational proof: neural and generic development-hold RMSE (n = 34)](../figures/current-origin-loss.png)

The two underlying records are in `data/plot-current-origin.csv`; exact evidence and denominators are in `data/performance-evidence.json.gz`. There are 34 predictions per final refit. The proof's run wall time is 239.143 seconds; its independent recovered report takes another 18.719 seconds. CPU, wall-time and per-predictor accounting follows in [§](#cpu-wall-time-and-per-predictor-accounting); the status summary is [§](01-current-status.md#latest-measured-operational-proof).

### CPU, wall time and per-predictor accounting

The current one-dataset proof completed 964 worker commands: 834 completed and 130 failed. It counted 832 unique fitted predictors and 164 recipes, plus two final refits with 34 development-hold predictions each. Its run took 239.143121970 seconds; independent report recovery took 18.718997325 seconds. These values are measured at the qualified source and protocol, not projected full-panel throughput. The CPU totals are in the proof record above and the amortized denominators in the table below.

The supervisor figure is the exact `CPU_USAGE_NSEC=270889403000` event from local PID 1 systemd, linked to the precise supervisor unit, source, launch receipt and original experiment window. It includes controller/setup threads and child workers and excludes the outer launcher, immutable build and external preparation. It overlaps the worker CPU total; **never add them**. The rounded systemd stderr message is only diagnostic. No exact controller/setup overhead is derived by subtracting a rounded total. Isolated controller CPU, GPU seconds, token usage and currency cost remain unknown in the report.

| Denominator | Worker CPU seconds | Supervisor aggregate CPU seconds | Amortized run wall seconds |
|---|---:|---:|---:|
| One unique fitted predictor |0.2178134291|0.3255882248|0.2874316370|
| One unique recipe |1.1050047134|1.6517646524|1.4581897681|

These are accounting ratios for one retained run, not extrapolated throughput or equal-quality claims. A recipe and a fitted predictor differ: the same recipe can create distinct fitted artifacts on separate folds. Unique artifact bytes do not prove behavioral uniqueness. Every failed fit remains in the inventory; failed or unavailable commands are not silently counted as zero cost.

### Failures: dense-width refusals and the earlier failed reproducer

Every one of those 130 failed commands exited 1 with a verified stderr message `generated dense width N exceeds 512`. They represent 26 distinct candidate recipes with five failed folds each. Generation zero included 16 of 64 genuine neural-policy candidates failing this way (25%), compared with 4 of 80 generic candidates (5%); the remaining failures came from exploration or offspring. These jobs reached their process gates, but the failed candidates lack complete physical-fit counters, so the corresponding per-origin total remains unknown rather than inferred as zero or five. The 512-column check occurs inside fit execution; a verified prelaunch width rejection remains an optimization opportunity. Sources: `crates/feat-math/src/advanced_v3/feature_block.rs:109`, `crates/feat-math/src/advanced_v3/grammar.rs:2299-2300`; retained 130 finished/stderr hashes and report origin counts.

The earlier failed reproducer at source `d0f5748b7db1` failed after 194.789382994 seconds at scorer replay. All 964 commands had terminal receipts, with 834 completed and 130 failed. Its exact supervisor CPU was 252.614611 seconds and worker CPU 181.863951 seconds. It remains a failed reproducer; the old journal and deadline were not repaired or reopened.

### The one-hour, ten-million-token reference scenario

The supplied one-hour/ten-million-token scenario has 3,600 seconds and 10,000,000 tokens by stipulation. This run used 239.143 seconds of wall time but has no measured token or monetary receipt. A dollar comparison, token savings claim, or one-hour quality extrapolation would require actual token usage, contemporaneous input/output/cache rates and an equivalent task/quality protocol. None is invented here.

The supplied one-hour/ten-million-token scenario is a contextual scenario only. Its workload, model, hardware, predictor inventory and token/cost traces are unavailable. This proof used approximately 6.64% of an hour for its run, but that fraction is not a matched speedup or token-efficiency result. Dollar cost and token usage remain unmeasured.

## Existing historical rankings and Elo

### Retained snapshots and plot data

Six committed snapshots are available and their ladder and score SHA-256 values were verified against retained bytes. No new Elo computation occurred in this audit. The snapshots use their own method pools and dataset protocols, so their Elo numbers cannot be compared across snapshots as a universal score.

| Snapshot | Rankable rows | Methods | Datasets | Interpretation |
|---|---:|---:|---:|---|
| same-cut-v1-screen |110|11|10|historical_same_cut_reproducible_snapshot_not_current_ASM30|
| same-cut-v1-deep |50|11|10|historical_same_cut_reproducible_snapshot_not_current_ASM30|
| same-cut-v2-screen |150|15|10|historical_same_cut_reproducible_snapshot_not_current_ASM30|
| same-cut-v2-deep |50|15|10|historical_same_cut_reproducible_snapshot_not_current_ASM30|
| historical-sotarena-published |8763|305|45|historical_published_best_known_test_selected_not_benchmark_valid|
| historical-sotarena-benchmark-valid |8718|304|45|historical_corrected_public_development|

For plots, use `performance-evidence.json.gz`: 6 complete ladder row sets; 520 same-cut score cells, including missing/rejected statuses; 20 per-dataset reports with exact cut identities, source heads and screen/deep timings; 8,763 historical SOTArena score rows; and the 2 current-origin refit loss rows. Do not blend these into one chart without explicit separate protocol panels. The frozen panel (asm-private-30-v1) matrix had no completed cells as of 2026-09-23; represent it as pending, not a fabricated ranking.

### Ranking figures: interpretation contract

The historical ranking plots that follow use their separately pinned snapshots. They are not new model runs and are not a frozen-panel (asm-private-30-v1) campaign. Benchmark-valid and public-development views retain their original eligibility distinctions. Same-cut deep-stage coverage varies because only finalists were deepened; absent cells are not zero losses. See [§](08-evaluation-protocol.md#the-bradleyterry-model-tie-rule-and-defaults) for the exact Elo model and tie rule, [§](08-evaluation-protocol.md#interpreting-a-rating) for coverage limits and [§](08-evaluation-protocol.md#adaptive-development-exposure) for the adaptive-development interpretation.

All machine-readable rows are retained in `data/performance-evidence.json.gz` and `data/plot-historical-elo.csv`. A displayed top-method subset does not remove the other methods from those inventories. Dataset selection and historical protocol caveats travel with each figure.

### SOTArena published and corrected ladders

#### Historical benchmark-valid Elo

This view retains 304 methods and the historical corrected eligibility filter. The historical `jope-dime-regressor` entrant is rank 31 with Elo 1687.0 in this snapshot. “Benchmark-valid” means no retained artifact contradicts a row's flag; comparator protocols were not independently verified. It does not certify that every comparator obeyed today's frozen ASM protocol.

![Historical benchmark-valid Elo ladder and complete rank distribution](../figures/fig01-elo-benchmark-valid.png)

The top-method view is paired with the distribution of the full method universe. Dataset coverage and the identity of the historical entrant remain visible. These numbers describe the retained historical ladder, not the current accepted step-60000 model's thirty-dataset performance.

#### Why the two historical ladders differ

The published SOTArena snapshot puts historical `jope-dime` at rank 1/Elo 2259.8 with 11 wins and 45/45 coverage. All 45 rows are test-selected: 42 best-test evolution-envelope rows plus 3 test-adaptive accepted replays. The corrected benchmark-valid snapshot excludes those 45; **that rank 1 is not a held-out ASM claim**. The corrected snapshot has 304 methods/8,718 rankable rows. Its fixed-seed historical `jope-dime-regressor` entry is rank 31/Elo 1687.0, 1 win, 42/45 coverage; TabFM Ensemble is rank 1/Elo 1934.9. It remains historical public-development evidence.

The published view includes 45 test-selected `jope-dime` rows; the corrected benchmark-valid view removes them. The historical `jope-dime` rank-one Elo of 2259.8 belongs to the former view. It is a best-known test-selected result, not a held-out claim. An eligibility change alters the overlap graph and the fitted ladder, so these ratings must not be combined into a single current score.

![Historical published and benchmark-valid ladders with their distinct eligibility](../figures/fig02-ladders-valid-vs-adaptive.png)

### Historical same-cut screening and deepening

#### Same-cut v1 and v2 snapshots with screen and deep Elo

Same-cut v1 covers native heads plus resampled search; v2 adds XGBoost 2.1.4, LightGBM 4.6.0, CatBoost 1.2.10 and scikit-learn 1.7.2 random forest under Python 3.12.3. Both use the favorable historical test-10 regression selection, seed 20260907, five folds, top-five deepening and 32×8 search. Deep rows include only screened finalists; coverage must be printed with rankings. Their timing fields are then-recorded wall times under uncontrolled host load, not the present isolated-worker timing protocol.

Same-cut v2 compares all fifteen methods during screening on ten regression datasets. Only the five finalists per dataset are deepened across all five cuts, so the deep-stage denominator differs by method. Hollow markers preserve the earlier v1 snapshot; they are a separate run, not confidence intervals. The historical searched program is distinct from the current qualified neural proposal path.

![Historical same-cut screen and deep Elo with coverage](../figures/fig03-samecut-deep-elo.png)

The ten historical datasets were a favorable selection for the earlier engine. Their ranking cannot establish performance on a broader corpus or substitute for the frozen panel (asm-private-30-v1). External engine versions printed in these plots are the historical run versions, not a claim about currently installed adapter versions.

#### Historical relative loss and per-dataset rank

Each heatmap cell is a method's screen RMSE divided by the best screen RMSE on the same dataset. A ratio of one is best; the color saturates at three while displayed values are rounded. This normalization supports within-dataset visual comparison without averaging incompatible raw target units. Outlines show which cells reached the deep stage.

![Historical same-cut relative RMSE heatmap](../figures/fig04-samecut-rmse-heatmap.png)

The rank-distribution plot shows all ten per-dataset screening ranks for each method, with mean and median. It reveals variability hidden by a single aggregate ladder. A rank difference conveys ordering, not the magnitude of a loss improvement or a confidence interval.

![Historical same-cut per-dataset rank distribution](../figures/fig05-samecut-rank-distribution.png)

### Historical production time and the supplied reference scenario

The retained same-cut reports record mean program-production wall time 2.1837062093 seconds, median 0.9870389355 seconds and maximum 8.299870126 seconds across ten datasets. These are recorded timings under the host's contemporary load, not isolated controlled latency measurements and not the current paired ASM proof's elapsed time.

![Historical program-production times and an explicitly unmeasured scenario](../figures/fig06-seconds-per-program.png)

The one-hour/ten-million-token marker is the user's contextual scenario. Workload, hardware, token trace, currency and result quality were not measured under matching conditions. The chart therefore establishes no matched speedup, cost saving, token-efficiency ratio or quality equivalence. The measured current proof and its per-predictor/per-recipe accounting remain separately reported.
