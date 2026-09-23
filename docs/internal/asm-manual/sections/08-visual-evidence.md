# Visual evidence: data, measured performance and historical rankings

## Current measured ASM outcome

This is the current bounded operational proof, with one preselected regression dataset and matched search-arm allowances. It demonstrates authenticated neural-origin fits and fitted descendants. The generic arm has lower development RMSE. The complete thirty-dataset ASM panel has not run, so this figure cannot establish its wins, ranking or Elo.

![Current operational proof: neural and generic development RMSE](../figures/current-origin-loss.png)

The two underlying records are in `data/plot-current-origin.csv`; exact evidence and denominators are in `data/performance-evidence.json`. There are 34 predictions per final refit. The proof's run wall time is 239.143 seconds; its independent recovered report takes another 18.719 seconds. Numerical measurements, worker counts and costs are detailed in the status and benchmark chapters.

## Frozen dataset coverage

The following chart covers all thirty immutable datasets in task order: regression, binary, multiclass. It preserves the original feature-type metadata and row assignments. Sealed counts come from membership metadata; sealed target bodies were not opened. The complete statistics appendix gives readable numeric tables, class/category frequencies and every feature record.

![All thirty datasets: frozen rows and feature types](../figures/inventory-frozen30.png)

Totals are 55,029 rows, 44,048 development rows, 10,981 sealed membership records and 522 summed feature records. Feature counts are not a union of semantic variables across datasets. The chart and the detailed tables share the same pinned inventory.

## Broader corpus and metadata concerns

The broader catalog has 3,859 real datasets and 6,208,051 metadata-reported rows. It is a different collection from the frozen panel. Its `multi_label` task is not interchangeable with admitted multiclass classification. Per-payload statistics for the entire corpus were not recomputed.

![Corpus task counts and license metadata completeness](../figures/inventory-corpus.png)

The catalog has 3,755 entries with unknown license metadata and 104 with a recorded license. The collection's older prose count of 1,416 unknown licenses is inconsistent with this current pinned catalog. This is a metadata quality gap; neither the chart nor a repository-level license resolves each dataset's provenance. See the estate chapter for exact storage bytes, prefixes and observation scope.

## Ranking figures: interpretation contract

The historical ranking plots that follow use their separately pinned snapshots. They are not new model runs and are not the current ASM30 campaign. Benchmark-valid and public-development views retain their original eligibility distinctions. Same-cut deep-stage coverage varies because only finalists were deepened; absent cells are not zero losses. Refer to the benchmark-methodology chapter for the exact Elo model, tie rule, coverage limits and adaptive-development interpretation.

All machine-readable rows are retained in `data/performance-evidence.json` and `data/plot-historical-elo.csv`. A displayed top-method subset does not remove the other methods from those inventories. Dataset selection and historical protocol caveats travel with each figure.

## Historical benchmark-valid Elo

This view retains 304 methods and the historical corrected eligibility filter. The historical `jope-dime-regressor` entrant is rank 31 with Elo 1687.0 in this snapshot. “Benchmark-valid” means no retained artifact contradicts a row's flag; comparator protocols were not independently verified. It does not certify that every comparator obeyed today's frozen ASM protocol.

![Historical benchmark-valid Elo ladder and complete rank distribution](../figures/fig01-elo-benchmark-valid.png)

The top-method view is paired with the distribution of the full method universe. Dataset coverage and the identity of the historical entrant remain visible. These numbers describe the retained historical ladder, not the current accepted step-60000 model's thirty-dataset performance.

## Why the two historical ladders differ

The published view includes development-selected rows; the corrected benchmark-valid view removes 45 test-selected rows. The historical `jope-dime` rank-one Elo of 2259.8 belongs to the former view. It is a best-known development result, not a held-out claim. An eligibility change alters the overlap graph and the fitted ladder, so these ratings must not be combined into a single current score.

![Historical published and benchmark-valid ladders with their distinct eligibility](../figures/fig02-ladders-valid-vs-adaptive.png)

## Historical same-cut screening and deepening

Same-cut v2 compares all fifteen methods during screening on ten regression datasets. Only the five finalists per dataset are deepened across all five cuts, so the deep-stage denominator differs by method. Hollow markers preserve the earlier v1 snapshot; they are a separate run, not confidence intervals. The historical searched program is distinct from the current qualified neural proposal path.

![Historical same-cut screen and deep Elo with coverage](../figures/fig03-samecut-deep-elo.png)

The ten historical datasets were a favorable selection for the earlier engine. Their ranking cannot establish performance on a broader corpus or substitute for the unchanged frozen thirty. External engine versions printed in these plots are the historical run versions, not a claim about currently installed adapter versions.

## Historical relative loss and per-dataset rank

Each heatmap cell is a method's screen RMSE divided by the best screen RMSE on the same dataset. A ratio of one is best; the color saturates at three while displayed values are rounded. This normalization supports within-dataset visual comparison without averaging incompatible raw target units. Outlines show which cells reached the deep stage.

![Historical same-cut relative RMSE heatmap](../figures/fig04-samecut-rmse-heatmap.png)

The rank-distribution plot shows all ten per-dataset screening ranks for each method, with mean and median. It reveals variability hidden by a single aggregate ladder. A rank difference conveys ordering, not the magnitude of a loss improvement or a confidence interval.

![Historical same-cut per-dataset rank distribution](../figures/fig05-samecut-rank-distribution.png)

## Historical production time and the supplied reference scenario

The retained same-cut reports record mean program-production wall time 2.1837062093 seconds, median 0.9870389355 seconds and maximum 8.299870126 seconds across ten datasets. These are recorded timings under the host's contemporary load, not isolated controlled latency measurements and not the current paired ASM proof's elapsed time.

![Historical program-production times and an explicitly unmeasured scenario](../figures/fig06-seconds-per-program.png)

The one-hour/ten-million-token marker is the user's contextual scenario. Workload, hardware, token trace, currency and result quality were not measured under matching conditions. The chart therefore establishes no matched speedup, cost saving, token-efficiency ratio or quality equivalence. The measured current proof and its per-predictor/per-recipe accounting remain separately reported.
