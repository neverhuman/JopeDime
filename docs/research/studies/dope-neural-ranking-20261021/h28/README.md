# H28: rare failure transitions explain the apparent landmark differences

This posthoc diagnostic retains every one of the 7,650 fixed real-state choices (850 contexts, three seeds, three landmark arms) and the original equal-component population weights. It does not refit, rerank, remove failed choices, clip fitness, select a checkpoint or calculate a new confidence interval.

| Comparison (lower constrained fitness is better) | Original total difference | Both choices pass: additive contribution | Gate transitions: additive contribution |
| --- | ---: | ---: | ---: |
| Five-fold minus zero | -0.1055719651 | -0.0020359877 | -0.1035359774 |
| Five-fold minus one-fold | +0.1482177800 | +0.0044846237 | +0.1437331563 |
| One-fold minus zero (exploratory) | -0.2537897451 | -0.0065206114 | -0.2472691337 |

About 97–98% of each net point difference comes from rare transitions between a passing and failing candidate. One component contributes -0.2056735520 to the exploratory one-fold-minus-zero difference, about 81% of that net point estimate. Conditional successful-choice comparisons would change the population if used as a new primary; the table instead preserves their additive contribution to the original population.

The original two H27 intervals remain unchanged and include harm. This does not establish landmark lift, failure prediction, causality, independent generalization or a new significant result. It motivates prospective failure-aware selection and broader independent support rather than more tuning on these outcomes.

Producer extraction: 18 fixed prediction files, 17.865583735 wall seconds / 2.047502070 CPU seconds, zero fits. Separate independent saved-choice arithmetic reconstructed the context/seed/component hierarchy and all gate cells, concentrations, seed differences and original intervals in 1.598409549 wall seconds / 0.252136051 CPU seconds. Four saved files totaling 2,460,916 bytes were read; no additional arrays, model forwards or APIs were used. That arithmetic inherits the original extraction and model semantics.

See [aggregate results](results.json), [independent arithmetic receipt](independent-arithmetic-verification.json), and the [original H27 primary results](../h27/nn-fixed400-independent-results.json).
