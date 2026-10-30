# Posthoc fixed gate-penalty arithmetic

This supplement describes the already fixed selected choices. It adds no fits,
reranking, confidence intervals, primary endpoints or recipe selection.

| Contrast | Existing raw-fitness difference | Explicit 100 × weighted gate-failure difference | Remainder |
|---|---:|---:|---:|
| D−M | −0.2320793901 | −0.2469135802 | +0.0148341901 |
| D−S | −0.1093396674 | −0.1234567901 | +0.0141171227 |

The penalty contribution uses equal-component/context-weighted failure fractions:
D 0.0020576131687242796, M 0.004526748971193416 and S 0.0032921810699588477.
The raw counts are D 1, M 3 and S 2 of 850. Raw count divided by 850 is a different
weighting and must not be substituted in this decomposition.

The fixed penalty component is more favorable than the full raw-fitness difference.
After subtracting that declared term, the remaining fixed-formula difference is
small and positive (adverse). This is arithmetic, not a causal explanation and not
conditional passing-model RMSE. It does not independently reconstruct the complete
original S1 fitness formula or assign effects to the other terms.

No uncertainty interval is computed for this posthoc remainder. Both original
central 98.75% primary intervals remain unchanged, include harm, and establish no
dispersion lift. The existing exposed-development, inherited-source and same-engine
limits remain. See [the exact arithmetic receipt](posthoc-gate-penalty-arithmetic.json).
