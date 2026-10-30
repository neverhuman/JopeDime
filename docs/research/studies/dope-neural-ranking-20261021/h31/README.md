# H31: probe-fold dispersion — verified development result

Six fixed-120-tree models and all six final OOF evaluations are closed. Across 850 historically exposed tasks and 162 current components, the producer reports favorable point estimates for actual dispersion versus mean-only and scrambled dispersion. Both adjusted intervals include harm, so there is no established benefit. Independent replay passed: all six models, 2,937,600 FP64 score bits and 2,550 picks reproduced exactly, with zero new fits or API calls. The same pinned prediction engine remains a limitation.

| Contrast, lower raw fitness better | Point | Central98.75% interval |
|---|---:|---:|
| D−M |−0.2320793901|[−0.7344936137,+0.0293917177]|
| D−S |−0.1093396674|[−0.4903898880,+0.0348107377]|

Raw failed picks are D 1, M 3 and S 2 of 850 each. These descriptive counts do not substitute for the preregistered raw-fitness contrasts. The two intervals use the same 100,000 paired resamples and familywise 97.5% coverage.

All 36 [TRAIN observations](train-curves.json) are retained across six separate panels. These are postfit prefix diagnostics, not validation, monitor or intermediate OOF curves. The fixed 120 endpoint was never selected from these curves. No new GP, Dope, NN or probe fits were needed; six new selector fits and two separately counted synthetic selector fits were completed.

The hypothesis was frozen before the source reviewer inspected H30 endpoints; the current results are not used to choose a next recipe. Historical H30/H27/H18/H20 null and adverse findings remain unchanged. This is exposed development evidence, with zero new numerical or independent data admissions and no serving or weight promotion.

See [results](results.json), [fixed protocol](protocol.json), [costs/failures](costs-and-failures.json), [critical report](critical-report.md) and [review guide](review-guide.md).
