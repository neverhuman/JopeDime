# External research review guide

This folder publishes the measured research history so another agent can challenge the methods and propose a useful next test. Start with the current summary linked from the study README, then inspect the source-bound JSON and every arm rather than the best seed.

## Questions the evidence currently answers

- **Does extra Dope neural state establish better candidate routing?** No. H18 and H20 preserve trained/random/zero controls; their primary intervals include harm. These tests are conditional state ablations, not all-Dope-versus-no-Dope comparisons.
- **Do rapid diverse GP probes establish additional lift for the neural ranker?** No. H27 completed the full 34,000-fit feature bank, fresh permitted Dope encodings, 36 matched neural models and independent endpoint replay. Both predeclared contrasts include harm. One-fold has the best point estimate, but that comparison was not the primary and H28 shows rare failed choices dominate its apparent benefit.
- **Should we train longer?** Existing long training plateaued under its recipe. In H27, TRAIN loss continues down while 31 of 36 internal-monitor losses worsen between updates 100 and 400. This motivates a prospective breadth-by-duration test, not retrospective selection of update 100.
- **Do we have millions of independent examples?** No. Candidate outcomes are correlated within contexts and source components. H27 has 850 historically exposed TRAIN contexts, 162 lineage components and 979,200 archived context-program outcomes. Metadata frontiers and stored object counts are not additional admitted tasks or independent validation.

## How to interpret the files

H-series percentage CV-ratio improvements, raw constrained GP fitness, context high-water-mark R²/AUC, native training loss and shadow top-1 gain are different endpoints. Their numerical values are not a common leaderboard. Raw constrained fitness is minimized; negative treatment-minus-control favors treatment. Each result retains its original population, direction, uncertainty and exposure limits.

Actual TRAIN/internal-monitor learning curves are separate from final out-of-fold endpoints. H27 did not evaluate intermediate outer-OOF checkpoints. A successful replay establishes implementation consistency within its declared scope; it does not create new independent families, prove every archived GP execution or remove prior development exposure. The historical wave capsule remains HISTORICAL_PARTIAL and is not an admitted governed wave. Optimizer updates do not prove committed-consumption exposure.

The public archive contains aggregate reports, source hashes, plans, cost/failure ledgers and per-model learning observations. It excludes credentials, private object locators, raw task cells, individual candidate outcomes, context/family identities and bulk checkpoints. Aggregate plots and JSON can be inspected from this checkout. Full bootstrap intervals or original training cannot be independently reconstructed from aggregate-only JSON; private source-bound evidence is needed. Missing artifacts or telemetry are explicitly unavailable rather than synthesized.

## High-value challenges

1. Identify any mismatch between decision-time available inputs and fitted feature generation, especially task preprocessing and probe CV folds.
2. Check whether target, slate, fold, evaluator, failure and gate contracts really match across comparisons. Equal fold counts alone do not establish compatibility.
3. Challenge dependence, historical family exposure, multiplicity and heavy-tail uncertainty. Preserve failed selections and the original equal-component weights.
4. Compare a strong tabular selector and qualified task breadth before proposing another encoder or broad hyperparameter search. Review the earlier continuous-cost, reduction, bilinear, surrogate and pairwise experiments first.
5. Propose a falsifiable mechanism, matched controls, complete work counts, exact source/role dependencies, raw decision utility and failure metrics, and a prospective independent confirmation boundary.

Feedback should cite a repository path, study/arm, exact commit or source hash, and the claim being challenged. A concrete counterexample or executable aggregate check is more useful than a favorable curve or a renamed version of an already-null experiment. File a GitHub issue or comment on the research publication PR; no production or resource authority is delegated by these documents.
