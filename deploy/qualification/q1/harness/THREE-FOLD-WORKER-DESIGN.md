# Q1 three-fold worker (ENG T0710)

ENG's 2026-11-01 T0710 ruling establishes three-fold frozen F for Q1. The amended FIT packet already pins three disjoint validation folds, but the v2 worker rejects any fold count other than two. Merely updating the cut consumer would charge failed attempts.

Require exactly three full-FIT complementary train/validation splits at production Q1 admission: repeat zero, each validation row once, no duplicate/overlapping/out-of-range rows, at least two rows per side. Refuse malformed bindings before reading TRAIN or reserving an attempt. Both Q1 policies use the same three folds and must return all three scores.

Extend the existing private v2 numerical callback to execute all three folds. Frozen S1 fitting/scaling stays unchanged. Aggregate the three log RMSE-to-training-mean-baseline ratios with arithmetic mean, maximum tail, population standard deviation, all-three-wins gate and unchanged geometric-ratio threshold 0.90, complexity penalty and gate ranges. User objective loss/value use the equal-weight mean of all three folds. This implements the ENG ruling; the historical RES-05 two-fold reference is not relabelled as three-fold evidence.

Retain existing two-fold v2 calls outside Q1, including their exact aggregation arithmetic, for historical research/runtime compatibility. Q1 cannot use that compatibility path. Reject other counts. No wire fields, version selector, schemas, model/checkpoint/DOPE pins, authoring rows or Q1 budgets change.

Alternatives rejected: accepting three fold arrays while scoring only the first two; replacing frozen S1 with the generic clipped predictor; silently changing preregistered cuts; breaking non-Q1 two-fold consumers. Risks are third-fold omission and budget-consuming malformed plans. Reproduce the original native callback refusal; prove third-fold influence, exact Q1 partition/pre-reservation refusal, numerical execution and legacy two-fold parity. Run service Q1/learned tests, strict Clippy/fmt and five records gates on the frozen head. Q1 scientific qualification and signing remain separate.

Authority: `~/nudges/lanes_t0710.md`; spec §13–14 Q1, §15 review. Two substantive COMPLEX reviews (QLT05 and DSC08) precede merger-owned integration.
