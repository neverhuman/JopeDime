# H372 — NL15 ranker reg_lambda 5

PREREGISTERED; treatment not started. Hypothesis: stronger L2 may damp NL15 leaf weights and improve paired drop10. Only existing ranker reg_lambda1→5 changes; leaf15/depth3/minchild20, original LR/tree choices, all labels/groups/seeds/sampling and veto remain fixed. Primary frozen H352 NL15; secondary BASE. See [PREREG.md](PREREG.md), source pins and tests.

36 TRAIN-only tasks,9cells per arm. Strict advance bar: paired-vs-NL15 drop10>=.002/positiveLCB95,90%gain retention/nonincreasingharm/two positive roots. Whole-family bootstrap10000/RNG372191; sequential H348–H372 multiplicity. Any advance needs separate dev91 authorization; none read here. Elo reference = r882 slot0 (DL-Train outer); r3175 not measured on TRAIN.

Preregistration commit: bound after non-force publication and before fitting. Fixed16:45compute/17:00close Nov5; one poolA admission; corrected60-secondCPU/three60-secondload108 guard; no replay/reduced design. No live adoption. DOPE c51a56d0 unloaded, CPU selection layer only.
