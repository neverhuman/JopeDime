# H363 — NL15 ranker min_child_samples 10

PREREGISTERED; treatment not started. Hypothesis: lowering H352 NL15's ranker leaf floor from20 to10 resolves smaller context groups. Everything else, including depth3, all learning-rate/tree choices, labels/groups/seeds and veto, stays fixed. Primary frozen reference is H352 NL15; BASE is secondary. See [PREREG.md](PREREG.md), source pins and tests.

TRAIN only:36 fold/root tasks,9 cells per arm. Advance requires paired drop10>=+.002 with positiveLCB95, gain/harm retention and two positive roots. Bootstrap10000/RNG363191; sequential H348–H363 multiplicity applies. Separate dev91 authorization would be required after any advance; H363 reads none. Elo reference = r882 slot0 (DL-Train outer); r3175 not measured on TRAIN.

Preregistration commit: to be bound after non-force publication and before fitting. Fixed compute15:15Z/close15:30Z Nov5; one poolA admission; corrected60-secondCPU/3×60-secondload108 guard; no replay or reduced design. No live adoption. DOPE c51a56d0 unloaded, CPU selection layer only.
