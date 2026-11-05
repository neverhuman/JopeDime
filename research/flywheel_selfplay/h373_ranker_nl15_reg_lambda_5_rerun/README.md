# H373 — NL15 ranker reg_lambda 5 rerun

PREREGISTERED; treatment not started. Exactly frozen H372 science (prereg c54a101f7d7b886abd8826e22efc158a842938b0, bound cced0c9522af3871539f00b9f1bec85af245da6d). H372 remains permanently INCOMPLETE_CPU_ADMISSION/NOT_EVALUATED with zero fits; this is a separate owner allocation. Only identifiers/paths, bootstrap RNG373191, window and union-pool admission change.

Hypothesis: stronger L2 on NL15 damps leaf weights and improves paired drop10. L2R5 changes only existing ranker reg_lambda1→5; H352 leaf15/depth3/minchild20/all other options/seeds/labels/groups/veto remain fixed. Primary frozen NL15; secondary BASE. See [PREREG.md](PREREG.md) and tests.

36 TRAIN-only tasks/9cells per arm. Advance requires paired drop10≥.002 and positiveLCB95,90%NL15gain/nonincreasingharm/two positive roots; whole-family10000/RNG373191. MultiplicityH348–H374. Any advance needs separate dev91 authorization; no dev91 here. Elo reference = r882 slot0 (DL-Train outer); r3175 not measured on TRAIN.

Preregistration commit: bound after non-force publication and before fitting. ONE5s gate over A0–14+sibs64–78 UNION A2_38–45+sibs102–109, first<=8/min4 passing bothSMT>=85idle/hostCPU<=70. Fixed17:30compute/17:45close Nov5; correctedCPU60/load108three60/Mem20GiB/scratch2GB own-only guard; no reprobe/restart/cuts. DOPEc51a56d0 unloaded, CPU selection layer only; no live adoption.
