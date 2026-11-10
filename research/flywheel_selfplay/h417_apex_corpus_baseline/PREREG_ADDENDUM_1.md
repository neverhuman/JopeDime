# H417 PREREG addendum 1 — operational changes (no change to any rule or readout)

Published 2026-11-10 while the deploy runs; no H417 AUDIT value has been read.

1. **Gate G0 ran on xbabe3**, not xbabe0 (owner order 08:45Z to use all hosts). Same staged code (`pub-p1`), same
   CPU model and worker binary; xbabe3's sandbox pauses while a GitHub runner container is busy (12 pauses during
   G0), and an evaluation in flight during a pause can time out. The gate compares three arms on the same host, so
   the rule of §6.2 is unaffected; timeouts per arm are reported.
2. **The deploy was restarted as eight context-disjoint processes** (`deploy_shard.py`, stage `pub-p1s` = `pub-p1`
   plus that file) at 10:58Z. A single `train_apex.py deploy` process is bound by its Python thread: it had finished
   25 of 221 contexts in 64 minutes with two thirds of its cores idle. The shards use the same unit function, design,
   run directory, tag and run config (checked for equality at start); the 25 finished contexts are store hits; each
   context's files have one writing process. The stored evaluation limit of this deploy is 300 s (the H416 deploy
   default), not the 60 s of the search runs.
3. **The search runs took about 2.9 h (GS, GE) and 3.9 h (GL)** on 32 cores each, for the same reason.
