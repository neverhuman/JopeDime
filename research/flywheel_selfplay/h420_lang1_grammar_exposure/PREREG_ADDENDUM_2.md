# H420 PREREG addendum 2 — deploy limit and process sharding (operational)

**Status.** Published before any H420 deploy on a scored panel has started and before any H420 AUDIT value is read.

1. **Deploy evaluation limit = 300 s.** `train_h420.py` first defaulted to 60 s, the search limit. The stored H416 and
   H417 deploys use 300 s for their fold fits, and the seeded memo holds results made under that limit. H420's deploys
   now use 300 s, so reused and new evaluations share one rule. The GX search run keeps the 60 s search limit, as GL
   had.
2. **Context shards.** A deploy process is bound by its Python thread (measured on H417: one core busy, 25 of 221
   contexts in the first hour). Each variant now runs as context-disjoint processes of one run directory
   (`train_h420.py deploy --shard k/n`), the engine's documented multi-process mode: every context's memo and stage
   files have one writing process. Results do not depend on the shard count. The GX search run on `DEV` was
   restarted the same way (16 processes instead of 4, same run directory and memo) before it had finished a context.
3. **Replay check** runs on 2 contexts (it is single-threaded).

Tests, thresholds, verdicts and addendum 1 are unchanged.
