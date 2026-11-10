# H420 PREREG addendum 1 — reuse H417's stored evaluations; defer FIT-SEL

**Status.** Published on `origin/v1/next` before any H420 deploy on a scored panel has started and before any H420
AUDIT value is read. At publication the only H420 work on a scored panel is the GX search run on `DEV` (in progress)
and a GX run on `FIT-SEL` that was stopped with no finished context.

**Reason.** Owner instruction, 2026-11-10 10:35Z: no unnecessary calculation; use what is already computed.

## Changes to the PREREG
1. **Variant `B` is not recomputed.** The baseline is H417's stored deploy of the same contexts (`APEX_BASE`, tag
   `BASE`, xbabe0). §2's "same-host replica" is dropped.
2. **The run memo is seeded.** `run_h420.sh seed` builds each context's memo as H417's post-deploy memo (every
   search, fold and AUDIT evaluation H417 made for that context) plus the GX memo. The variants then evaluate only
   programs that are not in it. Worker evaluations are deterministic given the program and the table, so a reused
   result equals a recomputed one; the exception is the 60 s limit, where a reused timeout stays a timeout.
3. **Order and scope.** `DEV` only: `TBOTH` (primary), then `TLIB`, then `TGX`. The tests, thresholds and verdicts
   of §3 are unchanged, with H417's `SXB` per context as the paired baseline.
4. **`FIT-SEL` is deferred.** Its GX run and deploys are run only if a `DEV` test passes, as training data for the
   stacker wave. The "reported, not tested" `FIT-SEL` comparisons of §3 are dropped unless that happens.
5. **Identity evidence instead of a replica.** On the two smoke contexts, variant `B` is run from a seeded memo and
   must (a) need 0 worker runs and (b) equal H417's smoke deploy on xbabe0 for `SXB` and `KFB` within 1e-9. Only the
   differences are read, not the values. If (b) fails, variant `B` is run from the seeded memo on all `DEV` contexts
   and becomes the baseline, as the PREREG first said.
6. **Readout.** `readout_h420.py` takes the baseline from the pulled H417 run when the run directory has no `B`.

## Compute saved
The dropped `B` replica on two panels and the `FIT-SEL` treatment variants were about 60% of the planned H420
worker time; the seeded memo also removes the re-evaluation of every program the variants share with H417.
