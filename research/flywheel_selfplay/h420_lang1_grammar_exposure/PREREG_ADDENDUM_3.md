# H420 PREREG addendum 3 — the FIT-SEL replication runs after all

**Status.** Published before the `FIT-SEL` search run resumes. No H420 deploy on a scored panel has started; no
AUDIT value has been read.

**Reason.** Owner instruction, 2026-11-10 12:55Z: maximum utilisation of the CPUs and GPUs of xbabe0–3. Addendum 1
(item 4) had deferred `FIT-SEL` until a `DEV` test passed, to avoid computation that might not be needed. xbabe3's
CPUs are idle, so the deferral is lifted.

- **GX search run on `FIT-SEL`:** resumed on xbabe3 (docker sandbox, cpus 32–63, 96–127, nice 19, paused under the
  CI rule) from the run directory that was stopped on xbabe1 at 10:36Z with no finished context (its memo is kept
  and copied). Same staged code and arguments; 16 processes.
- **`FIT-SEL` deploys:** after H417's deploy finishes, the same seeded chain as `DEV` (TBOTH, TLIB, TGX against
  H417's stored deploy) runs on xbabe3. As in the PREREG, `FIT-SEL` is **reported, not tested**; its matrices are
  training data for the stacker wave if a route passes on `DEV`.
- **Honest label:** this work is used only if a `DEV` test passes. If H420 is null it will have been unnecessary.

Tests, thresholds and verdicts are unchanged.
