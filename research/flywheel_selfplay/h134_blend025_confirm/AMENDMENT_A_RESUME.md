# H134 Amendment A-resume: one resume after the host-disk abort

This amendment implements owner decision (a) from hwave overseer `rd_opus_t1715`. It is committed and merged before
any resumed fit, and before any H134 outcome exists.
[PREREG.md](PREREG.md) and [runner.py](runner.py) are unchanged. The runner still checks the PREREG bytes against
`7e4537dbf`.

**What happened.**
- `run-20261031T2036Z` ran from source `11fba6d03` with preregistration `7e4537dbf`, merged as #1177 `9734924f3`.
- At 22:45Z the preregistered disk guard stopped it, with `/` at 93.97% (limit 93.8%), before fit 40/45
  (`seed143-exclude12`). Other activity on the host had filled the disk.
- 39 fits were complete. **No arm outcome, readout or verdict had been computed, before or since.**
- The partial evidence is pinned by manifest SHA-256 `f0205dba…`. Copies are in S3 at
  `H134/opus-h134-20261031a-partial/` and in `jope-rescue`.

**What is fixed (immutable).**
- Fold seed 134141, sampling seed 134142, model seeds 141–143.
- λ = 0.25, with BLEND_025 against r882 as the sole primary.
- H129 runner sha `e34d4a3f…` and H134 runner sha `5f38be27…`.
- All five admission gates, the tail checks, the Elo estimator, and the RNGs 134000 and 134001.

**What the resume does.** [resume.py](resume.py) is a launcher that imports the byte-pinned runner and repeats its
`run()` steps, with two differences:
1. Every file listed in the partial manifest must match its SHA-256 before anything runs. That covers the 39 fit
   predictions and receipts, `split.json`, the r882 reference costs and the DOPE receipt. The rebuilt split must also
   be byte-identical to the pinned one, and each reused prediction must match its receipt.
2. Only the **6 missing fits** are run: `seed143-exclude12, 13, 14, 23, 24, 34`. The readout is the runner's own
   `summary()`, and it runs only after all 45 fits exist.

The r882 reference costs and the frozen DOPE pass come from the aborted attempt's pinned files. DOPE `c51a56d0` is
recorded before and after in its receipt.

**Guards.**
- CPU only, nice 19, 4 threads, no `qualification.lock`.
- Worker RSS ≤ 6 GiB and MemAvailable ≥ 30 GiB.
- **The disk stays a hard abort at 93.8%.** Launch is withheld if the disk is above 93.5%.
- Host load stays a pause-and-wait (the H124 `headroom` loop) and never aborts.
- The launcher refuses a second resume, and refuses any root whose files differ from the manifest.

**If any guard aborts the resume, H134 is closed INCOMPLETE with no further retries.**
