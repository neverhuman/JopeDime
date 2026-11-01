# H133 — optimizer geometry / step size vs harm

**Status: A6 amended; execution not yet started.** H133 compares CONTROL, LOWLR_LONG, and SGDM against unchanged r882 on the family-held-out fold 0. The assessment fold and all admission/statistical gates remain frozen. A6 changes the seed roster to 133, 134, 138 and uses a monitored PAUSE-AND-WAIT policy for load and available memory; every other resource guard remains a hard abort.

See [PREREG.md](PREREG.md), [AMENDMENT_A6.md](AMENDMENT_A6.md), and [manifest.json](manifest.json) for immutable protocol details and source pins. The launcher records host, resources, pause intervals, and its four-hour elapsed-time cap. The first post-merge action is the exact r882 regression gate, which must reproduce family utility 1.1377942 before any treatment cell runs.

A6 discloses all attempts 001–012, including the qualification.lock deviation and the 40 CONTROL / 100 LOWLR_LONG unscored fit updates. The next authorized launch is attempt 013 on CPU at nice 19, preferably xbabe1 if the full staged input closure stays below 2 GB. No assessment tuning, DL-Val/outer-test/sealed/holdout reads, GPU use, resume, or implicit retry is permitted.
