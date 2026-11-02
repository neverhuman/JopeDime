# H211 — guarded GPU improvement-head rerun

**PREREGISTERED, NOT RUN.** H209 passed its native baseline but stopped before
training when SIGALRM reentered buffered resource logging. H211 preserves the
same scientific hypothesis and gates, adds the tested reentry guard, and uses
fresh seeds and a new one-hour, one-attempt budget.

[PREREG.md](PREREG.md) states the full protocol and explicit owner authorization
to launch after an exact-head PR is open. [manifest.json](manifest.json) pins
inputs, source and resource limits; [fold_manifest.json](fold_manifest.json)
freezes the splits. [provenance.json](provenance.json) retains real-S3 provenance.
No H209 budget is reset and no existing attempt is resumed.

Actual CUDA training: 30 heads / 3,840 updates on xbabe2 GPU0, disk<=93.8%,
one numeric thread, exclusive qualification lock. Native91 parity precedes
updates; all B choices lock before A. No result or serving qualification exists.
Acceptance: both H209/H211 CPU suites, Ruff, five records gates, source pins,
exact PR identity, then fresh GPU/resource/attempt admission. No self-merge.
