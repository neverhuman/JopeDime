# H214 — pool diversity at fixed width16

Preregistration and synthetic protocol tests only; no attempt launched and no
H214 utility readout. See [PREREG.md](PREREG.md) and [manifest.json](manifest.json).

The comparison is NATIVE16 / FPS16-of64 / TEMP16, frozen r882, three fresh seeds,
DL-Val91/19 families and exactly16 scored programs including the parent.
The primary decision uses the median over seed-level equal-family gains,
10,000 shared family bootstrap draws and Holm m=2; strict LCB95>0.001546.

Launch is blocked until H212's real end-to-end smoke is merged and passes at
the exact merged H214 head. The current pinned proposer has no temperature
argument, and H212 N16 is4 anchors+16 proposals; the manifest records the
required pre-outcome temperature and width resolutions. The code refuses an
unsupported TEMP16 call and stale/incomplete smoke receipt. No approximate
temperature substitute or silently altered control is permitted.

The source currently supplies data-free selection/statistics and admission
helpers, not a complete executable study runner. Reuse the fixed H212 runtime
after its prerequisite merges, binding the exact source and frozen inputs
before the single xbabe1 CPU attempt. No training, GPU, live-loop or protected
data access is authorized. Final utility, CI and Elo remain unmeasured.
