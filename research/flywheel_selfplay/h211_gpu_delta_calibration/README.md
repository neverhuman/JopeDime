# H211 — INCOMPLETE after host-load guard

The single attempt stopped at 2026-11-02 08:15:53 UTC: load1 **106.101** exceeded
**86.7** (85% of 102 effective CPUs). This is **INCOMPLETE, no scientific verdict**.
The hypothesis was never evaluated on A; no retry or budget reset occurred.

- Executed source: `ddfca6624f0a1977116ec39edc7fa40c65b7dca1`.
- Worker time: 1,249.643 s; supervisor time: 1,255.022 s.
- Optimizer updates: **2,724 / 3,840**; 21 complete saved heads and B panels,
  plus 36 updates of the next head whose state was not persisted.
- Native91 utilities passed bit-exact at **1.1377941898164525**.
- No global fitted-head/selection lock; no A scoring. Gain, adoption, Elo and U4
  are unmeasured (null with reasons in [metrics.json](metrics.json)).
- Per-head loss/gradient/state-change proofs remained in RAM until the global
  lock and were lost on abort. Saved checkpoint bytes do not replace those proofs.
- At refusal: CPU32%, available RAM79,493,812,224B, disk93.183325%, RSS1,950,064,640B;
  the instantaneous load1 limit was the failing gate. GPU and lock are released.

[PREREG.md](PREREG.md) preserves the hypothesis/setup: frozen r882/DOPE, separate
GPU utility-delta head, raw vs B-gated selections and shuffled-label control on
historical real-S3 TRAIN pools. The execution code, seeds and folds are unchanged.
H213 is a separately authorized prospective study with fresh seeds and a sustained
load rule; H211 is not resumed. No scientific rejection or serving promotion is inferred.

Custody: **46 exact-VersionId S3 objects re-read and rehashed**, including 21 heads
and 21 B panels; **56 runtime archive members recovered and individually rehashed**.
No new A evaluation, model inference or B outcome analysis was performed for closeout.

`s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261101/H211/astra-t0040/closeout-t0125/manifest.json`

VersionId `YCYXKEun40dkPlWnxeC.8lG6nqWnx3hq`; SHA256
`0d36f6b637eb37caf528ffc758c8f60bf2eff159575b2a513946cb51b576afb6`.

[manifest.json](manifest.json) binds custody; historical source hashes refer to the
executed commit. Local small receipts remain in
`/home/ubuntu/jope-rescue-20260921/h211-astra/closeout-t0125/`.
