# H285: fresh ODD CPU transfer and adapter averaging

**NOT_TRANSFERRED_FRESH**. Preregistered source `2affd94cfeb3824f4a64009a52ea9b8ea5b69f24`. One CPU attempt; zero neural updates.

The primary arm applies the three unchanged H278 RERANK8_VETO policies to fresh ODD-family r3175 CPU pools. The secondary arm averages four distinct adapters (r2750/r2795/r3100/r3175); r4330 duplicates r3175. B is exploratory and single-run, and cannot ADVANCE.

[Preregistered hypothesis, setup, rules and exact inputs](PREREG.md). [Machine-readable metrics](metrics.json). [Versioned full-output custody](full-output-custody.json).

Scored 33 contexts from 33 fresh families, one per family. The outcome-blind smoke rule fixed a prefix of 60; 27 structural exclusions for pools with fewer than eight candidates. Shared cohort SHA `12d2b1f37bf0e3c48ee61a24949b4f504251fe56b9b26b3081398623982f323c`, ODD half. No replacement cohort or refit.

Primary median family gain **+0.193003887**, paired-family bootstrap CI95 **[+0.014182603, +0.450019714]** (seed 285150, 10000 draws). Baseline Elo 1500; median A Elo **1577.325**.

| Frozen seed | Gain | Family CI95 | W/T/L | Harmed >.01 | Invalid / base | Changed | A Elo |
|---|---:|---|---|---:|---|---:|---:|
| 278101 | +0.214891374 | [+0.026874104, +0.467046734] | 13/15/5 | 4 | 0 / 2 | 18 | 1577.325 |
| 278102 | +0.193003887 | [+0.003449852, +0.449746160] | 10/19/4 | 3 | 0 / 2 | 14 | 1561.601 |
| 278103 | +0.087756531 | [-0.000125901, +0.245365483] | 13/15/5 | 4 | 1 / 2 | 19 | 1597.153 |

All primary gates: `{"all_three_lcb95_positive": false, "harmed_at_most_35_percent": true, "invalid_not_above_base": true, "median_gt_0_010": true, "random_null_fails": false}`. The random-valid null median gain is +0.323563119, CI95 [+0.039135931, +0.709127739]; its full conjunction passes: True. This null uses validity labels and is descriptive.

**B fresh: NOT (single run).** SWA gain +0.000000000, family CI95 [+0.000000000, +0.000000000], W/T/L {'wins': 0, 'ties': 33, 'losses': 0}, harmed 0, invalid 2 vs base 2, changed picks 1. Joint-policy median SWA Elo 1500.000; the repeated B array is not three independent runs.

**B reused DL-Val91:** gain +0.001613009, paired-family CI95 [-0.0016967073465243784, 0.006535733314364622], W/T/L {'wins': 1, 'ties': 17, 'losses': 1}, harmed 1, invalid 0 vs base 0, changed 3, two-policy Elo [1500.0, 1503.6535056965154]. These 91 contexts/19 families are reused, not fresh evidence.

Regression passed: r3175 CPU mean 1.1366829009431036 and 91/91 H278 cost vectors bit-exact. The rebuilt training table, pool and all eight H58 feature arrays were also bit-exact to the corpus reference. Saved outcomes, picks, bootstrap, null, prefix and DOPE were replay-audited without neural forwards. [Fresh cost/feature packets](fresh-pools.json), without picks or labels, are published for H284.

Next: retain r3175; use this failure of the preregistered transfer gates to design a separately authorized study. Do not refit or select on this cohort.

Resource and lifecycle notes: CPU 48,50–52, nice 19, numeric threads 1, aggregate threads 4 and 4 GiB RSS limits; pressure guards retained, no GPU/writer/observer intervention. Before science, two publication-helper probes exceeded the aggregate-thread guard (Go CLI runtime, then nested Git subprocesses); they were terminated and replaced by serial, directly guarded metadata commands. They performed no neural forwards and do not constitute scientific retries. Full outputs were exact-version GET/member-hash verified before this compact result. All own rebuildable checkpoint copies are reclaimed only after verified custody and terminal dependency checks.

The shared list uses first 60 buildable, positively P3-admitted families; historical missing/no_family context metadata is resolved by the pinned S1 admission. The approved teacher train-only view and GP recorded fitness are preserved. The auxiliary feature regression allowed numeric 1 versus historical numeric 4 tolerance; all tested arrays actually matched bit-for-bit. Hosted CI was not run under the owner research split; publication is non-force v1/next only, with exact-head local checks and same-agent review. No deployment qualification or live flywheel improvement is claimed.
