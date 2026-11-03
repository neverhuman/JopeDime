# H236: Schedule-Free AdamW — COMPLETE / NO_ADVANCE

Hypothesis: schedule tuning partly causes the 40-update r882 plateau; test SF-AdamW averaged-x scoring while retaining the exact H170 CTRL loss, batches and H235 strict selector.

Execution source `39fd7d58258a2aefd9aa2eadef201ae4b5069812`, prereg PR #2073. Primary SF and secondary SF_LR3 (Holm m=2), seeds 236101–236103 and fold seeds 236141–236143, offline r882 COPY, CPU xbabe2 cores 40–43, nice 19, one math thread. Only 47 adapter tensors/29,211 parameters; no serving promotion.

Native91 PASS: all 91 utility values exact, mean **1.1377941898164525**, cached/native forward bit-exact before any TRAIN pools. TRAIN smoke PASS: 3 disposable updates with averaged-x scoring and exact r882 restoration. Full study: 15/15 cells, 1,800 updates, 180 averaged-x snapshots independently hash-recounted. Frozen DOPE before/after `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`; no protected input refusal.

All 45 strict trajectory selections retained step 0 (r882). The trained step 40 adapter snapshots changed in every trajectory, but none passed the frozen selector. The resulting policy vectors duplicate the incumbent; degenerate [0, 0] intervals describe that fallback and do not establish independent equivalence of the trained optimizers.

| DL-Val contrast | Median equal-family gain | Family CI95 | Holm p | Advance |
|---|---:|---|---:|---|
| SF vs r882 | +0.0000000000 | [+0.0000000000, +0.0000000000] | 1.00000000 | False |
| SF_LR3 vs r882 | +0.0000000000 | [+0.0000000000, +0.0000000000] | 1.00000000 | False |

Primary SF verdict **NO_ADVANCE**; aggregate per-arm verdict **NO_ADVANCE**. ADVANCE requires gain > .002, family LCB95 > 0 and Holm p < .05; ties reject. Beating CONTROL alone is descriptive. Compact metrics retain CONTROL contrasts, Elo vs r882 and CONTROL, secondary TRAIN OOF 1219/205, selections/duplicates/drift and descriptive harm. The context-mean comparison to plateau best 1.1389572 (r3175) is descriptive. No AUC or speed verdict.

| Elo lift, median across 3 seeds [family CI95] | vs r882 | vs CONTROL |
|---|---|---|
| CONTROL | +0.000000 [+0.000000, +0.000000] | +0.000000 [+0.000000, +0.000000] |
| SF | +0.000000 [+0.000000, +0.000000] | +0.000000 [+0.000000, +0.000000] |
| SF_LR3 | +0.000000 [+0.000000, +0.000000] | +0.000000 [+0.000000, +0.000000] |

Elo intervals use 1,000 paired family draws of the three-seed median. Zero lift and intervals touching 0 do not establish improvement; the frozen primary utility margin/CI/Holm gate determines the recorded advance verdict.

| DL-Val context utility (descriptive) | Median across 3 seeds | Minus r3175 plateau 1.1389572 |
|---|---:|---:|
| r882 | 1.1377941898 | -0.0011630102 |
| CONTROL | 1.1377941898 | -0.0011630102 |
| SF | 1.1377941898 | -0.0011630102 |
| SF_LR3 | 1.1377941898 | -0.0011630102 |

This historical plateau comparison uses context means; the primary gate above uses paired equal-family gains against r882.

Resource guard terminal PASS: exit 0; peak RSS 3.14 GiB (3,373,522,944 bytes), peak threads 3, load pause events 0. Own process group only. No retry.

Red-team PASS: actual TRAIN/selector/assessment family sets disjoint and all exclude DL-Val; no has_meta_test/sealed/holdout input; higher utility is better; all 3 seeds reported; 47 adapter tensors / 29,211 parameters only; frozen DOPE c51a56d0. Resource observations: maximum raw host CPU 98.4% before the closeout instruction (all samples retained), maximum since the closeout instruction was read at 00:59:40Z: 48.0%; minimum MemAvailable 44,263,620,608 bytes and disk 750,147,272,704 bytes. No OOM/disk failure.

Next step: retain the incumbent; do not promote either optimizer treatment from this wave. No follow-on compute is authorized here.

Custody: full metrics, source capsule and raw receipts in `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261103/H236/res08-t0005/terminal-39fd7d58258a-25896e350e2d75a4/H236-execution-output.zip` VersionId `Ai.ohaMBoeHcqO9PZSXpB6.twVyCDnlJ`, SHA256 `25896e350e2d75a44d06ac0a89f7c5dbe2166a1f7bc2b496506e4130b998e9a6`. Exact-version bytes and all 270 members replay PASS. Manifest `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261103/H236/res08-t0005/terminal-39fd7d58258a-25896e350e2d75a4/H236-terminal-manifest.json` VersionId `tRHPa2QFW3Llb_pEDHx3.iLKvrFvD6SJ`. Both raw roots retained: `/home/ubuntu/jope_plan/evidence/H236/res08-t0005` and `/home/ubuntu/jope-data/h236`. Prereg first-prefix date error is retained in its original receipt; canonical prereg prefix is 20261103.

## Deviations

A1, recorded **2026-11-03T00:29:13.046507+00:00 before model construction, Native91, TRAIN smoke or any study outcome**: owner amendment `hw_t0030_res08_h236_a1_load.md`, SHA256 `948f02832adfc5f7231f8145dc29720cbe6df86cf556e5c063eb9d8bea0a98bf`. Reason: "load5 inflated by foreign nice'd rustc + iowait; CPU idle ~58%". Admission is load5 <= 100; run-time pause is load5 > 108 and resume <= 100. CPU cores 40–43, nice 19, one math thread, RSS < 4 GiB, MemAvailable >= 20 GiB, disk >= 300 GB, own-process-group signalling only, single attempt/no retries, compute STOP 06:00Z and closeout 06:30Z remain unchanged. The scientific source `39fd7d58258a2aefd9aa2eadef201ae4b5069812`, arms, seeds/folds, data, optimizer, selection and statistical gates remain unchanged. Original data-free admission controller was replaced before any primary attempt claim; this is not a scientific retry.

Publication validation: exact-head same-agent review, JSON/schema/link/hash and diff checks; 15 data-free pytest cases and Ruff repeated at the unchanged source bytes of the closeout head. Static coverage validation passed at preregistration. Hosted CI NOT_RUN/NOT_A_GATE, admin squash explicitly authorized by `hw_t0005_res08_h236_sfadamw.md`.

Original frozen protocol: [PREREG.md](PREREG.md); no analysis/tolerance change.
