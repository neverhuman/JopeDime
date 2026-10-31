# H122 — Lookahead-only retest

**Status: CPU-prepared only.** Owner `luna_v1_hwv03`; assignment `~/nudges/h122_lookahead_retest_t0730.md`. H122 retests Lookahead because H93's LA cell never reached scoring; H93's partial CONTROL result is not an input and cannot be pooled with H122.

## Question and input boundary

Does Lookahead (`k=5`, `alpha=0.5`) improve the official unrevealed-candidate argmin utility over a fresh r882 reference on a new family-held-out slice of DL-Train? H97 EMA's reported positive CONTROL-relative signal is motivation only; no H97 or H93 outcome arrays, adapters, selections, or fitted values enter H122.

Use only the pinned real-S3 H50 DL-Train packet (1,219 regression contexts in 205 families) and an offline copy of round-882. Read the selected train member's identity metadata, require `dl_split=DL-Train`, regression task, train/inner source split, real origin when present, valid unique context hashes, and no `has_meta_test` row. Never read DL-Val, holdout, sealed, meta-test, outer-test, or live-loop payloads. The H50 manifest, selected roster and fused source are hash-pinned in `runner.py`. DOPE stays frozen at `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`; train only the 29,211 adapter parameters.

Fresh whole-family DL-Train split seed: **122042** (chosen for this H-number; distinct from H50/H93's 4800/4900 and H99's 9900). Sort family IDs, permute with NumPy `default_rng(122042)`, split into five folds; fold 0 (41 families) is the one assessment fold and folds 1–4 (164 families) supply all training rows. This is conditional development evidence: r882 was previously trained over all H50 DL-Train families, including the assessment families. It is not an unseen-family or serving qualification.

## Fixed protocol

Exactly six scored cells: CONTROL and LA for each fresh paired seed **48, 49, 50**. No timing-only cell. Each cell starts from a fresh, byte-verified r882 COPY in a unique `scratch/<ARM>-seed<seed>-attempt<N>/` directory. The runner checks the launch disk gate before creating the attempt directory or copying the checkpoint; a failed preflight leaves no staged file. Do not reuse H93's seed 30–32 or any H93 result.

Both arms use the H82 CONTROL recipe: 20 rounds × 2 optimizer updates (40 total), batch 16, slate 192, AdamW learning rate `1e-4`, weight decay `1e-4`, final gradient clip 1, and SmoothL1 plus `0.1 * KL` at temperature `0.2`. LA applies `k=5`, `alpha=0.5` slow-weight synchronization after fast steps 5, 10, …, 40; CONTROL has no slow weights. No extra steps, early stopping, best-of selection, arm tuning, or outcome-dependent fold/seed changes.

The cumulative GPU cell-wall cap is **2,700 seconds** for all six cells, including setup and voids; cumulative FIFO/resource wait is at most three hours. A missing, duplicate, void, or scored-incomplete cell blocks analysis and admission. No pre-registered retry may overwrite a physical attempt directory.

## Readouts and admission

Use H61's official unrevealed argmin evaluation (`EVAL_MODE_UNREV`, indices 4..P-1) and its clipped context utility. For each family, average its context utilities, then give each of the 41 assessment families equal weight. The primary is LA minus fresh r882, paired within seed and family. Report each seed's mean family gain and their median. Bootstrap 10,000 common family-cluster resamples (RNG 122000), retaining all three paired seed values per sampled family; report two-sided 95% percentile interval and one-sided LCB95. Admit LA only if the median over seeds is **strictly > 0**, LCB95 is **strictly > +0.001**, and the harm fraction is **<= 10%**. Harm means the across-seed mean paired family gain is below `-0.01`; this matches the adjacent H105 family-harm convention. Exact threshold ties fail.

Secondary: LA versus same-seed CONTROL, with paired family gains and descriptive Bradley–Terry Elo. Report per-seed Elo and the median with 95% family-bootstrap intervals versus r882 (rated 1500) and versus CONTROL. Use RNG 122001 for Elo resamples. Report Elo, never AUC. Secondary comparisons do not change the primary admission rule. No 95% CI, Elo, or admission statistic is emitted unless the exact six-cell panel and common identity checks pass.

## FIFO and resource gate

Future execution, if separately authorized, queues H122 **after H121**, preserving the older order H99 → H100 → H105 → H109 → H121 → H122. This CPU-preparation PR creates no FIFO ticket, takes no qualification lock, launches no timing/outcome cell, and does not authorize execution. Before any later cell: host CPU <=85/102, two checks 30 seconds apart; `nice 19`; OMP/MKL <=4; OpenBLAS/NumExpr 1; MemAvailable >=16 GiB; disk launch usage <94.5% and >=1 GiB free; child RSS <6 GiB. Never touch xbabe0 or the live xbabe1 loop.
