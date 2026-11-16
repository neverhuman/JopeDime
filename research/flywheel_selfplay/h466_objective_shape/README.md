# H466: objective shape on the gated pairnf step (offline, adapter-only)

**Status: `RUNNING` (pre-registration record).** This README records the design, code hash and launch before any CV result exists. The verdict, metrics and red-team will be added when all 3 seeds write `summary.json`.

## Hypothesis
H457 (stabilizers), H461 (dose, family balance) and H463 (recalibrated gate, cosine dose) were all `NO_LIFT`. They moved how many changes the z >= 0.40 gate keeps, never the ~57% win rate of the kept changes. H466 keeps the dose (16 steps, lr 3e-4) and the gate (z >= 0.40) fixed and changes the **shape of the training objective**. Every arm shares all random streams with the control (`rng_arm = pairnf`: same init perturbation, dropout, context draws and slates), so only the loss differs:
- `pairnf_z40`: the control, H452-H463 pairnf + z >= 0.40, 16 steps. Adapter shas must equal the H463 `pairnf_z40` cells.
- `ctxn_z40`: per-context loss normalization. Each context's loss row is rescaled to the batch-mean magnitude, so every context carries equal gradient weight and high-loss tail contexts (gate-fail -2.0 picks, long fitness tails) cannot dominate.
- `clip05_z40`: robust pointwise term. The smooth-L1 target is clipped at (context best + 0.5), so gate-fail 2.0 targets and long tails cannot dominate the pointwise loss. Pairwise, listwise and hinge terms are unchanged.
- `pw2_z40`: pairwise Bradley-Terry weight 2.0 (vs 1.0), more weight on the non-floor pairwise term that carries the H451-H463 non-floor lead.

Not tested before on this instrument: no wave since H444 changed the loss shape of the gated pairnf adapter step. (H268/H271 used context-normalized *labels* on the older H265 feature reranker, a different learner.) Already falsified and not repeated: Polyak/EMA tail average and ASAM (H457), raw s32 and fbal (H461), recalibrated gate and cosine dose (H463).

## Pre-registered rules (unchanged from H452-H463)
- **Selection** uses Elo only: per seed, the max mean non-floor OOF Elo over the two partitions (ties broken by pooled gain).
- **Strict:** pooled OOF gain > 0.002 and gain CI low > 0; pooled Elo > 0 and Elo CI low > 0; non-floor Elo CI low > 0; Elo > 0 on each partition.
- An arm counts only if it also beats the control in the paired family bootstrap (arm - control CI low > 0). Ties never win.
- **LIFT** requires a non-control arm that beats the control on the median across 3 seeds with the paired CI above 0, plus a clean red-team and val91 not_worse on every seed. Otherwise the verdict is `NO_LIFT`.

## Setup
- **Code:** `h466_trainfold_cv.py`, sha256 `c90d359001ab88eb32bbad5b886a84d2aceae1225b83cea517dd5f718fa290d8` (checked with `sha256sum -c` in the scratch dir before launch). It differs from `h463_trainfold_cv.py` only in the arm table and docstring. CPU only, so no GPU and no `qualification.lock`. It runs at `nice 19` with 16 threads per seed and aborts itself at MemAvailable < 15 GB.
- **Scratch:** `~/jope-data/hwave-h466` on xbabe2, a copy of `round-3175.pt`, never the live run directory. The run never touches xbabe0.
- **Data:** the real S3 packet: 1,219 DL-Train contexts (205 families) and 91 DL-Val contexts (19 families). val91 is read once per seed, for the CV-selected arm only, as a regression check. No sealed or holdout data is read.
- **DOPE frozen:** composite sha `c51a56d0...` is checked before and after. Only the 29,211 adapter parameters train.
- **Partitions:** the two H452 floor-stratified, family-grouped 5-fold partitions (`h452a/`, `h452b/`), 4 trained bases x 2 partitions x 5 folds per seed, plus 1 full-train run for the selected arm.
- **Probe:** PROBE_OK at 2026-11-16 03:06 UTC, DOPE after `c51a56d0...`, peak RSS 6.2 GB, min MemAvailable 33.5 GB, incumbent val parity 1.1389572403748425.
- **Launch:** `launch.sh` is gated on MemAvailable >= 30 GB and machine CPU busy% + 16 threads <= 85%. Seeds 1/2/3 launched 2026-11-16 03:08 / 03:18 / 03:28 UTC (mem 60 / 49 / 45 GB, CPU busy 10 / 14 / 14%). `waiter.sh` relaunches a dead seed (any cause), resume-safe via `results/fold_sig`, at most 3 times per seed, and SIGTERMs only its own seed process if it hangs more than 10 min after `summary.json`. Expected close about 13:00-16:00 UTC Nov 16 (H463 took ~10 h per seed with 4 arms).

## Aggregation (when every seed has `results/seedN/summary.json`)
Copy the aggregator and red-team scripts from H463, then run `WAVE=H466 python3 h466_aggregate.py aggregate.json` and `ARMS=pairnf_z40,ctxn_z40,clip05_z40,pw2_z40 python3 h466_redteam.py redteam.json`.

Raw logs, losses and fold signatures stay on xbabe2 in `~/jope-data/hwave-h466`.
