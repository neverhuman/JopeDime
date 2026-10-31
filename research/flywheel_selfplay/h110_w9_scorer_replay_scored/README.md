# H110 — proposal-distribution scorer and W9 memory replay: scored run of the H106 preregistration

Status: **Primary A PASS, primary B FAIL.**

- **A:** a scorer trained on logged native reveals of proposer programs is the first learner in these waves to beat
  random selection among new programs under the common gate. The gain is +0.144 [+0.023, +0.367], median +0.0095
  [+0.004, +0.025], with 4.5% harm and **+269 Elo**.
- **B:** at matched rows, memory acquired by the program prior does not make a better common learner than uniformly
  acquired memory (+0.022, LCB −0.016). The W9 "better guidance acquires better data" claim is **not supported** for
  the non-neural learner class.

## Protocol

This run executes [H106's preregistration](../h106_w9_proposal_scorer_replay/PREREG.md), merged in #1004 at
11:18Z before the run, with the merged runner (`run_h106.py` sha256 `13730ebf…`, the same in the run and analyze
receipts).

- **Training memories.** The H94 and H103 reveal logs, recovered exactly by stream replay: UNIFORM 546, PRIOR 601,
  ADAPTIVE 439 and CONTEXT 437 rows. The four guided memories are cut to matched rows per fold.
- **Learner.** The same recipe for every memory: HistGBM on `[prog412 − champion, prog412]` → within-context
  percentile. One model per W1 fold, with each context's whole family held out.
- **Prospective programs.** H67 roster (50 contexts / 44 families) × fresh seeds 801–803, 16 new proposals each.
  **Every proposal was evaluated**: 1,631 distinct native fits, because the proposer repeats some programs across
  seeds of a context. Every arm is therefore scored on a fully labelled batch. Picks were fixed from scores before
  the fit store returned any outcome.
- **Statistics.** Bonferroni m = 2 under the common gate, a 10,000-draw family bootstrap, and family Elo with random4
  = 1500. The completeness gate passed: 0 gaps, 0 shortfalls, 0 execution errors.

## Outcome (44 families, 150 / 150 context-seeds)

| Contrast | Mean [95% CI] | Adj. LCB | Median [95% CI] | Losing > .01 | W/T/L | Gate |
|---|---|---:|---|---:|---|---|
| **A: S_ALL − random4** | **+0.1439** [+0.0233, +0.3674] | **+0.0233** | **+0.0095** [+0.0043, +0.0245] | 4.5% | 36/1/7 | **PASS** |
| **B: S_PRIOR − S_UNIFORM** | +0.0221 [−0.0163, +0.0812] | −0.0163 | 0.0000 [−0.0001, +0.0008] | 13.6% | 21/7/16 | **FAIL** |
| S_ADAPTIVE − S_UNIFORM | +0.0351 | −0.0159 | 0.0000 | 15.9% | 20/7/17 | — |
| S_CONTEXT − S_UNIFORM | +0.0176 | −0.0368 | 0.0000 | 18.2% | 20/7/17 | — |
| S_ALL − prog_top4 | +0.1465 [+0.0301, +0.3112] | +0.0393 | +0.0072 [+0.0034, +0.0198] | 2.3% | 35/4/5 | — |
| prog_top4 − random4 (H103 replication) | −0.0026 [−0.123, +0.127] | −0.1003 | −0.0016 | 36.4% | 20/0/24 | — |

The primary LCBs are one-sided at .025 (Bonferroni m = 2); the secondary LCBs are one-sided at .05.

| Arm | Family-mean best of four | Family Elo (random4 = 1500) |
|---|---:|---:|
| ORACLE4 (ceiling, not a contender) | 2.0284 | — |
| **S_ALL** | **2.0215** | **1769.1** |
| S_ADAPTIVE | 1.9774 | 1726.1 |
| S_PRIOR | 1.9644 | 1714.3 |
| S_CONTEXT | 1.9599 | 1746.0 |
| S_UNIFORM | 1.9423 | 1703.6 |
| random4 | 1.8776 | 1500.0 |
| prog_top4 | 1.8750 | 1505.4 |

## Verdict and reading

- **A: learning on the proposal distribution works where cached-pool priors failed.**
  - All five scorers beat random by about 200–270 Elo, even the one trained on only about 340 matched rows per fold.
  - The cached-pool program prior again ties random. This replicates H103 on a fresh stream.
  - S_ALL closes 95% of the family-mean gap to the four-pick oracle. Its top four contain the batch's best program
    in 81 of 150 context-seeds; random would be about 25%.
  - This is the first positive prospective learner result on new programs in the R&D waves.
- **B: guide-acquired memory is not better memory.**
  - PRIOR, ADAPTIVE and CONTEXT memories each beat UNIFORM in family mean (+0.018 to +0.035), but every median is 0
    and every LCB is below zero.
  - At this scale the learner's gain comes from the volume and kind of data (outcomes of proposer programs), not from
    which guide chose them. ALL, at about 2.7× the matched rows, is the best scorer.
  - This is consistent with H94/H103: the guides had no new-program signal to pass on.
- **For W9.** This is the non-neural half of H91's replay matrix: no causal guidance → data → learner effect is shown.
  The neural cells remain with H75, which is parked; the GPU queue is unchanged.

## Red-team checks

- **Is S_ALL just "stay near the champion"? No.** In a descriptive check that was not preregistered (`redteam/` in
  S3), NEAR4 picks the four proposals with the smallest prog412 L1 distance to the champion. It scores 1.8963, and
  S_ALL beats it by +0.125 (median +0.0021, family W/T/L 29/4/11). Nearness alone is worth only +0.019 over random.
- **Leakage.**
  - Each context is scored by the fold model trained without its whole family; the fold-k models exclude all of fold
    k's logged rows.
  - Training targets are within-context ranks of outcomes logged before H106 was preregistered.
  - The evaluation proposals come from fresh seeds 801–803. Picks are fixed before any outcome is read.
  - Re-fitting the scorers gives sha256 `c12c4c78…`, identical to the dry run, so the training is deterministic.
- **Heavy tail.** The mean gain (+0.144) is driven by contexts with large utility scales. The preregistered median
  (+0.0095, CI excluding 0) and the 36/1/7 family W/T/L show that the effect is broad.
- **Repeated proposals.** Within a context, seeds 801–803 share programs: 1,631 distinct fits against 2,400 slots.
  Seeds are therefore not independent draws. Inference is family-clustered, so the intervals do not treat them as
  independent.
- **Development status.**
  - These are fit-only roster contexts, which serve as development evidence. The memory came from the same roster's
    other folds.
  - A held-family, non-roster test and a live-loop ablation are needed before any adoption.
  - Adoption itself is an owner decision.
- **Data scope.**
  - DL-Train only.
  - Zero DL-Val, outer-test, sealed or holdout reads.
  - DOPE is not loaded, and no GPU is used.
- **Reporting.** Elo, not AUC, with ties scored 0.5.

## Next step

1. **Confirm A out of roster.** Run the S_ALL scorer, trained on all roster memory, against random4 and the live
   proposer order on fresh non-roster DL-Train families, as a new preregistered stratum.
2. **Live-loop ablation (owner-gated).** Rank the live proposer's candidates by the proposal scorer instead of the
   cached prior or raw retrieval. H83 found retrieval ranks below random, and H103/H110 found the same for the prior.
3. **W9.** Record B FAIL for the non-neural class. The neural replay cells wait on H75's GPU slot.

## Reproduce

```sh
cd research/flywheel_selfplay/h106_w9_proposal_scorer_replay
A="--out ~/jope-data/h110/<run_id> --run-id <run_id> --h94-run ~/jope-data/h94/opus-h94-20261031a \
   --h103-run ~/jope-data/h103/opus-h103-20261031a --prog-models <H94 models_prog.pkl, sha256 5938a354...>"
nice -n 10 python3 -I run_h106.py run $A --workers 12 && nice -n 10 python3 -I run_h106.py analyze $A
```

## Evidence

Run `opus-h110-20261031a` used xbabe2 (driver at nice 10, native fits at nice 15) with 12 workers: 9:51 wall,
163 MB peak, 1,631 native fits and zero GPU. [metrics.json](metrics.json) holds every number above, the red-team
check and the S3 object versions. The objects live under
`s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/W9/opus-h110-20261031a/`: manifest, metrics,
run receipt, memory, scorers, rows, fits, ledger, logs, the runner and `redteam/`. Each was written create-only with
a version-pinned readback hash.
