# H451: non-floor pairwise loss and confidence-gated override on the trust_big step (offline, adapter-only)

**Verdict: `NO_LIFT`.** 0 of 12 arm-seeds pass the strict rule, and every Elo CI spans 0. The CV-selected arm is `pairnf_k25` on both seeds: OOF gain +0.0098 / +0.0077, Elo +6.6 / +6.0 (Elo CI [−1.6, +14.4] / [−2.1, +13.9]). The val91 regression check is flat: gain +0.0030 / +0.0034 (both CIs span 0), Elo +7.6 / 0.0 (W/T/L 8/77/6 and 7/77/7), 0 new gate failures. H451 is still the first wave in H447–H451 with a small, consistent **non-floor** signal. `pairnf_k25` has positive non-floor gain on all 5 folds on both seeds (+0.001 to +0.005 per fold), non-floor Elo +5.5 / +5.2, and Elo +4.6 / +3.9 with fold 1 dropped. That signal is not significant, and it carries winner's-curse optimism from picking the best of 6 arms. Seed 3 was skipped because of memory (see Setup).

## Hypothesis
H450 showed that every trust_big-family gain is recovery of the incumbent's 19 TRAIN −2.0 gate failures. On the 1,200 non-floor contexts, the changed picks are coin flips, so Elo stays at about 0. H451 targets Elo directly:
- `pairnf`: trust_big step + gate hinge 0.5 + Bradley-Terry pairwise loss against the incumbent's own argmin. The pairwise term is restricted to non-floor contexts (incumbent pick passes) and passing programs, and each context is weighted equally.
- confidence-gated override (`_k50`, `_k25`): among the evaluated contexts whose pick changes, only the top 50% / 25% by z = (pred[inc] − pred[cand]) / std(pred) keep the change. The rest revert to the incumbent pick and become ties. The gate is label-free (predictions only), so it is legal on held-out folds and val91.
- Arms: `big_ctrl` (H450 big_ctrl reproduction), `big_k50`, `big_k25`, `pairnf`, `pairnf_k50`, `pairnf_k25`. Two trained bases (trust_big and pairnf); the gated arms are eval-only.

## Setup
- **Code:** `h451_trainfold_cv.py` (as run; CPU only, `nice 19`, 16 threads, abort if MemAvailable < 15 GB) on xbabe1 against a scratch copy, `~/jope-data/hwave-h451`. The live run directory was never written to. Seeds 1–2 launched 2026-11-12 23:45 MST and finished 2026-11-13 02:50 MST.
- **Seed 3: skipped for memory.** LAUNCH.md requires MemAvailable ≥ 38 GB for a third process. xbabe1 had 28.7 GB at launch and 28–30 GB after seeds 1–2 finished (other tenants hold the rest), so the wave is written up on 2 seeds. The seeds are near-replicates anyway: they differ on 16–118 of 1,219 OOF contexts.
- **Start point:** incumbent `round-3175.pt` plus its optimizer state, deep-copied per candidate. Parity reproduces the live val utility exactly (1.1389572403748425, asserted).
- **DOPE frozen:** composite sha `c51a56d0…` before and after on every seed. Only the 29,211 adapter parameters train.
- **Data:** real S3 packet, 1,219 DL-Train contexts (205 families) and 91 DL-Val contexts (19 families), 0 family overlap. Every row is DL-Train or DL-Val (asserted). No sealed or holdout data was touched.
- **Instrument:** same as H444–H450: 5 family-grouped TRAIN folds (244/244/244/244/243, `h444/` salt), pooled OOF paired gains vs the incumbent, family-bootstrap 95% CIs. **Strict (H448 rule):** mean > 0.002 **and** gain CI low > 0 **and** Elo > 0 **and** Elo CI low > 0. Ties never win (they count 0.5 in Elo). Per-seed selection is by OOF Elo with no val input. val91 is read once per seed, for the selected arm only.
- **Control:** `big_ctrl` reproduces H450 `big_ctrl` bit-for-bit (5 of 5 fold adapter shas and OOF vectors identical on both seeds).

## Outcome
**CV (pooled OOF over 1,219 TRAIN contexts, seeds 1 / 2):**

| Arm | OOF gain | Gain CI (s1) | Elo | Elo CI (s1) | W / T / L (s1) | Non-floor Elo | Recovered / new gate fails (s1) | Strict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| big_ctrl | +0.0244 / +0.0200 | [+0.008, +0.042] | +0.6 / −2.3 | [−11.8, +12.7] | 238/745/236 | −2.3 / −5.2 | 10 / 0 | 0/2 |
| big_k50 | +0.0058 / +0.0054 | [−0.010, +0.020] | −0.3 / −2.6 | [−9.0, +7.6] | 119/980/120 | −1.7 / −4.1 | 5 / 0 | 0/2 |
| big_k25 | +0.0015 / −0.0011 | [−0.009, +0.011] | −0.9 / −2.0 | [−7.1, +4.9] | 59/1098/62 | −1.4 / −2.6 | 2 / 0 | 0/2 |
| pairnf | +0.0095 / +0.0102 | [−0.015, +0.034] | −7.1 / −5.1 | [−23.3, +8.3] | 340/514/365 | −10.7 / −8.4 | 15 / 3 | 0/2 |
| pairnf_k50 | +0.0095 / +0.0044 | [−0.003, +0.024] | −1.7 / −2.6 | [−14.0, +9.4] | 174/865/180 | −4.1 / −4.1 | 8 / 0 | 0/2 |
| **pairnf_k25** | **+0.0098 / +0.0077** | [+0.003, +0.019] | **+6.6 / +6.0** | [−1.6, +14.4] | 100/1042/77 | **+5.5 / +5.2** | 4 / 0 | 0/2 |

Fold gains (seed 1): `pairnf_k25` +0.005, +0.038, +0.002, +0.003, +0.001; `big_ctrl` +0.015, +0.052, +0.031, +0.008, +0.016; `pairnf` +0.034, +0.056, −0.012, −0.014, −0.016 (all seeds in `metrics.json`).

**val91 regression check (one read per seed, CV-selected arm = `pairnf_k25` on both seeds):**

| Seed | Gain vs incumbent | Gain CI | W / T / L | Elo | Changed / kept | New gate failures | not_worse |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | +0.0030 | [−0.0022, +0.0085] | 8/77/6 | +7.6 | 54 / 14 | 0 | true |
| 2 | +0.0034 | [−0.0011, +0.0082] | 7/77/7 | 0.0 | 54 / 14 | 0 | true |

val91 has 0 incumbent gate failures, so the whole val91 read is non-floor. It is the first H447+ regression check that is not negative on Elo for every seed (H450: −19.1 / −11.5 / −15.3), but 14 kept changes per seed are far too few to separate from 0.

## Red-team (`h451_redteam.py`, output under `redteam` in `metrics.json`)
### (a) Floor recovery vs non-floor
The incumbent's 19 TRAIN gate failures fall 5/7/4/2/1 over folds 0–4. Per-arm split (seed 1; seed 2 in `metrics.json`):

| Arm | Gain from 19 floors | Non-floor mean gain [CI] | Non-floor W/T/L | Non-floor Elo [CI] |
| --- | --- | --- | --- | --- |
| big_ctrl | +0.0213 | +0.0031 [−0.008, +0.013] | 228/736/236 | −2.3 [−15.6, +10.5] |
| pairnf | +0.0318 | −0.0145 [−0.031, −0.0003] | 325/510/362 | −10.7 [−26.6, +5.0] |
| pairnf_k50 | +0.0145 | −0.0051 [−0.014, +0.002] | 166/854/180 | −4.1 [−15.5, +7.1] |
| pairnf_k25 | +0.0072 | +0.0026 [−0.0015, +0.0069] | 96/1027/77 | +5.5 [−2.8, +13.9] |

Ungated `pairnf` recovers the most floors (15 of 19) but loses on non-floor contexts (seed 1 non-floor gain CI below 0) and adds 3–4 new gate failures. The pairwise loss alone makes changed picks worse. The 25% gate is what turns it positive: it keeps about 180 of 705 changes, and on those W/L is 96/77.

### (b) Fold-1 concentration
For the selected arm, fold 1 carries 76–78% of the **total** OOF gain, but that share is floor recovery. The **non-floor** fold gains of `pairnf_k25` are positive on every fold and both seeds (s1 +0.005, +0.002, +0.002, +0.003, +0.001; s2 +0.001, +0.002, +0.002, +0.004, +0.002). With fold 1 dropped, Elo is still +4.6 / +3.9 (CI [−4.0, +12.8] / [−4.5, +11.8]). So for the first time in H447–H451 the fold-1 artefact is not the whole story, but the remaining effect is small. It is still one fixed partition, which H452 addresses.

### (c) Selection and winner's curse
- All 6 arms × 2 seeds are reported. Selection used OOF Elo only, with no val input.
- On 1,000 family bootstraps, `pairnf_k25` stays the max-Elo arm 74% / 82% of the time. The max-arm Elo CI is [−0.5, +15.7] / [−1.2, +14.5], so the best of 6 is not separated from 0 even before any optimism correction.
- The z-gate thresholds are quantiles over the evaluated set (the 244-context held-out fold, or val91), which is transductive but label-free. On val91, k25 keeps 14 of 54 changes. An absolute z threshold would be set-independent; H452 tests one.
- z-margin vs outcome (ungated `pairnf`, changed decisions): win rate rises only in the top quartile (W/L 100/77 and 102/74 vs W 68–92 / L 85–109 in the lower three). Within z ≥ 0.3 it does not keep rising (z 0.3–0.4: 54/41; 0.4–0.5: 41/27; 0.5–0.6: 14/9; ≥ 0.6: 2/4 on seed 1), so a tighter gate mostly shrinks n.

### (d) Seed noise and independence
- Fold-gain vectors differ for every seed pair and arm, and every (arm, fold) has distinct adapter shas (aggregator asserts). The val91 adapters are distinct too.
- The seeds differ on only 16–118 OOF contexts (pairnf_k25: 38), so 2 seeds are close to 1 effective replicate. The verdict rests on the family-bootstrap CIs.
- A consensus check (keep a change only where big_ctrl and pairnf make the same change) gives 141 / 143 contexts with W/L 72/69 and 71/72. Agreement between objectives carries no extra signal.

### (e) Other checks
- **Leakage:** train/val family overlap is 0. Folds are family-disjoint and exclude val families (asserted). The pairwise loss and hinge use training-fold contexts only. The gate uses predictions only. val91 was read once per seed, for the CV-selected arm.
- **Metric direction and ties:** utility is higher-is-better (negated fitness, −2.0 on gate fail). The aggregator recomputes both strict rules from per-seed fields. Ties count 0.5 in Elo and never pass strict. The gated arms create ties by design. Their Elo gain comes from W > L on kept changes, not from ties.
- **Data / safety:** DL-Train/DL-Val only, with no sealed or holdout reads. DOPE `c51a56d0` is intact on every DONE line. There was no guard abort and no OOM (minimum MemAvailable 17.5 GB, peak RSS 6.2 GB per seed). xbabe1 disk is at 98% (50 GB free). The wave directory is 465 MB, and nothing large is committed.

## Meaning
Pairwise training against the incumbent's pick, on its own, makes the candidate change more picks and lose more of them. The label-free confidence gate is the only H444–H451 lever that keeps W > L on non-floor contexts. It gives about +5 non-floor Elo across all folds and both seeds and is not negative on val91, but it is inside the noise (CI ≈ ±8 Elo) and comes from a best-of-6 selection on one fixed partition. It is a lead to confirm, not a lift.

## Next step
H452 (`~/jope-data/hwave-h452` on xbabe1) is a confirmation-and-dose wave for the gate:
- **Fresh partitions:** two new family-grouped 5-fold partitions (`h452a/`, `h452b/` salts), each stratified so the families holding the incumbent's TRAIN gate failures are spread round-robin over folds.
- **Bases:** `pairnf` (unchanged) and `pairnf_w3` (pair weight 3.0, listwise weight 0.5).
- **Gates:** none, `_k25`, and an absolute, set-independent `_z40` (z ≥ 0.40).
- **Selection:** by mean non-floor OOF Elo over the two partitions.
- **Strict:** adds non-floor Elo CI low > 0 and Elo > 0 on each partition.

No adoption nudge was sent to the flywheel owner.

Artifacts: `metrics.json` (aggregate per arm and seed, fold gains, regression checks, decision diversity, big_ctrl vs H450 bit-exactness, z-margin quartiles, CV gate split, peaks, plus the `redteam` block and seed-3 status), `h451_trainfold_cv.py` (as run), `h451_aggregate.py` and `h451_redteam.py`. Raw logs and fold signatures stay on xbabe1 in `~/jope-data/hwave-h451`.
