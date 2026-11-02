# H200 — nested lambda selection and rank-space GBM blend

**COMPLETE; NO_ADVANCE, with protocol deviations recorded.** The missing closeout
was a publication gap: attempt-001 finished on xbabe1 on 2026-11-01, with 60/60
GBM fits, 15 locked outer cells and complete three-seed A panels. It was neither
abandoned nor replaced by H203. Original owner: luna_ns_hbaseline; recovered
closeout: rd_astra / [ASTRA], under `gbw_2040_rd_astra.md`.

## Hypothesis and setup

[PREREG.md](PREREG.md) asks whether nested FIT-only lambda selection improves the
frozen r882/GBM blend, with a fixed lambda0.25 control, rank-space blend and
shuffled-score placebo. The real-S3 H139/H195 cohort has 1,219 TRAIN contexts in
205 families; five family folds per seed use A=k, B=(k+1)%5 and three FIT folds.
Seeds are 200101–200103; fold seeds 200141–200143; bootstrap seed200150. The
primary is NEST_L, with Holm m=2 over NEST_L/RANK025. Neural optimizer updates=0;
DOPE stayed `c51a56d0`; CPU-only, no serving promotion.

Execution source was `af2825452ec98bd0fc044c826f3c812f1fc02339` after prereg
[#1668](https://github.com/neverhuman/JopeDime/pull/1668). The original stdout
records exact native r882 **91/91 DL-Val utility equality**, mean
**1.1377941898164525**, before TRAIN outcomes/arms. DL-Val was a baseline check,
not a treatment endpoint. The TRAIN gains below cannot be added to that DL-Val
mean. Original execution bytes remain in
[execution-manifest-attempt-001.json](execution-manifest-attempt-001.json).

## Results against native r882

All numbers are equal-family TRAIN A gains; CI is the registered 10,000-draw
family bootstrap of the three-seed median. Harm means a family's three-seed
mean gain is below −0.01. Elo anchors r882 at1500 and is descriptive.

| Arm | Gains by seed (200101 / 200102 / 200103) | Median gain [95% CI] | Harm | Elo median |
|---|---|---|---|---|
| NEST_L (primary) | +0.035741 / +0.020570 / +0.037995 | +0.035741 [−0.000677, +0.072160] | 71/205 (34.63%) | 1510.23 |
| FIX025 | +0.041455 / +0.026303 / +0.029400 | +0.029400 [+0.004677, +0.064115] | 64/205 (31.22%) | 1515.33 |
| RANK025 | −0.049718 / −0.024359 / −0.020487 | −0.024359 [−0.120742, +0.042223] | 82/205 (40.00%) | 1521.53 |
| NEST_SHUF | −0.025076 / −0.000549 / −0.003151 | −0.003151 [−0.025157, +0.009563] | 58/205 (28.29%) | 1499.94 |

NEST_L raw one-sided p=0.032397, Holm p=**0.064794**. It fails all three
registered admission gates: LCB95>0, Holm p<0.05 and harm<=10%. Its 3/3 positive
seed gains do not override those failures. NEST_L minus FIX025 median gain is
−0.005714 [−0.015520, +0.011317]. Selected NEST_L lambdas by outer fold:
200101=[0.4,0.5,0.3,0.5,0.4], 200102=[0.4,0.3,0.3,0.5,0.4],
200103=[0.1,0.3,0.2,0.4,0.3]. RANK025's positive win-based Elo and negative
utility gain measure different quantities; Elo does not establish admission.
Full precision, ties, seed Elo/CI, placebo selections and exact −2 counts are in
[metrics.json](metrics.json).

## Verification and limitations

[replay.json](replay.json) independently reconstructs every reported contrast,
bootstrap interval/p, harm count, Elo interval, Holm correction and verdict from
the saved panels, exactly. It also verifies all60 prediction-file hashes,15
selection locks, family folds and9 execution-source pins. This was a CPU-only
saved-output replay, not a new fit or native inference run. Native91 equality
is supported by the original stdout and source assertion; it was not rerun.

**Protocol deviations:** seven fits ended before300 iterations (187–295), despite
the preregistration saying no early stopping. The runner inherited H124's model
configuration without explicitly disabling sklearn early stopping. Preserve the
observed fits and result as executed; this is not a fully conformant test of the
written no-early-stopping protocol. A future repair would require a new prereg,
not a relabel or silent rerun of this attempt.

The original monitor reported exit0/PASS after7,472.797s, peak RSS3.765GB and
worker CPU3.991cores. It enforced >=40GB free disk but **did not enforce the93.8%
usage ceiling**: launch recorded94.242%, observed samples ranged94.129–95.795%.
Its historical PASS is not a93.8%-guard compliance claim. The present closeout
and saved-output replay use the current93.8% guard; no run is resumed. Peak13
threads is a historical observation under the original four-core CPU limit,
not the later H196/H207 aggregate-thread rule.

## Custody and what to try next

[custody.json](custody.json) binds a create-only H200 S3 archive with all recovered
attempt outputs, fitted prediction arrays, panels, folds, lock, stdout, resource
samples, original source and closeout replay receipts. Exact-version GET, payload
SHA256 and every archived member were verified; original xbabe1 files also match
the copied evidence. [manifest.json](manifest.json) records these URI/hash pins.
Input corpus payloads remain in the pinned H139/H195 lineage; this output archive
does not claim to duplicate the entire input closure. No HWM candidate is proposed.

Keep the measured NO_ADVANCE verdict. Mean gain alone did not meet the harm gate;
follow the separately preregistered [H203 harm-gated fixed blend](../h203_blend025_gated/PREREG.md)
under its owner's existing budget and protocol. H203 cites H199 and is a related
question, not evidence that H200 was superseded or that H203 succeeded. Before any
new nested-lambda attempt, explicitly disable early stopping, test the iteration
contract, enforce the current disk guard and preregister fresh execution. This
closeout authorizes no additional scientific attempt. Central merger only.
