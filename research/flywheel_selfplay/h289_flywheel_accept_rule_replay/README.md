# H289: recorded flywheel acceptance-rule replay

**COMPLETE — KEEP_RULE.** No tested rule accepted a later candidate. This is evidence from reused DL-Val91 records, with no fresh evaluation or permission to change the live writer.

## Hypothesis and setup

The hypothesis was that r3175's lead over the plateau falls within family-bootstrap noise and that a robust acceptance rule would admit a later candidate without accepting ties. The frozen [preregistration](PREREG.md) was published at `5773d6e29fd8600174f94ccf973f9624f4da8ee0` before the single analysis attempt. It specifies every transform, margin, resampling rule and decision condition.

Read-only capture on xbabe1 covered r3175–4675. The analysis qualified **1,499 later candidates**, each with the same 91 contexts, 19 families, evaluation identity and DOPE identity. Round **4331 was explicitly unevaluated** (`candidate_evaluated=false`, candidate utility absent) and was excluded, never filled with an incumbent or zero. The last-200 window, r4476–4675, is complete. Utilities come from `candidate_dev_utility`; the restored incumbent's `dev_utility` is not a candidate outcome. Recorded selected-program hashes were joined to the immutable DL-Val pool fitness/gate arrays, with exact utility equality and no reevaluation.

The baseline regression reproduced **1.1389572403748425 bit-exactly**. Bootstrap seeds 289101/289102/289103 each use 10,000 paired draws of 19 families. Raw gain comparisons retain the original context weighting. Cross-seed medians and interval endpoints summarize Monte Carlo stability on one dataset, not three independent training runs.

## Outcome

| Rule | Accepts by seed | Accepted rounds | Final incumbent |
| --- | --- | --- | --- |
| R0: raw mean, excluding gains ≤0.0001 | 0 / 0 / 0 | none | r3175 |
| R1: paired-family median >0.002 and bootstrap LCB >0 | 0 / 0 / 0 | none | r3175 |
| R2: clip context utility below at −1, then R1 | 0 / 0 / 0 | none | r3175 |
| R3: past-only within-family ranks, then R1 and raw tie guard | 0 / 0 / 0 | none | r3175 |

The literal `mean > best` audit also yields zero accepts and matches the recorded journal. Every later raw mean is below r3175 (best gain −0.000400626; median gain −0.008365374). For R1–R3, every observed paired-family median gain is exactly zero. They fail the strict observed margin before an LCB calculation is needed; those event CIs are **null**, not fabricated zero intervals. The 10,000-draw bootstrap is performed for the winner/null and reported comparisons; it is short-circuited for these already-failed acceptance gates. All four terminal policies equal r3175: raw gain 0, CI [0, 0], Elo 1500.

| Winner comparison | Bootstrap median | 95% interval |
| --- | ---: | ---: |
| r3175 minus last-200 plateau profile | +0.006733783 | [+0.000047127, +0.015372054] |
| Centered best-of-200 null maximum | +0.002550294 | [−0.002483474, +0.011188826] |
| Final recorded candidate r4675 minus r3175 | −0.003370307 | [−0.008919191, 0] |
| r4675 pairwise Elo, r3175 fixed at 1500 | 1492.692 | [1481.716, 1500] |

The observed r3175-minus-plateau mean is +0.007055611. Each seed's plateau interval narrowly excludes zero: lower endpoints +0.000047127 / +0.000025882 / +0.000051455. The observed lead is below each centered best-of-200 null's upper endpoint, but the **conjunction** required by the preregistration fails on all three seeds. This does not establish the absence of selection bias: the centered null is conditional on these recorded candidates and does not reconstruct selection over the earlier adaptive history.

For individual comparisons, **157/200** paired intervals include zero on each seed. Across the last 200 rounds, bootstrap P(r3175 > candidate) has per-seed medians 0.9617 / 0.9593 / 0.9636 (ranges 0.6435–0.9885 / 0.6381–0.9876 / 0.6348–0.9882). These are bootstrap fractions, not posterior probabilities. Candidate Elo bootstrap medians range 1481.716–1496.346; their descriptive cross-round median is 1489.036. Complete per-round probabilities, strict-win fractions, Elo, medians and intervals, plus best-of-N distributions for N=1/5/10/25/50/100/200, are preserved in the evidence archive.

## Loss attribution and interpretation

In r3175 and the last 200 candidate vectors, **zero selected programs are invalid**. The −2 invalid tail contributes **0%** of the +0.007055611 plateau lead. Clipping at −1 removes zero of that lead; the whole difference is between valid selections. That does **not** mean broad improvement across families: four contexts account for **96.843% of the net lead**, and only seven families have nonzero plateau contributions. The family-median rules remain at zero under these sparse changes, so they do not resolve the acceptance drought in this replay.

Neither required condition supports a rule change: there is no accepted qualifying witness, and the preregistered winner-noise conjunction fails. **KEEP_RULE** means retain the current rule under this bounded test; it does not qualify the learning loop or prove the incumbent generalizes. These are recorded-path policy replays: changed acceptance would not reproduce the subsequent optimizer or GP trajectory. Any future alternative remains **SHADOW ONLY** and requires a fresh test before adoption. No writer/config change or additional experiment was performed.

## Verification and custody

The independent audit reconstructs chronological incumbents and past-only rank references, verifies all **17,988** rule decisions and saved family draws, and checks the null, intervals and invalid-tail decomposition. Eight synthetic tests passed, along with source checks, formatting, JSON/hash/link checks and exact-head diff checks. Hosted CI is **NOT_RUN** under the owner's research publication split; this is not a claim of CI coverage. The audit is same-agent verification, not an external review.

Before analysis, the preparation parser was corrected to allow the selected packet's existing train/inner source rows (all DL-Val); the initial train-only assertion had stopped before constructing a replay matrix. No outcomes were computed then, and the corrected source was frozen before the sole analysis attempt. There were no scientific retries or post-outcome rule changes.

The analysis used CPU48,50–52, nice19, one numeric thread; monitored peak aggregate threads 2 and RSS 154,963,968 bytes. The guard enforced aggregate ≤4, RSS ≤4 GiB and MemAvailable ≥10 GiB. There were **zero neural forwards, zero updates, no checkpoint loads, no fresh-cohort packets and no GPU work**. DOPE `c51a56d0` and the live writer/observer were untouched. The remote read-only capture process exited.

[Metrics](metrics.json), [manifest](manifest.json), [audit](audit.json) and [output custody](output-custody.json) bind the result to the frozen source and exact-version S3 evidence. [Input custody](input-custody.json) records the 120,115,338-byte compressed / 295,027,563-byte raw capture; [prepared custody](prepared-custody.json) records the matrix and selected-program evidence. Full outcomes were archived and exact-VersionId GET/member-hash verified **before** producing compact Git metrics. The output archive includes all 200 comparisons, 17,988 events, draws, per-context attribution, source, command/resource logs and review receipts available at analysis close. All keys are create-only under `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H289/astra-t0005/`.

Publication is non-force to `origin/v1/next` only, with the two H289 index rows marked `GIT_COMMITTED`. No main/fastlane merge or hosted CI dispatch. Final publication receipts are retained under the same S3 prefix's `lifecycle/` and local custody `/home/ubuntu/jope-rescue-20260921/h289-astra/`.
