# H143 — shadow replay of the strict-margin admission rule (H139 next step)

**Owner:** `rd_opus`. Claim: `~/jope_plan/ns/receipts/H143_claim.txt`. Brief: hwave overseer
`h143_rd_opus_t1905`. CPU only, **zero optimizer updates**, offline.

The live flywheel is untouched: reads are copies of `run-20261030` on xbabe1, and the live directory is never
written. This file is committed and merged **before** any replay statistic is computed.

## Hypothesis

H139 found r882's plateau to be a selection artefact of the single-mean argmax rule, which is dominated by
invalid-pick (−2) cliffs. Replaying the loop's admission history under R_STRICT should change the incumbent
trajectory. R_STRICT accepts only when the family-cluster bootstrap median gain over the incumbent exceeds +0.001,
and rejects ties. The final incumbent should then be no worse than r882 on DL-Train and should come from fewer
cliff-driven admissions.

**How this differs from earlier waves.**
- H61 compared S0–S3 on identical live-step candidates.
- H114 and H123 audited the lattice and the tie-break.
- H143 replays the **whole trajectory** under the H139 rule.

## Data (pinned at claim time, 2026-11-01T01:03:15Z, copy-only)

**Window.** Rounds **775–2530** (1,756 contiguous rounds), evaluation identity `e79a7715…`. Each per-round receipt
`round-N.json` records:
- `candidate_dev_utility.per_context`, the 91 per-context val utilities of that round's candidate;
- `dev_utility`, the incumbent after the decision.

The copies are pinned as follows:
- receipts: SHA-256 list `b7d3a5a5…`;
- `adapter_admissions` lines 1–1,756: `833d27cd…`;
- `best_dev_checkpoint.json`: `30853cad…`.

**Coverage.** Coverage is **complete**: all 1,756 candidates have per-context val utilities, so no rejected
candidate needs rescoring for the replay.

Structural checks run before this preregistration. They compare fields only, with no rule statistic computed:
- each candidate mean equals the logged `candidate_utility` (1,756/1,756);
- the incumbent before round t equals `dev_utility` of round t−1 (1,755/1,755, with one 1-ulp mean difference at
  round 775);
- the starting incumbent is round 775's `initial_dev_utility`, the checkpoint-774 state.

**Checkpoints** (copies) are used to score final incumbents on DL-Train:

| Round | Copied from | SHA-256 |
|---|---|---|
| 774 | `flywheel-gp-fused/run-20261029/round-774.pt` | `60430aa6…` |
| 782 | retained | `427b0d46…` |
| 785 | retained | `ef6696f6…` |
| 859 | retained | `5802245f…` |
| 882 | retained | `0084479a…` |

**Rejected candidates as final incumbents.** If a replayed final incumbent is a rejected candidate from round
≥ 1322, its exact checkpoint is fetched from that round's S3 retention archive (`rejected/round-N.pt`). The fetch is
version-pinned and sha-verified against the archive manifest, and recorded in `extra_checkpoints.json` before
DL-Train scoring. Rejected candidates from rounds 775–1321 have no retained checkpoint: a seed whose final incumbent
is one of them is **unavailable** on DL-Train.

**Limitation (plain).** The candidate stream is the *logged* one. A different admission rule would have changed the
incumbent the loop trained from, and so the later candidates. Those counterfactual candidates cannot be reproduced
offline. H143 measures what R_STRICT would have **admitted from the candidates that actually existed**.

## Arms and rules

The 19 val families come from the selected packet's DL-Val rows, in contract order.

| Arm | Rule |
|---|---|
| **S_LIVE** | The logged rule: accept when the 91-context mean is greater than or equal to the incumbent's (ties accepted). Used as the regression check. |
| **R_STRICT** | Per-context gain = candidate − incumbent. Take the per-family mean over the 19 families, then 2,000 family-cluster resamples of the equal-family mean. **Accept iff the median exceeds +0.001**; exact ties therefore fail. |
| **R_STRICT_NOINV** | Same as R_STRICT, but contexts where either side picked an invalid program (−2) are excluded, and families left empty are dropped. Sensitivity arm. |

- **RNG.** Each strict arm runs with seeds **143101, 143102, 143103**. One RNG per seed is consumed sequentially
  over the rounds.
- **Reporting.** Invalid-pick counts on both sides are logged for every round. Each admission is also flagged
  *cliff-driven* when its invalid-excluded mean family gain is ≤ +0.001.

## Steps and readout

1. **Regression (stage `replay`).**
   - S_LIVE must reproduce every logged action (accept_improvement, accept_tie or restore) for all 1,756 rounds.
   - Its final incumbent must be round **882**, whose per-context val utilities must equal `best_dev_checkpoint.json`
     on **91/91** contexts, with mean 1.1377942.
   - Otherwise the run stops.
2. **Replay.** R_STRICT and R_STRICT_NOINV run for each seed. For every arm and seed, report:
   - the final incumbent;
   - the number of admissions and the list;
   - the final val mean, which is **descriptive only**, because val is the selection data;
   - invalid counts;
   - cliff-driven admissions;
   - whether trajectories are identical across seeds.
3. **Extra checkpoints.** Fetch any archived checkpoint that the rejected-candidate procedure above requires. Nothing else is
   fetched.
4. **DL-Train (stage `dltrain`).** Score each distinct final incumbent and R882 with the pinned H124 forward pass
   (`9ebce965`): native endpoint on the 1,219 DL-Train contexts in 205 families.
   - Report the equal-family gain against R882, with a family-bootstrap 95% CI (10,000 draws, RNG 143000).
   - Report Elo with R882 = 1500 (canonical tracker, 1,000 family resamples, RNG 143001). AUC is not used.
   - Report U4 and DL-Train invalid counts descriptively.
   - R882's rescored val must again reproduce the reference on 91/91 contexts.
   - Because the final incumbents are fixed and deterministic, five-fold family cross-fit partitions give identical
     estimates (as disclosed in H139). Inference is the family bootstrap.
   - DOPE `c51a56d0` is checked before and after every forward pass.

## Verdict (a priori, R_STRICT)

| Verdict | Condition |
|---|---|
| **RULE_HELPS** | R_STRICT's final incumbent beats R882 on DL-Train (CI lower bound > 0) on ≥ 2 of 3 seeds |
| **RULE_HURTS** | CI upper bound < 0 on ≥ 2 of 3 seeds |
| **RULE_NEUTRAL** | otherwise; this includes a final incumbent equal to r882 |
| **INCOMPLETE_COVERAGE** | DL-Train is unavailable for ≥ 2 of 3 seeds, or the regression fails |

R_STRICT_NOINV is a sensitivity arm and never sets the verdict.

If the verdict is RULE_HELPS, the closeout adds an adoption note for the flywheel owner: shadow trial first, with no
live change.

## Caveats and resources

- **Caveats.**
  - DL-Train is training data for every checkpoint. It is independent of the selection, not of training.
  - Val utilities are selection data and are reported descriptively only.
- **Data closure.**
  - Zero sealed, holdout or meta-test reads.
  - `source_manifest.json` and `cache_manifest.json` are not opened.
  - Real S3 / custody data only.
- **Resources.**
  - xbabe1, CPU only, nice 19, OMP/MKL/OpenBLAS = 4.
  - No `qualification.lock` and no GPU.
  - RSS < 6 GiB (H124 watchdog).
  - **Abort if host free disk < 40 GB.**
  - Scratch < 3 GB, deleted at closeout after S3 custody.
  - The run executes once.
