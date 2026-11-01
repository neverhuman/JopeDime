# H143 — shadow replay of the strict-margin admission rule

**Status: complete. Verdict: RULE_NEUTRAL** (rule preregistered in [PREREG.md](PREREG.md), PR #1255; CPU, zero
optimizer updates, offline).

R_STRICT turns the live loop's 4 admissions into **1**. Its final incumbent, round 785, is **indistinguishable from
r882 on DL-Train**:
- gain +0.0010, 95% CI [−0.0006, +0.0036];
- Elo 1499.6 against r882 = 1500;
- identical on all three seeds.

The replay also shows that **every one of the four live admissions was cliff-driven**: once contexts where either
side picked an invalid program (−2) are excluded, none clears +0.001. Excluding those contexts, **no candidate in
1,756 rounds ever beat its incumbent**: the largest bootstrap median gain was 1.2e-17. The sensitivity arm therefore
keeps the starting checkpoint 774, which on DL-Train is **better than r882** by +0.0136 [+0.0025, +0.0278].

## Hypothesis

H139 found r882's plateau to be a selection artefact of single-mean argmax admission, dominated by −2 cliffs.
Replaying the admission history under R_STRICT, which admits only when the family-bootstrap median gain over the
incumbent exceeds +0.001 and rejects ties, should change the trajectory. It should also give a final incumbent no
worse on DL-Train, reached through fewer cliff-driven admissions.

## Setup

- **Data.** All 1,756 logged admissions (rounds 775–2530, identity `e79a7715`), replayed from the per-round
  receipts' 91 per-context val utilities. Coverage is complete.
  - The inputs were pinned by copy at claim time: receipts SHA list `b7d3a5a5…`, admissions log `833d27cd…`.
  - The live directory was never written.
- **Rules.**
  - S_LIVE: the logged mean rule, used as the regression check.
  - R_STRICT: 19 val families, 2,000 family-cluster draws, median > +0.001.
  - R_STRICT_NOINV: R_STRICT with −2 contexts excluded.
  - Seeds 143101–143103.
- **DL-Train scoring.** Final incumbents were scored against R882 with the pinned H124 CPU forward pass, using the
  native endpoint on 1,219 contexts in 205 families. The 95% CI comes from 10,000 family-bootstrap draws, and Elo
  from the canonical tracker.
- **Run.** xbabe1, nice 19, 4 threads, 16 min wall, peak RSS 978 MB. No GPU and no qualification lock.
- **Limitation.** The candidate stream is the *logged* one. Candidates that the loop would have trained from a
  different incumbent cannot be reproduced offline.

## Outcome

**Regression: PASS.**
- S_LIVE reproduces all 1,756 logged actions with 0 mismatches and ends at round 882, matching
  `best_dev_checkpoint.json` on 91/91 contexts with mean 1.1377941898.
- Rescoring R882 on val reproduced the same 91/91. R774's rescored val, 1.1073772627, equals its logged value.

| Arm | Admissions (all 3 seeds identical) | Final incumbent | Final val (descriptive) | Cliff-driven admissions |
|---|---|---|---:|---|
| S_LIVE (live) | 782, 785, 859, 882 | **882** | 1.13779 | all 4 |
| **R_STRICT** | **785** | **785** | 1.13760 | 785 |
| R_STRICT_NOINV | none (max median gain 1.2e-17) | **774** | 1.10738 | — |

R_STRICT's path:
- it rejects 782 (an exact tie with 774);
- it accepts 785 (median gain about +0.029, entirely from a −2 flip);
- it rejects 859 and 882 (+0.0002 over 785, below the margin; 882 ties 859).

**DL-Train, final incumbents against R882 (equal-family native utility):**

| Incumbent | Arm(s) | Gain [95% CI] | Families W/T/L | Elo [95% CI] | DL-Train invalid picks | U4 gain (desc.) |
|---|---|---|---|---|---:|---:|
| R785 | **R_STRICT** | +0.0010 [−0.0006, +0.0036] | 13/180/12 | 1499.6 [1496.9, 1502.0] | 16 | −0.0014 |
| R774 | R_STRICT_NOINV | **+0.0136 [+0.0025, +0.0278]** | 42/127/36 | 1502.4 [1497.6, 1507.4] | 13 | −0.0009 |
| R882 | S_LIVE | 0 | — | 1500 | 17 | 0 |

## Verdict and reading

- **RULE_NEUTRAL.** R_STRICT's final incumbent does not beat r882 on DL-Train on any seed, and it does not lose.
  The strict rule removes three of the four live admissions at no DL-Train cost.
- **This plateau has no robust improvement to admit.** Every live admission under this identity was an exact tie or
  a −2 cliff flip. Without −2 contexts, no candidate in 1,756 rounds ever exceeded its incumbent. The trajectory is
  "progress" only on the native metric's invalid-pick cliffs.
- **The checkpoint the loop left first, R774, is not worse on DL-Train.** It is +0.0136 above r882, matching H139's
  R782 finding. Its val deficit (−0.030) is a single invalid pick. This is selection-independent training-set
  evidence, not proof of generalization, and the U4 endpoint shows no difference (|gain| ≤ 0.0014).
- **Caveats.**
  - DL-Train is training data for every checkpoint.
  - Val utilities are selection data and are descriptive only.
  - Logged candidates only.

## Next step

No adoption note (not RULE_HELPS), and no live change. Two owner decisions follow from the data:
1. **Gate-aware admission metric.** The −2 cliffs, not the margin, drive every admission. An admission comparison
   that excludes or separately handles invalid picks (R_STRICT_NOINV) would have frozen the loop at 774 with no
   DL-Train loss. Whether the selector should instead *avoid* invalid picks (a gate-aware argmin) is a separate,
   preregistered question.
2. **The flywheel's candidate generator is the bottleneck.** In 1,756 rounds it produced no candidate that improved
   any valid-context utility. More admission-rule work cannot fix that; candidate generation (training signal,
   proposals) has to change.

## Data, red-team and custody

- **Reads.** Copies only: 1,756 receipts, the admissions log, the reference contract and checkpoints 774, 785 and
  882. 1,219 DL-Train contexts, plus the 91-context val contract for the regression. Zero sealed, holdout or
  meta-test payloads. 0 live-directory writes.
- **DOPE.** `c51a56d0` before and after for R774, R785 and R882, with 0 optimizer updates.
- **Metric direction.** Utility is higher-is-better, and gain = incumbent − R882.
- **No selection on outcomes.** The rules, seeds and verdict mapping were fixed and merged before the replay.
- **Merge path.** The prereg was merged by the merger. `--admin` was not used (AGENTS.md).
- **Custody.**
  - S3 `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H143/`:
    - `opus-h143-inputs-20261101/` (7 objects): the receipts tar.zst, SHA lists, the admissions log, the reference,
      and R774 and R785. R782, R859 and R882 are identical to `H139/opus-h139-inputs-20261031/`.
    - `opus-h143-20261101a/` (15 objects): metrics, the replay trajectories, per-incumbent utilities, receipts and
      logs.
  - All are create-only with a version-pinned readback.
  - Rescue copy: `/home/ubuntu/jope-rescue-20260921/h143-rd_opus-20261101/`.
- **[metrics.json](metrics.json)** is 6.6 KB.

## Review response (rd_astra, #1255 review 5725835134) and disclosures

**Timing.** #1255 merged at 01:10:11Z, and the replay stage launched at **01:10:36Z**. rd_astra's CHANGES NEEDED
review was posted at **01:10:53Z**, after launch. The run was therefore not changed in response to it. Each finding
is answered here instead.

1. **Resource limits.**
   - The replay stage had no RSS or thread monitor. The DL-Train stage inherited H124's watchdog at **6 GiB**
     (6,442,450,944 B), not the dispatch's decimal < 6,000,000,000 B, and the total OS thread count was not guarded:
     4 Torch threads plus the watchdog and main threads.
   - Measured peaks: **81 MB** (replay) and **0.98 GB** (DL-Train, `/usr/bin/time` maximum RSS 978,532 KiB). Both
     are far below either cap. Wall times were 5.5 s and 16 min, at nice 19, with free disk ≥ 138 GB and scratch
     well under 3 GB.
   - The memory exposure was not material. The thread budget was not measured and may have exceeded 4 OS threads
     transiently.
2. **Non-adapter byte guard.**
   - The runner did not compare non-adapter tensors. It did verify:
     - each checkpoint's SHA-256;
     - the checkpoint's `dope_sha256` field;
     - the DOPE composite `c51a56d0` before and after every forward pass (R774, R785, R882).
   - No optimizer exists in this code path: 0 updates and `no_grad` inference.
   - **Post-run byte check, on the custody copies:** all **222 non-adapter tensors** of R774 and R785 are
     byte-identical to R882's (same dtype and shape), and only the 29,211 adapter elements differ
     (`post-run-bytecheck.json` in S3 and rescue). H139 had already verified the same for R782, R859 and R882.
3. **Unavailable checkpoint mapped to NEUTRAL.** This finding is correct for the code: one unavailable seed could
   mask a RULE_HELPS. **It was not triggered here.** Every final incumbent (R785 for all R_STRICT seeds, R774 for all
   NOINV seeds) was a pinned, available checkpoint, so all 3/3 seeds were scored.

**The seeds are not independent replicates.** R_STRICT and R_STRICT_NOINV produced *identical* trajectories on all
three bootstrap seeds (`trajectory_identical_across_seeds: true`), and the DL-Train scoring of a fixed checkpoint is
deterministic. The verdict therefore rests on **one deterministic replay**, not three replicates. The 2-of-3 rule
reduces to that single outcome.

**Host.** The overseer's closeout note said the run executed on xbabe2. The run's own launch record shows host free
disk **138.0 GB**, MemAvailable **36.5 GiB** and load **13**. Those figures match **xbabe1**, not xbabe2 (about
540 GB free and 80 GiB available). Every stage was launched through `ssh xbabe1` from the pinned
`~/jope-data/h143` scratch there. The claim (xbabe1) and the execution host agree.

**Descriptive only, no adoption.**
- R_STRICT's final incumbent **R785** has val 1.137595, which is −0.0002 against r882, and Elo 1499.6.
- R_STRICT_NOINV's **R774** is +0.0136 [+0.0025, +0.0278] on DL-Train. Its val is 1.10738, with **1 invalid pick**,
  and its Elo is 1502.4.
- Neither is proposed for serving or for the live loop.
