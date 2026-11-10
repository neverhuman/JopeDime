# H288: confirmatory, powered, GPU-exact fresh test on ODD cohort v3

**Status: complete. There is no serving or promotion change.**

| Arm | Role | Verdict |
|---|---|---|
| BAG_SWITCH | **primary** | **NOT_TRANSFERRED_FRESH** |
| BAG_SWITCH_NESTED | **primary** | **NOT_TRANSFERRED_FRESH** |
| RERANK8_VETO | secondary replication | **FRESH_TRANSFER_PASS** |
| ENS_MEAN | H287 ensemble | **FRESH_TRANSFER_PASS** |
| ENS_VOTE2 | H287 ensemble | **FRESH_TRANSFER_PASS** |
| ENS_VOTE3 | H287 ensemble | **FRESH_TRANSFER_PASS** |
| CLIP | H287 ensemble | **FRESH_TRANSFER_PASS** |

Both nulls fail.

**What it means.** On 300 fresh ODD families, scored GPU-exact with r3175 exact in-process (91/91):
- **The primary single-seed BAG arms** pass everything but one part. Seed 279101's lower bound is **−0.004**. Their
  sign tests are p < 1e-4 on every seed, harm is 3–11%, and the median gains are +0.074 and +0.079.
- **The H287 seed ensembles of the same members pass the full tail-aware rule:**
  - **ENS_VOTE2:** +0.0854 [+0.040, +0.138], W/T/L 60/221/19, harm 4.7%;
  - **ENS_MEAN:** +0.0607;
  - **CLIP:** +0.0741;
  - **ENS_VOTE3:** +0.0143, very conservative.

  This supports **the second half of the hypothesis: the deterministic seed ensemble passes where the single seeds do
  not.**
- **H286 + H288 pooled** (500 families; descriptive; H286 portion already seen): **every arm passes** and both nulls
  fail.

**Caveats:**
- For the deterministic ensembles, "3 seeds" means 3 **bootstrap** seeds on one policy. That is weaker than 3
  independent fits.
- **Seven arms** were tested, with no multiplicity correction.
- **RERANK8_VETO passes on ODD** but failed on EVEN in H286. The halves differ in level (slot-0 mean 1.156 here against
  1.339).

## Run record

- **Cohort v3:** published at 00:04Z.
- **Prereg:** published on `origin/v1/next` as `4851f6bc` at 00:51:37Z, before any fresh context was built or scored
  (same-agent review receipt sha256 `057eb800…`). H287's export (commit `213b30ad`, 00:47:46Z) was on `v1/next` first,
  so its four named arms are pre-declared.
- **GPU attempt-001 (00:53:19Z):** refused, NOT_ADMITTED_VRAM. Only 1.23 GB was free: the writer and a foreign Blender
  render (PID 3940093, untouched) held the GPU. No forward pass ran.
- **GPU attempt-002 (00:53:49–00:54:04Z):** admitted through a declared admission wait, then scored, under
  `~/qualification.lock` with a 3 GiB memory cap.
- **Evaluation:** xbabe2 CPU, cores 44–47.
- **Integrity:** 0 neural updates. DOPE `c51a56d0` and the r3175 copy were unchanged. The writer (PID 1402163) was OK
  before and after and was never touched; neither was the shadow observer.

## Hypothesis

On fresh cohort v3 (300 ODD families with non-empty pools, disjoint from H285's 60, H283's 60, H286's 200 and
everything any flywheel round, RERANK fit or selection has seen), frozen BAG_SWITCH beats r3175's slot 0, GPU-exact. A
deterministic seed ensemble from H287 passes where single seeds do not.

## Setup

1. **Cohort v3.** [cohort_v3.py](cohort_v3.py) is H286's rule with ODD, minus H285's 60, first 300: sha256
   `e247983d…`, S3 VersionId `jSdAj.Zn…`, `~/jope_plan/ns/receipts/fresh_cohort_v3_H288.json`.
   - **Receipts:** 692 scanned; 219 had no buildable context and 173 had fewer than 2 accepts.
   - **Result:** 89 `train_fit` and 211 `train_nonpool`, all with 1,152-candidate pools. **300/300 built.**
   - **Reserved:** 649 later ODD families, untouched.
2. **Build and score.** H286's [build_fresh.py](build_fresh.py) and [gpu_rescore.py](gpu_rescore.py) unchanged. The
   r3175 DL-Val91 regression was **exact (91/91, 1.1389572403748425)** in-process before the fresh contexts.
3. **Arms.** All frozen, no refit; every file was fetched by S3 VersionId with sha256 enforced, and each was scored
   with its exporter's own scorer.
   - **BAG_SWITCH and BAG_SWITCH_NESTED (primary):** H284's export.
   - **RERANK8_VETO:** H278's export.
   - **H287:** ENS_MEAN, ENS_VOTE2 and ENS_VOTE3 (deterministic) and CLIP (3 refit seeds), scored with
     `ensemble.score_all_h287_arms`. H287's own per-seed BAG_SWITCH picks matched H284's exactly.
4. **Nulls.** Random valid slot (288101–288103) and random slot with no labels (288201–288203).
5. **Rule.** [analysis.py](analysis.py): per-seed LCB95 > 0 on 3/3 seeds; median > +0.010; **one-sided sign test
   p < 0.05 on 3/3 seeds**; harmed ≤ 35%; invalid not above slot 0; both nulls fail. The bootstrap uses seed 288150
   (288150–288152 on one policy for the deterministic arms), and ties (|d| ≤ 1e-4) are not wins.
6. **Regression.** H281's r3175 panel was reproduced bit-exactly before scoring.

## Outcome (300 fresh ODD families; r3175 slot-0 fresh mean 1.1559, top-8 oracle 1.3648)

| Arm | Family-mean gain by seed | Per-seed CI95 | Median [pooled CI95] | W/T/L by seed | Sign-test p by seed | Elo | Harmed | Invalid (slot 0 / arm) | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| **BAG_SWITCH** (primary) | +0.0524, +0.0741, +0.0790 | **[−0.004, +0.111]**, [+0.026, +0.126], [+0.023, +0.138] | +0.0741 [+0.023, +0.123] | 86/174/40, 58/227/15, 77/193/30 | < 1e-4 × 3 | 1633, 1735, 1664 | 11%, 3.3%, 8% | 10/8, 10/6, 10/6 | NOT_TRANSFERRED_FRESH (LCB, seed 279101) |
| **BAG_SWITCH_NESTED** (primary) | +0.0524, +0.0791, +0.0790 | **[−0.004, +0.111]**, [+0.030, +0.133], [+0.023, +0.138] | +0.0790 [+0.023, +0.128] | 86/174/40, 68/211/21, 77/193/30 | < 1e-4 × 3 | 1633, 1704, 1664 | 11%, 5.3%, 8% | 10/8, 10/6, 10/6 | NOT_TRANSFERRED_FRESH (LCB, seed 279101) |
| RERANK8_VETO | +0.1072, +0.0891, +0.0812 | [+0.050, +0.170], [+0.027, +0.156], [+0.018, +0.149] | **+0.0891 [+0.031, +0.157]** | 104/131/65, 81/167/52, 108/120/72 | 0.0017, 0.0074, 0.0045 | 1582, 1577, 1570 | 16%, 15%, 20% | 10/1, 10/3, 10/5 | **PASS** |
| ENS_MEAN † | +0.0607 | [+0.014, +0.112] (×3) | **+0.0607 [+0.028, +0.094]** | 69/202/29 | < 1e-4 | 1651 | 7% | 10/8 | **PASS** |
| ENS_VOTE2 † | +0.0854 | [+0.040, +0.138] (×3) | **+0.0854 [+0.053, +0.119]** | 60/221/19 | < 1e-4 | 1700 | 4.7% | 10/4 | **PASS** |
| ENS_VOTE3 † | +0.0143 | [+0.002, +0.032] (×3) | **+0.0143 [+0.005, +0.026]** | 22/273/5 | 0.0008 | 1757 | 1% | 10/9 | **PASS** |
| CLIP | +0.0604, +0.0741, +0.0790 | [+0.013, +0.112], [+0.026, +0.126], [+0.023, +0.138] | **+0.0741 [+0.026, +0.122]** | 74/190/36, 58/227/15, 77/193/30 | 0.0002, < 1e-4, < 1e-4 | 1625, 1735, 1664 | 9.7%, 3.3%, 8% | 10/8, 10/6, 10/6 | **PASS** |
| Null: random valid slot | −0.0213, +0.0386, +0.0213 | all include 0 | +0.0213 [−0.064, +0.097] | 122/16/162, 137/16/147, 139/15/146 | 0.99, 0.74, 0.68 | 1451, 1488, 1491 | 44%, 41%, 40% | 10/0 | fails |
| Null: random slot, no labels | −0.0161, −0.0138, −0.0595 | all include 0 | −0.0161 [−0.097, +0.047] | 127/56/117, 114/38/148, 112/46/142 | 0.28, 0.98, 0.97 | 1514, 1455, 1459 | 32%, 41%, 40% | 10/6, 10/6, 10/12 | fails |

† A deterministic single policy: the three "seeds" are bootstrap seeds 288150–288152 on the same picks (W/T/L, sign
test, harm and invalid picks are identical). The per-seed CIs differ only in the third decimal.

**Secondary (cannot flip a verdict):**
- **10%-trimmed family-mean gain:** positive with a CI above 0 for every arm except ENS_VOTE3, which is 0. Examples:
  RERANK8_VETO +0.0157, +0.0114, +0.0149; BAG_SWITCH +0.0124, +0.0049, +0.0120; ENS_MEAN +0.0082; ENS_VOTE2 +0.0052.
- **Normalized gain:** the median is 0 for every arm (237 families kept, 63 dropped), so most families keep slot 0's
  choice.

**Always-harmed families** (harmed on all three seeds; descriptive; disjoint from H286's 18):

| Arm | Always harmed | Of which dedupe-keyed | Source-file groups |
|---|---:|---:|---|
| BAG_SWITCH | 7 | 5 | `qsar-aqu…`, `satellit…` |
| BAG_SWITCH_NESTED | 10 | 8 | `qsar-aqu…`, `satellit…` |
| RERANK8_VETO | 21 | 15 | `rdataset…` (×2), `nasa-phm…`, `rmftsa-c…`, `qsar-aqu…`, `satellit…` |

The `rdataset…` source type recurs from H286.

**POOLED, H286 EVEN v2 + H288 ODD v3** (pre-declared, descriptive; "pooled, H286 portion already seen"; 500 families;
every arm and null re-scored on H286's packets, with H286's VETO panel reproduced exactly):

| Arm | Gain by seed | Seeds with LCB > 0 | Sign test | Verdict |
|---|---|---|---|---|
| RERANK8_VETO | +0.082, +0.069, +0.073 | 3/3 | p ≤ 0.005 | PASS |
| BAG_SWITCH | +0.059, +0.070, +0.062 | 3/3 | p < 1e-4 | PASS |
| BAG_SWITCH_NESTED | median +0.062 | 3/3 | p < 1e-4 | PASS |
| ENS_MEAN | +0.062 | 3/3 | p < 1e-4 | PASS |
| ENS_VOTE2 | +0.067 | 3/3 | p < 1e-4 | PASS |
| ENS_VOTE3 | +0.015 | 3/3 | p < 1e-3 | PASS |
| CLIP | median +0.062 | 3/3 | p ≤ 0.002 | PASS |

Both nulls fail.

## Verdict

- **PRIMARY (BAG_SWITCH, BAG_SWITCH_NESTED): NOT_TRANSFERRED_FRESH.** The near miss is one seed's lower bound of
  −0.004; everything else passes strongly.
- **H287 ENS_MEAN, ENS_VOTE2, ENS_VOTE3 and CLIP: FRESH_TRANSFER_PASS.**
- **RERANK8_VETO (secondary replication): FRESH_TRANSFER_PASS.**

**The ensemble result.** The seed ensemble of the same BAG_SWITCH members removes the single-seed weakness
(seed 279101) and passes the full tail-aware rule on fresh, GPU-exact pools, with both nulls failing. **ENS_VOTE2** has
the best gain/harm balance: +0.085, harm 4.7%, 4 invalid picks against slot 0's 10.

**The limits:**
- the ensembles' "3/3" is three bootstrap seeds on one policy, not three independent fits;
- seven arms were tested;
- the H286 EVEN half (no ensemble arms there) failed for all single-policy arms.

## Next step

1. **Confirm ENS_VOTE2 (and ENS_MEAN)** on the 649 reserved ODD families and/or a fresh EVEN v4. Use a
   **single pre-declared ensemble arm**, to remove arm multiplicity.
2. **Shadow consideration (GOV).** ENS_VOTE2 is the leading candidate for live shadow readouts against new pools. The
   shadow observer needs the H287 contract (topk features, ensemble spec).
3. **Inspect the recurring `rdataset…` harms** before adoption.

## Deviations

- **GPU admission.** attempt-001 was refused NOT_ADMITTED_VRAM before any forward pass, because a foreign Blender
  render (PID 3940093) held 5.8 GB. A declared admission-wait wrapper (`run_gpu_wait.sh`, sha256 `24d670d2…`, in S3
  custody) polled for ≥ 4.5 GiB free plus the lock and MemAvailable. It then ran the **published scorer command
  unchanged** into attempt-002, the first and only scoring. This is not a scientific retry; the refusal record is kept
  in custody.
- **H287 arms included:** only the four the assignment named (ENS_MEAN, ENS_VOTE2, ENS_VOTE3, CLIP), not
  ENS_MEAN_PROPOSER or CLIP_ENS_VOTE2.
- **Trimmed-mean CI:** it uses 2,000 bootstrap draws, as the prereg stated.

## Files and custody

- **In this directory:**
  - [metrics.json](metrics.json): 25 KB, holding every arm's summary, secondary readouts and verdict, the nulls, the
    GPU records, the pooled readout, the overlap and pointers;
  - [PREREG.md](PREREG.md), [cohort_v3.py](cohort_v3.py), [build_fresh.py](build_fresh.py),
    [gpu_rescore.py](gpu_rescore.py), [run_gpu.sh](run_gpu.sh), [evaluate.py](evaluate.py),
    [analysis.py](analysis.py), [h286_always_harmed.json](h286_always_harmed.json), [test_h288.py](test_h288.py) (6
    data-free tests) and [manifest.json](manifest.json).
- **Fresh pools:** `~/jope_plan/ns/receipts/fresh_pools_H288.json` points to `…/H288/fresh_pools/`.
- **S3:**
  - `…/H288/attempt-002/` holds `evaluate.json` (all picks and harmed lists), the panels, the picks, both
    `rescore.json` records, the logs, `run_gpu_wait.sh` and `s3_manifest.json` (VersionId `dDAGoR0q…`);
  - `…/H288/prereg-4851f6bc/` holds the prereg sources;
  - `…/H288/cohort/` holds cohort v3.
