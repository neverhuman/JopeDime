# H294: fresh EVEN v5 confirmation of H287 ENS_VOTE2

**Status: complete. Verdict: ENS_VOTE2 CONFIRMED_FRESH.** ENS_VOTE2 is now a **shadow-only candidate**. There is **no
serving or promotion change** in this wave.

**The result.** On 358 fresh EVEN families (GPU-exact; r3175 exact in-process at 91/91; the published H292 cell
reproduced bit-exactly first), the single pre-declared primary passes **every part** of the rule, and both nulls fail:

| Part | Result |
|---|---|
| Gain | **+0.0463**, pooled CI **[+0.028, +0.066]** |
| One-sided 95% LCB | **+0.024** on 3/3 bootstrap seeds (two-sided 95% lower bound +0.021, also above 0) |
| Sign test | **p = 3.1×10⁻⁴** (W/T/L 61/269/28) |
| Harm | **6.1%** |
| Elo | 1635 |
| Invalid picks | **3, against slot 0's 7** (McNemar b = 0, c = 4) |

- **Invalid picks:** ENS_VOTE2 makes fewer invalid picks than slot 0, so it passes **both** GOV's non-inferiority
  clause and H292's original strict clause. **The verdict does not depend on the clause change.**
- **The nulls fail** clearly: random valid slot −0.0036; label-free random −0.110.

**Caveats.**
- ENS_VOTE2 is **one deterministic policy**: the "3/3" is three bootstrap seeds on the same picks.
- The **family-median gain is 0**: most families keep slot 0's choice, and the gain comes from the 61 wins against 28
  losses.

## Run record

- **Regression:** the published H292 cell was reproduced bit-exactly at 01:58Z.
- **Cohort v5:** published at 02:00Z.
- **Prereg:** published on `origin/v1/next` as `eb67eb7d` at 02:07:36Z, before any v5 context was built or scored
  (same-agent review receipt sha256 `76c27ab4…`).
- **GPU step:** xbabe1, 02:09:46–02:10:01Z, admitted on the first poll, under `~/qualification.lock` with a 3 GiB cap.
- **Evaluation:** xbabe2 CPU, cores 44–47.
- **Integrity:** 0 neural updates. DOPE `c51a56d0` was unchanged. The writer was OK before and after and was never
  touched; neither was the observer or any foreign job.

## Hypothesis

On a new, disjoint fresh EVEN cohort v5, the frozen H287 ENS_VOTE2 (export `213b30ad`) beats r3175's slot 0, GPU-exact,
under the H292 rule with GOV's non-inferiority invalid-pick clause.

## Setup

1. **Regressions.**
   - [regress_h292.py](regress_h292.py) reproduced H292's published ENS_VOTE2 cell (+0.06647400236769202, CI and W/T/L)
     **bit-exactly** from S3 artifacts fetched by VersionId, before v5.
   - r3175's DL-Val91 regression was **exact (91/91)** in-process.
   - H281's panel was reproduced bit-exactly.
   - The H291 HARMVETO port reproduced H291's published Part B seed gains on H286 **bit-exactly**.
2. **Cohort v5.** [cohort_v5.py](cohort_v5.py) applies H286's qualification rule unchanged to the EVEN half, skipping
   H283's 60 and H286's 480 scanned families: sha256 `ea5364ad…`, S3 VersionId `zfXFbXPD…`.
   - **All 358 remaining qualifying families** were taken (104 `train_fit`, 254 `train_nonpool`); 210 had no buildable
     context and 220 had fewer than 2 accepts.
   - **The EVEN reserve is now exhausted.**
   - Disjointness from H283, H285, H286, H288 and H292 is asserted in code.
   - 358/358 built.
3. **Build and score.** H292's `build_fresh.py` and `gpu_rescore.py` were used **byte-identical**.
4. **Arms.** All frozen and fetched by VersionId with sha256 enforced; each scored with its exporter's scorer.
   - **Primary:** ENS_VOTE2 (single policy; bootstrap seeds 294150–294152).
   - **Descriptive:**
     - ENS_VOTE2+HARMVETO (H291 fit lock `d8bc373b80`; locked taus 0.135, 0.165 and 0.135);
     - CLIP;
     - BAG_SWITCH.
   - **Nulls:** random valid slot (294101–294103) and label-free random slot (294201–294203).
5. **Rule.** [analysis.py](analysis.py), at one-sided α = 0.05:
   - the one-sided 95% LCB is > 0 on 3/3 seeds;
   - the median is > +0.010;
   - the sign test is p < 0.05 on 3/3 seeds;
   - harmed ≤ 35%;
   - **invalid non-inferiority:** the excess is ≤ max(2, ⌈0.01 n⌉) = 4 and the McNemar p is ≥ 0.05;
   - both nulls fail.

## Outcome (358 fresh EVEN families; r3175 slot-0 fresh mean 1.4039, top-8 oracle 1.6036)

| Arm | Gain by seed | Median [pooled CI95] | One-sided 95% LCB | W/T/L | Sign p | Elo | Harmed | Invalid (slot 0 / arm) | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| **ENS_VOTE2** (primary) † | +0.0463 | **+0.0463 [+0.028, +0.066]** | +0.024 (×3) | 61/269/28 | 3.1e-4 | 1635 | 6.1% | **7 / 3** | **CONFIRMED_FRESH** |
| ENS_VOTE2+HARMVETO | +0.0074, +0.0256, +0.0268 | +0.0256 [+0.007, +0.042] | +0.004, +0.010, +0.009 | 20/337/1, 35/312/11, 23/329/6 | ≤ 0.0012 | 2019, 1701, 1733 | 0%, 2.5%, 1.1% | 7/7, 7/5, 7/4 | passes (descriptive) |
| CLIP | +0.0494, +0.0538, +0.0542 | +0.0538 [+0.018, +0.089] | +0.018, +0.031, +0.023 | 79/233/46, 61/272/25, 80/243/35 | ≤ 0.002 | 1594, 1655, 1644 | 11%, 5.9%, 7% | 7/3 | passes (descriptive) |
| BAG_SWITCH | +0.0579, +0.0538, +0.0542 | +0.0542 [+0.027, +0.089] | +0.029, +0.031, +0.023 | 84/219/55, 61/272/25, 80/243/35 | ≤ 0.009 | 1574, 1655, 1644 | 13%, 5.9%, 7% | 7/2–3 | passes (descriptive) |
| Null: random valid slot | −0.023, −0.004, +0.001 | −0.004 [−0.056, +0.049] | < 0 | 161/15/182, … | ≥ 0.65 | ≈ 1480 | 43–45% | 7/0 | fails |
| Null: label-free random | −0.050, −0.120, −0.110 | −0.110 [−0.172, −0.027] | < 0 | 147/50/161, … | ≥ 0.80 | ≈ 1470 | 36–40% | 7/5, 7/17, 7/14 | fails (also on invalid NI) |

† A deterministic single policy: the three "seeds" are bootstrap seeds on the same picks.

- **Family-median gain is 0** for every non-null arm.
- **ENS_VOTE2+HARMVETO** trades gain for safety: it has the lowest harm (0–2.5%) and Elo up to 2019, but about half
  ENS_VOTE2's gain.
- **H291's vote rule** produces exactly H287's ENS_VOTE2 picks.

**Always-harmed families** (harmed on all three seeds; descriptive):
- **ENS_VOTE2:** 22 (its harmed set, since it is a single policy): 18 dedupe-keyed, **2 `srcfile:rdataset*`**,
  `uci-597`, and one legacy key.
- **CLIP / BAG_SWITCH:** 13 each, including 2 `rdataset*`.
- **ENS_VOTE2+HARMVETO:** 0.

The `srcfile:rdataset*` group recurs, as in H286, H288 and H292.

**POOLED, H288 v3 + H292 v4 + H294 v5** (descriptive only; "pooled, H288 and H292 portions already seen"; 953
families). **Every arm passes the rule** and both nulls fail:

| Arm | Gain | Pooled CI | One-sided LCB | Harm | Invalid (slot 0 / arm) |
|---|---|---|---|---|---|
| ENS_VOTE2 | +0.0649 | [+0.046, +0.086] | +0.041 | 5.5% | 23 / 14 |
| CLIP | +0.051 to +0.076 | | | | |
| BAG_SWITCH | +0.051 to +0.076 | | | | |
| ENS_VOTE2+HARMVETO | +0.017 to +0.030 | | | 0.7–1.9% | |

## Verdict

**ENS_VOTE2: CONFIRMED_FRESH.** It is a **shadow-only candidate**; there is no serving or promotion change.

**The evidence across waves.** On genuinely fresh, GPU-exact pools, the frozen H287 ENS_VOTE2 has now:
- passed H288 (ODD v3, 300 families);
- nearly passed H292 (ODD v4, 295 families; one invalid pick over the strict limit);
- **passed this pre-declared single-primary confirmation** on a disjoint EVEN cohort (358 families). Here it makes
  fewer invalid picks than r3175's own choice, so the result holds under either invalid-pick clause.

## Next step

1. **Shadow (flywheel owner / GOV).** Wire ENS_VOTE2 into the shadow observer through the H287 contract (the ensemble
   spec and the 60 member models by VersionId, plus H278 `topk` features). Collect live readouts on new pools; repeated
   r3175 rounds give byte-identical packets and no new information.
2. **Consider ENS_VOTE2+HARMVETO as the safety variant.** It has the lowest harm and zero always-harmed families, at
   about half the gain. It deserves a pre-declared head-to-head against ENS_VOTE2 in the shadow.
3. **Inspect the recurring `srcfile:rdataset*` harms.**
4. **Both fresh reserves are now exhausted** (ODD after H292, EVEN after H294). A further fresh test needs new catalog
   data or a new family-level split.

## Deviations

- **Lock path:** as declared in the prereg, the GPU step used `~/qualification.lock`; the writer's
  `~/jope_plan/qualification.lock` was never touched.
- **`run_gpu.sh`** differs from H292's only in its data path and its 05:00Z stop time.
- **Cohort size:** 358 families, more than the assignment's "about 150–200" expectation; all qualifying were taken, as
  instructed.

## Files and custody

- **In this directory:**
  - [metrics.json](metrics.json): 27 KB, holding every arm's summary, criteria (including the invalid NI detail) and
    verdict, the regressions, the GPU record, the pooled readout and pointers;
  - [PREREG.md](PREREG.md), [regress_h292.py](regress_h292.py), [cohort_v5.py](cohort_v5.py),
    [build_fresh.py](build_fresh.py), [gpu_rescore.py](gpu_rescore.py), [run_gpu.sh](run_gpu.sh),
    [evaluate.py](evaluate.py), [analysis.py](analysis.py), [test_h294.py](test_h294.py) (5 data-free tests) and
    [manifest.json](manifest.json).
- **Fresh pools:** `~/jope_plan/ns/receipts/fresh_pools_H294.json` points to `…/H294/fresh_pools/`.
- **S3:**
  - `…/H294/attempt-001/` holds `evaluate.json`, the panels, the picks, `rescore.json`, the log and
    `s3_manifest.json` (VersionId `9eSpZuyM…`);
  - `…/H294/prereg-eb67eb7d/` holds the sources, the regression record and cohort v5.
