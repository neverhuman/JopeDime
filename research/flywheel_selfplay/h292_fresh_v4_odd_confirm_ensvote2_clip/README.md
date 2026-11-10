# H292: one clean confirmation of H287 ENS_VOTE2 and CLIP on fresh cohort v4

**Status: complete. Verdicts: ENS_VOTE2 NOT_CONFIRMED_FRESH; CLIP NOT_CONFIRMED_FRESH** (Holm, 2 pre-declared
primaries). There is no serving or promotion change, and no shadow candidate from this wave.

**The result.** On 295 fresh ODD families (GPU-exact; r3175 exact in-process at 91/91; the published H288 cell
reproduced bit-exactly first):
- **ENS_VOTE2 was tested first** (arm p 0.0135, Holm level 0.025). It **passes every statistical part**:
  - Holm-level one-sided LCB **+0.0057, +0.0060 and +0.0058** on all 3 seeds;
  - gain **+0.0665**, pooled CI [+0.021, +0.122];
  - sign test p = 3×10⁻⁵ (W/T/L 50/228/17), harm 5.4%, Elo 1687.

  It **fails only part 5: it makes 7 invalid (−2.0) picks against slot 0's 6.** The pre-declared rule says no
  more than slot 0, so ENS_VOTE2 is **NOT_CONFIRMED_FRESH by one invalid pick**.
- **CLIP:** Holm stops, so CLIP is not confirmed. It would also have failed on its own:
  - its seed 2 is **bit-identical to BAG_SWITCH**, with gain +0.0236 and LCB −0.014 at 0.05;
  - that seed also makes 7 invalid picks against 6.

  Its refit seeds 1 and 3 are strong (+0.102 and +0.100).
- **Both nulls fail**, the label-free null clearly so (−0.116).

This is the red-team's CLIP concern made concrete: **CLIP's support rests on its refit seed**.

## Run record

- **Regression:** the published H288 cell was reproduced bit-exactly at 01:27Z.
- **Cohort v4:** published at 01:29Z.
- **Prereg:** published on `origin/v1/next` as `330f770c` at 01:35:44Z, before any v4 context was built or scored
  (same-agent review receipt sha256 `4712434f…`).
- **GPU step:** xbabe1, 01:37:11–01:37:26Z, admitted on the first poll, under `~/qualification.lock` with a 3 GiB cap.
- **Evaluation:** xbabe2 CPU, cores 44–47.
- **Integrity:** 0 neural updates. DOPE `c51a56d0` was unchanged. The writer was OK before and after and was never
  touched; neither was the observer or the foreign render.

## Hypothesis

On fresh cohort v4, the frozen H287 arms ENS_VOTE2 and CLIP (the only two primaries, Holm-corrected) beat r3175's slot
0, GPU-exact, with LCB > 0 and a sign test on every seed, while both nulls fail.

## Setup

1. **Regressions.**
   - [regress_h288.py](regress_h288.py) reproduced H288's ENS_VOTE2 cell (bootstrap 288150: +0.08537299387889595,
     CI and W/T/L) **bit-exactly** from S3-published artifacts fetched by VersionId, before v4.
   - r3175's DL-Val91 regression was **exact (91/91, 1.1389572403748425)** in-process.
   - H281's panel was reproduced bit-exactly.
2. **Cohort v4.** [cohort_v4.py](cohort_v4.py) applies H288's rule unchanged to its 649-family reserve: sha256
   `7f7ff09d…`, S3 VersionId `Ie7wT7_g…`.
   - **295 = every qualifying reserved family** (174 had no buildable context and 180 had fewer than 2 accepts).
   - **The ODD reserve is now exhausted.**
   - Disjointness from H283, H285, H286 and H288 is asserted in code.
   - 295/295 built.
3. **Build and score.** H288's `build_fresh.py` and `gpu_rescore.py` were used **byte-identical**.
   [run_gpu.sh](run_gpu.sh) includes the declared admission wait, which was not needed.
4. **Arms.** All frozen and fetched by VersionId with sha256 enforced; each scored with its exporter's scorer.
   - **Primaries:** ENS_VOTE2 (single policy; bootstrap seeds 292150–292152) and CLIP (3 refit seeds).
   - **Descriptive:** BAG_SWITCH, ENS_MEAN, ENS_VOTE3 and RERANK8_VETO.
   - **Nulls:** random valid slot (292101–292103) and label-free random slot (292201–292203).
5. **Rule.** [analysis.py](analysis.py): Holm at one-sided α = 0.05.
   - **Arm p:** the maximum over seeds of max(bootstrap p, sign p).
   - **Levels:** the first arm is tested at 0.025, the second at 0.05, stopping at the first failure.
   - **Requirements:** LCB > 0 on 3/3 seeds, median > +0.010, sign test 3/3, harmed ≤ 35%, invalid not above slot 0,
     and both nulls fail at 0.05.
   - **Ties** (|d| ≤ 1e-4) are never wins.

## Outcome (295 fresh ODD families; r3175 slot-0 fresh mean 1.5703, top-8 oracle 1.8465)

**Holm.** The order was ENS_VOTE2 (arm p 0.0135), then CLIP (arm p 0.1448).

| Arm | Holm level | LCB at level | Median gain [pooled CI] | W/T/L | Sign p | Harmed | Invalid (slot 0 / arm) | Failed parts | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| **ENS_VOTE2** † | 0.025 | **+0.0057, +0.0060, +0.0058** | **+0.0665 [+0.021, +0.122]** | 50/228/17 | 3e-5 | 5.4% | **6 / 7** | invalid only | **NOT_CONFIRMED_FRESH** |
| **CLIP** | 0.05 (not tested: Holm stopped) | +0.036, **−0.014**, +0.038 | +0.0995 [+0.023, +0.187] | 69/198/28, 42/238/15, 68/204/23 | 2e-5, 2e-4, < 1e-5 | 7.8%, 4.7%, 6.8% | 6/4, **6/7**, 6/5 | LCB, invalid (seed 2) | **NOT_CONFIRMED_FRESH** |

† A deterministic single policy; the three "seeds" are bootstrap seeds on the same picks.

**Descriptive arms** (unadjusted; cannot flip a verdict):

| Arm | Gain by seed | Median [pooled CI] | W/T/L | Harmed | Invalid (slot 0 / arm) |
|---|---|---|---|---|---|
| ENS_MEAN † | +0.0695 | +0.0695 [+0.020, +0.127] | 55/219/21 | 6.4% | 6/6 |
| ENS_VOTE3 † | +0.0110 | +0.011 [−0.006, +0.028] | 20/269/6 | 2% | 6/7 |
| BAG_SWITCH | +0.0732, +0.0236, +0.0995 | +0.073 [+0.005, +0.164] | 70/193/32, 42/238/15, 68/204/23 | 9.5%, 4.7%, 6.8% | 6/6, 6/7, 6/5 |
| RERANK8_VETO | +0.1150, +0.0733, +0.0853 | +0.085 [+0.010, +0.183] | 111/133/51, 81/175/39, 108/133/54 | 15%, 11%, 17% | 6/3, 6/4, 6/3 |
| Null: random valid slot | −0.026, +0.047, −0.040 | −0.026 [−0.100, +0.115] | | 38–43% | 6/0 |
| Null: label-free random | −0.116, −0.100, −0.138 | −0.116 [−0.217, −0.027] | | 36–38% | up to 6/13 |

- **Family-median gain is 0** for every non-null arm on every seed: most families keep slot 0's choice.
- **CLIP's seeds 2–3 equal BAG_SWITCH's picks** (`[false, true, true]`). Seed 2 is the weak one on v4.

**Always-harmed families** (harmed on all three seeds), by source group:

| Arm | Always harmed | Of which dedupe-keyed | Source-file groups | `srcfile:rdataset*` |
|---|---:|---:|---|---:|
| ENS_VOTE2 | 16 | 12 | `nasa-phm`, `rdataset`, `uci-206`, `us-crime` | 1 |
| CLIP | 11 | 7 | `nasa-phm`, `rdataset`, `uci-206`, `us-crime` | 1 |
| RERANK8_VETO | 18 | 14 | `airfoil`, `nasa-phm`, `rdataset` (×2) | 2 |

- **`srcfile:rdataset*`** recurs in every arm (1–2 families), as it did in H286 and H288.

**POOLED, H288 v3 + H292 v4** (descriptive only; "pooled, H288 portion already seen"; 595 ODD families; same Holm
procedure). **Both primaries would be CONFIRMED:**
- **ENS_VOTE2:** +0.076, pooled CI [+0.047, +0.108], LCB at 0.025 +0.035, invalid 11 against slot 0's 16, harm 5%;
- **CLIP:** +0.081, +0.049 and +0.089 by seed; LCB at 0.025 +0.035, +0.016 and +0.043.

Both nulls fail.

## Verdict

**ENS_VOTE2: NOT_CONFIRMED_FRESH. CLIP: NOT_CONFIRMED_FRESH.** There is no shadow candidate from this wave.
- **ENS_VOTE2's statistical evidence on fresh v4 is strong:** the Holm 97.5% one-sided LCB is above 0 on every seed,
  the sign test gives p = 3×10⁻⁵, harm is 5.4% and both nulls fail. **But it made 7 invalid picks against slot 0's 6,
  and the pre-declared rule forbids even one extra.** Across H288 + H292 it makes fewer invalid picks than slot 0
  (11 against 16).
- **CLIP is not confirmed.** Its non-refit seed 2, identical to BAG_SWITCH, is weak on v4. This bears out the red-team
  point (b).

## Next step

1. **A confirmation decision for GOV.** ENS_VOTE2 now has one powered fresh pass (H288) and one fresh near-pass (H292),
   whose only failure is the invalid-pick count (+1 against slot 0). Over all 595 ODD families it makes fewer invalid
   picks than slot 0. The choice is either:
   - a pre-declared invalid-pick rule with tolerance (for example a rate test, or "not above slot 0 + 1%"), tested on a
     fresh **EVEN v4**, which needs a new cohort since the ODD reserve is exhausted; or
   - accepting the shadow-only stage with invalid picks monitored live.
2. **Drop CLIP as a primary.** Its seeds 2–3 duplicate BAG_SWITCH, so it adds no independent evidence.
3. **`srcfile:rdataset*`** families recur among the always-harmed in every wave. Inspect them before any adoption.

## Deviations

- **Lock path.** The assignment names `~/jope_plan/qualification.lock`, but `lslocks` shows it **held by the live writer
  for its lifetime**. As in H281–H288, the GPU step used `~/qualification.lock` (declared in the prereg); the writer's
  lock was never touched.
- **Cohort size.** It is **295, not 300**: the reserve held only 295 qualifying families, all of which were taken.
- **Comms.** The assignment's `~/nudges/comms.md` line was written at claim and close.

## Files and custody

- **In this directory:**
  - [metrics.json](metrics.json): 26 KB, holding the Holm procedure, every arm's summary and criteria, always-harmed
    counts, regressions, GPU record, the pooled readout and pointers;
  - [PREREG.md](PREREG.md), [regress_h288.py](regress_h288.py), [cohort_v4.py](cohort_v4.py),
    [build_fresh.py](build_fresh.py), [gpu_rescore.py](gpu_rescore.py), [run_gpu.sh](run_gpu.sh),
    [evaluate.py](evaluate.py), [analysis.py](analysis.py), [test_h292.py](test_h292.py) (6 data-free tests) and
    [manifest.json](manifest.json).
- **Fresh pools:** `~/jope_plan/ns/receipts/fresh_pools_H292.json` points to `…/H292/fresh_pools/`.
- **S3:**
  - `…/H292/attempt-001/` holds `evaluate.json` (all picks and harmed lists), the panels, the picks, `rescore.json`,
    the log and `s3_manifest.json` (VersionId `kqzWyNCi…`);
  - `…/H292/prereg-330f770c/` holds the sources, the regression record and cohort v4.
