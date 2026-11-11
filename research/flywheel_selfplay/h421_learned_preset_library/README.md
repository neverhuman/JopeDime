# H421 — data-selected preset library: `PORT_NULL` on DEV (closeout)

**Verdict: `PORT_NULL`.** On the 115 fresh `DEV` families, adding the 20 data-selected presets (`TPORT`) to the H420
recipe (`B*` = H420 `TBOTH`) does not pass the pre-registered test: clipped family-mean gain +0.0047 u (about 0.5%
lower RMSE), sign-flip p 0.234. **Consequence (PREREG §5): the library stays as H420 left it.** The estimate is
positive in every comparison but far below what this test can detect (about 2% RMSE).
TRAIN-side cohorts only: no dev91, DL-Val, DL-Holdout or sealed data was read.

Plan: [`PREREG.md`](PREREG.md), addenda [1](PREREG_ADDENDUM_1.md), [2](PREREG_ADDENDUM_2.md). Numbers:
[`metrics.json`](metrics.json). Selection: [`selection.json`](selection.json), [`LIB_PORT.json`](LIB_PORT.json),
[`LIB_RAND.json`](LIB_RAND.json), [`LIB_COV.json`](LIB_COV.json). Gates: [`gates_h421.sh`](gates_h421.sh).

## Result (DEV, 115 families, entrant `P3+LIB3.SXB`, value = AUDIT u, paired by family)
| Comparison | Mean gain (u) | Clipped mean | Sign-flip p | Leave-one-family-out min | Median | Win / tie / loss | ΔElo [95% interval] | Test |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **`TPORT` vs `B*` (primary)** | +0.0045 | **+0.0047** | **0.234** | +0.0017 | 0.0000 | 49 / 32 / 34 | +37.9 [−12.9, +88.8] | **fail** |
| `TPORT` vs `TRAND` (control) | +0.0038 | +0.0040 | 0.264 | +0.0010 | 0.0000 | 45 / 31 / 39 | +28.0 [−23.5, +81.0] | — |
| `TRAND` vs `B*` (reported) | +0.0007 | +0.0007 | 0.049 | +0.0004 | 0.0000 | 25 / 65 / 25 | +9.8 [−25.8, +45.7] | — |
| `TCOV` vs `B*` (reported) | +0.0032 | +0.0032 | 0.056 | +0.0017 | 0.0000 | 37 / 43 / 35 | +3.8 [−44.2, +50.2] | — |
| `TCOV` vs `TPORT` (reported) | −0.0013 | −0.0015 | 0.615 | −0.0059 | 0.0000 | 33 / 37 / 45 | −34.1 [−83.3, +12.2] | — |

| Variant | Mean u, SXB | Mean u, KFB | Contexts with an added preset in the stack | Presets used somewhere |
| --- | --- | --- | --- | --- |
| `B*` (stored H420 `TBOTH`, not recomputed) | 1.9483 | 1.9677 | — | — |
| `TPORT` (20 selected by stack gain) | 1.9528 | 1.9736 | 74 of 115 | 20 of 20 |
| `TRAND` (20 random harvested candidates) | 1.9490 | 1.9680 | 55 of 115 | 19 of 20 |
| `TCOV` (20 selected by best single column) | 1.9515 | 1.9706 | 76 of 115 | 20 of 20 |

- The test needs a clipped gain above 0 with p below 0.025; p is 0.234.
- **KFB entrant (reported):** `TPORT` vs `B*` clipped +0.0059, p 0.066, 41 / 51 / 23; also not a pass.
- No output was invalid in any variant. The single-run stock value is identical across variants (maximum difference
  0), so the variants differ only by the 20 presets.

## What the result does and does not say
- **The selected presets are used but add little.** They take stack weight in 74 of 115 families and all 20 are used
  somewhere, yet the held-out error barely moves.
- **The TRAIN-side estimate was about three times too high.** On held-out design families the selection beat the
  random set by +0.0127 u (clipped); on `DEV` the same comparison is +0.0040 u. The selection measured gains on
  SEARCH rows, the rows the stack is fitted on; the test measures AUDIT rows. Other differences that may
  contribute, not separated here: 2-fold fits in the fill against 5-fold fits in the deploy, and design candidate
  sets that came from older deploys without the extended search run.
- **The library lever is flattening.** The first 20 added presets (H420, hand-written) gave +0.028 u; the next 20
  (data-selected) give about +0.005 u. Random harvested presets give +0.0007 u.
- **Not shown:** that data selection is useless. `TPORT` is ahead of `TRAND` in direction (+0.0040 u, p 0.26), and
  this test cannot tell a true +0.5% from zero. It is also not shown that a larger learned library would fail;
  only K = 20 was tested.
- **Selection by stack gain against selection by single-column quality:** no measurable difference (`TCOV` vs
  `TPORT` −0.0015 u, p 0.62).

## Integrity gates (`gates_h421.sh`, run before the readout; all 8 pass)
| Gate | Evidence |
| --- | --- |
| Records and units | 115 result records per variant, 0 errors, 0 invalid SXB outputs, 115 units in 32 status files each |
| Guard and bank | guard installed in every process, 0 refusals, 0 same-family bank hits (deploys and fill) |
| Self-check and replay | stock stream equal in every process; replay check before the first variant: 0 worker runs |
| Design contexts only | 0 of the 291 design contexts are `DEV` contexts; the fill ran `search` evaluations only; one pool hash |
| TRAIN-side gate | passed (addendum 2 statistic): clipped +0.0127 u, 114 / 119 / 58, p 0.00001 |
| Presets published before DEV | preset files of commit `c8a3726472` (01:30:17Z) equal the files each variant ran; first DEV config created 02:04Z |
| Baseline unchanged | the H420 `TBOTH` records in the run directory still match H420's pulled manifest |
| Published stage, one config | stage `pub-q2` hash intact, PREREG equal to the published one, one manifest and one code set across variants |

## Cost
| Step | Used | Cap |
| --- | --- | --- |
| Fill (291 design contexts × 142 candidates) | 14.6 worker-hours, 23,107 worker runs | 60 worker-hours |
| Selection and gate | 5.6 core-hours | 4 core-hours (**exceeded**, see below) |
| DEV deploys, 3 variants | 3.65 worker-hours of new evaluations (9,256 worker runs); 4.0 hours on 64 CPUs | 15 worker-hours; 6 hours |

## Deviations and weaknesses
1. **Hosts (addendum 1).** Fill and DEV deploys ran on xbabe2 instead of xbabe3 / xbabe1 (disk on xbabe3).
2. **First selection void; gate statistic changed afterwards (addendum 2).** The first selection run had a
   numerical error and is void. The corrected selection uses a clipped objective and a clipped gate statistic,
   decided after the void run had shown a pass. The DEV test and its thresholds were not changed.
3. **Selection objective differs from PREREG §3.3 (addendum 1).** It models the deploy's candidate-set rule; published
   before the selection ran.
4. **Selection exceeded its compute cap:** 5.6 core-hours against 4 (numerical-library threads were not limited).
5. **The gate and readout scripts were published at 02:08Z, four minutes after the first DEV variant started.**
   No DEV record of this wave existed in readable form before then, and none was read before the gates ran.
6. `TRAND` logged 91 resource pauses (other load on the host); no timeout and no failed evaluation in any variant.
7. The test is under-powered for an effect of this size, as PREREG §5 said in advance.

## Timeline (UTC, 2026-11-11)
| Time | Event |
| --- | --- |
| 00:22 | go-ahead relayed by the owner's watcher (xbabe2, bounded CPU, host change recorded first) |
| 00:47 | addendum 1 and code published (`492412f3da`); harvest: 291 design contexts, pool of 142 |
| 00:57–01:19 | fill |
| 01:21 | first selection: void |
| 01:25 | addendum 2 published (`5e9c807105`) |
| 01:28 | corrected selection; TRAIN-side gate passes |
| 01:30 | selection and preset files published (`c8a3726472`) |
| 01:31–02:04 | code smoke on 2 design tables; replay check on DEV (0 worker runs) |
| 02:04–06:04 | `TPORT`, `TRAND`, `TCOV` |
| 02:08 | gate and readout scripts published (`a8b600cdd3`) |
| 06:06 | pulled copy verified against the source, gates (8 of 8), readout |

Evidence (xbabe2): `~/jope_plan/evidence/H421/` — `pulled/H421_DEV` (2,272 files, manifest sha256 `5229ca0d5906201f…`),
`pulled/H421_FILL` (302 files, `fef21353a3dee5b2…`), `results/H421_DEV.gates.json`, `results/H421_DEV.readout.json`,
`results/selection2`, `results/selection` (void), `logs/`.
