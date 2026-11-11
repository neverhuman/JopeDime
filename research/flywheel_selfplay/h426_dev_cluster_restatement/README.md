# H426 — the DEV tests of H420–H425 with clusters as the unit: no conclusion flips (restatement; closeout)

**Result.** Under the pre-declared clustering the 115 `DEV` families become 110 units, and **no comparison changes
its pass / fail**. P-values move little. **No verdict is changed by this wave.**
**One thing to watch:** H420's primary (`TBOTH` vs `B`) is borderline. Its p rises from 0.012 to 0.020 under the
pre-declared clustering and to 0.026 (above 0.025) under the stricter `copy` clustering. H420's conclusion that the
library presets carry the gain does not depend on it: `TLIB` vs `B` passes under every clustering.
Stored values only: no table, model, GPU or GP worker; `CONFIRM` and dev91 untouched.

Plan: [`PREREG.md`](PREREG.md). Numbers: [`metrics.json`](metrics.json). Gates: [`gates_h426.sh`](gates_h426.sh).

## Units
| Clustering | Units | Clusters larger than one |
| --- | --- | --- |
| original (families) | 115 | — |
| **`feature` (pre-declared)** | **110** | 3, 2, 2, 2 |
| `copy` | 112 | 3, 2 |
| `any` (with target-only matches) | 104 | 8, 3, 2, 2 |
| `shape` (crude, pessimistic) | 95 | 8, 5, 4, 3, 2, 2, 2, 2 |

## Original against cluster level (clipped unit-mean gain in u; one-sided sign-flip p; test = the waves' own)
| Wave | Comparison | Original: gain, p, test | `feature` clusters: gain, p, test | Flip | p under `copy` / `any` / `shape` |
| --- | --- | --- | --- | --- | --- |
| H420 | `TBOTH` vs `B` (primary) | +0.0252, 0.0121, pass | +0.0223, 0.0200, pass | no | **0.0257** / 0.0241 / 0.0246 |
| H420 | `TLIB` vs `B` | +0.0282, < 0.0001, pass | +0.0245, < 0.0001, pass | no | < 0.0001 in all |
| H420 | `TGX` vs `B` | +0.0039, 0.36, fail | +0.0033, 0.38, fail | no | 0.41 / 0.42 / 0.41 |
| H421 | `TPORT` vs `B*` (primary) | +0.0047, 0.23, fail | +0.0043, 0.26, fail | no | 0.24 / 0.27 / 0.31 |
| H421 | `TPORT` vs `TRAND` (control) | +0.0040, 0.26, fail | +0.0036, 0.29, fail | no | 0.27 / 0.30 / 0.34 |
| H422 | `S` vs `B` (primary) | +0.0045, 0.24, fail | +0.0016, 0.35, fail | no | 0.25 / 0.40 / 0.44 |
| H422 | `A` vs `B` (reported) | +0.0065, 0.10, fail | +0.0061, 0.13, fail | no | 0.12 / 0.17 / 0.23 |
| H422 | `SA` vs `B` (reported) | +0.0191, 0.012, pass | +0.0162, 0.020, pass | no | 0.015 / **0.031** / **0.047** |
| H424 | `I` vs `B` (primary) | +0.0092, 0.00006, pass | +0.0096, 0.00004, pass | no | 0.00005 / 0.00007 / 0.0001 |
| H425 | `I` vs `B` without feature flags | +0.0107, 0.0002, pass (94) | +0.0107, 0.0002, pass (94) | no | 0.0002 / 0.0003 / 0.0005 |

- **Flips under the pre-declared clustering: none.** H422's three-part rule fails both ways.
- **Flips under other clusterings:** H420 `TBOTH` vs `B` under `copy` (p 0.0257); H422 `SA` vs `B` under `any` and
  `shape` (p 0.031, 0.047).
- Two-arm ΔElo with a cluster bootstrap (`feature`): H420 `TBOTH` +36 [−27, +101]; H420 `TLIB` +92 [+33, +155];
  H422 `SA` +102 [+39, +170]; H424 `I` +126 [+72, +185]. (Two-arm values; the waves reported multi-arm ones.)

## What the result does and does not say
- **The in-context column (H424) is the most robust result of the set:** p stays near 0.0001 in every clustering,
  including the pessimistic one with 95 units.
- **The library presets (H420 `TLIB`) are equally robust.** H420's combined arm, which was its primary, is not:
  it sits at the threshold. H420's README already said the Elo interval of that arm included 0.
- **H422's lead (wider admission + combiner) is weaker than it looked:** its gain falls from +0.019 to +0.016 u and
  its p from 0.012 to between 0.015 and 0.047. It was already not a verdict.
- **H422's stacker alone loses most of its estimate** (+0.0045 → +0.0016 u): part of it sat in families that the
  clustering merges into one unit.
- **Not shown:** that 110 is the true number of independent sources. The stored lists link `DEV` contexts only
  through a shared `FIT-ICL` match; copies among `DEV` contexts with no such match stay separate. The `shape`
  clustering (95 units) is a crude bound from the manifest, not a measurement.

## Gates (6 of 6 pass)
Every stored original reproduced exactly (largest difference 0); the screen file is the one the frozen list was cut
from; PREREG (`cc83f15fe4`) older than the output; the H425 list equals the published one and the pulled record
copies match their manifests; the sandbox view still holds no `CONFIRM` table; 75 seconds on 8 threads.

## Deviations and limits
1. The gate script ran after the restatement (it reads its output); the restatement's own checks ran before any
   number was written.
2. H423 (P201b) is not restated: no frozen list exists for P201b.
3. H424's second read was not available when this ran; it is not in the table.
4. ΔElo here is two-arm and differs slightly from the multi-arm values in the waves' READMEs.

## Queued follow-ups
1. "Screen `DEV` against itself and against `FIT-SEL` with the H425 code (tables in the sandbox, features only) to
   replace the `shape` stand-in by measured clusters; then rerun this restatement."
2. "Use clusters as the unit in every future `DEV` or `CONFIRM` test, pre-registered, with the clustering frozen
   before the readout."

Evidence (xbabe2): `~/jope_plan/evidence/H426/results/H426_DEV.restatement.{json,txt}`, `H426.gates.json`.
