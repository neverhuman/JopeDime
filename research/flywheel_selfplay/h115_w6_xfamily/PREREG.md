# H115 preregistration — W6 cross-family outcome transfer on new families

> **Numbering.** This work was opened as H114 (#1027). H114 was taken on main by #1028 18 s earlier, so it is H115.
> The runner's frozen identifiers keep the original `h114` label: file names `run_h114.py`/`test_run_h114.py`, row tag
> `H114`, and hash salts `h114-panel`, `h114-context`, `h114-stream`, `h114-random`, `h114-shuffle`. Renaming them
> would change the runner sha256 (`ed7b30a0…`) and the panel, and invalidate the dry-run receipt.

Fixed on 2026-10-31, before any H115 panel outcome. `run_h114.py` implements this design. Its frozen sha256, recorded
in every receipt, is authoritative; any difference from this text is a defect.

## Hypothesis (W6)

From [PROPOSALS.md](../rd_breakthrough/PROPOSALS.md#w6--cross-family-outcome-transfer): "conditional preferences
across recurring programs and structural neighbors improve new-family proposal selection beyond the program-only
prior." The test here asks whether outcomes observed on *other* families, conditioned on the new family's dataset
features, choose better new programs than a generic proposal prior and current retrieval.

## Why this design

- [H83](../h83_w6_cross_family_cached/README.md) and [H92](../h92_w6_downside_controlled/README.md) found context
  gains only on cached pools. [H94](../h94_w6_prospective_hedge/README.md) found them reversed on new programs.
- [H110](../h110_w9_scorer_replay_scored/README.md) found a generic proposal-distribution scorer that beats random4
  on held roster families.
- H115 moves to families outside the roster entirely, and tests whether **context conditioning** adds to that
  generic scorer.
- **Disclosure:** the design follows those outcomes.

## Design

| Item | Fixed value |
|---|---|
| Donor memory | All logged native outcomes on the H67 roster: the H106 memory (H94 + H103) plus all H110 fits (`fits.jsonl` sha256 `239b9d26…`), recovered by stream replay and de-duplicated, with conflicting utilities refused. 2,306 rows, 50 contexts. |
| Panel | **38 DL-Train contexts, one per family, all from families outside the roster.** Families are taken in sha256(`h114-panel\|family`) order, and the context in sha256(`h114-context\|context`) order. Earlier dry-run contexts (H96, H106, H112) are excluded. The W1 manifest records 0 cross-family table and dataset aliases, so no donor shares a family or a physical table with the panel ([panel_manifest.json](panel_manifest.json)). |
| Programs | Seed 1001, 16 new proposals per context (stream `h114-stream`, admitted pool excluded). **All 16 are evaluated natively**, so every arm is scored on a fully labelled batch. All picks are fixed before any outcome is read. |
| `X_SCORER` | GBM on `[prog412 − champion, prog412, lm48_z(context)]` → within-context utility percentile (H106 hyperparameters, `random_state` 1131): context-conditioned transfer |
| `G_SCORER` | The same without `lm48_z` (`random_state` 1130): the generic proposal prior (H110 recipe) and the Elo anchor |
| `X_SHUF` | `X_SCORER` trained with every donor context's `lm48_z` replaced by another donor's (sha256-keyed permutation; all 50 moved): a context-shuffle placebo |
| `RETR` | Current raw-lm12 retrieval (H83 rule): −min proj16 distance to the champions of the 4 donor contexts nearest by `lm48_z[:12]` |
| `PROG_PRIOR` | The W1-recipe program-only refit (H94 `models_prog.pkl`, sha256 `5938a354…`) for the context's W1 fold |
| `random4` | Seeded uniform four (`h114-random`) |
| Endpoint | Best utility among the arm's top 4 of 16 (zero history). One context per family, so contexts are families. |
| **Primaries** | `X_SCORER − G_SCORER` and `X_SCORER − RETR`. Bonferroni m = 2: one-sided .025 LCB > +.001, family median ≥ 0 and ≤ 10% of families losing > .01; a 10,000-draw family bootstrap, seed 20261031. |
| **Advance** | Both primaries pass **and** `X_SCORER − X_SHUF` > 0 (family mean) **and**, on the unseen-template panel, `X_SCORER − G_SCORER` ≥ 0 (family mean) |
| Unseen-template panel | Proposals whose template (DSL with numeric literals masked) occurs in no donor row. Every arm picks its top 4 among them; contexts with fewer than 6 such proposals are excluded. If fewer than 10 families qualify, the check is UNIDENTIFIED and the verdict is WITHHELD (W6: "coverage too sparse"). |
| **Kill** | NO_ADVANCE if a primary fails, the shuffle matches, or the benefit disappears on unseen templates (W6 kill rule) |
| Secondary | `G_SCORER − random4` (H110 transfer to new families), `PROG_PRIOR − random4` and `RETR − random4`; an exact-program overlap census; median 95% CIs; W/T/L |
| Elo | Canonical family-grouped Bradley–Terry, `G_SCORER` = 1500, tie 1e-4, prior .01, 400 iterations. No AUC. |
| Completeness | A verdict only with all 38 contexts × 6 arms complete |
| Budget | 608 native fits. The W6 ceiling is 1,200: H94 used 556, this prep's dry run used 16, so the total is 1,180. Deadline 2026-11-01T18:00Z; xbabe2 (x86-64 evaluator), nice 15, load-gated at 85%, 0 GPU. |

## Not included, and why

- **Fit-family program mean.** This W6 arm is covered by H101 (#983, owned by rd_argon).
- **The rank-8 bilinear residual.** H83 and H92 tested it on cached pools, where it was not separable from its
  placebo. Here, context interaction is carried by the GBM with `lm48_z` features.
- **val_reg_inner.** Its contract does not apply to the new-program endpoint.

## Data scope

DL-Train only. No DL-Val, outer-test, sealed or holdout row is read. DOPE stays frozen at `c51a56d0` and is not loaded.
