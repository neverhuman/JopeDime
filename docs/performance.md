# Regression performance: current evidence and reproduction

Reviewed on 2026-10-23. Rankings below retain their original data, opponents and
exposure. This audit changes presentation and validation; it performs no new
model fitting, inference, training or sealed-label evaluation.

## Latest complete comparison

The October 22 four-entrant result is the newest complete regression Elo panel
committed to this repository. It covers **64 families**, three folds and three
seeds (41, 73, 109): 576 fits per entrant, 2,304 fits altogether. All six pairs
have complete coverage. The historical three entrants' 1,728 fits are reused;
Gemini contributed 576 new fits. Reuse is not a new experiment.

[RESULTS.json](../assets/automl-elo/v1/RESULTS.json) is the numerical authority;
[SOURCES.md](../assets/automl-elo/v1/SOURCES.md) describes the implementation,
custody and actual resources. The paper's
[exposure summary](paper/innovationzero/evidence/elo-summary.json) binds those
exact result bytes and records limitations that the short source description
alone does not make clear:

- These are existing training families with prepared inner-CV contexts, not new
  independent validation families. Stored preprocessing was fitted before the
  original split. Scoring held-out rows inside that preparation does not repair
  the earlier exposure.
- Seed 41 had been inspected before successor seeds 73 and 109 were fixed.
- Historical entrants and Gemini ran on different hosts. Fit times describe those
  runs; they are not a controlled hardware speed comparison.
- Ratings are fitted jointly against this opponent pool. A higher numeric Elo on
  another ladder is not evidence of improvement.

**Opus5p5_Extra has the highest Elo. searchT6p1_Sol_Max has the lowest
geometric-mean RMSE point estimate.** The pairwise ratio intervals do not establish
Sol's error advantage over Opus or Gemini. All three Gemini RMSE comparisons fail
the Holm-adjusted significance threshold. The README shows the complete table
and intervals, rather than a favorable selected subset or seed.

The model names identify LLM-authored regression algorithms. The benchmark fits
those algorithms; it does not call the named language models for each dataset.
It also does not evaluate the separately trained neural proposal policy. An Elo
entry does not guarantee a shipped CLI selector: the product native registry
contains Sol and Gemini; the same-cut benchmark also has a separate Opus entrant
adapter. Preserve the distinct source/registration boundaries.

## Plots checked

The README now leads with the existing four-entrant Elo and pairwise figures.
The table, Elo intervals, six pairwise columns, complete 64-family coverage and
all six SVG/PNG hashes are checked by:

```sh
bash assets/readme-performance/v1/render.sh --check
# Regenerate the current block after an owning evidence update:
bash assets/readme-performance/v1/render.sh --write
```

The [same-agent numerical replay](../assets/readme-performance/v1/replay.json)
recomputed the point estimate plus all 10,000 original family-bootstrap fits.
Every rating and interval endpoint matched exactly (maximum absolute difference
0.0). It used the retained match matrix and original bootstrap indices under
2 CPU / 2 GiB / zero-swap limits, with no new regression fits, inference or
label reads. This reuses the original numerical core; it is not an independent
scientific validation or a new dataset benchmark. Bootstrap indices and raw
fit custody remain outside Git as the original sources document.

The command reads committed aggregates and hashes only. It performs no numerical
fits and changes no original benchmark artifacts. The pairwise figure's short
label says "share of families"; its cell values are actually **the mean of
three seed-specific outcomes within each family, then averaged over families**.
The README caption states that definition explicitly. A 0.66 cell is not an
integer majority-family win fraction. Ties receive 0.5.

The older `readme-clover-overview.svg` remains byte-identical and linked inside a
clearly dated historical section. Its left panel is the September `test-10`
finalist ladder; its right panel is CLOVER v1 quality-head accuracy and a
shadow-only surrogate. It does not show the latest native-regressor ladder,
current trained-checkpoint Elo or a production flywheel improvement. Keeping it
as historical evidence preserves its paper/claim-ledger pins.

## Neural inference and refinement

The accepted step-60000 proposal policy, CLOVER-Learn v1 and the newer Dope
ranking studies are separate lineages. "Newest" and "lowest training loss" do
not establish the best downstream predictive model.

| Lineage | Actual recorded evidence | Current ranking boundary |
| --- | --- | --- |
| Accepted lineage-B step-60000 | A historical one-dataset neural-consumption proof completed genuine neural-origin fits and descendants. Its selected neural-arm winner came from exploration. Development RMSE was 166.318 versus generic 155.259 on 34 held rows. | No complete matched 0/4/16/64-generation Elo panel. This one-dataset proof is not a broad quality ranking. |
| CLOVER-Learn v1, final absolute step 17,773 | 40 completed epochs; full validation over 263 contexts had best L6 0.8968201 at epoch 28 and final L6 0.8974925. Fresh-program quality-head pairwise accuracy was 0.506 [0.441, 0.568] on 32 development contexts. | L6 and pairwise accuracy are not Elo or end-to-end regression RMSE. Final-checkpoint matched Elo has not been computed. |
| H27 Dope/landmark rankers | 36 fixed-endpoint routing models, 850 exposed TRAIN contexts, 34,000 probe fits; final predictions and choices replayed separately. Both predeclared incremental-lift intervals include harm. | Internal crossfit research; no new independent-family validation, serving promotion or neural-mode Elo. |

Sources: [the historical bounded proof](internal/asm-manual/sections/status.md#summary-of-the-one-dataset-proof),
[CLOVER v1 model record](../deploy/dl-flywheel/docs/model.md), and the [H27 fixed-endpoint study](research/studies/dope-neural-ranking-20261021/h27/nn-fixed400-study.md).
The H27 publication's current index supersedes older "pending" historical prose;
its claims do not promote weights. The original immutable experiment reports
remain the authorities for their own cohorts and endpoints.

### Requested mode labels

For a matched comparison after deep-learning inference, the owner's allowed
labels are:

| Profile | Subsequent search generations | Measured regression Elo |
| --- | ---: | --- |
| fast | 0 | Not measured |
| low | 4 | Not measured |
| medium | 16 | Not measured |
| high | 64 | Not measured |

These are **benchmark profile definitions**, not claimed shipped flags or results.
Zero means no subsequent refinement; it still includes encoding, neural proposals,
fitting and training-side model selection. Generation counts are not candidate
evaluation counts, and a generation-zero candidate set is not one inference-only
prediction. A generic `jope-bench --search-gens` run cannot be relabelled as
post-neural inference: that route uses generic search and has no neural proposals.

The existing metadata-selected
[panel-selection.json](../benchmarks/panel-selection.json) supplies ten regression
problems without selecting on this comparison's outcomes: CPU activity, Puma,
machine CPU, PM10, pollen, Friedman, insurance price, galaxy, geyser and wind.
It excludes the favorable historical `test-10`, includes both real and simulated
data, and is a suitable bounded development panel within the requested 10–40
problems. Its exposure is not newly independent. Preserve date/duplicate groups
and every selected problem, failed arm and unavailable outcome.

A valid new mode ranking needs one frozen consumed checkpoint, exact source and
encoder/worker qualification, the same initial proposal set and seed for
all four modes on each fold, declared candidate-set size, actual completed
neural-origin fits, and independent training-side search inside each scored fold.
Include a matched generic control and fixed baselines, full loading/encoding/
proposal/fit/refinement costs, predictions and explicit failures. Freeze panel,
seeds, budgets and the failure rule before outcomes. Report all four profiles,
Elo uncertainty and error sizes together. Do not reuse sealed test labels for
selection, silently substitute generic fallback for neural consumption, or
produce a current-policy Elo by renaming a historical result.

## Git-history audit

| Commit | What it establishes |
| --- | --- |
| `2468be138` (2026-09-23) | Golden42 envelope rows were demoted along with the test-adaptive rows; published rank 1/Elo 2259.8 is best-test-selected evidence. |
| `c852d7c60` (2026-09-23) | Explicitly records favorable `test-10` selection. |
| `aa5943167` (2026-09-23) | Same-cut v2 adds four external engines and renders the September table. Its deep ladder contains finalists only, with uneven coverage. |
| `5404ecacb` (2026-10-13) | CLOVER documentation refresh distinguishes v1 training, scratch comparisons and absent final-checkpoint matched Elo. |
| `5b8b8d56b` (2026-10-17) | Deterministic README/CLOVER SVGs document the named historical evidence. |
| `8dc79c587` (2026-10-22, #407) | Adds Gemini and the complete four-entrant 64-family comparison. README previously linked it only from the crate-layout row. |
| `5390576ff` (2026-10-22, #415) | The combined paper preserves the four-entrant comparison and its training/exposure limitations. |

The historical benchmark-valid Jope Dime Regressor rating remains **1687.0,
rank 31/304, 42/45 coverage**, under its own published development protocol.
The September same-cut search remains **1556.6, rank 3/15, finalist on 5/10**, on
favorable `test-10`. The published best-test envelope remains **2259.8, rank 1**,
excluded from the corrected benchmark-valid ladder. These are different
questions and opponent pools; none supplies a missing neural-mode Elo.

Reproduce their original committed blocks with
`cargo run --locked -p jope-records -- readme-benchmarks --check`.
No historical source, result, correction, fitted state, SVG or claim ledger was
rewritten to improve the displayed standing.
