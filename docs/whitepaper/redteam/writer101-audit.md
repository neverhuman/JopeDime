# Flywheel text/figure synchronization and Section 4–6 source review — t2245

Reviewed main `a327af9ae49f14b61d1b7fcac851b0af3da13666` on 1 November 2026. This pass reads committed aggregate CSVs, provenance and source only. No new observations, remote logs, dataset/weight files, sealed/holdout payloads or experiments are used. Opus owns Sections 1–3/7+/appendices and float placement; those source lines and all data/figure/generator files remain unchanged.

## Findings and corrections

The values quoted in Section 6 agree with their respective source windows. The drift is in coverage wording after the admission figures were extended to round 2633:

1. Figure 6's missing **summaries** are now named as such; admission records for the intervening rounds are present in the complete prefix. The 25-summary gap estimate is unchanged.
2. Figure 7's caption now bounds the summary points to 1342–1469 and the timestamp intervals to 775–1469. It states the existing positive, under-600-second filter, distinct-context throughput and rolling completed-round program-event rate. These earlier operating windows are distinguished from the extended admission prefix.
3. Figure 8 and its narrative use the absolute 1838–1857 window. Calling it a later slice beside the now-extended trajectory was ambiguous. Its candidate utility agrees with every overlapping admission record.
4. Figure 9 and its narrative name rounds 2005–2024; the follow-up pointer names 2082–2101. Their event counts remain separate and no missing rounds are joined.
5. The 41.6 rounds/hour and 45.8 training-program events/hour statement is explicitly limited to 2343–2362. It is not an adapter admission rate across the full prefix.
6. The appendix pointers explicitly retain 2343–2362 and 2508–2527. No appendix source is edited and no program-event count/rate is extrapolated through 2633.

Only caption/narrative wording changes in `sections/06_method/flywheel_status.tex`. All existing literal measurement values are retained; absolute bounds and the existing filter threshold are made explicit. Every substitution preserves its source line count, protecting all 28 TBD source locations and the 20 Needs anchors. Figure labels, source order, widths and float options are unchanged.

## Numerical reconciliation

| Figure / record | Retained observations and bounds | Reconciliation |
| --- | --- | --- |
| Figures 5 and 12 | 1,859 admission rounds, 775–2633 | Four accepted: strict gains at 785/859, ties at 782/882. All 1,751 rounds after 882 rejected. Final recorded best 1.1377941898164525 rounds to 1.1378; logged identity 882 is an accepted tie, not a strict gain. |
| Figure 6 | 25 summaries: 1342–1345 and 1449–1469 | Train minus candidate-validation utility ranges from 0.2823019198833662 to 0.2903527105229568, which rounds to the quoted 0.282–0.290. All 25 validation scalars equal their admission records. |
| Figure 7, left/right | Same 25 summaries | Distinct contexts divided by completed-round wall seconds; strict program-event rate uses up to 20 completed rounds in the reporting process, including qualification waits. It is not recomputed over only the sparse retained points. |
| Figure 7, middle | Earlier 775–1469 admission prefix | 692 consecutive positive timestamp differences under 600 seconds; logged wall-time squares are the 25 summaries. Timestamp differences are not accepted-update counts or deployment costs. |
| Figure 8 | 20 summaries, 1838–1857 | 22 strict training-program events; every candidate-validation scalar matches the full admission prefix. |
| Figure 9 | 20 summaries, 2005–2024 | 15 program events, zero adapter admissions; all overlapping candidate/best utilities and flags match. |
| Appendix follow-up window | 20 summaries, 2082–2101 | 10 program events, zero adapter admissions; all overlapping utility/flag fields match. |
| Later window / completed-time view | 20 summaries, 2343–2362 | 22 events in 1730.12998260400001 logged seconds: 41.61537036173061122373001406 rounds/hour and 45.77690739790367234610301547 events/hour. The manuscript's 1730.1, 41.6 and 45.8 are correctly rounded. |
| Subsequent window / utility / rates | 20 summaries, 2508–2527 | 15 events in 1731.449021203 seconds: 41.58366727423187329483085585 rounds/hour and 31.18775045567390497112314189 events/hour. Zero adapter admissions and best identity 882 match the complete prefix. |
| Historical target summary (Figure 10) | Three aggregate rows | Arm counts sum to 159 and lower-score counts to 64. Distinct-cut totals 55/23 are reported by committed provenance; per-cut records are not re-opened here. These are not admission counts. |
| Historical trajectory (Figure 11) | Three arms, 71 generations each | All endpoints agree. First endpoint generations are {"Generic": 42, "Policy-seeded": 43, "Shuffled": 42}; the policy-seeded arm reaches the reported endpoint one generation after each control. These historical evaluator scores are distinct from adapter pool utility. |

No scalar mismatch was found. All candidate/best values and admission flags in the five retained program-event windows agree with the corresponding rows of `live_admissions.csv`. The 1,000-evaluation historical search budget remains the value in the committed provenance. No independent-performance or confidence claim is introduced.

## Final source review

Re-read each writer/map/checkpoint claim in Sections 4–6 against the current native model, decoder, grammar, ranking/seeds/parent route, objective/sampler, context/checkpoint contracts and the inspected adapter loop. All 20 source/note pins from #1304/#1317 remain unchanged; `crates/jope-telemetry/src/hooks/flywheel.rs` is additionally inspected for the timing/rate formula. The implemented pool and ranking paths, historical reference specifications and proposed one-draw deployment remain distinguished. Exact checkpoint/runtime binding remains TBD. No further writer/value-net overclaim was found.

Validation receipts retain all 21 source and 27 data/provenance SHA-256 pins, exact decimal arithmetic, overlap checks, source substitutions, clean nice19 serial make, references/TBD/import checks, page proofs, five static records and a head-bound same-agent review. Results, Meaning and the abstract remain byte-identical. These checks do not qualify a live learning loop or replace the pending independent evaluation.

Visual review: sixteen before/after page proofs cover pages 13–16 and 19–22. Caption reflow changes some page assignments, including Table 9 from page 19 to 20 while Appendix A remains on page 19. The 7.3 heading retains text beneath it; no clipping or overlap is visible. Figure/table numbering and all float source settings/order are unchanged. Opus retains broader layout ownership.
