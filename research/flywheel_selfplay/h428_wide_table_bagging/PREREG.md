# H428 PREREG — feature bagging for wide tables on the FIT-SEL monitor (prompt A of H427)

**Status.** Published on `origin/v1/next` before any output on a `FIT-SEL` context exists. Claim:
`~/jope_plan/ns/receipts/H428_claim.txt` (2026-11-11T11:08Z).

**Authority.** Relayed by ENG:watcher on 2026-11-11 (stamped 4:08 AM MST): take prompt A of H427; no GPU; estimate the
CPU cost first and run only if it fits 16 single-thread processes on free xbabe2 CPUs and 4 hours, else a
pre-declared subsample; PREREG with success and failure criteria before any output; `FIT-SEL` only; prompts B and C
held; the hour-24 driver and all training runs are left alone.

**Closed.** dev91-audit-v1, DL-Val, DL-Holdout and sealed data are never read. `DEV` is not rescored. `CONFIRM` is
not staged, opened or read. No training run is touched; no GPU is used.

## 1. Question
H427: the in-context column earns nothing on tables with more than 30 features, and the model was trained mostly on
narrow tables. Does showing a wide table to the model as several 30-feature tables give a better readout than the
full-width readout, with the same checkpoint?

## 2. Data, model, cost
- **Data:** the stored `FIT-SEL` monitor bank of H419 (sha256 `8e517921b7d5…`; 106 contexts, SEARCH rows only). Task
  per context as in the monitor: query = fold 0 of the stored fold labels, context = folds 1–4.
- **Contexts tested: all 33 with more than 30 features** (25 with 31–100, 8 with more than 100, widest 398). The
  73 narrower contexts are untouched by the method and are not run.
- **Model:** H419 `ICL_M_r3`, checkpoint `ckpt_h020.pt`, sha256 `e3c8fd9972ce…` (the H424 checkpoint), EMA weights,
  H419's published code.
- **Cost estimate, before launch.** One thread, measured in H424: a forward pass takes about (rows / 1,600) ×
  (35 + 0.72 × features) seconds. Control: one full-width pass per context, about 2,500 core-seconds in all.
  Two bagged variants: 16 passes of 30 features per context, about 20,400 core-seconds. **Total about 6.4
  core-hours; at most 860 seconds per context; about 45 minutes with 16 processes. It fits the 4-hour limit, so no
  subsample is taken.**

## 3. Arms (same checkpoint, same query rows)
- **Control:** the monitored residual-of-ridge readout on all features, with the monitor's seed.
- **T2 (the queued prompt, literally):** that readout on each of 8 seeded random subsets of 30 features (ridge and
  model both see only the subset), predictions averaged.
- **T1 (variant added here):** ridge on all features as in the control; the model predicts that ridge's residual
  from each of the same 8 subsets; the 8 residual predictions are averaged and added to the ridge.
  - Why it is added: T2 throws away the linear part carried by the features outside a subset; T1 keeps it. I
    expect T1 to be the better of the two and T2 to lose on the widest tables. Both are fixed here with one
    multiplicity rule, so neither can be picked afterwards.
- Subsets: `numpy.default_rng(sha256("H428-subset" ‖ NUL ‖ context ‖ NUL ‖ k))`, 30 features without replacement.
  The number of subsets and their size are not tuned.

## 4. Test, success and failure
u = −ln(RMSE / RMSE of the context-mean predictor) on the query rows, clamped to [−2, 6] as in the monitor.
- **Units:** clusters of related `FIT-SEL` contexts (H426's `feature` clustering on the stored H425 rehearsal screen
  of `FIT-SEL`), because H425 showed that several `FIT-SEL` families are one source. The same statistics with
  single contexts as units are reported beside them.
- **Statistic per variant:** `h335_stats.primary_test` of the variant against the control: clipped unit-mean gain,
  exact one-sided sign-flip p, leave-one-unit-out minimum, median.
- **Two variants, Holm:** the smaller p must be below 0.0125, the larger below 0.025. A variant passes if its
  clipped gain is above 0, its p meets its threshold, its leave-one-unit-out minimum is above 0 and its median is
  not negative.

| Verdict | Condition | Consequence |
| --- | --- | --- |
| `BAG_CONFIRMED` | at least one variant passes | the passing variant with the larger clipped gain becomes the readout for tables with more than 30 features in the next stack-column test (its own PREREG) |
| `BAG_HARM` | no variant passes and one has a clipped gain below 0 with mirror p below 0.0125 | recorded; not retried with other subset counts or sizes |
| `BAG_NULL` | anything else | recorded as a negative result; prompt B (training mix) remains the route for wide tables |

- Reported: mean gain and wins / losses for 31–100 and for more than 100 features; the control's margin over
  plain ridge; contexts without a result (they score the control in both variants and are counted).

## 5. Budget and limits
- 16 single-thread processes on xbabe2 cpus 40–55 (`CUDA_VISIBLE_DEVICES` empty), away from cpus 16–31, which the
  H424 second read will use. Hard cap 4 hours: no new context after 3 h 30 min, stopped at 3 h 50 min.
- Checkpointing: one create-once file per context; a restart continues.

## 6. Gates (`gates_h428.sh`, after inference, before the readout)
This PREREG is older than the first output and no readout exists yet; the pinned checkpoint and the monitor bank by
sha256; the 33 wide `FIT-SEL` contexts and nothing else; CPU only, at most 16 threads, no CUDA device visible; the
4-hour cap; the sandbox view still holds no `CONFIRM` table; published stage.

## 7. What was looked at before this PREREG
- The stored monitor readings by width (H427's README) and H427's analysis.
- A code smoke on a **synthetic** bank of random numbers (three made-up tables): finite outputs; no real context.
- **No output of this method on any `FIT-SEL` context exists.**

## 8. Weaknesses
- 33 contexts in fewer clusters: low power; only a clear effect can pass.
- `FIT-SEL` is the model's monitor cohort and guided two training changes in H419; it is not a fresh cohort. A pass
  still has to show up in a stack-column test on other families.
- One checkpoint, one readout family, one subset size.
- The monitor task (one fold as query) is not the stack column (five folds plus AUDIT rows).

## 9. Files
`bag_h428.py` (CPU inference), `readout_h428.py`, `gates_h428.sh`, `stage_h428.sh`. Reused at published bytes: H419's
`icl_model.py`, `icl_resid.py`, `icl_diag.py`, `icl_inject.py`; `restate_h426.py` (clusters).
