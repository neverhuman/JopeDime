# H424 PREREG — the in-context model as one more stack column (`icl_stack_column`)

**Status.** Published on `origin/v1/next` before any `DEV` table is opened by this wave and before any stack value
with the model's column exists on `DEV`. What was looked at is in §8. Claim: `~/jope_plan/ns/receipts/H424_claim.txt`
(2026-11-11T07:54Z, `cl_apex`).

**Authority.** Owner order of 2026-11-10 (APEX campaign; our own in-context model as stack columns). Relayed by
ENG:watcher on 2026-11-11 (stamped 12:58 AM MST): the 24M model's predictions from its latest saved checkpoint as
one more stack column, against the stored H420 stack on `DEV` families only; CPU-only inference, at most 16 threads
on xbabe2, no GPU; hard cap 4 hours; primary = clipped mean gain at one-sided 0.025; pre-declared secondary = win
share with the ±0.5-clipped contexts excluded; rerun with the hour-24 checkpoint only if the primary passes.

**Closed.** dev91-audit-v1, DL-Val, DL-Holdout and sealed data are never read. `CONFIRM` is not read. No GP worker
evaluation runs. The 104M and 24M training runs are not touched: one checkpoint file was copied read-only.

## 1. Question
Does the stored stack get better on fresh families when our in-context model's prediction is offered to it as a
65th candidate?

## 2. Baseline, model and data
- **`B`** = the stored `h416.SXB` value of H420's `TBOTH` recipe on the 115 `DEV` contexts (the stack H421, H422 used
  as baseline). Not recomputed.
- **Model:** H419 `ICL_M_r3` (23.66M parameters), checkpoint `ckpt_h020.pt` (20.0 training hours), sha256
  `e3c8fd9972ce0c1684630ff06b159e950f6f498c4461d0454e0c6a6f44d38dd7`, EMA weights. It is the latest *named*
  checkpoint at the time of writing; `ckpt_last.pt` is 20 to 40 minutes newer but is overwritten every 20 minutes
  and cannot be pinned on the training host. The model code is H419's published code (stage `r9`).
- **Training data of the model:** synthetic tasks and the `FIT-ICL` families. No `DEV` family. `FIT-SEL` was its
  monitor.
- **`DEV` tables** are read once, inside the sandbox with the guard, to write an inference bank: SEARCH rows
  (features, targets, fold labels) and AUDIT **features**. No AUDIT target leaves the sandbox; only the sha256 of
  the AUDIT target vector is written, so the stack step can prove row order. The bank is kept apart from the
  model's training bank.

## 3. Method
- **The column** ("residual-of-ridge" readout, the one H419 monitors and trains for): ridge on the context rows
  plus the model's prediction of the ridge residual (`icl_diag.run_variant("ridgeres")`, published bytes).
  - SEARCH rows: 5-fold out-of-fold on the stored fold labels (each fold is the query, the other four the
    context), the same construction as the stored candidates' out-of-fold matrix.
  - AUDIT rows: "bag" = mean of the five fold-context predictions; "refit" = prediction with all SEARCH rows as
    context. These are the two columns every stored candidate has.
  - The model sees SEARCH targets of context rows only. Six forward passes per context.
- **Arm `I`:** the stored 64 candidates plus this column; the stored weight builder (bootstrap Caruana,
  `meta_h416`, same row bootstraps) on 65 columns; the stored mix of bag and refit predictions. Nothing else
  changes. No winsorising, no wider admission.
- **Alignment checks per context** before any value is computed: SEARCH targets equal to the stored ones, fold
  labels equal, and the sha256 of the stored AUDIT targets equal to the one written inside the sandbox.
- **Not computed:** the plain readout as a second column (CPU time is spent on the primary column only).

## 4. Test and verdicts (115 `DEV` families, paired by family, value = AUDIT u)
Primary: `I` vs `B`, `h335_stats.primary_test`.

| Verdict | Condition |
| --- | --- |
| `ICL_CONFIRMED` | the test passes: clipped family-mean gain above 0, **one-sided sign-flip p < 0.025**, leave-one-family-out minimum above 0, median not negative |
| `ICL_HARM` | clipped family-mean gain below 0 with mirror sign-flip p < 0.025 |
| `ICL_NULL` | anything else; recorded as a negative result |

- **Pre-declared secondary:** with every context whose gain is at the ±0.5 clip excluded: wins, ties, losses, the
  win share of decided families, and the mean gain.
- Reported, no verdict: ΔElo with its 95% family-bootstrap interval (10,000 draws, seed 424150); in how many
  families the column gets weight and how much; the column's AUDIT value alone against `B`; the same test on the
  contexts whose column was computed, if some were not.
- **Contexts without a column** (inference not reached before the budget ended, or a non-finite prediction) score
  `B` in arm `I` (gain 0) and are counted. Inference runs cheapest table first, so any such context is a wide one.
- **Consequences.** `ICL_CONFIRMED`: the run is repeated once with the hour-24 checkpoint under this PREREG
  (reported as a second read, named as such), and the column becomes a candidate for the champion recipe.
  Otherwise no rerun; the stack stays as it is.

## 5. Budget and limits (recorded before launch)
- xbabe2 CPUs only. Inference: 16 single-thread processes on cpus 16–31 (`CUDA_VISIBLE_DEVICES` empty; the model
  is never moved off the CPU). The stack step uses the same 16 CPUs.
- **Hard cap: 4 hours from the start of the bank step to the end of the stack step.** Inference starts no new
  context after 3 h 10 min and is stopped at 3 h 40 min.
- Expected: about 12 core-hours of inference, about 1 hour (development timing, one thread: 37 s per pass at 3
  features, 49 s at 19, about 90 s at 78;
  six passes per context; the widest tables, up to 400 features, take about half an hour each).
- Checkpointing: one create-once file per context; a restart continues.

## 6. Integrity gates (`gates_h424.sh`, after the stack step, before the readout)
1. The stored `DEV` matrices match their manifest; the checkpoint is the pinned one and equals the file on the
   training host.
2. The bank holds exactly the 115 `DEV` contexts, no AUDIT target, guard installed; the stack step used that bank.
3. Alignment holds on every context; recomputed stack weights equal the stored ones (difference 0) and the
   recomputed baseline value is within 1e-9 u of the record.
4. This PREREG's commit is older than the bank file; the staged PREREG equals the published one.
5. Inference ran on CPU, 16 threads at most in total, no CUDA device visible, with the pinned checkpoint.
6. The 4-hour cap was kept.
7. Published stage, code hash verified (the stage also checks the model code against H419's training stage).

## 7. Expected outcome
- On `FIT-SEL` SEARCH rows (H419's stack test, 12-hour checkpoint) the column was used in 49% of contexts with
  weight 0.11 and added +0.005 u **on the rows the stack is fitted on**. H421 and H422 saw such gains shrink to a
  third or a quarter on held-out rows.
- **Expected on `DEV`: about +0.002 u, far below what 115 families can detect (about +0.02 u). `ICL_NULL` is the
  likely verdict** even if the column helps slightly.
- At 18 hours (the last monitor reading) the model alone is well below the stack: median u 1.01 against 1.23 for
  the GP blend on the `FIT-SEL` monitor.

## 8. What was looked at before this PREREG (design cohort and timing only)
- The checkpoint was copied from xbabe1 and its sha256 compared.
- A `FIT-SEL` inference bank (design cohort) was written in the sandbox, and columns were computed for 10 `FIT-SEL`
  contexts to time CPU inference and to check the join: alignment holds on all 106; replay difference 3.5e-11.
- On the 8 smallest `FIT-SEL` contexts (3 to 7 features) the column got weight in 4; gains over the stored stack
  were +0.004, 0, 0, 0, −0.009, +0.001, 0, 0 u. Not results: 8 contexts, design cohort.
- **No `DEV` table has been opened by this wave and no `DEV` column exists.**

## 9. Weaknesses
- One column, one checkpoint, one readout; the model is still training (46-hour run).
- CPU inference is slow (the model's forward pass does not use more than about one core), so the plain readout
  and repeated-permutation averaging are left out.
- The test cannot detect an effect of the expected size.
- `DEV` has been read by H420, H421 and H422; this is its fourth pre-registered read.

## 10. Files
`bank_h424.py` (sandbox; inference bank), `cols_h424.py` (CPU inference), `stack_h424.py` (`apply`, `readout`),
`gates_h424.sh`, `stage_h424.sh`. Reused at published bytes: `cache_h422.py`, `meta_h416.py`, and H419's
`icl_model.py`, `icl_resid.py`, `icl_diag.py`, `icl_inject.py`.
