# JopeDime continuous GP policy — v2.0.0 learner (merged source) and v1.0.0 history

Updated 2026-10-13. This document describes the v2.0.0 continual-policy learner whose source merged to `main` in PR #292 (`9e5baa28f`, 2026-10-12) under `deploy/dl-flywheel/pipelines/continuous_v6/` and four companion packages. It also keeps a short record of the v1.0.0 research candidate that v2 continues. v2 exists as source, offline fixtures and build-qualification receipts. It has not trained live. The 72-hour live run, the R9 drain of v1's recorded proposals and any promotion are Jepson-only decisions, and none has been taken. No receipt shows that any v2 objective improves executed-program quality, GP win rate or L6 over frozen v1.

## Status on 2026-10-13

Source: merged. `continuous_v6` is build r3.1 (the R23 anchored objective and the KL guard), and #293's caller-completion durability for overlays landed in the same squash merge. Offline: `deploy/dl-flywheel/run_v2_offline.py` checks an exact roster of 230 CPU fixture identities with no skips: `continuous_v6` 83, `native_result_overlay_v1` 55, `generated_target_preparation_v2` 36, `final_capture_v1` 32 and `boundary_admission_v2` 24. Build qualification: CUDA next-update parity and CPU/CUDA dry runs on the r3 bytes, recorded as lane receipts on xbabe2. Live: no v2 campaign has been launched, no adoption has run against real state, the KL guard has never run on a live campaign, and no S3 write path is qualified. Evidence that a v2 objective beats frozen v1: none. The scratch sweeps and the 2026-10-13 results described below (#304, #306 and #307 merged; R25, R26 and R27; the J21 and sweep 6 looks; the L6-1148 set, whose first verdict is a REGRESSION ALERT for all three checkpoints) are non-qualifying.

## Lineage: from the sealed v1 capture to a fresh v2 schedule

v1 ended at absolute step 17,773. At 00:29:05Z on 2026-10-10 the systemd runtime limit stopped its unit while it was idle; no step had been in flight since step 17,773 at 22:05:30Z. Owner ruling R17 accepted this as a clean stop, conditional on a read-only final capture proving four things:

- the unit was inactive with an empty cgroup;
- nothing in the campaign root changed after the in-loop deadline;
- a private planner backup had 0 unresolved reservations, and its last committed step's checkpoint equalled the rolling checkpoint and progress;
- a CPU restore of `rolling/000017773` gave the same parameter SHA256, `ea078456690b0665592652b2ed5b04a0deb39b9e604de97a513dc885e467f9fb`.

v1's missing `status.json` and `result.json` are recorded as absent and were never reconstructed. The sealed capture (`final-capture.json` `b4db984d…`, capture receipt `364eb9a2…`) records 40 completed epochs, 17,773 optimizer steps, 284,006 presentations and 6,982 exposed training contexts.

`genesis.py` builds a new v2 campaign root from that capture and refuses to reuse an existing one. Genesis:

- byte-copies v1's final checkpoint;
- keeps the train-only normalizer bytes;
- opens a fresh planner on v1's active manifest that inherits v1's ancestry, so every historical batch fingerprint stays banned;
- opens a fresh v2 outbox.

v1 stopped partway through planner epoch 40 (lineage epoch 41), with 5,712 contexts consumed and 1,270 remaining. v2 does not continue that partial epoch (Codex finding F5). Genesis records it exactly and declares a fresh full pass over the active manifest. v1's partial epoch is never counted as completed, and v2 never schedules v1's own next batch.

The owner rulings that govern v2 are AskUserQuestion receipts in `.agent/proposals/claude-rlfw-20261009/rulings/` on xbabe2:

- R7–R16 (`1b30486e…`), which includes R11's 72 h / 80,000-update window and R14;
- R17 (`2a1b0a5e…`);
- R19 (`556e3e55…`): build only, and ask again before the drain or a live launch;
- R22 (`55d9452c…`): scratch-only training;
- R23 (`8bd7544a…`).

The r3 build-qualification campaign config binds each of these by hash.

## R14 warm restart

When v1 ended, both learning-rate groups sat at the 1e-6 floor after nine halvings, and the fixed-panel plateau best was 0.8705178834. R14 applies one logged control event at genesis (`training/control-event-r14.json`):

- both groups are set to 3.75e-5;
- the plateau state is reset, with patience 5 and the 1e-6 floor unchanged;
- the AdamW moments are kept.

Genesis applies R14 only after three gates have passed on v1's unchanged control:

1. bitwise CPU restore of the captured checkpoint;
2. bitwise CUDA restore, round-trip and RNG parity (xbabe3 RTX 3090, receipt `7cb7dac6…`);
3. the exact next-update parity receipt described under build qualification.

A probe root built with `probe=True` keeps v1's control, carries no R14 event and can never be a live root. CPU next-update parity was not run.

## Objective: L6 plus an anchored KL term (R23)

The label objective is still v1's L6, unchanged (see the v1 history below). Owner ruling R23 (decision D10, 2026-10-11T00:41Z) amends R8 ("L6 unchanged") and keeps R14 and the build-only limits of R19/R20. It makes the v2 training objective:

`L = L6 + 0.2 * KL(policy || v1 anchor)`

`anchored_step.run_microbatches_anchored` takes each update in three parts:

1. **L6:** frozen `neural_native_train.run_microbatches`, copied line for line.
2. **KL term:** form `exact_trie_kl_eval_mode_v1`, computed only on the batch's first context (`contexts_per_step` 1):
   - the exact chain-rule KL over every branching node of the deployed train-catalog trie, EOS included, under the deployed child mask;
   - temperature 1.0, and the runner requires `decode_temperature == 1.0`;
   - the policy in eval mode, so the term draws no random numbers;
   - the policy pass carries gradient, with activation checkpointing per 16,384-token trie chunk; the anchor pass runs without gradient.
3. **Frozen tail:** finite-gradient check, clip at 1.0, optimizer step.

The anchor is v1's final policy (`ea078456…`). It is loaded read-only from the sealed capture, once per segment, and shared with the KL guard. A single-program trie contributes nothing.

`loss` stays the L6 total, so the loss history, plateau control, G32 guard and parity comparisons stay on the pre-R23 scale. Proofs and optimizer history report the anchor term separately, as `anchor_kl` and `objective_total`.

The training-objective identity is `{schema: jope.v2-training-objective/v1, l6: OBJECTIVE, anchor_kl: <block>}`. For the real capture its digest, `training_objective_sha256`, is `1c0c050f877b576ac37fc54842ecf3cbd17d44641af81c9320c87e639453a55e`.

- It must equal the config's value, the campaign identity's and the value sealed at genesis.
- It is bound into every batch proof, `progress.json`, the terminal intent, `result-verification.json` and `result.json`.
- A genesis or closure without it (r2.x) is refused; there is no migration path.

The C10 contract's `objective_sha256` stays the L6 label objective (`bf02f87341bc…`):

- the contract, generated-target preparation and the native-result overlays pin how targets and supervision are built;
- the anchor term reads no labels or targets;
- changing the contract would invalidate every preparation proof and overlay chain for no semantic reason.

At the anchor the KL value is exactly 0, but its gradient is not: the gradient keeps a float64 rounding residual of at most about 3e-17. In the fixtures and on CUDA, the accumulated gradient and AdamW absorbed it, and the first anchored update equalled the frozen v1 step bitwise. This is an observed property, not a structural guarantee, which is why the next-update parity gate was re-run on r3.

R23 rests on R22 scratch sweep 4a, which is non-qualifying (see the evidence-status section). At a learning rate of 3.75e-5, the probe KL mean from the anchor behaved as follows:

- **Pure L6:** 0.031 after one epoch and 0.053 after two, past the 0.05 HOLD line.
- **L6 + 0.2·KL:** between 0.004 and 0.011 for all eight epochs.

## Adoption guards

Two guards act at every closed epoch, inside the same `close_epoch` step and before any adoption. They share these properties:

- they hold adoptions only, and L6+KL training continues on the active snapshot;
- once tripped they are sticky and raise an alert for owner review;
- their state is sealed in every verified closure, recovered only from that seal, and absent at genesis.

**G32 learning guard.**

- It compares the latest fixed-panel validation L6 with the sealed v1 reference, 0.8705178834 (v1's plateau best on the fixed original panel).
- Three consecutive closed epochs above 1.02 × the reference trip it (`learning-guard.json`). A good epoch resets the count before a trip, but never clears a trip.
- Restart refuses a deleted, reset, malformed or symlinked sidecar.
- Each closed epoch's boundary check runs exactly once (`last_checked_epoch`). Boundary work deferred at a deadline runs at the next start, before any adoption or reservation, or is deferred again.
- The guard makes no claim about how much was learned.

**KL guard (gap T12).**

- It measures the exact trie KL(current ‖ v1 anchor) on 64 native TRAIN probe contexts chosen by a campaign-seeded digest. Validation and outer-test contexts are never probes.
- Each probe also checks leaf-probability closure to 1e-9. A failed check makes the probe unavailable; it never yields an approximate value.
- A mean of 0.03 or more is a recorded, non-blocking alarm. A mean of 0.05 or more, or a maximum of 0.15 or more, trips a sticky HOLD (`kl-guard.json`).
- The probe runs first in the boundary, before any boundary side effect. It is deadline-bounded: it checks the deadline at entry, after every chunk forward, after every probe and before returning.
- A probe that cannot finish in time is unavailable. The whole boundary is then deferred, with no change to guard state, evidence or plateau.
- Per-epoch evidence is written to `kl-guard-epoch-<epoch>.json`, and the history keeps the last 50 checks.
- So far the KL guard has measured only in fixtures. The r3 dry runs sealed its empty initial state, because no epoch closed within their two or three updates. It has never measured on real training data.

## Custody and terminal closure

**Verified watermark.** After every closed epoch, `custody_v6` writes a verified watermark (`verified-watermark.json`). Each pass:

- chain-verifies planner and outbox events from the previous watermark's head;
- joins every committed step to its proof (bytes equal to the planner's commit evidence), its durable outbox ACK row and the outbox `batch.emitted` event;
- checks that counts stay continuous.

Work on the suffix is bounded, and the prefix is re-joined by its head SHA.

**Prefix integrity.** A watermark certifies only the bytes seen when it was written. So before any clean closure and any resume:

- `prefix_integrity` re-hashes every committed proof file against its planner-chain evidence (about 101 KB per proof; projected at about one minute for 80,000 proofs);
- `proof_inventory` requires the proof filenames to be exactly the committed steps, all regular files.

**Terminal closure order.**

1. `segment.json` names each run segment before any reservation.
2. `terminal-intent.json` is published. An intent alone is not a clean stop.
3. The suffix joins, prefix re-hash, proof inventory and full indexed planner and outbox verification run, all within `closure_budget_seconds`.
4. `result-verification.json` is written. It binds the intent, watermark, heads, control, checkpoint, execution and both guard states.
5. `result.json` is written, binding both.

Any closure error publishes `closure-failure.json` and no clean result. Resume refuses a missing, stale, failed or rebound closure, and any `fatal-*.json`.

**Tick-safe stop.** The v1 rule carries over with tighter bounds:

- a reservation starts only when the remaining in-loop time covers the bounded step budget;
- closed-epoch boundary work starts only when `epoch_boundary_budget_seconds` remain;
- the emission cap counts only v2's own unresolved rows. `emissions_per_step` is 1 (the training proposal only, as in the r3 build root) or 2 (training plus validation, as in v1);
- when memory is below the floor, the loop waits without a reservation.

The r3 CPU and CUDA dry runs closed in 0.47 s and 0.46 s.

## Adoption path: C10 inbox and native overlay

**C10b inbox.** At startup (when the current epoch is closed) and after every closed epoch, the boundary driver reconciles the C10b inbox. It adopts at most one candidate, and none while a guard HOLD is active.

- A candidate may carry a variant component (X1/C10a prepared roots) and native-overlay components.
- An inbox HOLD keeps training on the active snapshot and is published; a refused inbox stops the run.
- A prepared root is consumed only with its operator acceptance envelope: exact terminal marker bytes, the sealed original bound and physical in-time success.
- A fresh authority proof is taken after hydration and checked again inside the CAS.
- First exposures are joined to committed batches.

**Native overlay.** Native verified results reach training through `native_result_overlay_v1` as train-only delta packets.

- Every overlay record is CE-false. CE still applies only to the frozen HWM winners, so overlays teach the quality, ranking, feasibility and runtime heads, never imitation.
- The builder never certifies itself. Only after `build_packet` returns does the caller call `accept_publication(root, physical=...)`. That call samples the clocks first, then writes one create-only envelope at `<parent>/<root>.acceptance.json` (`jope.native-overlay-bounded-terminal-acceptance/v1`).
- The envelope pins the exact marker bytes, the sealed packet and receipt, the unchanged original bound, the post-return sample, and physical evidence (`in_process_return` or `systemd_unit`).

**Return witness (#293, landed with #292).**

- The bounded publisher also file-fsyncs a publication candidate and links it to `<root>.acceptance-complete.json`.
- After those final operations it rechecks the original bound.
- The operator then records a return witness at `<root>.acceptance-qualified/completion.json`, only after the publisher has returned inside that bound.

**Admission rules.** `validate_chain` and `completed_packet` require the accepted envelope, the publication candidate and the return witness for every packet. The inbox relies on them at publication, adoption and restart.

- Admission never relies on the absence of a refusal.
- A build that raised, a mid-build read, a returned-but-unaccepted root and a late caller all admit nothing.
- Callers must accept each packet before building its successor.
- The envelope is not signed. It guards against accidental and late admission, not against an adversary that can write files.

## Build qualification (not live qualification)

**Offline.** `run_v2_offline.py` enforces the exact 230-identity roster with no skips, verifies each package's pins before import, and hides CUDA. Before #293 added its durability fixtures, the r3.1 candidate passed 213/213 under both torch 2.7.0 and CI's torch 2.14.0+cpu. The cross-implementation KL pins run in-process against vendored, SHA-pinned reference copies.

**CUDA next-update parity.** The probe makes one isolated, never-committed update from v1's final checkpoint on v1's own next batch (`9a70e4f5…`, epoch 40, index 357, absolute step 17,774). That batch is reserved on a byte copy of the planner backup, with v1's control unchanged.

- Side A runs the frozen v1 step primitives. Sides B1 and B2 run `runner_v6` with one and two emissions.
- All three sides are bitwise equal on 48 model tensors, 144 AdamW tensors, 2 RNG tensors and the state file, with loss 0.9540845. This held in all eight receipts from r2 to r3:
  - `abec3214` (r2), `13ada28b` (r2.1), `c19fa1d2` (r2.2) and `c15aea30` (r2.3);
  - `7c3da3a5` (r2.4), `d03aac03` (r2.6) and `364ebfa4` (r2.7);
  - `10153efb` (r3, the R23 anchored step at KL 0).
- The scope is update-only. Emitted programs differ between sides by design, and v2 never schedules that batch.

**r3 build qualification** (`BUILD-QUALIFICATION-r3.json`, 2026-10-11T03:09Z, `24a8f505…`).

- A CPU dry run reached absolute step 17,775, with anchor KL 0 and then 1.4e-4.
- A CUDA dry run reached 17,776, with anchor KL 0, 1.5e-4 and 4.2e-4.
- Both ended `maximum_updates_reached`. Each sealed the KL guard state in `result-verification.json` with an empty history, so no guard measurement was made.
- It checked 21 runtime files against the r3.1 PR tree. On today's `main`, 19 of them are still byte-equal. The other two, `native_result_overlay_v1/packet.py` and its `run_offline.py`, changed in #293 after the qualification, so the merged overlay admission path has fixture coverage only.
- Not claimed:
  - operator physical proofs;
  - the 80,000-update closure time, which is a linear projection;
  - GPU step time and VRAM of the KL term, which were not measured separately.

## Evidence status: R22 scratch sweeps (NON-QUALIFYING)

R22 (2026-10-10) allowed scratch training on native detail-shard measurements, labelled scratch-only and non-qualifying. It allowed no live learner, drain, waves, S3 writes or merges. Validation and outer-test rows were evaluation-only.

Sweeps 3, 4a, 4b, 5 and 6 ran on RTX 3090s from v1's final parameters, with v1's AdamW moments. Earlier R22 runs (a first run and sweep 2) are not summarized here. Every receipt:

- is labelled `R22 scratch-only, non-qualifying`;
- sets `benchmark_valid` and `promotion_input` to false;
- lists the qualification guards it bypassed, including the seven-artifact native/Rust closure of its targets.

The results are directional only. In the table, "probe KL" is the mean over 64 train probes, given as its range over each arm's complete epochs. An arm marked "frozen" was stopped by the harness.

| Sweep | Receipt | LR | Probe KL by arm |
|---|---|---|---|
| 3 | `ccdc9e17` | 1e-5 | L6 control 0.012–0.044; L_PI 0.01 + 0.2·KL 0.004–0.008; L_PI 0.03 0.020–0.036; L_PI 0.1 0.102 at epoch 1 (frozen) |
| 4a | `f8b54e53` | 3.75e-5 | control 0.031 then 0.053 (frozen at 2); `kl_only` (L6 + 0.2·KL) 0.004–0.011 over 8; L_PI 0.01 0.011–0.020; L_PI 0.03 reached 0.057 at 4 (frozen) |
| 4b | `ad010b1e` | 1e-5 | control 0.011–0.041; L_PI 0.01 0.005–0.010; L_PI 0.03 0.024–0.042 |
| 5 | `71493c89` | 3.75e-5 | control 0.035 then 0.052 (frozen at 2); `r23` 0.004–0.008; `r23_lpi01` 0.009–0.016 |
| 6 | `773040d5` | 3.75e-5 | control 0.031 at both epochs (2 by design); `r23` 0.006–0.014; `r23_lpi01` 0.013–0.036 (7 complete epochs; the 8th was cut by the deadline) |

Sweeps 4b, 5 and 6 also scored a held-out split of 885 train contexts from 167 parent lineages, excluded from their training. Read directionally:

- **The KL anchor bounds drift.** At 3.75e-5, pure L6 reached a probe KL of 0.052–0.053 within two epochs in sweeps 4a and 5 (0.031 in sweep 6, stopped at 2 by design). The L6 + 0.2·KL arms stayed at or below 0.014 for up to eight epochs. At 1e-5, pure L6 still drifted to 0.041–0.044 by epoch 8. Drift grows with the L_PI weight: 0.03 reached 0.057, and 0.1 reached 0.102 in one epoch.
- **L_PI improves held-out L_PI.**
  - Held-out L_PI starts at 7.080. At the last complete epoch, the L_PI arms ended below their comparators:
    - sweep 4b: 6.813 (weight 0.01) and 6.591 (0.03), against 7.211 for the control;
    - sweep 5: 6.865, against 7.066 for `r23`;
    - sweep 6: 6.675, against 7.088 at epoch 7.
  - L_PI is the trained objective, and its targets lack the seven-artifact closure. Held-out L_PI is not a measure of executed-program quality.
  - Held-out L6 did not improve in any arm by its last evaluated epoch: it moved from 0.8610 to between 0.8621 and 0.8660.
- **No objective has beaten frozen v1 L6 under qualified evaluation; no qualified evaluation exists yet.**
  - Frozen v1's full-validation L6 is 0.8973272 over 263 contexts. The scratch arms' last evaluated epochs range from 0.8963 to 0.9022.
  - Six arms ended below frozen v1, by 0.00003 to 0.00098; the lowest is sweep 6's `r23` arm at 0.8963494. Two of them, the sweep 6 arms at epoch 8, were judged against frozen v1 at 04:47Z and are INCONCLUSIVE (next section). The other four have no paired verdict against frozen v1; sweep 3's two were judged only against their own L6 control, at epochs 1–3. None of the six is a result.
  - The `kl_only` arm's 16-context fixed panel moved from 0.8712 to 0.8649, while its full validation moved from 0.8973 to 0.8977.
  - The G-JUDGE scratch judge requires paired per-context series over at least 881 contexts before it labels a comparison GREATER or WORSE. Full validation has 263 contexts. So every comparison it judged through 2026-10-13T00:06Z is INCONCLUSIVE and none is GREATER. That covers sweep 3, the CX-OP RLOO/PKPO arms and the G-ABL IPO/DPO arms.
  - Sweeps 4a, 4b and 5 have no G-JUDGE verdict. Sweep 6 was judged at 04:47Z and is INCONCLUSIVE (next section).
  - The 881 figure is itself not an applicable sample size: it came from reading one pair's standard error as a paired standard deviation. The correction is recorded in `rlfw/judge/POWER_TABLE.json` (`se_pair_unit_check`), merged in #302. A qualifying sample size is still to be set.

## Evidence status: 2026-10-13 scratch results (NON-QUALIFYING)

Since the sweeps above, three PRs merged and three more R22 scratch training runs completed. The judge recorded three more paired looks. None of it is qualifying, and none of it shows an objective beating frozen v1 L6. L6 is a validation loss: lower is better.

**The first look large enough for G-JUDGE's rule (L6-1148, 11:38Z) raised a REGRESSION ALERT against frozen v1 for all three checkpoints it judged**: two runs of the anchored recipe, and R25, which adds kNN-CE (details at the end of this section).

**Merged:**
- #304 (GB-03): verified archive targets and frozen-round actor updates. Build-only libraries with CPU fixtures.
- #306: the bounded GP expansion study and the closed H1 pilot.
- #307: the R25-kNN-CE scratch arm source, `pipelines/rlfw/r25_knnce/`. It is harness v7, `L6 + 0.2·KL(π‖v1) + 0.25·CE_kNN`, where CE_kNN uses the frozen HWM teachers of a context's k = 4 nearest TRAIN neighbours.

**Paired looks against frozen v1** on the 263 validation contexts. Each difference is arm minus v1, so negative is better. Intervals are the per-context 95 % interval, then the 54-cluster interval. G-JUDGE labels a difference only at n ≥ 881, and never from a cluster interval, so every row is INCONCLUSIVE.

| Checkpoint | Updates | Arm − v1 | Per-context | Cluster | Record |
|---|---|---|---|---|---|
| sweep 6 `r23`, epoch 8 (incomplete) | 2,621 | −0.000978 | [−0.00446, +0.00250] | none (no parent ids) | `sweep6-verdict-8b7aaead.json` |
| sweep 6 `r23_lpi01`, epoch 8 (incomplete) | 2,621 | −0.000315 | [−0.00365, +0.00302] | none | same |
| matched control (= `r23`), epoch 1 | 352 | −0.00107 | [−0.00335, +0.00121] | [−0.00437, +0.00198] | `j21-paired-verdict-98fa563d.json` |
| matched control, epoch 2 | 704 | +0.00399 | [+0.00119, +0.00679] | [+0.00006, +0.00947] | same |
| matched control, epoch 3 | 1,056 | −0.00321 | [−0.00600, −0.00042] | [−0.00746, +0.00043] | same |
| matched control, epoch 4 | 1,408 | −0.00163 | [−0.00414, +0.00088] | [−0.00492, +0.00124] | same; bitwise equal to sweep 6 `r23` at 1,408 |
| R25-kNN-CE, epoch 1 (KL stop) | 352 | +0.00012 | [−0.00189, +0.00213] | [−0.00331, +0.00303] | same; `r25-e1-verdict-de66cb6a.json` |

All records are under `.agent/rlfw-coord/lanes/G-JUDGE/`. Reading the table:

- The matched control's per-context interval is above zero (a regression) at epoch 2 and below zero at epoch 3. 263 contexts cannot separate effects of this size.
- R25's mean probe KL reached 0.1079 at epoch 1, about twice the 0.05 stop, so it froze at 352 updates.

**Training telemetry with no paired verdict** (full-validation L6 by epoch 1–4; frozen v1 is 0.8973272):

| Run | Delta from the matched control | L6 by epoch | Mean probe KL | Receipt |
|---|---|---|---|---|
| R26 (CX-OP) | dropout seed 20261026 | 0.8965543, 0.8989283, 0.8944658, 0.8951989 | ≤ 0.0106 | `lanes/CX-OP/r26-seedrep/TRAIN_EXEC_EXIT.json` `ea5c0a86` |
| R27 (G-ABL) | kNN-CE weight 0.05 | 0.8957436, 0.9002211, 0.8946769, 0.8951011 | 0.0191 → 0.0377 (alarm at epoch 4) | `4c3a3efd` |

Further seed replicates (R28, R29) started at 11:16–11:17Z.

**L6-1148**, a larger pre-registered set. G-JUDGE adopted it at 10:43:44Z (`lanes/G-JUDGE/l6-1148-prereg.json`, `5db24253`), before any paired look on it.
- Contents: val263 plus the 885 lineage-held-out TRAIN contexts that sweeps 4b–6 and every later fine-tuning run excluded (sweeps 3 and 4a had no holdout). That is 1,148 contexts in 221 parent clusters.
- Pre-registered looks: three checkpoints, cluster bootstrap, Holm correction.
- Conservative for a challenger: the held-out lineages were in v1's own training, so this is not an unseen-family test.
- Rows: CPU-evaluated per context for v1 and the three checkpoints. The per-term consistency check against the training-time numerators passed all 6,888 checks of each checkpoint under the amended tolerance (relative ≤ 1e-5 or absolute ≤ 1e-7), registered at 11:20:58Z, before the look. Under the relative test alone, 4 feasibility terms (absolute differences about 1.3e-8) failed.

**L6-1148 verdict** (`lanes/G-JUDGE/l6-1148-verdict.json`, `370b5e09`, 11:38:25Z). Differences are arm − v1 on the contract cost over all 1,148 contexts, with the 221-cluster bootstrap interval:

| Checkpoint | Pooled | Holm | Label | val263 (not decisive) | held-out lineages (not decisive) |
|---|---|---|---|---|---|
| R25 E1 | +0.00511 [+0.00413, +0.00611] | step 1, p 1/2000 | REGRESSION ALERT | +0.00119 [−0.00073, +0.00294] | +0.00628 [+0.00524, +0.00737] |
| `r23` E4 = control E4 | +0.00310 [+0.00195, +0.00434] | step 2, p 1/2000 | REGRESSION ALERT | −0.00070 [−0.00214, +0.00060] | +0.00423 [+0.00286, +0.00563] |
| R26 E4 | +0.00333 [+0.00214, +0.00461] | step 3, p 1/2000 | REGRESSION ALERT | −0.00108 [−0.00277, +0.00044] | +0.00463 [+0.00329, +0.00609] |

Reading:
- The L6 + 0.2·KL fine-tune gives no detectable change on validation lineages.
- It is clearly worse on the lineages that v1 trained on and the fine-tunes excluded. Both strata are reported and marked not decisive; read directionally, the pattern is consistent with forgetting, which the 0.2 KL anchor did not prevent.
- The verdict is scored non-qualifying: no claim, no seal, no promotion.
- It rules out "this objective beats frozen v1" for these checkpoints.

## Live decisions not taken

Three decisions belong to Jepson alone (R19 and R23 build-only; gap board rule 6), and none has been taken:

- **The 72 h live campaign.** Under R11 the window is fixed once at first launch, with at most 80,000 new updates. The CLI's live mode is for an owner-approved launch only.
- **The R9 drain:** executing and scoring v1's roughly 32.7k recorded proposals.
- **Any seal, promotion or serving change.**

R13's earlier "green gates suffice" permission does not apply now: R19 and R23 keep v2 build-only, and gap board rule 6 reserves these steps for Jepson. Green build gates do not stand in for that approval.

## CLOVER direction

`docs/internal/clover/CLOVER_VNEXT_FINAL.md` (2026-10-12) and its digest `CLOVER_VNEXT_SUMMARY.md` set the direction. The final document calls itself a research proposal, not an approved implementation or a demonstrated improvement. It splits the system in two:

- **CLOVER-Discover** is a qualified program-discovery engine: conventional GP, reproducible execution, independent semantic checks and controlled evaluation. It must be useful with no neural controller, and it is the foundational architecture.
- **CLOVER-Learn** is a learned evaluation-allocation controller. It is accepted only when controlled experiments show incremental value over Discover.

The final decision keeps PolicyLM, outcome distillation, edit-gain modeling and policy-gradient methods as candidates rather than commitments. PolicyLM is "one candidate model, not the assumed winner". **The v2 learner is therefore a CLOVER-Learn candidate.**

The hypotheses are tested separately:

- **H1:** GP repeatedly finds useful programs outside the 6,640-program trie.
- **H2:** a learned selector picks better programs than conventional selection from identical frozen pools.
- **H3:** fresh feedback beats frozen, offline-replay and restart-only controls.
- **H4 (conditional):** learned guidance improves GP generation itself.

If only H1 holds, the plan is to ship a verified GP engine and not claim an RL flywheel. Phases 0 to 5 are gates, not commitments, and there is no requirement to advance through all of them:

0. comparability;
1. useful support expansion;
2. ranking learnability;
3. fresh-feedback value;
4. guided discovery;
5. independent qualification and handoff.

The Phase 0 foundations merged in #296. The bounded GP expansion study is gap GB-06.

**The closed H1 pilot**, merged in #306 (`docs/internal/clover/GP_EXPANSION_STUDY.md`), is NON-QUAL. It compared three GP arms on the same TRAIN inner folds: ordinary, seeded from frozen-v1 policy draws, and seeded from the context's own HWM. 17 contexts were complete for all three arms.
- Own-HWM seeding beat ordinary in 16 of the 17.
- Frozen-v1 seeding versus ordinary was 10 better, 2 equal, 5 worse, with a median log ratio of −0.00035.
- 40 of the 51 champions are outside the 6,640-program catalog.
- 48 of the 51 passed bitwise native/emitted parity.
- 18 arms were interrupted, without retry.

H1 is unestablished. The flat result for policy seeds is the H4 baseline that a learned policy has to beat.

**Next gap: EI-1, the first expert-iteration round.** It is planned only (`.agent/proposals/claude-rlfw-20261009/ei-1/EI1_PLAN.md`).
- **Search** a fresh 48-context TRAIN roster. The roster is disjoint from L6-1148 stratum h (`EI1_DISJOINTNESS.json`) and from the H1, GB-06 and SP-1 rosters.
- **Qualify** the winners.
- **Freeze** a GB-05 successor support that gives them a finite trie likelihood.
- **Fine-tune** against a matched zero-weight control.
- **Judge** on L6-1148.
- **Re-seed** GP from the fine-tuned policy.

It needs GB-05 merged and owner sign-off before anything runs. SP-1 and GB-06 are owned by `jopedime_clover`.

Proposed amendment after the L6-1148 verdict (EI1_PLAN.md, Amendment 1; awaiting sign-off):
- EI-1 fine-tunes with the same base recipe, so arm vs v1 would mostly measure that recipe's forgetting.
- The primary becomes the EI-1 arm vs its matched zero-archive-CE control, so the base recipe cancels.
- Arm vs v1 is reported but not decisive.
- Forgetting mitigation is a separate, later pre-registered question.

The governing rule: no self-improving-RL claim until fresh feedback causally improves independently confirmed selection beyond the strongest non-learning baseline.

For v2 this has four consequences:

- L6 and the anchor KL are training and stability quantities, not success metrics. The CLOVER synthesis rejects L6 as the success metric.
- The 0.05 trie-KL HOLD is defined only on the current support. Once support expands, CLOVER requires a positive-support reference and a new, versioned stability protocol; the old trie KL threshold does not carry over.
- v2's same-start restart and frozen comparisons are inputs to H3, not evidence for it until matched controls run.
- A v2 live run, if Jepson approves one, would test a CLOVER-Learn candidate. It would not establish the flywheel.

## v1.0.0 (history)

v1.0.0 was the full-corpus PolicyLM continuation that ran on xbabe2 from 2026-10-09. Its original service was `jopedime-neural-continuous-20261009-v1`, with config `fec184217e7c…` and root `/home/ubuntu/jopedime_probe_exp/ofm-20261008/dl_final/flywheel_neural_continuous_20261009_v1`; it ended as the `recovery5` unit described under lineage. Its source is on `main` under `deploy/dl-flywheel/pipelines/continuous/`, added by commit `c9983160c` (2026-10-09).

**Model.**

- The native PolicyLM has 893,957 parameters: context projection plus causal prefix Transformer, width 128, two layers, four heads, feed-forward width 512, dropout 0.1, vocabulary 282 and maximum sequence length 640.
- Its only input is NVIDIA Kumo-Tabular Small: 2,051 values, which are 2,048 pooled hidden values plus three query diagnostics. Revision `3c3e10bbdb59…`; weight SHA256 `fc672f536501…`.
- Normalization means and scales were frozen from the original training corpus. Zero scales become one, and standardized inputs are clipped at ±10.
- Decoding samples model logits through a fixed trie of 6,640 programs derived only from original training teachers. It is restricted program selection; v1 did not claim arbitrary unseen GP topology generation.
- A macro-batch holds up to 16 training contexts, at most one per parent lineage. Micro-batches hold at most 8,192 padded tokens.
- Teacher windows rotate up to eight measured candidates plus an eligible winner per context.
- Each update emitted two proposals: one training context and one validation context.

**Objective.** The objective was the support-masked L6:

`L6 = CE + 0.25 quality_Huber + 0.25 pairwise_Bradley_Terry + 0.10 listwise + 0.10 feasibility_CE + 0.05 runtime_Huber`

Quality is `q = -0.5 * sum(log(max(GP_RMSE, epsilon) / max(mean_baseline_RMSE, epsilon)))` against the complementary-fold mean predictor, clipped to ±log(1000) for optimization. CatBoost-relative quality was not active.

**Training.** AdamW started at 3e-4 with weight decay 0.01 and gradient clip 1. The plateau controller halved the learning rate with patience 5, down to a 1e-6 floor.

**Data.**

- The original inventory held 14,987 data runs from 3,003 dataset versions, of which 12,581 were eligible.
- 7,120 had a complete teacher/input join: 6,857 training and 263 validation.
- The other 5,461 eligible contexts are an explicit join backlog.
- Roles were assigned per parent lineage: 1,068 parents, 1,014 training and 54 validation.

**Execution.** Every proposal was queued durably before resolution. Confirmed misses ran natively and as generated Rust on both inner folds, and had to agree within `1e-9 * max(1, abs(reference))`. Artifacts were create-only and verified by exact-VersionId GET plus SHA256, with the result record published last. Parity failures are explicit failures, never fabricated RMSEs. The S3 research prefix was `experiments/gp-policy-dl/v2-trainonly/` in `veox-jopedime-learning-use2`.

**Trajectory.**

- The first pass ran 429 updates over the 6,857 training contexts in 844.3 s: input preparation 36.6%, decoding 29.9%, optimizer 3.4%.
- The continuation reached 17,773 updates.
- The learning rate halved nine times between steps 2,477 and 5,677, then trained 12,096 updates at the 1e-6 floor.
- Full validation over 263 contexts fell from 1.0055 at epoch 2 to a best of 0.8968 at epoch 28, and ended at 0.8975 at epoch 40. It stayed within [0.8968, 0.8978] for all 31 epochs from 10 to 40.
- The 16-context fixed validation panel went from 1.3814 at step 429 to a best of 0.87052 at step 16,813, ending at 0.87121.
- On 32 development contexts the quality head's pairwise ranking of fresh programs was 0.506 [0.441, 0.568], which is chance.
- Matched win rate and Elo for the final v1 checkpoint have not been computed. The historical development-panel matches used a mid-training checkpoint and 32 kept contexts. Historical 13-context comparisons must not be relabelled as current-model results.
