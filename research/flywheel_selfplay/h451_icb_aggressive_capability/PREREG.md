# H451 — clean ICB_v1 capability battery on the H450 aggressive all-data fine-tune (pre-registration)

- **Owner and authority:** cl_fwriter (Claude, claude-opus-5-5).
  - Jepson in the cl_fwriter session, 2026-11-12 about 19:00Z: "Fix list + clean battery (Recommended)".
  - The approved joint plan, step 4.
  - The claim receipt is `ns/receipts/H451_claim.txt`.
- **Question:** Does the H450 aggressive fine-tune (pub-r5, maximum-entropy random splits on every non-sealed regression table, dev91 included) keep the in-context capabilities that ICB_v1 measures?
  - ICB_v1 is a synthetic battery. It contains no table from any bank, so it is the one clean check left after H450 put FIT-SEL, P201b, APEX-DEV and dev91 into training.
- **Code:** the frozen H431 stage `~/apex-stage/h431/pub-r3`, used unchanged.
  - CODE.sha256 prefix is `f9df082d1800cad9`.
  - The command is `icb_run.py battery` on EMA weights, run through `icl_model.ICL`, which is state-dict compatible with ICLH450.
  - Every checkpoint is scored with the same forward.
- **Input:** `ICB_v1.npz`, sha256 `492bb143672958247b65fef72adafa99b327766f293c7cf7b5cebd5f562a58a5`.
- **Compute:** CPU only on xbabe2 cpus 64-79, nice 19, `CUDA_VISIBLE_DEVICES=""`.
  - M: all 340 tasks, one single-thread shard per cpu.
  - L: the 68 rep-0 tasks (`--rep0-only`), 4 shards × 4 threads, as H431 scored L_h018.
- **Outputs:** create-once, read-only, under `~/jope-data/cw-battery/h451/`.

## 1. Entrants
- `init_L` and `init_M`: the H450 init checkpoints, H449 `ckpts_v2/all_non_d91_{L,M}/ckpt_unfair_final.pt`.
  - L sha256 `b854b518b07b2c243bc62069b36e9cbd05b0a5cd77ca5faba60b5347623a3a29`.
  - M sha256 `5aadefdfec053ac7a955381194bfb233d0ce8341f413b4d0d89699140715d4a0`.
- `final_<run>_ep<NN>`: H450 pub-r5 epoch checkpoints `ckpt_epoch_NN.pt`, or the final checkpoint.
  - There are at most 4 per size. The latest epoch of each seed-A run comes first.
  - Each is compared with the init checkpoint of the same size.

## 2. Metrics
Computed by `summarize_h451.sh` (jq), with the CW-4 `readout_cw4.metrics` definitions minus the monitor metric:
- K = mean of the family means of cap over B3, B4 and B5.
- B9 and B10 are family means.
- L = B1 tasks with noise 0.
- P = mean u8 over noise-free tasks, excluding B12.
- S6 = mean of (L, B3, B4, B5, B9, B10).

Readouts:
- The primary readout is `ridgeres`, the way the ICL column is used in the stacks.
- The `full` readout is reported with the same rule.
- M uses all 340 tasks. L uses the 68 rep-0 tasks.

## 3. Noise unit and rule
The noise unit is n_k = |m_k(M_h020) − m_k(M_h024)|, from the stored H431 batteries, for each metric and readout, on the same task subset. It is computed before any H451 run:

| subset / readout | K | L | S6 |
|---|---|---|---|
| all / ridgeres | 0.0110 | 0.0003 | 0.0095 |
| all / full | 0.0100 | 0.0048 | 0.0110 |
| rep0 / ridgeres | 0.0557 | 0.0133 | 0.0138 |
| rep0 / full | 0.0604 | 0.0828 | 0.0285 |

The threshold is thr_k = max(0.05, 2 · n_k). Per pair (final vs init, same size, same tasks), on `ridgeres`:
- `CAPABILITY_LOST` if K, L or S6 drops by more than thr_k.
- Otherwise `CAPABILITY_KEPT`, or `CAPABILITY_KEPT_S6_GAIN` when ΔS6 ≥ thr_S6.
- `INCOMPLETE` (no verdict) unless both runs have every task with `ok`.

B9, B10 and P are reported descriptively. There is no p-value: the tasks are fixed and the rule is a tolerance band.

## 4. Gates
- **Determinism:** `run_h451.sh det` re-scores shard 0/34 of the stored M_h024 battery (10 tasks, 1 thread) and compares every task file with the stored one, wall_s removed. All must be identical before any measured run.
- **Preflight:** `run_h451.sh battery` refuses unless this PREREG and both scripts match their published bytes at the PREREG commit, the H431 stage and ICB_v1 verify, and the checkpoint sha256 matches.

## 5. What this wave does not do
- It opens no table, no monitor bank, and no DEV, CONFIRM, DL-Holdout or dev91 file.
- It uses no GPU.
- It changes no H450 file.
- It is a capability check, not a ranking.
