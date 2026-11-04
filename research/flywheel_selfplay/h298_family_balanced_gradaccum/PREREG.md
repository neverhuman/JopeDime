# H298 (`h298_family_balanced_gradaccum`) — Preregistration

- **Wave ID**: `H298`
- **Slug**: `h298_family_balanced_gradaccum`
- **Owner**: `rd_argon` (`[ARGON]`)
- **Stage**: Preregistration pushed to `origin/v1/next` **before any H298 arm is scored**.
- **Preregistered UTC**: `2026-11-04T02:52:00Z` (deadline `03:40Z Nov 4`; closeout deadline `05:45Z Nov 4`)

## 1. Hypothesis

The 4 contexts that carry `r3175`'s lead (`H289`) reflect a few large families dominating the training gradient (in baseline `reporter.training_order(order_seed, 8192, False)`, the largest `TRAIN` family receives `252` presentations per update while 1-context families receive `6–7` presentations per update — a `42x` presentation skew across the `205` `TRAIN` families). Family-balanced sampling (`1/205` uniform across `TRAIN` families, then uniform temperature `1.0` over contexts within each sampled family) combined with `4x` gradient accumulation (`32,768` context presentations per optimizer step, accumulated over microbatches of `32` before `clip_grad_norm_(1.0)`) gives a lower-noise, family-unbiased update direction that changes more `dev91` argmin choices in the right direction.

## 2. Step 1 Regression Verification (Completed Before Any Arm)

Executed on an isolated copy at `xbabe1:/home/ubuntu/jope-data/H298-rd_argon/` (`CUDA_VISIBLE_DEVICES="" taskset -c 64-67 nice -n 19`) and recorded in [`regression_receipt.json`](regression_receipt.json):

- `round-3175.pt` SHA256: `1d6ea900ebed8d2a05c3b3f37576adc15aadd2b53c585bfd5cb71a103e9ff97b`
- `selected.json` SHA256: `981960ad628f5029f8c54552c7b1cb1d440072e7e9dcffecdea52c6a3493a0f1` (`1,219` `TRAIN` contexts across `205` families; `91` `dev` contexts across `19` families)
- Frozen DOPE SHA256: `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814` (`29,211` adapter trainable parameters only; frozen non-adapter state SHA256 `d4dde909145d9da75ec74fd61b9827d8fc2ec9236505b7cdf72b4161994551b7`)
- `dev91` evaluation identity: `e79a7715c1531ccb8c4055ab96971101cd4034ec3a71d83bca6d25b0e604ca83`
- **`r3175` replay**: `1.1389572403748425` (`91/91` program choices bit-exact match)
- **`t1845` control seed `184501` 2-update replay**: `1.1354387884248622` (`round(u, 10) == 1.1354387884`, `91/91` program choices bit-exact match, `adapter_sha256 == fd690ea216af7e93fe246e849e9f66a876d23306492cca3ad6866c57e28d56f1`)

## 3. Experimental Design & Arms (`seeds 29801, 29802, 29803`)

All three arms start from an identical copy of `r3175` (`round-3175.pt`), restore the incumbent `AdamW` state (`weight_decay=1e-4`), and apply the same seeded v2 adapter warm-restart (`warm_restart_adapter(net, opt, seed, rnd=seed)`) per seed `s in (29801, 29802, 29803)` so initial adapter weights, AdamW moments, and learning rate are strictly paired across arms. Each arm executes **`8` `TRAIN`-only updates** per seed (`SmoothL1 + 0.1 * KL(softmax(-target/0.2) || softmax(-pred/0.2))`, gate-false target `2.0`, microbatch `32`, `clip_grad_norm_(params, 1.0)` once per optimizer step, `dope_freeze_guard` verified on every update, no `dev91` gradients, no intermediate `dev91` selection, no rollback inside a trajectory). Each terminal candidate checkpoint is frozen and hashed before touching `dev91` **once** for terminal scoring.

- **PRIMARY (`FAMBAL_ACC4`)**:
  - Uniform sampling across the `205` `TRAIN` families, then uniform (`temperature = 1.0`) over `TRAIN` contexts within the sampled family.
  - `4x` presentations per update (`32,768` presentations = `1,024` microbatches of `32` accumulated with `weight = len(cc) / 32768` before `clip_grad_norm_(1.0)` and `opt.step()`), `8` updates per seed.
- **Secondary (`FAMBAL_ACC1`, descriptive)**:
  - Same family-balanced sampling (`1/205` uniform family, then uniform context within family, `temperature = 1.0`), `1x` presentations per update (`8,192` presentations = `256` microbatches of `32`, exact prefix of `FAMBAL_ACC4` at each step), `8` updates per seed.
- **Paired Control (`PAIRED_CONTROL`)**:
  - `t1845` baseline sampling (`reporter.training_order(order_seed, 8192, False)`), `8` updates per seed, same loss, microbatch `32`, LR, and grad clip `1.0`.

All batch orders and 192-candidate slates (`training_indices(pools[c], 192)`) for all 3 seeds and all 3 arms are pre-generated and pinned in [`preregistered_batches.json`](preregistered_batches.json) (`sha256 == 51e93dff787ee5d38f12fc41c2fb4b8021929ed59926a708922c5340bd359f95`).

## 4. Strict Decision Rule (Ties Never Win)

Primary (`FAMBAL_ACC4`) achieves **`DEV_ADVANCE`** iff on **`3/3` seeds** (`29801`, `29802`, `29803`):
1. `dev91` canonical mean `> 1.1389572403748425 + 0.001` (`> 1.1399572403748425`) **AND** `> its paired control (`PAIRED_CONTROL`)`.
2. Family-grouped (`19` `dev` families) `10,000`-resample cluster bootstrap `LCB95` (`2.5%` quantile) of paired per-context delta vs `r3175` `> 0.0`.
3. Positive gain is not concentrated: top-4 contexts carry `< 80%` (`< 0.80`) of total positive per-context gain vs `r3175`.

Otherwise the verdict is **`NOT_ADVANCED`**.

## 5. Required Reporting

- Per-seed and median-of-seeds lift vs `r3175` and vs `PAIRED_CONTROL` with 19-family `10k` bootstrap `95%` CI.
- Per-context `W/T/L` and changed-selection counts vs `r3175` and vs `PAIRED_CONTROL`.
- Top-4 positive gain concentration share per seed.
- MM Bradley-Terry Elo (`r3175 = 1500`) with 19-family bootstrap `95%` CI.
- S3 archival of checkpoints and full outputs to `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H298/` followed by verified deletion of `/home/ubuntu/jope-data/H298-rd_argon/` on `xbabe1`.
