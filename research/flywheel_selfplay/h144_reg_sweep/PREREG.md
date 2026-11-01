# H144 — prospective regularization sweep from r882

Owner: rd_astra [ASTRA]. Authority: `~/nudges/h144_rd_astra_t1940.md`.
Status: preregistration before all scored inference and optimizer updates. This study is exploratory research on repeatedly exposed DL-Val, not sealed-test qualification or serving authorization.

## Hypothesis and fixed arms

Stronger decoupled weight decay may reduce continuation drift and invalid-pick cliffs without losing ranking gain. Exactly three arms are executed, all starting from the same frozen checkpoint:

| Arm | AdamW learning rate | Decoupled weight decay | Steps | Batch/slate | Gradient clip |
|---|---|---|---|---|---|
| CTRL | 0.0001 | 0.0001 | 40 | 16/192 | 1.0 |
| WD_1E2 | 0.0001 | 0.01 | 40 | 16/192 | 1.0 |
| WD_1E1 | 0.0001 | 0.1 | 40 | 16/192 | 1.0 |

**DROP_00 and DROP_30 are removed prospectively, before preregistration.** There are no dropout modules in the non-anchor adapter. Both adapter attention modules have dropout zero. Although `modality_dropout_p=0.10` exists, `cpu_exact_fused.py` installs `_sync_free_encode_context_and_candidates`, which bypasses the original functional modality-dropout branch. The H140 recipe also leaves the constructed adapter in eval mode. Changing that unused attribute cannot test dropout; enabling new stochastic code or changing frozen DOPE is outside this dispatch. Constructor-only `smoke.json` verifies the actual module census and effective encoder before merge. No DROP outcome will be computed. Keep the explicitly requested **Holm family size four** conservatively, treating the two unexecuted hypotheses as p=1.

The exact H140 control loss is SmoothL1 (beta=1), plus 0.1 target-to-prediction listwise KL at temperature0.2. Target cost is passing fitness clipped to [-4,2], otherwise2, matching H140's H59 target function. This is not a hinge objective. Predictors use zero context584/base prior/SOS, scale0.4208, frozen10x32 tokens, and first four slate positions as revealed fitness/gate history. Eval mode is retained throughout, matching H140; gradient computation and AdamW updates remain enabled only for47 named non-anchor adapter tensors (29,211 parameters). No L2 penalty, projection, early stopping or adaptive hyperparameter change.

Fresh seeds:144101,144102,144103. Fresh family fold seed144140. Sort205 DL-Train family IDs, NumPy default_rng permutation, split into five41-family folds. Fold0 has249 contexts; the other four contain970 fit contexts. Fold0 is descriptive only. The manifest records the exact assignment.

Use20 rounds x2 steps=40 per cell; H140 paired stream hashing and candidate sampler consume all eight candidate draws per round and retain the first two. Batch16 contexts with replacement, slate192 (first48 fitness-ranked candidates plus144 sampled from the remainder); every arm shares the exact context/candidate plan and Torch seed per step. Forty sequential updates x3arms x3seeds =360 updates total. Every cell starts from r882 with a new optimizer. No cached-cell reuse or implicit restart; an existing attempt directory/started ledger is refused.

## Input identities and boundaries

Checkpoint: owned copy `/home/ubuntu/jope-data/h144/inputs/round-882.pt`, SHA256 `0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64`. Source was the dispatch-authorized H142 copy, never the live loop.
Reference: owned `best_dev_checkpoint.json`, SHA256 `30853cad78d860870338316dfb5a4e869e46dc5c22d42fe4ab9d5f24f31c373a`.
Evaluation identity: `e79a7715c1531ccb8c4055ab96971101cd4034ec3a71d83bca6d25b0e604ca83`.

Reuse the real-S3-derived H50 selected train+val packet (`981960ad...`), source manifest (`99a53569...`), per-context SHA-pinned pools, fused source (`b9f1468a...`) and H58 token cache (`75704e82...`). Full hashes, constructor assets, source files, fold and preregistration are pinned in [manifest.json](manifest.json). [input-provenance.json](input-provenance.json) binds the retained S3 custody/catalog chain. These are the authorized1219 DL-Train/91 DL-Val contexts,205/19 disjoint families; no outer-test/holdout/sealed/meta-test records are admitted. Audit hooks allow only the exact selected data paths and owned output directory and refuse protected/live paths and writes to inputs. Constructor assets are frozen model assets. Broad S3-prefix scans or protected object fetches are forbidden.

The source manifest contains exclusion *counts*, not excluded payloads. Allowed cache CIDs must equal the selected train+val union. Every pool byte SHA is checked before decode. No validation contexts enter minibatches. There is no scored validation-dependent change to arms, steps, seeds or parameters.

## Regression and frozen-state gates

Before **any cell**, r882 must reproduce all91 reference native utilities with exact zero maximum per-context error and exact zero mean error (approximately1.1377942). No tolerance relaxation/reordering or fallback. Inference chooses native argmin cost, higher utility is better; failed gate =>-2. Baseline fails => INCOMPLETE, zero updates.

Before/after every optimizer step and after inference, compare the complete non-adapter state (including frozen anchor, normalizers and buffers) by raw-byte digest with names/dtypes/shapes; abort on any difference. Require canonical DOPE `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814` throughout. Log actual updates, loss, adapter SHA and L2 drift for every cell; frozen-state hashes and source/input/arm/seed/batch-plan identities are retained. Zero/unchanged adapter update is a failed cell, never successful training. Save all nine adapter arrays outside Git for replay custody.

## One consistent primary estimand and inference

Primary is equal-family native argmin utility on19 DL-Val families: mean contexts within each family, then mean19 families within each seed, then median over the three seed gains. Report gain versus r882 and paired same-seed CTRL. The1.1377942 baseline is the familiar *context* mean; it is not incorrectly substituted for the equal-family baseline when computing gains.

Use10,000 paired family-cluster bootstrap draws, RNG144141. For each draw sample19 families with replacement, share those sampled family IDs across all three seeds, recompute the equal-family gain in each seed, then take the median across seeds. The point statistic, CI and p-value use this same median estimator. Report two-sided percentile95% CI (0.025,0.975). One-sided bootstrap tail p=(1+count(draw median<=0))/(10000+1). Apply Holm step-down across the two executed and two prospectively dropped p=1 hypotheses; report monotone adjusted p. This is an exploratory bootstrap tail test, not an independent confirmation sample.

ADVANCE for an executed non-control arm iff all gates hold:
- median seed equal-family gain vs r882 strictly>+0.001;
- two-sided95% bootstrap CI lower bound strictly>0;
- at least2/3 seed equal-family gains strictly>0;
- at most80% of total positive family gain is concentrated in the top<=3 families;
- Holm-adjusted p strictly<0.05.

For concentration, select the seed whose scalar gain defines the median (ties break by ascending fixed seed). Its family gain vector sums to the point estimand; use positive components only for top3/total-positive share. No positive gain => share1 and reject. Tied utilities do not count as wins; zero margin/LCB fails. CTRL comparison is reported, not an undeclared additional gate. Different adapter hashes do not establish independent data. Missing any of nine cells => INCOMPLETE, never NO_ADVANCE. No valid arm => NO_ADVANCE. ADVANCE only produces a candidate adoption note; no live promotion.

Report invalid-pick counts separately for baseline and each seed/arm. Report paired family Elo relative to r882=1500, exact family ties worth0.5, wins/losses regularized with0.01 each, median across seeds and the same family bootstrap95% interval. CTRL-relative Elo uses the same1500 reference convention and is explicitly labeled. No AUC.

Secondary: the held-out DL-Train Fold0 gets the same fixed native utility contrasts, invalid counts and Elo, strictly descriptive. No secondary statistic changes the primary decision.

## Execution, resources and closeout

Only xbabe2 CPU; supervisor waits for H142 PID1137290 to exit before even the constructor smoke. Nice19 for supervisor/child; one numerical thread. Poll every0.25s, summing total OS threads and RSS of supervisor plus worker and descendants: threads<=4, RSS<6,000,000,000B. Disk available>=40,000,000,000B; owned scratch<3,000,000,000B. Check before launch, during imports/training/inference/statistics. SIGSTOP the owned worker group when load1>85 or MemAvailable<16GiB; SIGCONT when load1<80 and memory recovers. Record pauses; never signal other work. No GPU, qualification lock or live-directory write.

Targeted admission checks: data-free unit controls, Ruff, exact hashes/JSON/links, constructor-only smoke with zero scored contexts/updates, and `git diff --check`. Same-agent exact-head review and read-only ARGON/OPUS cross-review; owner authorizes squash-admin merge after targeted checks. No claim of full hosted CI. Run only byte-verified merged source, once. OOM/disk failures are INCOMPLETE; at most one explicitly recorded retry allowed by dispatch, same frozen protocol, no silent cached cells.

Closeout includes README hypothesis/setup/outcome/metrics/verdict/next step, metrics<100KB and one H144 index row. Create-only S3 custody at `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H144/astra-t1940/`, exact-version SHA readback and archive-member verification precede deletion of owned bulk outputs/input copies. Original shared H50/H58/H142 data are retained. Preserve incomplete receipts honestly; never infer unknown outcomes as zeros.
