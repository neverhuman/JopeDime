# [ASTRA] Code and data evidence

Investigation source: `ca4c64b36c38b9c3e3f450b09c93896c50d98ee0`, 2026-10-31 UTC. These are source/retained-evidence observations. No live worker, GPU, protected payload, or training job was accessed. The brief's runtime counts remain attributed to the brief/H56, not independently remeasured here.

## Actual execution closure

The fused runner inserts `/home/ubuntu/JopeDimeMicro/python` before importing `rust_evaluator`, then constructs the custody-pinned `cpu_exact_fused.py`. Reading the adjacent repository `model.py` or `selfplay_loop.py` alone does not describe this model. Vendored code provides stable reviewable copies:

| Component | SHA256 | Receipt |
|---|---|---|
| CPU-exact fused constructor | `b9f1468a61aa4dc888c8f2931f9c759f7d4f057793a77ccb05eed6300fb4f9ab` | `run_fused_closed_loop.py:298-307`; `custody/files/home/ubuntu/jope-data/res-01/input/cpu_exact_fused.py` |
| Micro evaluator | `53422343699172b61ed69ed707015558f7cf1a344e5449f30e1eb9d708aef900` | `custody/PROPOSER.md`; local read-only hash check matched |
| Micro model | `0aa76360a7e5e691e420bd5a88a05b343b5b83edcc645e5f50fb33a3d5756f5a` | `h58/prepare_cache.py:33-35`; local read-only hash check matched |
| Micro multimodal adapter | `7f5d30280de5cd47e8af5aa6787e5cb2bff321d984f9cae7d261eabd0a9c0d7a` | `h58/prepare_cache.py:35`; local read-only hash check matched |
| Program feature-name order | `c4e4373d1bc7e99bc38e44eb4acda22d0f9ecc9e332fa8cb9c6a0b463fe37962` | `custody/files/home/ubuntu/JopeDimeMicro/assets/projections/program-feature-names.json`; read-only hash check |

Unless otherwise stated, paths below are relative to `research/flywheel_selfplay/`. Long `cpu_exact_fused.py` and `phase12_multimodal_model.py` references mean the vendored custody copies above. Historical r782 custody does not certify the current live process.

## Loop, targets and information flow

1. `load_packet` enforces the 1219/91 full cohort, regression task, family disjointness, role membership, pool/table SHA and zero outer-row JTT1 envelope (`run_fused_closed_loop.py:94-155`). Full mode is not the older 16/8 training subset. Dataset rows and original labels are not rewritten by proposed learning changes.
2. Only non-anchor adapter parameters train. DOPE is checked against `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`; optimizer exclusion and post-step checks are explicit (`run_fused_closed_loop.py:313-330,1023,1128`). All proposals retain this pin.
3. `training_indices` deterministically puts the lowest-fitness 48 first; `predict` reveals the first four *slate positions*. `utility` scores full pools in storage order. The reveal-set mismatch is measured in existing RES-05 on 1219/1219 initial pools and 128/128 pre-proposal snapshots (`automatic_promotion.py:323-333`; `run_fused_closed_loop.py:411-481`; `custody/SURROGATE.md:1-19`). This is not proof of the size of its quality effect.
4. The fused loss uses raw passing fitness, failing cost +2, SmoothL1 plus 0.1 KL at temperature .2 (`run_fused_closed_loop.py:974-982`). Legacy `selfplay_loop.py:527-557` instead includes pairwise, gate and directional-evolution losses. Neither coefficient magnitude nor scalar loss decrease establishes useful ranking gradients.
5. Native utility returns `−fitness` when passing and −2 otherwise. `val_reg` is the context mean, higher better; family means are additional fields (`crates/jope-telemetry/src/hooks/flywheel.rs:207-256`). H58's `clip(1−regret/(2*spread),−1,1)` and family average are different metrics (`h58/runner.py:1054-1137`). Preserve raw `val_reg` and common-panel family Elo alongside normalized diagnostics.
6. Full mode overrides generic retry policy to retain the declared maximum two updates; rejects restore model plus optimizer, ties are accepted (`run_fused_closed_loop.py:220-223,874-878`; `dev_incumbent.py:472-499`). Each full round perturbs trainable parameters and selects a round-hashed log-uniform LR/noise scale (`dev_incumbent.py:681-728`). Persistent-learning experiments must distinguish removal of rollback, noise and averaging; H55 already disfavors a simple EMA explanation under its conditions.
7. GP continues after adapter rejection. It samples 16 training families, ranks 8/16 proposals, and evaluates at most four/context, with two random slots during exploration (`run_fused_closed_loop.py:1153-1255`; `dev_incumbent.py:607-616`). Only strict full-training-CV winners enter training pools, apart from historical admissions (`automatic_promotion.py:224-287`). Losing evaluations have score/tree evidence but are not used by this learning path.

## Additional mechanisms worth discriminating

**Slate dependence beyond reveal mismatch.** The executed constructor overrides candidate encoding. It subtracts the current slate's 24D semantic mean before forming a query and nonlinear multimodal attention (`cpu_exact_fused.py:651-703,746-748`). Training uses 192 top-enriched candidates; full-pool evaluation often uses 1152; proposal ranking uses 12/20 including seeds. A scalar output-mean subtraction alone cannot change argmin, but this *input* centering can change relative predictions. Test the same real candidate pair and fixed revealed history under different real distractor sets. Rank-change frequency and decision regret are unknown. H58 retains the same top-enriched 192-wide sampling, so its reveal repair does not isolate this mechanism (`h58/runner.py:444-525`).

**Prediction units.** Anchor residuals use F/scale; the adapter history also carries uncapped F/scale (`phase12_multimodal_model.py:1364-1412`; `custody/SURROGATE.md:51-59`). Current training targets raw F. This is a potential calibration burden, not a proof that the composed model has a single exact output unit. H58 changes scale to `max(std(fitness[:4]),.10)` while retaining raw-F targets (`h58/prepare_cache.py:273-296`; `h58/runner.py:1415-1425`). Test a unit-consistent residual loss without changing evaluation fitness; H59's loss reweighting is a different intervention. Require exact baseline parity first.

**A configured dropout need not execute.** `phase12_multimodal_model.py:1242-1246` has training modality dropout; the constructor's bound replacement at `cpu_exact_fused.py:651-703` does not. H58 copies that bound function (`h58/runner.py:239-244`). A constructor argument of `.10` alone is not evidence of stochastic modality masking. No claim that enabling it would improve quality.

**Priors and initialization provenance.** H58 resets anchor/adapter and program normalization from fit-family revealed anchors (`h58/runner.py:181-277`), but precomputes calibrated priors from frozen assets once (`h58/prepare_cache.py:127-143,198-215,273-296`). Fold-clean adapter weights do not by themselves establish fold-clean provenance of every inherited outcome-derived prior or normalizer. The prior estimator fits all its `strict_train` roster after a cross-fit audit (`phase12_multimodal_model.py:922-1000`). Actual overlap with H58 folds requires asset-training manifests; do not assert leakage from names alone. Keep DOPE frozen, and label any residual pretrained exposure honestly.

**H52 anchor is wired.** `h52/runner.py:213-216,301-319` adds λ||θ−θ0||²/(2·29211), so its gradient is λ(θ−θ0)/29211 and zero at initialization. Same hard choices under λ1/λ10 do not establish a disconnected loss. Compare gradient component norms if revisiting regularization.

**Program representation.** `rank_prepare.py:6-32` includes operator counts, positions, parameters and estimator identity; the 412-name asset contains 302 position features. A blanket claim that this is order-blind is false. Seeds are skipped explicitly. Any representation-collision wave first needs a real collision census and conflicting-outcome evidence; it is lower priority than known mismatches.

## Search/data affordances and limitations

`custody/PROPOSER.md` records family-excluding retrieval over the first 12 *raw SVD* coordinates, not the named raw landmark gaps. Generation is retrieval, then cyclic edits, then fallback synthesis; the output is not an independent uniform operator mixture. Outcome-conditioned retrieval or transition-value training is a prospective change, with equal fit budgets and explicit zero-filled/missing-outcome distinctions.

The HWM score is deterministic two-fold constrained log-RMSE cost at pinned worker bytes. Source comparison is arithmetic replay, not independent evaluation. `custody/SURROGATE.md` reports 512/512 matches, but also documents exact row/fold/source identities and non-independent adaptive observations. Additional resplits or rungs create a new comparison identity; never merge their scores as if they were original-F replicas.

`input_manifest.py:272-320,396,451-458` distinguishes metadata inventory from numeric training admission. Newly found S3 keys cannot enter training merely because they exist or share a dataset name. Join exact table/context/program/family/role, evaluator, source prefix/version and budget before any labels are used. Untouched families remain outside every training input closure.

## What existing waves already cover

- H57: direct-utility ES, six sigma/seed cells plus null, not another generic loss experiment (`h57/PREREGISTRATION.md`).
- H58: reveal/sampling and prior restoration, completed-outcome replay/hard negatives, listwise loss; the chained design does not isolate every subfix. It limits native generation to fast contexts in seven training families (`h58/PREREG.md`; `h58/runner.py:1373-1402`). A small fast-context result needs a separate broad-runtime qualification.
- H59: clipped/context-normalized loss from exposed r882 (`h59/PREREG.md`), with a stated fold-clean sidecar requirement for broader claims.
- H55: six completed averaging cells, no admissions and zero terminal DL-Val reads. SWA-8 had a small positive mean with zero lower bound; EMA-25 degraded. This conditional result is not proof that optimizer design never matters (`docs/research/studies/dope-neural-ranking-20261021/h55/README.md`).

The final wave list should contain a causal flywheel test: frozen vs updated ranker crossed with frozen vs fresh outcome memory, fixed acquisition budgets and terminal scoring. Without it, better oracle pools and more optimizer updates remain two separate improvements rather than demonstrated positive feedback.

## H58 prior follow-up

The cached H58 prior is the frozen 150D ridge `prior10` (`cpu_exact_fused.py:384-403,1121-1159`): median plus standardized semantic24, complexity6 and context-interaction120 features times fixed weights. Its historical fitting target was static `prior_cost`, not direct current candidate fitness (`phase12_multimodal_model.py:922-1000`). H58 loads it through `prepare_cache.py:127-143,198-215,273-296` and `runner.py:411-432`.

It enters the anchor effective prior additively, but is also subtracted from revealed F/scale residuals (`model.py:245-249,272-299,371-382`). The outer adapter receives a frozen/detached anchor score and adds multimodal, semantic-kernel and trajectory corrections (`phase12_multimodal_model.py:1321-1337,1437-1444`). A poor prior-only ranking therefore motivates an intervention; it does not alone prove domination of final rankings. The same-choice/component diagnostic and matched zero/ridge/learned-prior comparison are required.

Standalone persistent learning overlaps completed H48-PF and H51-STEPS. H48-PF minus control was −0.0150 [−0.0877,+0.0297] in its exposed development comparison, and H51 larger regression updates hurt continuity scores; their selection/exposure limitations remain. See `docs/research/studies/dope-neural-ranking-20261021/h48/README.md` and `h51/README.md`. Prior/tree and loss proposals must also acknowledge H30/H31/H34's different 850-context closure rather than claim those ideas were never tested.

## Source refresh before publication

A read-only diff against the later main commit `c63cc320983b722acac83a3bed2ec4fd30ee143b` (PR #744) found the strict-admission implementation: `DevIncumbent` now supports explicit `--strict-adapter-admission` plus a required finite nonnegative `--adapter-min-gain`. In this opt-in mode it accepts only candidate > incumbent + margin; defaults still accept ties. The change does not alter the reviewed loss, reveal construction, slate encoding or rollback mechanism. `FLYWHEEL_LOOP.md` explicitly says this flag does not authorize a live restart; this investigation does not establish live activation. Original line references above remain bound to the initial source commit rather than drifting with main.

ARGON's source/probe correction is accepted: the dataset builder sorts initial archive entries by program hash (`dataset_builder.py:216`), so 'seed anchors' in older notes means the four operationally revealed storage entries, not four chronological generation-zero programs. Every comparison must use the same declared history; an old r882 score cannot be compared to a newly changed history's best-known floor.

## Context feature provenance follow-up

At investigation source `ca4c64b36c38b9c3e3f450b09c93896c50d98ee0`, `h58/prepare_cache.py:83-89` rejects a JTT1 table unless `nq==0`; lines 239–248 check the selected context's `train_only_jtt1_sha256` before extracting features from that table. Lines 94–124 normalize its labels and compute frozen DOPE/landmark features. Construction is training-row-only per context.

The cache also contains DL-Val context vectors (`:193-195`). OPUS's `context_probe.py:12-14` loads whole `lm48_z` and `dope629` arrays, then its `selected["train"]` loop selects only DL-Train contexts for fitting and scoring. This is not literally zero access to dev cached features, although no new dev outcome scoring is reported. ASTRA inspected source only. Existing-cache reuse checks IDs and field presence (`prepare_cache.py:217-228`); a future admitted run must verify cache bytes against `cache_manifest.json` (`:389-415`) and independently audit inherited feature weights/normalization. Per-table row closure is not proof of fold-clean pretraining. A bundled landmark+DOPE improvement does not identify DOPE's separate contribution.

A metadata-only read of the 2,122-byte `~/jope-data/h58/cache/cache_manifest.json` found the expected packet, selected roster, fused source and unchanged before/after DOPE hashes. Manifest file SHA256: `66b1df1374242580d482b7d6f9b72b6fe8f65830f90f658ca662a698aa229897`. Its declared context-cache SHA256 is `75704e8279c19fee8a7843ccbea801e3cd4b67e0ee08d282b070ddae7ab82faf`. ASTRA did not hash or load that numeric cache, so this is a manifest declaration, not independent current-byte verification.
