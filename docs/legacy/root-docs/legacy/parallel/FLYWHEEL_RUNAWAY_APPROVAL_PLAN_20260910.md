# Flywheel runaway plan — approve before the next scientific deploy

Status: **Phase A live** (Codex hold/cadence manifest `779dacc4…`, independently replayed `held` at 14:05 UTC). Serving stays recovery `c2f229c9…`. GP PID 774452 unchanged. Do not lift the hold. Next work is Stage D from existing ledgers, not another train.

This is not a rewrite of GP as MCTS and not a request to open sealed tests.

## 0. What is actually running

| Surface | Identity | Verdict |
|---|---|---|
| Serving | recovery train-2 `c2f229c9…`, `rolled_back_to_recovery` | Correct. Do not walk it. |
| Train binary | native-v3, batch 256×8, no `--replay-policy` | Sound checkpoint chain from recovery |
| GP | native-v5-cpu, 104 workers, ~2300 queued, 1000/600 matched RNG | Leave up |
| Campaign | Codex combined-v2, `automatic_training=true` | Collects GP; wastes GPU on failed refits |
| Corpus | teaching shards, `reference_v1`, ~8.87M × 532-byte records | Incremental pack works (6–18 rebuilt / ~7870 reused) |
| Hyperion gate | empty `cuts: {}`, fail-closed `raw` | Jobs report a view; they do not teach cell |
| Promotion | 41/41 `inconclusive` (`manifest=None`) | Paper gate never fires; diagnostics reject |

Cycles **082–100** all fail both frozen guards versus recovery NLL 1.394498 / BT 0.533676. Typical latest: NLL +0.6–1.0%, BT +1.2–3.2%. Frozen monitor of cycle-000100: NLL 1.40475 / BT 0.54741, `training_selection_used=false`, `independent_evaluation=false`. Journal: 19 `diagnostic_rejected` in the last 400 events. Cycle-000101 completed 3000 steps from recovery on snapshot-000408 while later snapshots kept landing.

This is **not** a flywheel. It is GP collection plus repeated warm-starts that lose ranking.

## 1. AlphaGo Zero / AlphaZero — what to steal, what not to

Silver et al. 2017 (AZ) and 2018 (AlphaZero) are a closed loop: **prior → search improvement → train on the improvement → replace the prior only if the new net wins a fixed evaluation**.

| AZ mechanism | Honest mapping here | Do not do |
|---|---|---|
| Policy prior *during* search | 80 neural seeds **and** mutation/crossover parent weights aligned to **ranked survivors** | Treat 80 root seeds as if they were MCTS |
| Train on search-improved targets π | Weight / sample programs that beat the **neural seed and the teacher**; keep near-misses at lower percentile weight | Relabel GP visit counts as MCTS π |
| Value target z is a finished game | `value = 1/(1+rmse/median)` is a within-group score, **not** a fixed outcome | Add a new loss term first |
| Replace best only after a fixed eval panel (AZ 2017, ~55%) | Frozen NLL +0.5% / BT +1% **and** 16-cut `nseed<t` / exclusive-wins probe **and** the paper 1% PURE RMSE gate | Copy AlphaZero 2018 “always use latest” — our latest is worse |
| Sliding recent buffer | Teaching-store shards already avoid rewriting 7.8k clean groups | Split the 532-byte record format (new corpus identity) |
| Self-play vs current best | Net proposes → GP searches that cut → better programs become teachers → net trains from **serving**, not from the latest loser | Serve a rejected block |

Expert Iteration is the closer paper: the search is the expert improvement operator; the net is the apprentice. Our expert is 1000/600 GP, not MCTS. Keep that.

**Closed loop we want after approval:**

1. Serve only a net that beat recovery on the frozen panel **and** the 16-cut probe.
2. Propose from that serving checkpoint (DL → GP kernel).
3. Run a few matched generations chasing **train-cut CV / teacher RMSE**, folds respected (`meta_train` / `meta_validation` only).
4. Pack only new first-label programs; do not treat dirty-shard rebuilds as “teachers improved”.
5. Train the next net on **search-beating** groups once ranking-BT is at or below recovery. Near-misses stay in the group with `percentile^4` weight.
6. Promote or reject on the **same frozen panel**. Repeat.

## 2. Audit — issues ranked by certainty

### Sure, already corrected in source (not live)

1. **`rebuilt_groups >= 1` was treated as teacher improvement.** Pack dirties a group when *any* new program is inserted. `reference_v1` still recomputes every percentile and `value` in that shard (median can move). That is label thickening, not a new teacher. `teachers_improved` now requires `reference_training_growth >= 1` only.
2. **Ranking `score` was written into `policy_logprob`.** Those are not log-probabilities. `exp(score)` would scramble parent selection if a GP binary ever read them. Removed.
3. **`policy_weighted_parent` indexed seed-population order into `ranked[]`.** Different programs. Workspace `gp.rs` now weights **ranked genomes**. Do **not** swap live GP 774452 for this; current production seeds have no logprobs, so live search RNG is unchanged.
4. **No brake after repeated frozen-guard failures.** Plan-exec now holds automatic train after 3 consecutive `diagnostic_rejected`. Live combined-v2 does not. That is why 8 A100s keep burning 3000-step blocks.

### Sure, live, do not “fix” by mutating serving or GP

5. Serving is correctly recovery. Cycle-000100 was correctly `diagnostic_rejected`. On-disk shards are consistent (`reference_v1`, incremental reuse). **No torn pack, no serving drift.**
6. `--pause` while `pack`/`train` children exist **kills those children**. I did not pause. A hold must be a `training_work` return, not a global pause, unless GPU children are empty.
7. Paper `promotion_gate` request is always `manifest=None` → `inconclusive`. That is why 41 candidates never serve. **Do not weaken the paper gate.** Diagnostic accept is the flywheel close; campaign serve stays the paper contract.
8. In-run validation shares adaptive labels (`independent_evaluation=false`). Do not call frozen-panel NLL “independent evaluation”.
9. Compact quality-report winner keys mix identities (reporting reviewer r1). Do not treat compact `nseed<t` as the paper gate.

### Measured, not yet a licensed intervention

10. Claude’s first-visit read: ~5.9% of new programs beat the incumbent; median challenger ~0.17% worse; 72.8% of searches within 1% of incumbent; neural vs generic vs shuffled not significant; ~11% timeout waste; `histgbm2`+`gpr` dominate successful wall. Gens 29–48 are recovery re-runs on the same frozen board.
11. Strict 72-cut replay failed frozen NLL because **none** of the 226 frozen-panel validation contexts appear in that 72-cut corpus. Full recovery support (6836 train cuts) is the matched panel. Do not train the next challenger on the 72-cut subset and expect frozen NLL to move the right way.
12. `enrich_neural_seeds` is live and **unqualified** (Codex 74). Cross-cut teachers / subtree seeds are a reasonable prior *as extra seeds*, not as new grammar tokens. Hold further enrichment changes until a matched-arm receipt exists. **Never add `macro_*` tokens** (`vocabulary_sha256` invalidates every checkpoint).
13. `search_beating_training` is false. `SEARCH_BEATING_SAMPLER_ID` is receipt-only until ranking-BT is stable.
14. GPU: ~57–70% kernel during train, ~23 GiB / rank, hour occupancy ~72%, kernel mean ~39%, handoff p95 ~59s (observer fail). Pack I/O is not the bottleneck. Residual tax is controller cadence and **training blocks that cannot promote**.

## 3. Loss, labels, packing — what would actually unlock the loop

Current train loss (keep the weights; do not retune first):

`L = NLL + 0.25 BT + 0.10 listwise + 0.10 Huber + 0.05 ablation-BT`

Stability and `loss_log_vars` stay ×0.

**Label problem:** even under `reference_v1`, adding a program rebuilds the shard and rescales everyone’s percentile and `value`. AZ’s z does not move after the game ends. Near-misses should stay in the comparison group, but **old programs should not have their targets rewritten** just because a worse or equal program arrived.

Approval options, in order:

1. **Hold 3000-step trains** after 3 frozen rejects (source ready; deploy at a no-child-GPU boundary). GP + pack continue.
2. Pack timing grows a real `new_teacher_beats` count (new program RMSE < prior teacher). Train only when that is ≥ 1, not when `rebuilt_groups ≥ 1`.
3. Optional later: freeze percentile/value for already-admitted programs; only the new row gets a first label. New corpus identity. Qualify before any serve.
4. **Search-beating sampler** only after frozen BT ≤ recovery×1.01. Sample groups where final GP beat both neural seed and teacher. Near-misses remain as the seven comparators.
5. Uncontended 256 vs 1024 microbatch **after** (1) so we are not timing a doomed refit. Prod stays 256 until that bake.

Do **not** split tokens/labels into a second stream. `records.bin` is already 532-byte packed and GPU-resident. Incremental shards are the world-class part we already have.

## 4. GP engine — priors without breaking RNG

Keep 1000 evaluations / 600 seconds and matched-arm RNG.

Use other population leaders as **seeds**, not as grammar:

- Same-family cross-cut teachers (already in enrich; unqualified — hold at current behavior until a receipt).
- Frequent parenthesized subtrees as extra seed genomes (same).
- After a **new** GP binary is qualified on the node: populate `policy_logprob` from `score_pool` **by genome**, so `policy_weighted_parent` actually weights the top-10 RMSE survivors.

A cheap “few generations chasing train CV” loop, once serving is allowed to move:

1. Neural propose from serving.
2. One matched triplet generation (neural / generic / shuffled).
3. If neural exclusive wins or `nseed<t` rose on **train** cuts, enqueue one more generation seeded from those winners.
4. Stop at 2 extra generations or when train-cut teacher RMSE does not improve. Validation folds are never used as training targets.

Timeout waste (~11%) is the next GP efficiency cut: do not pay a full fit for a projection duplicate (already rejected) and consider a cheap policy reject only after policy scores are real.

## 5. Hyperion

Two different objects get the same name:

- **512-d Hyperion** is already concatenated into the 2048-d neural context. Leave it.
- **Tabular `hyperion_cell` vs `raw`** is a per-cut feature view. The lift is real on some cuts in `scripts/hyperion_cv_search_protected.py` and absent on others.

Smart flag (already coded, empty gate): fail closed to `raw`. One generation `hyperion-view-report` before any cell view teaches. No dual context assemblies. No fourth arm. Fill `hyperion-gate.json` only from a frozen train-only CV receipt, cut by cut.

## 6. Breadth, folds, training-set quality

- Respect `meta_train` / `meta_validation`. Sealed / final tests stay closed (`final_tests_opened=false`).
- Frozen diagnostic panel is 2048 appearances / 226 validation contexts / 8 families. Those contexts are **not** in the 72-cut strict corpus. Challenger training must keep recovery-scale support or we will keep failing NLL for a support reason, not a learning reason.
- Best training set is not “all 8.87M rows equally.” It is: champion + seven comparators per group (already the sampler), with **fresh mass** on programs that beat the previous teacher, once BT is stable.
- Checkpoints: nominate `validation_best` (already in live finish_task). Frozen monitor must set `training_selection_used=true` when that checkpoint is the one evaluated. In-run val NLL is **not** the AZ panel.

## 7. Phased approval (execute only what is checked)

### Phase A — stop lighting GPUs on known losers (ready)

- Deploy a **separately versioned** controller: consecutive-reject hold (3) + growth-only `teachers_improved`.
- Same scientific gate, same paper promotion contract, same GP binary, same serving.
- Qualify exact source; restart campaign only when no `train`/`proposal`/`monitor` child is running. Pack may finish first. **Do not `--pause` while those children exist.**
- Hold `enrich_neural_seeds` changes and `search_beating_training` off.

Success: no new 3000-step block starts while `consecutive_diagnostic_rejects ≥ 3`. GP queue keeps draining. Serving unchanged.

### Phase B — audit pack + cadence (operational, Codex-owned)

- Keep incremental shards. Persist `packing-timing.json` (already live).
- Review Codex scheduler-cadence (start ready GPU work before the 9–10s snapshot scan; subtract tick from the 10s sleep). Fresh exact-source qualification.
- Bounded 104/312 refill stays.

Success: hour assessor can pass occupancy without calling that “learning.”

### Phase C — prior during search (new GP binary, node-built)

- Ship the ranked-genome `policy_weighted_parent` fix.
- Populate logprobs from `score_pool` by genome.
- Matched-arm RNG tests must still pass when logprobs are absent (uniform).

Success: neural exclusive wins and `nseed<t` on the 16 val cuts stop falling, then rise, versus the serving generation. Final GP RMSE on those cuts no worse than serving.

### Phase D — search-beating teacher + Hyperion cuts

- Enable `--replay-policy` / `search_beating` **only if** frozen BT ≤ recovery×1.01.
- Fill Hyperion gate from the existing CV screen; cell jobs teach only after the generation report.

Success: a challenger beats recovery on frozen NLL **and** BT **and** the 16-cut probe. Then, and only then, the paper gate may run.

### Phase E — few-generation CV chase (the actual flywheel)

- After a diagnostic-passing serve: 1 + at most 2 extra generations seeded from winners, train-cut CV only.
- Pack `new_teacher_beats`; train the next net from that serving checkpoint.

Success: serving walks forward once, then the next block is trained from the new prior, not from a stack of rejected latest.json files.

## 8. Non-goals

- Stop or replace native GP / v44 neural.
- Open sealed tests or rewrite gens 00–20 receipts.
- Change `featmath` vocabulary identity / add `macro_*`.
- Full MCTS, new loss terms as the first knob, dual Hyperion encodings.
- Call frozen-panel NLL independent evaluation.
- Forge `#replay*` family names.
- Swap resident/train to native-v5 GPU binary without qualification (GLIBC / SHA).

## 9. Ask

Approve **Phase A** as the next live deploy. Phases B–E stay source-versioned until A is live and the next frozen monitor is another reject or a real pass.
