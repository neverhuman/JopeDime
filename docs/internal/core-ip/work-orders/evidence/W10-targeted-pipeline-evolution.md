# W10 — targeted pipeline evolution (a CPU witness adjacent to W10-T01)

## 0. Status, scope and source identity

| Item | Value |
|---|---|
| Task | Adjacent to W10-T01 ([`W10-learning-transfer-studies.tasks.md`](../W10-learning-transfer-studies.tasks.md)). **This is not the W10 real-loop witness, and it closes no W10 task.** |
| What it shows | The product's CPU pipeline drives one chosen dataset to better GP programs over rounds on fixed cuts. Every step re-derives from its receipts. The learning store records the improvements, and the champion bank keeps the improved champions as targets for later rounds. |
| What it does not exercise | A proposal policy, corpus, teacher, training update, checkpoint, export restore, S3 destination, three-node topology or no-update training control. W10-T01's operational witness requires all of these. None is claimed here. |
| Learning-store status | Pipeline HWM records are unbound, not replayable, not benchmark-valid and never `train_oof`, so they are **never teaching-eligible**. Nothing here is `live-qualified` or a production activation. The only `CI-verified` part is the e2e-smoke mini-lineage, once hosted CI runs it. |
| Source | Worktree `.agent/worktrees/bench`, branch `rr2/gp-evolve`, based on `origin/main` `cee1fc8bd242688972c352133847793c6927279f`. The demo binaries were built at `888b61387bec182c9fc33b1eeb4d4b1fc5ca0ca4`: `jope` sha256 `2f185e2b…70cb`, `jope-learning` sha256 `43550d71…1957`. |
| Toolchain observed | rustc 1.97.1 (8bab26f4f 2026-07-14), release profile, CPU only. |
| Run receipts | `/home/ubuntu/jope-rescue-20260921/regression-remediation-20260926/claude-w0-w12/runs/p5-*`. Each run holds HEAD, status, the diff, the command, the log and the exit code. See section 5. |

Progress words follow [`EXECUTION-CONTRACT.md`](../EXECUTION-CONTRACT.md).

## 1. What exists (`cee1fc8bd..rr2/gp-evolve`)

| Commit | Capability | Named tests |
|---|---|---|
| `cd3a3c946` | `jope pipeline run --seed-genomes <jope.pipeline-seeds/v1>` and `--incumbent <receipt>`. Seeds are canonicalised, deduplicated and capped at the elite count. They enter generation 0 after the bootstrap with origin `incumbent` and are re-measured on this run's cuts. Receipt v3 `search.seeding` records their sha256s, the parent receipt sha256 and the seeds-file sha256. v1 and v2 receipts still validate. `verify-receipt --evidence` re-derives a receipt from its run directory. | `run.rs`: seed in gen 0 as incumbent; same seed and rng replay; champion never worse than its seed; a tampered parent is refused before anything is written; a budget that cannot re-measure the seeds is refused. `seeds.rs` (3). `schema_agreement` (v2/v3). |
| `feabdbacc` | `--threads N`: a generation's admitted candidates are evaluated on a dedicated pool. Admission and the fold of results happen in slot order, so the thread count never changes a decision. `--cpu-ceiling <millicores>`, `sched_getaffinity` and `available_parallelism` bound the workers. | Byte-identical runs at 1 and 8 threads (frozen clock). Identical decisions under the wall clock. Oversubscription refused. |
| `38b6dbda4` | `--islands M --migration-interval K`: M random streams over the same cuts. Every K generations, each island's top-1 enters the other islands' elites only if it beats one of them. The pure `islands::next_seeds(champions, k)` is used both for migration and for round-to-round seeding. Receipt v3 `search.islands` records the migrations, and they are re-derived. The worker count is `islands × threads`. | `islands.rs` (3); island replay at 1 and 4 workers; one island equals the plain search; a migrant enters only an island it improves; islands × threads above the ceiling is refused. |
| `939a3b24c` | `--source-commit <sha>`: recorded as `git_sha`, and `git` is never spawned. SIGTERM/SIGINT stop between evaluations: no receipt is written and the exit code is 143. `tests/dependency_fence.rs` shows jope-pipeline depends on neither jope-learning nor jope-parallel. | `pipeline_cli`: a fake `git` is never called; the real binary is killed mid-search. `dependency_fence` (2). |
| `ea79139d6` | `jope pipeline evolve --rounds N [--control] [--seeds]` runs on identical cuts. `lineage.json` (`jope.pipeline-lineage/v1`) is rewritten atomically and fsynced after every completed round. `jope pipeline lineage verify [--require-discovery]` checks it. | `lineage.rs` (6): a 3-round lineage plus control chains and never worsens; tampering is refused; 1 and 8 threads write byte-identical lineages; a stop keeps only completed rounds. `pipeline_cli`: SIGTERM during round 2. |
| `d93057a63` | `jope-learning hwm ingest --source pipeline <receipt>`: a verified receipt becomes a `jope.hwm/v1` record under a complete comparison identity (dataset hash, cut id over the prepared-cut digests, evaluator/tool version, protocol, rmse lower-is-better). | `pipeline_hwm` (6): initialized then improved; tie and not-better rejected; another cut set or dataset is another identity; never teaching-eligible; an unverifiable receipt writes nothing; CLI. |
| `edea0f26f` | `jope-learning bank add`, `bank list` and `bank export`: an append-only, content-addressed bank. It holds HWM-`improved` champions only. `bank export` writes `jope.pipeline-seeds/v1`. | `pipeline_bank` (5). |
| `d57955bfb` | `jope pipeline score --receipt <r> --split test`: refits on train.csv and scores the sha-verified test.csv once per (dataset, genome) in a separate `jope.pipeline-test-score/v1`. It never feeds selection. | `test_score.rs` (3). |
| `51ef28559` | `lineage verify` binds each round's island model to the lineage's. | Extended tamper test. |
| `6ee177599` | `deploy/ci/e2e-smoke.sh` runs a 2-round mini-lineage on `pmlb-229-pwlinear` (evaluations=40). `lineage verify --require-discovery` checks it in Rust. | The e2e-smoke lane. |
| `888b61387` | `docs/campaigns.md`: the targeted single-dataset workflow. | — |

## 2. Local dry demo (2026-09-27, 16:57–17:39Z)

**Setup.** Dataset `datasets/test/uci-parkinsons-telemonitoring` (id `21fbfe8b53dbcd60`, regression):

- train.csv sha256 `d9a0b97e91ff9d04bc87fefa4671a49c40cee21dfac05be58a7fc7ce08045931`, 1,000 rows;
- test.csv sha256 `9786badb9d822cd0dc34b730019c13feff1fd02fe85da61a59c00a27e461f61c`, 999 rows;
- prepared cuts 0–2 of a 5-fold partition, fold seed 20260907, prepared sha256 `5d998c33…ab99`.

The run was `jope pipeline evolve --rounds 3 --control --cuts 3 --budget evaluations=200,seconds=600 --threads 8 --policy skip --emit rust --source-commit 888b61387…`. It used 1 island and 8 workers inside a 16-core, 48 GiB user scope. The multi-cut budget reserves half of each cut's allowance for comparison, so exploration had 100 evaluations and 300 s per cut. Total evolve wall time was 2,399 s.

**Per-round results.** Mean held-out RMSE and R2 are over the identical three cuts (original target units).

| Round | Kind | Seeds | Mean RMSE | Mean R2 | Gen ≥ 1 measured | Evaluations | Seconds | Exploration stop (cut 0/1/2) |
|---|---|---|---|---|---|---|---|---|
| 1 | unseeded | 0 | 0.020931 | 0.991525 | 264 | 324 | 482.2 | evaluations / evaluations / evaluations |
| 2 | seeded from round 1 | 4 | 0.019224 | 0.992768 | 244 | 316 | 563.0 | evaluations / **seconds** (92) / evaluations |
| 3 | seeded from round 2 | 4 | **0.019101** | 0.992987 | 252 | 324 | 562.6 | evaluations / evaluations / evaluations |
| control | unseeded, round 3's configuration | 0 | 0.019344 | 0.992761 | 215 | 275 | 791.5 | evaluations / **seconds** (51) / evaluations |

Champions:

- round 1: `(pipe X (blom append=true) (cheb append=true degree=6) (histgbm depth=7 … rounds=272))`, program `0a79df8b…`;
- round 2: `(pipe X (blom append=true) (poly_interact append=true) (blom append=false) (blom append=false) (histgbm depth=8 … rounds=272))`, program `3f83f3c0…`;
- round 3: `(pipe X (blom append=true) (blom append=true) (poly_interact append=false) (blom append=true) (histgbm depth=7 … rounds=273))`, program `831e2ff6…`;
- control: `(pipe X (robust_scale …) (winsor_clip …) (hinge knots=9) (hinge knots=7) (histgbm2 bags=3 … rounds=500 …))`, program `25869a2e…`.

`lineage verify --require-discovery` passed before and after ingest, banking and scoring. All four receipts re-derive with `verify-receipt --evidence`.

**HWM ledger** (`jope-learning hwm ingest --source pipeline`, one comparison identity `7080f66f…`):

| Receipt | Decision |
|---|---|
| round 1 | `initialized` (0.020931) |
| round 2 | `improved` (0.019224, displaces round 1) |
| round 3 | `improved` (0.019101, displaces round 2) |
| control | rejected `not-better` (0.019344 against incumbent 0.019101) |

`hwm verify` found 4 events (3 accepted, 1 rejected), 1 identity, 1 pointer, 4 blobs and 0 breaks.

**Champion bank.** `bank add` admitted rounds 2 and 3, refused round 1 (`initialized`) and the control (`not-better`), and wrote nothing when round 3 was re-added. `bank export` wrote 2 genomes, best first, to `jope.pipeline-seeds/v1` (sha256 `9b306bf0…77f3`), and refused a second export to the same path.

**Seeds consumption.** One `evolve --seeds <bank export>` round on the same cuts (evaluations=24, generation 0 only) did three things:

- recorded the export's sha256 in the lineage and the receipt;
- admitted both banked genomes as generation-0 `incumbent` seeds on all 3 cuts;
- re-measured them to exactly the recorded mean RMSEs (0.019101081869404863 and 0.019223874482223152). Its champion is the banked round-3 genome.

**Sealed test split** (`jope pipeline score`, once per (dataset, genome); refit on all 1,000 training rows):

| Champion | Development mean RMSE | Test RMSE | Test R2 |
|---|---|---|---|
| round 3 (final) | 0.019101 | 0.013780 | 0.996138 |
| control | 0.019344 | **0.011447** | 0.997335 |

Repeating the round-3 score was refused (exit 2) before anything was fitted. The round receipts still re-derive after scoring.

**What the numbers do and do not say.**

- Seeding carried each champion forward, and the lineage improved the development RMSE by 8.7% over round 1. Rounds 2 and 3 had more cumulative search than round 1, so the matched comparison is the control. The final round beats the control by 1.3% on development RMSE.
- **On the sealed test split the unseeded control's champion is better.** One dataset, one seed, 1,000 training rows and a small budget cannot separate seeding from noise. No generalisation gain from seeding is claimed.
- The control is at an equal logical budget but not an equal effective one. Its cut-1 exploration hit the 300 s bound after 51 of 100 evaluations (6 generations). The control's fits were the slowest of the four runs (p90 33 s, max 65 s). Round 2's cut 1 stopped at 92. A seconds bound makes the result wall-clock dependent.
- Rounds 1–3 reproduced an earlier run of the same command (11:18Z, same binary, another host load) genome for genome and RMSE for RMSE. That earlier run's control made 283 evaluations, not 275, and selected the same champion.
- **Parallel efficiency at 8 workers.** Fit-seconds per wall-second was 2.6–3.1 (round 1: 1,264 fit-s in 482 s; control: 2,447 in 792). Fit times are heavy-tailed (median about 1 s, p90 15–33 s, max 65 s). Each generation waits for its slowest fit, and the comparison phase runs at most one worker per cut.

## 3. Sizing the live demo (a projection, not a measurement)

- **The worker count is `islands × threads`**, which `--cpu-ceiling` bounds. A 100,000 mc GP rank therefore needs `islands × threads ≤ 100`. Deriving `--threads = ceiling / 1000` while also passing `--islands 9` is refused.
- **Per-generation width is 8 × islands.** That is `POPULATION − ELITES` new candidates per island (12 plus seeds at generation 0), so threads beyond 8 per island sit idle. Cuts are explored one after another, not concurrently, so 3 cuts do not triple the width. The comparison phase uses at most `cuts` workers.
- **A starting point per host** is `--islands 12 --threads 8 --cpu-ceiling 100000` (96 workers), with `--cuts 3 --budget evaluations=1000,seconds=600` (the per-cut maxima). Use `--migration-interval 2`: 500 exploration evaluations spread over 96-wide generations give only about 4–5 generations per cut per round.
- Rounds per run come from the round time measured in Window B. Gc uses the identical flags without seeds.
- On this evidence, one evolve process per host is **unlikely to reach the 100-CPU (78%) utilisation bar**. The per-generation barrier over heavy-tailed fits, the sequential cuts and a comparison phase at most `cuts` wide all hold it down. Measure it in Window B before claiming 80%.

## 4. Open

- W10-T01's operational witness (policy, corpus, teacher, update, checkpoint, export restore, S3, three hosts, no-update control) is untouched.
- The comparison between seeded rounds and the control is one small sample. A campaign across datasets and seeds would be needed for any claim about seeding.
- Utilisation at 96 or more workers is unmeasured (section 3).

## 5. Evidence

Receipts are kept outside the tree under
`/home/ubuntu/jope-rescue-20260921/regression-remediation-20260926/claude-w0-w12/runs/`:

- `p5-{seeding,parallel,islands,hwm,bank,evolve,score,lineage-islands,ci}-{pre,post,gate}-*` for each item;
- `p5-phase6-gate-*`;
- `p5-final-*` and `p5-final2-*` for the full verification;
- `p5-demo-20260927T111803Z` and `p5-demo-data-20260927T111803Z` for the first dry demo;
- `p5-drydemo-20260927T165746Z` for this dry demo's log.

This dry demo's lineage, receipts, HWM and bank state, bank export, seeds check and test-score registry are in `/home/ubuntu/.cache/claude-session-scratch/agents/p5/dry-demo/`, with `lineage.json` sha256 `db186dc2…8880`.
