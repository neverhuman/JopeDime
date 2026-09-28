# W10 — targeted pipeline evolution (a CPU witness adjacent to W10-T01)

## 0. Status, scope and source identity

| Item | Value |
|---|---|
| Task | Adjacent to W10-T01 ([`W10-learning-transfer-studies.tasks.md`](../W10-learning-transfer-studies.tasks.md)). **This is not the W10 real-loop witness, and it closes no W10 task.** |
| What it shows | The product's CPU pipeline runs targeted evolution on one chosen dataset: rounds on fixed cuts, each seeded from the previous round's champion, every step re-derived from its receipts. The learning store records the rounds under one comparison identity, and the champion bank keeps the HWM-improved champions as starting points for later rounds. |
| What it does not show | That later rounds' programs generalise better. Every round selects on the same held-out folds, so a round cannot lose ground there by construction and part of any round-over-round gain is selection on reused folds. The control is one round's budget, not a compute-matched comparison. The sealed test split (section 2) is the only generalisation evidence here, and on it the unseeded control's champion is better than the final round's. |
| What it does not exercise | A proposal policy, corpus, teacher, training update, checkpoint, export restore, S3 destination, three-node topology or no-update training control. W10-T01's operational witness requires all of these. None is claimed here. |
| Learning-store status | Pipeline HWM records are unbound (no evaluator binary hash), not replayable, not benchmark-valid and never `train_oof`, so they are **never teaching-eligible**. Nothing here is `live-qualified` or a production activation. The only `CI-verified` part is the e2e-smoke mini-lineage, once hosted CI runs it. |
| Source | Worktree `.agent/worktrees/bench`, branch `rr2/gp-evolve`. On 2026-09-28 the branch was rebased from `origin/main` `cee1fc8bd242688972c352133847793c6927279f` onto `origin/main` `f46a9048fbb2697d6d1f4463a688083ea9dfcce1`. `git range-diff` shows all 21 commits replayed as identical patches; section 6 maps the old SHAs to the new ones. **The measured source commit is the pre-rebase `078058006edf2a88df178cb4c69584169f4ec434`**, taken after the review fixes. The demo in section 2 ran binaries built there (`jope` sha256 `b50d4038…1b51`, `jope-learning` sha256 `0a83e434…216d`), and its HWM identity binds that commit. Its rebased counterpart, `db480f79a`, was not re-measured; a run there would be another comparison identity. The pre-rebase history (tip `538362b1588cf9df14e0a1e2e8d54c39dc568017`) is kept on origin as branch `archive/rr2-gp-evolve-pre-rebase-538362b15` and in a custody bundle (section 5). |
| Toolchain observed | rustc 1.97.1 (8bab26f4f 2026-07-14), release profile, CPU only. |
| Run receipts | `/home/ubuntu/jope-rescue-20260921/regression-remediation-20260926/claude-w0-w12/runs/p5-*`. Each run holds HEAD, status, the diff, the command, the log and the exit code. See section 5. |

Progress words follow [`EXECUTION-CONTRACT.md`](../EXECUTION-CONTRACT.md).

## 1. What exists (`f46a9048f..rr2/gp-evolve`; before the rebase, `cee1fc8bd..538362b15`)

The SHAs below are the rebased commits. Section 6 gives each one's pre-rebase SHA, which is the SHA the custody runs in section 5 record.

| Commit | Capability | Named tests |
|---|---|---|
| `e436a9b80` | `jope pipeline run --seed-genomes <jope.pipeline-seeds/v1>` and `--incumbent <receipt>`. Seeds are canonicalised, deduplicated and capped at the elite count. They enter generation 0 after the bootstrap with origin `incumbent` and are re-measured on this run's cuts. Receipt v3 `search.seeding` records their sha256s, the parent receipt sha256 and the seeds-file sha256. v1 and v2 receipts still validate. `verify-receipt --evidence` re-derives a receipt from its run directory. | `run.rs`: seed in gen 0 as incumbent; same seed and rng replay; champion never worse than its seed; a tampered parent is refused before anything is written; a budget that cannot re-measure the seeds is refused. `seeds.rs` (3). `schema_agreement` (v2/v3). |
| `19ac2a342` | `--threads N`: a generation's admitted candidates are evaluated on a dedicated pool. Admission and the fold of results happen in slot order, so the thread count changes no decision while the evaluation budget ends every phase. Under a binding seconds deadline the admitted candidates depend on timing, and so on the thread count; the receipt records such a phase as stopped by `seconds`. `--cpu-ceiling <millicores>`, `sched_getaffinity` and `available_parallelism` bound the workers. | Byte-identical runs at 1 and 8 threads (frozen clock, so the evaluation budget ends every phase). Identical decisions under the wall clock with a seconds budget that does not bind. Oversubscription refused. |
| `2c8df328c` | `--islands M --migration-interval K`: M random streams over the same cuts. Every K generations, each island's top-1 enters the other islands' elites only if it beats one of them. The pure `islands::next_seeds(champions, k)` is used both for migration and for round-to-round seeding. Receipt v3 `search.islands` records the migrations, and they are re-derived. The worker count is `islands × threads`. | `islands.rs` (3); island replay at 1 and 4 workers (frozen clock); one island equals the plain search; a migrant enters only an island it improves; islands × threads above the ceiling is refused. |
| `d00c7fbe1` | `--source-commit <sha>`: recorded as `git_sha`, and `git` is never spawned. SIGTERM/SIGINT stop between evaluations: no receipt is written and the exit code is 143. `tests/dependency_fence.rs` shows jope-pipeline depends on neither jope-learning nor jope-parallel. | `pipeline_cli`: a fake `git` is never called; the real binary is killed mid-search. `dependency_fence` (2). |
| `259a2a6ab` | `jope pipeline evolve --rounds N [--control] [--seeds]` runs on identical cuts. `lineage.json` (`jope.pipeline-lineage/v1`) is rewritten atomically and fsynced after every completed round. `jope pipeline lineage verify` checks it (its flag, then `--require-discovery`, is `--require-bred-candidates` since `445bd75df`). | `lineage.rs`: a 3-round lineage plus control chains and never worsens; tampering is refused; 1 and 8 threads write byte-identical lineages (frozen clock); a stop keeps only completed rounds. `pipeline_cli`: SIGTERM during round 2. |
| `c4c6a8e88` | `jope-learning hwm ingest --source pipeline <receipt>`: a verified receipt becomes a `jope.hwm/v1` record under a comparison identity naming the dataset hash, a cut id over the prepared-cut digests, the evaluator and the protocol. **Corrected by `e7b5d20ca`:** as first committed the identity could not tell two code revisions apart (the tool version never moves and the engine string is a constant), and it never contained the metric (`decide` checks that separately); its commit message and this note called it "complete". | `pipeline_hwm`: initialized then improved; tie and not-better rejected; another cut set or dataset is another identity; never teaching-eligible; an unverifiable receipt writes nothing; CLI. |
| `cf09f3bfc` | `jope-learning bank add`, `bank list` and `bank export`: an append-only, content-addressed bank. It holds HWM-`improved` champions only. `bank export` writes `jope.pipeline-seeds/v1` (ranked within one comparison identity since `db480f79a`). | `pipeline_bank`. |
| `f9273f7b8` | `jope pipeline score --receipt <r> --split test`: refits on train.csv and scores the sha-verified test.csv once per (dataset, genome) in a registry, in a separate `jope.pipeline-test-score/v1`. Nothing in the tools reads it back into selection (see `37ef8ea97` for what that does and does not enforce). | `test_score.rs`. |
| `9bdc1dd11` | `lineage verify` binds each round's island model to the lineage's. | Extended tamper test. |
| `6eb4fc828` | `deploy/ci/e2e-smoke.sh` runs a 2-round mini-lineage on `pmlb-229-pwlinear` (evaluations=40), checked by `lineage verify` in Rust (now `--require-bred-candidates`). | The e2e-smoke lane. |
| `e0ce8a772` | `docs/campaigns.md`: the targeted single-dataset workflow. | — |
| `85e3ca56c` | This note (first version). | — |

Review fixes (adversarial review of `rr2/gp-evolve`, 2026-09-27). Each has a pre-fix reproduction and a post-fix gate in custody (section 5).

| Commit | Finding and fix | Named tests |
|---|---|---|
| `4d769e62a` | `evolve` created `round-001/` and copied `--seeds` before `run` made its refusals, so any refusal left a directory that turned every corrected retry into "already holds a lineage". `run::check` now makes every refusal (seeds that are not pipelines, an unresolvable or non-regression dataset, a budget that cannot re-measure round 1's seeds or a later round's four) before anything is written. | `lineage.rs`: a refused lineage writes nothing and the corrected retry runs. |
| `c279ab6dd` | The receipt's v3 seeding invariant accepted a seed covered by a comparison re-measurement, while evidence verification requires an `incumbent` observation in generation 0 on every cut; the seconds bound could stop generation 0 before a seed started, and the run wrote a receipt its own verification refused. Generation 0's bootstrap and seeds are now exempt from the seconds bound (not from the evaluation bound or a stop request). | `search.rs`, `run.rs` with a test-only `Clock::Expired`: seeds measured on every cut at 1 and 3 workers past the bound; the receipt verifies; a stop still stops. |
| `a3c3eb031` | The evaluation pool had rayon's default 2 MiB stacks while `--threads 1` fitted on the caller's thread, so a deep fit could measure at one worker and abort the process at eight. Every fit, the test-split refit included, now runs on a 32 MiB evaluation stack on any worker count. | `parallel.rs`: a 10 MiB-deep fit survives at 1 and 2 workers (it aborted with SIGABRT before). |
| `445bd75df` | `--require-discovery` counted any measured bred candidate, improvement or not. Renamed `--require-bred-candidates`; the report adds `champion_found_in_round` and `improved_on_previous_round`, and evolve prints each champion as found in its round, carried from its seeds, or the bootstrap. | `lineage.rs`: a round whose champion is its own seed is not reported as a discovery. |
| `3b627f3bb` | The control was presented as the matched comparison that separates seeding from search, and campaigns.md said rounds drive a dataset "to better GP programs". The docs now say the control is one round's budget and that round gains are on reused folds; `lineage verify` reports `rounds_completed_evaluations` beside `control_completed_evaluations`. | The 3-round + control lineage test asserts both totals. |
| `37ef8ea97` | "Scored once" is per (dataset, genome) in a caller-chosen registry, not once per dataset. Each test score now records `earlier_looks` (the genomes already scored on that test split in the registry) and the command warns; the docs say what is and is not enforced. | `test_score.rs`: a score names the earlier looks at its dataset's test split. |
| `e7b5d20ca` | The pipeline HWM identity now names the source commit (`;source_commit=<git_sha>`); another commit is another identity, and a receipt without a commit is refused. Uncommitted changes and the binary itself are still not bound. | `pipeline_hwm`: another source commit is another identity and an unbound one is refused. |
| `db480f79a` | `bank export` ranked champions from different comparison identities against each other. It now ranks within one identity and refuses, listing them, when a dataset is banked under several and no `--identity` is given. Before the rebase this commit was `078058006`, the measured source commit. | `pipeline_bank`: an export never ranks across comparison identities. |

`7c0a3034e` updated this note with the demo re-run after the review fixes (section 2). After the rebase, `4b272d14e` changed prose only, in `docs/campaigns.md`, `docs/learning.md` and `docs/pipeline.md`, to pass the shipped-text rule in `deploy/ci/check.sh`. It changes no behaviour, schema or command, and the measured commit does not contain it. Section 7 records the fixes for the second review (`0dd98d103`, `d38941adb`).

## 2. Local dry demo (re-run after the review fixes, 2026-09-27, 18:39–19:20Z)

**Setup.** Dataset `datasets/test/uci-parkinsons-telemonitoring` (id `21fbfe8b53dbcd60`, regression):

- train.csv sha256 `d9a0b97e91ff9d04bc87fefa4671a49c40cee21dfac05be58a7fc7ce08045931`, 1,000 rows;
- test.csv sha256 `9786badb9d822cd0dc34b730019c13feff1fd02fe85da61a59c00a27e461f61c`, 999 rows;
- prepared cuts 0–2 of a 5-fold partition, fold seed 20260907, prepared sha256 `5d998c33…ab99`.

The run was `jope pipeline evolve --rounds 3 --control --cuts 3 --budget evaluations=200,seconds=600 --threads 8 --policy skip --emit rust --source-commit 078058006…`, with the pre-rebase measured commit (section 0). It used 1 island and 8 workers inside a 16-core, 48 GiB user scope. The multi-cut budget reserves half of each cut's allowance for comparison, so exploration had 100 evaluations and 300 s per cut. Total evolve wall time was 2,389 s.

**Per-round results.** Mean held-out RMSE and R2 are over the identical three cuts (original target units), which every round also selected on.

| Round | Kind | Seeds | Mean RMSE | Mean R2 | Gen ≥ 1 measured | Evaluations | Seconds | Exploration stop (cut 0/1/2) | Champion |
|---|---|---|---|---|---|---|---|---|---|
| 1 | unseeded | 0 | 0.020931 | 0.991525 | 264 | 324 | 472.4 | evaluations / evaluations / evaluations | found in the round |
| 2 | seeded from round 1 | 4 | 0.019224 | 0.992768 | 244 | 316 | 560.9 | evaluations / **seconds** (92) / evaluations | found in the round |
| 3 | seeded from round 2 | 4 | **0.019101** | 0.992987 | 252 | 324 | 564.1 | evaluations / evaluations / evaluations | found in the round |
| control | unseeded, round 3's configuration and stream, one round's budget | 0 | 0.019344 | 0.992761 | 215 | 275 | 791.3 | evaluations / **seconds** (51) / evaluations | found in the run |

`lineage verify --require-bred-candidates` reports `improved_on_previous_round: [true, true]`, `champion_found_in_round: [true, true, true]`, and **964 completed evaluations over the three rounds against the control's 275**. It passed before and after ingest, banking and scoring. All four receipts re-derive with `verify-receipt --evidence`.

Champions:

- round 1: `(pipe X (blom append=true) (cheb append=true degree=6) (histgbm depth=7 … rounds=272))`, program `0a79df8b…`;
- round 2: `(pipe X (blom append=true) (poly_interact append=true) (blom append=false) (blom append=false) (histgbm depth=8 … rounds=272))`, program `3f83f3c0…`;
- round 3: `(pipe X (blom append=true) (blom append=true) (poly_interact append=false) (blom append=true) (histgbm depth=7 … rounds=273))`, program `831e2ff6…`;
- control: `(pipe X (robust_scale …) (winsor_clip …) (hinge knots=9) (hinge knots=7) (histgbm2 bags=3 … rounds=500 …))`, program `25869a2e…`.

This re-run reproduced the pre-fix run of the same command (16:57Z, binaries at the pre-rebase `888b61387`, which is in the archived history) genome for genome and RMSE for RMSE, including the control's 275 evaluations and its cut-1 stop at 51. The review fixes change no decision in this regime: every seed started well before its deadline.

**HWM ledger** (`jope-learning hwm ingest --source pipeline`, one comparison identity `eb404c6f…`, whose protocol names `source_commit=078058006…`):

| Receipt | Decision |
|---|---|
| round 1 | `initialized` (0.020931) |
| round 2 | `improved` (0.019224, displaces round 1) |
| round 3 | `improved` (0.019101, displaces round 2) |
| control | rejected `not-better` (0.019344 against incumbent 0.019101) |

`hwm verify` found 4 events (3 accepted, 1 rejected), 1 identity, 1 pointer, 4 blobs and 0 breaks.

**Champion bank.** `bank add` admitted rounds 2 and 3, refused round 1 (`initialized`) and the control (`not-better`), and wrote nothing when round 3 was re-added. `bank export` ranked within the one identity and wrote 2 genomes, best first, to `jope.pipeline-seeds/v1` (sha256 `a1216f51…dcb1`), and refused a second export to the same path.

**Seeds consumption.** One `evolve --seeds <bank export>` round on the same cuts (evaluations=24, generation 0 only) recorded the export's sha256 in the lineage and the receipt, admitted both banked genomes as generation-0 `incumbent` seeds on all 3 cuts, and re-measured them to exactly the recorded mean RMSEs (0.019101081869404863 and 0.019223874482223152). Its champion is the banked round-3 genome, reported as carried from its seeds.

**Sealed test split** (`jope pipeline score`; refit on all 1,000 training rows). The final champion was chosen before any test score was seen and scored first; the control's score is a second look at the same test split, and its receipt lists the first in `earlier_looks` (the command warned).

| Champion | Look | Development mean RMSE | Test RMSE | Test R2 |
|---|---|---|---|---|
| round 3 (final) | 1st | 0.019101 | 0.013780 | 0.996138 |
| control | 2nd (`earlier_looks: [831e2ff6…]`) | 0.019344 | **0.011447** | 0.997335 |

Repeating the round-3 score was refused (exit 2) before anything was fitted. The round receipts still re-derive after scoring.

**What the numbers do and do not say.**

- The development RMSE fell from 0.020931 (round 1) to 0.019101 (round 3), 8.7%, on the same three folds every round selected on. A seeded round re-measures its parent's champion there, so it cannot lose ground on those folds; the drop shows the lineage machinery working, not programs that generalise better.
- The control is not a matched comparison. It is one unseeded round at one round's budget on round 3's stream: 275 evaluations against the rounds' 964, and its cut-1 exploration stopped on the seconds bound after 51 of 100 evaluations. Its 1.3% higher development RMSE cannot be read as the effect of seeding; one control on one stream cannot separate seeding from the extra compute or from noise.
- **On the sealed test split the unseeded control's champion is better** (0.011447 against 0.013780). One dataset, one seed, 1,000 training rows and a small budget: no generalisation gain from seeding is claimed.
- A seconds bound makes these runs wall-clock dependent. The control's fits were the slowest of the four runs (p90 32 s, max 67 s); round 2's cut 1 also stopped on seconds, at 92.
- **Parallel efficiency at 8 workers.** Fit-seconds per wall-second was 2.6–3.1 (round 1: 1,233 fit-s in 472 s; control: 2,434 in 791). Fit times are heavy-tailed (median 0.6–1.6 s, p90 15–32 s, max 67 s). Each generation waits for its slowest fit, and the comparison phase runs at most one worker per cut.

## 3. Sizing the live demo (a projection, not a measurement)

- **The worker count is `islands × threads`**, which `--cpu-ceiling` bounds. A 100,000 mc GP rank therefore needs `islands × threads ≤ 100`. Deriving `--threads = ceiling / 1000` while also passing `--islands 9` is refused.
- **Per-generation width is 8 × islands.** That is `POPULATION − ELITES` new candidates per island (12 plus seeds at generation 0), so threads beyond 8 per island sit idle. Cuts are explored one after another, not concurrently, so 3 cuts do not triple the width. The comparison phase uses at most `cuts` workers.
- **A starting point per host** is `--islands 12 --threads 8 --cpu-ceiling 100000` (96 workers), with `--cuts 3 --budget evaluations=1000,seconds=600` (the per-cut maxima). Use `--migration-interval 2`: 500 exploration evaluations spread over 96-wide generations give only about 4–5 generations per cut per round.
- Rounds per run come from the round time measured in Window B. A control uses the identical flags without seeds, and is one round's budget: a compute-matched comparison needs an unseeded run (or several) with the lineage's cumulative budget, over several seeds.
- On this evidence, one evolve process per host is **unlikely to reach the 100-CPU (78%) utilisation bar**. The per-generation barrier over heavy-tailed fits, the sequential cuts and a comparison phase at most `cuts` wide all hold it down. Measure it in Window B before claiming 80%.

## 4. Open

- W10-T01's operational witness (policy, corpus, teacher, update, checkpoint, export restore, S3, three hosts, no-update control) is untouched.
- The comparison between seeded rounds and the control is one small, compute-unmatched sample. A campaign across datasets and seeds, with budget-matched unseeded runs, would be needed for any claim about seeding.
- The pipeline HWM identity binds the source commit, not the evaluator binary or uncommitted changes (`evaluator_sha256` is `unbound`).
- The test-split registry is a local record, not access control; several looks are disclosed, not prevented.
- Utilisation at 96 or more workers is unmeasured (section 3).

## 5. Evidence

Receipts are kept outside the tree under
`/home/ubuntu/jope-rescue-20260921/regression-remediation-20260926/claude-w0-w12/runs/`:

- `p5-{seeding,parallel,islands,hwm,bank,evolve,score,lineage-islands,ci}-{pre,post,gate}-*` for each item;
- `p5-phase6-gate-*`;
- `p5-final-*` and `p5-final2-*` for the full verification before the review fixes;
- review fixes: `p5-rv1-evolve-refusals-pre2-20260927T180644Z` and `-post-20260927T180725Z`; `p5-rv2-seeds-deadline-pre-20260927T181200Z` and `-post-20260927T181329Z`; `p5-rv349-pre-20260927T181701Z` (stack overflow, earlier looks, bred-not-discovery reproductions); `p5-rv3-eval-stack-post2-20260927T181938Z`; `p5-rv9-bred-post-20260927T182255Z`; `p5-rv7-control-compute-post-20260927T182645Z`; `p5-rv56-hwm-bank-pre-20260927T181209Z`; `p5-rv456-post-20260927T183059Z`;
- `p5-demo-20260927T111803Z` and `p5-demo-data-20260927T111803Z` for the first dry demo;
- `p5-drydemo-20260927T165746Z` (log) and `p5-drydemo-data-20260927T180250Z` (lineage, receipts, HWM and bank state, bank export, seeds check, test-score registry; `lineage.json` sha256 `db186dc2…8880`) for the pre-fix dry demo;
- `p5-drydemo2-20260927T183604Z` (log) and **`p5-drydemo2-data-20260927T192104Z`** for the dry demo in section 2: its lineage, receipts, HWM and bank state, bank export, seeds check and test-score registry, with `SHA256SUMS` (99 files, verified against the source) and `PROVENANCE.txt`. Its `lineage.json` sha256 is `02b7366d…788b`;
- rebase: `gp-rebase-pre-20260928T033037Z` holds the pre-rebase identity and `rr2-gp-evolve-538362b15.bundle` (sha256 `49481cca…30be`). The bundle's one head is `538362b15`, with prerequisite `cee1fc8bd`; it contains the measured `078058006` and the pre-fix `888b61387`. `gp-rebase-post-20260928T033252Z` and `gp-rebase-post-20260928T034212Z` hold the local CI lanes after the rebase;
- `gp-rebase-rev-{pre,post}-*`: every commit this note cites, checked as reachable from the branch or from the archived pre-rebase tip.

## 6. Commits before and after the 2026-09-28 rebase

The custody runs before 2026-09-28 (`p5-*`) record the pre-rebase SHAs. Branch `archive/rr2-gp-evolve-pre-rebase-538362b15` on origin keeps them reachable. `git range-diff cee1fc8bd..538362b15 f46a9048f..7c0a3034e` pairs every commit below as an identical patch.

| Before (on `cee1fc8bd`) | After (on `f46a9048f`) | Subject |
|---|---|---|
| `cd3a3c946` | `e436a9b80` | pipeline: seed generation 0 from a verified parent receipt or a seeds file |
| `feabdbacc` | `19ac2a342` | pipeline: evaluate each generation on --threads workers, deterministically |
| `38b6dbda4` | `2c8df328c` | pipeline: island model with deterministic champion migration |
| `939a3b24c` | `d00c7fbe1` | pipeline: --source-commit, clean SIGTERM stop, and a dependency fence |
| `ea79139d6` | `259a2a6ab` | pipeline: `evolve` rounds on fixed cuts, chained in a verified lineage |
| `d93057a63` | `c4c6a8e88` | learning: ingest verified pipeline receipts into the HWM ledger |
| `edea0f26f` | `cf09f3bfc` | learning: champion bank of HWM-improved pipeline champions |
| `d57955bfb` | `f9273f7b8` | pipeline: score a champion once on the sealed test split |
| `51ef28559` | `9bdc1dd11` | pipeline: lineage verify binds the island model |
| `6ee177599` | `6eb4fc828` | ci(e2e-smoke): a mini-lineage checked in Rust |
| `888b61387` | `e0ce8a772` | docs(campaigns): the targeted single-dataset evolution workflow (the pre-fix demo's binaries) |
| `fb8bd1802` | `85e3ca56c` | docs(evidence): W10 note for targeted pipeline evolution |
| `3f782be48` | `4d769e62a` | pipeline: evolve refuses everything before it writes, so a retry runs |
| `aa8673525` | `c279ab6dd` | pipeline: generation 0 always re-measures its seeds, whatever the clock |
| `10fcd87e4` | `a3c3eb031` | pipeline: every fit runs on a 32 MiB evaluation stack, on any worker count |
| `b353f10a9` | `445bd75df` | pipeline: lineage verify says what it checks: bred candidates, not discovery |
| `ee6e048fb` | `3b627f3bb` | docs, pipeline: the control is one round's budget, and round gains are on reused folds |
| `32f93c945` | `37ef8ea97` | pipeline: a test score lists the earlier looks at its dataset's test split |
| `8e5d961ee` | `e7b5d20ca` | learning: a pipeline HWM identity names the source commit that measured |
| **`078058006`** | `db480f79a` | learning: bank export ranks within one HWM identity (**the measured source commit**) |
| `538362b15` | `7c0a3034e` | docs(evidence): W10 note re-run after the review fixes, data in custody |

## 7. Second review: review 5334007364 on PR #118 (2026-09-28)

An independent review of head `d7e89f211` requested changes on two P2 findings. Its receipts are under `/home/ubuntu/jope-rescue-20260921/regression-remediation-20260926/claude-w0-w12/independent-review-pr118-d7e89f211/`. Both findings were reproduced on the committed `datasets/test/pmlb-229-pwlinear` fixture before anything was changed. These commits change no measured result in section 2: the first adds refusals, the second changes wording and tests only.

| Commit | Finding and fix | Named tests |
|---|---|---|
| `0dd98d103` | `run::preflight` accepted any positive `--cuts`, but the preparation accepts only 1 to 5. `jope pipeline evolve --rounds 2 --cuts 6` exited 2 (`--cuts must be 1..=5, got 6`) after it had created `round-001/`, so the corrected `--cuts 3` into the same `--output` exited 2 with "already holds a lineage". `jope pipeline run --cuts 6` left an empty `--output` behind. `preflight` now checks the whole range with `prepare::check_cuts`, in the same words (`--cuts 0` now gets that message instead of "at least 1"). The preparation is split into a pure `prepare::partition`, which makes every refusal, and `Partition::write`, which only publishes; `build` is the two together, byte for byte as before. `run::check`, which `evolve` runs before round 1, now partitions the dataset, and `run` partitions before it creates `--output`. A dataset that resolves and parses but cannot be cut (a cut with fewer than two rows on either side, or a constant held-out target) is therefore also refused with nothing written. | `prepare.rs`: the cut range; a partition writes nothing until it is published, and publishes `build`'s bytes. `run.rs`: `--cuts 0` and `6` and a constant-target copy of the fixture are refused by `check` and by `run` with no `--output`, and the corrected run into it succeeds. `lineage.rs`: the review's reproduction (cuts 6, cuts 0, constant target), then `--cuts 3`, 2 rounds, into the same `--output`, verified. `pipeline_cli`: the real binary exits 2 and writes nothing for `evolve --cuts 6`, `evolve --cuts 0` and `run --cuts 6`, and the corrected `evolve --cuts 3` runs. |
| `d38941adb` | `docs/pipeline.md` said the thread count "never changes a decision" and is not recorded in the receipt "because it cannot change it". `docs/campaigns.md`, section 1 of this note and comments in `search.rs`, `parallel.rs`, `run.rs`, `cli.rs`, `clock.rs` and `lineage.rs` made the same claim without a condition. It holds only while the evaluation budget ends every phase. The seconds bound is checked again before each fit starts, on its worker, so a binding bound admits a timing-dependent number of candidates. The reviewer measured this on the fixture (`--rng-seed 7`, one cut, `evaluations=200,seconds=1`): 2 completed evaluations at `--threads 1` and 12 at `--threads 8`, both recorded as stopped by `seconds`. The two runs selected the same program, but their observations differ (`deadline-1/` and `deadline-8/` in the review's custody). The wording now scopes the claim and says that such a receipt records `seconds` as the phase's stop reason. Behaviour is unchanged. | `run.rs`: the 1-and-8-thread byte-identity test now asserts that no phase stopped on `seconds`; `a_phase_the_seconds_bound_ended_is_recorded_as_stopped_by_seconds` (test-only expired clock, so deterministic) shows that such a phase is recorded as `seconds` and that a frozen-clock run never is. `search.rs`: the wall-clock decision test is renamed `thread_count_changes_no_decision_when_the_evaluation_budget_binds` and asserts that it ran in that regime. |

The demo in section 2 ran with `seconds=600`. Round 2's cut 1 and the control's cut 1 stopped exploring on seconds, as that section says, so those two runs are timing-dependent in this sense. Their retained receipts (`p5-drydemo2-data-20260927T192104Z`) show that every other exploration phase stopped on evaluations and that every comparison phase ended with its pool exhausted.

Evidence, under `/home/ubuntu/jope-rescue-20260921/regression-remediation-20260926/claude-w0-w12/runs/`:

- `gp-r1-pre-20260928T044031Z`: the reproduction at `d7e89f211` (debug `jope` sha256 `40bbffff…edf8`). `evolve --cuts 6` exits 2 and leaves `round-001/`; the corrected `--cuts 3` into it exits 2 with "already holds a lineage"; `evolve --cuts 0` exits 2 with nothing written (the old check covered 0); `run --cuts 6` exits 2 and leaves an empty `--output`.
- `gp-r1-c1-gate2-20260928T044828Z`: the gate for `0dd98d103`: `cargo fmt --check`, `git diff --check`, `cargo check --workspace --all-targets --locked`, the jope-pipeline lib (127 passed, 2 ignored), `pipeline_cli` (4 passed), the fence suites (`fit_boundary` 16, `gp_contracts` 8, `view_contracts` 17, `gp::fit_contract_tests` 44 passed and 3 ignored), coverage validation and the test inventory, all exit 0. `gp-r1-c1-gate-20260928T044457Z` is its first attempt, which failed on rustfmt line wrapping and on a wrong package name for the fence suites.
- `gp-r1-post-20260928T045207Z`: the same commands at `0dd98d103` (debug `jope` sha256 `544ecf8c…7b04`). Every refusal exits 2 and writes nothing; the corrected `--cuts 3` into the refused `--output` exits 0, and `lineage verify` on it exits 0.
- `gp-r1-c2-gate-20260928T045417Z` and `gp-r1-c2-fmt-20260928T045829Z`: the gate for `d38941adb`, with the shipped-text rule added (jope-pipeline lib 128 passed, 2 ignored; the other counts as above). The first run's only failure was rustfmt line wrapping in a new test; the second re-ran the format, diff and workspace checks and the new test after `cargo fmt`.
- `gp-r1-final-*`: the final verification of the branch head, including the `rust-check`, `rust-test` and `e2e-smoke` lanes as `.github/workflows/ci.yml` runs them.
- Follow-ups from the verification of `843f8c73e` (its report is kept as `verifier-check-report.md` in `gp-r2-final-*`), none of which changes a measured result in section 2: `8b54b84c4` makes `jope pipeline evolve` refuse an `--output` that exists and is not an empty directory before it writes anything (a stale `control/` or `round-002/` had let the earlier rounds write `lineage.json`, and round 2 its `seeds.json`, before the refusal, so the corrected retry was refused as "already holds a lineage"; an unrelated file was accepted); `a233ea7c7` makes `jope pipeline run`'s check of the entries it writes see a dangling symbolic link (a dangling `emit` had let it write `prepared/` and `search/` and then fail, and a dangling `receipt.json` had the receipt written outside `--output`); `7204daa41` qualifies the replay sentence in `search.rs` and says in `docs/pipeline.md` that an exploration phase the deadline cut short shows in `accounting.exploration.stopped_by`. Evidence: `gp-r2-pre-20260928T052521Z` (the reproduction at `843f8c73e`), `gp-r2-c1-gate-20260928T052759Z`, `gp-r2-c2-gate-20260928T053241Z`, `gp-r2-c3-gate-20260928T053701Z`, `gp-r2-post-20260928T053759Z` (the same commands at `7204daa41`) and `gp-r2-final-*`.
