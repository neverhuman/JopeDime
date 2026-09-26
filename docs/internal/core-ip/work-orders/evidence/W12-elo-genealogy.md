# W12-T04 — Historical Elo and genealogy reconciliation

Observed 2026-09-26 on the `feat/regression-remediation` lane (source head at observation
`1aaec28a24f7b4bdd27a169a87733586784042ee` plus the uncommitted files named below). Read-only
reconstruction from retained repository bytes: no fit, training, benchmark, Cargo build or host
run. It addresses the source/fixture part of findings `S08-G09` and `S08-K09`
(`findings/sources/shard-08.md:74-79`, `:146-151`). Both findings remain open.

The machine record is [`elo-history/histories.json`](../../elo-history/histories.json). The
verifier was first written as `tools/elo_history.py` (18 tests) and has since been ported,
operation for operation, to Rust: [`crates/jope-records/src/elo_history.rs`](../../../../../crates/jope-records/src/elo_history.rs),
tested by `crates/jope-records/tests/records/elo_history.rs` (the 18 cases plus additions). The
port reproduces the Python replay's outputs exactly on the committed record, including the
2,000-iteration `--sensitivity` fits, and like it never calls `jope-arena`:

```sh
cargo run -q -p jope-records -- elo-history                  # verify the record, replay every ladder
cargo run -q -p jope-records -- elo-history --require-all    # also fail if a ladder cannot be replayed here
cargo run -q -p jope-records -- elo-history --sensitivity H-sotarena-golden45-published
cargo nextest run -p jope-records -E 'test(/elo_history/)'
```

Two ladders read `scores.csv.zst`. The Python replay needed a `zstd` binary on `PATH`; the Rust
port decodes zstd in process (`ruzstd`), so these ladders replay wherever the repository is
checked out. A build without a decoder would still list them under `not_replayed_here`, never
count them as replayed and leave them out of the exact totals, and `--require-all` turns that
condition into a failure.

## 1. The recovered construction

Every retained ladder uses one algorithm, `regularized-bradley-terry-elo-scale`. This repository
has two implementations of it: the per-match port of `sotarena/elo.py` and the reporter's version,
which aggregates pair statistics first (`crates/jope-arena/src/elo.rs:1-17`).

| Element the finding asks for | What was used | Anchor |
|---|---|---|
| Participants | Rows with `status = ok` and a value; a method with no match is unrated | `crates/jope-arena/src/scores.rs:92-95`, `elo.rs:269-281` |
| Match events | One equal-weight match for each dataset and each pair of methods that both have a score there | `elo.rs:108-142` |
| Outcome conversion | Direction taken from `metric_direction` (RMSE is minimized); 0.5 when `|a−b| ≤ 1e−12·max(|a|,|b|,1)`, otherwise 1 or 0 | `elo.rs:309-321` |
| Initialization and anchor | Every rating starts at 1500; a final mean-centring shift to 1500 is a convention, not a reference competitor | `elo.rs:267-307` |
| Update | Synchronous diagonal Newton steps, L2 `1e−4` toward 1500, each iteration's step clamped to ±40, at most 250 iterations, stop when the largest step is below `1e−8` | `elo.rs:155-228` |
| Ordering | Sorted datasets and upper-triangle method pairs, added sequentially; the ladder sorts by (−rating, method id) | `elo.rs:1-17`, `leaderboard.rs:370-371` |
| Algorithm version | Golden45 published ladder: `sotarena/elo.py` in the benchmark clone at `e5db30a0…` (2026-07-25). Same-cut and benchmark-valid ladders: the `jope-arena` port at each report's `git_sha` | `assets/regression/SOURCES.md:180-220`, `assets/same-cut/v*/SOURCES.md` |
| Duplicate delivery | The Rust port inserts into a map, so a second delivery of one (dataset, method) cell silently replaces the first. The independent replay refuses it instead. No retained ladder contains a duplicate, because all six replay under the refusing parser | `leaderboard.rs:331-334`, `jope_records::elo_history::parse_scores` |

The independent replay was written from this specification in plain Python and is now its Rust
port with the same accumulation order. It uses aggregated pair statistics (match count and
outcome sum), not a copy of the per-match port.

## 2. Disposition of each history

| History | Disposition | Retained inputs → outputs | Result of this reconciliation |
|---|---|---|---|
| `H-same-cut-v1-deep` | reproducible-with-receipts | track `scores.csv` `f6438041…` → leaderboard `b19314a5…` | Replays byte for byte; 11 methods, 100 matches; one connected component |
| `H-same-cut-v1-screen` | reproducible-with-receipts | `7d11173a…` → `af5088ea…` | Every cell matches. Only the order of the exchangeable pair `baseline.elasticnet`/`baseline.ridge` (both 1388.7) differs from the committed ladder |
| `H-same-cut-v2-deep` | reproducible-with-receipts | `f945ec48…` → `bf2c6dc0…` | Replays byte for byte; 15 methods |
| `H-same-cut-v2-screen` | reproducible-with-receipts | `5175ea13…` → `03890d9c…` | Replays byte for byte. Contains two exchangeable pairs; their committed order happens to agree with the replay |
| `H-sotarena-golden45-published` | reproducible-with-receipts | `scores.csv` (decoded `5ec86898…`) and ledger → `2d28ffbd…` | Every cell matches. Only the order of the exchangeable pair `ngboost`/`xgboost_hist_trees900_depth4_lr0p02` (ranks 81 and 82, 1576.5) differs. The fit ends at the 250-iteration cap without converging |
| `H-sotarena-golden45-benchmark-valid` | reproducible-with-receipts | same scores plus the corrections file `6a831c4a…` → `4c39f51c…` | Replays byte for byte, with 45 rows demoted by the corrections. Also ends at the iteration cap. This is a ladder derived in this repository, not a benchmark publication |
| `H-campaign-redteam-bradley-terry` | scorer-parity-only | — | Rust-to-Rust parity with `jope-arena` on three synthetic datasets (`elo-parity-followup.md`; CI inputs lane, `deploy/ci/fast-lane.sh:125-126`). No red-team ladder from a past campaign is retained |
| `H-sotarena-reporter-aggregated` | scorer-parity-only | — | The ported reporter agrees with the per-match fit to within 1e−6 Elo on a synthetic fixture (`elo.rs::tests::aggregated_and_per_match_fits_agree`). The reporter's own data was not copied |
| `H-sotarena-golden42-v1` | irrecoverable | none | `scores.csv` `b943def7…` is not in this repository. The "jope-dime 2259.9" value is a statement made at port time, not a replayable rating |
| `H-legacy-round1-internal-elo` | irrecoverable | `elo_trajectory.csv` `5d1cc168…` (generations 1–30) | Only paper summaries are retained. Match events, participants, algorithm, K/rounds and update order are missing, and there is no generation-31 row |
| `H-render-elo-ranking-static` | irrecoverable | the figure script only | Its rating literals (for example 1260.94) come from a test-adaptive run whose inputs are not retained |
| `H-fable-official-ladder-replica` | irrecoverable | `elo_replica.py`, `elo_meter.py` | A different algorithm (60 sweeps, K=64, initial rating 1000). The official reference, the parity records and all outputs are missing |
| `H-task-elo-strict-corpus` | irrecoverable | none | Source only; the SQLite inputs and `ops/task-elo.v1.json` live outside the repository |
| `H-task-rating-snapshots` | irrecoverable | none | No `JopeTaskRatingSnapshot.v1` file is committed |

`jope_records::elo_history::authorize` enforces these dispositions:

- Ranking and rating claims are allowed only for `reproducible-with-receipts` histories, and only
  for the `historical-ranking-only` consumer.
- Teaching, HWM, transfer and promotion consumers are refused for every history.
- Continuity between ladders is always refused, because each rating is relative to its own method
  universe.
- An `irrecoverable` or `scorer-parity-only` record cannot carry a replay, ratings or a ranking,
  and must say what is missing.

**Exact totals.** The totals cover only the six replayed histories: 17,841 rankable rows and
1,708,614 pairwise matches. The eight other histories are listed under
`excluded_from_exact_totals` and add nothing.

## 3. What the replay found

1. **The published order of exchangeable methods is an artefact of floating-point arithmetic.**
   Bradley–Terry ratings depend on the data only through match counts and each method's total
   score. Two methods with equal match counts against every other method, and equal totals, have
   equal ratings in exact arithmetic even when their records against individual opponents differ.
   For example, against `afm.lite-race.v1.resampled` `baseline.elasticnet` scores 2 of 10 and
   `baseline.ridge` scores 2.5 of 10, yet their totals are equal. Among such methods the committed
   order was set by the original accumulation order.

   The verifier accepts a changed order only inside such a group. Every other cell must match
   byte for byte, and it lists the groups:

   - Same-cut v1 screen has 1 group and v2 screen has 2.
   - Golden45 published has 5 groups, including a 4-way group of `validation_gp_full_…` variants.
   - The benchmark-valid ladder has 5 groups, including a 3-way group.

   No claim may rest on the relative rank of methods in one group.
2. **Identical score vectors (possible aliases).** Three groups in both Golden45 ladders have
   identical scores on every dataset:
   - `catboost_mae_trees300_depth6`, `catboost_quantile_alpha0p5_trees300_depth6`
   - `jope-lime`, `jope_prime`
   - the four `validation_gp_full_20260710_143239__ab0239b9ef0ba28e__*` variants

   Duplicated competitors shift the ratings of all other methods. This record does not decide
   whether they are aliases.
3. **The Golden45 ratings are the value at the iteration cap, not converged values.** Both
   Golden45 fits stop at the 250-iteration cap before reaching the `1e−8` tolerance.
   `--sensitivity` measured how much a 2,000-iteration fit on the same inputs moves them. Neither
   longer fit converged either.

   | Ladder | Largest rating shift | Ladder positions that change | Top 10 |
   |---|---:|---:|---|
   | Published | 0.31 Elo | 6 | unchanged |
   | Benchmark-valid | 0.54 Elo | 6 | unchanged |

   The same-cut fits converge; v1 screen, for example, converges in 123 iterations with no shift.
   The retained Golden45 values therefore depend on the iteration setting, and they are
   reproduced exactly only at display precision.
4. **Graph coverage.** Every retained ladder forms one connected comparison graph, so no order
   between separate groups is imposed by the regularizer. Coverage per method varies widely: for
   example, `adapter.rf` is scored on 1 of 10 datasets in same-cut v2 deep. Ratings from partial
   coverage rest on fewer matches. The ladders already show that coverage beside the rating.
5. **Consumers.** A search of `crates/jope-learning/src`, `crates/jope-engine/src` and
   `crates/jope-policy/src` found no Elo consumer. `jope-campaigns` uses `jope-arena` only as a
   dev-dependency (`crates/jope-campaigns/Cargo.toml:32`), and `jope-bench` uses it to write
   ladders. The README renders ladders for publication. No teaching, HWM or transfer gate reads
   Elo, so the severity stays as `S08-G09` states it: a risk to historical reproducibility and
   interpretation.
6. **Derived plot table.** All 661 rows of `docs/internal/asm-manual/data/plot-historical-elo.csv`
   repeat their digest-pinned snapshots (rank, Elo and coverage), and every snapshot is plotted
   completely.

## 4. Scorer parity is not recovery of historical matches

The two `scorer-parity-only` histories show that the implementations in use today agree with one
another on synthetic fixtures. They recover no historical match graph, and they cannot upgrade
`H-sotarena-golden42-v1`, the legacy round-1 Elo or any campaign ladder. Likewise, the
byte-identical Golden45 replay starts from the retained `scores.csv`. That file aggregates 397
upstream evidence files that were verified in place when the port was made and are not retained
here, so the replay attests the ladder arithmetic, not the upstream scores.

## 5. Genealogy: lineage and exposure edges

| Lineage | Disposition | Retained table | Parent edges | Edges to parents that are not retained |
|---|---|---|---|---|
| `L-lineage-b-a100-recovery` | reproducible-with-receipts | `a100-checkpoint-lineage.csv` `a141f361…` | 78 checkpoints, 22 blocks, 22 corpora; 74 parents retained | 4 edges, all to the step-3000 state `20dca670…` |
| `L-native-audit-108-blocks` | reproducible-with-receipts | `block-endpoint-metrics.csv` `5278b22f…` | 108 checkpoints, blocks and corpora; 73 parents retained | 35 edges: 34 to `c2f229c9…` and 1 to `20dca670…`, the campaign64 qualification checkpoints for cycles 0000 and 0001 at step 3000 |
| `L-lineage-a-xbabe2` | irrecoverable | none | — | Live control-plane state and later descendants were declared lost (`sections/09.md:29`) |
| `L-legacy-round1-generations` | irrecoverable | none | — | No checkpoint identities from one generation to the next (`sections/09.md:84`, `:96-104`) |

- **What a parent edge records.** A parent edge links a block to the state it started from, so
  the order of steps inside a block is not a recorded edge.
- **Accepted step 60,000.** `jope_records::elo_history::continuity` confirms that it descends from step
  6,000 through 18 retained block edges. It refuses any continuity claim that reaches step 3,000
  or earlier, and any claim between steps in the same block, such as 4,000 and 4,477.
- **Shared unretained root.** Both native lineages descend from the unretained state `20dca670…`,
  and the 108-block audit also descends from `c2f229c9…`. Neither table retains either state.
- **Exposure.** Each block names one corpus digest. Which datasets each corpus contains is not
  retained, so `exposure()` refuses exposure totals for every lineage. Checkpoint exposure to the
  frozen panel therefore remains unknown (`sections/02.md:32`), and the corresponding
  never-seen-family and transfer claims stay blocked.

## 6. Limits and open items

- Unrounded historical ratings were never retained. The published values are matched at display
  precision: 0.1 Elo for ratings and 0.001 for average rank.
- This work changes no Rust. The port's silent replacement of a duplicate cell
  (`leaderboard.rs:331-334`) is recorded here, not changed. A change to refuse duplicates belongs
  to `jope-arena` and its own review.
- The Python replay needed a `zstd` binary, which the hosted hygiene runner was never verified to
  provide. The Rust port decodes zstd in process, so the Golden45 histories now replay in hosted
  CI as well. Without a decoder they would be reported as not replayed, never as passed.
- `S08-G09` and `S08-K09` stay open in `findings/dispositions.json`. Any disposition record there
  is a separate, reviewed step.

## 7. Appended 2026-09-26: jope-arena refuses a duplicate cell delivery (W12-T04; branch `rr2/integration-c`)

Section 6 recorded that the Rust port silently replaced a duplicate cell. This section fixes it
in `jope-arena`.

- **Reproduction** `w12c-t04-pre-20260926T212047Z` (1/1 FAIL):
  - `jope-arena leaderboard::tests::a_duplicate_cell_delivery_is_refused_not_silently_replaced`
    delivers `(d1, a)` twice. The second value turns a's win on `d1` into a loss.
  - `build_leaderboard` accepted it. The map insert replaced the first value, and the ladder came
    out `["b", "a"]` ("a duplicate (dataset, method) cell was silently replaced").
  - `rankable_rows` also counted both deliveries.
- **Fix.** `build_leaderboard` refuses a second rankable delivery of one `(dataset, method)`
  cell with "duplicate delivery of {dataset}/{method} cannot add or replace a match". It does
  so in both ladders, before the benchmark-valid filter, so a duplicate the valid ladder would
  drop is still refused. This is the independent replay's rule
  (`jope_records::elo_history::parse_scores`), so the port and the replay now agree.
  Non-rankable rows (not `ok`) are unchanged.
- **Controls.**
  - Positive: one delivery per cell ranks as before.
  - Every existing `jope-arena` test passes, including the reference-shape and maximize ladders
    and the committed-asset snapshot tests.
  - The `jope-records` Elo-history replays pass, so no retained ladder contained a duplicate.
- **Negative controls** (all refused):
  - the flipping duplicate, in the published ladder and in the benchmark-valid ladder;
  - a benchmark-invalid duplicate under the benchmark-valid ladder.
- **Runs.**
  - Pre `w12c-t04-pre-20260926T212047Z`: 1/1 FAIL as above.
  - Post `w12c-t04-post-20260926T212117Z`: `jope-arena` 41/41 (all targets) and the `jope-records` Elo-history replays 23/23.
  - Pre-commit `w12c-t04-c-20260926T212412Z`: fence suites and `gp::fit_contract_tests` 85/85; workspace all-targets check, coverage (0 unassigned), verify-test-inventory, fmt and diff-check exit 0.
  - The first gate, `w12c-t04-c-20260926T212221Z`, is kept. It failed one fence test unrelated to this change, `fit_boundary::isolated_model_export_obeys_the_same_authority_fence` (6.3 s under load). That test passed 3/3 in isolation (`w12c-t04-fence-rerun-20260926T212356Z`, 2.5 s each) and in the full rerun. It is a load-sensitive flake of the authority fence fixture, not a change in the fence.
- Exposure edges are still not retained (section 5). `S08-G09` and `S08-K09` stay open.
