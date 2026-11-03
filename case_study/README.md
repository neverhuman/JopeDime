# Case studies: tuning programs for what a business pays for

Each directory here is one named, self-contained research item on a real regression dataset from
our corpus. Each runs unchanged through the product program search, `jope pipeline run --fitness`.
A case asks one business question, such as *"which program gives the best day-ahead price
forecast that can be retrained in under 30 s?"*. It answers by ranking candidate programs with a
**resource-aware objective**: accuracy traded against, or limited by, what the candidate cost to
train and to serve. The variables are listed in
[docs/regression-objectives.md](../docs/regression-objectives.md#resource-aware-objectives-training-time-latency-and-memory).

## Layout of a case

| File | Contents |
| --- | --- |
| `README.md` | Business question, dataset provenance, why this objective, arms, how to read the result, caveats. |
| `case.json` | `jope.case-study/v1` manifest: the dataset id, budget, RNG seeds, seed limit and arms. |
| `objective.json` | The resource-aware objective (`jope.regression-fitness/v1`). |
| `baseline.json` | Accuracy-only ranking with non-binding limits, so the same resource readings are recorded for comparison. |
| `seeds.dl.json` | Generation-0 seeds proposed by our deep-learning model (`jope.pipeline-seeds/v1`), when produced. |
| `seeds.provenance.json` | Exactly which model, checkpoint and inputs produced the seeds, or which fallback did. |

## Arms

Every case runs a 2×2 design on one compute-matched evaluation budget:

| Arm | Generation 0 | Ranked by |
| --- | --- | --- |
| A | bootstrap + 11 random draws + DL seeds | `objective.json` |
| B | bootstrap + 11 random draws | `objective.json` |
| C | bootstrap + 11 random draws + DL seeds | `baseline.json` |
| D | bootstrap + 11 random draws | `baseline.json` |

At one `--rng-seed` the random draws are identical in every arm. A vs B isolates the
deep-learning seeds, A vs C isolates the objective, and C vs D is the seeds' effect on plain
accuracy. Each later generation breeds 8 children from the 4 best, so an evaluation budget of
`12 + seeds + 8·G` buys about G generations. The default budget gives 8–10.

## Running

```sh
cargo build --locked --release -p jope-cli --bin jope          # once per lane
case_study/bin/check_sealed.sh 127a1b5eccc09a84                # refuses sealed/held-out data
case_study/bin/run_case.sh case_study/01_energy_price_fast_retrain --smoke   # one seed, tiny budget
case_study/bin/run_case.sh case_study/01_energy_price_fast_retrain           # all arms and seeds
case_study/bin/run_all.sh calibrate --jobs 8      # set every case's limits/prices (declared rule)
case_study/bin/run_all.sh main --jobs 8           # every case x arm x seed, 8 searches at a time
```

`calibrate.sh` turns a case's `objective_template` into `objective.json`. It uses a calibration run
of arm D on its own RNG seed (never a main seed). The rule is fixed in advance:

- **Limits:** the median reading over the calibration candidates, to 2 significant figures, so a
  limit binds for about half of them.
- **Composite resource terms:** scaled by that median reading, so a candidate at the median pays
  0.1 of a train-target σ.

`case.json` records the calibration run and its observations hash.

`run_case.sh` behaves as follows:

- **Data.** It fetches the dataset into the local cache on first use (`AWS_PROFILE`, default
  `veox-jepson-prod`).
- **Sealed data.** It refuses sealed families.
- **Resources.** Every run uses `--threads 1`, because measured readings are only per-candidate
  while one candidate runs at a time. It also applies `nice -n 19` and one numeric thread
  (optionally `JOPE_CASE_CPUS` for taskset).
- **Verification.** After each run it calls `jope pipeline verify-receipt --evidence`.
- **Output.** It writes everything under `~/jope-data/case_study/<case>/<UTC>/` (never into the
  repository or `/tmp`):
  - `run.json`: the source commit, binary hash, budget and limits;
  - `runs.jsonl`: the exit and verification status of every run;
  - `generations.csv`;
  - `SUMMARY.md`.

`summarize.sh` reports, per generation, the best-so-far objective, and the accuracy and readings
of the best-so-far program.

## Extending, comparing and reporting

```sh
case_study/bin/run_all.sh extended --cases "01 03" --rounds 3     # chained rounds per arm-A seed
case_study/bin/collect.sh                 # copy runs from the other hosts (never overwrites)
case_study/bin/compare.sh --extended "<stamps>"     # ours vs alternatives, same objective
case_study/bin/leaderboard.sh --extended "<stamps>" # cases ranked by leakage risk, then margin
case_study/bin/final_test.sh <case>       # the single pre-declared test look (after TEST WINDOW)
```

- `extend_case.sh` chains rounds of up to 1000 evaluations or 600 s each. Round r+1 starts from
  the 11 best feasible programs found so far, and every round is a verified receipt.
- `extend_case.sh --cuts 5` ranks every program by its mean over the 5 folds of `train.csv` (the
  pipeline's cross-cut selection): a train-only cross-validated KPI. `test.csv` is never read.
- Arm **E** seeds generation 0 with [`seeds.bestpractice.json`](seeds.bestpractice.json): our
  grammar's equivalents of the XGBoost, LightGBM and scikit-learn defaults, the same for every case.
- The alternatives (CatBoost, LightGBM, XGBoost, HistGB, AutoGluon, TabPFN v2, Kumo Tabular,
  and classical models) are fitted outside the repository on the identical cut-0 rows. Each is
  scored by `jopedime fitness evaluate` under the case's own `objective.json`.
- `compare.sh` and `leaderboard.sh` read only those score files and our receipts and observations.
- The interim results are summarized in the
  [case-study brief](../docs/case-study-brief/README.md) (not the white paper). It is regenerated
  from the evidence with `make -C docs/case-study-brief`.

## Rules every case keeps

- **Sealed data.** Only datasets that `check_sealed.sh` clears: no sealed meta-test family, no
  JRB-01 exclusion, no private benchmark panel member.
- **Test split.** `test.csv` is never read by the search. Scoring the test split is one look per
  (dataset, genome) through `jope pipeline score --split test`, recorded in its registry.
- **Seed provenance.** A seed file states its true source. Seeds that came from the champion bank
  or a fallback are labelled as such, never as deep-learning proposals.
- **Interpreting readings.** Measured readings depend on the host and its load. Compare arms run
  together on one host, report the host load recorded in `run.json`, and prefer the structural
  proxies (`model_bytes`, `program_nodes`) for claims that must reproduce elsewhere.
- **What results establish.** These are engineering demonstrations of the tuning capability. They
  carry no learning-gain claim. A claim about the DL model needs the preregistered protocol in
  `research/flywheel_selfplay/PREREG_TEMPLATE.md`.
- **Validation order.** `jope pipeline run` validates on an unordered fold partition of
  `train.csv`. Cases on time-ordered data say so, and do not claim chronological (regime)
  validation.

## Known limitation and next work

A `fit_seconds` limit refuses an over-budget program only after its fit finishes, so the search
still pays for it. In the first calibration run, one random generation-0 program took 365 s of a
388 s run. The fix is a preemptive per-candidate fit deadline derived from the objective's limit:
a cooperative stage-boundary check in feat-math plus a watchdog for blocking linear algebra.
The feat-math part moves the ADR-0004 evaluator identity, so it is separate work. Design note by
agy_resource_fitness: `~/jope-data/resfit-2071/PREEMPTIVE_FIT_DEADLINE_DESIGN.md` on xbabe2.

## Cases

| Case | Dataset | Business constraint | Objective |
| --- | --- | --- | --- |
| [`01_energy_price_fast_retrain`](01_energy_price_fast_retrain/README.md) | Polish Day-Ahead Electricity Market, Demand, Renewables and Weather (`127a1b5eccc09a84`, 9901×8) | fit_seconds max | `constrained` on `pinball` |
| [`02_energy_price_hourly_retrain_cost`](02_energy_price_hourly_retrain_cost/README.md) | Hourly Day-Ahead Electricity Prices with Exogenous Variables (`14bc1b3a41c041b5`, 6720×11) | price per CPU second | `composite` on `rmse` |
| [`03_taxi_fare_online_latency`](03_taxi_fare_online_latency/README.md) | NYC Green Taxi Trip Sample with Fare and Distance (`7394ac9e4f167075`, 10000×11) | row_latency_p99_us max | `constrained` on `rmse` |
| [`04_ct_slice_wide_training_memory`](04_ct_slice_wide_training_memory/README.md) | Relative Location of CT Slices on Axial Images (UCI 206 slice_localization) (`2056a3aad9468d87`, 50000×384) | price per heap byte | `composite` on `rmse` |
| [`05_consumer_finance_edge_model`](05_consumer_finance_edge_model/README.md) | 2022 Survey of Consumer Finances — Summary Extract (`9721aa73ef1df237`, 10000×346) | model_bytes max | `constrained` on `rmse` |
| [`06_etf_frequent_retrain`](06_etf_frequent_retrain/README.md) | Taiwan 0050 ETF Historical OHLCV (`f285fd52a7c14111`, 10000×3) | price per training second | `composite` on `rmse` |
| [`07_loan_rate_batch_throughput`](07_loan_rate_batch_throughput/README.md) | OpenIntro loans_full_schema (Lending Club), interest_rate (`18f905fa597f92d3`, 10000×41) | predict_rows_per_second min | `constrained` on `mae` |
| [`08_hotel_rate_serving_cost`](08_hotel_rate_serving_cost/README.md) | modeldata hotel_rates, avg_price_per_room (`78dc8d077285ce09`, 15402×26) | serving cost per row | `custom` on `mae` |
| [`09_ultrawide_parsimony`](09_ultrawide_parsimony/README.md) | mtp2 (`9ae61bc7da3ba01c`, 274×1142) | price per program node | `composite` on `rmse` |
| [`10_city_power_realtime_dispatch`](10_city_power_realtime_dispatch/README.md) | Power Consumption of Tetouan City with Weather (`ca8b2f30cf07dd87`, 10000×7) | row_latency_p50_us max + fit_seconds max | `constrained` on `rmse` |
| [`11_bike_rebalancing_latency`](11_bike_rebalancing_latency/README.md) | Bike Sharing Hour (`bbc0bb2e8fcf2ba6`, 1999×16) | row_latency_p99_us max | `constrained` on `asymmetric_absolute` |
| [`12_hdb_resale_valuation_api`](12_hdb_resale_valuation_api/README.md) | Singapore HDB Resale Flat Prices from January 2017 (`f2d74feab0307e24`, 10000×7) | row_latency_p99_us max + model_bytes max | `constrained` on `mape` |
| [`13_loan_default_wide_memory`](13_loan_default_wide_memory/README.md) | LoanDefaultPrediction (`053a91a333d4f3a7`, 1999×735) | fit_peak_heap_bytes max | `constrained` on `rmse` |
| [`14_transformer_load_edge_device`](14_transformer_load_edge_device/README.md) | Electricity Transformer Temperature and Load Measurements (ETTm1) (`6ce1759b1a18e332`, 10000×6) | model_bytes max + row_latency_p99_us max | `constrained` on `rmse` |
