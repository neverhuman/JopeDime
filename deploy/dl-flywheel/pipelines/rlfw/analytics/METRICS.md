# D9 measurement record (CX-MET, BUILD ONLY)

> **Generated outputs live outside the tree.** The parquet partitions and HTML dashboards this page
> describes were removed from the source tree on 2026-10-21 (about 30 MB of regenerable output; the
> repository keeps bulk data out of working trees). Links below open them at commit
> [`ca6e15f1c`](https://github.com/neverhuman/JopeDime/tree/ca6e15f1c4dad721987f30825fe19b50bbf4e408/deploy/dl-flywheel/pipelines/rlfw/analytics/), the last one that carries them; the
> `*SNAPSHOT.json` receipts beside this page keep their paths and SHA-256 digests.

Daily Parquet, a static HTML dashboard, and read-only receipt adapters live here.
This package measures retained Lane A observations. It does not execute programs,
drain an outbox, launch a learner, publish to S3, spend a seal, or promote a model.
Missing measurements are Parquet NULL and display as **unmeasured**.

Open [dashboard.html](https://github.com/neverhuman/JopeDime/blob/ca6e15f1c4dad721987f30825fe19b50bbf4e408/deploy/dl-flywheel/pipelines/rlfw/analytics/dashboard.html); its rows come from the committed
[daily/2026-10-09.parquet](https://github.com/neverhuman/JopeDime/blob/ca6e15f1c4dad721987f30825fe19b50bbf4e408/deploy/dl-flywheel/pipelines/rlfw/analytics/daily/2026-10-09.parquet) and
[daily/2026-10-10.parquet](https://github.com/neverhuman/JopeDime/blob/ca6e15f1c4dad721987f30825fe19b50bbf4e408/deploy/dl-flywheel/pipelines/rlfw/analytics/daily/2026-10-10.parquet) readback.
[SNAPSHOT.json](SNAPSHOT.json) pins 1,116 rows and the full custody receipt.
The extraction implementation commit is
`cbeca37700daf9404afd4b9fec3414573b919462`. Later optional rolling-adapter
repairs do not change this retained extraction; its exact code is pinned separately.

## Retained measurements: 2026-10-09

The retained extraction (`.agent/rlfw-coord/lanes/CX-MET/run-final-s3/analytics/`)
read four retained D5.1 analysis outboxes. These are copies of completed/failed
producer snapshots, not live SQLite handles. The fixed denominator is the
unmodified `BENCH/cache/D5-kept-core53.json:1` roster: 53 contexts / 53 parents.
`BENCH` means `/home/ubuntu/jopedime_probe_exp/rlfw-20261009`.

| Measurement | Observed value | Evidence limit |
|---|---:|---|
| Comparable program/incumbent fold pairs | 1,034 | Unique comparison identity + program; fixed core53 only |
| Median `SE_pair` | 0.05296631947 | 2,000 paired bootstrap draws per pair |
| Median `abs(d0-d1)` | 0.07298555377 | Fold log RMSE ratios against the frozen incumbent |
| Median bootstrap SE | 0.04543960570 | Saved inner-fold predictions and targets only |
| Median `tau_c=max(.01,1.96*SE_pair)` | 0.1038139862 | Exploratory uncertainty; no teacher promotion |
| Positive observed gain lower bound, fixed53 parent macro | 0.001712830189 q | Unresolved/censored programs may increase the maximum; full ΔH unmeasured |
| Observed distinct useful program-parent pairs | 6 | Above frozen comparator; lifetime novelty and U per second unmeasured |
| Native compile phase median | 0.874378221 s | 1,883 retained unique noncached qualified results; selective cohort |
| Native execution fold 0 / fold 1 medians | 0.169979888 / 0.163582913 s | Phase service durations; no concurrency speed claim |

Numerical sources: `run-final-s3/analytics/daily/2026-10-09.parquet` pair and summary
rows, generated from `run-final-s3/analytics/observations.jsonl:1` and the hash/length
pinned evaluation references recorded there. Raw metadata and producer identity:
`BENCH/runs/d5b/analysis-owner1620-1791589225/analysis.json:1`, source commit
`1aea2c1041aa8d859ec9b475a28d73d1cc2c1e04`. Producer baseline source commit:
`586c4ae7abad47d1c578fc5b07a0497385865ddc`. Exact input SHA256/byte counts are
retained in each extraction's `receipt.json`, not inferred from filenames.

| Producer | Physical responses / elapsed seconds | Responses/s | Physical censoring including missing responses |
|---|---:|---:|---:|
| baseline_T0_D5b | 278 / 919.2681496 | 0.3024144806 | 31 / 280 = 11.0714% |
| baseline_T1_D5b | 759 / 2884.677469 | 0.2631143371 | 53 / 767 = 6.9100% |
| baseline_extended_D5b | 730 / 4359.833025 | 0.1674376050 | 51 / 734 = 6.9482% |
| rerank_T1_D5b | 306 / 1543.381559 | 0.1982659429 | 15 / 306 = 4.9020% |

Source: the same `analysis.json:1` physical audit; elapsed/status source is
`BENCH/runs/d5b/<producer without _D5b>/completion.json:1` at the producer
commits above. `baseline_extended` **failed** with an AssertionError; its work
and elapsed cost remain in the table. Excluded work remains in physical totals;
imported parent work is not charged as new physical work. Retained logical slots
have zero censor flags because censored contexts were excluded before those
final slot exports. The dashboard reports logical and physical rates separately.

## Definitions and missing evidence

`metrics.py:49` computes parent-macro U, G and ΔH in a half-open UTC window with
an explicit fixed context roster. Comparisons must use the same complete HWM
identity; predicted q never enters the input. U counts distinct verified program
SHAs that clear the frozen comparator and have a lifetime first-seen proof. G
counts actual `improvement` ledger events, excluding `first_label`, divided by
per-parent evaluator **cgroup** CPU seconds / 3600. Process and child CPU timing,
worker counts and elapsed wall time cannot replace that denominator. ΔH averages
positive gains within each parent, then across the fixed parent roster. Missing
comparator or unresolved context coverage keeps the complete metric unmeasured.

The current retained snapshot does not supply lifetime novelty, a dated live
research HWM ledger/cgroup join, or an actual generated-result learner exposure.
Therefore U_useful, G, complete ΔH, T_feedback and each admission-ladder stage
latency are **unmeasured**. The observed gain lower bound is separately labelled.
Production bytes added and USD/day remain unmeasured. G-S3GP supplied its
seven-row role census: **20,833 objects / 18,178,859,605 bytes**, with a provider
storage estimate of **$0.418113770915/month** for that single objects prefix.
This is observed storage stock, not production traffic or daily billing. Source:
G-S3GP `rlfw-grok/receipts/objects-census.parquet` (SHA256
`0be916761ae2808d4742d3ae0dc7647e9809866fda5ea11998de94c181ffd91d`)
and `rlfw-grok/receipts/dry-run-v1.json:1`, provider source commit
`50969b90aafc997aac7fd1eeb3272b0853383835`. The provider ledger event binds
the reporting day 2026-10-10; it supplies no census wall-clock timestamp.
CX-MET consumes sealed local copies and implements no S3 lister or pricing model.

`metrics.py:132` recomputes fold RMSE from saved vectors, checks ordered targets,
and performs 2,000 paired row resamples within each inner fold. It sets
`SE_pair=max(bootstrap_SE,abs(d0-d1)/2)`. The floor comes from the existing
`continuous/consumer/dltrain/labels.py:82` robust-scale and `:93` epsilon rule,
SHA256 `c2554a33554e69ee36f7c9eeab094989609c212b9e5c954ccf6f712c35b0befd`,
identical to BENCH's frozen labels copy and product base commit
`edf36182b3fb3254afd244c9be1f1f6f274b9e0c`. Seed is the first 64 bits of the
SHA256 of context+challenger SHA. This is conditional row-resampling uncertainty
on the retained two-fold cohort, not refit-split robustness or a generalization
claim. No outer test rows or labels are decoded.

`metrics.py:205` joins only explicit proposal/context/program/result-ref/snapshot/
term exposure identities. Admission is not first exposure. Missing endpoints
remain null and proposals without exposure remain right censored; stage clocks
out of order are refused. `windows.py:12` consumes hash-pinned owner exports for
registered rolling 1 h / 24 h windows and the merged `research_hwm` event schema.
Proposal cohorts are selected by commit time in the half-open window. T uses
equal context/parent weights and remains null with missing or censored joins;
completed-only p50 and stage quantiles are separate diagnostics.
That ledger schema has no publication clock, so the adapter requires an exact
`(hwm_key,version)` UTC receipt join and does not invent timestamps from mtime.
Any undated qualified publication for a fixed HWM identity keeps U, G and ΔH
unmeasured: its membership in the window is unknown, even with CPU receipts.

Phase p50/p95/p99 and `samples/sum(service seconds)` use individual unique
noncached receipts. The serial service rate and actual producer responses/wall-s
have different denominators. Nested phases are never added together.
Gate history records retained directional Lane A verdicts (including
INCONCLUSIVE); these observations are not G2/G3 gates or promotions.

## Reproduce safely

From the lane root, use the locked Python environment (numpy and pyarrow already
appear in `python/pyproject.toml`). Choose new output/copy directories under
CX-MET custody, outside BENCH and outside any live service directory:

```sh
systemd-run --user --scope -p CPUQuota=200% -p MemoryMax=4G \
  env PYTHONPATH=deploy/dl-flywheel/pipelines PYTHONDONTWRITEBYTECODE=1 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python3 -m rlfw.analytics.build \
  --bench /home/ubuntu/jopedime_probe_exp/rlfw-20261009 \
  --retained /home/ubuntu/jopedime_probe_exp/rlfw-20261009/runs/d5b/analysis-owner1620-1791589225 \
  --copies /home/ubuntu/JopeDime/.agent/rlfw-coord/lanes/CX-MET/NEW-RUN/inputs \
  --output /home/ubuntu/JopeDime/.agent/rlfw-coord/lanes/CX-MET/NEW-RUN/analytics
```

The CLI reads back actual cgroup limits and refuses uncapped builds. Copies use
content-addressed exclusive creation, source/copy SHA validation, fsync and
read-only permissions. SQLite opens use `mode=ro&immutable=1` plus `query_only`;
WAL/journal-bearing sources require an owner's quiescent export. Dashboard HTML
is generated from the Parquet **readback** and includes no external resources.
Raw observations and input copies remain outside Git; only compact analytics
snapshots belong here. New daily outputs refuse to overwrite prior partitions.

Optional `--s3-export` accepts G-S3GP's local daily Parquet columns
`date,bytes,usd,cost_kind,source`; cost_kind is estimate/billed/unmeasured and
NULL cost stays unmeasured. `--s3-census-manifest` additionally consumes the
provider role census plus pinned report/ledger, validates matching totals, and
keeps its monthly estimate separate from daily production costs. Export subtotals must be disjoint, not overlapping
bucket and prefix totals. Optional `--window-manifest` has schema
`jope.rlfw-measurement-input/v1`, pins contexts/timeline/ledger/feedback files
by path/SHA256/bytes, and supplies named start/end windows with optional pinned
`jope.evaluator-cgroup-window/v1` per-parent CPU receipts. See the isolated
`tests/test_io.py` end-to-end ledger fixture for the exact schema.

Acceptance: `analytics/run_offline.py --receipt <owned path>` under the capped
scope (set TMPDIR to owned custody), source manifest/classification validation,
`git diff --check`, the repository required gate and exact-head CI. Tests cover
both refusal and authorized successful paths. A configured pipeline, measured
retained cohort, green CI, and a qualified live learning loop remain separate.

## Owner 19:50Z scratch-run ingest: 2026-10-10

The owner-authorized CPU receipt ingest adds [scratch-dashboard.html](https://github.com/neverhuman/JopeDime/blob/ca6e15f1c4dad721987f30825fe19b50bbf4e408/deploy/dl-flywheel/pipelines/rlfw/analytics/scratch-dashboard.html)
with per-arm validation-loss and KL curves alongside the retained D9 dashboard.
[scratch-daily/2026-10-10.parquet](https://github.com/neverhuman/JopeDime/blob/ca6e15f1c4dad721987f30825fe19b50bbf4e408/deploy/dl-flywheel/pipelines/rlfw/analytics/scratch-daily/2026-10-10.parquet) has **349 rows**
from **20 distinct receipt/partial sources**, observed at 20:09Z; no scoring or
training runs execute in this analytics lane. [SCRATCH_SNAPSHOT.json](SCRATCH_SNAPSHOT.json)
pins the extraction code commit `d7919d669a6232fd5d578590b4b307c5e3681894`, raw input
copies, JSONL companions, provider table, output hashes and actual kernel caps.
This snapshot is NON-QUALIFYING, benchmark_valid=false, promotion_input=false.

| Run / final epoch / arm | LPI validation loss (98 contexts) | L6 panel loss (16) | KL mean to anchor (64 probes) |
|---|---:|---:|---:|
| CL run1 / 1 / control | 6.851773111 | 0.870878999 | 0.030999528 |
| CL run1 / 1 / LPI | 4.482191993 | 0.885834814 | 0.969761412 (disqualified) |
| CL run1 / 1 / calibrated LPI | 5.702575065 | 0.872007923 | 0.082915115 (disqualified) |
| CL sweep2 / 3 / control | 6.909926911 | 0.869012540 | 0.027887610 |
| CL sweep2 / 3 / LPI .01 | 6.327138563 | 0.869949479 | 0.006917915 |
| CL sweep2 / 3 / LPI .03 | 5.920710794 | 0.870727482 | 0.030059133 |

Source run1: `/home/ubuntu/JopeDime/.agent/proposals/claude-rlfw-20261009/rlfw-takeover/lpi-scratch/receipts/runs/claude-rlfw-lpi-scratch-20261010T174612Z/receipt.json:125`, SHA256 `1f85f3b69b45643e7dadc64a6c2c4736a8fd3d1811c5f556adbaaa93294b0298`.
Source sweep2: `/home/ubuntu/JopeDime/.agent/proposals/claude-rlfw-20261009/rlfw-takeover/lpi-scratch/frozen/sweep2/claude-rlfw-lpi-scratch-sweep2-20261010T181758Z/receipt.json:107`, SHA256 `532032d39c2afc9d5d3ae1156f4c92dfdaa278342815433b6fec2a8638666835`.
CL receipts do not assert a Git source commit; that provider commit is unmeasured.
Harness and input identities are retained separately in the copied closure and receipt.

Run1 reports **0.224825876 GPU wall-hours**; sweep2 reports **0.572977312 GPU wall-hours**.
Each is charged once for the shared run, without dividing across arms or claiming
GPU utilization. Run-bound cgroup CPU meters are learner costs; **G per evaluator
core-hour remains unmeasured** without a qualifying HWM improvement/evaluator CPU join.
Failed OP receipts preserve their reported costs and error; absent final update
counts remain unknown. Same-run loss deltas require matching epoch, sample count
and reported update budget; missing joins are NULL. Smoke and synthetic fixture
observations remain labelled and never establish real-pair or qualification coverage.

The retained median **SE_pair=0.05296631947 q** is displayed as a reference.
`1.96*median(SE_pair)=0.1038139862 q` describes that retained cohort only.
It is not the paired uncertainty of validation loss, KL, M2 ranking, or a new q
comparison. Their resolvability is **unmeasured**. A q noise screen requires a
physical SHA/length-pinned paired evidence export, matching context/dataset/cut/
evaluator/vocabulary/protocol and challenger/incumbent identities, its own SE,
and the same q unit. No scratch receipt currently supplies that evidence; no
scratch win, family gate, or improvement rate is inferred from these curves.

CL sweep3 and OP RLOO/PKPO live JSONL are copied by read-only SSH stat/hash/cat/hash
from their assigned xbabe3 metadata roots. Changed files are deferred and labelled
unmeasured; partial prefixes keep observed epochs but lack final costs/closure.
G-ABL real-pair receipts use a native adapter: training loss is not validation
loss, and labelled-support KL is separate from full-trie KL. Before such a receipt
lands, real-pair validation and costs stay unmeasured. CX-KV EVAL_TABLE is preserved
as a raw provider copy; numeric checkpoint receipts report one-context development
KL and selected-support M2 across 66 contexts. Those are distinct from R22's 64
probes; their training epochs are unknown and never assigned epoch zero.

The bounded watcher `cx-met-scratch-watch-2012.scope`, PID1224020, started
2026-10-10T20:09:33Z with CPU200%, MemoryMax4GiB, swap0, external RuntimeMax7200s.
It refreshes every1800s and at its declared 2026-10-10T22:04:33Z deadline.
Future snapshots and their tables/curves are durable outside Git at
`.agent/rlfw-coord/lanes/CX-MET/scratch-ingest/watch/`; each pass logs the table to
COORD.md under its lock. The committed snapshot retains its original input pins.
The watcher pins source bytes and refuses changes during its run.

```sh
systemd-run --user --scope -p CPUQuota=200% -p MemoryMax=4G \
  -p MemorySwapMax=0 -p RuntimeMaxSec=7200 \
  env CUDA_VISIBLE_DEVICES= PYTHONPATH=deploy/dl-flywheel/pipelines \
  PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  uv run --project python --locked --extra cpu python -m rlfw.analytics.build_scratch \
  --manifest /home/ubuntu/JopeDime/.agent/rlfw-coord/lanes/CX-MET/scratch-ingest/input.json \
  --output-root /home/ubuntu/JopeDime/.agent/rlfw-coord/lanes/CX-MET/NEW-SCRATCH-RUN \
  --watch-seconds 6900 --interval-seconds 1800 \
  --board /home/ubuntu/JopeDime/.agent/rlfw-coord/COORD.md
```

Analytics fixtures now collect and execute **39 tests**, with zero failures,
errors or skips. Source/test registration includes the scratch adapters and
`tests/test_scratch.py`; the repository required gate and changed-head CI/review
remain separate evidence. Source implementation is in `build_scratch.py` and
`scratch.py` at the extraction commit above; the snapshot inventory binds every
native provider receipt to its copied hash rather than assuming its filename.

## Owner 20:10Z update: measured context variance and scope costs

The initial 20:09 watcher and its local gate were stopped only on this lane's
CPU scopes (exit143, not PASS) to apply the owner's logging/paired-cost update.
GPU jobs were not interrupted. Future ingests emit **one** COORD summary, with
count, path and headline deltas; per-arm tables remain in HTML/Parquet. The
original 22:04:33Z watch deadline is preserved by `--deadline-unix`.

The collector now consumes CL's native paired index and per-context rows, plus
CX-OP's [context-costs.json contract](CONTEXT_EXPORT.md). `context_costs.py`
computes sample between-context paired-delta variance, each arm's cost variance,
SE of the paired mean, normal CI, and an exploratory n for delta=.005,
alpha=.05/power=.80. Epochs are separate cohorts and never increase independent n.
Validation, unused-training prefixes and full unused-training cohorts remain
separate. Missing/mismatched coverage leaves full-roster statistics unmeasured.

On the native sweep3 epoch-1 **263 validation contexts**, weighted contributions
reconcile the saved L6 macro. Control-minus-arm values are:

| Arm | Paired mean delta | Sample variance of paired delta | Mean SE | Normal n estimate for .005 |
|---|---:|---:|---:|---:|
| LPI .01 | +0.00009719383 | 0.000004339198116 | 0.00012844786 | 2 |
| LPI .03 | -0.0005873106 | 0.00001529178275 | 0.00024112997 | 5 |
| LPI .10 | -0.002314618 | 0.00006029175848 | 0.00047879665 | 19 |

These are observed-variance normal approximations assuming independent contexts.
They replace the fold-q stand-in for this specific L6 contribution definition;
they do not qualify a family gate or establish power for another effect size.
At .01 the exploratory CI crosses zero; the .03/.10 CIs are below zero. G-JUDGE
owns verdicts and no seal is spent here. Raw context loss means are reported
separately (`l6_raw_*`); they differ from supported-term pooled L6 costs. Native
missing protocol/input identity hashes remain explicitly unmeasured.

The first measured read is pinned in custody at
`lanes/CX-MET/scratch-ingest/context-measured/20261010T202635Z-000/receipt.json:1`
and `measured-power-readback.json:1`; its exact source bytes are in `watch.json`.
Native input: xbabe3
`/home/ubuntu/claude-rlfw-lane/rlfw-takeover/lpi-scratch/run/claude-rlfw-lpi-scratch-sweep3-20261010T195546Z-paired/index.json:1`,
SHA256 `3cb805a52eccd835ce1f008f88f8de91675e0c9c9778f1651dec0bf2affb3d29`;
verified JSONL prefix SHA256
`c1f938da610667acd0cc20f47d369f370c71a287a4c73573c0f460ef58ebfaac`.
This preliminary read preceded the committed extraction; the refreshed committed
snapshot separately pins an exact code commit and identical epoch-1 statistics.

Read-only systemctl properties additionally capture actual scope UTC start/stop
or observed time and cumulative CPUUsageNSec. At the 20:27Z read, assigned GPU
scope occupancy was CL sweep3 **0.521608814h**, G-ABL **0.387899595h**, OP R4
**0.146071155h**, all partial. These include launch/init time and are alternative
measures to reported process wall-hours: never sum them. RuntimeMax is a resource
cap, not elapsed time. Removed OP R3 unit properties remain unmeasured; its
reported failed-run process cost is retained. Scope metadata/source hashes live
in that extraction's copied inputs and Parquet rows.

Growing JSONL streams use a twice-verified complete-line prefix snapshot,
retaining byte count/hash and deferring rewritten prefixes. Source files stay
read-only. The collector never imports the trainer, loads weights, or opens a
live writable DB. Per-context pairs stay in Parquet while dashboard tables show
cohort summaries, validation/KL curves and costs. G/evaluator-core-hour remains
unmeasured because there is no qualifying HWM/evaluator CPU join.

Committed refresh: [context-dashboard.html](https://github.com/neverhuman/JopeDime/blob/ca6e15f1c4dad721987f30825fe19b50bbf4e408/deploy/dl-flywheel/pipelines/rlfw/analytics/context-dashboard.html),
[context-daily/2026-10-10.parquet](https://github.com/neverhuman/JopeDime/blob/ca6e15f1c4dad721987f30825fe19b50bbf4e408/deploy/dl-flywheel/pipelines/rlfw/analytics/context-daily/2026-10-10.parquet), and
[CONTEXT_SNAPSHOT.json](CONTEXT_SNAPSHOT.json) pin 13296 rows at 2026-10-10T20:32:52.817730+00:00
with exact extraction source `430f8b377569dd4007cfe706566329b9040ffbce`.
Source/fixture coverage is now 52 tests, all executed with zero errors/failures/skips.
The restarted watcher is `cx-met-scratch-watch-v3-2035.scope`, PID2376817,
CPU200%/4GiB/swap0/Runtime5711.0s; original declared22:04:33Z
and outer22:09:33Z deadlines retained. In-place runtime update failed; only the
CPU watcher was replaced with the remaining launch budget. No GPU jobs changed.

## Owner 2026-10-11 19:30Z refresh

[dashboard-20261011.html](https://github.com/neverhuman/JopeDime/blob/ca6e15f1c4dad721987f30825fe19b50bbf4e408/deploy/dl-flywheel/pipelines/rlfw/analytics/dashboard-20261011.html) reads back **148,485 rows** from the new [daily-20261011/](https://github.com/neverhuman/JopeDime/tree/ca6e15f1c4dad721987f30825fe19b50bbf4e408/deploy/dl-flywheel/pipelines/rlfw/analytics/daily-20261011/) partitions. The snapshot observes provider metadata through **2026-10-11T19:56:05.102008+00:00**, including **147,369 scratch rows**, **1,116 retained D9 rows** and **1,034 retained fold pairs**. [DAY_20261011_SNAPSHOT.json](DAY_20261011_SNAPSHOT.json) binds every physical copy, output hash, kernel limit, successful driver and clean extraction commit `144b6041a8ca6a6c12d09bd4158b24232d5d3eab`. The large daily partition is stored as four shards below the repository's 5 MiB file limit; a row multiset hash proves that every record is retained. Older committed snapshots remain dated observations.

The refresh consumes G-ABL equal/tau/tau0/tau0-s11 receipts, OP R7/R8 terminals and R9 running receipts/paired metadata, current CX-KV and OP native KV inference, and CL sweep6 `773040d562805fc7ad11a254dab44fe10c8bdbe367a5a3e5465e514ec909d11c`. CL source and jobs are unchanged. The native indexed join checks ordered roster hashes, checkpoint parameters/updates and saved L6 macros; `r23_control` is now recognized as OP's comparator. The before/after failure is preserved in custody `r23-control-reproduction.json:1`.

| Latest cohort / arm | Epoch | Paired contexts | Control minus arm L6 | Paired variance | Mean SE | Exploratory required n |
|---|---:|---:|---:|---:|---:|---:|
| CL/claude-rlfw-lpi-scratch-sweep6-20261011T024759Z/set-a/vs-r23/control | 2 | 263 | -0.0008880623427 | 5.86548243e-05 | 0.0004722522055 | 19 |
| CL/claude-rlfw-lpi-scratch-sweep6-20261011T024759Z/set-a/vs-r23/r23_lpi01 | 7 | 263 | -0.0007935499296 | 2.657785221e-05 | 0.0003178938066 | 9 |
| CL/claude-rlfw-lpi-scratch-sweep6-20261011T024759Z/set-b/vs-r23/control | 2 | 1000 | -0.0001226925316 | 8.474621375e-06 | 9.205770677e-05 | 3 |
| CL/claude-rlfw-lpi-scratch-sweep6-20261011T024759Z/set-b/vs-r23/r23_lpi01 | 7 | 1000 | -0.0002450622655 | 2.945453449e-06 | 5.427203192e-05 | 1 |
| CL/claude-rlfw-lpi-scratch-sweep6-20261011T024759Z/set-heldout/vs-r23/control | 2 | 885 | -0.001478900035 | 1.378616228e-05 | 0.0001248101939 | 5 |
| CL/claude-rlfw-lpi-scratch-sweep6-20261011T024759Z/set-heldout/vs-r23/r23_lpi01 | 7 | 885 | -0.0001431444356 | 6.866873716e-06 | 8.808620403e-05 | 3 |
| CX-OP/cxop-r9-scratch-r23-20261011T184027Z/pkpo_b8 | 5 | 263 | 2.558551784e-08 | 4.872049289e-14 | 1.361062168e-08 | 1 |
| CX-OP/cxop-r9-scratch-r23-20261011T184027Z/rloo | 5 | 263 | 3.923873393e-05 | 2.943632442e-07 | 3.3455222e-05 | 1 |
| CX-OP/cxop-r7-r23-20261011T141653Z/pkpo_b8 | 8 | 263 | -3.895240371e-08 | 1.845686433e-13 | 2.649116671e-08 | 1 |
| CX-OP/cxop-r7-r23-20261011T141653Z/rloo | 8 | 263 | -0.0004203270354 | 1.662864841e-05 | 0.0002514493986 | 6 |
| CX-OP/cxop-r8-r23-20261011T161734Z/pkpo_b8 | 8 | 263 | -6.575823653e-09 | 2.71845294e-14 | 1.016677086e-08 | 1 |
| CX-OP/cxop-r8-r23-20261011T161734Z/rloo | 8 | 263 | 3.583254449e-05 | 3.31162011e-07 | 3.54848016e-05 | 1 |

Positive delta means lower measured cost for the arm. Required n uses the measured paired SD, delta .005, two-sided alpha .05 and power .80 under the stated independent-context normal approximation. These are exploratory single-cohort estimates; no family gate or replacement qualification threshold is asserted. Validation a (263), the b sample (1,000) and lineage-heldout (885) remain separate; repeated epochs do not add independent contexts. Unequal control/treatment updates remain unmeasured. The retained q `SE_pair=.05296631947` is a separate unit and cohort.

Numerical pairing sources (the full copied JSONL and indexes are pinned in the snapshot):

- `/home/ubuntu/JopeDime/.agent/rlfw-coord/lanes/CX-OP/scratch-train/r7/result/context-costs.json:1 sha256=a5e73aaa47a57445a0c421e1ac888f4edbb19fa75c562f3561301effaa7ceb03`
- `/home/ubuntu/JopeDime/.agent/rlfw-coord/lanes/CX-OP/scratch-train/r8/result/context-costs.json:1 sha256=d9d47b42debf778e3edcf9a12d57f13d1a5748f31dbc5425bd880d63c89d0cda`
- `/home/ubuntu/JopeDime/.agent/proposals/claude-rlfw-20261009/rlfw-takeover/lpi-scratch/x2/run/claude-rlfw-lpi-scratch-sweep6-20261011T024759Z/paired-index.json:1 index_sha256=ccb7edaf74471df223f09da30538235e8894ebd877c2ffdd577d8880e21bed60 rows_sha256=8b7aaeadc445ed2f54598e154f1433340ec3013f073c64478088f66a58b613ab checkpoint_index_sha256=f79f763c9936441e7b42eb7c0eaaaaac667589123988c85ad9808905bee8f7e4 parent_sha256=773040d562805fc7ad11a254dab44fe10c8bdbe367a5a3e5465e514ec909d11c`
- `xbabe3:/home/ubuntu/jope-evidence/rlfw-cx-op/scratch-train/r9-scratch/results/cxop-r9-scratch-r23-20261011T184027Z/context-costs.json:1 sha256=159cbb7ab307492016d440467ffd1ee05196b9a615520b45c0588ff0d54ddf03`

Computation: `context_costs.py:204` (paired moments) and `:240` (power), extraction commit `144b6041a8ca6a6c12d09bd4158b24232d5d3eab`. The snapshot's full receipt includes source-line and SHA identity for every provider; a provider Git commit that is absent stays unmeasured.

| Run cost / metric | Observed value | Unit |
|---|---:|---|
| CX-OP/r7/all arms shared / gpu_process_wall_hours | 1.748833577 | GPU wall-hours |
| CX-OP/r7/all arms shared / learner_cpu_core_hours | 1.748223229 | learner cgroup core-hours |
| CX-OP/r7/all arms shared / G_per_evaluator_core_hour | unmeasured | improvements/core-hour |
| CX-OP/r7/pkpo_b8 / optimizer_updates_reported | 1136 | updates |
| CX-OP/r7/r23_control / optimizer_updates_reported | 1136 | updates |
| CX-OP/r7/rloo / optimizer_updates_reported | 1136 | updates |
| CX-OP/r8/all arms shared / gpu_process_wall_hours | 1.812542595 | GPU wall-hours |
| CX-OP/r8/all arms shared / learner_cpu_core_hours | 1.812545372 | learner cgroup core-hours |
| CX-OP/r8/all arms shared / G_per_evaluator_core_hour | unmeasured | improvements/core-hour |
| CX-OP/r8/pkpo_b8 / optimizer_updates_reported | 1080 | updates |
| CX-OP/r8/r23_control / optimizer_updates_reported | 1080 | updates |
| CX-OP/r8/rloo / optimizer_updates_reported | 1080 | updates |
| CX-OP/r9-scratch/LIVE_STATUS/all arms shared / gpu_process_wall_hours | unmeasured | GPU wall-hours |
| CX-OP/r9-scratch/LIVE_STATUS/all arms shared / learner_cpu_core_hours | unmeasured | learner cgroup core-hours |
| CX-OP/r9-scratch/LIVE_STATUS/all arms shared / G_per_evaluator_core_hour | unmeasured | improvements/core-hour |
| CX-OP/r9-scratch/LIVE_STATUS/r23_control / optimizer_updates_reported | 747 | updates |
| CX-OP/r9-scratch/LIVE_STATUS/rloo / optimizer_updates_reported | 747 | updates |
| CX-OP/r9-scratch/LIVE_STATUS/pkpo_b8 / optimizer_updates_reported | 747 | updates |
| CX-OP/cxop-r9-scratch-r23-20261011T184027Z/all arms shared / gpu_process_wall_hours | unmeasured | GPU wall-hours |
| CX-OP/cxop-r9-scratch-r23-20261011T184027Z/all arms shared / learner_cpu_core_hours | unmeasured | learner cgroup core-hours |
| CX-OP/cxop-r9-scratch-r23-20261011T184027Z/all arms shared / G_per_evaluator_core_hour | unmeasured | improvements/core-hour |
| CX-OP/cxop-r9-scratch-r23-20261011T184027Z/r23_control / optimizer_updates_reported | 749 | updates |
| CX-OP/cxop-r9-scratch-r23-20261011T184027Z/rloo / optimizer_updates_reported | 749 | updates |
| CX-OP/cxop-r9-scratch-r23-20261011T184027Z/pkpo_b8 / optimizer_updates_reported | 749 | updates |

| Run | Recorded process wall seconds |
|---|---:|
| CX-OP/r7/all arms shared | 6295.8008787632 |
| CX-OP/r8/all arms shared | 6525.1533429623 |

Costs are charged once per shared run. GPU process wall-hours and scope occupancy are alternative descriptions, never added; scope CPU is the measured kernel counter, not CPUQuota times elapsed wall time. R7/R8 terminal CPU joins use copied TRAIN_EXEC_EXIT and final-scope-properties files pinned by `daily-20261011/closures/r7.json:1` and `r8.json:1`. Unclosed R9/G-ABL terminal totals and unavailable R5/R6 CPU joins remain unmeasured. These are run totals, not allocations of training to today's UTC day. G per evaluator core-hour still lacks a qualifying HWM/evaluator join.

At the snapshot observation, R9's still-running assigned scope reports **1.259456237 wall/GPU-assigned hours and 1.258521238 cumulative CPU core-hours**. The G-ABL DPO-seeds scope reports **0.3386246247 wall/GPU-assigned hours and 0.3372986422 CPU core-hours**. These are partial scope counters, including scope overhead, not closed training totals or GPU busy time. Sources are the physical scope-property exports in `DAY_20261011_SNAPSHOT.json` for `cxop-r9-scratch-r23-20261011T184027Z.scope` and `g-abl-r23-dpo-seeds-20261011T1935Z.scope`; transformation: `context_costs.py:598`, extraction commit `144b6041a8ca6a6c12d09bd4158b24232d5d3eab`.

| Host/device observation | Point value | Unit |
|---|---:|---|
| CX-MET/xbabe2-gpu-observation/GPU-44c63d19-492a-ab46-cc6a-56e12a2ef51b / gpu_utilization_sample_percent | 90 | % |
| CX-MET/xbabe2-gpu-observation/GPU-44c63d19-492a-ab46-cc6a-56e12a2ef51b / gpu_memory_used_sample_mb | 4387 | MiB |
| CX-MET/xbabe3-gpu-observation/GPU-2857566c-23c5-293a-df02-7c66a7571ac4 / gpu_utilization_sample_percent | 52 | % |
| CX-MET/xbabe3-gpu-observation/GPU-2857566c-23c5-293a-df02-7c66a7571ac4 / gpu_memory_used_sample_mb | 4642 | MiB |

GPU points come from copied read-only `nvidia-smi` metadata with command, UTC observation clock and raw CSV (`build_scratch.py:137`, same extraction commit). They include every host workload, not lane utilization. Full-day utilization is unmeasured without a time trace. No GPU job is started or interrupted by this lane.

G-S3GP's `s3-daily-20261011.parquet` reports **0 LastModified bytes in the provider's trailing 24-hour objects-prefix window; USD/day remains unmeasured**. This is not a stock total, production throughput or billed daily rate. Its exact copied hash/size is in the retained receipt; the dated 20,833-object / 18,178,859,605-byte stock census remains separate. CX-MET performs no S3 listing or write.

U_useful, qualified G, complete fixed-context ΔH and proposal-to-first-learner-exposure latency remain unmeasured where the required identities/clocks do not exist. Missing values are NULL, not zero. G-ABL tau0's provider objective metadata has an independent review finding; the original receipt bytes are preserved and its curves remain NON-QUALIFYING. No model win, qualification or promotion is asserted.

Validation before extraction: **58/58 analytics fixtures passed**, zero failures/errors/skips, with capped CPU200% / MemoryMax4GiB / swap0 scopes. The source and artifact commits still require the repository gate, current-head CI and fresh independent review; an approval of the earlier PR head does not cover this refresh.

## Owner 2026-10-11 20:20Z expert-review metrics

[dashboard-gap-20261011.html](https://github.com/neverhuman/JopeDime/blob/ca6e15f1c4dad721987f30825fe19b50bbf4e408/deploy/dl-flywheel/pipelines/rlfw/analytics/dashboard-gap-20261011.html) reads **198,859 rows** from [daily-gap-20261011/](https://github.com/neverhuman/JopeDime/tree/ca6e15f1c4dad721987f30825fe19b50bbf4e408/deploy/dl-flywheel/pipelines/rlfw/analytics/daily-gap-20261011/), including **44,349 operational/parent records**. Nine shards preserve the complete row multiset and every file stays below 5 MiB. [DAY_GAP_20261011_SNAPSHOT.json](DAY_GAP_20261011_SNAPSHOT.json) pins source code `599295bb572e2f0dfd72824f69b375bc8a7596ff`, physical copies, driver limits and hashes. The native ingest observed at 2026-10-11T21:09:04Z; its base watch closed after three passes at the original deadline. Earlier snapshots remain dated evidence.

G-S3GP denominator export `throughput-denominator-20261011.json:1` reconciles with the copied terminal `scratch-gp-xb3/receipt.json:1` at provider commit `9a030299620df3ca1ebc4b510ee086d0ef094ad7`. These are closed-run rates, not UTC-day allocations or qualified production rates.

| Measurement | Value | Unit |
|---|---:|---|
| programs_scored | 2815 | program evaluations |
| active_cpu_core_hours | 2.95403807 | active CPU core-hours |
| allocated_core_hours | 53.9292249913 | allocated core-hours |
| wall_seconds | 6067.03781152 | seconds |
| programs_per_active_cpu_hour | 952.932878079 | programs/active CPU-hour |
| programs_per_allocated_core_hour | 52.1980429063 | programs/allocated core-hour |
| programs_per_wall_hour | 1670.337373 | programs/wall-hour |
| compiles | 2794 | compiles |
| cached_disclosures | 21 | cached disclosures |

Active CPU is producer RUSAGE_SELF plus children. Allocated capacity is 32 declared cores over the measured 6,067.0378115 seconds. The review’s rounded 100-minute illustration gives 52.78125; the actual allocated rate is 52.1980429063. Neither rate fills qualified context-program outcomes/hour or qualified non-catalog programs/hour. Consumer arithmetic: `operational.py:89` at the pinned code commit.

The explicit native registry joins all **263 validation contexts to 54 parents**, with **zero missing or ambiguous IDs**. It maps 7,120 contexts using `retained_lineage_context_sha256 -> parent_lineage_sha256`, never names. Sources: copied spec `05dc9558…:1`, parent registry `8fe40984…:1`, custody `gap-close-20261011/parent-map-proof.json:1` and `parent-map.json:1`. Parent means average cuts within each parent, then weight parents equally. SE assumes independent parents; family independence and new-family generalization remain unqualified. Epochs, updates, rosters and physical exports are separate.

| Latest complete comparison | Epoch / updates | Contexts / parents | Parent mean ΔL6 | Parent mean SE |
|---|---:|---:|---:|---:|
| CL/claude-rlfw-lpi-scratch-sweep6-20261011T024759Z/set-a/vs-control/r23 | 2 / 704 | 263 / 54 | 0.000918893657707 | 0.000599369344405 |
| CL/claude-rlfw-lpi-scratch-sweep6-20261011T024759Z/set-a/vs-control/r23_lpi01 | 2 / 704 | 263 / 54 | 0.00013911954201 | 0.000568851209788 |
| CL/claude-rlfw-lpi-scratch-sweep6-20261011T024759Z/set-a/vs-r23/control | 2 / 704 | 263 / 54 | -0.000918893657707 | 0.000599369344405 |
| CL/claude-rlfw-lpi-scratch-sweep6-20261011T024759Z/set-a/vs-r23/r23_lpi01 | 7 / 2464 | 263 / 54 | -0.000486967967188 | 0.000518357199798 |
| CL/claude-rlfw-lpi-scratch-sweep6-20261011T024759Z/set-b/vs-control/r23 | 2 / 704 | 1000 / 455 | 0.000272979732953 | 0.000126905564294 |
| CL/claude-rlfw-lpi-scratch-sweep6-20261011T024759Z/set-b/vs-control/r23_lpi01 | 2 / 704 | 1000 / 455 | 0.000118721324967 | 0.00015526463352 |
| CL/claude-rlfw-lpi-scratch-sweep6-20261011T024759Z/set-b/vs-r23/control | 2 / 704 | 1000 / 455 | -0.000272979732953 | 0.000126905564294 |
| CL/claude-rlfw-lpi-scratch-sweep6-20261011T024759Z/set-b/vs-r23/r23_lpi01 | 7 / 2464 | 1000 / 455 | -0.000253267427568 | 7.14081172439e-05 |
| CL/claude-rlfw-lpi-scratch-sweep6-20261011T024759Z/set-heldout/vs-control/r23 | 2 / 704 | 885 / 167 | 0.00149995293316 | 0.000201631179044 |
| CL/claude-rlfw-lpi-scratch-sweep6-20261011T024759Z/set-heldout/vs-control/r23_lpi01 | 2 / 704 | 885 / 167 | 0.00160191182405 | 0.000258525920748 |
| CL/claude-rlfw-lpi-scratch-sweep6-20261011T024759Z/set-heldout/vs-r23/control | 2 / 704 | 885 / 167 | -0.00149995293316 | 0.000201631179044 |
| CL/claude-rlfw-lpi-scratch-sweep6-20261011T024759Z/set-heldout/vs-r23/r23_lpi01 | 7 / 2464 | 885 / 167 | -0.000146681562894 | 0.000123385497975 |
| CX-OP/cxop-r7-r23-20261011T141653Z/pkpo_b8 | 8 / 1136 | 263 / 54 | -3.3816431061e-08 | 3.06673738747e-08 |
| CX-OP/cxop-r7-r23-20261011T141653Z/rloo | 8 / 1136 | 263 / 54 | -0.000362717292385 | 0.000299389359272 |
| CX-OP/cxop-r8-r23-20261011T161734Z/pkpo_b8 | 8 / 1080 | 263 / 54 | -1.18150264806e-09 | 1.1188702131e-08 |
| CX-OP/cxop-r8-r23-20261011T161734Z/rloo | 8 / 1080 | 263 / 54 | 3.4394339541e-05 | 4.26648617958e-05 |
| CX-OP/cxop-r9-scratch-r23-20261011T184027Z/pkpo_b8 | 8 / 1089 | 263 / 54 | 4.42766489174e-09 | 1.10355230743e-08 |
| CX-OP/cxop-r9-scratch-r23-20261011T184027Z/rloo | 8 / 1089 | 263 / 54 | 9.45732051452e-05 | 4.60884349288e-05 |

These are descriptive NON-QUALIFYING L6 inputs. They do not replace retained q fold noise or family qualification. All per-parent inputs, variances and source file:line references are retained in Parquet. The native parent join/count proof and 4,208 identical local/remote paired aliases are retained in custody; aliases count once in each aggregate, while every raw record remains in the snapshot. Conflicting observations under one export hash are refused.

R10 archive audit `INITIAL_MEASURED_AUDIT.json:1` SHA `54bb54cab48f0191c7f8120e67d8fe983454d6f8a90da3491e98c92e2a5dfb26` reports gradient probes on four fixed contexts at **epoch 0 / updates 0**. Its separate completed-training count of 35 never relabels the probes. “Applied” is the coefficient-scaled term gradient, not a consumed optimizer gradient. Diagnostic source SHA `d6d884faa9bbd1ad28daff8e4523198a254952d77cc2e5569dcde6454c9171ac`; trainer SHA `f314a2b0d24e9a070cf4655d2b9b936b738d8d34c0ddae592a729982a85064f4`.

| Probe scope / term | Applied actor L2 norm |
|---|---:|
| CX-OP/R10-archive/r23_control/probe/epoch0/updates0 / actor_probe_grad_norm:L_archive_PI:applied | 0 |
| CX-OP/R10-archive/archive_lpi/probe/epoch0/updates0 / actor_probe_grad_norm:L_archive_PI:applied | 0.186786544761 |
| CX-OP/R10-archive/rloo/probe/epoch0/updates0 / actor_probe_grad_norm:L_RLOO:applied | 0.0282498368735 |

expected_distinct_D_B:full_trie at B=8 ranges **7.98546770822–7.98942167351** across the producer’s four fixed probe contexts. These are reported expectations, not observed distinct draws; full trie and measured-leaf conditional pi_M are separate policies.

expected_distinct_D_B:pi_M at B=8 ranges **4.29851702305–5.91950775082** across the producer’s four fixed probe contexts. These are reported expectations, not observed distinct draws; full trie and measured-leaf conditional pi_M are separate policies.

Clipped-utility scratch HWM support has **47 known-positive / 2,267 total contexts**: observed lower bound **0.0207322452581**. Among 2,266 explicitly comparable contexts it is **47/2,266**; one NULL incumbent stays unknown. Raw-q positives (52), q-grid positives (44) and clipped-utility positives (47) are different counters. This does not qualify a reward/evaluator closure.

Qualified outcomes/hour, qualified non-catalog programs/hour, feedback-to-first-update latency, actual contexts receiving actor targets, complete HWM support, actual update gradient norms, duplicate-draw rate, archive exposure by parent and verified-improvement probability movement remain **unmeasured**. First-trained must be an actual optimizer update using the exact feedback identity; adoption, fixture probes and evaluation cannot fill it. The seven explicit clocks are proposed, compute-started, locally verified, remotely qualified, adopted, first-trained and first-evaluated (`operational.py:197`; contracts in [CONTEXT_EXPORT.md](CONTEXT_EXPORT.md)).

First local required gate PASS belongs to `e941893140496210100df1c28b16421a83068608` only. The final candidate repeats local required, exact-head isolated analytics, hosted CI and independent review; predecessor approvals are not transferred. No GPU, learner/scoring, bootstrap, S3 write or promotion was performed by CX-MET.
