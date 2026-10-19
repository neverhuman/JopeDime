# KPI catalog and instrumentation backlog

This is the product brainstorm and measurement contract, not a claim that every KPI exists.
**Live** means implemented with a source when configured; **snapshot** means dated evidence;
**next** requires additional producer instrumentation. Unknown means unknown, never zero.
Every KPI needs unit, scope, timestamp, source/config/data identities and a useful denominator.
No blended “percent complete” substitutes for engineering/scientific acceptance gates.

## Training, optimization and model state

| KPI | Definition / denominator | Source / state |
| --- | --- | --- |
| Optimizer steps | Actual completed updates per run, not loop/process ticks | Config-bound progress; live |
| Completed epochs | Producer-reported completed corpus passes | Planner/progress; live |
| Budget progress | Steps / min(max_updates, epoch_batches × max_epochs) | Config + loaded train membership; live |
| Unique train exposure | Distinct training context IDs actually presented | Progress; live |
| Total presentations | All context presentations, including repeats | Progress; live |
| Train coverage | Unique presented / admitted train contexts | Loaded receipt + progress; live |
| Validation coverage | Unique scored / eligible validation contexts | Loaded receipt + progress; live |
| Fixed-panel train loss | Measured objective with dropout=false | Panel/objective-bound curve; live |
| Fixed-panel dev loss | Same panel and objective across steps | Separate series; live |
| Full validation loss | All required validation members, with support counts | Full-validation receipt; live |
| Generalization gap | Comparable dev loss minus train loss | Same objective/panels only; next derived |
| Loss by component | CE, BT, quality, feasibility, listwise, runtime terms | Curve term records; live API; next dedicated chart |
| Sparse support | Numerator/denominator per loss term/context | Objective receipts; next |
| Gradient norm / clipping | Pre/post clip norms and clipped update fraction | Optimizer hook; next |
| Learning rate / schedule | Effective rate and scheduler phase | Config/checkpoint; live base rate; next dynamic |
| Weight decay / optimizer | Actual optimizer family and hyperparameters | Checkpoint manifest; live |
| Parameter count / architecture | Layers, heads, width, context/vocab/sequence limits | Config/checkpoint; live |
| Parameter identity | Canonical parameter hash and changed-tensor proof | Manifest live; independent tensor proof next linked |
| Optimizer-state checkpoint | State payload hash, bytes and step binding | Manifest; live |
| Initialization / resume lineage | Fresh/resumed source, parent checkpoint and seed | Config live; parent graph next |
| Reproducibility | Multi-seed variance, deterministic replay deviations | Independent exact-protocol experiments; next |
| Failure / nonfinite updates | NaN, overflow, skipped/failed update counts | Failure receipts; next |
| Time to target loss | Wall time/updates to a predeclared same-panel threshold | Bound objective history; next |
| Best checkpoint | Best qualified full-validation metric under selection rule | Selection receipt; next; never guessed from panel loss |

## Scientific and algorithmic self-play outcomes

| KPI | Definition / denominator | Source / state |
| --- | --- | --- |
| Generated proposals | Durable model emissions, not rewards | Outbox progress; live |
| Novel executed programs | Qualified novel executions, distinct from emissions | Progress; live |
| Published results | Producer-reported verified result publication count | Progress; live; independent closure separate |
| Comparison games | Completed same-protocol games | Progress; live |
| Win rate vs random | Wins / qualified games with interval and seed count | Evaluator-bound comparison receipt; next |
| Win rate vs HWM / incumbent | Same task/dataset/cut/evaluator/vocab/protocol | Qualified comparison receipt; next |
| Realized improvement | Candidate score minus matched baseline, correct direction | Paired baseline receipt; next |
| HWM promotions | Authorized novel improvements / admitted comparisons | Promotion ledger; next |
| Elo / ranking uncertainty | Model, fit, confidence and opponent population | Valid comparison games; next; no games => unknown |
| Successful self-play cycles | Generate → execute → evaluate → learn closure | Cycle receipts; next |
| Novelty / uniqueness rate | Distinct canonical programs / emitted programs | Outbox/canonicalizer; next |
| Compile / type / feasibility success | Successes / attempted candidates, by reason | Execution receipts; next |
| Reward attribution coverage | Outcomes with exact causal proposal binding / outcomes | Native evaluator closure; next |
| Self-play data contribution | Newly qualified labels / optimizer input labels | Corpus provenance; next |
| Negative results | Failed/inconclusive/timeout experiments retained | Wave cards/metrics; snapshot/live files |
| Held-out performance | Untouched family/outer-fold metrics under sealed admission | Evaluator receipt; next; research dev split never substitutes |
| Statistical confidence | Paired effect interval, sample and independent seed counts | Predeclared protocol and analysis; next |
| Compute efficiency of gains | Qualified improvement / GPU-hour or execution budget | Bound experiment + resource receipts; next |

## Storage, datasets and training corpus

| KPI | Definition / denominator | Source / state |
| --- | --- | --- |
| S3 physical storage | Bytes per configured bucket incl physical copies | Cached daily CloudWatch; live daily |
| S3 objects | Daily NumberOfObjects across all storage types | Cached metadata; live daily |
| Estate coverage | Successful buckets / configured buckets, measurement age | Collector output; live |
| Current vs noncurrent bytes | Version class bytes; avoid combining with payload census | Detailed inventory; next |
| Unique logical payload bytes | Content-addressed deduplicated size | Hash inventory; next |
| Replication overhead | Physical copy bytes / unique payload bytes | Replica + digest census; next |
| Compression ratio | Decoded/member payload bytes / compressed bytes | Archive receipts; snapshot; next aggregate |
| Storage growth | Comparable daily estate deltas with coverage held fixed | Persisted daily metrics; next |
| Storage class distribution | Standard, IA, archive and overhead bytes | CloudWatch series; next breakdown |
| Multipart/version overhead | Incomplete upload and metadata/version storage | Inventory/lifecycle evidence; next |
| Dataset IDs | Unique catalog identities, not objects/contexts/rows | Versioned catalog; snapshot |
| Dataset origin / family coverage | Count by real/synthetic/source/family | Catalog; snapshot/next views |
| Dataset rows/features/task | Per-dataset shape and semantic schema | Dataset manifest; next |
| License completeness | Recorded license IDs / datasets | Catalog; snapshot |
| Corpus contexts | Selected train/dev contexts vs globally eligible views | Loaded receipt; live |
| Pending input joins | Eligible contexts awaiting valid encoding/label join | Loaded receipt; live |
| Declared vs exposed records | Inventory rows vs rows actually read/qualified | Separate counters; next |
| Label / encoding schema | Policy, vocabulary, encoder/checkpoint and dimensions | Config/corpus pins; live partial; next dedicated view |
| Split / leakage qualification | Parent-family overlap and protected exclusions | Role/admission receipts; next |
| Artifact body closure | Exact-version/member hashes verified / required | Preservation receipts; next |
| Corruption / missing artifacts | Digest or membership failures by stage | Restoration/corpus receipts; next |

## GP library and drill-down

| KPI | Definition / denominator | Source / state |
| --- | --- | --- |
| Unique GP programs | Distinct declared canonical program identities | Hash-pinned configured snapshot; live |
| Source scope / coverage | Snapshots included, date, hash, incomplete global scope | Source cards; live |
| Family distribution | Unique programs by recognized estimator family | Library DSL metadata; live |
| Operator composition | Operators present per program, searchable | DSL drill-down; live |
| Context occurrences | Source records referencing a program | Snapshot, not global execution count; live |
| Exact DSL / identity | Original bounded program string and declared ID | Library inspect dialog; live |
| Program complexity | Nodes, depth, token length, reusable subexpressions | Canonical IR receipts; next |
| Fit / prediction cost | Measured latency, memory, failures and sample size | Same-input execution receipts; next |
| Score / evaluator identity | Dataset/cut/metric/protocol-bound measurements | Not present in basic teacher library; next |
| Pareto frontier | Qualified quality vs cost/complexity tradeoff | Comparable execution receipts; next |
| Evolution lineage | Parent IDs, mutations, crossover, generation | Native proposal lineage; next |
| Library churn | New, superseded, duplicate and retired identities by wave | Dated library snapshots; next |
| Admission / serving state | Research, qualified, selected, deployed | Native admission/promotion receipts; next |
| Reuse and transfer | Qualified reuse across tasks/families and resulting lift | Matched transfer experiments; next |

## Fleet throughput, resource use and reliability

| KPI | Definition / denominator | Source / state |
| --- | --- | --- |
| Host freshness | Timestamp/age per xbabe1..4 | Optional node snapshots; live |
| CPU capacity / pressure | Cores, load; next busy%, throttling and PSI | Node snapshots; live partial |
| Memory / disk headroom | Available bytes, next pressure/OOM/error counters | Node snapshots; live |
| GPU utilization / memory | Host-wide device counters, exact device model | Node snapshots; live |
| GPU owner / resource grant | Actual run/device/cgroup/task attribution | Resource lease receipt; next |
| Step throughput | Optimizer updates / elapsed wall time | Run performance; live |
| Input / decode / optimizer time | Separate accumulated phase seconds | Run performance; live API |
| Training utilization efficiency | Compute vs input/decode/wait wall fractions | Nonoverlapping phase timings; next |
| Checkpoint overhead | Write duration, throughput and stall fraction | Writer timings; next |
| Power / energy | W and integrated Wh per qualified experiment | Device telemetry + exact interval; next |
| Budget / time remaining | Fixed grant deadline and remaining useful budget | Original allocation/window receipt; next |
| Queue depth / wait age | Ready, blocked, admitted and active work | Owned scheduler feed; next |
| Worker progress | Deliverables and counter deltas, not process presence | Acceptance receipts; next |
| Failure / retry rate | Original failure classes / attempts, finite retry budgets | Run receipts; next |
| Run stagnation | No qualified progress over workload-specific interval | Counter/history, not HTTP health; next |
| Provider exhaustion | Exact model/provider blocker and retry disposition | Owner receipts; next; no settings change |

## Versioning, Git, preservation and engineering health

| KPI | Definition / denominator | Source / state |
| --- | --- | --- |
| Training code commit | Actual copied runtime closure vs pinned Git commit | External source proof; live |
| Dashboard code/build | Actual full SHA, dirty state, binary/UI manifest | Operator code identity; live partial |
| Source/config/seed/input IDs | Reproducible recipe identity | Config/checkpoint/corpus; live |
| Checkpoint lineage and size | Step, model/parameter/config hashes and bytes | Latest manifests; live |
| Git waves / measurements | Wave count and recorded assertion count | Compact wave files; live |
| Git publication | Commit/PR/main reachability, not local marker alone | GitHub + receiving-main receipt; next automatic |
| Archive publication | Exact URI, VersionId, compressed/decoded/member hashes | Wave manifest; snapshot; next dedicated state graph |
| Independent restoration | Exact restore proof vs hash-only archival claim | Restoration receipt; next |
| Information density | Analyzable assertions/context per compressed byte | Wave index + archive sizes; next |
| Preservation lag | Completed run → Git card/archive/restore delay | Terminal/publication receipt timestamps; next |
| Open PR queue / staleness | Current queue, conflict, owner, ready age | GitHub feed; next; captain retains queue ownership |
| Exact-head CI/review | Applicable checks + independent native approval | GitHub receipts; next |
| Receiving-main validation | Actual merged main build/test/runtime identities | Release receipt; next |
| Release gap | Engineering gates vs scientific/custody gates separately | Current47-task tracker; next; old66 distinct |
| Dirty / unpushed work | Owned changes and unique commits; no destructive auto-clean | Read-only Git inventory; next |
| Cleanup eligibility | Preserved, independently restored, no consumers/symlinks | Owner custody ledger; next; dashboard never deletes |

Priority after this first usable release: independent results/archive state graph, full-validation
and loss-term charts, artifact/dataset browsing, per-program measured fitness/lineage, exact grant
and device ownership, then statistically qualified self-play gains. Storage cost estimates require
an explicit pricing date and billing inputs; do not invent cost from bytes or spend new capacity.
