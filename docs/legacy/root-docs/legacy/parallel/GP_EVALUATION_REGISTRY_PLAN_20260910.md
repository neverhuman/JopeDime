# Prevent duplicate GP fits across workers, searches and generations

Status: detailed design for review, September 10, 2026. This extends the [flywheel audit](FLYWHEEL_DEEP_AUDIT_AND_APPROVAL_PLAN_20260910.md). It does not claim that a global registry is deployed. The production GP executable remains `974e9383…`, and automatic neural training remains held after failed diagnostics. The user's new constraint supersedes any earlier proposal to benchmark cache-on/cache-off by fitting the same evaluation again.

The required invariant is **one fit dispatch at most for each program and exact evaluation contract across the entire campaign**, including concurrent searches and restarts. Reusing a program's verified fitness is useful; fitting it again to rediscover that fitness is waste. A required evaluation on a different training/validation fold is distinct work and must be visible in the dataset/program fold inventory. Search RNG, generation numbers, output directories and new ancestors are not reasons to refit.

## 1. What the production source actually prevents

The captured native-v5 GP source is SHA `29d3452f…`; the independent reviewer checked it against both frozen native-v5 archives. [Exact inspected source](../runs/flywheel-implementation-20260910/deep-audit-20260910/node-source/gp/gp.rs).

| Mechanism | Current scope | Remaining hole |
|---|---|---|
| Initial population `seen` set | One search; raw program strings | Not shared with another job or prior generation |
| `evaluated_programs` | Canonical programs **after** vocabulary projection; restored from that search's ledger | Correctly rejects projection aliases within a search, then resets for a new output/search |
| Prediction `semantic_sha256` | Deduplicates ranked predictions **after fitting** | Cannot save the cost of that fit; equal predictions on one cut do not prove program equivalence elsewhere |
| Retained teacher bank | Supplies strong structures before neural seeds | Supplied teachers are evaluated again; no global measurement lookup precedes dispatch |
| Saved candidate intent | Restores candidate identity within one interrupted search | An unfinished intent without a result can be dispatched again; execution may have happened before the crash |
| Search RNG | Advances mutation/crossover choices | Separate from the emitted `evaluation_seed=0`; must not fragment evaluation reuse |

An older Python [pair registry](../../scripts/test_exact_pair_registry.py) already uses a transactional unique key and an attempt/event history. It is not called by the native GP path. Its protocol key and infrastructure-retry rules do not satisfy the new contract. Reuse the useful transaction/state-machine ideas, not its identity omissions or retry permission. Its misleading `test_` filename also means it must not be confused with a harmless unit test during broad test discovery.

The live controller additionally retains an unqualified neural-only seed enrichment step, and its manifest does not enable `generation_seeded_v1`. Proposal diversity and evaluation reuse are separate problems: advance the qualified proposal RNG to discover new structures, and consult the registry regardless of how a structure was proposed. The current v44 proposal executable and generation policy must be versioned explicitly before changing them.

## 2. Separate program, evaluation and search identities

Use three records rather than a single overloaded hash.

**Program identity** binds a domain-separated SHA-256 of the exact canonical typed program after vocabulary projection and macro expansion, its grammar/IR meaning version and parameter values. Store the canonical bytes as well as the digest and check them on collision. Canonicalization must preserve operator order and parameter attachment. Do not sort tokens, commute floating-point operators, or merge programs merely because they produced equal predictions. A new equivalence rewrite needs a separately tested, versioned semantic rule.

**PhysicalFoldKey** binds one program's actual fit and evaluation on one exact fold. It includes everything that can change fitted predictions, reported quality or the meaning of that physical outcome:

| Field family | Required content |
|---|---|
| Data | Source content and schema, selected feature/target view, exact ordered training and evaluation memberships, sealed numeric input digest |
| Preparation | Training-fitted preprocessing and learned-transform/reference-bank identities; missing-value/category treatment; target transform, scale and metric units |
| Program | Canonical program identity, interpreter/compiler meaning and executable identity |
| Numerics | Evaluator protocol, dependency/build identity, relevant hardware/numerical backend, precision/determinism policy and thread settings |
| Randomness | Actual fit RNG seeds and any stochastic program parameters; distinguish these from search/proposal RNG |
| Resources | Total candidate deadline, memory/thread limits, process protocol and failure/censoring semantics |
| Physical outcome | Metric definition, target units and one physical fold's outcome protocol; no CV weights or aggregate-only settings |

Human-readable dataset names, cut indices, filesystem paths, whole-generation manifests and search ancestry belong in provenance. Where exact sealed data and fold/preprocessing contracts already determine computation, incidental path or generation changes must not force a new evaluation key. Conservative compatibility must be proved before collapsing older identities; do not discard a field merely to increase hit rate.

**AggregateKey** binds an ordered set of PhysicalFoldKey/result references, CV weights, aggregation recipe and partial-failure policy. It authorizes a derived calculation only. Changing those weights or recomputing a summary cannot authorize any new fold fit. The current synthetic prototype rejects aggregate requests, keeping them outside its physical dispatch API.

**Search provenance** records the requesting search, generation, comparison arm, original proposal slot, parent/mutation/crossover, checkpoint, proposal RNG, logical candidate index and allowed data scope. Many search requests may reference one evaluation. A reuse receipt points to the original clean-completion receipt and original execution identity; it never rewrites the original attempt or claims a fresh measurement under a new search index.

A program changing because its numeric seed parameter changed is a different program. Deliberately incrementing an otherwise irrelevant fit seed or protocol version just to bypass a tombstone is prohibited. The current rule permits no intentional repeat-measurement study. Any exception requires a new explicit amendment from the user; an internal experiment manifest, agent decision or recovery policy cannot grant it.

A later implementation should separate the numerical evaluator executable from GP orchestration and reporting. Keep the numerical build pinned while changing queue or receipt code, and bind the launcher/completion build independently in provenance. This reduces unnecessary compatibility breaks without dropping evaluator identity from the physical key. It requires a versioned adapter and lifecycle qualification: the current v4 admission proposal still checks the combined build contract. A controller rebuild must not create a loophole for repeating an already attempted numerical cell; unresolved cross-build history remains blocked. Small parity samples alone cannot prove universal compatibility.

## 3. Registry states and the dispatch boundary

Start with one authoritative SQLite registry on the node's local reliable filesystem, using WAL, full durability, transactions, unique binary keys and immutable external result blobs. The existing native process already links SQLite. This is sufficient to test the actual 104-worker load before adding a distributed database or a separate network service.

| State | Meaning | Request action |
|---|---|---|
| Absent | No known claim or historical attempt | May atomically claim, subject to scope and budget |
| Claimed, not dispatch-authorized | Durable owner/contract reservation exists | Other workers wait briefly or propose another program |
| Dispatch-authorized / running | Durable one-use dispatch identity exists | Never launch another fit; reconcile the owner/process |
| Completed | Clean, verified result committed | Reuse the original result if caller scope permits |
| Terminal failed / timed out | Observed outcome under this contract | Reuse outcome; never invent an RMSE or automatically retry |
| Unknown after interruption | Dispatch may have occurred; evidence incomplete | Quarantine and reconcile; do not dispatch again |
| Historical seen, not reusable | Prior attempt lacks enough evidence for trustworthy reuse | Preserve the exclusion and evidence gap; do not call it a fresh miss |
| Conflict / corrupt evidence | Identity or result disagreement | Block reuse and new dispatch; emit an incident |

The fast path is: canonicalize → compute the prepared evaluation key → lookup → either reuse/wait/skip or atomically claim. A read followed by an unguarded insert is insufficient. `INSERT … ON CONFLICT`/an equivalent immediate transaction must elect one owner, and the evaluator launch path itself must require the committed one-use authorization. Every path—teacher seed, neural seed, random candidate, mutation, crossover, resumed intent and direct-proposal evaluation—uses this boundary.

Use a per-evaluation OS lock or a sole authoritative executor to prevent two processes carrying the same owner token from launching concurrently. The start token must also be consumed exactly once in durable state: an OS lock alone cannot prevent sequential reuse after its first owner releases it. Bind process identity to node boot, PID birth and a task-specific process group. Keep a monotonically fenced dispatch record. Publish authorization durably **before** spawning; retain custody through completion. A crash between authorization and spawn may leave an unknown that never actually ran. Under an absolute no-repeat rule, accepting that inconclusive slot is safer than guessing and launching it again.

Expiry of a heartbeat or lease never proves that evaluation did not run. A replacement controller first reconciles original receipts, surviving processes and committed results. It can continue supervising the original process or commit a valid original completion. It cannot delete the claim and start over. The narrow current worker-readiness fix does not by itself provide this GP dispatch guarantee.

This is an at-most-once dispatch contract, not a promise of exactly one successful measurement despite arbitrary crashes or data loss. Strict no-repeat and automatic retry of uncertain work cannot both be guaranteed. Make `unknown_after_dispatch` visible and retain it indefinitely unless new original evidence resolves it. The historical importer must additionally maintain conservative partial-identity tombstones, indexed by the strongest verified program/input identity available. Missing runtime or executable fields cannot make an overlapping historical attempt appear absent under a newly complete key. Resolve or explicitly block that uncertainty before first dispatch. Known past duplicate executions remain recorded; activation establishes a prospective boundary and cannot make the earlier campaign at-most-once retroactively.

## 4. Fast lookups without weakening correctness

Use fixed-size evaluation keys in a `WITHOUT ROWID` primary table, a separate program table, small status/result-reference rows and indexed dataset/program/fold membership. Large metrics, predictions and trained artifacts live in immutable content-addressed files. A lookup need not parse every historic ledger or load prediction vectors.

Maintain small per-process caches only for immutable terminal entries whose evidence identity was already verified. A local miss does not authorize execution; it falls through to the authoritative transaction. A stale negative cache is especially dangerous under concurrent requests. Bloom filters, if later useful, are hints only: a positive cannot exclude a novel program without an exact lookup, and a negative cannot bypass the atomic claim.

Reuse one prepared evaluation-contract object per immutable cut/feature view. Hash exact sealed input once under its verified identity, not once per candidate. Cache exact tokenization/canonical IR and grammar transitions by program meaning identity. This complements the existing binary teaching shards; an evaluation registry is not another copy of the entire packed corpus.

Do not hold a database write transaction while fitting, reading a large result file, building a corpus or waiting on another worker. Keep claim/commit transactions short. A bounded notification or local polling path wakes waiters; the rest of a worker's proposal queue can continue while another worker owns that individual. If measured fsync contention becomes material, consider a durable group-commit executor that acknowledges claims only after their shared fsync. Measure the simpler design first.

Proposed performance criteria, to be measured under representative corpus I/O and 104 concurrent clients: terminal-hit p99 at most 2 ms, uncontended durable claim p99 at most 10 ms, throughput at least twice measured peak candidate-request rate, and no more than one CPU core of aggregate registry overhead at observed production load. These are engineering targets, not achieved benchmarks. Report hits, misses, conflicts, waits, fsync latency, busy time and false filter hits separately. A p99 miss above budget triggers optimization; it never enables an unsafe direct-fit fallback.

## 5. Preserve folds, failures and scientific comparisons

A dataset/program registry should expose its required-fold bitmap and outcome references. A five-fold CV request fills only missing authorized cells. Existing complete cells are reused; running cells are shared; unknown cells make the aggregate incomplete. Never replace missing folds with the current winner's score or silently aggregate only successful folds. Fold-local preprocessing and Hyperion reference banks remain mandatory. A CV aggregate is a derivation over named fold evaluations, not permission for five uncontrolled repeats of one fit.

Keep dataset/family eligibility outside the physical cache key as an enforced access policy. A score existing on disk does not grant a training task access to validation or sealed-test evidence. An ineligible result is an access denial, not a cache miss followed by a fresh fit. Registry enumeration must not turn validation winners into an implicit leader bank. The ten sealed test families stay unopened.

Under the user's no-repeat rule, do **not** run a real-data cache-off arm that intentionally recomputes the cache-on arm's exact fits. Instead, qualify correctness with synthetic evaluators and first-execution-versus-reuse receipts. For real search comparison, predeclare a shared measurement oracle with the same eligibility and result availability for all arms. Each arm can request only programs it actually proposes; a hit consumes its logical candidate slot. Report logical requests, unique executed fits, reused fits, waits, actual CPU/wall time and original-measurement/virtual charges independently.

A mutable shared registry can otherwise give later arms free results unavailable to earlier arms. Freeze measurement access/accounting at the experiment level, and define how simultaneously requested shared fits are charged. Do not call this a causal cache-speedup experiment merely because fewer processes run. Replay original execution traces or model avoided cost using recorded original runtimes, clearly labeled as estimates, rather than refitting to manufacture a counterfactual.

Reuse is not discovery. The teaching admission path stores an observation once by evaluation identity and keeps requesting searches as references. Reused programs do not increment newly admitted, improved or actual-fresh-exposure counters. Teacher reporting retains the original discoverer's ancestry, the later reuse event and ties; cache delivery does not turn a teacher into a neural discovery.

## 6. Historical import and rollout

1. **Freeze the census method.** Inventory exact source/evaluator versions, complete searches and pending attempts. Count within-search and cross-search repetitions separately, at both dataset/program and full known evaluation identity. Separate fold changes, fit-seed changes and conflicts. Report records/attempts rather than claiming all finite metrics are clean results.
2. **Build an immutable history index.** Reuse a score only when complete execution, data, unit and scope identities can be validated. Otherwise import a seen/unreusable record or explicit ambiguity. Missing evidence must not become a trustworthy cache hit, and a conflicting repeated score must not be resolved by “first row wins.” Preserve originals and record import-source hashes and coverage.
3. **Finish clean-completion/admission integration.** The source proposals add receipt validation, outcome separation and journal-to-champion recovery. Current historic success rows do not automatically satisfy that contract. Cache reuse must reference the validated original receipt rather than forge a new completion.
4. **Shadow lookup only.** Observe existing committed requests and count would-reuse/wait/block decisions without dispatching extra fits or changing queues. This checks identity coverage and estimates avoidable work. It is not proof the new dispatch fence works.
5. **Prove the boundary with synthetic workers.** Run concurrency/crash/rollback tests below; type-check/build the exact native integration. Then use a bounded private target fixture with synthetic evaluation behavior, not sealed or repeated real data.
6. **Handoff at a durable GP boundary.** Preserve the existing queue, results, serving checkpoint, leases and pending attempts. Stop old-format new admission, drain/reconcile already active evaluations, reconcile the history high-water mark, then start only the versioned registry-aware executable. Old and new independent dispatchers must never race over the same key.
7. **Observe live exclusions/reuse.** Require zero duplicate authorized dispatches; sample original receipt chains, scope enforcement, performance and GP progress. An exhausted novel proposal stream reports exhaustion and triggers a separately approved coverage/diversity change, not a fit retry.

An old database backup must not erase knowledge of later dispatches. Bind backups and registry rebuilds to durable dispatch-journal high-water marks and complete receipt inventories. Restore in read-only/reconciliation mode until every later claim is accounted for. Missing journal evidence blocks new dispatch. Compaction may discard redundant payload copies only after preserving all evaluation tombstones and original provenance. Disk-full, corruption, lost locks and registry unavailability fail closed for new fits; known valid completed entries may remain readable.

## 7. Required verification and completion criteria

| Test | Required observation |
|---|---|
| 104 simultaneous requests for one exact key | Exactly one synthetic evaluator start; all others reuse/wait/skip |
| Same program across different searches/generations/RNG/paths | Same evaluation key and no additional fit |
| Canonical aliases and projection/macro collapse | Collision detected before evaluator launch |
| Order/parameter/feature/cut/unit/evaluator/fit-RNG changes | Correctly distinct computation or explicit incompatible-policy rejection; no accidental reuse |
| Crash before claim, after claim, around spawn and before PID receipt | No duplicate launch; genuinely uncertain dispatch stays unknown |
| Crash after result, before registry commit, before champion update | Original valid receipt reconciles without fitting again |
| Expired lease with surviving worker; PID reuse; controller restart | Original custody or quarantine; no lease-expiry duplicate |
| Success, fit error, nonfinite result, timeout, cancellation | Separate outcomes; failures never become fabricated RMSE labels |
| Tampered result/program/input/units and conflicting old rows | Reuse rejected and incident retained |
| Old database backup plus newer dispatch journal | Reconciliation restores exclusions before new work |
| Registry unavailable/full/corrupt; lock contention | No unregistered evaluator fallback |
| CV partly cached, partly running, partly unknown | Each cell at most once; missing evidence blocks complete aggregate |
| Cache hit across comparison arms | Same policy and explicit logical/physical budget accounting |
| Reused winner and repeated label publication | No fake fresh-discovery, exposure or ancestor credit |

The evaluator itself should emit a process-start record under the dispatch identity. Reconcile those starts against registry authorizations and completion receipts, so the invariant is measured at the fit boundary rather than inferred from a dashboard counter. Any second start for one identity is an incident that blocks new admission. A durable state-machine test, successful import and a sustained live window with zero violations are required before claiming the duplication problem solved.

## 8. CI and local rendered testing

Run ordinary unit/integration checks against exact source identities, preserving failures and fixing their causes rather than weakening selection. Tuiwright and Playwright checks requested by the user are **xbabe2-only local checks**, excluded from ordinary CI. The test entry point must refuse a different hostname and a CI environment before launching an authenticated tool, and must not copy credentials or modify provider enrollment. Rendered tests should verify registry/reuse/wait/unknown/failure displays, keyboard navigation, resize and recovery status against a private fixture; any authenticated end-to-end scenario needs the existing development-node runner and its own bounded receipt.

The active implementation and CI scope here is jope-dime. Bullet's separate agents own their four-repository CI work, and coordination is recorded in their requested chat file without changing those sources. Actual Jope dashboard Tuiwright checks run locally on xbabe2; no Jope browser application or Playwright target has been identified. No claim that “all CI passes” is made until the relevant inventory and exact results, including local rendered checks, are complete.

Approve the identity/state/never-retry-unknown contract and the staged implementation/verification sequence. No new GP fitting, data-family expansion, live cache swap or altered scientific comparison is authorized merely by this design document.
