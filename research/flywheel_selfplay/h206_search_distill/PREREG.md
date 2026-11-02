# H206: Rust-CV best-of-k search distillation

Authority: **Jepson via Grok Bot, card answer 8:22 PM MT**.
Owner: gpu_fused_redteam_codex; backup rd_astra.
Status: **QUEUED / NOT RUN / FACTORY AND PHASED STUDY SOURCE IMPLEMENTED**. Target: xbabe3 RTX 3090.
The initial prereg registered CPU preparation; final sections bind later source. It authorizes
no launch now. A later owner queue-release nudge, reviewed executor and immutable
input/model bindings are required. No GPU claim has been made.

## Hypothesis and fixed comparisons (EZ-G02)

Three expert-iteration rounds distill Rust-CV search into a k=1 proposer.
Teacher budgets are k={16,64,256}; add a compute-matched K1_SELF_DISTILL control.
A frozen native r882 is the reference, with no update and no search.
Keep four selection slots to reuse the exact H177 pooled selector calibrated by
H178: CTRL=K1_SELF_DISTILL, ACC4=K16, ACC16=K64, OC15=K256. These are an explicit
H206 name mapping, not the original H177 interventions. H178's empirical power
estimate does not prove power for these new interventions.

Fix one width128 widened adapter architecture and initialization mapping from r882,
using H204's architecture if its source is available. Pin the implementation and
initial state before execution; there is no outcome-selected checkpoint/model
fallback. Do not import an H204 model trained on the held-out families. A compatible
fold-specific checkpoint could be used only by a new pre-outcome amendment.
The current architecture/initialization and executor bindings are pending, so
this skeleton refuses execution rather than substituting a different model.

On separate research copies, train DOPE and the widened adapter via the dedicated
research opt-in/run ID API. Keep the native anchor, global normalizer buffers and
DOPE eval mode fixed; save their hashes. Serving/flywheel freeze policy remains
strict. Persist the actual dope_policy in every run launch.json before updates.

## Inputs, closure and scored-forward order

Use the SHA/version-pinned real S3 TRAIN closure: 1,219 contexts / 205 families.
Do not silently expand the training population. Any future FRESH/D4 TRAIN
expansion requires the merged JRB-01 real-catalog fixes and a prereg amendment.
Neither sealed D4 confirmation families nor their aliases can enter training.
Pin all context IDs, family IDs, S3 VersionIds, content hashes, r882 assets,
grammar/evaluator versions, program sources and the three family partitions.

Before ANY candidate forward, reproduce native r882 91/91 reference utilities
and exact context mean **1.1377941898164525**. Score one context per forward
everywhere, including cached-anchor parity and candidate evaluation. Cross-batch
agreement or unchanged argmins cannot replace full native-cost exact identity.
A mismatch stops INCOMPLETE and saves the failing vectors/identities if custody
is available. Native DL-Val access is only for that mandated gate and the final
descriptive read; no DL-Val-driven tuning, training or checkpoint choice.

Outer seeds: 206101,206102,206103. Fold seeds: 206141,206142,206143.
For each seed, permute sorted family IDs with NumPy default_rng(fold_seed),
split into five groups of 41. In cell j, A=j, B=(j+1)%5, FIT=the other three.
Prove FIT/B/A family and context disjointness and exact-once pooled B/A coverage.
Save the full fold manifest and pairwise Jaccard matrices before any inference.
Inherited r882 training is common to all arms; new H206 labels/updates are FIT-only.

## Search and distillation, three rounds

For each FIT context and arm/seed/cell/round, generate a deterministic ordered
slate of 256 distinct, grammar-valid Rust programs, bounded to 4,096 proposal
attempts. Canonical program-source SHA identifies duplicates. Rank the valid
slate by the current proposer, break ties by canonical source SHA, and retain
that order. Candidate-generation source, sampling distribution and evaluator
bindings must be reviewed and pinned before release; they are not implemented
by this skeleton. Failure to obtain 256 valid programs is INCOMPLETE.

Rust inner CV uses that context's permitted training rows/cuts only. The teacher
is the first minimum exact Rust CV cost among the ordered first k programs
(k=1 for the control). The native outer evaluation target is never a teacher
label. Record invalid-program/CV failures; never silently replace them with
oracle utility. All-failed teachers stop INCOMPLETE. Each arm shares the same
initial model and random-stream definition, but after round1 its proposal
distribution may differ: do not claim shared candidate prefixes across models.

Train against the teacher index with cross entropy over the full 256 predicted
negative-cost logits, temperature1. Thus the k=1 self-distillation control has
the same nontrivial loss/slate shape as the other arms. Optimizer: AdamW,
lr=1e-4, weight_decay=1e-4, effective batch_size32 (mean gradients from32
singleton forwards), exactly600 updates per cell/round.
Carry each cell's own weights/optimizer into its next round; no cross-cell or
cross-seed state sharing. Terminal candidate is round3/update1800 for EVERY arm.
No mean-only checkpoint retention, outcome stopping, B-selected round, best
seed, best snapshot or fallback to an earlier successful round.
Stop the whole attempt INCOMPLETE if a required fit or guard cannot finish.

Report the k=1 student's quality and inference cost separately from Rust search
quality/cost for k=1,16,64,256 at each fixed round. On B/A, any search comparator
uses only the context's revealed/training rows for inner CV, locks its chosen
program before reading evaluation utility, and never supplies training labels.
It cannot change the deployed student or selector. Keep search comparators
descriptive; the sole primary endpoint is the selected terminal k=1 student
against native r882. Count all comparator search work in the CPU budget.

## Selection, confirmation and decision

Produce all four terminal B panels with shape(3,1219), role-local at every cell.
Use the exact imported H177 pooled_b_select predicate, with a private module
instance and candidate-step metadata bound to1800 (no predicate changes):
average three seeds within each family, equal-family weighting, 10,000 paired
family-bootstrap draws, mean gain>0.001+1e-12, LCB95>0, joint four-arm Holm p<0.05.
Bootstrap seed206150. Additionally reject any candidate harming >10% of pooled
B families (seed-averaged candidate utility below seed-averaged r882 utility).
Among admitted arms choose largest pooled B mean gain; ties follow
K1_SELF_DISTILL,K16,K64,K256. No admission means native r882 selected.

Fsync a B-selection lock binding every terminal checkpoint, fold/input/source
hash, B panel, harm result and decision BEFORE A evaluation. Then read each
fixed arm's A panel, assigning each context exactly once per seed/arm to its
corresponding held-out cell model. All raw arms are retained; A cannot choose
an arm, checkpoint or round.
Report raw and selected panels, per-seed gains, family-mean pooled gain and CI,
median-seed companion diagnostics, four-arm Holm values and harm counts.
Do not present mean-B selection and median-A diagnostics as the same statistic.
A requires LCB95>0 and harm<=10% for a cross-fit ADVANCE; no A-driven reselection.

The rotated pooled B and A panels reuse families and related models. Report this
dependence; their agreement is not independent confirmation. Any final positive
claim additionally requires one sanctioned sealed D4 confirmation read, on
families never used for training or selection, through the authorized evaluator.
No sealed payload access is implemented or attempted here. Until that read is
completed under its own sealed access authority, the strongest result is
CROSS_FIT_CANDIDATE, never confirmed lift or serving promotion.

## Budgets and launch queue

Maximum400 CPU core-hours plus3x6 GPU-hours (18 total), including controls,
search comparators, native checks and evaluation. For each round reserve up to
5 GPU-hours for60 fits (4arms x3seeds x5cells; at most300s each) and1 GPU-hour
for gates/evaluation. Exactly600 updates must finish within each fit cap.
Meter process-tree user+system CPU time and GPU reservation elapsed time;
record actual work, failures and unused budget. Stage GPU proposal generation
before the long Rust CPU-labeling phase, persist its source/checkpoint-bound
slates, and release the GPU during CPU labeling. Reclaim only an idle GPU for
the next training/evaluation phase after fresh running-resource admission.
Count every GPU reservation in the6h/round budget; do not reserve the cache twice.
No retries, budget resets or
partial-arm efficacy claims. Cache replay cannot hide the cost of its creation.

nice19, host CPU<=85%, <=4 numeric threads (OMP/MKL/OpenBLAS/torch intra-op);
CUDA runtime/driver threads exempt. Use the shared PREREG_TEMPLATE and
hwave_numeric_threads guard, with real per-worker runtime reports and no nested
parallelism. At most four numeric workers of one thread each; GPU student work
uses one numerical worker. Enforce MemAvailable>=16GiB and RAM tree RSS<16GiB.

H206 remains queued even if a preparatory probe appears favorable. Launch needs:
this prereg committed and PR open; later explicit queue-release nudge; reviewed
full executor and all immutable pins; xbabe3 RTX3090 unclaimed with one GPU job;
disk used<=88% BEFORE reserve, >=50,000,000,000B headroom, cache cap40,000,000,000B,
and used<93.8% AFTER reserving that cache. Keep the running93.8% guard.
Stream exact-version objects from S3; do not clean foreign files to manufacture
admission. CLAIM GPU / RELEASE GPU under AGENT_CHAT flock on any future launch.
Never touch xbabe0 or xbabe1's live flywheel PID2297929.

## Evidence, acceptance and current limits

S3 root: s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261101/H206/
Use a new run-id prefix and immutable launch/source/input/fold bindings.
Stream large search/label/checkpoint evidence to versioned S3, GET exact versions,
verify SHA256 before reclaiming owned duplicates. Retain small panels and receipts.
Final PREREG.md, README.md(verdict), metrics.json and manifest.json must distinguish
not-run/partial/completed and bind S3 URI+VersionId+SHA256. Current JSON nulls mean
unmeasured, not zero performance or a passed native/selector gate.

This PR's acceptance is CPU-only data-free controls, Ruff, shell syntax,
just lint-paths and diff checks, plus exact-head central review. Remote CI waived
by n2022 policy; no author merge. The executor is intentionally unimplemented:
runtime gradient propagation, model/slate compatibility, native parity, numeric
resource admission and science have NOT been qualified by skeleton tests.


## Prospective source/initialization amendment — owner resume 2026-11-02

Authority: **Jepson via Grok Bot, card answer 8:22 PM MT**, continued by the
H206-OWNER-RESUME packet. This amendment precedes any H206 scientific outcome
or actual model/data run and awaits the assigned merger's review.

Fix the allowed widened-adapter option concretely: frozen r882 adapter_10tok
anchor plus a112,641-parameter residual MLP, Linear749->128/GELU/128/GELU/1,
no dropout, zero output layer, native unscaled cost residual. The749 inputs are
412 program features,16 projection coordinates,320 masked live token coordinates
and one frozen native-anchor cost. Use seed206101/102/103 for the corresponding
outer-seed initialization. Model-content hashes, original r882 checkpoint
ObjectRef and transitive assets remain required before real data use. Do not
substitute any H204 trained/checkpoint winner. ARCHITECTURE and its digest are
recorded in student.py and launch.json. The only trainable closure is DOPE plus
this residual. Original r882 tokens feed only the frozen anchor; fresh inherited
DOPE/router/column gates and Stage1B SVD feed the residual. There is no cached
embedding difference trick. Verify r882 native91 first; this source amendment
and its generated fixtures do not satisfy that gate.

Each optimizer update averages32 singleton-forward/backward examples, with
family components sampled uniformly then a context uniformly within component,
using the context's pinned seed. Save the batch-plan digest.600 updates per
round and3 rounds remain unchanged. Short fixture updates are logged as
non-endpoint source controls; B scoring is refused before round3/update1800.
Record intended parameter/gradient/forward deltas within EACH round; cumulative
prior progress cannot hide a later no-op. Resume model+optimizer+counts from
exact-version verified bytes and reuse the existing cost ledger.

Native bridge: require original TRAIN-only JTT1 with nq0, exactly two folds,
>=50 rows and>=25 per inner fold,1..512 columns, and<=16M cells after padding.
The pinned S1 parser requires nq>=10 but never decodes query cells. Supply10
query feature rows taken cyclically from original TRAIN X, with zero query
labels; never fetch or decode outer/test data. Preserve the original TRAIN
ObjectRef/bodySHA/cut and separately hash the actual native input. Freeze this
bridge's protocol/environment identity. The actual pinned binary succeeds on
an admitted-size generated fixture. Every scientific CV field is exact under
a change of TRAIN-derived query rows; only runtime timestamps/durations differ.
The original undersized refusal and raw timing-inclusive comparison failure
remain in evidence/H206/executor-20261102/native-protocol. This is one synthetic
protocol proof, not broad real-data/native91 qualification or permission to use
unapproved labels. Native failures/hard negatives stay in persisted outcomes.

Input role admission and learner qualification remain separate. Bind approved
original TRAIN refs, full role/alias exclusion closure, encoder source/model
and all tensor shapes/dtypes/hashes before body reads. Native cache identity
includes the complete context, encoder, source, environment, protocol, ordered
slate, actor and seed; identical semantics alone permit reuse. Acquisition
interfaces invoke real inherited GP mutations and enforce256 distinct DSLs
within4096 attempts. Production parent banks, feature/projection generation
and first actual admitted-context transport remain unresolved bindings.

Native teachers are flushed create-only to S3, GET-verified at exact VersionId,
then consumed by the learner from those same hashed bytes. Persist checkpoints,
per-round source/identity/cost/novelty proofs and fixed cost panels before any
wait or stop. Component-level family novelty is distinct from repeated contexts,
new actor/native labels or cache hits. The400 CPU-core-hour/18 GPU-hour original
campaign caps remain cumulative; BudgetLedger requires an existing reconciled
ledger and never silently creates or resets it. Setup, waits, failures, helpers,
search, tokens, updates and custody are included in charged operation scopes;
per-stage breakdowns are not charged twice.

Current source implements CPU cell/round interfaces and rejects CUDA models.
The GPU reservation controller, full multicell/H178 selection wiring, actual
model/input/native91 qualification and source/openPR/resource/queue gates still
require binding. --execute remains refused. See MISSING_PINS.md for exact
artifacts/owners. No scientific training, real-data verdict, GPU allocation,
serving change or queue release follows from this source PR.


## Prospective phase implementation binding — RESEARCH-EXEC-0918

No H206 scientific candidate forward, optimizer update or GPU allocation has
occurred. The implementation now binds the original constructor/state and
explicit GPU/CPU phase lifecycle without changing k1/16/64/256, three seeds,
five folds, three rounds,600 updates, matched initialization/batches or
terminal-only selection. The original CPU wrapper remains CPU-only; a separate
GPU worker requires actual merged source, immutable genuine qualification,
the parent's canonical GPU reservation/attempt and the assigned3090.

Qualified actor-independent slate templates may be prepared by the pinned
actual GP generator for each cell/round/role. They must contain256 distinct
programs, actual generator source/seed/attempt receipts and the same admitted
original TRAIN/cut/tensor identities. They are not cached1152-pool resamples.
The current measured actor is bound and ranks the full slate anew; no common
ranked prefix is asserted across trained models. No template/payload is opened
for a transitively protected role, no B score is allowed before1800, and no A
payload/score is allowed before the durable checkpoint-bound B lock.

Persist step proofs/losses and terminal partial state before advancing, including
during a storage or resource failure. The only runtime reclamation is verified
owned native scratch/cache duplication after exact-version S3 recovery.
Creation and custody costs remain charged once to the original cumulative
campaign. A terminated scientific attempt is INCOMPLETE, not a retry/reset or
partial-arm efficacy result. Source fixtures and metadata-only full-controller
controls never establish actual GPU/model/native/data scientific qualification.


## Receiving-main admission continuation (after PR1822)

The actual CPU91 singleton full-vector/pick/utility gate is already established
at15f365; it is not repeated by this continuation and does not qualify CUDA.
Receiving main merged the stricter storage-alias/buffer/anchor freeze guard in
PR1833 while the H206 dependency manifest still named its predecessor. Keep
both identities in the receiving receipt; update the source pin only with
new targeted stricter-guard and controlled real-learner checks. The original
r882 model/assets/scorer/reference and scientific comparisons do not change.

The existing StudyController and PhaseSupervisor now distinguish study from
runtime_qualification authority. There is no second study or controller.
Scientific study admission additionally requires an actual source/input/fold/
model-bound CUDA runtime proof and explicit purpose-matched queue release.
CPU compatibility or an unscoped queue document cannot satisfy that gate.

A prospective FIT-only runtime qualification uses the existing first seed206101,
fold0 four registered arms (k1/16/64/256), all three expert rounds, the same600
updates/round, singleton32-example batches, actor/native/learner interfaces,
per-fit300s and original cumulative/per-round budgets. All four heads start
from the pinned original seed206101 state dbe16fbce2060a0493b4129c5e5579414a874e1fde3d156ad6c2b44339313a2d.
The qualification never opens the baseline/B/A panels, applies the selector,
claims an efficacy endpoint or supplies a checkpoint to the scientific study.
All qualification updates, native fits, setup/finalization, failures and queue
reservations consume the ORIGINAL400CPUh/18GPUh; no extra window is granted.
A separate single qualification claim is retained beside the original single
scientific attempt, with no automatic retry, overwrite or reset. Both use the
same canonical deadline, which cannot be renewed between purposes.

Qualification itself still needs merged exact source, genuine approved complete
original-TRAIN/role/encoder/slate inputs, actual model/CPU91 bindings, reconciled
canonical costs, fresh resource/device admission and explicit queue authority
for runtime_qualification. Its presence in source authorizes no compute now.
Current accounting is UNRECONCILED and execution stays disabled.

Every actual phase emits an immutable source-bound worker/numeric/device/exit
receipt. Each of twelve qualification round proofs must bind600 durable step
objects,19200 examples, positive DOPE/residual gradients and parameter/forward/
checkpoint changes. All36 actual phase receipts and the complete round objects
are restored for acceptance. The controller writes PENDING_FINAL_ACCOUNTING;
PASS requires external observed controller/worker exit, closed logs, reconciled
canonical36-phase cost range and no unresolved tail. A live controller, a missing
phase, stale source/input, unknown allowance or a renewed deadline refuses.

The learner rechecks the original campaign/phase deadline before each training
forward, backward and optimizer step. A deadline consumed by scoring or verified
persistence leaves zero additional steps; any already completed step, gradient/
loss/state journal and partial checkpoint remain durable on the terminal path.
Templates cannot override the registered cell/seed/fold/arm/k/role partition,
even when a candidate input-payload hash happens to match that template.


Production Context now distinguishes native TRAIN JTT1 from original_train and
derivation ObjectRefs. The approved role contract binds both. Before production
payload use, verify the actual original VersionId/body and approved producer
witness linking row/cut/dtype/native matrix/raw encoder/tensor source identities.
Native receipts retain both original_train_sha256 and native_train_sha256.
This consumer proves custody/linkage and the native header/matrix/cut shape;
ARGON retains semantic original-CSV row/encoder derivation qualification. The
consumer never infers that qualification from tokens, hashes or a boolean alone.
Generated CPU fixture contexts remain separate from production admission.

After actual controller exit, prepare_runtime_acceptance requires an existing
positive original-budget runtime-qualification-acceptance reservation. Its raw
proof-restoration/publication work is charged once by that parent. It returns
PENDING_ACCEPTANCE_PUBLISHER_COSTS, never automatic CUDA PASS. Scientific
admission also requires the original owner's versioned acceptance candidate and
final publisher observed-exit/tail cost receipt, with no unresolved cost scope.
No helper can spend from an unknown allowance or claim its own final tail is zero.
