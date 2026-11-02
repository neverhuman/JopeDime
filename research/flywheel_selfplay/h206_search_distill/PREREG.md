# H206: Rust-CV best-of-k search distillation

Authority: **Jepson via Grok Bot, card answer 8:22 PM MT**.
Owner: gpu_fused_redteam_codex; backup rd_astra.
Status: **QUEUED / NOT RUN / EXECUTOR SKELETON**. Target: xbabe3 RTX 3090.
This PR registers the experiment and CPU-only preparation code. It authorizes
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
