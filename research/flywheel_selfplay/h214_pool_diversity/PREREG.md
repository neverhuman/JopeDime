# H214 — candidate-pool diversity at fixed evaluated width

Owner: luna_v1_res08. Authority: ENG H-wave nudges
`~/nudges/hw_t0200_res08.md` and FINAL `~/nudges/hw_t0230_res08.md`.
Claim: `~/jope_plan/ns/receipts/H214_claim.txt`. Registration precedes H214
utility readouts. This is offline research with zero optimizer updates.

## Frozen state and executable arms

Use a COPY of r882, SHA256
`0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64`.
Re-hash checkpoint bytes and DOPE composite
`c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`
before and after. Model training=false, all requires_grad=false, no_grad inference,
CUDA_VISIBLE_DEVICES="", CUDA uninitialized. No live-loop directory interaction.

Executable arms are NATIVE16 and FPS16 only. Both score EXACTLY16 unique
programs: the common parent in slot0 plus15 proposals. The four H212 revealed
anchors condition frozen r882 predict_costs, but are not four extra candidates
in H214's argmin. H212 N16 is4+16; this difference is explicitly approved by
the final coordinator ruling and is not an open question.

The approved packet contains TRAIN champions and no DL-Val champion DSLs.
Before any outcome, the parent protocol is therefore fixed to the native
proposer's first retrieved family-disjoint TRAIN champion: smallest squared
distance between target landmark12 and stored TRAIN landmark_summary, with
program_sha256 as exact tie break. All arms use the same parent. No utility
is used to choose it; no live journal or missing-DSL fallback is allowed.

NATIVE16 requests16 native proposals and keeps the first15 unique nonparent
programs in generator order. FPS16 requests64 native proposals, starts with the
parent and greedily adds15 programs maximizing minimum squared distance to
selected programs in native proj16. L2-normalize each16-vector; a zero vector
stays zero. Exact distance ties take the earliest native generator position.
Pool selection uses no costs, fitness, labels or fitted feature transform.
Deduplicate by program_sha256, checking that each hash matches its DSL.
Reject insufficient unique output: no refill or reduced scored width.

TEMP16 is NOT_RUN and is dropped from executable arms by the final coordinator
ruling because no native temperature hook exists. There is no surrogate,
post-proposal temperature weighting, seed replacement or neural weight change.
It is absent from the executable contrast family; Holm m=1.

Only the selected16 in each arm receive Rust evaluation. Share identical
evaluations across arms while preserving exact arm order. Unselected FPS
proposals receive features only. Oracle-best-in-pool means best among the
scored16; it is descriptive and cannot trigger extra evaluations.

## Input closure, native utility and randomness

Reuse H212's content-pinned H50/H58 real-S3 regression packet:
1,219 TRAIN contexts/205 families and DL-Val91/19 disjoint families. DL-Val91
is H214's authorized primary endpoint; TRAIN supplies retrieval and one
prelaunch smoke context only. Validate every bank record is DL-Train and
exclude the target family explicitly.

Retain the pinned selected packet, source manifest, caches, champion bank,
model code, baseline reference, native proposer and Rust worker. The existing
H212 manifest SHA and fixed smoke runner SHA are bound in manifest.json.
The checkpoint is relocated to an owned H214 copy with unchanged bytes.
For authorized DL-Val contexts, create fresh H214 CV envelopes from their
pinned train-only H50 raw tables using the unchanged pinned H58 envelope
function. H58's frozen TRAIN cache is never regenerated or changed. Record
raw-table/envelope hashes. Rust always uses run_outer_test=false and the same
pinned evaluator, fitness and CV gate. No outer-test, sealed, holdout or
meta-test payload is read.

Require the unchanged frozen r882 reference to match91/91 exactly, with mean
1.1377941898164525. This baseline is an identity check and the descriptive
r882 comparator. An identity failure stops before study proposals.
Use native first np.argmin over the ordered16 costs; exact ties select the
parent-first position. Utility is -fitness for a completed gate-passing Rust
result and -2.0 for a completed CV-gate failure. Execution errors, timeouts,
missing/incorrectly ordered results or identity mismatch mean INCOMPLETE.

Fresh proposal seeds214101,214102,214103. Derive each context stream seed as
the low31 bits of the little-endian first8 bytes of SHA256 of
`H214:propose:<seed>:<context_sha256>`. Pair seed, target, parent and TRAIN
bank across arms. Native generation is width-dependent: do not assume16/64
outputs are nested prefixes.

## Estimator and strict decision

Per seed, average paired FPS16-minus-NATIVE16 context utility within each of
the19 DL-Val families, then average families equally. The point estimate is
the median of those three seed-level family means. With NumPy default_rng(214150),
make10,000 shared family-cluster bootstrap draws: sample19 families with
replacement, use the same indices across seeds, recompute each seed mean
and their median. Report percentile95% CI [2.5%,97.5%]; LCB95 is its lower bound.

The sole contrast is FPS16 vs NATIVE16, Holm m=1. Margin bootstrap p is
`(1 + count(bootstrap median <= 0.001546)) / 10001`.
ADVANCE only if median gain>0, LCB95>0.001546 and Holm-adjusted p<0.05.
All bounds are strict: ties REJECT. Otherwise NO_ADVANCE; missing any
registered seed/family/panel yields INCOMPLETE. TEMP16 stays NOT_RUN.

Report family-grouped Bradley–Terry Elo using the existing H124 helper
(tie_margin=1e-4, prior=0.01, n_iters=400), separately anchored to r882=1500
and NATIVE16=1500, per seed and median over seeds. Elo is descriptive.
Report unique programs, mean pairwise Euclidean distance on normalized proj16,
and oracle-best-of-scored16 utility. No AUC or speed/tokens-per-second verdict.

## Exact-head admission, limits and closeout

Reuse merged H212 smoke fix#1784. From the exact merged H214 head, run one
real TRAIN-context smoke covering H212 widths16/32/64 through proposal, Rust,
pool record and analysis row, plus H214's two16-total panels. Bind the returned
PASS to that full source SHA. No scientific launch from a stale/synthetic
receipt or unmerged source. No actual H214 utility was read during registration.

One fresh `~/jope-data/h214/attempt-001`, CPU xbabe1, nice19, at most4 active
numeric/evaluator threads (native workers4, single-thread inner math), GPU off.
Use the owned registered xbabe1 worktree and its existing build directory.
The2h wall deadline starts with the real TRAIN smoke and is carried unchanged
into the study; pauses/resume never reset it. Instantaneous stop: disk>93.8%,
free<40GB decimal, MemAvailable<16GiB, aggregate worker RSS>=6GB decimal.
Stop immediately if load5>0.85*online cores, or after300 continuous seconds of
load1>0.85*cores or host CPU>85%. A separate supervisor monitors the process tree
and stops only its own workers. Do not touch xbabe0, xbabe3 or H210's processes.
Retain raw per-panel evidence before reclaiming only fresh evaluator scratch;
never delete source, inputs, prior evidence or another campaign's files.

Open prereg PR by10:30Z. Targeted local checks and self squash merge are
authorized; hosted checks are waived, not passed. The two hygiene failures
reproduced on unchanged main4d9e9d1c (compiler-cache path assertion and DL
encodings.py custody pin) are explicitly nonblocking under the final ruling.
After merge, the single real TRAIN smoke must pass before attempt launch.
At finish, small README/metrics.json(<100KB)/index closeout PR and squash merge.
Keep bulk inputs/arrays/runtime evidence outside Git. Post host/PID/run root/
launch time/deadline to H214_claim.txt.


## Prelaunch source-pin correction (2026-11-02 11:30Z)

The first supervisor refused before creating a worker, reading model/cohort data,
or consuming a scientific attempt: the registered H212 runner bytes had changed
on main. Bind H212 commit `c39a96589f340e3d8faa833840e8ddf224aed2a5` (#1788),
whose only runner change removes duplicate scratch-directory creation in its
unused `run` entry point. The reused smoke/evaluator/predictor helpers and analysis
are unchanged. Verify the committed and working runner hashes, pin analysis before
import, and read the original input manifest from that immutable Git commit.
Later H212 S3 receipt metadata does not change H214's registered input closure.
A positive data-free regression now loads the real upstream helper and manifest;
negative controls require refusal on runner/analysis drift. No estimand, seed,
width, arm, model, evaluator or input identity changes.

The conservative original campaign clock remains 11:08:24.326111Z to
13:08:24.326111Z on 2026-11-02, including this repair. No budget reset.
Scientific attempts consumed and utility readouts remain zero before real smoke.
