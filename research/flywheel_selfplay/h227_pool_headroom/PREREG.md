# H227 preregistration — frozen candidate-generation headroom

Preregister before proposals or model construction. Zero optimizer updates; frozen
r882 COPY and DOPE c51a56d0. Real-S3 H139/H195 cohort only: TRAIN1219/205,
DL-Val91/19; no sealed/holdout/meta-test, outer test, GPU or live-loop access.
All91 r882 baseline utilities and mean1.1377941898164525 must match bit-for-bit
before any proposal. The physical source remains untouched; execute exact merged
Git blobs and pinned dependencies.

## Arms and source binding

One paired context/seed and the same TRAIN-only retrieval parent feed native
calls for15 and63 children. NATIVE16 is parent+15; NATIVE64 parent+63.
FPS16of64 retains that same parent and15 greedy farthest-point selections from
the same63 using normalized native proj16 vectors. Earliest generator order
breaks ties. Width-specific native calls are not assumed to be nested prefixes.
No fitness, oracle result or evaluation label participates in construction.
Durable per-unit candidate locks precede native evaluation. Shared candidates
are evaluated once; each arm is scored separately with frozen r882. Four known
anchors condition the scorer; only the registered arm programs are eligible.

The source planner declares16 proposals in exploration,8 in exploitation, and
caps native fits at4. This offline control binds the requested16-program
exploration width; it is not a replay of that4-fit live budget. No live process,
run directory or custody directory is inspected or changed. Exact source and
all input identities are in manifest.json.

## Estimands and decision

Primary: NATIVE64-NATIVE16 on DL-Val91, averaging within each of19 families
then weighting families equally. Secondary: FPS16of64-NATIVE16. Also report
the separate91-context mean plateau metric. Seeds227101–227103; fresh grouped
fold seeds227141–227143. With zero fitting, folds support TRAIN smoke bank
exclusion and Val family diagnostics; they are not outer-test generalization.
For each contrast report the median over three seed equal-family means and
10,000 shared paired19-family bootstrap draws (seed227150), two-sided95% CI,
per-seed gains/signs and harm. One-sided bootstrap p=(1+draws<=0)/10001;
Holm correction over both contrasts (m=2).

Primary ADVANCE requires median seed context-mean plateau gain strictly
>+0.002, equal-family CI lower strictly>0 and Holm-adjusted p<0.05. Ties
reject. A secondary positive result cannot rescue the primary verdict.
Missing panels or unqualified execution are INCOMPLETE; no partial efficacy
verdict. Paired family Bradley-Terry Elo anchors r882 at1500, exact ties,
prior0.01; descriptive only. Oracle-best budgeted utility per width and its
selected-program gap are diagnostic only, never selection or a decision gate.
No AUC or speed/cost verdict; native evaluation counts and elapsed time are logs.

## Execution and engineering gate

Up to two TRAIN-only engineering smokes, each exercising the complete proposal,
Rust evaluation, external-ID dynamic prediction, all three treatments and an
analysis row on one TRAIN context. Required91-context baseline parity is a
separate prerequisite. Preserve receipts/results and delete only own derived
scratch after durable evidence. Primary requires exact-source full-path smoke
PASS, then exactly one273-unit attempt (three seeds x91 contexts). No primary
retry or implicit resume. Explicit failed/gate-false native results retain
utility-2 per native convention; missing/unordered/nonfinite results or no
completed native result in a unit cause operational INCOMPLETE. No silent
replacement or reduction of width.

CPU xbabe1 only, nice19, whole tree<=4 cores, single-thread math, RSS<=6GiB,
MemAvailable>=10GiB, free disk>=50GB, output<2GB. Between units pause when
load5>70, recheck60s, resume<=60. RSS/RAM/disk/output/deadline remain hard stops;
the supervisor does not abort an in-progress unit because of load. All child
threads inherit and are checked against the same affinity. No other owner is
signalled. Prereg merged by17:15Z, smoke by17:45Z, compute stop21:30Z and
closeout22:00Z on2026-11-02, without reset. Final source/metrics/index/custody
must preserve unknown outcomes and all raw evidence.
