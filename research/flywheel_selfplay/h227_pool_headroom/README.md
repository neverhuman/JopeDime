# H227 — Candidate-generation headroom

**CLOSED INCOMPLETE.** The only primary attempt stopped on a hard resource
guard after76/273 context-seed units. No complete three-seed efficacy verdict,
gain interval, Elo or aggregate oracle diagnostic is available. Missing values
remain null. No primary retry, resource waiver or deadline reset occurred.

## Hypothesis and setup

Test whether a larger frozen-r882 proposal pool improves selected-program
utility: NATIVE16 = one fixed TRAIN-retrieval parent +15 native proposals;
NATIVE64 = the same parent +63; FPS16of64 = parent +15 outcome-free farthest
points from those same63, using normalized native proj16. Each pool locks
before native evaluation, and each arm is scored separately by frozen r882.
Oracle-best native utility is diagnostic only and never selects a pool.
See [PREREG.md](PREREG.md), [manifest.json](manifest.json),
[fold_manifest.json](fold_manifest.json) and [metrics.json](metrics.json).

Approved real-S3 H139/H195 cohort: TRAIN1219/205 and DL-Val91/19. Seeds
227101–227103; fresh grouped fold seeds227141–227143. Zero optimizer updates;
DOPE c51a56d0 and an offline r882 COPY. Native first-cost-argmin utility is
negative fitness for a passed gate, otherwise-2. Primary NATIVE64-NATIVE16
uses equal-family means and a paired19-family bootstrap; the separate91-context
plateau mean must improve strictly>0.002. CI lower>0 and Holm m=2 adjusted
p<0.05 are also required. Secondary FPS16of64 cannot rescue the primary.
The source planner has16/8 generated proposals and4 exact fits; this offline
16-program exploration control is not a full replay of that live fit budget.
No live process, run/custody directory, sealed/holdout/meta-test or GPU was used.

## Execution and outcome

Job1 [index backfill #1892](https://github.com/neverhuman/JopeDime/pull/1892)
merged and was verified before H227 implementation.
[Preregistration #1902](https://github.com/neverhuman/JopeDime/pull/1902)
merged at16:07:08Z as `1b463762b90c2f5f4db8143483026802e2947d03`. Its complete tree matched the
reviewed source.22 data-free tests, Ruff/format/compile, source imports and
registry/fold checks passed; all30 xbabe1 input/native-worker identities matched.
Source main-guard succeeded. Full/hosted CI was not run or claimed passing.

The first full-path TRAIN smoke passed at16:16:48Z, actual exit0: proposals,
Rust evaluation, external-context-ID prediction, all three treatment scores and
an analysis row. All64 returned native evaluations completed. Baseline utilities
matched91/91 bit-for-bit, mean1.1377941898164525, before proposals. DOPE/r882
before/after matched, CUDA stayed uninitialized and optimizer updates were0.
Own generated scratch was deleted after durable results. Independent replay
verified widths16/64/16, FPS subset, hashes, first argmin and oracle rows.

The single primary worker started16:20:02Z and stopped at16:45:06Z, actual
exit-15, with `hard RSS/MemAvailable/disk/output/deadline guard`.76 complete units, all in seed227101,
contain4,869 returned native evaluations, all completed; the other seeds did
not start. A77th pool was locked and interrupted. Its10 native JSON files are
preserved, but its outcome and additional native-call count remain unknown.
Including the TRAIN smoke,4,933 returned native records are known. All
completed raw first-argmin, oracle and FPS rows replayed independently.

The supervisor did not persist the exact violating observation. The last
resource sample at2026-11-02T16:45:04.979956+00:00 showed RAM11105603584 bytes, tree RSS
416927744 bytes, free disk192175931392 bytes and load5=56.79150390625.
RAM was near the10GiB minimum; that sample does not prove which field triggered.
The final in-memory DOPE observation is unavailable. A terminal audit found all
30 physical input pins unchanged; this does not substitute for that missing
observation. The76/273 partial panel supplies no qualified three-seed gain,
CI, Elo or aggregate headroom diagnosis. This is INCOMPLETE, with no ADVANCE
or NO_ADVANCE efficacy claim.

CPU xbabe1 used nice19, a four-core tree and single-thread math. Load pauses
were configured at>70/recheck60s/resume<=60; other guards were hard stops:
RSS6GiB, available RAM10GiB, free disk50GB, output<2GB and compute21:30Z.
The original22:00Z closeout deadline remained unchanged.

## Versioned custody

The4,410,203-byte output/source capsule and terminal manifest were uploaded
create-only and downloaded at their exact returned VersionIds. All81 unique
archive members passed byte-size/SHA-256 replay, including the interrupted
native JSON files. Initial model/dataset/native-binary payloads and scratch
tables are excluded and retained locally under their pinned identities.
Manifest `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261102/H227/res08-t1530/terminal-1b463762b90c-c4fde112a6889695/H227-terminal-manifest.json`,
VersionId `4jjn4yf3dDrH1_pbwlr025U5OpRA23tz`,
SHA-256 `c6d2c8ccb64d7d992a421a1b33a0632f72a44d891253f2fd0afcc9f60bb070df`. Both pointers are in manifest.json.

Raw roots `/home/ubuntu/jope-data/h227/attempt-001` and
`/home/ubuntu/jope-data/h227/smoke-001` remain intact. Receipts and independent
replay are under `/home/ubuntu/jope_plan/evidence/H227/res08-t1530/`.

## Next step

Retain the partial evidence. A future complete measurement needs a prospective
resource-admission window and a receipt that records the exact violating
observation. The complete diagnostic needed to choose H228 remains unavailable;
this closeout authorizes no H227 retry or H228 execution.
