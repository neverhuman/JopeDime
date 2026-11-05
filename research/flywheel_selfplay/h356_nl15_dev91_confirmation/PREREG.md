# H356 — one-shot confirmation of frozen H352 NL15 on dev91

Authority: `/home/ubuntu/nudges/hw_t0935_jrb01_h356_nl15_dev91.md` (Nov 5, 2026).
Publish this protocol and all executed source before development payload decoding
or prediction. H352 source/artifacts: `02dfee2f7e0b8543324494dd4f99d074cd6b93b6`;
closeout SHA256 `042905333cc018c3baef067f670a86261ef6f0919d732f676c92465c62256eeb`.
No new TRAIN fit, full-TRAIN fit, model update, hyperparameter selection, or GPU.

The candidate is the **nine existing H352 policies**, P0/P1/P2 × roots
30001–30003. Each policy retains the original family-hash route to its four
frozen fold locks; routing extends unchanged to dev families. Use all nine,
never choose a dev winner, average folds, or refit. Retain each lock's chosen
(tau, k, a), three frozen members, margins, iteration counts and eight bags.
All 1,080 final model hashes and 36 lock hashes must match H352 custody. Native
rankers remain LambdaRank, num_leaves15, max_depth3 (actual leaves<=8), L2=1,
implicit L1=0, min_child_samples20; unchanged veto. No new constructor is fitted.

One **original r882 top-eight input view**, identical to H352, produces nine
91-context pick vectors. Do not substitute r3175-ordered input features or
introduce another candidate view. Compare these same picks against r3175 slot0
(primary) and r882 slot0 (secondary). Use only the pinned 91 regression DL-Val
contexts / 19 families in `dev91_contexts.json`, joined to real catalog v246;
TRAIN families are disjoint. The immutable roster preserves source_split
train75/inner16. No other DL-Val, sealed, holdout, meta-test, fresh-cohort or
DL-Test rows. No raw dataset train/test files. DOPE c51a56d0 stays unloaded.

Before inference: exact-VersionId GET, whole-object SHA, and each used member
SHA for existing S3 H352/H144/H281 custody. Three retained derivatives (priors,
r882 costs, r3175 reference JSON) have prior H287/H310 hashes but no established
original derivative S3 versions. Preserve byte-identical, create-only H356
custody copies, then exact-VersionId GET/hash them; identify them as **copies**,
not newly discovered original versions. Opaque whole-archive checksum transport
is distinct from row decoding: extract only the 91 pool members, descriptors,
roster and r882 reference; never extract/decode neural checkpoints or other pool
members. Decode mixed-cache features with exact dev91 os.pread ranges only.

Lock source, models and input hashes; create exclusive inference/group markers.
Decode program/features only, predict each nonempty fold/root batch once, and
fsync picks plus raw predictions. Only after `PICKS-LOCK.json` decode fitness,
gate and reference utilities. Raw utility is gate ? -fitness : -2; no clipping.
Before challenger evaluation, regress r3175 and r882 91/91 per-context utilities
and exact means 1.1389572403748425 / 1.1377941898164525.

Statistics: within-family means, equal weight to each of 19 families; report
per-policy/root and pooled medians, paired utility lift, literal family median,
W/T/L (1e-4 ties), harm/invalid, Elo and CIs. Elo uses the pinned H302 two-player
MM Bradley–Terry point calculation, reference anchored at1500, and its original
bootstrap formula. Bootstrap10,000 whole-family resamples, RNG356191, identical
draw shared across all nine dependent policies and both references. Primary
pooled utility lift is median of nine family-equal means; LCB95 is its 5th
bootstrap percentile. **NL15_DEV91_CONFIRMED iff primary median Elo>1500 AND
paired utility-lift LCB95>0**; ties/below either bar yield
NL15_DEV91_NOT_CONFIRMED. Elo interval bounds are also reported. No AUC, no new
axis, no adaptive rerun. Missing/partial results remain unknown, never negative.

xbabe2 only, nice19, numeric1, systemd6GiB/Swap0, own scratch<=2GB. ONE fresh5s
physical+SMT >=85%-idle sample on freed pool A0–14+sibs64–78; take first<=8,
minimum4, same-interval hostCPU<=70%. One sequential inference worker uses the
admitted affinity. Stop only own cgroup for CPU/load>85, MemAvailable<20GiB,
scratch>2GB or fixed compute stop12:15Z; close12:30Z Nov5, no extension/reprobe.
CI_FREEZE+BATCH_WINDOW: nonforce fresh-parent v1/next/private index only; no
PR/main/fastlane/hostedCI, qualification.lock, live writer or foreign-job change.
Archive all own evidence create-only with exact-version/member recovery proof
before scratch cleanup. CONFIRMED permits an ADOPT_SAFE **nudge only** for the
flywheel owner; it never mutates a live run. Otherwise wait H353 before another
owner-authorized axis. This reused 19-family development panel is not fresh
confirmation, and no full-TRAIN deployment model is introduced by H356.
