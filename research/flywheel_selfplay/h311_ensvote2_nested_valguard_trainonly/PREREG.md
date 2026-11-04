# H311 — nested TRAIN-only VALGUARD

Owner directive hw_t0905_astra_h311.md, 2026-11-04. Shadow research only.
Publish this document nonforce to origin/v1/next before **any new fit**, including
regression. No serving, flywheel or fresh-population qualification is authorized.

## Question and frozen dependencies

Does H295's validity guard improve invalid picks over H303 NESTED while preserving
gain, and does it outperform its shuffled-validity-label control?
DL-Train only: 1,219 contexts, 205 families. No dev91, DL-Val, DL-Test, fresh,
sealed or holdout reads. Use H308's byte-exact TRAIN-prefix loader from
2e2a22eda5, SHA256
65a86b6d636461761eaca9bdba413ba5dec34140cd49d4e0ca020aa849a26c6d.
P0/P1/P2 assign SHA256(salt + NUL + family).digest()[-1] mod 4, with salts
H302-outer-v1 / H308-outer-v1 / H308-outer-v2. Root seeds 30001/30002/30003.
H303 choice and member models frozen at 7bef8870f7; H308 at 6826ee680c.
H295 source export at 5964c67df3baca949785fc81f9f376a5db3c179b preserves
the earlier 891af1e69b prereg/fit-lock implementation. Hash-pin dependencies.

## Mandatory regressions

First fresh H308-style P0 fold0/root30001 with the TRAIN-only loader must match
H303's choice (.05,5,2), inner mean .09612510193329017, full inner table,
member model hashes, and picks SHA
18fdb2538ceabd9a9e5d07ac4a80cf9ff0eaf7597d9dc5fd1645acf0566bd3aa.
Reproduce H295's ENS_VOTE2 **reference** cell, which is H287 ENS_VOTE2:
rounded gains [.053903,.059932,.062204], invalid [14,13,14], and exact
published utility vectors. This is not a regression of the later guarded arm.
Either mismatch => REGRESSION_FAIL; stop without new guard fits.

The mandated historical regression uses H308's packet-construction protocol:
all TRAIN labels are built in its separate regression process, with task access
guarded. It is explicitly distinct from the stricter guard-task boundary below.

## Arms and task boundary

A: archived H303/H308 NESTED picks, hash-checked. Reuse its models, do not refit
the selector except the mandated regression. Score saved refit members to recover
their votes/raw scores for H295 fallback; require reproduced outer picks equal
the archived A arrays. Preserve H303's chosen agreement a (including a=3), never
silently replace it with H295's old 2-vote rule. Fallback candidates retain H295's
ordering: member votes, bag votes, mean raw score, then smaller candidate id.

B: A plus the frozen H295 568/652-dimensional features, hybrid 50/50
LGBMClassifier + standardized L2 LogisticRegression(C=.05,max_iter=500), and
the exact fallback/slot0-escape constants (.055 and .015). Only guard models
and guard tau are fitted within each outer_train.
N: identical guard/threshold procedure, with validity labels permuted inside
each fitting subset; null seeds 311201/311202/311203 correspond to root seeds.
Do not permute labels across an inner or outer boundary. True training outcomes
remain the common threshold-selection target for B and N, as in H295.

Guard inner folds: sort outer_train families, shuffle with
stream_seed(root_seed,"H311-inner",partition,outer_fold), assign round-robin
to three folds. Model and permutation streams include arm, partition, outer
fold and inner fold; persist their exact integers. Train B and N on the same
inner splits. Threshold grid {.030,.035,.040,.045,.050,.060}.
Threshold objective retains H295 lexicographic (feasible, -invalid, -harmed,
family-equal gain), where feasible means invalid<=slot0, gain>=.85*A,
harmed<=A. Exact objective ties choose larger tau, as owner instructed.
No post-outcome grid, model, tie or decision adjustment.

For threshold evaluation, use **this outer fold's frozen selector refit** on
outer_train, not other folds' archived OOF choices (those models may have seen
the current outer-test labels). Selector predictions here are in-sample with
respect to the frozen selector; guard probabilities are 3-fold OOF. This
conditional inner tuning is a limitation, not an unbiased inner performance
claim. The final outer-test remains absent from all H311 fitting and choices.

Build reusable features without reading pool fitness/gate arrays. Each guard
task reads outcomes only for its outer_train; its label accessor rejects
outer-test indices before lock and counts attempts. Do not materialize global
TRAIN labels in the fit parent. Save each final B/N model and tau plus source,
input and training-scope hashes; fsync model files and fit_lock before permitting
any outer-test validity/utility access. Inventory all 36 task locks before
scoring any B/N outer test. Negative tests must exercise forbidden reads.
Score phase disables fitting and verifies models/locks. Each outer test once.

## Statistics and decision

Per partition/seed and median across all nine cells, report A/B/N vs r882
slot0 and paired B-A/B-N family-equal utility gain. Average contexts within
each of the 205 families, then average families. Bootstrap 10,000 whole-family
draws with RNG302150, the **same family multiplicities across all arms, seeds
and partitions**; pooled draw statistic is median of the nine cell means.
CI95 is 2.5/97.5 percentiles; LCB95 is 5th percentile. Cells are dependent,
not nine independent datasets. Also show each partition's three-seed median.
Report invalid counts and excess versus slot0, B-N invalid delta, harmed
families (gain<-.01), W/T/L (ties not wins), drop-top5 means and intervals.
Drop-top5 removes each arm's five highest family gains; re-trim each bootstrap
draw. MM Bradley–Terry Elo follows H303 with symmetric .01 decisive counts;
label exactly: "reference = r882 slot0 (DL-Train OOF); r3175-anchored Elo unmeasured".
No AUC or speed verdict.

VALGUARD_NESTED_ADVANCE iff all hold:
1. median invalid(B) <= median invalid(A)-2;
2. invalid(B) < invalid(N) in at least 6/9 cells;
3. median paired B-A gain >= -.002 and pooled LCB95 >= -.005;
4. harmed(B) <= harmed(A) in at least 6/9 cells.
If clause2 fails: VALGUARD_NULL_EQUIVALENT. Otherwise if any other clause fails:
VALGUARD_NOT_ADVANCED. Ties in invalid B/N do not win. Report all clauses.

## Resources, custody and limits

xbabe2 CPU only, <=6 idle cores (initial52,56,57,58,59,62), excluding H310
95,106,117,119,120,121. nice19, numerical threads1, systemd MemoryMax8GiB,
aggregate RSS8GiB, hostCPU<=85%, MemAvailable>=20GiB. Scratch<=3,000,000,000B
under /home/ubuntu/jope-data/h311-astra. Stop compute11:15Z, close11:30Z Nov4;
resume never resets deadlines. No other hosts/jobs/GPU/live loop/neural updates.
Create-only S3 prefix experiments/rd_breakthrough_20261030/H311/astra-20261104T085652Z/;
exact-version GET + whole archive and member hashes before own scratch deletion.
Nonforce v1/next only under CI freeze; no PR/main/fastlane/CI action.

Historical H287/H295/H303 design exposure and P0 near-duplicate seeds remain.
H295 null's high absolute gain can reflect the strong shared A selector; the
incremental B-vs-N and invalid contrasts, not null absolute gain alone, test
guard signal. H310 dev91 work is separate and is not read or borrowed here.
