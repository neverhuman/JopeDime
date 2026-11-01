# H148: cross-fit checkpoint selection

Authority: `~/nudges/h148_res08_t2030.md`, accepted 2026-11-01 before any H148
forward or update. Does selection on a larger family panel improve continuation
checkpoint choice relative to the 91-context DL-Val selector?

## Cohort and fit

Use the exact H145 real-S3 regression roster: 1,219 DL-Train contexts, 205 resolved
families; 91 DL-Val contexts, 19 disjoint families. The H50 selected manifest and
source manifest, H58 feature cache, pool hashes, r882 checkpoint and native
validation reference are pinned in manifest.json. Check cache identities before
decoding features. No DL-Holdout, sealed, meta-test, live-loop or GPU inputs.
The inherited roster's sealed exclusion uses known identities/families; a globally
complete alias roster is unavailable. This experiment cannot establish universal
alias exclusion. Input lineage is the previously audited real-S3 H145/H50 panel,
not newly generated synthetic data.

Reuse H145's family partition algorithm with explicitly assigned NEW seed148140:
sorted families, default_rng permutation, array_split into five groups of41.
For A=0..4, B=(A+1)%5 selects; fit uses only the other three groups (123 families).
All context members move together. A is assessed once per seed, never fitted or
used for selection within its cell. Historical r882 exposure to these TRAIN
families remains: this is conditional continuation generalization, not a fresh
unseen-family test. DL-Val also historically selected r882.

Each of15 cells (five A × seeds148101–148103) starts from an owned COPY of r882
and a fresh AdamW(lr1e-4,wd1e-4). Exactly40 updates, snapshots0/10/20/30/40;
600 updates total, one shared trajectory per cell for both selection rules.
Use the pinned H146 CTRL live recipe: eval mode/no effective dropout; batch16,
slate192 (48 fitness-ranked +144 random); H140/H146 paired sampler consumes eight
steps per round, retains first two,20 rounds. Same seed stream for each A, but
only its admitted fit IDs. SmoothL1 +0.1 KL, temperature0.2, targets clip fitness
to[-4,2], failed gate cost2; gradient norm clip1.0. No alternative recipes.
Only47 adapter tensors/29,211 parameters train. All other state and DOPE composite
c51a56d0 must remain byte-identical after every update and evaluation.

## Selection, assessment, and decision

Native utility is minus original fitness for the first minimum-cost candidate,
or -2 for failed gate; do not gate-filter candidates or clip evaluation utility.
Score each snapshot on B for XFIT, on91 DL-Val for VALSEL. Within each panel,
average contexts within family, then families equally. Choose the highest mean;
exact ties choose the earliest step. Admit that best step only if paired gain
versus step0 is strictly >0.001 AND the two-sided95% family-bootstrap lower bound
is strictly >0; otherwise choose0. Treat gains within1e-12 of0.001 as threshold ties. Do not try a second-best step after rejection.
10,000 paired family draws, percentile2.5/97.5, seed148141+100*A+10*seed_index
(+1 for VALSEL). These are selection diagnostics without multiplicity claims.

Persist both decisions and their SHA before any outer-A prediction in that cell.
Evaluate only the selected steps plus0 on A. Primary: assemble every A once into
1,219-context panels, equal-family XFIT minus VALSEL within each seed, median
across three seeds. Shared205-family bootstrap10,000 draws, seed148141, computes
the median seed contrast per draw. RULE_HELPS requires median>0.001, lower95%>0,
and at least2/3 positive seeds. RULE_HURTS requires median<-0.001, upper95%<0,
and at least2/3 negative seeds. Otherwise RULE_NEUTRAL; all30 selections0 is
explicitly RULE_NEUTRAL. Ties reject. Missing cells/nonfinite inputs/guard or
baseline failure => INCOMPLETE, never imputed zero and never automatically retried.

Secondary descriptive: both choices versus r882 on91 DL-Val. Average each
context's utilities across the five outer-fit policies per seed (uniform mixture
of policies, not an ensemble prediction); family inference remains19 families,
not455 independent contexts. VALSEL's same-panel secondary is selection biased.
Elo on both panels uses H145's canonical400-iteration Bradley–Terry MM,
family-averaged context-pairwise wins, tie margin1e-4, prior0.01, r882=1500;
three arms r882/XFIT/VALSEL, median seed ratings,1,000 shared-family bootstrap
draws seed148142 and median seed Elo per draw. Report identical seed panels and
snapshot hashes rather than claiming pseudoreplicates as independent evidence.
This is regression only; no AUC, binary inference or serving promotion.

## Execution and custody

Before updates, constructor-only smoke must show0 updates/0 scored contexts,
47/29,211 trainables, frozen hashes and effective mode. Data-free tests cover
family isolation, weighting, strict thresholds, best-step ties, incomplete panels,
canonical Elo and batch closure. Freeze this source, rd_argon read-only review,
then squash-merge prereg PR. Launch only from verified merged source, with
prereg merge/review/manifest/source identities in external launch.json. First
scored operation must reproduce all91 native r882 utilities exactly and mean
1.1377941898164525. Failure consumes the attempt; no tuning/retry.

One wave on xbabe2 CPU, nice19; math libraries1, worker1 and supervisor1 OS thread
normally, aggregate hard maximum4; aggregate RSS strictly<6e9 B. Free disk≥40GB;
new H148 data+custody<1GB. Supervisor pauses only its own child process group at
load>.85×affinity cores or MemAvailable<12GiB, resumes below.80×cores with12GiB;
never signals foreign work. Wall limit4h including pauses; hard deadline
2026-11-01T07:30:00Z; one-thread worker makes16 CPU-h ceiling conservative.
Constructor smoke limit600s. No deadline/budget reset on session resume.

Sources are reused by hash from H146 inference/freeze helpers and H145 scientific
functions with attribution in code; peer runtime files and jobs are untouched.
All raw scores, train traces, adapter snapshots and resource samples stay outside
Git in `/home/ubuntu/jope-data/h148/` and
`/home/ubuntu/jope_plan/evidence/H148/`. Only small metrics/receipts enter Git.
Prereg source pins are immutable through closeout. A neutral all-zero result
means no eligible continuation under these rules/candidates; it is consistent
with a generation bottleneck, not proof that every selection method is exhausted.
