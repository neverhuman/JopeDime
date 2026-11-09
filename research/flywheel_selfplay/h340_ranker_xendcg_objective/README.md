# H340: rank_xendcg objective

**INCOMPLETE_RUNTIME_FORECAST. Numerical outcome UNKNOWN; zero optimizer fits and
zero XENDCG outer scores.** The one capacity gate admitted only cores1,2,3
(minimum4). The full36 forecast at 2026-11-04T23:12:45.933878+00:00 ended
2026-11-05T03:14:27.922629+00:00, beyond fixed Nov5 02:15Z compute stop. No re-probe,
borrowed core, reduced design, partial score, restart or deadline reset.

Preregistration was pushed before controller launch and any fit:
`2e299f427c6d56184f486620f6a488c951fc1f4b`. See [PREREG.md](PREREG.md) and [metrics.json](metrics.json).
H334/H333/H329/H326 remain closed. This record does not advance or reject XENDCG.

## Hypothesis and setup

Changing the listwise ranking surrogate may reduce sensitivity to noisy failure
contexts. The fixed intervention changes only the existing ranker objective from
`lambdarank` to `rank_xendcg` in the two pinned H287 constructors. Restoring that
value reproduces both complete BASE ASTs exactly; no keyword is added. Veto,
labels/groups/features/seeds, GBDT, rounds, all other constructor arguments,
18-config selection, tie/switch/margin rules and -2.0 sentinel stay fixed.
`label_gain` is ignored by rank_xendcg; no gain override or label transform.
Native objective/label rejection would stop INCOMPLETE_SPEC without workaround;
that branch and actual native objective acceptance were NOT_RUN because no fit
was admitted. Native parameter dictionary fixtures are not live native-fit proof.

Reuse H334 harness and actual33343.799676 CPU seconds as the planning reference.
P0/P1/P2 x roots30001-30003 x4 family-grouped outer folds was the mandatory36-task
TRAIN-only design. Numeric1,nice19,systemd MemoryMax6G/MemorySwapMax0, xbabe2 CPU.
The single5-second sample requires both assigned core and SMT sibling>=85%idle.
Only1(86.01%),2(87.42%),3(88.98%) qualified;7(84.77%),9(24.02%),10(0%),12(64.61%),
14(68.99%) failed. No second sample. Forecast=CPU/(3*.9)*1.15+300=14501.988750889s;
objective cost multiplier1.0 is an assumption, not measured runtime or a speed verdict.

## Outcome and verification

Exact-version archived BASE GET, whole-object and selected-member hashes PASS.
All9 panels,36 fold/root pick+lock hashes and three partition medians were exact:
P0 .0643163334285782; P1 .04821480030677507; P2 .04360142642936269.
No BASE refit or fresh BASE model prediction. Current1219 TRAIN pool hashes,
selected/r882 costs and4 consumed mixed-cache TRAIN feature prefixes matched
archived input identities exactly. BASE arithmetic used TRAIN packets only.
No DL-Val/dev91/DL-Test or protected data rows were loaded.

31 datafree controls PASS (31 executed,0 failed/skipped): full inverse ASTs,
unchanged constructor keysets/round streams, native parameter refusal fixtures,
physical scope/lock refusals, complete report fixture, strict advance/tie/seed
rules, paired bootstrap and resource/forecast boundaries. Actual campaign exit2
is the expected admission refusal, not a successful experiment exit. The
invocation ID matches launch and terminal resource receipts. Model/native fit
logs, treatment locks, saved models and once-only outer markers are all absent.
Physical fit guards were not exercised by an actual fit. Preregistered source
and vendor pins reverified unchanged after controller exit.

## Metrics, decision and next step

`reference = r882 slot0 (DL-Train outer); r3175 not measured on TRAIN`.
All XENDCG gain/CI/drop1/5/10/LOFO/W-T-L/harm/invalid/switch/Elo/configuration
metrics, per-root medians and paired bootstrap CI are UNKNOWN (JSON null), not0.
The locked advance bar remains paired drop10 median>=+.002 with LCB95>0,
full gain>=90%BASE,harmed<=BASE,and positive root medians on>=2/3roots.
Planned paired bootstrap10000/RNG340191 and ordinary10000/RNG302150 were not run
on treatment outcomes. No ADVANCE or XENDCG_NOT_ADVANCED verdict is assigned.
H340 is closed; any future attempt needs a separate owner assignment/window.
No dev91 or serving promotion follows this incomplete run.

## Custody and publication

Create-only S3 `experiments/rd_breakthrough_20261030/H340/gov03-closeout-20261104T231435Z/h340-scratch.tar`; exact version `BD8.wLWRG68mrvSf5ZTjiKoXk4w9p9v0`;
SHA256 `5bef3cdebdda0b1c47e694911996d7aa1dfdd3461a022fdd938323dd6aba578b`. Exact-version GET, whole SHA and all137 file members
verified before deleting2414608 bytes of own scratch. Generated restore removed;
immutable local archive and all source/evidence retained at `/home/ubuntu/jope_plan/evidence/H340/gov03-20261104-t2300`.
Actual controller peak 608878592 bytes, swap0; host limits
held. Both owned units have MainPID0; no foreign job was stopped or modified.

Nonforce v1/next only, private Git index/objects, no models in Git. Hosted CI
NOT_RUN under CI_FREEZE + MASTER_MERGE_HOLD; no PR/main/fastlane/CI dispatch.
DOPE c51a56d0 unchanged and not loaded; CPU selection layer only. No GPU or
qualification.lock access, protected cohorts, live writer changes or foreign jobs.
