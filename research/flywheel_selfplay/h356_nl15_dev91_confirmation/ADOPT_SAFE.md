# ADOPT_SAFE — H352 NL15 / H356 dev91 owner handoff

Read-only adoption preparation, directed by
`/home/ubuntu/nudges/hw_t1020_flywheel_h356_nl15_adopt_safe.md`.
H356 is CLOSED_RELEASED at `7b37158919e4c041fdbd7ba9125beccbc2482f79`.
This note prepares a future owner-approved adoption; it performs no live change.

The exact change is **CPU selection-layer ranker `num_leaves=7` → `15`**, covering
inner fits, final refits, and all eight bag rankers. `max_depth=3` remains fixed,
so actual binary ranker trees have at most eight leaves. Everything else matches
BASE: LambdaRank/default gains/sigmoid1, ranker L1=0/L2=1/min_child_samples20,
features, labels, groups, seeds, iteration choices, thresholds, tie/sentinel
rules, and the unchanged veto GBDT. This is selection-layer work only: there is
no neural adapter/weight change. DOPE `c51a56d0` stays frozen and unloaded.

The result chain is pinned:

- [H352 TRAIN result](../h352_ranker_num_leaves_15/README.md), source
  `02dfee2f7e0b8543324494dd4f99d074cd6b93b6`: ADVANCE; paired drop-top-10
  +0.006371494, LCB95 +0.000973848; Elo1734.47 vs r882 slot0 on TRAIN outer folds.
- [H356 preregistration](PREREG.md), `83219744233a5c2f9bad039ec2272c665980d487`,
  preceded development decoding/inference; picks were locked before fitness/gate
  decoding. [Result](README.md) / [metrics](metrics.json):
  **NL15_DEV91_CONFIRMED**. Versus r3175 (dev91 reference1500), median Elo1658.92,
  CI95[1500,1778.90], one-sided Elo LCB95 1526.74; paired family-equal lift
  +0.016227004, CI95[+0.005464389,+0.029163386], LCB95 +0.006809689.
  The same picks versus r882 give median Elo1725.29 and lift+0.017341084.
- H356 used 91 contexts/19 development families, nine frozen dependent H352
  policies, 1,080 unchanged final models and 36 unchanged TRAIN-selected locks.
  Model fits, TRAIN refits and weight updates were all zero. The full audit and
  exact-head review passed. [Input/artifact pins](input_pins.json) bind the files.

Offline procedure for the flywheel owner, **under a separately approved future
qualification allocation**:

1. Freeze the intended serving policy and input/evaluator contract before any
   further evaluation. H352/H356 contain nine partition/root policies with the
   original family-hash fold routing; their median is a statistic. Do not choose
   the best dev91 root after seeing results, average folds into a new policy, or
   introduce an unqualified all-TRAIN deployment refit.
2. On a **checkpoint COPY**, rebuild the ranker artifact using pinned real-S3
   TRAIN-only inputs (1,219 contexts/205 families), H352 source, original
   TRAIN-chosen settings/seeds and pinned package versions. Preserve the frozen
   feature pipeline and cached descriptors; keep DOPE/neural weights unloaded
   and unchanged. Dev91, other DL-Val, DL-Test, sealed/meta-test and holdout rows
   must never enter fitting, tuning, or policy choice. No rebuild is run here.
3. Compare rebuilt trained-model SHAs and the copied selection/lock identities
   against archived H352 pins. Check native parameters, max-depth/actual-leaf
   bounds, and all 108 unchanged veto SHAs. Keep original fit locks/choices;
   record timing/container differences separately rather than claim unequal
   trained models are identical. Any model/input/hash mismatch blocks adoption.
4. Run an offline **dry selection replay** on the locked TRAIN replay inputs.
   Require exact feature/pick/reload checks and unchanged selection rules; the
   stored H352/H356 replay receipts are references. Use no live writer state.
   This document performs no replay or additional model prediction.
5. Before serving, complete **fresh-cohort qualification through the approved
   pinned evaluator only**, binding cohort/context identities, evaluator binary
   SHA and the exact proposed artifact. Research lanes never touch sealed or
   meta-test data; any restricted evaluation belongs to the owner-approved
   qualification operator. Require an **Elo-lift check versus the incumbent**
   under that frozen evaluation contract. There are **no speed gates**.
6. Apply only at a **future writer restart explicitly approved by Jepson**,
   after the above gates. Preserve the reviewed artifact and rollback copy.
   Rollback restores ranker `num_leaves=7` in inner/final/eight-bag fits, keeps
   depth3 and the rest of BASE (including veto/DOPE) fixed, and uses the known
   BASE artifact/selection contract. No rollback or restart is performed here.

Explicit prohibitions for this handoff:

- No live flywheel/writer mutation, model swap, config edit, pause, or restart.
  Keep the xbabe1 writer stopped at **R5049** and leave
  **AUTORESTART_DISABLED** untouched, as directed by the owner; this task does
  not access that host or independently recheck its runtime state.
- No access to xbabe1, xbabe0, or xbabe3; no GPU or qualification.lock operation;
  no serving-config merge and no foreign-process operation.
- **CI_FREEZE + BATCH_WINDOW** remain in force: non-force `origin/v1/next` only,
  private index; no PR/main/fastlane/hosted CI. No new wave, fit, evaluation,
  rescore, or capacity reprobe is started by this note.

Carry the limits into any owner decision: dev91 reuses 19 development families
from earlier confirmation waves (multiplicity); the nine policies are dependent.
The pooled two-sided Elo CI touches1500 and literal family-median lift versus
r3175 is zero. The one-sided Elo and utility bounds pass the frozen gates.
The inherited finite-MM/closed-form-bootstrap discrepancy in one zero-loss
secondary cell is disclosed in README/metrics; primary points and pooled medians
are unchanged. **Fresh-cohort and serving qualification remain outstanding.**

Scientific closeout:
`/home/ubuntu/jope_plan/evidence/H356/jrb01-20261105-t0935/CLOSEOUT.json`,
SHA256 `b49f8482faad940c095af16b21a4b6ae8852d52dbb7ac6e2188269e90e078105`.
The original SHA-bound local handoff remains unchanged at
`/home/ubuntu/nudges/H356_NL15_DEV91_ADOPT_SAFE.md`.
