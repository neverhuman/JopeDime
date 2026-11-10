# H326: bootstrap-CI configuration selection

**Status: `INCOMPLETE_INPUT_COVERAGE`. BSEL numerical verdict: unmeasured.**

The frozen BASE replay passed all nine panels, all 36 fold/root pick hashes, and all three exact published medians. All 14 data-free selector, cache and physical-boundary controls passed. The input gate stopped with expected exit2 before any BSEL selection, member fit or outer score.

## Hypothesis and setup

A noisy inner mean argmax may select gains concentrated in a few families. BSEL ranks the same eighteen TAU/k/agreement configurations by a whole-family bootstrap median and requires paired median improvement >=+0.002 plus one-sided LCB95>0 before departing from the H303-selected configuration. Same H303 members, seeds, features, LambdaRank, veto and folds. The exact inner/outer bootstrap definitions and tie rule are fixed in [PREREG.md](PREREG.md).

Preregistration `79acbc247b4b6e8069366e87f3a8e44c79aac64f` was published non-force to `origin/v1/next` before recovery, fitting or BSEL scoring. Owner assignment: `~/nudges/hw_t1625_h326_bootsel.md`. TRAIN only: H308 P0/P1/P2 × roots30001–30003 × four outer folds;1,219 contexts/205 families. CPU selection layer only; DOPE `c51a56d0` loaded=false and unchanged; no GPU/dev91/DL-Val feature rows/DL-Test/sealed/holdout/fresh-cohort action.

## Verified outcome

| gate | actual result |
|---|---|
| Archived BASE panels | 9/9 exact |
| Fold/root BASE pick hashes | 36/36 exact |
| P0/P1/P2 medians | 0.0643163334285782 / 0.04821480030677507 / 0.04360142642936269, exact |
| Data-free controls | 14 collected,14 passed,0 failed,0 skipped |
| Required cached config/cell policies | 648 |
| Available selected-config policies | 36 |
| Missing policies | 612 |
| Catalogued inner model files | 12,960; inner policy metadata files0 |
| Campaign BSEL selection / member refit / outer score calls | 0 / 0 / 0 |
| BSEL gain,CI,drop-top-1/5/10,LOFO,W/T/L,harmed,invalid,switches,Elo,Jaccard,acceptance/abstain | Unmeasured |

**Elo label:** `reference = r882 slot0 (DL-Train outer); r3175 not measured on TRAIN`.

The exact-version H318 archive stores `inner_selected_oof.npz` with `picks[N]` and `cids[N]` for the one BASE-selected configuration in each cell. Its fit locks retain eighteen aggregate means/switch counts, which do not recover the required family gain vectors. Every per-cell inner-model catalog lists360 booster text files and zero serialized inner policy metadata files. A verified byte-identical reconstruction of all eighteen policies is unavailable without the missing selected-iteration/strict-margin information. [cache_coverage.json](cache_coverage.json) records all36 checked cells.

The assignment states “no member refits unless a different config is chosen.” A different config cannot be selected from the incomplete grid. The gate therefore preserves unknown acceptance/abstain rates and BSEL metrics; it assigns no scientific `BSEL_NOT_ADVANCED` or `ABSTAIN_DOMINANT` label. No partial scoring or refit exception was inferred.

## Next step

Supply byte-identical full-grid inner OOF policies with matching member/source/fold provenance, or obtain an explicit owner amendment permitting reconstruction before selection. Complete BSEL selection/refit/outer-score qualification after that input/authority gap is resolved. The original compute stop18:45Z and close19:00Z on Nov4 were retained; no deadline reset. [selection.py](selection.py) is the tested selector; [audit_cache.py](audit_cache.py) enforces the present input gate.

## Evidence and custody

Input: exact H318 VersionId `HiO.tOo0ons7PxKtDAPWzfqznixHZ0AP`, SHA256 `c853051526c5eafb0e393e81534ee9e0c75905eff8e14062ba064e8c8848def7`;1,227 selected BASE/cache/catalog members verified. [regression_receipt.json](regression_receipt.json) retains all36 hashes and nine panel comparisons; [input_recovery_summary.json](input_recovery_summary.json) binds recovery. [metrics.json](metrics.json) is 13991bytes and keeps BSEL values null.

Create-only H326 archive: `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H326/gov03-closeout-20261104T170737Z/h326-scratch.tar`, exact VersionId `qGlyfmAGWe_0Zdnk.m7a74vtNYXzMrz4`, SHA256 `c514244d55b6d7777a8315c31369505f9181d3d01425dd59e105be9abd8bebd4`. Exact-version GET and all1269 file hashes were verified before deleting only own `/home/ubuntu/jope-data/h326` (259004682bytes). Receipt object VersionId `jXY9OpEU.wJxwN5qsMikdpBNNcWJkOeJ`. Immutable local archive and lifecycle receipts remain under `/home/ubuntu/jope_plan/evidence/H326/gov03-20261104-t1640`; the generated verification copy was reclaimed. No models enter Git.

CI_FREEZE: source/records published non-force to `origin/v1/next` through a private index. No PR/main/fastlane/CI, foreign checkout/job, credential-store or DOPE change.
