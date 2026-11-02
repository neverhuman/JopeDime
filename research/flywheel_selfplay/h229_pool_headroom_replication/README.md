# H229 — Fresh-seed candidate-pool headroom replication

**CLOSED_COMPLETE: NO_ADVANCE.** The single primary completed
273/273 context-seed units. No scientific retry or deadline reset.

## Hypothesis and setup

Test whether NATIVE64 improves frozen-r882 selected-program utility over
NATIVE16; FPS16of64 is secondary. Reuse H227 source
`1b463762b90c2f5f4db8143483026802e2947d03`, preserving generation/scoring/statistics.
Fresh replicate seeds229101–229103 and grouped folds229141–229143; no prior
partial outcomes reused. See [PREREG.md](PREREG.md), [manifest.json](manifest.json),
[fold_manifest.json](fold_manifest.json), [metrics.json](metrics.json).

Real-S3 TRAIN1219/205 and DL-Val91/19 only. Same TRAIN retrieval parent,
paired native15/63-child proposals; outcome-free normalized-proj16 FPS from
the same64; locks before evaluation; frozen r882 scores each arm separately,
first-cost argmin. Oracle-best utility is diagnostic only. Source planner16/8
proposals and4 native fits are disclosed; offline16 control is not a complete
live-fit-budget replay. No sealed/holdout/meta-test, GPU or live-wave access.
Zero optimizer updates; frozen offline r882 COPY and DOPE c51a56d0.

Authoritative preregistered ADVANCE requires positive median primary equal-family
gain, paired19-family CI lower>0, Holm m2 adjusted p<0.05, AND median over3 seeds
of NATIVE64's91-context mean lift vs r8821.1377941898164525 strictly>0.002.
Ties reject. Elo and oracle gap are descriptive; no speed/AUC verdict.

## Execution, metrics and verdict

[Preregistration #2005](https://github.com/neverhuman/JopeDime/pull/2005) merged 2026-11-02T18:39:06Z,
source `d1e8dcbea161d4204156b6c67fe7af29791e8dea`. All11 study blobs match the reviewed source.
25 data-free tests, Ruff/format/compile/import/fold/registry checks PASS.
Full/hosted CI NOT_RUN under assigned targeted-local workflow.

A later main commit6dc949d5cffa8f90351b5562db616c4cb2c48257
([#2026](https://github.com/neverhuman/JopeDime/pull/2026)) changed backend.py,
launcher.py and test_h229.py loaders after preregistration. The original
xbabe1 process executed immutable d1e8dcbe bytes throughout. This closeout
preserves those receiving edits. The25 tests, smoke and scientific verdict
qualify the pinned executed source only; the later loaders were not run or
qualified here. Exact executed/receiving hashes are in metrics.json.

The one full-path TRAIN smoke passed: all3 treatments, 64
returned native records, actual exit0, scratch removed after durable results.
Before proposals baseline91/91 bit-exact, mean1.1377941898164525; smoke DOPE/r882
before/after matched and CUDA stayed uninitialized. Primary worker exit
0; supervisor error `None`.
The xbabe2 observer was INTERRUPTED at19:27Z by host reboot and re-armed at19:46Z. The original xbabe1 supervisor3749369/worker3749416 continued uninterrupted; no second worker, primary restart or deadline reset.

Complete units by seed: `{'229101': 91, '229102': 91, '229103': 91}`. Complete units contain
17483 returned native records, statuses
`{'completed': 17480, 'failed': 3}`. 273 pools locked; 0
interrupted native JSON files retained. Total actual native-call count is
not inferred from returned records. Independent output-only replay verified
complete-row DSL hashes, pool widths, first argmin, FPS and oracle fields;
terminal physical input audit: all30 match=`True`. Final in-memory observation is
present and matches the frozen before-state.

Primary NATIVE64-NATIVE16 median equal-family gain **-0.0680616601**, paired19-family95% CI **[-0.17044949092945047, -0.00019279094999959053]**, Holm m2 adjusted p **1**. Authoritative final gate: `{'primary_median_positive': False, 'primary_ci_lower_positive': False, 'holm_m2_adjusted_p_below_005': False, 'absolute_r882_plateau_lift_strictly_above_0002': False}`. Inherited raw H227 decision `NO_ADVANCE` is preserved separately. FPS secondary cannot rescue primary.

| Arm | Median91-context utility | Median lift vs r882 | Equal-family lift95% CI | Median Elo |
| --- | ---: | ---: | --- | ---: |
| NATIVE16 | 0.561883514 | -0.575910676 | [-1.0196665808057126, -0.183295422314758] | 1322.7 |
| NATIVE64 | 0.484672075 | -0.653122115 | [-1.0876086918084202, -0.25322793800175103] | 1306.52 |
| FPS16of64 | 0.521807814 | -0.615986375 | [-1.0353413897867985, -0.2283515586701002] | 1324.97 |

Oracle diagnostic only (never selection or a gate):

| Arm | Median oracle context utility | Median oracle-selected gap |
| --- | ---: | ---: |
| NATIVE16 | 0.991134093 | 0.435408636 |
| NATIVE64 | 1.07201042 | 0.585470517 |
| FPS16of64 | 1.04933075 | 0.530114324 |


CPU xbabe1, nice19,4cores/math1. Pause RAM<12GiB OR load5>70, resume RAM>=16GiB
AND load5<=60,60s recheck; shared latch carries pressure to the next dispatch.
Hard stops ownRSS>6GiB, RAM<6GiB, free<80GB, output>2GB or22:30Z deadline.
Exact hard-stop fields `None`; all raw pause/stop
observations are preserved in supervisor-guard.jsonl and compact metrics.
Compute22:30Z and closeout23:00Z November2 UTC remain unchanged.

## Versioned custody and next step

Create-only archive 12398040 bytes / 80 unique
members; exact returned-VersionId archive/manifest bytes and every member
size/SHA-256/CRC replay PASS. Manifest `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261102/H229/res08-t1830/terminal-d1e8dcbea161-94808e7012891f78/H229-terminal-manifest.json`,
VersionId `Yt8jsq4D_qITsxEzjQSvatsildKoSLji`, SHA-256 `f38ef39f99f6c716921ca67e9d8461a0e230a41e536562d54bf88bc00dc5512f`.
Raw roots remain `/home/ubuntu/jope-data/h229/attempt-001` and `smoke-001`.
Model/dataset/native-binary payloads and scratch tables remain outside Git and
are excluded from the small output capsule under pinned identities. Local
receipts: `/home/ubuntu/jope_plan/evidence/H229/res08-t1830/`.

NO_ADVANCE: keep the frozen live budget unchanged. N64 oracle context means1.0701–1.0779 remain below r8821.137794, while its median oracle-selected gap is0.58547. Larger pools do not solve this scoped candidate-quality/selection deficit. Retain results; any joint proposal-quality/reranking study needs separate prospective authorization; no live promotion.
