# H63 — W8 original-training-row resplit diagnostic

Status: **execution preparation; no result or verdict yet**. H63 is reserved for W8 under the owner’s `overnight15_t2125.md` assignment. OPUS owns diagnostic execution; ASTRA owns result review, custody and publication. This folder does not authorize duplicate runs.

The scientific protocol is [W8](../rd_breakthrough/PROPOSALS.md#w8--robust-outcomes-on-training-only-resplits); the shared [CPU and S3 contract](../rd_breakthrough/CPU_S3.md) applies. The diagnostic uses zero GPU, at most **2,400 native evaluations and four aggregate CPU-hours**, including pilots, failed attempts and acquisition controls. Fan-out does not multiply these budgets.

Freeze up to 200 hash-selected incumbent/challenger pairs before inspecting resplit outcomes. Prove both programs’ original context, evaluator and completed outcome identities. Seeds 701, 702 and 703 define three additional inner resplits using exactly the original context’s admitted `nq=0` training rows. Both programs use identical folds within each pair. The paired arm spends at most 200 × 3 × 2 = 1,200 evaluations; the equal-cost new-candidate acquisition control gets at most 1,200. Record the original and each new cut identity separately; original scores remain immutable.

No sibling-view labels or tables, outer-test rows, DL-Val, sealed data or synthetic substitute data enter this diagnostic. Missing pair provenance or unverified row closure blocks the affected work; do not replace it with a different study and retain the W8 label.

Report sign and magnitude stability, paired raw utility and costs first. Prospective family Elo, `val_reg_inner` and verified U4 belong to an actually evaluated policy under the common contract; otherwise record null with reasons. Correlated resplits are not independent families. Training-cut sensitivity does not identify selection bias on the repeatedly used development panel.

## Result admission

Once OPUS supplies the verdict, add a small `SUMMARY.md` and `metrics.json` here. They must identify:

- Source SHA, exact command, times, exit status, hosts, shard ledger, seeds and pair/fold roster hashes.
- Manifest S3 URI/version/SHA256, original row/table/program/evaluator identities and verified output custody receipts.
- Intended, attempted, completed, failed and missing evaluations for each arm; total CPU seconds across all hosts and carried budget on resume.
- Stability by original margin and family, all prespecified controls, cost-matched conclusions and the literal stable/unstable/incomplete verdict.
- Any protocol blocker, reviewer disposition bound to result hashes, ready PR URL and merger handoff.

Stop if original gains are stable or reliability loses to equal-cost search, as specified in W8. An interval or incomplete run remains informative without authorizing a reliability head or a serving change. Bulk tables, predictions and receipts stay outside Git; retain only small summaries and immutable custody pointers here.

## Verified preparation custody

Preparation source: `ac86cd7ce93695dc626a27d96d2fdabb7091974c`. Manifest: `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/W8/astra-prep-20261031T033326Z-3199084c/manifest.json`. Version: `x9.3OB8X5Pz.3Uv7sDJ31L6xS5IqdYNi`. SHA256: `75003769950ed92c7790255b5178a786895563b85cf647c9334b9c186276c003`. All listed objects and the manifest passed downloaded-byte SHA256 verification. This preserves instructions, folder setup, host preflight and documentation validation; it contains **no diagnostic result**, and all outcome metrics are null with reasons. No HWM candidate pointer was created.
