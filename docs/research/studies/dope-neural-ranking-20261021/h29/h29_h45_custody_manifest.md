# HWV-02 evidence custody manifest

Task: v1/HWV-02. Digest authority for the recovered bundle is `~/jope_plan/ns/receipts/H00_H41_MASTER_INDEX.md`. All 13 H38–H41 indexed files were copied from the read-only `/home/ubuntu/JopeDime/.agent/worktrees/flywheel-selfplay-20261023/docs/research/studies/dope-neural-ranking-20261021/` tree and verified by SHA-256; all are under 5 MB.

| Wave | Artifact | SHA-256 | Disposition |
|---|---|---|---|
| H38 | `h38/results.json` | `f0d4c413f0c4def25ae0ffbe40f19b089fa0177e4be079f044c3eea5c7849e9b` | FOUND (SHA ok) |
| H38 | `h38/results_v2_reconciled_noncompute.json` | `0afee6439cd69f63edc1a827621b337c77120c9c03c092635b8663433cc4c545` | FOUND (SHA ok) |
| H38 | `h38/h38_bounded_additive_budget_ledger.json` | `8f48b6185eaa9c4a33129f8a58d86785ee0f4451657c30d522241cc12a4b2656` | FOUND (SHA ok) |
| H39 | `h39/results.json` | `686bb13b8b75af8ca64c731744fb666da67bc7b3d148b44cf1d214a99b9ce34b` | FOUND (SHA ok) |
| H39 | `h39/h39_bounded_additive_budget_ledger.json` | `a09ed0e844caae9ef4a6a2dc6ef0b87399b997907e8c504ce785e4bebdc16115` | FOUND (SHA ok) |
| H39 | `h39/h39_reconciled_execution_vs_denied_reservation_ledger.json` | `7d3f9109c38f6a84086574ca173c18e8d8a226cd76d46d5d58b430817086d6a2` | FOUND (SHA ok) |
| H40 | `h40/results.json` | `5d7e0d4034b12b1c7e284db7eed27c3984365f15ff263963c5dd5d356daf70c5` | FOUND (SHA ok) |
| H40 | `h40/h40_bounded_additive_budget_ledger.json` | `9b61e8f51f9d050cf3ab3abf74fcfe1ec2717264baa82856867f2622b06242be` | FOUND (SHA ok) |
| H40 | `h40/h40_timing_decomposition_and_resource_estimate_v2_closeout.json` | `682b74756f697b69aca67d42bc16321f1ebb24a2a04cb51df75ad98638f223e5` | FOUND (SHA ok) |
| H40 | `h40/h40_timing_decomposition_and_resource_estimate_v3_closeout.json` | `a058e36eaff6ac7da8049d1e252de2e4dcf36d616757744da26cfefe621dc60e` | FOUND (SHA ok) |
| H41 | `h41/results.json` | `57df980cef8423029f9e808a66d7d8155c22be4658e802673d0bcf349323439b` | FOUND (SHA ok) |
| H41 | `h41/results_v2_closeout_reconciled.json` | `4a02ed926ffffb6c61529b24f81f25979c8a14f51053556603c59d72f302b257` | FOUND (SHA ok) |
| H41 | `h41/results_v3_closeout_reconciled.json` | `d91bbede45c158e9c4e9b32c9b6079bfec9a1e4eb893d2d6d8ccc102fa7f1e20` | FOUND (SHA ok) |

## H29–H34 result evidence

Disposition per ENG:watcher ruling `lanes_0545.md`: **EVIDENCE_MISSING** for H29, H30, H31, H32, H33 and H34. This is not `NEVER_RUN` and not `PASS`. Current eligible-source searches found no recoverable wave directories or result artifacts in:

- Current xbabe2 endpoint `Mac-Studio.local`: `/mnt/jope-archive/jopedime-research/dope-neural-ranking-20261021` does not exist.
- xbabe1 and xbabe3: task-authorized `~/jope-data`, `~/jope_plan`, `~/.codex-screen-monitor`, worktree roots, and named archive path; no H29–H34 directories appeared in the searched roots.
- S3 `s3://veox-jopedime-archive-use2/evidence/`: object metadata enumeration found no H29–H34 keys.

A prior filename-only inventory in this session reported `/mnt/jope-archive/jopedime-research/dope-neural-ranking-20261021/h29/g064-m0-stat-preparation/results.json`, conflicting with current host/path availability. Thus this manifest does not infer `NEVER_RUN`. H29–H34 remain unresolved until that observed archive path is available or its custody owner supplies an index-bound artifact/search receipt. No contents from that inventory were opened.

Wave dispositions: H29 `EVIDENCE_MISSING`; H30 `EVIDENCE_MISSING`; H31 `EVIDENCE_MISSING`; H32 `EVIDENCE_MISSING`; H33 `EVIDENCE_MISSING`; H34 `EVIDENCE_MISSING`.

## Safety and verification notes

- No sealed meta-test, holdout, roster, prediction array, checkpoint, or model-weight payload was copied or opened. S3 inspection was object-list metadata only.
- No DOPE dataset content was accessed or changed; frozen checksum `c51a56d0` was not recomputed because that would require opening the corpus.
- No large binary was copied. No new metrics are claimed or recomputed.
