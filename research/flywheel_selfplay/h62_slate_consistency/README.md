# H62 — W2 slate-consistency diagnostic

Status: **diagnostic complete; author stop verdict corrected during result review**. See [SUMMARY.md](SUMMARY.md) and [metrics.json](metrics.json). H62 is reserved for W2 under the owner’s `overnight15_t2125.md` assignment. OPUS owns diagnostic execution; ASTRA owns result review, custody and publication. This folder does not authorize duplicate runs.

The scientific protocol is [W2](../rd_breakthrough/PROPOSALS.md#w2--slate-consistent-scoring); the shared [CPU and S3 contract](../rd_breakthrough/CPU_S3.md) applies. The diagnostic uses zero GPU and zero new native fits. It does not wait for W1 training.

Use frozen offline r882, 100 hash-selected DL-Train contexts and sampling seed 701. Hold candidate pairs, candidate features, revealed IDs and history fixed while varying real distractors across sizes 12, 20, 192 and the full pool. Compare the current candidate-slate semantic mean with a fixed reference computed only from revealed anchors. Preserve the checkpoint and record the exact implementation and input hashes.

Report pairwise rank reversals separately from common score shifts and numerical ties. Include reversal numerators/denominators, tolerances, per-context results, full-choice regret, legacy `val_reg_inner` and separately named verified U4 wherever evaluated. Report family Bradley–Terry Elo using the common contender panel; unavailable metrics remain null with reasons. r882 is an exposed continuity control, not a fold-clean model.

If real pair order is invariant, record the W2 kill verdict and stop this mechanism before training. If reversals occur, review their magnitude and the scoring contract before later waves inherit it. A diagnostic mechanism finding does not establish a learned improvement, serving promotion, or permission for the optional GPU follow-up.

## Result admission

Once OPUS supplies the verdict, add a small `SUMMARY.md` and `metrics.json` here. They must identify:

- Frozen source SHA, command, start/end times, exit status, hosts, deterministic shard roster and actual CPU seconds, including failed work.
- Manifest S3 URI/version/SHA256, exact input identities and verified output custody receipts.
- Intended and completed contexts/pairs, failures, exclusions and missing shards; zero GPU/native-fit counts.
- Current versus anchor-reference results on identical pairs, the numerical tie rule, and the literal kill/continue/incomplete verdict with its limits.
- Reviewer disposition bound to the result hashes, plus the ready PR URL in FINDINGS section 7 and the merger handoff.

Bulk predictions and checkpoints stay outside Git. Never replace an incomplete diagnostic with a success-shaped placeholder or treat a preparation receipt as an experiment result.

## Verified preparation custody

Preparation source: `ac86cd7ce93695dc626a27d96d2fdabb7091974c`. Manifest: `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/W2/astra-prep-20261031T033326Z-3199084c/manifest.json`. Version: `gCwZ3RDIhMWE4OHqME9hLDNOe8E9oPF8`. SHA256: `80f2df9c744088ca942eb403ef9d03c5875717d49529038f12853abb60ef90f3`. All listed objects and the manifest passed downloaded-byte SHA256 verification. This preserves instructions, folder setup, host preflight and documentation validation; it contains **no diagnostic result**, and all outcome metrics are null with reasons. No HWM candidate pointer was created.
