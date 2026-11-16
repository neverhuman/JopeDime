# H458 Track D results: look 2 on all 91 dev91 contexts

**Label on every number here: recall.** The specialists were trained on all rows of the scored tables, AUDIT labels
included, on random splits (`in_corpus_dev91_disclosure = true`, `canonical_promotion_forbidden = true`).

| Quantity | Value |
|---|---|
| Arm | `h458d.RECALL_CANON_E2000_ALL91`: legacy stack + one canonical-slot specialist per family at 2,000 exposures |
| Elo | **2343.8** (95% interval 2181 to 2619 in the look-2 receipt's own fit; 2195 to 2612 in the 35-row ladder re-fit), rank 1 |
| Same stack without the specialist block | 1859.0 (in-corpus), rank 5 |
| Kumo Large | 2070.2; families won, tied, lost against it: 16, 0, 3 |
| Pair test against the stack alone | 17-1-1 families, +1.035 u, sign-flip p 7.6e-6, pass |
| Gate D-4 | not met: 11 of 16 target families swept, 13 needed |
| Scored | 2026-11-15 03:27Z to 03:32Z; ledger lines 3 (pre) and 4 (look); receipt sha256 0bde5d23bdab8e74... |

The full account, with the per-family table, the ladder, the training curves and what followed, is in
[docs/research/waves/h458-h465-black-hole-20261116/context.md](../../../docs/research/waves/h458-h465-black-hole-20261116/context.md).

## Files in `evidence/`

Copies of the records under `/home/ubuntu/jope_plan/evidence/H458` and of run logs under `~/jope-data` on the
training host, taken 2026-11-16. `evidence/SHA256SUMS` lists every file. The originals are unchanged.

- `LOOK_LEDGER.snapshot_*.jsonl`: the look ledger.
- `trackD/look2/`: spec, runner and its output, block sidecar, `SCORE.json`, `per_ctx_u.jsonl`.
- `trackD/specialist_runs/<run>/`: `config.json`, `FINAL.json`, `log.jsonl` of the 19 specialists.
- `trackD/family_pool_v1/`: pool manifest and hash (no rows).
- `trackD/probe/`: probe scores. `trackD/SELFTEST_H458_RECEIPT.json`: the 18 CPU gates.
- `trackD/elo_report_20261114/`: the Elo report's inputs, scripts, figures, latest PDF and `REVISIONS.txt`. Earlier
  PDF revisions are not copied; their hashes are in `REVISIONS.txt`.
- `trackS/`, `trackG/`: scorer-track and harvester-track records.
- `training_curves/`: training logs of the starting checkpoint's run, of `r4_L_A` and `r4_L_B`, of the H464 all-data
  generalists, and the specialists' SEARCH-only probes at milestone checkpoints.

Not copied, by rule: `.npz` blocks, checkpoints, pools, banks, label files.

The official-split check (that no training episode used the official SEARCH/AUDIT split) has not been run; see
section 2 of the context document.
