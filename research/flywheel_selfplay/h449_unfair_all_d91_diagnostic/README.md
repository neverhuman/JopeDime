# H449 — Quarantined Matched Comparison: `ALL-NON-DEV91-COMPLETE` (`0%` `dev91`) vs. `ENTIRE` (`100%` `dev91` In-Corpus Fit) + Token & Compute Cost Ledger

> **STRICT QUARANTINE & SCIENTIFIC INTERPRETATION (`quarantined_unfair_diagnostic = true`, `canonical_promotion_forbidden = true`):**
> - Zero files in `research/elo/reference-board/` or canonical checkpoint directories were modified. The official fair `dev91-audit-v1` score remains **`h448.APEX_FAIR_CHAMPION`** (`Wave H448`).
> - Arms fine-tuned on `everything_all_d91_bank.npz` (`100%` `dev91` included in training) measure **in-corpus fit to `dev91`, NOT out-of-sample generalisation**, and are kept outside the canonical `dev91-audit-v1` ranking and outside any generalisation comparison with external reference models.

# Fair (`NON-DEV91`) vs. Unfair (`ENTIRE` incl. `100%` `dev91`) Elo & Token/Compute Cost Ledger

| Entrant | Regime (`dev91` Exposure) | Board Elo (`91` ctx, `95% CI`) | Clear-Holdout Elo (`H425`) | Gap vs Kumo Large (`2070.2`) | Real Tables Seen | Cumul. Cell-Tokens (Adapt Cell-Tokens) | `dev91` Train Cell-Tokens | Approx Train FLOPs | Train Compute (GPU-h + CPU-h) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `h448.APEX_FAIR_CHAMPION` | FAIR CANONICAL (0% dev91; STACK_R4_E8_WC) | **1821.5** (`[1697.0, 1989.8]`) | `1789.2` | `-248.7` | `1,368` (`0` d91) | `339.55B` (`+1.20B`) | `0.0M` | `9.55e+19` | `101.58` GPU-h + `0.0` CPU-h |
| `h449.FAIR_ALL_NON_D91_STACK` | FAIR COMPLETE NON-DEV91 STACK (0% dev91; 3392 complete tables) | **1828.6** (`[1699.7, 2001.8]`) | `1794.1` | `-245.7` | `3,392` (`0` d91) | `340.35B` (`+2.00B`) | `0.0M` | `9.58e+19` | `101.92` GPU-h + `0.0` CPU-h |
| `h449.FAIR_ALL_NON_D91_ICL_ONLY` | FAIR COMPLETE NON-DEV91 PURE NEURAL ICL (0% dev91; 3392 complete tables) | **1672.4** (`[1604.7, 1764.9]`) | `1653.8` | `-399.0` | `3,392` (`0` d91) | `340.35B` (`+2.00B`) | `0.0M` | `9.58e+19` | `101.92` GPU-h + `0.0` CPU-h |
| `h449.UNFAIR_ALL_D91_STACK` | UNFAIR ENTIRE STACK (100% dev91; 3525 complete tables) | **1830.2** (`[1701.0, 2003.7]`) | `1795.9` | `-244.1` | `3,525` (`133` d91) | `340.37B` (`+2.02B`) | `6.5M` | `9.58e+19` | `101.92` GPU-h + `23.8` CPU-h |
| `h449.UNFAIR_ALL_D91_ICL_ONLY` | UNFAIR ENTIRE PURE NEURAL ICL (100% dev91; 3525 complete tables) | **1676.1** (`[1605.6, 1772.4]`) | `1657.3` | `-395.8` | `3,525` (`133` d91) | `340.37B` (`+2.02B`) | `6.5M` | `9.58e+19` | `101.92` GPU-h + `23.8` CPU-h |
| `kumo_tabular_large_e8.v1` | External Proprietary Pretrained Foundation Model (8-way ensemble) | **2070.2** (`[1892.5, 2335.7]`) | `2014.0` | — | undisclosed | undisclosed (proprietary KumoAI corpus) | unknown | undisclosed | eval `18.04` CPU-h |
| `kumo_tabular_small_e8.v1` | External Proprietary Pretrained Foundation Model (8-way ensemble) | **1900.2** (`[1745.3, 2124.5]`) | `1853.2` | — | undisclosed | undisclosed (proprietary KumoAI corpus) | unknown | undisclosed | eval `3.49` CPU-h |
| `tabfm_ensemble.v1` | External Pretrained Tabular Foundation Ensemble (6.59 GB weights) | **1915.6** (`[1752.0, 2134.8]`) | `1878.7` | — | undisclosed | undisclosed (external pretrained checkpoints) | unknown | undisclosed | undisclosed |


## Key Findings & Provenance
- `h449.FAIR_ALL_NON_D91_STACK` (`0%` `dev91`, `3392` complete tables): **`1828.6` Elo** (`[1699.7, 2001.8]`, vs Kumo Large `-245.7` Elo)
- `h449.UNFAIR_ALL_D91_STACK` (`100%` `dev91` **in-corpus fit, not generalisation**, `3525` complete tables): **`1830.2` BT-Elo** (`[1701.0, 2003.7]`, virtual position `#7/34`, nominal ΔBT-Elo vs stored Kumo Large reference `-244.1`, W/T/L `1/0/18`, Δu `-0.2594`, in-corpus inflation vs canonical `H448` `+10.6`)
- `h449.UNFAIR_ALL_D91_ICL_ONLY` (`100%` `dev91` **in-corpus fit**, pure neural `R4-e8`): **`1676.1` BT-Elo** (`[1605.6, 1772.4]`)
- **Quarantined Lock SHA256:** `0e8097bd2f647b3eaf9b68e449ca7ad2fbcb9506300fc55fa45ff69293960815`
- **Readout & Token Ledger JSON:** `/home/ubuntu/jope_plan/evidence/H449/diag_results/H449_UNFAIR_DEV91.json`, `/home/ubuntu/jope_plan/evidence/H449/diag_results/TOKEN_AND_COMPUTE_COST_LEDGER.json`
