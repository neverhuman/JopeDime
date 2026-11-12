# H446 — All-Legal Real Regression Pretraining Banks (`MAX_FEATS = 400`, `6/6` Gates Passed)

- **Wave:** `H446` (`h446_all_legal_real_bank`)
- **Owner:** `@agy_apex_w10`
- **PREREG Commit (`pub-r6`):** `5e20d976c2e19d5e1e3bb5c971b3d806d877c1bc`
- **Pinned Stage:** `/home/ubuntu/apex-stage/h446/pub-r6` (`CODE.sha256` prefix `66b0580dbcd36ec6`)
- **Gates (`H446.gates.json`):** **`6/6` PASSED (`"pass": true`, `sha256 = d12ddea435f8081356eb48ef59469ff3dbba7c2e5d05f050e02fb1d070445847`)**

## Summary of Constructed Banks (`/home/ubuntu/jope-data/h335-cl/h446/pub-r6/banks/`)

| Bank | Distinct Datasets | Distinct Families | Wide (`F > 100`) | Cells | SHA-256 |
| --- | ---: | ---: | ---: | ---: | --- |
| `fit_icl_wide400_bank.npz` (`FIT-ICL` strictly) | `907` | `866` | `50` | `22,741,110` | `0fef6b435baa9489d4987895e65b356dfbccc05c7be7ef7e00717b25cd400e7b` |
| `all_legal_excl_p201b_bank.npz` (`P201b` family-disjoint) | `1,368` | `1,319` | `78` | `35,137,085` | `bc5c7d6cd1f42096d0f3b31ac95686e81e7bfc439d07f42f1b6992f20b35763e` |
| `all_legal_bank.npz` (max legal pretraining corpus) | `1,609` | `1,509` | `85` | `39,621,927` | `7d4164f9a9b5e618a2c47b8437c7e06ef87b1de11f3fd36ede2a48d7bfc2dc04` |

## Leakage & Contamination Audit (`g4_zero_holdout_and_audit_leakage = true`)

- `confirm_table_opens = 0`, `dev91_table_opens = 0`, `p201b_audit_rows_written = 0`, `fitsel_audit_rows_written = 0`, `delta_refusal_log_lines = 0`, `stage_v_unlocked = false`, `s3_write_operations = 0`, `sandbox/tables = 661`.
- Dropped `5` `H425` `S4` family-name matches against `APEX-CONFIRM` / `APEX-DEV` / `DL-Val` metadata (`0` table opens) and `214` strict `H425` `S1–S4` feature-sketch matches against the `115` core `APEX-DEV` tables; unmodified `screen_h425.screen` verified **`0 / 115` `flag_feature` hits (`0` `copy`, `0` `possible`)** on `APEX-DEV`.
