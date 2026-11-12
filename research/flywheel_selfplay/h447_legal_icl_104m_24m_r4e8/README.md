# H447 — Legal `104M + 24M` Plain-`ICL` `R4-e8` `[X, S]` Stack Evaluation & Fair Champion Freeze

- **Status:** `COMPLETE` (`6/6` integrity gates passed; `0` `APEX-CONFIRM` opens; `0` `dev91` opens)
- **Frozen Champion Selected (`FROZEN_FAIR_CHAMPION.json`):** **`STACK_R4_E8_WC`**
- **Selection Reason:** WC beat BASE and GM_CW12 on P201b excl10 with FIT-SEL clean-88 non-harm

## Key Results

| Panel | Metric | `BASE_SXB` | `GM_CW12` (No ICL) | `STACK_R4_M20` (`24M`) | `STACK_R4_E8_BASE` (`104M+24M`) | `STACK_R4_E8_WC` (`104M+24M WC`) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `FIT-SEL clean-88` (`n=88`) | Family-mean utility | `1.9691` | `2.2419` | `2.2459` | `2.2461` | **`2.2467`** |
| `P201b excl10` (`n=191`) | Family-mean utility | `2.0407` | `2.2543` | `2.2671` | `2.2700` | **`2.2713`** |
| `P201b excl10` (`n=191`) | Gain vs `BASE_SXB` | `0.0000` | `+0.2135` | `+0.2263` | `+0.2293` | **`+0.2305`** |
| `P201b excl10` (`n=191`) | Gain vs `GM_CW12` | — | `0.0000` | `+0.0128` | `+0.0158` | **`+0.0170`** |

- **P201b excl10 Board Elo (`n=191`, `25` entrants):** `h447.STACK_R4_E8_WC` = **`1856.6`** (rank **`#2/25`**; `kumo_tabular_large_e8.v1` = `1970.3`).
- **Exact Replay & Completeness (`H447.gates.json`):** `FIT-SEL` `106/106`, `P201b` `201/201`, `replay_max_w_diff = 0.0`, `replay_max_u_diff = 3.95e-10`.
- **Next Step:** Wave `H448` scores the frozen `FROZEN_FAIR_CHAMPION.json` recipe once on `dev91-audit-v1`.
