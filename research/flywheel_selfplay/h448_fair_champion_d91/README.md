# H448 — Official Fair `dev91-audit-v1` Once-Score of `h448.APEX_FAIR_CHAMPION` Ranked with Kumo

- **Status:** `COMPLETE` (`SCORED`, all pre-readout gates passed; `0` `dev91` bytes read during training or freeze)
- **Entrant:** `h448.APEX_FAIR_CHAMPION` (`STACK_R4_E8_WC`, frozen in `H447` `FROZEN_FAIR_CHAMPION.json`)
- **LOCK `sha256`:** `7d58297314bd306fe8ec19e8510812a7f6bdc54df26b035649f328b32382de68`

## Headline Fair Readouts vs `kumo_tabular_large_e8.v1`

| Slice | Contexts / Families | `h448.APEX_FAIR_CHAMPION` Elo (`95% CI`) | Rank | `kumo_tabular_large_e8.v1` Elo | Gap vs Kumo L | Gain vs `h416.SXB` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **1. Strict Clean-Holdout (`H425` `without_feature` clear)** | `75 / 17` | **`1789.2`** (`[1659.9, 1962.9]`) | **`#7/33`** | `2014.0` | **`-224.9`** | `+63.1` Elo (`+0.0586` u) |
| **2. Canonical `dev91-audit-v1` Board (All 91)** | `91 / 19` | **`1821.5`** (`[1697.0, 1989.8]`) | **`#7/33`** | `2070.2` | **`-248.7`** | `+60.3` Elo (`+0.0521` u) |

## Contamination Screen & Block Ablations

- **`H425` Feature-Only Screen (`X_search` vs `{'clear_families_without_feature': 17, 'clear_without_feature': 75, 'contexts': 91, 'families': 19, 'flag_feature': 16, 's1': 16, 's2': 0, 's3': 0, 's4': 0, 'tiers': {'clear': 75, 'possible': 16}}`, `0` `AUDIT` target reads):** `{'clear_families_without_feature': 17, 'clear_without_feature': 75, 'contexts': 91, 'families': 19, 'flag_feature': 16, 's1': 16, 's2': 0, 's3': 0, 's4': 0, 'tiers': {'clear': 75, 'possible': 16}}`
- **Block Ablations on Canonical 91-Context Board:**
  - `h416.SXB` (stored baseline): `1761.1` Elo
  - `G_ONLY` (`Guard+2L` on GP pool only): `1784.8` Elo (`u = 1.6474`)
  - `GM_CW12` (`Guard+2L + M7 + K3 + F4`, no ICL): `1820.5` Elo (`u = 1.6404`)
  - **`h448.APEX_FAIR_CHAMPION` (`+ R4-e8 104M+24M ICL`):** **`1821.5` Elo** (`u = 1.6737`)
