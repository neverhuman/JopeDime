# H448 — Official Fair `dev91-audit-v1` Once-Score of `h448.APEX_FAIR_CHAMPION` Ranked with Kumo

- **Status:** `COMPLETE` (`SCORED`, `0` `dev91` bytes read during training or freeze)
- **Entrant:** `h448.APEX_FAIR_CHAMPION` (`STACK_R4_E8_WC`, frozen in `H447` `FROZEN_FAIR_CHAMPION.json`)
- **LOCK `sha256`:** `7d58297314bd306fe8ec19e8510812a7f6bdc54df26b035649f328b32382de68`
- **Precondition & Authorization Disclosure:** Scored **before `H445` `READOUT_DEV.json` existed**, under `OWNER_DIRECT_D91` in session `e066f19a` (`2026-11-12T14:15:51Z`: *"Please ensure we are moving quickly towards getting our first real ELO launch"* and `2026-11-12T14:37:44Z`: `Proceed`). In `metrics.json`, `h445_precondition: true` alongside `h445_dev_status: "OWNER_DIRECT_D91"` records this explicit owner-directed launch authorization rather than a completed `H445` readout.

## Headline Fair Readouts vs `kumo_tabular_large_e8.v1`

| Slice | Contexts / Families | `h448.APEX_FAIR_CHAMPION` Elo (`95% CI`) | Rank | `kumo_tabular_large_e8.v1` Elo | Gap vs Kumo L | Comparison vs `h416.SXB` |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| **1. Strict Clean-Holdout (`H425` `without_feature` clear)** | `75 / 17` | **`1789.2`** (`[1659.9, 1962.9]`) | **`#7/33`** | `2014.0` | **`-224.9`** | leads `h416.SXB` by `+63.1` Elo (`+0.0586` u point estimate) |
| **2. Canonical `dev91-audit-v1` Board (All 91)** | `91 / 19` | **`1821.5`** (`[1697.0, 1989.8]`) | **`#7/33`** | `2070.2` | **`-248.7`** | leads `h416.SXB` by `+60.3` Elo (`+0.0521` u point estimate; primary paired test not passed: clipped `+0.0210`, `p = 0.239`, LOFO `+0.0043`, families `16/1/2`; vs `h364.XSTACK_2L` `p = 0.043`, vs `h335.H335_AGG` `p = 0.041`, both `> 0.025`) |

## Contamination Screen & Block Ablations

- **`H425` Feature-Only Screen (`X_search` only, `0` `AUDIT` target reads):** `75 / 91` contexts (`17 / 19` families) `clear` without feature-distribution flags; `16` contexts (`2` families) flagged `s1` (`possible`), `s2 = 0`, `s3 = 0`, `s4 = 0`.
- **Block Ablations on Canonical 91-Context Board:**
  - `h416.SXB` (stored baseline): `1761.1` Elo (`u = 1.6215`)
  - `G_ONLY` (`Guard+2L` on GP pool only): `1784.8` Elo (`u = 1.6474`, `+23.7` Elo over `SXB`)
  - `GM_CW12` (`Guard+2L + M7 + K3 + F4`, no ICL): `1820.5` Elo (`u = 1.6404`, `+35.7` Elo over `G_ONLY`)
  - **`h448.APEX_FAIR_CHAMPION` (`+ R4-e8 104M+24M ICL`):** **`1821.5` Elo** (`u = 1.6737`, `+1.0` Elo / `+0.0333` u over `GM_CW12` — nearly all of the `+60.3` Elo point lead over `SXB` in `H448` comes from `Guard+2L` and `M7 + K3 + F4` rather than the `R4-e8` ICL block).
