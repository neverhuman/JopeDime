# H418 — H416 `SXB` once-scored on the fixed holdout board `dev91-audit-v1`

**Result: `h416.SXB` = 1766.0 AUDIT-Elo** (95% CI 1661–1912). It ranks 8 of 32.

- It is the **highest-ranked model this project has put on this board**:
  - +62.8 Elo over `h364.XSTACK_2L` (1703.2), ΔElo 95% CI [+2.1, +117.1];
  - +78.9 Elo over `h335.H335_AGG` (1687.1), ΔElo 95% CI [+16.1, +138.6].
- It is **not a new leader.** Seven foundation-model entrants stay ahead: Kumo L 2063.9, TabFM ensemble 1907.7, Kumo S
  1894.9, TabFM 1882.2, TabPFN 3.5 1843.8, TabPFN 3.5 (e1) 1828.6 and TabPFN v3 1786.7. The gap to Kumo L is −297.9
  [−460.1, −175.5].
- **Its paired gains over our two earlier entrants do not pass the H335 primary procedure.** The sign-flip p is 0.076
  against `h364.XSTACK_2L` and 0.074 against `h335.H335_AGG`, against the 0.025 bar. The PREREG reports these
  comparisons; it did not pre-register them as tests.
- **One cell was scored under Amendment 2** (below). With the deploy's own value in that cell instead, the Elo is
  1766.9 and the rank is still 8 (descriptive only).
- **Field effect.** Elo depends on the field, so every stored entrant moved when SXB joined, by at most 6.6. For
  example, Kumo L went from 2058.7 to 2063.9 and `h364.XSTACK_2L` from 1705.9 to 1703.2. No stored entrant was refit,
  rerun or rescored.
- **No adoption claim.** dev91 is a reuse cohort. SXB was chosen on TRAIN P201e and confirmed on TRAIN P201b
  (H416, `SXB_CONFIRMED`).

- **PREREG:** [`PREREG.md`](PREREG.md) at `43020dc51e`, published before the LOCK and the one-shot unlock.
  [`AMENDMENT_2.md`](AMENDMENT_2.md) at `f57a655508`, published before the amended analysis ran.
- **Owner:** `cl_fwriter`, Claude Opus 5.5.
- **Authority:**
  - Jepson typed at 2026-11-10T09:26Z: "confirm: score h416.SXB once on dev91-audit-v1".
  - The PR gate (4:09 AM MST): run only after PRs 2180 and 2181 merged.
  - Amendment 2 (3:22 PM MST): option B.
- **Numbers:** [`metrics.json`](metrics.json). The board files are `scores/h416.SXB.jsonl`, `LADDER.md`,
  `ladder.json` and `ARCHIVE.json` under `research/elo/reference-board/boards/dev91-audit-v1/`. The focus ladder is
  [`LADDER_focus_SXB.md`](LADDER_focus_SXB.md).

## SXB vs our earlier entrants and the leader (paired over 19 families)
| vs | ΔElo [95% CI] | Families W/T/L | Gain | RMSE cut | Clipped mean | Sign-flip p | LOFO min |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `h364.XSTACK_2L` | +62.8 [+2.1, +117.1] | 18 / 0 / 1 | +0.0610 | +5.9% | +0.0507 | 0.076 | +0.0337 |
| `h335.H335_AGG` | +78.9 [+16.1, +138.6] | 17 / 0 / 2 | +0.0625 | +6.1% | +0.0518 | 0.074 | +0.0348 |
| `kumo_tabular_large_e8.v1` | −297.9 [−460.1, −175.5] | 1 / 0 / 18 | −0.3139 | −36.9% | −0.1659 | 0.999 | −0.1935 |

- **Definitions.**
  - ΔElo: family bootstrap, 10,000 draws, seed 335160.
  - Gain: the family-equal mean of u(SXB) − u(other).
  - RMSE cut: 1 − exp(−gain).
  - Clipped mean, sign-flip p and LOFO: the H335 primary procedure (clip ±0.5, one-sided sign-flip).
- **TRAIN vs dev91.**
  - On TRAIN P201b, H416 measured SXB at 11.4% RMSE cut over the locked recipe X0b (sign-flip p < 1e-6).
  - On dev91, X0b is the board's `h335.H335_AGG`. The cut there is 6.1%: the direction holds in 17 of 19 families,
    but the effect is about half the size.
- **Against other entrants** (see [`LADDER_focus_SXB.md`](LADDER_focus_SXB.md)):
  - **The six frontier-LLM entrants:** SXB beats all of them and passes against each. The RMSE cut is 23–53%:

    | vs | RMSE cut | Families W/T/L |
    | --- | --- | --- |
    | Gemini 4 argon high | 23.1% | 15 / 0 / 4 |
    | GPT 6.1 Sol Max | 23.5% | 17 / 0 / 2 |
    | Opus 5.5 Extra | 27.9% | 16 / 0 / 3 |
    | Kimi K3.1 max | 37.6% | 18 / 0 / 1 |
    | GPT 6 Astra Max | 45.1% | 19 / 0 / 0 |
    | Grok 4.7 xhigh | 53.1% | 19 / 0 / 0 |

  - **The classic learners and AutoGluon 300 s:** SXB passes against every one, with RMSE cuts of 30.9–50.4%.
  - **The foundation models:**

    | vs | RMSE cut | Families W/T/L |
    | --- | --- | --- |
    | TabICL v2 | +12.5% | 10 / 0 / 9 |
    | TabDPT 1.3 | +9.9% | 11 / 0 / 8 |
    | Mitra v2 | +6.0% | 9 / 0 / 10 |
    | TabPFN 3.5 (e1) | +3.2% | 5 / 0 / 14 |
    | TabPFN 3.5 | −2.0% | 6 / 0 / 13 |
    | TabPFN v3 | −2.1% | 7 / 0 / 12 |
    | TabFM | −10.4% | 5 / 0 / 14 |
    | TabFM ensemble | −15.7% | 3 / 0 / 16 |
    | Kumo S | −17.8% | 2 / 0 / 17 |
    | Kumo L | −36.9% | 1 / 0 / 18 |

    SXB passes the H335 procedure against none of them.

## Updated ladder (`tools/ladder.py`, anchor `h335.H257_REPLICA` = 1500)
| Rank | Entrant | AUDIT-Elo | 95% CI | Before H418 |
| --- | --- | --- | --- | --- |
| 1 | kumo_tabular_large_e8 | 2063.9 | 1887–2326 | 2058.7 (1) |
| 2 | tabfm_ensemble | 1907.7 | 1745–2123 | 1901.2 (2) |
| 3 | kumo_tabular_small_e8 | 1894.9 | 1741–2116 | 1888.3 (3) |
| 4 | tabfm | 1882.2 | 1722–2093 | 1876.5 (4) |
| 5 | tabpfn_v3p5 | 1843.8 | 1675–2087 | 1837.7 (5) |
| 6 | tabpfn_v3p5_e1 | 1828.6 | 1653–2067 | 1824.0 (6) |
| 7 | tabpfn_v3 | 1786.7 | 1655–1954 | 1782.3 (7) |
| **8** | **h416.SXB (new)** | **1766.0** | 1661–1912 | — |
| 9 | mitra_v2_zeroshot | 1719.0 | 1565–1915 | 1715.1 (8) |
| 10 | h364.XSTACK_2L | 1703.2 | 1618–1820 | 1705.9 (9) |
| 11 | tabicl_v2 | 1700.1 | 1555–1882 | 1697.3 (10) |
| 12 | h335.H335_AGG | 1687.1 | 1607–1798 | 1689.5 (11) |
| 13–32 | tabdpt_1p3 1686.3 … h412.grok4p7_xhigh 1121.7; each down one rank, same order as before | | | |

The full table is in `LADDER.md`.

## Descriptive only (PREREG §5; never a headline)
| Reading | AUDIT-Elo [95% CI] | Rank of 32 | Gap to `h364.XSTACK_2L` |
| --- | --- | --- | --- |
| Board value (mean of the two seeds) | 1766.0 [1661.1, 1911.7] | 8 | +62.8 |
| Seed 335101 alone | 1775.7 [1667.2, 1925.1] | 8 | +73.4 |
| Seed 335102 alone | 1744.2 [1639.6, 1890.8] | 8 | +39.6 |
| Without the five H381/H410-exposed contexts (86 contexts) | 1797.8 [1700.5, 1939.9] | 7 | +83.2 |
| Amendment-2 cell at the deploy's own SXB value | 1766.9 [1661.7, 1912.6] | 8 | +63.7 |

In the Amendment-2 cell (`43022ae6a8bc`, seed 335102), the deploy's own SXB u is 1.34449. The scored X0b u is 1.32802.

## How it was scored
- **Entrant:** `h416.SXB` (design `BCAR.mix`), frozen at its H416 bytes. It runs over the pooled GP population of
  three screener arms per (context, seed):
  - **GS:** the stored H335 Stage V run, replayed from its memo;
  - **GE** (`H381-E2-S3fz`) and **GL** (`H415-L`): new runs of the same locked AGG search.
- **Board value:** the mean of the per-seed values for seeds 335101 and 335102, as for every GP entrant of ours on
  this board.
- **Value sources** (182 (context, seed) units):

  | Source | Seed 335101 | Seed 335102 |
  | --- | --- | --- |
  | SXB deploy value | 91 | 90 |
  | `a2_exempt_x0b` | 0 | 1 |

  No unit lacked a usable H program, and no amendment-1 memo-conflict context occurred.
- **Harness gates:** all 24 passed. Tolerances: replay 1e-9, derivation 1e-6, ladder 1e-6.
  - **Replays:** GS 182 / 182 identical (Δ 0.0 vs stored and vs board). GL 182 / 182 identical. GE 181 / 182
    identical; the one difference is the Amendment-2 cell.
  - **SXB re-derived offline** from the stored deploy matrices: max |Δ| 2.7e-15. The KCAR100 bag equals KFB exactly,
    and the reference predictions match within 8.9e-16.
  - **Envelopes and board:** AUDIT envelopes match the board on 182 / 182 units. The 33 board files match the LOCKed
    manifest. The stored ladder is reproduced exactly (max |Δ| 0.0). No `h416.*` file existed before.
  - **LOCK:** `f923e5b7…`, receipt `ns/receipts/H418_LOCK_sha256.txt`.
- **The ladder is regenerated by the published `tools/ladder.py` from stored scores only.** Its SXB Elo equals the
  analysis's within 1e-9, and the panel and bootstrap Elo agree exactly.

## Run (xbabe0, cpus 16-63,80-127, nice 19, docker `h317-empty:v1`, CPU only)
| Step | Units | Wall | Engine counters |
| --- | --- | --- | --- |
| GE arm | 182 | 1.21 h | 58,229 program runs; 7 timeouts at the 60 s limit; 10 retries |
| GL arm (in parallel with GE) | 182 | 3.04 h | 58,464 program runs; 28 timeouts; 30 retries |
| D1 deploy (seed 335101) | 91 | 3.69 h | 81,733 program runs; 20 evaluation failures; 37 resource pauses (185 s) |
| D2 deploy (seed 335102) | 91 | 3.82 h | 80,950 program runs; 19 evaluation failures; 15 resource pauses (75 s) |

- **Timeline:** the unlock was at 14:41Z on 2026-11-10. The last deploy finished at about 21:48Z.
- **The guard refused nothing:** 0 DL-Val, sealed or denylisted path refusals in every run.
- **The counters are program-level events inside the engine.** A timed-out or failed program is scored as failed
  within its unit, which is the recipe's normal behaviour. Every unit completed.
- **GL took 3.0 h against the PREREG's 1.6 h estimate.** This changes timing only: each unit is deterministic given
  its seed and memo.

## Deviations and disclosures
1. **[`DEVIATION_1.md`](DEVIATION_1.md): two driver defects, both before any evaluation.** No code, LOCK, receipt
   or recipe parameter changed, and the unlock was not repeated.
   - **Missing view link.** The first GE launch failed in 5 s because the merged TRAIN bank-view link
     `neural/views_c32_train1219.npz` was missing from the H418 data root. My port of `d91_h410.py` had dropped the
     line that makes it. The fix was the H381/H410 symlink, sha `eb042276…`, which equals the H335 and H410 LOCK value.
   - **Resume refusal.** The second launch was refused because `gp_lab` appends `screener_manifest` to `config.json`,
     so a `gp` step cannot resume on its own run dir. The attempt-1 dir (config only, no evaluation) was moved to
     `superseded/`, with a `.SUPERSEDED` note.
2. **The first analysis was `INCOMPLETE`** ([`INCOMPLETE_1.md`](INCOMPLETE_1.md);
   [`INCOMPLETE_1.analysis.json`](INCOMPLETE_1.analysis.json), sha `ee9652bf…`).
   - **What happened:** `analyze_h418.py` (21:49–21:55Z) stopped at gate `ge_replay_335102`. In D2's replay of the GE
     search for context `43022ae6a8bc`, one of the 12 `AGG.g50.CV3@12` stock members differed (value Δ 3.43e-5).
     The other 363 GE and GL replays were identical.
   - **No SXB value was written, printed or read.** That output is kept unchanged, and `run_h418_a2.sh` sha-checked
     it before and after the amended analysis.
   - **Likely cause:** each GE arm process ran both seeds of a context, sharing the screener's per-(context, program)
     score cache. Each deploy ran one process per seed.
3. **[`AMENDMENT_2.md`](AMENDMENT_2.md): the owner chose option B (Jepson, 3:22 PM MST).**
   - **Rule:** a listed failing (context, seed, source) cell scores X0b, the stored `h335.H335_AGG` per-seed value,
     with label `a2_exempt_x0b`. The deploy's value is reported as descriptive only.
   - **The list:** [`a2_cells.json`](a2_cells.json) has exactly one cell. The gate passes only if the failing set
     equals the list.
   - **Before the analysis:**
     - The amendment was published at `f57a655508` before the amended analysis.
     - `selftest_h418_a2.py` passed 29 / 29 synthetic cases from the published stage.
   - **The run:** the amended analysis ran once (receipt `ns/receipts/H418_A2_ANALYZE.txt`, 22:38:49Z, exit 0) into
     the new file `results/H418_DEV91_A2.json`.
   - **Effect:** substituting the deploy value moves the Elo by +0.9 and leaves the rank unchanged.
4. **Seed convention.** On TRAIN P201b, H416 scored one seed (335101), because that board's GP entrants are
   single-seed. Here SXB uses both dev91 seeds, as the board's other GP entrants do. This was fixed in the PREREG
   before any dev91 read.
5. **dev91 is a reuse cohort.**
   - H257 was tuned on it.
   - Five contexts were exposed by H381 and H410. Without them, SXB reads 1797.8 (rank 7, descriptive).
   - The H416 design and constants were chosen on TRAIN only.
   - This is a ranking on a reused holdout, not an unbiased generalisation estimate.

## Not read or reported
- **The deploy unit's other outputs** (KFB, BAG, HFR, …) are not reported (PREREG §6). They entered only the harness
  identities above, as pass/fail plus maximum deltas.
- **Untouched:** DL-Val outside `dev91-audit-v1`, DL-Test, sealed data, DOPE `c51a56d0` and `provenance_bind.py`. The
  flywheel writer stays stopped.

## Steps as run (a record: the unlock and the amended analysis each refuse a second run)
```sh
R=research/flywheel_selfplay/h418_sxb_dev91
bash $R/stage_h418.sh pub p1; export H418_STAGE=/home/ubuntu/h418-stage/pub-p1; X="bash $H418_STAGE/code/run_h418.sh"
$X selftest DIR; $X gate; $X precheck; $X import-check; $X inputs; $X lock
$X announce 16-63,80-127 7; $X unlock
$X gp GE 16-63 12 32g; $X gp GL 80-127 12 32g
$X deploy-prep; $X memo-conflicts
$X deploy 335101 16-63 8 48g; $X deploy 335102 80-127 8 48g; $X pull
$X analyze                                   # INCOMPLETE_1 (gate ge_replay_335102); kept unchanged
A="bash $R/run_h418_a2.sh"                  # Amendment 2, after f57a655508
$A stage; $A selftest DIR; $A analyze; $A board
```
The closeout bundle (the pulled run records, analysis outputs and receipts) is archived create-only in S3; see
[`ARCHIVE.json`](ARCHIVE.json).
