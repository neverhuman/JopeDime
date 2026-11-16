# H458 Track S, S9: AMENDMENT_6, the pilot's arm-list read, A1L's place, the load-matched read, ultra, A5 receipts, the selection record, and errata

- **Author:** cl_mtc (Claude, xbabe2, session 7afc9e1c).
- **Frozen:** 2026-11-15T03:44Z, before any P201B column is scored and before the high tier starts.
  - The medium tier is running on both hosts (xbabe3 since about 02:50Z, xbabe2 since 03:26:43Z).
  - agy's A5 block p201b80_medium_prod_pure has been delivered (03:25Z) and is not scored.
  - No P201B receipt exists: `~/jope-data/h458s-quarantine/s9_stack/` is empty.
- **Rulings and reviews this records:**
  - cl_mtc's posts of 03:27Z (rulings) and 03:39Z (pilot read);
  - cl_h458 03:21Z and 03:22Z (A1L's place, the full-load rule, the a4 disclosure), 03:29Z (the pilot blocks);
  - cl_fwriter's reviews of AMENDMENT_4's tools and of AMENDMENT_5 (W1, W2, the errata in §1), and 03:22Z and 03:27Z;
  - agy 03:17Z and 03:32Z (A5 receipts; no objection to "ultra never selects").
- **Unchanged:** everything else in PREREG_S9.md and AMENDMENT_1 to AMENDMENT_5. That includes R1, R2, R1c, R2c, the
  selection rule, the Holm read, the dominance rule, the noise rule, the boards and the scorer s9_p201b_stack.py
  (7b90ae79).

## 1. Errata to AMENDMENT_5

Found by cl_fwriter's review of AMENDMENT_5. None of them changes a rule.

1. **§2, A1L's seeds.** "The 8 calls of 8 seeds use the same 64 seeds as A4" is wrong.
   - A1L's 8 engine calls use seeds 458 + 17j, for stage j = 0 to 7.
   - A4's use 458 + 101(o + 1) + 17j, for outer fold o = 0 to 3 and stage j = 0 or 1.
   - Both come from the driver's `pipeline()` (stage seed + 17j) and its outer-fold loop (+ 101(o + 1)). The two arms
     share no engine seed.
2. **§6, the CPU limit.** "inside cl_fwriter's 85% cap" should read "inside Jepson's 85% average host CPU limit". The
   limit is the owner's, not cl_fwriter's.
3. **Cited times.** AMENDMENT_5 cites two cl_fwriter posts by their labels: 03:14Z (the guard, §4) and 03:20Z (N1, §7).
   - Those labels ran ahead of the host clock (cl_fwriter's own correction).
   - In shared_CHAT.md the posts sit before cl_mtc's 03:03Z post (line 2229) and before cl_mtc's 03:09Z post
     (line 2306) respectively.
   - Both therefore came before AMENDMENT_5 was frozen at 03:08Z, and the order of events AMENDMENT_5 states holds.

## 2. The pilot read sets the P201B arm list: A1 and A3 join

**The rule** (PREREG §4 item 2): "A1 and A3 are added if, in `h458s9.PILOT_HIGH_GP_ALONE`, either one beats single_a on
at least 7 of the 10 contexts."

**The read.** It was scored once, at 03:38Z, by run_s9_pilot_probe.sh (e65c35ea) with score_probe_h458s.py (f0885672),
on cl_h458's packed blocks (SIDECARs 6f5f98b6… and 3d7580db…).
- Receipt: `~/jope-data/h458s-quarantine/s9_probe/pilot_high_v1/gp_alone/SCORE.json`, sha256
  ecd8a9ba4e485b00c104be43b908a1735f77d2c1e63c592d7737a6581e71f27b.
- chain (A1) against single_a: contexts W/T/L 7/0/3. The rule fires. cl_fwriter verified the receipt (03:41Z).
- oof_single (A3): 3/0/7. The rule adds both arms whichever of them fires it.
- **The trigger is fragile** (cl_fwriter 03:41Z). Two of the seven wins are tiny: 538e +0.0001 u and 598b +0.0013 u.
  The rule says "beats" and was fixed in advance, so it stands. The readout prints this beside every A1 and A3 result,
  so that nobody reads the pilot as evidence that A1 is strong.

**Consequence.**
- A1 (chain, 4 stages of G/4) and A3 (oof_single, 4 outer folds of G/4) are registered P201B arms at medium and at
  high, with the same standing as A2 and A4: in R1, R2, R1c, R2c, the dominance rule, the Holm read and the selection.
- They run with the frozen driver (code-2c3bf7446d) on each tier's HOSTS.json split.
- **At medium,** where the six other arms are already running (cl_fwriter 03:41Z):
  - on xbabe3 they are not stacked on the six under the 1600% belt. They start as the six's slots free up, at the same
    contexts side by side, as A1L does at high;
  - on xbabe2 they may start at once, if the total stays at or under 66 slots and CPUSTAT records it;
  - their medium contrasts carry the note "started after the six", with their time-out share per host, beside R1.
- **At high** they start with the six, making eight registered arms. A1L follows (§4).
- The wrapper takes them as its third argument: `run_s9_p201b_score.sh <tier> <tag> a1,a3 [descriptive columns]`.
- Nothing else in the pilot read decides anything. Its other three receipts are reported as read (03:39Z post):
  stage_curves c47237f3…, stack_with_icl 27306595…, ensemble_controls_v2 b5145635….

## 3. The load-matched read (descriptive)

**Why.** At medium on xbabe3, a4 ran out of step with the other arms (cl_h458's disclosure, 03:21Z). It reaches a
context about twice as late as the single runs, so its last contexts run under a lighter load and with fewer time-outs.
A1 and A3 (§2) start later at medium too.

**The rule** (cl_h458 03:22Z; cl_fwriter agreed 03:22Z).
- **The arms it covers** are the arms that started together (cl_fwriter 03:41Z):
  - at medium, the six (a0a, a0b, a0s, a0x, a2, a4). A1 and A3, started later (§2), are outside it, so a late start
    cannot empty the set the a4 read was named for;
  - at high, the eight registered arms. A1L runs after them and is outside it.
- Per host, T_first is the moment the first covered arm finishes its whole context list on that host.
- A context is full-load on that host if every covered arm's run window on it ends before T_first.
- The load-matched set is the full-load contexts of both hosts, by this one rule. It is applied to every host,
  xbabe2 included.
- cl_h458 writes the set as `full_load_contexts` in `<tier>/FREEZE_CALLS.json`, with T_first per host and the windows,
  before the tier's block is packed. It is fixed by the run's own clock, not by any score.

**The read.**
- Every GP-alone and stack contrast of the run (R1's, R2's, the noise contrast, the dominating-control contrast), on the
  load-matched set. It is printed beside R1 with each arm's time-out share per host.
- It decides nothing: not R1, R2, R1c, R2c, the dominance rule, the Holm read or the selection.
- **Size check** (cl_fwriter 03:22Z). The readout prints the set's size per real host. If the set has fewer than 10
  contexts, it says "too few to read" and quotes no T.
- It applies at medium and at high (one host at high). It applies to cl_h458's runs, whose per-context windows
  FREEZE_CALLS.json records. A5's runs carry agy's own host and pause records instead.

**How it is computed, with the frozen scorer.**
1. `s9_loadmatch_hosts.sh FREEZE_CALLS.json <tier>/HOSTS.json <tier> LOADMATCH_<tier>.json` writes a 0444 pseudo-hosts
   file: each of the 80 contexts maps to "full_load" or "rest". It carries the schema, tier and cids_80 fields, the
   per-host counts, T_first and both source shas.
2. The wrapper runs a second time with `S9_HOSTS=LOADMATCH_<tier>.json` and
   `S9_OUT=~/jope-data/h458s-quarantine/s9_stack_loadmatch`, tag `<tag>_loadmatch`. The scorer's host split then gives
   every contrast on the full-load set.
3. That receipt never sits in the selection directory, so the selection tool (§6) never reads it.

## 4. A1L runs after the registered arms at high

This replaces AMENDMENT_5 §6's "at the same time" (cl_h458 03:21Z option (iii), with cl_fwriter's change at 03:22Z;
ruled 03:27Z).
- A1L runs on xbabe2 right after the registered high arms finish. It uses the same 22 contexts side by side, the same
  CPU quota and the same CPUSTAT logging.
- Its level against A0a and A0x carries the note "same host and density, run after the registered arms".
- Everything else in AMENDMENT_5 stands: A1L is descriptive, and the curve a1l_s1 to a1l_s8 is read whole.
- high/HOSTS.json (133259d8…, all 80 contexts on xbabe2) is unchanged.

## 5. Ultra never selects, and its two descriptive reads

**Clarification** (no objection from cl_fwriter, 03:27Z, or agy, 03:32Z).
- PREREG §4 item 2 runs ultra for A0a, A0b and the high-tier winner only, with no A0x.
- R1 needs arm − A0x, so under AMENDMENT_4 §1 R1 is null at ultra.
- The highest tier at which an arm can be selected is therefore high. A high selection does not wait for ultra.

**The two ultra reads.** Both are descriptive and run on the first 20 of the 80 contexts. They cover both outcomes of
high.
- **If high has a winner:** A0a, A0b and the winner at G 1,250, as PREREG §4 item 2 says. An A5 winner runs with A5's
  own a0_s0 and a0_s1, and with no a0_2x.
- **If no arm passes R1 at high:** the null read (cl_h458's proposal, agreed in principle at 03:03Z). A0a, A0b and A2
  run on the same 20 contexts, so the null is stated at all three tiers.
- Either read is announced in shared_CHAT.md before it runs. It is scored with `--first-by-sha 20` and no
  `--dominating`: by run_s9_a5_score.sh for A5, and by a direct scorer call named in that announcement for cl_h458's
  arms.

## 6. The selection record (W1 fix)

cl_fwriter's W1 (MEDIUM) found three gaps in s9_write_selection.sh 81a07e8a:
- it trusted its arguments;
- it never read the higher tiers;
- it found tiers by file-name glob only.

J7 makes this record the only gate on FITSEL. The fixed tool, s9_write_selection.sh (412b560f…), applies PREREG §5
itself:
- **It reads every P201B receipt** in `~/jope-data/h458s-quarantine/s9_stack/` (raw files excepted), not only the
  ones named.
  - Each must be named `S9_STACK_P201B_<medium|high|ultra>_<tag>.json`.
  - Each must be 0444, from the frozen scorer, with the tier's contexts (the 80 of CIDS_80.txt; ultra its first 20),
    and with no column named for another tier.
- **Full accounting.** Every receipt in the directory must be in the expected list given on the command line, and
  every expected name must be present. A superseded receipt is renamed `*.json.SUPERSEDED`, never deleted.
- **"Highest tier".**
  - No receipt of a higher tier may have an R1-passing eligible arm.
  - A medium selection needs at least one high receipt.
  - The tier is medium or high (§5). An ultra receipt with an R1 passer is refused as a contradiction.
- **"Largest T over A0a".**
  - The arm must be the unique argmax of `rules[arm].T_over_ctrl_a` over every R1 passer of every receipt of the
    tier, cl_h458's and A5's alike. A5 is read against its own a0_s0 (AMENDMENT_3 §5).
  - A tie refuses.
  - No eligible arm name may occur in two receipts of one tier, since a re-scored arm would be a max-of-N.
- **What the record lists.** Every receipt (path, sha256, tier, controls, R1-passing list with T), plus the
  Holm-adjusted p over every R1 test run on P201B, pooled over all receipts. The Holm read is reported and decides
  nothing (AMENDMENT_2 §1).
- **Tests.** 26 of 26 pass on fake receipts: the argmax, a tie, a duplicate arm, a descriptive arm, a non-passing arm,
  a higher tier that passes, high missing, an unexpected receipt, a missing receipt, an untiered name, a writable
  receipt, wrong contexts, a column named for another tier, the ultra cases, a wrong scorer, a wrong board, raw files,
  create-once, and a null everywhere. Receipt `selftest/S9_SELECTION_TOOL_TESTS_20261115T034218Z.txt` with its script `selftest/s9_write_selection_tests_20261115T034218Z.sh` (both in SHA256SUMS).

## 7. A5 receipts

- Tags start with the tier: `{tier}_a5_prod_{anc}` and `{tier}_a5_cycle_pure` (agy 03:17Z, with cl_mtc's 03:27Z change,
  and agy's finalize script updated at 03:32Z).
- Each block is scored in its own run by run_s9_a5_score.sh (4ddb1f3f…).
- **Prod blocks.**
  - Eligible arms: `a5_{tier}_{anc}_shrunk` and `a5_{tier}_{anc}_ratchet`.
  - Controls: `a0_{tier}_{anc}_s0` (A0a), `_s1` (A0b) and `_2x` (A0x).
  - Every other column is descriptive. The descriptive contrasts are arm:`a6_…_comm` and arm:`a6b_…_runbag`.
- **Cycle blocks.** Every non-control column is descriptive.
- **Checks before scoring.**
  - The block is P201B's, from one producer, with prior ["dev91-free"], and names every column used.
  - The budget (0444) has a row of exactly the run's contexts for every arm and control.
  - The hosts file (0444, jope.h458.s9_hosts/v1) names exactly the run's contexts.
  - The run uses `--check`, `--procs 8` and `--n-boot 200`, as for cl_h458's runs.
- **Dry test.** For p201b80_medium_prod_pure, the command the wrapper builds matches agy's posted arguments column for
  column. It adds `--check`, and refuses a writable budget and the wrong tier.

## 8. Wrapper and readout fixes

- **run_s9_p201b_score.sh (d5acd881…; W2).** The log starts before the first line, so it also holds the start,
  affinity, thread and check lines. The previous version stays read-only as run_s9_p201b_score.903dcc3b.sh. The dry
  tests pass: rc 2 for a joined-string SIDECAR, rc 3 for the fake block, and both logs now begin with the start lines.
- **s9_readout.sh (83561dfb…).** First named here. It is the plain-text readout of one receipt. Changes:
  - the scorer sha is looked up by path, because the scorer runs as `__main__`;
  - the load-matched branch (§3), with "too few to read" under 10 contexts and the per-host counts from
    LOADMATCH_<tier>.json;
  - AMENDMENT_5 §7 (N1): R1, R2, R1c and R2c are never printed for a descriptive column;
  - the notes of §2 and §4, keyed on the tier in the receipt name: beside A1 and A3 the fragile-trigger note, at
    medium also "started after the six", and beside every a1l_sk column "same host and density, run after the
    registered arms".
  - Tested on fake receipts: the load-matched branch with 55, 7 and 0 full-load contexts, the notes at medium and at
    high, and the one-host and two-host host splits.

## 9. Tools

| Tool | Role | sha256 |
|---|---|---|
| `s9_p201b_stack.py` | the scorer, unchanged | `7b90ae79f83b7f3540bbfaaebd90c0812c2c7a602f7da37eabad445863b8bc47` |
| `run_s9_p201b_score.sh` | cl_h458's tiers (W2) | `d5acd881f261f94e0523e7eeb627e3b613e5e021fbaa7985175f0dcc77e166c8` |
| `run_s9_a5_score.sh` | A5's blocks (§7) | `4ddb1f3f6e3cfa6855cdc423733b3aad1d6c6d4861f56f27f9cc49cbdef10397` |
| `s9_write_selection.sh` | the selection record (W1) | `412b560f94ab7af02b73eec66a1c89daa5660257256f58beac69a0ea14a4e3c8` |
| `s9_loadmatch_hosts.sh` | LOADMATCH_<tier>.json (§3) | `4b04a1f3f072640a65aede1d69e3d2166fc722faad217a00dcc81e6fb69f6882` |
| `s9_readout.sh` | the readout (§8) | `83561dfb9a9b0d3f7da018359e0a9139397524d7dba6bd009c1184b6fb3feac9` |
| `s9_freeze_timeouts.sh` | the context-level freeze read, unchanged | `4c8fd53f9b35df54d3793be20f6f1e7bd7bb987addf8fcadfd22ce6c25fafb10` |
| `s9_budget_table.sh` | budget rows, unchanged | `425a5e587dee430a11df01e6cfad89e2583448ce7f70612da907a4d9dc8a246b` |
| `s9_ensemble_controls.py` | a6b, unchanged | `b739b4d1b5c086a99e7840d7511580e8ecc4c85af4e73c2acdfb45bb098b5d33` |
| `run_s9_pilot_probe.sh` | the pilot read, run once (§2) | `e65c35eafd573977ce0d8597f150ba5576cff58c64d6a2a2572f368436abdb4d` |

**Earlier versions, kept read-only:**
- run_s9_p201b_score.903dcc3b.sh, .02064344.sh, .a1e38fd9.sh and .a45bf077.sh;
- s9_write_selection.81a07e8a.sh.
