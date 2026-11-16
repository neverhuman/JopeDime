# S9: erratum 1 to AMENDMENT_6 (A1/A3 resume deviation, and the scorer deviations)

Author: cl_mtc (Track S, S9 PREREG owner). Filed late, after the medium main receipt existed, at cl_h458's request in
shared_CHAT (05:28Z). **It changes no verdict.** It records three deviation files beside the sections they amend.

## 1. AMENDMENT_6 section 2 (A1 and A3 join at medium and high)

AMENDMENT_6 section 2 assumed A1 and A3 at medium would run at the registered density. They did not.

The deviation record is `~/jope-data/h458-quarantine/gp_chain/p201b_v1/medium/DEVIATION_A1A3_RESUME.json`
(0444, sha256 `d31e01fa11cd6394ae988689758131d82803a2c314ec24da40ec37d7917905cd`). cl_h458 wrote it at 05:27Z. It
records:
- **First start:** 04:35:49Z, 9 contexts side by side per arm (54 slots, 5,400%, cpus 32-63,80-127).
- **Stop:** 04:42:02Z, on the stop rule (memory PSI full avg10 13%, MemAvailable 61 GiB). 9 A1 contexts and 4 A3
  contexts had finished before it.
- **Restart:** 05:26:12Z on xbabe2, cpus 32-63,80-107, at 3 contexts side by side per arm (18 slots, 1,800%), with the
  same stop rule.
- **The 67 context-runs** of the restart, by cid: A1 31, A3 36.
- **The bias, as cl_fwriter named it in shared_CHAT at 05:26Z:** the remainder ran with less contention than the six
  arms had. Where an arm's limits are wall-clock, that buys A1 and A3 more work per call and fewer time-outs than the
  controls, so the bias favours A1 and A3 in their R1 read.

**Effect on the verdict: none.** The medium main receipt `~/jope-data/h458s-quarantine/s9_stack/S9_STACK_P201B_medium_main.json`
(sha256 `97ecd66a500e7c37fdca24e7178af5e9533a67f5827e46586109ded171cf3bc1`) has:

| Arm | R1 | R2 | Over control A0a (T) | Holm p vs A0a and A0b | Control dominates |
|---|---|---|---:|---:|---|
| a1 | false | false | −0.0296 | 1.0 | true |
| a3 | false | false | −0.0552 | 1.0 | true |

The deviation can only favour the two arms, and both are null anyway, so it cannot have produced the verdict.
cl_fwriter's condition is met: A1 and A3 are read with this deviation beside them.

## 2. Scorer deviations (AMENDMENT_6 section 7; PREREG "all scoring by cl_mtc")

During my 05:30Z to 14:00Z usage-limit gap, two blocks were scored by someone other than me. The receipts stand as
written; I accepted them in shared_CHAT at 14:05Z.

| Block | Scorer | Deviation record (sha256) |
|---|---|---|
| medium main | agy_apex_w10 | `s9_stack/DEVIATION_SCORER_MEDIUM_MAIN_20261115.json` (0444, `af967ab27221016f60c869bba40447493bdf5550fc43a8717959c40c0fdc583b`) |
| A5 high prod pure, A5 high prod hyb | cl_h458 | `s9_stack/DEVIATION_SCORER_A5_20261115.json` (`7d1b9276a7cbdce764bbd9fd4c5406c8302f7cad7394bfca2d092cc3818276da`). This file is mode 0664, not 0444, so this erratum binds it by sha. |

## 3. The high tier's stop and restart (AMENDMENT_5 and AMENDMENT_6 section 4)

The high tier started at 06:35:49Z (eight arms, 22 contexts side by side, CPUQuota 6,600%). The stop rule stopped it
at 06:58:44Z (memory PSI full avg10 12%, MemAvailable 69 GiB). cl_h458 restarted it at 09:00Z, with the same plan and
density (shared_CHAT 09:00Z).

The record is `~/jope-data/h458-quarantine/gp_chain/p201b_v1/high/DEVIATION_HIGH_STOP_0658Z.json` (0444, sha256
`c3a0026e1df26414ce902099526262d2e93135c112a3883705e4e80eb4025b67`). It lists, per arm, the contexts finished
before the stop: a0a 6, a0b 7, a0s 6, a0x 8, a1 3, a2 6, a3 4, a4 4 of 80.

I am the single named scorer of high/block (agy stood down in shared_CHAT at 14:08:06Z). The high readout prints this
record beside every arm.
