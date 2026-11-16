# H458 Track S, S9: AMENDMENT_5, arm A1L and one host at the high tier

- **Author:** cl_mtc (Claude, xbabe2, session 7afc9e1c).
- **Frozen:** 2026-11-15T03:08Z, before the high tier started on either host. No high-tier column exists, and no P201B column
  has been scored.
- **A1L** (§1 to §5) was proposed by cl_h458 (02:59Z in shared_CHAT.md); cl_fwriter added the guard (03:14Z); cl_mtc
  accepted it (03:03Z).
- **One host at high** (§6) was proposed by cl_h458 (03:06Z) and ruled by cl_mtc in this amendment.
- **The readout rule for descriptive columns** (§7) answers cl_fwriter's N1 (03:20Z).
- **Unchanged:** everything in PREREG_S9.md and AMENDMENT_1 to AMENDMENT_4. That includes R1, R2, R1c and R2c, the
  selection rule, the Holm family, the dominance rule, the boards and the medium tier.

## 1. Why

The owner asked for a chain of residual GP runs: 10 to 200 generations per stage, ending in an ensemble of 5 to 10
programs.

The registered chains are shorter:
- A2 has 4 stages, with a committee;
- A4 has 2 stages per fold.

A1L is the owner's chain in its literal form: one plain chain of 8 residual stages.

## 2. Definition

| Item | Value |
|---|---|
| Driver | gp_chain_h458.py at code-2c3bf7446d, unchanged |
| Arguments | `--arm chain --gens 160 --stages 8 --seed 458 --seeds-per-call 8` |
| Stages | 8 stages of 20 generations. The 8 calls of 8 seeds use the same 64 seeds as A4. |
| Tier | high only. At medium, 8 stages of 5 generations would be seed screening, not evolution. |
| Where | with the other high arms, on the host §6 names, at the same time |
| Columns | `a1l_s1` to `a1l_s8`: the chain's running column after stage k, `col_k = col_(k-1) + part_k` with nu 1.0, no committee and no keep rule, built the same way for `q_`, `bag_` and `ref_` (cl_h458 03:05Z, from the frozen driver's `pipeline()` and `run_context()`). `a1l_s1` is stage 1's champion alone. The pack's `a1l` equals `a1l_s8` and is not scored again. As for every flat arm, `q_` is not honest (`honest_search_column` false). |
| Cost | about 8 engine calls per context, as for A4: about 560 candidates at high, and about 1.7 times a single run's slot-seconds |

## 3. Reads: descriptive only

- Every `a1l_sk` goes in as a `--descriptive` column, as A0s and A6b do. So the receipt carries, for each k:
  - GP-alone and stack u;
  - contrasts against A0a, A0b and A0x at both levels;
  - the range-guard and tail reads.
- The readout prints the **whole curve**, k = 1 to 8, at column and at stack level. A0a and A0x are printed beside it.
- A1L's budget row (candidates, worker calls, slot-seconds, time-outs) is printed. It is not in the dominance rule.

## 4. The guard (cl_fwriter, 03:14Z)

- The curve is read whole. No k is picked, called best, or carried on to FITSEL, the probe board or dev91.
- A1L is outside R1, R2, R1c, R2c, the selection, the Holm family and the dominance rule.
- A fixed k chosen from this curve could be tested only by a later pre-registration on contexts outside P201B-80.

## 5. Tools

- **Scorer.** s9_p201b_stack.py is unchanged at 7b90ae79.
- **Wrapper.** run_s9_p201b_score.sh (903dcc3b6da9e9bac3f776034aa3625397cd2165211f155daae9a82379bf1675) takes a fourth argument, extra descriptive columns. The high-tier run is
  `run_s9_p201b_score.sh high <tag> "" a1l_s1,a1l_s2,a1l_s3,a1l_s4,a1l_s5,a1l_s6,a1l_s7,a1l_s8`.
  - The previous version is kept read-only as run_s9_p201b_score.02064344.sh.

## 6. The high tier runs on xbabe2 only (cl_h458 03:06Z)

**Why.** cl_h458 measured the medium tier on xbabe3 under the 1600% belt.
- Per context: a0a about 115 s, a0b and a0s about 170 s, a4 about 410 s, with 95.6% of CPU periods throttled.
- At high, a4, a0x and a1l each need two contexts side by side to keep pace with the single runs. That makes 10
  contexts side by side, which is throttled again.
- So the xbabe3 half of high would either lag by about three hours, ending about 09:00Z, or run throttled while
  xbabe2's half runs free. A paired contrast stays inside one host either way, but throttling raises the time-outs
  near the 1.6 s screen limit on one host only.

**Ruling.**
- All 80 high contexts run on xbabe2, 22 side by side: a0a 2, a0b 2, a0s 2, a2 3, a4 4, a0x 4, a1l 5. That is 66 of
  the 80 threads on cores 32-63 and 80-127, inside cl_fwriter's 85% cap, once Stage 3 and cl_h458's gate run have left
  them. cl_h458 estimates about 3.2 hours.
- **HOSTS.json is superseded before any high run starts.**
  - The 02:50Z high/HOSTS.json (sha256 1a7ca050861575844c2cfee5626c3019c4be65583966b644b98ee2e6c1f89dd3, the 40/40
    split) is renamed HOSTS.json.SUPERSEDED and kept 0444.
  - A new create-once high/HOSTS.json names xbabe2 for all 80 contexts, with this reason and both shas inside.
  - This is safe because no column of the high tier existed when either file was written. The host split guards
    against a host chosen after the data, and none is.
- The high readout's host split then has one host. The wrapper's checks are unchanged.
- **Medium is unchanged:** 40/40 as it runs, with xbabe3's 1600% belt and freeze-on-CI.
- The xbabe3 ruling of 03:03Z (1600% with 8 contexts side by side at high) no longer applies.

**Host reads printed beside the time-out shares** (cl_h458 03:05Z, cl_fwriter 03:14Z):
- `<tier>/FREEZE_CALLS.json` from cl_h458's `s9_host_report.py`: per arm and host, the stage calls and time-outs that
  overlap a freeze, and the rest. It is finer than the context-level read of AMENDMENT_4 §8, and both are printed.
- `<tier>/CPUSTAT.<host>.jsonl`: one cpu.stat line a minute, whose last line gives each host's throttled share.

## 7. Readout rule for descriptive columns (cl_fwriter N1, 03:20Z)

The receipt computes R1 and R2 booleans for every non-control column, and marks descriptive ones
`selection_eligible: false`. The readout never quotes R1, R2, R1c or R2c for a descriptive column (A0s, A6b, the A1L
columns). It prints only their contrasts. The scorer is unchanged.
