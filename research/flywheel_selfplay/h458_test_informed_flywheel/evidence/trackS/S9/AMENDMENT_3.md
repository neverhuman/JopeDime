# S9 PREREG, amendment 3

- **Amends:** PREREG_S9.md (sha256 feaaed4e8a817a7ddb7174a661a45f45dc6b390a3e2fe305df340d3f6dba3456), as amended by
  AMENDMENT_1.md (85c3fc29…) and AMENDMENT_2.md (a956e490…).
- **Written:** 2026-11-15 about 02:05Z by cl_mtc. No P201B column has been run or scored.
- **Prompted by:**
  - cl_h458 01:46Z, the measured medium-tier budget gap;
  - cl_h458 01:51Z and 01:53Z, the work per candidate and `--seeds-per-call` at 2c3bf7446d;
  - cl_fwriter 01:47Z and 01:54Z, the reviews of 644b602e, 1a6f367a and 2c3bf744, none blocking;
  - cl_fwriter 01:53Z, the review of A5 at c6aefd23 (H1 to H4);
  - Jepson 01:42Z (via ENG:watcher): "fine to train on dev91 as long as we are doing different folds".

## 1. Seeds per engine call (option b)

**Problem.** Every engine call injects its generation-0 seeds. At 48 seeds per call, a multi-call arm at medium is
mostly seed screening. A single run at 2 G then fails to dominate oof_chain (about 512 against 300 candidates).

**Rule.** P201B and FITSEL use driver `gp_chain_h458.py` at **2c3bf7446d7a6044f9960d4e05940200dcae5465** (stage
code-2c3bf7446d). Every arm spends **48 generation-0 seeds in total**, split equally over its engine calls:

| Arm | Engine calls | `--seeds-per-call` |
|---|---|---|
| A0a, A0b, A0x (2 G) | 1 | 48 |
| A1, A2 (4 stages) | 4 | 12 |
| A3 (4 outer folds) | 4 | 12 |
| A4 (4 outer folds × 2 stages) | 8 | 8 |

A4's exact share would be 6, which has not been run. 8 is the smallest population that has been run. It adds 16 seeds:
about +9% candidates at medium and +3% at high.

**Expected candidates.** These come from cl_h458's measurement of about 3.1 candidates per generation plus about one per
seed:

| Tier | Single at G, A1–A3 | A4 | A0x |
|---|---|---|---|
| medium | about 172 | about 188 | about 296 |
| high | about 544 | about 560 | about 1,040 |

The driver keeps 48 seeds per call as its default. The value is part of the folder claim. The pilot keeps 48 seeds per
call (it is descriptive, and already running).

**Dominance check, mechanical.** The receipt prints, for every arm and tier, the measured unique candidates (mean and
maximum over contexts). If any arm's mean exceeds A0x's, that arm's "T > 0 against A0x" condition is reported as
"control not dominating", and R1 cannot pass for that arm at that tier.

## 2. Attribution control A0s (cl_fwriter 01:54Z)

**A0s:** single at G with `--seeds-per-call 12`, seed 458, at medium and high.

It is descriptive only: not in R1, R2, the selection or the Holm family. It separates "fewer seeds" from "residual
staging". The readout prints A2 − A0s beside A2 − A0a. cl_fwriter notes that seeds affect only the generation-0 injection
(build_bagged_gen0_population), so A0s ≈ A0a would mean that the A2 contrast is about staging.

## 3. Budget table

Unique candidates stays the unit, as frozen. Every arm row of every S9 receipt also prints worker calls and
slot-seconds (mean and maximum over contexts), read from the arm folders' per-context manifests.

**A5 rows.** A5 and cl_h458's driver spend different work per candidate. cl_h458 01:51Z measured about 2 to 3 times the
worker time behind one candidate. A5 against A0 to A4 is therefore printed as descriptive only, never as a like-for-like
compute contrast.

## 4. Pre-stated reading for A4 at medium

A4 at medium runs 8 calls of 5 generations with 8 seeds each. If A4 fails R1 at medium but not at high, the readout says
it "may be too small to evolve at medium", not "no residual value".

## 5. A5 on P201B and FITSEL (agy's arm, H459)

- **R1 for A5** uses A5's own paired controls from the same driver and partitions:
  - `a0_s0` for A0a;
  - `a0_s1` for A0b;
  - `a0_2x` for A0x.
- **Selection and Holm.** A5's R1 tests join the Holm family of AMENDMENT_2 §1, and A5 can be selected under the
  unchanged selection rule.
- **One scorer.** A5's P201B and FITSEL reads come from cl_mtc's `s9_p201b_stack.py`, the scorer for every S9 arm.
  - No arm gets a FITSEL score before cl_mtc posts the P201B selection.
  - A5's own `score_s9_boards_h459.py` does not score P201B or FITSEL for S9 (cl_fwriter H3).
- **Block format.** A5 is delivered as blocks in the `_load_col_block` layout:
  - files: `{cid}.npz` with `ok`, `q_<n>`, `bag_<n>` and `ref_<n>`;
  - `SIDECAR.json`: `names`, `closure`, `bank_sha256` and the prior.
- **Before any A5 P201B column counts:**
  - H1 is fixed: a missing anchor refuses the run or the context, never silent pure columns under hyb names;
  - H2 is settled: the wall cap is off for the S9 read, or the receipt shows it never bound;
  - H4 is settled: the closure is a required argument.

## 6. Prior

No change. P201B and FITSEL use qhat_v2.pt (dev91-free, bf05e040…), and so do the dev91 look's primary pair.

Jepson's 01:42Z ruling allows the in-corpus prior, so labels stay as cl_fwriter 01:43Z says. When the dev91 look is
named, after FITSEL, an in-corpus-prior pair may be added as a pre-named secondary in the same receipt. That is decided
in an amendment at naming time, never after the look.

## 7. Unchanged

Everything else in PREREG_S9.md, AMENDMENT_1.md and AMENDMENT_2.md stands.
