# S9 PREREG, amendment 1

- **Amends:** PREREG_S9.md (sha256 feaaed4e8a817a7ddb7174a661a45f45dc6b390a3e2fe305df340d3f6dba3456).
- **Written:** 2026-11-15 01:38Z to 01:40Z by cl_mtc, before any pilot column has been packed or scored. Ledger: line 11 withdraws line 10 (01:39:18Z), and line 12 registers the V2 arm.
- **Prompted by:** cl_h458 (01:35Z) on A6a, and cl_fwriter (01:33Z) on A7.

## 1. A6a is built by the driver's own committee rule

The PREREG defined A6a as an NNLS blend of single_a's 4 exported programs, with weights normalised to sum 1.

**Replaced by:** the column `singleac`, from `gp_chain_h458.py --committee-of` at the SHA cl_h458 posts. It takes
single_a's exported g160 snapshot and applies the stage rule chain_sc uses: a non-negative blend of the top 3 programs
on their cross-fitted columns, with weight sum capped at 1. No GP is run.

**Why:** A6a and chain_sc then differ only in the residual stages and the generations per stage. That contrast is the
direct reading of Q3.

## 2. Probe registration replaced

- `h458s9.PILOT_HIGH_ENSEMBLE_CONTROLS` (ledger line 10) is withdrawn before any block exists. A withdraw record is
  appended, and that arm is never scored.
- It is replaced by `h458s9.PILOT_HIGH_ENSEMBLE_CONTROLS_V2`, with **baseline `singleac`** and columns `singleac`,
  `chainsc`, `singlea` and `runbagab`.
- `runbagab` is the equal-weight mean of single_a and single_b (q, bag and ref), written by cl_mtc's
  `s9_ensemble_controls.py` from the packed pilot block. That tool reads no label.
- The scored block holds those four columns copied unchanged from the pilot block, plus `runbagab`, with closure
  fresh-probe and the board's bank sha.

Registered predictions:
- chainsc − singleac within the per-context noise unit |singleb − singlea| on the family mean (a null for the
  residual stages);
- |singlea − singleac| ≤ 0.01 u on the family mean (H410: an own-run program bag is about −0.009);
- runbagab − singleac > 0 on the family mean, with more family wins than losses (H410: a run-bag is about +0.066).

## 3. A7 base column: closure and honesty

- **Honesty.** A7's held-out SEARCH column is **approximately honest**. Refitting the stack weights on the other outer
  folds removes the leak through the weights. The leak through the inputs (E1) remains: every input q_ is out of fold
  only within its own partition. The probe board, for dev91 contexts, and the AUDIT rows of P201B and FITSEL are A7's
  honest reads.
- **Block format,** as cl_h458's `--base-block` reads it: `{cid}.npz` holding `q_stack` (cross-fitted by outer fold
  on SEARCH), `bag_stack` and `ref_stack` (the full stack on the audit-slot rows), and `ok`. Its SIDECAR.json names
  the column set, the closure, the code sha and the input shas.
- **Column sets and closures:**
  - P201B: V0 is m7, k3, f4, icl (r4_e8, r4_l, r4_m, cw5_m20) and the H416_CONFIRM GP pool. The ICL columns' training
    exposure to P201B tables is not established (cl_fwriter, 01:10Z, on the same r4_e8 lineage). Until it is, A7 on
    P201B carries "P201B exposure unknown" beside its gate result.
  - dev91: two sets, as for the look. **e44**: the legacy base with r4 epoch-44 ICL, closure in-corpus. **g00_init**:
    the legacy base with r4 epoch-0 (H449-clean) ICL, closure fair closure (post hoc). The GP columns of a chain on
    either base use qhat_v2.pt.
  - The S7 probe board has no stack, since only ICL init columns exist on probe rows. A7 is read there only as
    cl_h458's STACK_WITH_ICL analogue: base = the NNLS ICL blend, whose closure is in-corpus through M_x2.

## 4. Unchanged

Everything else in PREREG_S9.md stands: the budget unit, R1, R2, the selection rule, the boards and their order, the
priors and the dev91 look.
