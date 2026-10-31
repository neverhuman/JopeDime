# H93 one-time launch-gate continuation amendment

This note amends the H93 preregistration only for the exact stopped state recorded below. It authorizes one continuation after merge; it does not change the experiment, candidate admission rule, resource limits, retry policy for any other cell, or readout.

## Sealed stopped state

- Authorized run root: `/home/ubuntu/jope-data/h93/h93-wave`.
- H93 preregistration SHA-256: `92968f7920e6592d060ba37cddc381c2ea48f121e9ed306ec8250f132c44bd91`.
- `budget.json` SHA-256: `cd8acb213aa6100ff69e2dd5cb59e78b6629e3b382672b95316f5de2accba033`.
- Existing attempts, in ledger order:
  - EB4/seed30 timing attempt1 completed; 60.184124 seconds cell wall; receipt SHA-256 `522dd8f170b1c9c3e426db99b283463dce280ecdf726028f2446cf1b03090e76`.
  - CONTROL/seed30 attempt1 completed; 66.264630 seconds cell wall; receipt SHA-256 `5d979ffc7ac4062ee0e57feff687333266e2e411446719cffa89aee635fa5bef`.
  - LA/seed30 attempt1 stopped with `filesystem / launch gate requires below 94.5% used (94.50%)`; 32.776532 seconds cell wall and 256.624944 seconds resource/lock wait remain charged. It is a failed pre-scoring attempt, not a completed outcome.
- LA/seed30 attempt1 phase SHA-256: `1aad8e9acbadf411b09180336bb02a5caf4eecd6a4d34af05989bb604bad4f18`; phase is `stage=loading`, `outcome_started=false`, with the same attempt token as the ledger.
- LA/seed30 attempt1 log SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` (empty). No LA/seed30 attempt1 outcome receipt or adapter snapshot exists.
- The stopped `incomplete.json` SHA-256 is `8fa8b254cfe87d84d1b2c07d9cfe10064ae33d07c6fcae339bdc48585af6ad19`.

These hashes identify the operator-local stop state; the run artifacts and outcomes stay outside Git.

## Sole authorized continuation

1. Preserve the existing budget and all receipts in place. Count the failed LA/seed30 attempt1 and its wall/wait charges against the original 2,700-second cell-wall and three-hour wait caps. Never reset or reconstruct the ledger.
2. If and only if the exact sealed state above still matches, the phase remains pre-scoring, no LA/seed30 attempt1 result or adapter snapshot exists, and all original launch gates pass, run **LA/seed30 attempt2 once**. Retain attempt1 unchanged. The continuation must use the existing FIFO helper and H93 coordinator lock, and must remain behind older live FIFO tickets.
3. Do not repeat EB4/seed30 timing or CONTROL/seed30. After LA/seed30 attempt2 succeeds, run only the remaining cells in original seed/arm order: GN and EB4 for seed30, then all four arms for seeds31 and32. The existing host-load retry rule remains the only automatic retry rule for other cells.
4. If the disk launch gate fails again for LA/seed30 attempt2, stop the wave, preserve the new failed attempt and ledger, and record the unknown outcome. Do not lower the 94.5% disk gate, rerun a scored cell, or authorize any additional disk-gate retry under this note.
5. Keep the original 40-update recipes, fixed r882 COPY, adapter-only training, frozen DOPE hash, CPU/RSS/memory/disk limits, 12-cell completeness requirement, Elo/family-bootstrap readout, and admission thresholds unchanged. No holdout, outer-test, sealed, live-loop, or terminal DL-Val read is authorized by this continuation.

The continuation becomes effective only after this note is merged. A bounded operator entry point must verify the merged note/source identity and the sealed run-state hashes before resuming; it may not bypass the existing FIFO, coordinator lock, or resource checks.
