# H97 scratch-collision recovery

The first H97 coordinator stopped after its timing cell and the CONTROL/WD
seed-33 outcomes completed. The DROP seed-33 outcome attempt 1 failed before
scoring because timing and outcome cells shared the same scratch directory.
That 3.75-second physical attempt remains charged in the original budget
ledger. Its log, phase file, `incomplete.json`, and scratch directory are
preserved as failure evidence. The timing, CONTROL and WD results are retained
and must not be rerun.

The worker now includes the full cell identifier in its staging path, so
`timing-DROP-seed33-attempt1` and `DROP-seed33-attempt1` cannot collide. The
fix changes staging identity only; optimizer settings, data, scoring and
readout remain frozen by `PREREG.md`.

After this repair is merged, resume only the exact stopped prefix by passing
the repair PR number and `--resume-stopped-prefix` to `run_wave.py`. The
coordinator rechecks the live-H93 barrier and merged source, takes its
exclusive coordinator lock, and validates the four prior ledger entries,
receipt hashes, phase files, exact scratch-collision traceback, stop record
and incident receipt. It refuses a changed, scored, incomplete, repeated or
otherwise unknown prefix. It does not clear or recreate the ledger. The
continuation record preserves its prior ledger hash, records operator downtime
against the three-hour wait allowance, and starts DROP seed 33 at physical
attempt 2. The already complete timing, CONTROL and WD artifacts remain the
inputs for closeout; only missing outcome cells run.

Example, only after the repair PR has merged and H93 owns a live FIFO ticket:

```sh
nice -n 19 python3 research/flywheel_selfplay/h97_reg_ema/run_wave.py \
  --cpu-pr REPAIR_PR_NUMBER --resume-stopped-prefix
```

Existing timing and outcome receipts may name both the pre-repair and repair
source commits. Final metrics list every `source_commit`; the summary still
requires one preregistration, fold assignment and initial adapter identity.
No admission or publication is allowed before all twelve paired outcome cells
complete and the preregistered family-grouped Elo, median, confidence interval,
strict tie rule and harm gate are evaluated. Do not delete the failed scratch,
rewrite the budget, rescore CONTROL/WD, or rerun the timing cell.
