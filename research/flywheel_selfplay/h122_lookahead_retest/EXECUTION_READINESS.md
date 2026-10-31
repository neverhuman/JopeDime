# H122 launch-readiness audit

Owner transfer: `unstick_t0955_sol_v1_dsc01.md`, ENG:watcher for Jepson.
The merged #1050 supplied more than H121's synthetic scaffold: its `run_cell`
already performs adapter-only CONTROL/Lookahead training, guarded fresh-r882
staging, disjoint-family assessment, official unrevealed argmin evaluation and
reference replay. Its only CLI nevertheless required `--cpu-dry-run`; no
coordinator or receipt-bound collection path invoked that worker.

## CPU implementation

| Requirement | Implementation / verification |
| --- | --- |
| Callable real worker | `runner.py --arm ...` validates an inherited parent-owned qualification lease before payload/checkpoint/CUDA access. Parent-death handling precedes execution. |
| Fixed scientific recipe | Existing 40-update path extracted for direct CPU tests; real AdamW changes trainable parameters, frozen parameters stay exact, and LA synchronizes only at steps 5, 10, ..., 40. The existing DOPE optimizer guard remains invoked. |
| Six paired cells | `run_wave.py` reserves CONTROL then LA for each of seeds 48, 49, 50. No timing cell, retry, best-of choice or unknown-reservation resume. |
| Durable identity | Ledger binds clean merged source, protocol/preregistration and both executable hashes. Every worker result binds the input pins, split, adapter identity/change, trainable inventory, actual updates and phase token. |
| Resource/FIFO admission | Five predecessor closeout attestations precede root/FIFO effects. Two CPU load observations 30 seconds apart precede existing H61 FIFO admission. Existing disk, memory, GPU-free and inherited lease checks apply. |
| Bounded execution | One cumulative 2,700-second cell-wall cap, including setup and failures, plus a separate cumulative three-hour resource-wait cap. Each cell gets remaining wall minus a 10-second shutdown reserve; child RSS stays below 6 GiB. |
| Owned-process cleanup | Each worker has a new session; supervision terminates only that session and also removes descendants when the leader exits first. No foreign PID or peer ticket is accepted. |
| Complete-panel readout | Collector reads only canonical admitted paths, verifies result/phase bytes and reservation identities, refuses failed/unknown/over-budget panels, then recomputes the existing family-bootstrap/harm/Elo readout. Context utilities must reproduce each family mean. |
| Software evidence | CPU tests cover both update paths, official unrevealed selection and equal-family weighting, full fake-worker coordinator/collector success, failure retention, identity/tampering/budget refusal, source admission, CPU gate and owned-descendant cleanup. Synthetic results stay inside pytest fixtures. |

The previously accepted missing-preregistration receipt and unreachable worker
CLI have reproducing failures against the original source. These are software
controls; no actual H122 learning or CUDA qualification is asserted. The tracked
`metrics.json` remains the unexecuted study stub, not a launch certificate.

## Future operator handoff

CPU readiness does not authorize launch. After this source is reviewed and merged,
use a clean managed checkout whose HEAD is on fetched `origin/main`. Preserve
the preregistered H99 → H100 → H105 → H109 → H121 → H122 order. Obtain the
separate owner launch grant and verify each predecessor is terminal and has
released its qualification lease. Keep an immutable operator closeout JSON per
predecessor with these fields (the outcome may be positive or negative):

```json
{"wave":"H121","closed":true,"qualification_lock_released":true,"in_progress":null}
```

These are explicit operator attestations, not inferred scientific passes or
fabricated by the coordinator. Supply all five real paths; each is retained by
path and SHA-256 in the budget and revalidated at collection. A missing or open
predecessor prevents even creation of the H122 run root or FIFO ticket.

The following is a later, authorized launch example; **do not run it as part of
CPU preparation**. Use the canonical root on its first invocation. The CLI
refuses alternate roots, and an existing canonical root cannot be overwritten or
reset. Resource-wait time is retained on failure as well as success.

```sh
nice -n 19 env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  python3 research/flywheel_selfplay/h122_lookahead_retest/run_wave.py --execute \
  --root "$HOME/jope-data/h122/h122-wave" \
  --predecessor H99=/path/to/H99-closeout.json \
  --predecessor H100=/path/to/H100-closeout.json \
  --predecessor H105=/path/to/H105-closeout.json \
  --predecessor H109=/path/to/H109-closeout.json \
  --predecessor H121=/path/to/H121-closeout.json
```

The worker accepts only the original H50 input and offline r882 copy with their
pinned identities. It uses selected DL-Train members only; no DL-Val, sealed,
holdout, meta-test, outer-test or H93 result payload is part of execution. Never
use xbabe0 or the live xbabe1 loop. A failed run keeps its budget, logs, scratch,
phases and `incomplete.json`; no automatic retry or receipt deletion occurs.

After a successful panel, a read-only replay emits JSON to stdout:

```sh
python3 research/flywheel_selfplay/h122_lookahead_retest/run_wave.py \
  --collect --root "$HOME/jope-data/h122/h122-wave"
```

Remaining operational acceptance: actual authorized predecessor closeouts,
resource admission, inherited kernel lease, real CUDA execution, complete
six-cell result replay and empirical H122 verdict. These depend on runtime
evidence; the CPU tests do not satisfy them. No GPU job or waiting process is
started by this PR.
