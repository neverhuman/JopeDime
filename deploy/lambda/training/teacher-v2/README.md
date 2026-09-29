# Airfoil semantic-island Lambda campaign

Read `SCHEMA.md` before operating this campaign. The safe sequence is:

```bash
python3 analysis/materialize_control_semantics.py --parent-runtime ~/scratch/jopedime-test-adaptive-gp-v1-20260929 --output ~/scratch/jopedime-test-adaptive-gp-islands-v2-20260929/control-analysis --workers 96
python3 analysis/test_engine_protocol.py
python3 analysis/test_score_parity.py
python3 transport/evolve.py --qualify 16
ulimit -n 65535
python3 transport/evolve.py --run --generations 100
```

Use `python3 transport/evolve.py --status` only when the campaign process is stopped; it starts a
fresh deterministic engine and writes an orchestrator instance receipt. While running, read
`~/scratch/jopedime-test-adaptive-gp-islands-v2-20260929/status.json` or use the TUI.

Create `STOP` in the runtime directory for a draining stop. The coordinator stops admitting work,
records every in-flight return, writes the status, and resets Lambda reserved concurrency to zero.
