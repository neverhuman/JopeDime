# MCP-01 conformance spike

Two isolated, disposable stdio servers with the same `echo` and `slow_echo`
tools. Nothing is added to the product Cargo workspace.

Read [the design](DESIGN.md), [the report](REPORT.md) and
[the downstream LUNA cards](FOLLOW_UPS.md). OWNER ruled OD9 as `rmcp 3.8.0` in
the Jepson-approved t526u scope nudge. ENG:watcher publishes the local commit from
its bundle; this worker does not push. This experiment does not close product gate A4.

From the repository root:

```bash
bash research/mcp-spike/run.sh
python3 research/mcp-spike/check-dependencies.py
python3 research/mcp-spike/check-local.py
```

The first command uses the lane's existing `target/`, eight Cargo jobs, disabled
incremental compilation, line-table debug information and niceness 10. It refuses
to start above the host load limit. The replay receipt is
`evidence/local/replay.json`; matching `.jsonl` records contain the raw wire.
For the retained experiment see `evidence/replay.json` and `.jsonl`.

The check runner executes **all** card commands and records their exit statuses.
Its own zero exit means receipt collection completed, even when individual checks
failed; inspect `evidence/checks.json`. Similarly, the conformance driver's zero
exit means both candidates were measured, not that both conform. Each protocol
version and case has its own PASS/FAIL. Malformed-input probes are diagnostics:
they deliberately violate the stdio client's valid-message requirement.

The task card's banned-dependency shell pipeline masks failures. The strict
dependency scanner checks Cargo's exit status and exact crate names separately;
its zero exit requires all three measured closures to pass.

Tool inputs are bounded, synthetic protocol fixtures. No scientific dataset,
learning worker, DOPE weights, GPU, sealed family or outer fold is accessed.
