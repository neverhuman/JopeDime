# H122: Lookahead-only retest

**CPU executor and collection implemented; the H122 study has not run.** H93's Lookahead arm never reached scoring, so H122 preregisters a fresh CONTROL/LA comparison using seeds 48–50 and a new whole-family DL-Train assessment fold. H93 outcomes and seeds are excluded.

The [preregistration](PREREG.md) fixes the 122042 family split, six-cell 40-update protocol, H61 official argmin utility, primary +0.001 family-bootstrap LCB gate, 10% harm ceiling, Elo readout, H50 input identity, and later FIFO order. The runner's CPU dry-run uses only synthetic family metadata and update-plan fixtures. Its read-only scratch regression uses a tiny synthetic checkpoint under pytest's fixture directory; it preserves the pre-existing file and mode. Neither is experiment evidence.

The metrics stub reports `CPU_PREP_ONLY`, zero outcome cells and null scientific readouts. H122 remains not admitted / not assessed until a separately authorized, complete future panel is run after H121 in FIFO order.

The [launch-readiness audit and operating instructions](EXECUTION_READINESS.md)
distinguish tested CPU software from actual CUDA execution. `run_wave.py` now
coordinates the existing cell worker and collects only the six reserved,
hash-bound complete results. CPU fixtures exercise real optimizer steps on tiny
tensors; they are not H122 updates or experimental evidence. The original
preregistration, 40-update CONTROL/LA recipe and shared scientific pins are
unchanged. No ticket, qualification lock, real input/checkpoint or CUDA operation
is needed for either command:

```sh
python3 research/flywheel_selfplay/h122_lookahead_retest/runner.py --cpu-dry-run
python3 research/flywheel_selfplay/h122_lookahead_retest/run_wave.py --cpu-dry-run
```
