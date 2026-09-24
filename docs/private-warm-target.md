# Warm demonstration latency target

The warm ASM invocation targets less than 60 seconds. Its original declared resource budget
may exceed that target so that a real fitted demonstration can finish and record a measured
miss. `execution_complete`/`ok` describe authenticated neural consumption, bounded refinement, both
fitted outputs, predictions and an unclipped matched comparison. `target_met` and the retained
`under_60_seconds` field are true only when that execution succeeds in less than 60 seconds.
A successful execution with `target_met: false` does not pass the latency target.

The existing 60-second example with 20 seconds per arm is unqualified. In the retained
3fd7d5b2 proof the neural arm began 0.415 seconds after the original window started; policy
execution began at 22.129 seconds, took about 20.8 seconds, and the first fit began at 50.384
seconds. The pre-policy work occurred inside the arm. Those measurements do not establish
that a 20-second arm can consume a neural proposal or complete either fitted demonstration.

For an honest longer bounded demonstration, explicitly preregister a resource total such as
300 seconds with five seconds reporting reserve and the unchanged 120 seconds per arm. This
is a configuration example, not a promise that all required outputs finish within it. Keep
20 CPUs/workers, 64 GiB, zero swap/GPU, one thread per worker and 30 seconds per fit. Preserve
both full equal arm allowances inside the total minus reserve. Final reports retain actual
available time for each arm; clipped or missing arms still fail matched-comparison acceptance.
No default panel/proof budget, candidate count, original deadline or resume rule changes.

Record cold installation separately. Publish measured warm duration and both execution and
latency outcomes, retaining failed attempts and their receipts. A generic-only fixture cannot
stand in for this neural demonstration.
