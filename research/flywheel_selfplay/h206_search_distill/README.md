# H206 search distillation

**QUEUED / NOT RUN.** This is the preregistration and CPU-only runner/launch
skeleton requested by n2022. It contains no executable search, training, GPU
probe or sealed evaluator. Metrics remain unmeasured.

The fixed protocol is in [PREREG.md](PREREG.md): k=16/64/256 Rust-CV teachers,
a k=1 self-distillation control, three expert iterations, a terminal k=1 student,
four-arm pooled selection and separate sanctioned sealed confirmation.

Run `nice -n 19 bash research/flywheel_selfplay/h206_search_distill/launch.sh --plan`
to print the queued plan. `--output /path/to/new/launch.json` creates an exclusive
CPU preparation record with the authority string and research run ID.
`--execute` always refuses. A successful plan print is not resource admission.

Future work before launch: reviewed/pinned width128 architecture and r882
initialization, full Rust candidate/search/distillation executor, immutable
real-data/fold/source bindings, native91 parity, runtime gradient/resource
qualification, and the later H206 queue-release nudge. xbabe3 needs <=88% used
before reserving the40GB cache and >=50GB headroom; post-reservation used must
stay<93.8%, with the RTX3090 unclaimed. No foreign cleanup is authorized.

No efficacy, completed optimizer updates, scientific gate PASS or serving
promotion is claimed. Current manifest evidence is empty because no experiment
has run or uploaded artifacts.
