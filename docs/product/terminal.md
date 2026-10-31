# Terminal reference

The `jopedime` executable provides local model commands and commands that communicate with the shared background engine. Use `jopedime --help` to see the commands in the installed build.

## Engine status

```console
$ jopedime serve --status
stopped
```

The command exits with status 0 when it reads the engine state. In this example the engine was stopped and `JOPEDIME_HOME` pointed to a new isolated directory.

## Runs

The current CLI exposes `run`, `status`, and `events`. The captured help shows that `run` requires `--dataset`, `--target`, `--task`, `--split`, `--objective`, `--effort`, `--budget`, `--landmarks`, `--guidance`, `--device`, and `--seed`. `status` and `events` take a run identifier.

The service implements `system_info` and dataset registration. Dataset listing/inspection and run planning, starting, status, control, events, and algorithm lookup still return `CAPABILITY_UNAVAILABLE`. The CLI command names therefore describe available syntax; they do not imply that the corresponding engine operation is wired in this build.

Dataset upload is executable with `jopedime dataset add`; see [Quickstart](quickstart.md). The run, status, and events workflow remains pending until its service handlers are available.

## Exit statuses

The executable's top-level help lists these exit statuses: 0 for success, 2 for invalid input, 3 for integrity or compatibility errors, 4 for unavailable backends, 5 for a time budget exhausted, 6 for numerical failure, and 7 for output failure.
