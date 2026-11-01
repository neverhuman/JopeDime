# Terminal reference

The `jopedime` executable provides local model commands and commands that communicate with the shared background engine. Use `jopedime --help` to see the commands in the installed build.

## Interactive screen

Launching without arguments in a terminal opens the full-screen **JAIN by VEOX** landing view, with a large ASCII wordmark and **Powered by JopeDime**. Submit a command to start the transcript; the composer stays near the bottom.

Use **Page Up / Page Down** or the mouse wheel to scroll results. While reading history, new output keeps your position and increments a new-lines indicator; **End** returns to live output. Scrolling does not change your pending input. The view retains up to 2,000 lines / 1 MiB; durable run results remain in the engine.

`NO_COLOR` keeps the full-screen interface in monochrome. `TERM=dumb`, width below 40 columns, or height below 12 rows uses plain line mode. Resize fallback retains the transcript and draft. Exiting restores the previous terminal screen. Cloud account sync is not connected in this build.

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

## Leaving an interactive session

Press **Ctrl-C** to exit immediately and return to your shell on Linux or macOS, including while editing a draft or searching history. On macOS this is Control-C; Command-C remains the terminal’s copy shortcut. **Ctrl-D** exits when the input is empty; `\quit` also exits. The CLI restores terminal input and cursor visibility when it leaves.

An active run continues in the local background engine. Reconnect with `jopedime attach RUN`; use explicit `\pause` or `\cancel` commands to control it. Ctrl-C does not send either command.
