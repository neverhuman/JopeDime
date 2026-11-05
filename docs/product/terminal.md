# Terminal reference

The `jopedime` executable provides local model commands and commands that communicate with the shared background engine. Use `jopedime --help` to see the commands in the installed build.

## Interactive screen

Launching without arguments in a terminal opens the full-screen **jAIn by VEOX** landing view with an animated JAIN wordmark (`JAIN_NO_ANIMATION=1` shows the finished logo immediately). Submit a command to start the transcript; the composer stays near the bottom.

### First run

The first launch asks three questions in a dialog (or as prompts in line mode):

1. **Terms of Service** — read https://veox.ai/terms (version `2026-11-03`); Enter or `y` accepts, `q` or `n` exits without changing anything. A new terms version asks again.
2. **Usage statistics** — on by default; `n` turns them off. This is the same setting as `jopedime cloud telemetry on|off`. `DO_NOT_TRACK=1` or `JAIN_TELEMETRY=0` forces it off and skips the question.
3. **Log in to your free veox Cloud Workspace** or create one (starts the browser device login), or **continue with the bundled demos**.

Choices are stored in `<JOPEDIME_HOME>/state.json` (owner-only). Agents and scripts accept the terms without a prompt with `--accept-terms` (or `JAIN_ACCEPT_TERMS=1`) and choose usage statistics with `--telemetry on|off` (or `JAIN_TELEMETRY`; `DO_NOT_TRACK=1` still forces off), after showing the user https://veox.ai/terms; the dialog is then skipped. Both flags are global: before or after the subcommand.

### Demos

`/demo` lists the datasets bundled in the executable; `/demo NAME` (for example `/demo regression`) opens one with its target and task preset and starts a fast run. `/open example/<name>.csv` opens a bundled demo as if it were a file (a real file at that path takes precedence), and Tab completes `example/` names and `/demo` names. Signed out, the interactive session opens only the bundled demos; your own files need a free veox Cloud Workspace account (`/login`). `JAIN_ALLOW_LOCAL_FILES=1` lifts that for offline and CI use.

### Opening your own files

`/open` with no argument lists the CSV files and folders in the folder `jopedime` was started from, or, when there are none, names that folder and how to add a file (copy a `.csv` there, or start from the folder that holds your data). Tab after `/open ` lists that folder live: the bundled `example/` demos, then folders, then `.csv` files. Up/Down move through the list, Enter on a folder opens its listing, Enter on a file opens it, Esc closes the list. Signed out, `/open` opens only the bundled demos.

### Transcript

Each command you submit is shown as a highlighted `❯` row with a blank line before it; jAIn's reply is joined to it with `⎿`. Reasoning from the search (`Thinking` events) is set apart with `✻` and a `┊` rail, `✓` marks a completed step and `✗` an error. Text wraps at word boundaries; PgUp/PgDn and the mouse wheel scroll the history, a scrollbar shows the position, and End returns to live output.

### Usage

`/usage` shows the signed-in account and its plan (read from the workspace in the background), the current run, and run statistics recorded by this machine's engine for the last 7 days and in total (runs by outcome, evaluations and compute time), plus the usage-statistics and terms settings.

### Links

URLs printed by the jAIn CLI (login, terms, cloud runs, help) are always shown in full and are never split across lines when they fit. In terminals that support OSC 8 hyperlinks (iTerm2, WezTerm, kitty, VTE/GNOME Terminal, Windows Terminal, VS Code) they are also clickable; `JAIN_HYPERLINKS=1` forces links on and `JAIN_HYPERLINKS=0` turns them off. Only veox-hosted (`*.veox.ai`) and loopback URLs become links, and a link's target is always its visible text; escape sequences in data are always removed.

Use **Page Up / Page Down** or the mouse wheel to scroll results. While reading history, new output keeps your position and increments a new-lines indicator; **End** returns to live output. Scrolling does not change your pending input. The view retains up to 2,000 lines / 1 MiB; durable run results remain in the engine.

`NO_COLOR` keeps the full-screen interface in monochrome. `TERM=dumb`, width below 40 columns, or height below 12 rows uses plain line mode. Resize fallback retains the transcript and draft. Exiting restores the previous terminal screen. Use `/login` to connect the selected VEOX workspace.

## Engine status

```console
$ jopedime serve --status
stopped
```

The command exits with status 0 when it reads the engine state. In this example the engine was stopped and `JOPEDIME_HOME` pointed to a new isolated directory.

## Runs

The current CLI exposes `run`, `status`, and `events`. The captured help shows that `run` requires `--dataset`, `--target`, `--task`, `--split`, `--objective`, `--effort`, `--budget`, `--landmarks`, `--guidance`, `--device`, and `--seed`. `status` and `events` take a run identifier.

The development build supports the shared-engine regression workflow used by the connected CLI acceptance test, including planning, starting, status, control and events. Capabilities still depend on the selected task, guidance, landmarks and export codec; an exposed command does not establish every backend or release qualification.

Dataset upload is available with `jopedime dataset add`; see [Quickstart](quickstart.md). For the bounded development regression path, configure `/landmarks none`, `/lang rust`, `/target target --task regression`, `/open data.csv`, `/effort fast --evals 1 --time 20s`, then `/run`.

## Exit statuses

The executable's top-level help lists these exit statuses: 0 for success, 2 for invalid input, 3 for integrity or compatibility errors, 4 for unavailable backends, 5 for a time budget exhausted, 6 for numerical failure, and 7 for output failure.

## Leaving an interactive session

Press **Ctrl-C** to exit immediately and return to your shell on Linux or macOS, including while editing a draft or searching history. On macOS this is Control-C; Command-C remains the terminal’s copy shortcut. **Ctrl-D** exits when the input is empty; `\quit` also exits. The CLI restores terminal input and cursor visibility when it leaves.

An active run continues in the local background engine. Reconnect with `jopedime attach RUN`; use explicit `\pause` or `\cancel` commands to control it. Ctrl-C does not send either command.

## Connect to VEOX

Run `/login` in the interactive CLI, or `jain-dev auth login --workspace https://dev-xbabe0.veox.ai`. The default workspace is your saved choice, `JAIN_WORKSPACE`, or https://app.veox.ai. The CLI prints a short pairing code and opens the browser; over SSH, open that URL on your own machine. Sign in normally, check that the code matches, and approve the CLI. No password or browser cookie is copied into the terminal. `/logout` revokes this CLI credential and removes its local copy; `auth status` shows the selected account.

Device credentials grant account reads and connected-run/model publication, expire after 30 days, and can be revoked from the account page. Ordinary cloud-training and dataset-deletion writes remain forbidden. The CLI stores its credential in a private 0600 file inside a 0700 directory, bound to the selected account and HTTPS workspace origin. Automation may use `auth login --token-file PRIVATE_FILE` or redirected `--token-stdin`; token input is not accepted through an interactive echoed prompt.

New runs created in a signed-in interactive CLI synchronize status and metrics. Existing/attached runs are shared only after `/sync [RUN]`; use `jain-dev cloud sync --run RUN` outside the interface. Metadata queues survive an offline failure and retry without creating duplicate cloud runs. Dataset files and raw rows are not part of metadata sync. A run already associated with another account/workspace is refused rather than silently republished.

Left Arrow from empty input opens the runs panel. Left continues ordinary editing when text is present. The panel traverses local and account-visible cloud pages, shows status separately from Local only / Cloud only / Local + cloud, and includes the real workspace URL. Up/Down selects, Enter attaches a local run or opens a cloud-only run, `o` opens the cloud URL, `d` downloads a model, `r` refreshes, and Esc/Right returns to the composer. A metadata cache does not imply the model is downloaded.

`/publish [RUN]` explicitly exports and publishes the selected fitted model; `cloud publish RUN --model OBJECT_DIRECTORY` can choose an existing portable object. Fitted state can retain training rows, so publishing a model shares that state even though no dataset file is uploaded. Published objects are immutable per connected run in this development version; identical retries succeed and different replacements are refused. Limits are 32 MiB per model and 512 MiB of connected models per account.

`/fetch CLOUD_UUID` or `jain-dev cloud fetch CLOUD_UUID` downloads and verifies the model before installing it in the active account's local model store. It prints the model path and whether this build can execute its codec. Use `jopedime model verify --model PATH --require-executable` before local prediction. Connected-run status and metrics are client reported. Cloud scoring is not yet available for these newly published connected artifacts; existing supported cloud-trained runs retain their browser prediction workflow.

The development gateway is separate from VEOX account login. For that environment only, configure a private gateway JSON file with `origin`, `username`, and `password`, and pass `--gateway-auth-file PATH` or `JAIN_GATEWAY_AUTH_FILE`. Gateway credentials are refused for a different origin. The application token remains distinct and uses `X-Veox-Api-Token` behind Basic Auth. Never place account or gateway secrets in a URL or command argument.

## Local or cloud

`/run` decides where a run goes and says so in one line (`crate::cloud::route`):

- `/run --local` and `/run --cloud` decide outright. Bundled demos always run here and need no login.
- A Pro account's deep search (effort `medium`, `high` or `ultra`) runs in the veox Cloud Workspace.
- Otherwise the run is local, unless the table has more than 100,000 rows (Free's cloud ceiling), when
  the cloud is offered. A Free account is told that its cloud runs take 100,000 rows too.

The public build is CPU-only, so nothing probes for a GPU. A cloud run that needs a login says
"This run needs your veox Cloud Workspace — opening your browser to log in…", runs the device login and
then continues the same run. The upload is asked once per dataset (remembered per file hash, account and
workspace in `state.json`): "Upload <file> (<rows> rows, <size>) to your veox Cloud Workspace? [Y/n]".
Progress arrives as the same "Thinking" cards as a local run, then the closing narrative and the run's
workspace link. Tier limits read in plain words with "Run locally instead? [Y/n]" and the upgrade link.

`jopedime run CSV` follows the same rule but uploads only with `--cloud` (plus `--consent-upload`, or an
answer on a terminal, or a remembered consent); when the rule would use the cloud it runs locally and says
on stderr how to send it. The local MCP server always runs locally and adds a routing text line to
`run_start`: an agent cannot give the user's upload consent.

## Results sync

After a local run finishes while you are logged in, its fitted model, a PDF report and the generated
code go to your workspace, so it can be explored there like a cloud run. `jopedime cloud runs` lists every run, local and cloud (`--json`, `--local`, `--cloud`, `--unsynced`, `--limit N`). The first time, jAIn asks
"Sync results (model + report, not your data) to your workspace? [Y/n]"; `/sync on|off` (or
`jopedime cloud results on|off`) changes the answer. No training rows are uploaded, except that kernel,
smoother and Gaussian-process models keep the support rows they predict from inside the fitted model.

## Device id and update check

Workspace requests carry `X-Jain-Device`, a 32-hex HMAC-SHA256 pseudonym of the OS machine id (never the
id itself), unless telemetry is off. Once a day the terminal (and `--version` / `doctor` at a terminal)
reads the installer's signed stable channel pointer, verified with the pinned release key, and shows
"jAIn <v> is available — update: curl -fsSL https://veox.ai/install.sh | sh". It never installs.
`JAIN_NO_UPDATE_CHECK=1` or `DO_NOT_TRACK=1` turns it off; `JAIN_REGISTRY` selects an HTTPS mirror.

## The shared engine

Engine commands (`status`, `events`, `predict --model RUN`, `runs`, `export`, `code`, `control`,
`restart`, …) start the shared background engine on demand. An engine from another jAIn version is
restarted when idle ("Restarted the background engine: it was jAIn 1.0.1, now 1.0.0") and left alone,
with an explanation, while it runs a job. `jopedime serve --stop` says what a busy engine is running;
`--stop --force` stops it anyway. A directory written by `run --out` or `export --out` loads with
`predict --model DIR` and `model verify --model DIR`.
