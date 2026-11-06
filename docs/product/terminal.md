# Terminal reference

The jAIn CLI is installed as the `jain` command. It provides local model commands and commands that communicate with the shared background engine; `jain --help` lists the commands in the installed build.

Names: everything people read says jAIn / `jain` (`jain --version` prints `jain X.Y.Z`, errors start with `jain:`). The executable is still built as `jopedime`, and machine-readable names stay stable: `version --json` keeps `"product": "jopedime"` (release, qualification and installer checks match it) and adds `"display_name": "jAIn"` and `"command": "jain"`; the unrenamed release artifact (`jopedime-…`) still prints `jopedime X.Y.Z` for the release checks; the data directory stays `~/.local/share/jopedime` (`JOPEDIME_HOME`), because moving it would orphan existing runs, logins and settings.

Without a terminal (an agent, a script, a pipe), `jain` with no arguments says that the TUI needs a terminal, names the commands that work without one, and prints the help. The bundled demos are reachable without a terminal:

```console
$ jain demo list                                  # or --json
$ jain demo run regression_parkinsons --json      # the same run as /demo regression_parkinsons
$ jain run example/regression_parkinsons.csv --target target   # bundled demos run locally; your own data runs in the veox Cloud (jain auth login, then jain run FILE.csv --target COLUMN --cloud; it asks before uploading)
$ jain fit --train example/regression_parkinsons.csv --target target --output model.jope
```

`example/<name>.csv` is written once to `<JOPEDIME_HOME>/examples/` when used (a real file at that path takes precedence). `jain run --dataset ID` infers the task from the stored profile and defaults the export languages to Rust and Python, like a CSV run; a refused start names the flag to change.

## Interactive screen

Launching without arguments in a terminal opens the full-screen **jAIn by VEOX** landing view with an animated JAIN wordmark (`JAIN_NO_ANIMATION=1` shows the finished logo immediately). Submit a command to start the transcript; the composer stays near the bottom.

### First run

The first launch asks three questions in a dialog (or as prompts in line mode):

1. **Terms of Service** — read https://veox.ai/terms (version `2026-11-03`); Enter or `y` accepts, `q` or `n` exits without changing anything. A new terms version asks again.
2. **Usage statistics** — on by default (Enter shares; `n` turns them off). They are counts, bucketed sizes and scores rounded to tenths; never dataset rows, column names or file paths — the same words `jain cloud telemetry` uses. This is the same setting as `jain cloud telemetry on|off`. `DO_NOT_TRACK=1` or `JAIN_TELEMETRY=0` forces it off and skips the question.
3. **Log in to your free veox Cloud Workspace** or create one (starts the browser device login), or **continue with the bundled demos**.

Choices are stored in `<JOPEDIME_HOME>/state.json` (owner-only). Agents and scripts accept the terms without a prompt with `--accept-terms` (or `JAIN_ACCEPT_TERMS=1`) and choose usage statistics with `--telemetry on|off` (or `JAIN_TELEMETRY`; `DO_NOT_TRACK=1` still forces off), after showing the user https://veox.ai/terms; the dialog is then skipped. Both flags are global: before or after the subcommand.

### Demos

`/demo` lists the datasets bundled in the executable; `/demo NAME` (for example `/demo regression`) opens one with its target and task preset and starts a fast run. `/open example/<name>.csv` opens a bundled demo as if it were a file (a real file at that path takes precedence), and Tab completes `example/` names and `/demo` names. Signed out, the interactive session opens only the bundled demos; your own files need a free veox Cloud Workspace account (`/login`). `JAIN_ALLOW_LOCAL_FILES=1` lifts that for offline and CI use.

### Opening your own files

`/open PATH` accepts `~/…` for your home directory; after it opens, jAIn names the suggested target column and says "Type /run to start" (`/auto` runs the default search on it). `/open` with no argument lists the CSV files and folders in the folder `jain` was started from, or, when there are none, names that folder and how to add a file (copy a `.csv` there, or start from the folder that holds your data). Tab after `/open ` lists that folder live: the bundled `example/` demos, then folders, then `.csv` files. Up/Down move through the list, Enter on a folder opens its listing, Enter on a file opens it, Esc closes the list. Signed out, `/open` opens only the bundled demos.

### Transcript

Each command you submit is shown as a highlighted `❯` row with a blank line before it; jAIn's reply is joined to it with `⎿`. Reasoning from the search (`Thinking` events) is set apart with `✻` and a `┊` rail, `✓` marks a completed step and `✗` an error. Text wraps at word boundaries; PgUp/PgDn scroll the history (and the mouse wheel when mouse scrolling is on), a scrollbar shows the position, and End returns to live output. `/help` lists every command one per line; `/exit` is the same as `/quit`. After a run finishes the status bar shows `completed` and how it ended (`finished · time budget reached`), a run stopped by its time budget says "Stopped at the time budget (10 s) after 15 of 128 candidates." (and how long finishing took when it ran past the budget), and one "Next:" line names predicting, the code export, your own data and `/login`. Attaching a run already shown does not print its lines again. On quit jAIn says that the background engine stays up for about 30 minutes after its last job and that `jain serve --stop` stops it.

### Selecting and copying text

Everything the TUI shows can be selected and copied:

- **The mouse is not captured by default**, so your terminal's own selection (drag, double-click, copy) works on any text on screen. **F2** (or `/mouse on|off`) switches the mouse wheel to scrolling; while it is on, most terminals still select with **Shift-drag**. The footer shows the current mode (`F2 mouse: select text` or `F2 mouse: scroll (Shift-drag selects)`). `JAIN_MOUSE=1` starts with wheel scrolling on.
- **`/copy`** puts text on the clipboard through the terminal (OSC 52): `/copy` the output since your previous command, `/copy link` the last URL shown, `/copy run` the current run id, `/copy result` the last run result, `/copy all` the whole transcript. Links and run ids are also printed again on a line of their own, so they can be selected where the terminal ignores OSC 52 (tmux needs `set -g set-clipboard on`; xterm needs `allowWindowOps`).
- **`/transcript`** (or `/select`) shows the whole session as plain text on the normal screen, in your terminal's scrollback, with every line and URL whole (nothing cut at the screen edge); Enter returns to the TUI.

### Usage

`/usage` shows the signed-in account and its plan (read from the workspace in the background), the current run, and run statistics recorded by this machine's engine for the last 7 days and in total (runs by outcome, evaluations and compute time), plus the usage-statistics and terms settings.

### Links

URLs printed by the jAIn CLI (login, terms, cloud runs, help) are always shown in full and are never split across lines when they fit. They are also OSC 8 hyperlinks by default, clickable in terminals that support them (iTerm2, WezTerm, kitty, VTE/GNOME Terminal, Windows Terminal, VS Code, recent xterm and tmux); others ignore the sequence and show the same plain URL. Links are off for `TERM=dumb`, the Linux console and VTE before 0.50; `JAIN_HYPERLINKS=1` forces links on and `JAIN_HYPERLINKS=0` turns them off. `/login` opens your browser when one can show (not over SSH without a forwarded display, not on a machine without a graphical session, never with `JAIN_NO_BROWSER=1`; `BROWSER` is always tried) and otherwise says to open the link on any device. Only veox-hosted (`*.veox.ai`) and loopback URLs become links, and a link's target is always its visible text; escape sequences in data are always removed.

Use **Page Up / Page Down** (or the mouse wheel with F2 on) to scroll results. While reading history, new output keeps your position and increments a new-lines indicator; **End** returns to live output. Scrolling does not change your pending input. The view retains up to 2,000 lines / 1 MiB; durable run results remain in the engine.

`NO_COLOR` keeps the full-screen interface in monochrome. `TERM=dumb`, width below 40 columns, or height below 12 rows uses plain line mode. Resize fallback retains the transcript and draft. Exiting restores the previous terminal screen. Use `/login` to connect the selected VEOX workspace.

## Engine status

```console
$ jain serve --status
stopped
```

The command exits with status 0 when it reads the engine state. In this example the engine was stopped and `JOPEDIME_HOME` pointed to a new isolated directory.

## Runs

The current CLI exposes `run`, `status`, and `events`. The captured help shows that `run` requires `--dataset`, `--target`, `--task`, `--split`, `--objective`, `--effort`, `--budget`, `--landmarks`, `--guidance`, `--device`, and `--seed`. `status` and `events` take a run identifier.

The development build supports the shared-engine regression workflow used by the connected CLI acceptance test, including planning, starting, status, control and events. Capabilities still depend on the selected task, guidance, landmarks and export codec; an exposed command does not establish every backend or release qualification.

Dataset upload is available with `jain dataset add`; see [Quickstart](quickstart.md). For the bounded development regression path, configure `/landmarks none`, `/lang rust`, `/target target --task regression`, `/open data.csv`, `/effort fast --evals 1 --time 20s`, then `/run`.

## Exit statuses

The executable's top-level help lists these exit statuses: 0 for success, 2 for invalid input, 3 for integrity or compatibility errors, 4 for unavailable backends, 5 for a time budget exhausted, 6 for numerical failure, and 7 for output failure.

## Leaving an interactive session

Press **Ctrl-C** to exit immediately and return to your shell on Linux or macOS, including while editing a draft or searching history. On macOS this is Control-C; Command-C remains the terminal’s copy shortcut. **Ctrl-D** exits when the input is empty; `\quit` also exits. The CLI restores terminal input and cursor visibility when it leaves.

An active run continues in the local background engine. Reconnect with `jain attach RUN`; use explicit `\pause` or `\cancel` commands to control it. Ctrl-C does not send either command.

## Connect to VEOX

Run `/login` in the interactive CLI, or `jain-dev auth login --workspace https://dev-xbabe0.veox.ai`. The default workspace is your saved choice, `JAIN_WORKSPACE`, or https://app.veox.ai. The CLI prints a short pairing code and opens the browser; over SSH, open that URL on your own machine. Sign in normally, check that the code matches, and approve the CLI. No password or browser cookie is copied into the terminal. `/logout` revokes this CLI credential and removes its local copy; `auth status` shows the selected account.

Device credentials grant account reads and connected-run/model publication, expire after 30 days, and can be revoked from the account page. Ordinary cloud-training and dataset-deletion writes remain forbidden. The CLI stores its credential in a private 0600 file inside a 0700 directory, bound to the selected account and HTTPS workspace origin. Automation may use `auth login --token-file PRIVATE_FILE` or redirected `--token-stdin`; token input is not accepted through an interactive echoed prompt.

New runs created in a signed-in interactive CLI synchronize status and metrics. Existing/attached runs are shared only after `/sync [RUN]`; use `jain-dev cloud sync --run RUN` outside the interface. Metadata queues survive an offline failure and retry without creating duplicate cloud runs. Dataset files and raw rows are not part of metadata sync. A run already associated with another account/workspace is refused rather than silently republished.

Left Arrow from empty input opens the runs panel. Left continues ordinary editing when text is present. The panel traverses local and account-visible cloud pages, shows status separately from Local only / Cloud only / Local + cloud, and includes the real workspace URL. Up/Down selects, Enter attaches a local run or opens a cloud-only run, `o` opens the cloud URL, `d` downloads a model, `r` refreshes, and Esc/Right returns to the composer. A metadata cache does not imply the model is downloaded.

`/publish [RUN]` explicitly exports and publishes the selected fitted model; `cloud publish RUN --model OBJECT_DIRECTORY` can choose an existing portable object. Fitted state can retain training rows, so publishing a model shares that state even though no dataset file is uploaded. Published objects are immutable per connected run in this development version; identical retries succeed and different replacements are refused. Limits are 32 MiB per model and 512 MiB of connected models per account.

`/fetch CLOUD_UUID` or `jain-dev cloud fetch CLOUD_UUID` downloads and verifies the model before installing it in the active account's local model store. It prints the model path and whether this build can execute its codec. Use `jain model verify --model PATH --require-executable` before local prediction. Connected-run status and metrics are client reported. Cloud scoring is not yet available for these newly published connected artifacts; existing supported cloud-trained runs retain their browser prediction workflow.

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

`jain run CSV` follows the same rule but uploads only with `--cloud` (plus `--consent-upload`, or an
answer on a terminal, or a remembered consent); when the rule would use the cloud it runs locally and says
on stderr how to send it. The local MCP server always runs locally and adds a routing text line to
`run_start`: an agent cannot give the user's upload consent.

## Results sync

After a local run finishes while you are logged in, its fitted model and the generated code go to
your workspace (its events carry the target, task, metric and data profile; the report PDF waits for an
engine that writes one), so it can be explored there like a cloud run. `jain cloud runs` lists every run, local and cloud (`--json`, `--local`, `--cloud`, `--unsynced`, `--limit N`). The first time, jAIn asks
"Sync results (model + code, not your data) to your workspace, for this run and every later local run? … [Y/n]": the answer is a standing setting, announced as such ("Results sync on for this and later local runs · turn off with /sync off"), and `/sync on|off` (or
`jain cloud results on|off`) changes it. While a run is in progress only its status (progress and scores) is synced, shown once as "Status synced for RUN (progress and scores; results only with your consent)"; results go only after the answer. `jain cloud publish RUN` exports a finished run's model and code itself and sends both. A run's workspace title is its dataset's file or display name, also for `jain run --dataset ID`. No training rows are uploaded, except that kernel,
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
with an explanation, while it runs a job. Release candidates share one version number, so the engine's
build (the source commit recorded in `daemon.json`) is compared too: an engine from another build of the
same version, or one that predates build identities, is treated the same way ("it was jAIn 1.0.0 (build
51c0c1a), now 1.0.0 (build 2ab7f55)"). `jain serve --stop` says what a busy engine is running;
`--stop --force` stops it anyway. A directory written by `run --out` or `export --out` loads with
`predict --model DIR` and `model verify --model DIR`.
