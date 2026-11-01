# Terminal reference

The `jopedime` executable provides local model commands and commands that communicate with the shared background engine. Use `jopedime --help` to see the commands in the installed build.

## Interactive screen

Launching without arguments in a terminal opens the full-screen **JAIN by VEOX** landing view, with a large ASCII wordmark and **Powered by JopeDime**. Submit a command to start the transcript; the composer stays near the bottom.

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
