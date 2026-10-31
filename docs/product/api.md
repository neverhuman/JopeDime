# Local API and run commands

The command line includes `api` and `mcp` stdio modes, plus engine-backed dataset commands. Their exact arguments can vary by build; inspect the installed executable with `jopedime api --help`, `jopedime mcp --help`, or `jopedime --help` before configuring a client.

The CLI lists `add`, `list`, `show`, and `delete`. Only `add` is wired to a service operation in this build; the other dataset operations still return `CAPABILITY_UNAVAILABLE`.

## HTTP routes in the current service build

The local engine serves the router on IPv4 loopback. The system and registration examples below were executed against a locally built engine with an isolated `JOPEDIME_HOME`. `GET /v1/system` returns HTTP 200, and CSV registration through `POST /v1/datasets` returns HTTP 201. The store-bound engine also executes the bounded CPU regression path below. Other unavailable operations return HTTP 422 with `CAPABILITY_UNAVAILABLE`.

The successful system snapshot is:

```json
{
  "version": "0.4.1",
  "api_version": "1.0",
  "health": "ok",
  "local_paths": true,
  "capabilities": {
    "tasks": [],
    "languages": [],
    "devices": [],
    "learned_guidance": false
  },
  "warnings": [],
  "telemetry": []
}
```

This response was returned by the live local engine. The version follows the built package version.

The examples below use the bearer token in the mode-0600 `daemon.json` written under `JOPEDIME_HOME`. The test script reads it without printing it or placing it in a process argument.

```sh
daemon_file="$JOPEDIME_HOME/daemon.json"
url=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["url"])' "$daemon_file")
python3 - "$daemon_file" <<'PY' | curl --fail-with-body --silent --show-error --config - "$url/v1/system"
import json, sys
record = json.load(open(sys.argv[1]))
print('header = "Authorization: Bearer ' + record["token"] + '"')
PY
```

The `system_info` response has `health: ok`, `api_version: 1.0`, `local_paths: true`, and empty task, language, device, warning, and telemetry lists in this build.

Register the catalog CSV over the HTTP route with an idempotency key. This replay returns the same profile with HTTP 201:

```sh
python3 - "$daemon_file" <<'PY' | curl --fail-with-body --silent --show-error --config - \
  --output response.json --write-out '%{http_code}\n' \
  -X POST -H 'Content-Type: text/csv' \
  -H 'Idempotency-Key: rel05-s3-catalog-30-row-v1' \
  --data-binary "@$REL05_DATASET_CSV" "$url/v1/datasets?name=rel05-s3-catalog-30-row"
import json, sys
record = json.load(open(sys.argv[1]))
print('header = "Authorization: Bearer ' + record["token"] + '"')
PY
```

| Operation | Method and path | Current router result |
|---|---|---|
| `system_info` | `GET /v1/system` | `200 OK`, system snapshot above |
| `dataset_register` | `POST /v1/datasets` | `201 Created`; CSV upload verified |
| `dataset_inspect` | `GET /v1/datasets`, `GET /v1/datasets/{id}` | `422 CAPABILITY_UNAVAILABLE` |
| `dataset_delete` | `DELETE /v1/datasets/{id}` | `422 CAPABILITY_UNAVAILABLE` |
| `objective_validate` | `POST /v1/objectives/validate` | `422 CAPABILITY_UNAVAILABLE` |
| `run_start` | `POST /v1/runs` | `200` dry-run plan or `201` durable start; bounded CPU numeric regression |
| `run_status` | `GET /v1/runs`, `GET /v1/runs/{id}` | `200` paginated inventory or committed snapshot; long-poll supported |
| `run_control` | `POST /v1/runs/{id}/control` | `422 CAPABILITY_UNAVAILABLE` |
| `run_fork` | `POST /v1/runs/{id}/fork` | `422 CAPABILITY_UNAVAILABLE` |
| `run_delete` | `DELETE /v1/runs/{id}` | `422 CAPABILITY_UNAVAILABLE` |
| `run_events` | `GET /v1/runs/{id}/events` | `200` durable JSON page or authenticated SSE stream |
| `algorithm_get` | `GET /v1/runs/{id}/algorithms`, `GET /v1/runs/{id}/algorithms/{alg}` | `422 CAPABILITY_UNAVAILABLE` |
| `model_export` | `POST /v1/runs/{id}/exports` | `422 CAPABILITY_UNAVAILABLE` |
| `artifact_download` | `GET /v1/artifacts/{id}` | `422 CAPABILITY_UNAVAILABLE` |
| `model_predict` | `POST /v1/predictions` | `422 CAPABILITY_UNAVAILABLE` |
| `feedback_export` | `GET /v1/runs/{id}/feedback` | `422 CAPABILITY_UNAVAILABLE` |
| `store_gc` | `POST /v1/store/gc` | `422 CAPABILITY_UNAVAILABLE` |

Dataset registration is implemented by SVC-08 #1037. Dataset inspection/deletion, run control/fork/deletion, objective, algorithm, export, artifact download, prediction, feedback, and GC operations remain stubs. A stateless router without engine/store ownership retains explicit run capability refusals.

## Run lifecycle

The CLI provides `run`, `status`, and `events`. `run --help` lists dataset, target, task, split, objective, effort, budget, landmarks, guidance, device, and seed as required inputs. A run identifier is required by both `status` and `events`.

For a registered numeric dataset, this form executes native search and finalizes an actual fitted model:

```sh
jopedime run --dataset "$DATASET_ID" --target "$TARGET" --task regression \
  --split '{"kind":"random","holdout_fraction":0.2,"group_column":null,"time_column":null}' \
  --objective rmse --effort fast --budget '{"evaluations":"4","wall_seconds":"30"}' \
  --landmarks none --guidance heuristic --language rust --require-all-languages \
  --device cpu --seed 73 --idempotency-key my-bounded-run --json
```

The completed `RunSnapshot` includes the champion's finite CV score, fitted model artifact ID, completion reason, and event watermark. The batch fallback owns the store and exits after its run without creating a daemon. An existing daemon remains shared. The engine admits one active CPU run at a time. Retrying an identical request with its original key returns the original start response; read status for current state. A changed request under that key conflicts. Dry runs dispatch no workers; the returned `plan_digest` can bind a subsequent start to the resolved plan.

This backend supports explicit regression, numeric features, RMSE, a random internal assessment split, no landmarks, and Rust operations. `device:auto` resolves to CPU; `guidance:auto` reports the heuristic fallback warning. Other tasks/objectives/splits/languages, learned guidance, and separate test datasets are refused explicitly. Only FIT rows reach fitting workers; internal assessment rows are scored during finalization. Wall budgets retain discovery's wave-boundary accounting and may finish after the nominal time limit.

The terminal can configure this subset before opening a dataset: `\landmarks none`, `\lang rust`, `\target COLUMN --task regression`, and `\effort fast --evals 4 --time 30s`. `\run` accepts optional `--seed` and `--idempotency-key`. The integration suite submits the same request/key through batch, MCP, HTTP, in-process, and an actual PTY terminal, then compares the associated committed champion, scores, and event stream. This covers shared-run transport equivalence; independent rerun determinism remains a separate requirement.

Export/predict and run-control routes remain unavailable. Interrupted runs remain inspectable as paused; public continuation/recovery is unqualified. Full H1/H2/AG1 export flows, independent-rerun S1, and Linux/macOS hosted CI remain outstanding.
