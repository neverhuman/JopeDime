# Local API and run commands

The command line includes `api` and `mcp` stdio modes, plus engine-backed run and dataset commands. Their exact arguments can vary by build; inspect the installed executable with `jopedime api --help`, `jopedime mcp --help`, or `jopedime --help` before configuring a client.

The current binary describes the dataset command as “Manage datasets registered with the local engine” and lists `add`, `list`, `show`, and `delete`. Presence in CLI help does not mean the service operation is available in this build.

**TODO(SVC-05-F01):** verify and document dataset API requests and responses against a build with a `dataset_register` service handler. Then add an executed API or MCP request using a permitted catalog dataset.

## HTTP routes in the current service build

The router contract test exercises all 20 registered method/path pairs. `GET /v1/system` succeeds with HTTP 200. Every other pair currently returns HTTP 422 with `CAPABILITY_UNAVAILABLE`. The local engine serves this router on IPv4 loopback; the results below are from the router contract tests, not a live `curl` session.

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

This response is exercised by the service router tests. The version shown is the package version used by the recorded `0.4.1` build; the live response follows the built package version.

| Operation | Method and path | Current router result |
|---|---|---|
| `system_info` | `GET /v1/system` | `200 OK`, system snapshot above |
| `dataset_register` | `POST /v1/datasets` | Stub: `422 CAPABILITY_UNAVAILABLE` |
| `dataset_inspect` | `GET /v1/datasets`, `GET /v1/datasets/{id}` | `422 CAPABILITY_UNAVAILABLE` |
| `dataset_delete` | `DELETE /v1/datasets/{id}` | `422 CAPABILITY_UNAVAILABLE` |
| `objective_validate` | `POST /v1/objectives/validate` | `422 CAPABILITY_UNAVAILABLE` |
| `run_start` | `POST /v1/runs` | `422 CAPABILITY_UNAVAILABLE` |
| `run_status` | `GET /v1/runs`, `GET /v1/runs/{id}` | `422 CAPABILITY_UNAVAILABLE` |
| `run_control` | `POST /v1/runs/{id}/control` | `422 CAPABILITY_UNAVAILABLE` |
| `run_fork` | `POST /v1/runs/{id}/fork` | `422 CAPABILITY_UNAVAILABLE` |
| `run_delete` | `DELETE /v1/runs/{id}` | `422 CAPABILITY_UNAVAILABLE` |
| `run_events` | `GET /v1/runs/{id}/events` | `422 CAPABILITY_UNAVAILABLE` |
| `algorithm_get` | `GET /v1/runs/{id}/algorithms`, `GET /v1/runs/{id}/algorithms/{alg}` | `422 CAPABILITY_UNAVAILABLE` |
| `model_export` | `POST /v1/runs/{id}/exports` | `422 CAPABILITY_UNAVAILABLE` |
| `artifact_download` | `GET /v1/artifacts/{id}` | `422 CAPABILITY_UNAVAILABLE` |
| `model_predict` | `POST /v1/predictions` | `422 CAPABILITY_UNAVAILABLE` |
| `feedback_export` | `GET /v1/runs/{id}/feedback` | `422 CAPABILITY_UNAVAILABLE` |
| `store_gc` | `POST /v1/store/gc` | `422 CAPABILITY_UNAVAILABLE` |

The `dataset_register` row is intentionally a stub pending SVC-08's handler PR. Do not use CLI command presence or the OpenAPI registry as evidence that an operation is executable.

## Run lifecycle

The CLI provides `run`, `status`, and `events`. `run --help` lists dataset, target, task, split, objective, effort, budget, landmarks, guidance, device, and seed as required inputs. A run identifier is required by both `status` and `events`.

**TODO(SVC-05-F01):** add request and response examples for starting a run, reading its status, and following its NDJSON events after the dataset and run handlers work end to end.
