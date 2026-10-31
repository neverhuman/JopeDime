# Local API and run commands

The command line includes `api` and `mcp` stdio modes, plus engine-backed run and dataset commands. Their exact arguments can vary by build; inspect the installed executable with `jopedime api --help`, `jopedime mcp --help`, or `jopedime --help` before configuring a client.

The current binary describes the dataset command as “Manage datasets registered with the local engine” and lists `add`, `list`, `show`, and `delete`. Presence in CLI help does not mean the service operation is available in this build.

**TODO(SVC-05-F01):** verify and document dataset API requests and responses against a build with a `dataset_register` service handler. Then add an executed API or MCP request using a permitted catalog dataset.

## Run lifecycle

The CLI provides `run`, `status`, and `events`. `run --help` lists dataset, target, task, split, objective, effort, budget, landmarks, guidance, device, and seed as required inputs. A run identifier is required by both `status` and `events`.

**TODO(SVC-05-F01):** add request and response examples for starting a run, reading its status, and following its NDJSON events after dataset registration works end to end.
