# Troubleshooting

## The engine reports `stopped`

Run:

```console
$ jopedime serve --status
stopped
```

This is a successful status query: the engine is not running. The example was captured with an isolated `JOPEDIME_HOME`, so it did not inspect or alter a user's normal engine state.

## Dataset registration works; dataset listing is unavailable

Use `jopedime dataset add` with a CSV file, a name, and an idempotency key. The local engine accepts the registration and returns its dataset ID and profile.

The CLI also lists `dataset list` and `dataset show`, but those operations are not implemented yet. `jopedime dataset list --json` currently exits 2 and reports `Operation dataset_inspect is not implemented in this build.` Do not interpret a listed CLI command as a supported service operation.

The router also reports `CAPABILITY_UNAVAILABLE` for dataset deletion, objective validation, run lifecycle, algorithm lookup, export, artifact download, prediction, feedback export, and store garbage collection. `GET /v1/system` and `POST /v1/datasets` are the implemented operations in this build.

## Check version and build features

```console
$ jopedime --version --json
```

Inspect the `version`, `target`, `features`, `package`, and `source_sha40` fields when reporting a problem. A development build may report `package: null` and `trusts_development_key: true`; that is not release verification.
