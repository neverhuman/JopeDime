# Troubleshooting

## The engine reports `stopped`

Run:

```console
$ jopedime serve --status
stopped
```

This is a successful status query: the engine is not running. The example was captured with an isolated `JOPEDIME_HOME`, so it did not inspect or alter a user's normal engine state.

## Dataset registration is unavailable

The current executable lists dataset management commands, but the local service build does not implement the `dataset_register` operation. A registration attempt currently returns an “Operation dataset_register is not implemented in this build” error. Do not treat the CLI command's presence as a successful registration.

**TODO(SVC-05-F01):** replace this limitation note with the supported registration procedure and its verified recovery steps after the service handler is available.

The same router currently reports `CAPABILITY_UNAVAILABLE` for dataset inspection/deletion, objective validation, run lifecycle, algorithm lookup, export, artifact download, prediction, feedback export, and store garbage collection. `GET /v1/system` is the only implemented operation; its empty capability lists describe this service build, not a configured learning engine.

## Check version and build features

```console
$ jopedime --version --json
```

Inspect the `version`, `target`, `features`, `package`, and `source_sha40` fields when reporting a problem. A development build may report `package: null` and `trusts_development_key: true`; that is not release verification.
