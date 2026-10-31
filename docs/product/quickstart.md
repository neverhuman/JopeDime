# Quickstart

These examples were run against a default-feature `jopedime` binary built from source commit `75f5d704763c321f7586ea2db77cbdc771aa988e` (version 0.4.1, target `x86_64-unknown-linux-gnu`). `source_sha40` is `unknown` in a local development build. The binary is not a release-signed package.

## Check the executable

```console
$ jopedime --version --json
{
  "features": {
    "cuda": false,
    "dev_keys": false,
    "dope_cuda": false,
    "embedded": false
  },
  "package": null,
  "package_format": "jope.appliance-manifest/v1",
  "product": "jopedime",
  "runtime_abi": "jope.appliance-runtime/v1",
  "source_sha40": "unknown",
  "target": "x86_64-unknown-linux-gnu",
  "trusts_development_key": false,
  "version": "0.4.1"
}
```

## Start the local engine

Choose a dedicated data directory for the engine. Start it in one terminal; it serves only IPv4 loopback. In another terminal, point the client at the same `JOPEDIME_HOME`.

```console
$ export JOPEDIME_HOME="$HOME/.cache/jopedime-rel05-example"
$ jopedime serve --status
stopped
$ jopedime serve --idle-exit 10m
```

The `serve` command stays in the foreground. The local check harness starts and stops it automatically in an isolated home.

## Register the catalog example

Set `REL05_DATASET_CSV` to the permitted catalog CSV for dataset `b4c8807a75fc190f` (30 rows × 13 columns). This fixture is kept outside the repository. Add it with a stable idempotency key:

```console
$ jopedime dataset add "$REL05_DATASET_CSV" --name rel05-s3-catalog-30-row --idempotency-key rel05-s3-catalog-30-row-v1 --json
```

The local run returned dataset ID `ds_3fcf54fcaf3444a421c1f8de`, 30 rows, 13 numeric columns, and zero warnings. Repeating the same upload with the same key returned the same registration response.

The service implements `GET /v1/system` and `POST /v1/datasets`. Dataset listing and inspection, run planning and lifecycle, export, and prediction are not implemented in this build; those routes return `CAPABILITY_UNAVAILABLE`. The docs example check exercises the supported system and registration paths and verifies the expected refusal for dataset listing. See [Local API and run commands](api.md) for the route matrix.

Stop the local engine when finished:

```console
$ jopedime serve --stop
$ jopedime serve --status
stopped
```

## Start a run

`jopedime run` accepts a dataset identifier and a complete run configuration, but `run_start` is not implemented by the current service. A run example must wait for that handler and its qualified release path.
