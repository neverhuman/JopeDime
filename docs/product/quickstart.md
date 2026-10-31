# Quickstart

This page covers checks that work with the current `jopedime` executable. The examples below were run against the default-feature binary built from source revision `3a1adc560686251bc7c24c5a7dd03178d27558eb` (version 0.4.1, target `aarch64-unknown-linux-gnu`). The build reports `source_sha40: unknown`, so the source revision is recorded separately.

## Check the executable

```console
$ jopedime --version --json
{
  "features": {
    "cuda": false,
    "dev_keys": true,
    "dope_cuda": false,
    "embedded": false
  },
  "package": null,
  "package_format": "jope.appliance-manifest/v1",
  "product": "jopedime",
  "runtime_abi": "jope.appliance-runtime/v1",
  "source_sha40": "unknown",
  "target": "aarch64-unknown-linux-gnu",
  "trusts_development_key": true,
  "version": "0.4.1"
}
```

The command exited successfully. This development-key build is useful for local CLI preparation; it does not establish release signature or package trust.

## Check the background engine

```console
$ jopedime serve --status
stopped
```

This was run with a fresh, isolated `JOPEDIME_HOME`. It reports the engine state without starting it.

## Start a run

`jopedime run` requires a dataset identifier and a complete run configuration. A complete example depends on registering or otherwise provisioning a dataset in the engine.

**TODO(SVC-05-F01):** document and execute dataset registration, then add a complete run example with a real, permitted catalog dataset and its resulting run ID.
