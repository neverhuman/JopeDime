# Release process

The maintained release procedure is [release/README.md](release/README.md).
It defines the version source, signed assets, SHA256SUMS verification, static-key
custody, and independent verification. The dispatch-only GitHub release workflow
refuses publication; the reviewed host release script is the publication path.

Before publishing, preserve the exact source SHA, full CI run and attempt,
review receipt, release manifest, signed checksums and independent verification
results described in that procedure. A queued, skipped or missing check is not a
passing release gate. Never move a published tag or overwrite a published asset.

## Roll back a binary release

1. Select a previously published release compatible with the deployment's model
   and dataset manifests. Preserve the current deployment's source and asset
   hashes, configuration and receipts before changing the executable.
2. Download that release into an owned directory and execute every signature,
   SHA256SUMS, source/tag and binary-bound check in the maintained procedure.
   Stop if any verification fails; an existing filename is not proof of identity.
3. Use the deployment's existing controlled activation procedure to select the
   verified executable. Retain the previous and replacement asset hashes and the
   activation result. Model bundles and dataset collections are independently
   versioned: switching a binary does not authorize rewriting those objects.
4. Verify the service's health and command compatibility against the expected
   deployment configuration. If activation fails, retain the failure receipt and
   restore the previously verified deployment through the same procedure.

Do not delete or move a published tag, overwrite a release asset, bypass a
signature failure, or run the dispatch-only release workflow as a rollback.
