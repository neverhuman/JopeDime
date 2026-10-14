# Custody v2 local dry run

`jope_custody` consumes local, caller-pinned observations and payload files. It has
no S3 adapter, credential provider, deletion, lifecycle, scoring or training path.
All receipts remain nonqualifying. The injected library reader supports only
single-attempt version listing and exact-version reads; a live adapter requires
a separate scoped handoff and cannot hide retries from the physical-send guard.

The CLI input is a JSON object with `rows`, optional `facts`, `budgets` and
`members`. Each member contains an observed `row`, a `body_path` relative to the
input file (or an absolute local path), and a lowercase `body_sha256`. Optional
`pricing` contains captured `requests`, `rates` and a UTC `as_of` date. Pricing
uses those dated observations; unsupported classes, transfer, discounts and
actual billed cost remain unknown. The input is bounded to 32 MiB and must match
the caller's SHA before payload reads. Output must be outside every registered
Git checkout, including symlink destinations.

```sh
cargo run --locked -p jope-learning --bin jope_custody -- dry-run \
  --input "$CUSTODY_INPUT" --input-sha "$CUSTODY_INPUT_SHA" \
  --output "$CUSTODY_OUTPUT"
cargo run --locked -p jope-learning --bin jope_custody -- verify-complete \
  --directory "$CUSTODY_OUTPUT" --receipt-sha "$RETURNED_RECEIPT_SHA"
```

Use the receipt SHA returned by a successful publisher. An existing directory
or self-asserted `complete` field does not establish publication completion.
Incomplete attempts remain recoverable and are never reused. Local file and
parent-directory synchronization precede readback and terminal success.

`readback --manifest PATH` also reads archived `jope.custody-local-pack/v2`
manifests without Git or credentials. It authenticates the exact manifest bytes,
outer compressed SHA, all payload SHAs and the complete physical-ref/blob join;
it does not claim the historical producer completed durably. Archives are never
extracted. The reader bounds manifest, compressed, entire inflated TAR including
PAX and tails, unique payload, blob and reference rosters. Valid bounded PAX is
supported; unsafe effective names, links, duplicates and undeclared tails refuse.

Unknown keeper/hold facts retain bytes. The `other/14 days` field is an inert
historical proposal. The guard reports an alarm at 20,000,000,000 new-version
bytes/day and refuses its scoring recommendation at 25,000,000,000, unknown
counters, 1 TB hot bytes or 1,500,000 PUTs. It never changes services or storage
policy, and it does not prevent preserving custody evidence when counters are
unmeasured. A version listing is neither an atomic snapshot nor a net-new-byte
or PUT measurement.

The tests retain all 28 original Python test IDs and their 69 portable scenarios,
with the original eleven fixture files unchanged. Their generation metadata is
historical; actual Rust execution is reported separately. Additional tests cover
publication write/fsync failures, interrupted completion, fresh-process recovery,
full gzip/TAR tails, accepted PAX, overflow and CLI pin/output boundaries. A pack
sample qualifies only its captured physical references, not an entire prefix.
