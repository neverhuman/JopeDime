# Research recording integration

Use the existing opaque CaptureContext, pinned Plan, durable Discovery capture and
atomic outbox materializer. A small coordinator validates a fresh destination and
the approved worker bound before starting a run, drives the existing backend, and
exports only complete committed Engine records. It does not construct a provider,
accept an Engine JSON import, change admission, retry unknown outcomes or qualify
inputs. The separately owned trusted job resolver supplies the reviewed context.

Return an inventory derived from the committed outbox: identity, mode, exact
JSONL SHA256, rounds/proposals/admissions/dispatches/successes/failures/outcomes and
incomplete rounds. The standalone Python reader can compare that inventory while
hashing the same bytes it validates. Keep its existing output and import API.

Reject alternatives: counting requested budgets, comparing only round totals,
hashing a second file read, accepting a synthetic control as Engine, adding a
constructor merely because pins parse, or enabling the CLI before trusted job
resolution and actual qualification exist. Preserve closed round/checkpoint and
worker schemas (spec13.2, discovery5, private-input/provider contract).

Verify both dispatch modes with explicitly synthetic durable-engine controls,
including a real known worker failure; compare Rust and Python identity/hash/all
counts and reject mutations, overwrite and invalid worker bounds before execution.
Actual real-S3 Engine acceptance remains distinct and consumes qualified providers
when their owners deliver them. No protected rows, signing, GPU or CI.
