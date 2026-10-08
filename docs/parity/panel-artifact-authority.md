# Panel artifact producer authority

The private raw-table panel adapter uses trusted, installed model factories and
validates untrusted requests as raw features and training labels. Those factories
already execute code during fitting. They must not come from a request, downloaded
artifact or untrusted Python plugin.

`_panel.fit` issues process-local authority for the complete SHA-256 identity of
its fitted artifact. `_panel.replay` checks the supplied digest against an immutable
snapshot and requires that identity to have been issued by this adapter process
before calling `pickle.loads`. Issuance is rolled back if the fit's replay check
fails. The artifact schema, bytes and reported SHA-256 remain unchanged.

A foreign artifact plus its matching digest is refused before deserialization,
including an otherwise valid model with a recomputed manifest pin. The private
authority set is never populated from an input file, request or manifest. It is
not serialized: another process cannot replay a persisted pickle through this
route. Within one process, artifacts issued by earlier successful fits remain
admitted. The adapter CLI performs one fit per process.

This is a restricted local-producer contract, not authentication of arbitrary
external producers or Python-code isolation. Privileged caller code can mutate
module state; the boundary protects untrusted artifact bytes, not hostile code
already executing inside the adapter. A future cross-process replay route needs
an independently trusted producer root or a data-only model format. A checksum
read from the same untrusted artifact bundle does not provide that root.

The target-maker joblib bundle route is separate and still requires its own
producer/manifest qualification (audit ordinal529). This change addresses the
panel replay boundary at ordinal100 only.

Focused acceptance uses the maintained Python tests:

```sh
python3 -m unittest discover -s crates/jope-bench/adapters -p test_panel.py -v
```

Controls include successful local regression/classification replay, original
pin-mismatch and mutable-buffer protections, a matching-digest executable-pickle
canary that must never run, a changed valid model with a recomputed pin, fresh
process refusal, and failed-fit issuance rollback. No Cargo build is required.
