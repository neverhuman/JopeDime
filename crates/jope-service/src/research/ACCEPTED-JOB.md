# Bounded TRAIN recording job (QLT-03)

`qlt03-slump85-v1` is an explicitly admitted internal recording job, not a
scientific quality or release qualification. It connects spec §13.2's durable
producer to the internal feature-gated CLI and existing Python reader. It does
not enable learned guidance defaults or qualify RNK-10, Q1 or REL-07.

The source pins the authoritative catalog, exclusion-aware family closure,
dataset manifest, all 85 native TRAIN rows of slump (`b9be38a2d975279c`), and
signed r882. The loader checks metadata before reading the table and uses the
production signature/asset loader before constructing Engine capability. The
only CSV transformation prepends nine deterministic feature names and the
manifest target name; every TRAIN row and label is retained. No outer/test or
sealed payload is loaded. New jobs require source review, not a JSON registry.

Native FIT conditioning, feature/projection inputs, measured surrogate/gate,
actual generator arms, same-call behavior costs and policy admission enter the
existing capture/outbox path. A callback observes native generation without RNG
changes. Research requests select the existing v2 evaluator before journaling;
label-only calls use reserved slots 32–47. Record, worker and checkpoint schemas
remain unchanged. The opaque resolver binds the complete Plan to a retained
input-admission receipt whose quality/release qualification flags are false.

Each invocation is fresh: one CPU worker, two rounds, seed count plus four R0
attempts (ordinary) or eight (evaluate-all), 300 seconds, 10 seconds/1 GiB per
attempt, and 16 MiB capture. Promotion/refit work keeps the existing separately
bounded auxiliary path. Modes have different evaluation budgets and run IDs;
this smoke proof is not an equal-budget quality comparison. Output, input receipt,
JMAT and store stay private. Existing output/inventory destinations refuse.

Build an internal source-bound executable with `--features research` and
`JOPE_BUILD_SOURCE_SHA=<full source SHA>` (no embedded/dev-key release features).
Place the exact policy files and authenticated bundle in
`$JOPEDIME_HOME/research/jobs/qlt03-slump85-v1/`, then run:

```sh
jopedime research record --job qlt03-slump85-v1 --out /private/ordinary.jsonl
jopedime research record --job qlt03-slump85-v1 --out /private/all.jsonl --evaluate-all-proposals
python3 python/src/jope/north_star/record_reader.py /private/all.jsonl \
  --expected-inventory /private/all.inventory.json --detailed-counts --require-evaluate-all
```

Alternatives rejected: a caller-controlled job-to-Engine importer, inferred
operator labels, a second forward just for costs, or a seed-only recording proof.
Remaining risk: authentication/input admission does not prove evaluator parity,
learned utility or release readiness; those retain their separate owners/gates.
