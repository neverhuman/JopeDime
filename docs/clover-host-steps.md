# CLOVER quick start, steps 3 to 5: host inputs

Steps 1 and 2 of the [CLOVER quick start](../README.md#clover-self-play-flywheel-quick-start) run
from a clone. Steps 3 to 5 adapt R22 coverage, freeze candidate pools, evaluate selectors, distill a
results index and run the self-play flywheel against a matched-budget passive control. Like steps 1
and 2 they are offline and shadow-only: they fit selectors and surrogates on local files, train no
policy and write nothing to S3.

Steps 3 to 5 read host inputs that are not in Git. Put them in place first; each command stops at the first file it cannot read. They are built with `--release`: on xbabe2 on 2026-10-16, `evaluate` took 44 s that way and 767 s in the default debug profile, and `flywheel-distill` took 570 s that way and 8,125 s in debug.

| Input | What it is |
| --- | --- |
| `target/pools/coverage.json` | an R22 native trie-leaf coverage document (`jope.r22-lpi-scratch-coverage/v1`; scratch-only and non-qualifying) |
| `PROTOCOL_SHA256` | that measurement run's native snapshot spec SHA-256 (its receipt's `inputs.snapshot_spec_sha256`) |
| `target/pools/heldout.ids` | context ids to keep out of the pools, one per line, such as a held-out evaluation set |
| `target/pools/roster.jsonl`, `target/pools/roster-vectors.f32` | a joined native roster (`jope.joined-native-roster/v1`) and its f32 vector pack |
| `target/retention/GP_RESULTS_INDEX.json` | a results index built from the S3 results tree |

```sh
# 3. Adapt coverage (jope.r22-lpi-scratch-coverage/v1) for the train partition, leaving out the
#    held-out contexts (add --exclude-pairs to also drop each context's own incumbent), freeze
#    candidate pools, extract joined-roster context vectors (jope.joined-native-roster/v1), and
#    evaluate selectors:
cargo run --release --locked -p jope-learning --bin jope_pools -- ingest-coverage \
  --coverage target/pools/coverage.json \
  --exclude-contexts target/pools/heldout.ids \
  --protocol-sha256 "$PROTOCOL_SHA256" \
  --out target/pools/measurements.jsonl
cargo run --release --locked -p jope-learning --bin jope_pools -- freeze \
  --rows target/pools/measurements.jsonl \
  --source-grade native_nonqualifying_R22 \
  --out target/pools/frozen
cargo run --release --locked -p jope-learning --bin jope_pools -- ingest-roster-features \
  --roster target/pools/roster.jsonl \
  --pack target/pools/roster-vectors.f32 \
  --manifest target/pools/frozen/manifest.json \
  --out target/pools/context-features.jsonl
cargo run --release --locked -p jope-learning --bin jope_pools -- evaluate \
  --pools target/pools/frozen/pools.jsonl \
  --manifest target/pools/frozen/manifest.json \
  --reference pool-floor \
  --context-features target/pools/context-features.jsonl \
  --source-commit "$(git rev-parse HEAD)" \
  --out target/pools/eval-receipt.json

# 4. Distill a GP results index into a gp_distilled/v1 Parquet shard and run the multi-round
#    surrogate-guided self-play flywheel against a matched-budget passive control:
cargo run --release --locked -p jope-learning --bin jope_retention -- distill-gp-index \
  --gp-index target/retention/GP_RESULTS_INDEX.json \
  --split-role train \
  --source-commit "$(git rev-parse HEAD)" \
  --receipt target/retention/gp-results-index-s3-receipt.json \
  --out target/retention/gp-results-index-s3.parquet
cargo run --release --locked -p jope-learning --bin jope_pools -- flywheel-distill \
  --pools target/pools/frozen/pools.jsonl \
  --manifest target/pools/frozen/manifest.json \
  --reference pool-floor \
  --context-features target/pools/context-features.jsonl \
  --s3-prior-parquet target/retention/gp-results-index-s3.parquet \
  --rounds 5 \
  --initial-budget 2 \
  --step-budget 2 \
  --eval-budgets 1,4,8 \
  --parquet-dir target/pools/flywheel-shards \
  --source-commit "$(git rev-parse HEAD)" \
  --out target/pools/flywheel-receipt.json

# 5. Decode and inspect local Parquet pages (e.g. S3 Inventory or gp_distilled/v1 shards) with jope_retention:
cargo run --release --locked -p jope-learning --bin jope_retention -- inspect-parquet \
  --sample 3 target/retention/gp-results-index-s3.parquet
```

On 2026-10-16 these seven commands were run exactly as written from a fresh clone, with R22
scratch inputs. [`paper/NUMBERS_TRACE.md`](paper/NUMBERS_TRACE.md#fresh-clone-run) records the
inputs, exit statuses and timings. Output computed from scratch inputs is not a result.
