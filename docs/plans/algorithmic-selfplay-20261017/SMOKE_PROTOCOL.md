# I32 offline synthetic research-loop smoke protocol (draft v1)

This protocol exercises one small CPU mechanism trace. Its only inputs are generated synthetic
TRAIN rows. It cannot qualify a teacher, claim scientific benefit, change accepted serving state,
or authorize publication. The permission in the capsule is explicitly nonqualifying.

## Required prebuilt fixtures

The test harness never invokes Cargo and never falls back to an in-process mock update. The owner
must build and admit both binaries separately, then provide absolute paths and full binary hashes:

```sh
JOPE_RESEARCH_LOOP_FIXTURE_BIN=/absolute/path/research_loop_fixture
JOPE_RESEARCH_LOOP_FIXTURE_BIN_SHA256=<sha256>
JOPE_CHALLENGER_FIXTURE_BIN=/absolute/path/challenger_lineage_fixture
JOPE_CHALLENGER_FIXTURE_BIN_SHA256=<sha256>
```

The current harness resolves the digest environment name by appending `_SHA256` to each binary
variable, so the exact names above bind the executable bytes. Missing paths, digest values or
executable permissions are **BLOCKED**, never a passing mechanism trace. Receipts must also retain
the source commit/tree, enabled features, compiler/backend and exact build argv for both binaries.

## Engine fixture

`jope-parallel`'s `research_loop_fixture` is a neural-feature, CPU-only Cargo example. Its draft
CLI is:

```sh
research_loop_fixture --output <fresh-lane-target-directory>
```

It uses `Policy`, the repository's composite training objective, scalar FP32 AdamW and the native
atomic checkpoint writer for one synthetic update. It commits an initial accepted-parent
checkpoint, saves a changed child checkpoint, independently reconstructs the child actor from
the saved model and records its next-actor logits/proposal. The publication crash is injected at
`LatestStaged`, after the child checkpoint directory is complete but before its latest pointer is
published. A dry-run rejection leaves the accepted-parent pointer byte-identical. The sealed
canary role check occurs before objective construction. The child export is research evidence;
the capsule keeps publication `pending`.

The example writes `capsule.json` and prints the same JSON to stdout. Its `trace` order is
proposal, exact synthetic TRAIN objective, nonqualifying admission, native CPU update, checkpoint
commit, native reload, next actor, dry-run judge, capsule and pending publication. `objective_loss`
is training telemetry for the fixture, not held-out evaluation or a benefit result.

## Independent backend fixture

The draft I14 Candle fixture interface used by this adapter is:

```sh
challenger_lineage_fixture \
  --i32-reload-checkpoint <capsule.child_checkpoint> \
  --output <fresh-output>/independent-reload.json \
  --cpu --dry-run-next-actor
```

It must independently load the child, compare the parameter digest with the engine capsule,
produce next-actor behavior, and emit `independent_reload_verified`,
`parameter_digest_matches_engine`, `next_actor_behavior_changed` and `dry_run_only` as true.
This command/receipt shape is an integration draft because I14 has not been accepted into this
base. A missing binary or unsupported interface blocks closure; the test does not manufacture a
reload result.

## Adversarial closure and interpretation

The engine capsule retains the leakage-canary refusal, the crash phase, complete child checkpoint
state, unchanged accepted parent, and dry-run rejection. The test asserts those facts plus changed
native parameters and actor logits. The separate backend test asserts independent reload and
next-actor behavior. Both tests are required for an offline mechanism closure. Missing I14 source,
binary or receipt means the overall closure remains BLOCKED even when the native fixture subtest
passes. Fixture completion proves no science, live durability, accepted serving promotion or
publication authority.
