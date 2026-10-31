# H104 — W7 acquisition / QD / replay CPU preparation

**Preparation only: no fits, training, GPU or measured lift.** H104 extends
[H69](../h69_w7_informative_acquisition_qd_replay/README.md) with a real pinned
training-identity fold manifest, a deterministic acquisition schedule, a
feature-only selection helper and refusal controls. [DESIGN.md](DESIGN.md)
defines the preparation recipe and remaining admission gates.

- [config.json](config.json): three acquisition arms, seeds 701–703, frozen DOPE,
  equal archive/replay limits, common-learner matching and Elo contract.
- [fold_manifest.json](fold_manifest.json): byte-identical H72 S3 metadata,
  1,219 DL-Train contexts / 205 families / five family-and-table grouped folds.
  Independently reconstructed from the pinned H72 roster with the H66 builder.
- [runner.py](runner.py): requires `--dry-run`, checks input byte pins and fold
  separation, produces a deterministic schedule and 45 pending matching receipts.
  It has no model loader, outcome reader, fitter, network client or execution mode.
- [test_runner.py](test_runner.py): synthetic controls for fold leakage, duplicate
  identities, contract drift, uniform support, label-field refusal and output safety.

The complete plan has **300 arm-shared context rounds / 3,600 planned new fits**.
Each held fold uses 20 distinct families from its fit folds. Across five folds,
that is 100 context rounds per arm/seed, preserving H69's arithmetic. The
existing 4,800-fit / four CPU-hour / two GPU-hour ceilings remain unchanged.
No remaining-budget, deadline or execution-readiness claim is made.

The uniform4 arm is the concrete fresh-label uniform-acquisition comparator
required by the W7 data-value gate; it is a preparation interpretation to freeze
before execution. The archive and training sampler are design contracts, not
implemented execution capabilities. Pending lineage, completed-status, current
role/alias exclusion, learner/source pins, timing/power and cost gates still block
execution. In particular H72 fold consistency does not prove global sealed-family
exclusion. V3 imports remain disabled. No protected payload was opened.

## Reproduce the dry run

From the repository root, set `W7_EVIDENCE_DIR` to an existing external evidence
directory and select an output path that does not exist:

```sh
H104_DIR=research/flywheel_selfplay/h104_w7_acquisition_replay_prep
nice -n 15 python3 "$H104_DIR/runner.py" --dry-run \
  --config "$H104_DIR/config.json" \
  --config-sha256 828838fe32777ee39d813029ac981d66748f04c7ea6f18073a1b939ab2012ec7 \
  --fold-manifest "$H104_DIR/fold_manifest.json" \
  --output "$W7_EVIDENCE_DIR/dry_run.json"
python3 -m pytest -q "$H104_DIR/test_runner.py" \
  --basetemp="$W7_EVIDENCE_DIR/pytest-scratch" -p no:cacheprovider
```

`--folds 0`, `1`, `2`, and `3 4` are the four deterministic shards. Their schedules,
fold summaries and matching panels must concatenate to the full plan; their fit
counts sum to 3,600. The dry run reports zero actual fits/updates and null Elo,
`val_reg_inner` and U4. Refusing an existing output protects prior receipts.

[dry_run_summary.json](dry_run_summary.json) records the exact full-plan hash,
four-host reconciliation, tests and source/input pins. The full schedule and
shard receipts are in the create-only archive identified by [custody.json](custody.json).
The hosts are xbabe1/2/3 and xbabe4 Linux Docker; preflight keeps load/headroom
below 85%, xbabe1 free space above 150 GiB, one worker each, nice19 on xbabe1
and at least nice10 elsewhere. No GPU, live-loop, DOPE or qualification-lock action.

Validation: 32 synthetic controls passed, zero failed/skipped. Four-host measured
worker CPU was 0.059136525 seconds against the 120-second preparation ceiling;
collection/local dry-run/test/transport/publication overhead is unmeasured, not zero.
