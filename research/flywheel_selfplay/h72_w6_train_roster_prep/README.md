# H72 — W6 training-only roster and fold preparation

**Status: preparation only.** This handoff supplies a field-allowlisted W6
DL-Train roster and five family/table-grouped folds. It contains no W6 model
fit, outcome comparison, scientific verdict, or serving recommendation.

## Source and method

The source packet is the admitted H50 packet: manifest file SHA256
`99a53569fcaf59f9309e69329c830ae1684adff382da3ae22e0b2645c4ebc4d4`, packet
identity `3eb4a62acea512f528500b043299a5968e1c9b1d1b1759bbf77a426b9fa1ecc3`.
The committed builder reads only the packet's first top-level `train` array,
stops before the following DL-Val section, and decodes task metadata only for
dataset IDs referenced by those training rows. It projects exactly the schema
accepted by [H66's fold builder](../h66_w6_cross_family_transfer/prepare_folds.py).
The full selected and task-metadata files are SHA-pinned for source identity;
their unselected records are not parsed or used. The inventory records the 257
unique, version-pinned real S3 training table references and their canonical
physical-table hashes. No dataset object, pool, checkpoint, score, or outcome
body was opened.

The builder ran at source commit
`8f874b40a3560b245cf10b63236feff118ebb326`; its SHA256 is
`cb7d66dd5778760d0e78e3e492204d005fe6129a0f99ac11dc949a0a5d8f454e`.
H66's `prepare_folds.py` SHA256 is
`d14fa2a7a511f0eaf1c721bab60dc684f6b12578d520dcdb682837785f05558d`.

## Prepared roster

- 1,219 DL-Train regression contexts across 205 declared families.
- 257 distinct, version-pinned S3 dataset objects referenced by the admitted
  training metadata.
- Five whole-family/physical-table folds, seed 4900; 205 resolved
  family/table components.
- Zero DL-Val rows parsed, zero outcome values read, zero pool/table/checkpoint/
  score bodies opened, zero outer rows delivered, and zero excluded holdout
  payloads opened.

This roster resolves the input-roster availability issue recorded in the
original H66 preparation README. It does **not** resolve the missing W1 model
outputs or provide W6 outcome history. The H66 numerical study remains
unstarted; its 1.5 GPU-hour and 1,200 native-evaluation caps are untouched.

## S3 custody

Canonical run ID: `w6-roster-prep-res03-20261031T0451Z-d4`.

Manifest:
`s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/W6/w6-roster-prep-res03-20261031T0451Z-d4/manifest.json`

Manifest SHA256:
`75fd0a63370ee235a165fa842bf4a3a773eff3f2b38ff2c3e636896dd663009f`;
version `ZerRr5DdgXIBhJEdmt_oMzZTH1fCteGT`.

| Object | SHA256 | S3 version |
|---|---|---|
| `train_roster.json` | `9d3656d9817239d07ac27d1cb4f90d688bf3421b492e15b62ac078a9d183190d` | `n5uFqKsIE1lL8bIwF4iwhJBUpc.ikG7T` |
| `training_source_inventory.json` | `982148e1f35a8ba1707845c3efb4eb48dacc558290383320f0e26b3dd5197076` | `gFBqaSyGUn27uN0keMv8LNo20kvVdtdg` |
| `fold_manifest.json` | `129002bcd6cfc4b2f3a0b9e0562967cd4510dcbf88364c2d11d0bbe8801c2c73` | `R1L8dcRZZzhDQz4XGpXyo02kHhHz6Dh2` |

All four S3 objects, including the manifest, were written with create-only
`IfNoneMatch=*` semantics and read back by exact version ID. Downloaded bytes
matched each local SHA256. The earlier immutable run
`w6-roster-prep-res03-20261031T0449Z-c3` is retained; it contains the same
payloads but pins the pre-rebase source commit. Run `d4` is canonical and pins
the durable rebased source commit.

## CPU and reported metrics

Preparation ran on xbabe2, one worker at `nice -n 15`; load average was 42.46
on 102 logical CPUs at launch. Roster construction used 0.24 seconds wall / 0.20
user / 0.03 system; fold construction used 0.05 seconds wall / 0.03 user / 0.02
system. GPU hours and native evaluations were both zero.

Family Elo, `val_reg_inner`, and U4 are null because no model was fitted or
scored. These are unmeasured, not zero. No HWM candidate pointer was written.

## Reproduction and checks

From the repository root, with the H50 packet available at the pinned path:

```sh
nice -n 15 python3 -m unittest \
  research.flywheel_selfplay.h72_w6_train_roster_prep.test_build_train_roster -v
nice -n 15 python3 -m unittest discover \
  -s research/flywheel_selfplay/h66_w6_cross_family_transfer \
  -p 'test_*.py' -v
nice -n 15 python3 \
  research/flywheel_selfplay/h72_w6_train_roster_prep/build_train_roster.py \
  --input /home/ubuntu/jope-data/h50/input \
  --roster "$W6_RUN_DIR/train_roster.json" \
  --inventory "$W6_RUN_DIR/training_source_inventory.json"
nice -n 15 python3 \
  research/flywheel_selfplay/h66_w6_cross_family_transfer/prepare_folds.py \
  --roster "$W6_RUN_DIR/train_roster.json" \
  --out "$W6_RUN_DIR/fold_manifest.json"
```

The checks above passed on 2026-10-31. No GitHub CI was dispatched.
