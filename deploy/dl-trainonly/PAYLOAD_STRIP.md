# Published JTT1 payload strip (plan)

Status: plan only. This file adds no code and grants no publication, compute or runtime authority.

## Ruling

The owner ruling was delegated to ENG:watcher (sole sign-off; not Jepson-typed) on 2026-10-06 at 06:52 MT. It is recorded in `owner-decisions-20261006T125305Z.json` (sha256 `240e0af5521c03f95157c447916d61f3e49dffa2e564a72bc097f811df957baf`) in the lane's private custody. The decisions, verbatim:

> **PAYLOAD-STRIP-218**: #218 may merge as-is once grok_sit_watch red-team PASS at eb91ab02231266f5808592eefbfe4632fdf5b123 (no payload bytes changed; training reads only jtt1.train_view under the flip test)

> **PAYLOAD-STRIP-PUBLISH**: flip-test-only NOT accepted for publication: nothing carrying outer-test targets goes to S3; published encoder/teacher payloads hold no outer-test target values (drop the outer-test block, or keep features with target column removed); outer-test labels move to a separate scorer-only artifact with its own sha256, read only by the outer-test scorer adapter, never by encoder/teacher/selector/scaler code; must-fail test that fails if any encoder/teacher payload contains an outer-test target; provenance dry run stays 0 unresolved; lands in the RAW-GROUP follow-up PR or a small dedicated writer PR

> **PAYLOAD-STRIP-P8**: must not slip the 08:30 MT P8 DL start; P8 keeps reading through train_view

This is that small dedicated writer PR.

## Design

### Publication payload

The published `encoder.jtt1.gz` and `teacher.jtt1.gz` drop the outer-test block. They are JTT1 with `n_test = 0`:

- the same magic `JTT1`, `n_train`, `p` and `inner_folds`;
- the same inner-fold labels;
- the same train block;
- nothing after the train block.

`jtt1.parse_header` already admits `n_test = 0`, and the length rule becomes `17 + n_train + 8*(p+1)*n_train`.

`jtt1.train_view` (labels, X, y) and `jtt1.train_rows_sha256` (labels plus train block, header excluded) are therefore byte-identical to the unstripped file, so encodings computed through `train_view` do not change. A byte-compare receipt over every published context must prove this before publication.

The full-file sha256 does change. The publication manifest therefore binds the stripped raw and gz sha256 as the payload identity and records the unstripped raw sha256 only as lineage.

### Scorer-only outer-test artifact

The removed outer-test block moves to a separate artifact:

- **Magic:** its own magic (proposed `JTS1`), which `jtt1.parse_header` refuses. Encoder, teacher, selector and scaler code cannot decode it.
- **Content:**
  - the outer-test rows in their original order, as float64 LE, features plus target;
  - `n_test` and `p`;
  - the stripped payload's raw sha256;
  - the unstripped raw sha256.
- **Identity:** its own sha256, recorded in the context manifest.
- **Storage:** local-only and never published. It is read only by the outer-test scorer adapter.

A reconstruction check proves that concatenating the stripped payload's labels and train block with the artifact's rows, under the original header, returns the original raw bytes exactly. Nothing is lost or altered.

### Publisher gate

`tools/v2_publish.py` gets a row-level gate that runs before any client is created. For every JTT1 object in the publish set it:

- decompresses the object;
- parses the header;
- refuses the whole batch, fail-closed, if any `n_test > 0`, any length does not equal the `n_test = 0` length, or any object carries the scorer magic.

A refusal publishes nothing.

## Must-fail tests

1. An unstripped payload (`n_test > 0`) in a publish set is refused by the publisher gate before any client exists.
2. The strip tool refuses a source whose train block was tampered: the stripped `train_rows_sha256` must equal the source's before anything is written.
3. A tree scan fails if any encoder or teacher payload has `n_test > 0`, has bytes past the train block, or if a scorer artifact appears in the publication set.
4. A static import test (AST scan of `dltrain/` and `tools/`) fails if any module other than the outer-test scorer adapter imports the scorer-artifact reader or opens a scorer artifact path.

Each test is shown to fail on the current unstripped builders, then to pass with the strip.

## Provenance

The strip tool, the gate, the scorer adapter and the changed builders enter code custody before any publish run. The provenance dry run must stay at 0 unresolved (`tools/v2_provenance.py` fails closed with exit 2 otherwise).

## Delivery

- **Now:** this PR carries only this plan. It depends on #218 merging first and changes no file that #218 changes.
- **After #218 merges:** the lane code arrives as a later lane-handoff restage commit on this branch, with exact producer bytes, `verify_lane.py` pins, and the offline runner and coverage updates in the same form as #218. Before that commit, the branch is brought up to date with main by a history-preserving update. It is never force-pushed or rewritten.
- **P8:** keeps reading through `train_view` and is not blocked by this change.
- **Other work:** the RAW-GROUP follow-up PR stays separate.
