# H102 — W5 provenance gaps and W9 CPU preparation

**W5 remains HELD.** The owner accepted [H98's refusal](../h98_w5_provenance/README.md)
and requested three specific proofs. Fresh real S3 receipts and a complete v3b
training-pool schema inspection still cannot establish any of them. The directed
fallback is implemented as a W9 receipt plan; no learning experiment was launched.
H102 is the overseer-assigned identity for PR #984; H101 belongs to ARGON W6.

| Required proof | Result | Exact missing artifact |
| --- | --- | --- |
| Every W5 family absent from the global sealed meta-test manifest | Unproved; global overlap **null** | Complete authoritative **identity-only** sealed-family manifest, independent authority receipt and matching family-alias registry pin |
| Each child linked to an existing parent program in its pool | Unproved for all **504** admitted children | Context/child/parent/operator event ledger bound to the actual pool-membership byte receipt and immutable lineage source |
| Each original row traced to a source S3 object and SHA | Unproved for all **504** children | Per-child source-row locator, S3 key/version/byte SHA, verified byte receipt and original-training-row ledger |

The census covers all **254 v3b pools / 123 families**, not all historical W5
revisions. All 254 pool hashes matched the pinned manifest and H50 training
identities. Inspection read ZIP directory names only: the nine members contain
program, program hashes, fitness/gate/logratio, generation, birth and delivery
arrays, but no parent hash, operator or original-row source ledger. No array
member was decoded. The 83-object approved source-prefix census is unchanged
from H98. Four exact-version S3 inputs passed SHA-256 readback. The global roster
was not supplied by the inspected catalog records; this bounded search does not
establish that no suitable artifact exists elsewhere.

A whole-pool `parent_pool_sha256` is **not** an AST parent-program link. A
manifest's S3 identity does not establish original-row provenance. H98's known
sealed aliases, role conflicts, completion/evaluator/cut requirements and other
admission gates remain applicable. This is a critique of the evidence available
for the historical W5 roster, not a finding of actual protected-outcome use.

[check.py](check.py) checks the three metadata relations offline. It rejects
missing lineage/source evidence, incomplete sealed scope, duplicate children,
wrong pool membership and mismatched byte pins. Its hypothetical relation PASS
is conditional on independently admitted authenticity and completeness: strings
and a `complete` flag cannot authenticate a manifest. It never fetches a URI or
grants execution. The actual [results](results.json) are FAIL with CLI exit 1.

## W9 fallback

The checker expands [H91's ten common-learner replay types](../h91_w9_closeout_readiness/README.md)
over the [H75 protocol](../h75_closed_loop_evolution/protocol.json): five folds ×
three seeds × five memories (A/B/C/D/uniform fit-family acquisition) × two learners
(neural/strongest W1 tree) = **150 receipt slots**. [w9_cpu_prep.json](w9_cpu_prep.json)
records each slot's fit-fold complement and missing receipt. Uniform memory may
use fit-family acquisitions only, never terminal R held-context outcomes.

This is receipt preparation, not a replay runner or protocol amendment. All 150
receipts are pending; strongest-W1 recipe binding, actual campaign consumption,
remaining authorized budget and outcomes remain unknown. The existing W9 limits
and **2026-10-31 14:00 UTC deadline** are retained verbatim, including the original
H75 planning arithmetic; they do not certify the expanded panel fits the remaining
budget. H91's conditional expansion arithmetic still needs admission. OPUS retains
runner ownership. Elo, U4 and val_reg_inner are null; no HWM candidate exists.

## Evidence and replay

[Identity alias](IDENTITY_ALIAS.md): H102 is the current publication ID. The
immutable S3 archive, its paths, source bytes and serialized `jope.h101.*` schemas
retain their historical identity; no object or outcome was rewritten. The replay
command below intentionally uses the original path inside that archive.

[evidence.json](evidence.json) binds the source commit, input SHA/version receipts,
checker and test bytes, complete schema census, host results and resource scope.
[custody.json](custody.json) gives create-only output objects and exact-version
readback hashes. Its archive includes the real input projection, pinned current
and historical H75 protocols, collector/preparer/worker sources, all pool schema
receipts, source-prefix listing, controls and four-host command/preflight records.

After downloading the archive at its recorded VersionId and verifying its SHA,
extract it into an external evidence directory. Run from that directory:

```sh
python3 candidate/research/flywheel_selfplay/h101_w5_provenance_gaps/check.py \
  --input input_packet.json \
  --sha256 c633a50e9cb2b32b8a0eb1216958edfc9aa81728722ac89265db124a7ff2b276 \
  --w9-protocol h75_protocol.json \
  --w9-protocol-sha256 d9973e972e54db0fc814dd8999a20fefbc450446152939e4d4a42c914c3a8ef9 \
  --output fresh-replay
```

Expected exit **1**, with `results.json` and `w9_cpu_prep.json` byte-identical to
the committed outputs. Existing output directories are refused. Run the synthetic
controls separately with `python3 -m pytest -q test_check.py`, from this source
folder, using a host-approved external pytest scratch directory.

Validation: **33 controls passed**, zero failed/skipped; these are not scientific
observations. Deterministic gate shards ran on xbabe1/2/3 and xbabe4 Linux Docker,
one worker each, with load/headroom admission below 85% and xbabe1 above 150 GiB
free. Nice19 on xbabe1, at least nice10 elsewhere. Four-host measured worker CPU
was **0.03865017 s** against the 180 s ceiling; local preparation used 0.327526857 s.
S3/SSH/test/publication overhead is unmeasured, not zero. No protected row contents,
model fits, native evaluations, GPU, optimizer updates, live-loop changes or
qualification-lock actions occurred. Targeted local checks only; no hosted CI claim.
