# H84 — W6 exact-program transfer coverage

**Completed CPU diagnostic: 7.0370% of candidate slots have an exact program
identity in the fit folds; 92.9630% do not.** Equal-family mean context coverage
is 7.1240%. This measures where exact-ID transfer could apply, not its predictive
value or a W6 learning result. Elo, val_reg_inner and U4 are null because no
outcome values were evaluated and no policy was fitted or scored.

The shard follows `overnight16_t0050.md`. It uses the [H72 W6 roster and
folds](../h72_w6_train_roster_prep/README.md), independently of W1 model recovery
and the H81 prospective comparator. It does not change their acceptance gates,
select a W1 winner, supply an outcome export, or authorize GPU work.

## Result

Across 1,219 DL-Train contexts in 205 families, the 1,404,287 candidate slots
contain 1,316,986 unique program SHA256 identities. For each context in fold k,
we count a slot as covered only if that exact program appears in a context in
one of the other four folds. H72 groups each family and physical-table identity
within one fold; no same-family or same-fold match qualifies.

| Held fold | Contexts | Families | Candidate slots | Seen in fit folds | Coverage |
|---|---:|---:|---:|---:|---:|
| 0 | 263 | 41 | 302,976 | 21,783 | 7.1897% |
| 1 | 216 | 41 | 248,832 | 18,793 | 7.5525% |
| 2 | 277 | 41 | 319,104 | 23,126 | 7.2472% |
| 3 | 240 | 41 | 276,480 | 18,728 | 6.7737% |
| 4 | 223 | 41 | 256,895 | 16,390 | 6.3800% |

Total: **98,820 covered slots; 1,305,467 unseen slots**. Context coverage ranges
from 2.4306% to 20.6597%, with median 6.25%; no context has zero or complete
coverage. Slot-weighted coverage is 7.0370231%, equal-context coverage 7.0370230%,
and equal-family mean context coverage 7.1239523%. These are different estimands;
the family value averages context fractions within family, then families equally.

A looser count would find 102,206 slots repeated in another context. Of those,
203 repeat only within the same family. Another 3,183 have a match in another
family but only inside the same held fold. Both groups are excluded, leaving
98,820. This is exact byte-identity coverage, not structural-template similarity,
kNN coverage, novelty of generated proposals, or empirical-Bayes performance.

The earlier exploratory 7.26% cross-family overlap shorthand did not impose this
exact fit-fold exclusion. It cannot be substituted for the 7.0370% measurement.
An exact-ID-only outcome lookup would need a fallback for the uncovered slots;
feature-based transfer can still apply there. Even a covered slot needs admitted
completed outcomes, evaluator/cut identity and original-row provenance before its
outcome is usable. This count supplies no accuracy or causal-lift evidence.

## Input and computational closure

H72's real-S3 roster, fold manifest and run manifest were downloaded by their
published version IDs and SHA256-verified. Their manifest identifies the H50
packet `3eb4a62acea512f528500b043299a5968e1c9b1d1b1759bbf77a426b9fa1ecc3`,
source-manifest file SHA `99a53569fcaf59f9309e69329c830ae1684adff382da3ae22e0b2645c4ebc4d4`,
and selected-file SHA `981960ad628f5029f8c54552c7b1cb1d440072e7e9dcffecdea52c6a3493a0f1`.
The full selected file was hashed for identity; its reader decoded only the
first `train` array and stopped before the DL-Val partition. Unused aggregate
candidate/gate counts in that metadata are not labels for this diagnostic.

All 1,219 admitted local-mirror pool files matched their selected-row hashes.
Whole NPZ bytes (162,215,175 bytes) were hash-streamed for identity, but only the
`program_sha256` member was decompressed/decoded. Features, fitness, logratio,
gate arrays, raw DSLs, tables, checkpoints and mixed score bodies were not loaded.
Pool file size/inode/mtime/ctime were stable across each read. Original pool S3
versions were not independently retrieved; the provenance is the pinned admitted
mirror plus H72's S3 metadata, not a new claim of remote pool-version verification.
The physical family/table mapping is inherited from H72 and checked for internal
fold consistency; original row aliases were not independently re-audited.

Program hashes were mapped bijectively to integers to make a small transport
packet. The exact vocabulary and its hash are retained with the evidence.
Workers received only context/family/fold/integer-ID metadata. They used five-bit
fold membership to count matches; an independent direct-set implementation
reconstructed every one of the 1,219 context counts. Neither implementation
consults outcome values. Coverage is conditioned on H72's exact folds and pool
revision; another W1/W6 fold map must not silently inherit these values.

## Execution and checks

Eight CPU controls passed with zero failures or skips: same-family/same-fold
exclusion, same-fold other-family exclusion, no overlap, shard partition equality,
and refusal of crossed families, duplicate contexts/IDs and invalid IDs.
Tiny integer fixtures test the counter only; scientific counts use real admitted
program identities. There was no repeated scientific run or extra training seed.

- xbabe2: extraction and independent direct-set validation, nice10, one worker.
- xbabe1: fold0, nice19, one worker; preflight free disk exceeded 150 GiB.
- xbabe3: folds1/2, nice10, one worker.
- xbabe4: folds3/4 in the existing Linux colima `jope-ci` container image,
  one CPU, 512 MiB, network disabled, nice10. No native evaluator is involved.

Hosts passed CPU/load admission below 85%; observed Linux hosts report 128 logical
CPUs, while admission conservatively used the brief's 102. The Mac has 32 host logical
CPUs and the Linux VM reports 24. The first Mac bind-mount launch
failed before the worker opened any data; copying the identical source/packet
into an owned container repaired that transport issue. The two successful Linux
shards were not rerun. The failure and successful retry are retained.

Successful measured compute used **14.312621 process CPU-seconds** of the 300-second
shard ceiling. Extraction used 8.904845 CPU-seconds /16.513775s wall;
remote shard CPU totals sum to 1.611481s; independent combination used
3.796296 CPU-seconds. Budgets for the W6 learning experiment are not reset.

Measured cost excludes network/S3/test/publication overhead and the failed Mac
interpreter startup, whose CPU cost is unknown. It must not be reported as zero.
No GPU, native fit, training update, live-flywheel read, sealed or holdout payload
access occurred. CPU resource receipts and per-host commands are in the archive.

## Durable evidence and reproduction

The version-pinned input references, 27 archived source/input/result members,
per-pool hash receipts, encoded packet/vocabulary, commands, controls and three
host outputs are preserved under a unique create-only W6 S3 run. The archive
was downloaded by exact version and every archived member independently hashed
against its inventory. Source is stored by hash outside Git; the recorded Git SHA
binds inspected repository context rather than inventing a committed checker.

Manifest: `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/W6/astra-h84-coverage-20261031T065538Z-a56ff9d9/manifest.json`

Version `OPzZJWiAmW9RuEv3em9W6Hhd.fvQ_xPv`; SHA256
`0122052f1648997ab2dede234fa74872b480437faca6c335a787ca08b621618c`.

To reproduce the count, restore the archive, verify its member inventory, then
run Python3 under the recorded CPU limits:

```sh
python3 coverage_worker.py --input identity_packet.json.gz \
  --sha256 13376e15a33d210063ce8356538e2b5ea282fd1cfdf82d428884ab4a50193689 \
  --folds 0,1,2,3,4 --host-label replay
python3 combine.py
python3 -B -m unittest -v test_coverage_worker
```

These commands use only restored identity metadata. `combine.py` reconciles the
retained shard outputs by direct sets. Own local bulk and remote transport copies
were removed only after S3 download and archive-member recovery verification;
original pool inputs and small source/result receipts were retained. The separate
create-only `cleanup.json` under the same S3 run records that removal.

[metrics.json](metrics.json) contains exact numerators, denominators, fold counts,
resources and limits. No HWM candidate or launch admission was emitted.
