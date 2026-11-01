# Q1 pre-run amendment T1115 — unique owner families, five-wide floor

Authority: ENG:watcher owner direction in `lanes_t1115.md`, 2026-11-01 18:15 UTC.
Q1 has **NOT RUN** and no Q1 outcome has been observed. The source-bound
[amendment manifest](amendments/t1115.json) records the exact direction/input hashes.

## Selection and identity

Use the 97 frozen owner families recovered in [the identity table](identity-table.json),
including its 24 families among the 36 formerly unbound records. The family key
removes only the literal `fam-` prefix from an existing frozen owner family ID.
This is a reversible namespace encoding, not a new family hash or an inferred
filename/URL-parent group. Original catalog fingerprints remain in the historical
source packet; they are not treated as additional independent owner families.

Apply the existing outcome-blind selection rule: choose the smallest eligible
dataset SHA-256 **within each owner family**. The immutable 153-record source
packet remains byte-identical (`ab434d14...`). Selection retains 97 representatives
and records the other 56 same-family variants with their selected representative
and reason. Every retained dataset and three-fold cut hash is unchanged.
No family is minted, split, moved out of training or counted twice.

RECS 2015 and RECS 2020 share owner family `fam-66943d2a...`; RECS 2015 has the
smaller dataset SHA-256 and is retained. Counting the two catalog fingerprints as
independent families caused the earlier four-wide count.

The complete membership, selection/exclusion inventory and wide census are pinned
by `amendments/t1115.json`. The census checks all 4,750 records of the exact v246
catalog version, including the 14 sealed exclusions listed by the catalog Markdown,
through pinned exact/source-aware P2 keys and the frozen owner/P3 family identities.
Among the 97 owner families, it finds four wide datasets but only **three distinct
wide families**. No additional eligible wide member is bound by those pinned keys.
Unreviewed aliases and unrelated families are not substitutions.

## Five-family requirement and present deficit

The latest owner direction requires **at least five distinct families in every
subgroup**, including wide (`features > 20`). It supersedes R19's four-wide exception
for this new plan. The current source rejects `minimum_wide_families` overrides;
neither validation nor analysis has a four-family/descriptive-only execution path.
The dated R19 documents remain historical evidence of an unexecuted plan.

| Subgroup | Selected distinct families | Required | Admission |
|---|---:|---:|---|
| small | 70 | 5 | Count met |
| large | 27 | 5 | Count met |
| narrow | 94 | 5 | Count met |
| wide | 3 | 5 | **INCOMPLETE — two families short** |

The duplicate-family blocker is resolved by selection. The five-wide requirement
is **not met**, so this regenerated plan is not signable or executable. The wide
effect/interval/verdict remains unknown; no Q1 measurement is substituted. Do not
invent two identities, duplicate a member, weaken the floor or silently widen the
approved family pool to claim readiness.

## Unchanged execution and inference

The primary normalized median effect, paired 10,000-resample bootstrap, seed
20261030, percentile 95% interval, strictly positive primary lower endpoint,
subgroup margin -0.02 and Bonferroni lower quantile `0.05 / 4` remain unchanged.
All four subgroup checks apply once admissible. Candidate r782 remains bundle
`94fd9587675c7806bd0a0582b6e97540cd2fdc7ebbc51958411b905185cad85c`, with DOPE
`c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`.
Evaluation seeds remain 0/1/2, each policy gets 24 charged admissions and 600 seconds
per dataset/seed, and the persistent canonical ledger/single-use rules remain.
No retry, observation deadline or study budget resets.

Namespace-aware training/exposure closure is still required: owner family IDs and
training parent-lineage IDs are not a crosswalk. The proposed training-closure
bytes stay at `8e786b2d...`; a different string namespace does not prove disjointness.
Actual source-bound evaluator/runtime qualification, evaluator hash, owner signature
and pinned-evaluator-only sealed-row access remain separate gates. GOV03 owns the
runtime handoff. API01 never signs; no signing request is filed while readiness is
false. This change opens no payload, reserves no FIFO ticket and consumes no ledger.

Replay the source-bound metadata draft with:

```sh
python3 -B deploy/qualification/q1/harness/q1_dedup.py \
  --training-closure /home/ubuntu/jope_plan/evidence/QLT-04/res03-t0218/exclusion-aware-closure.json \
  --out-dir /home/ubuntu/jope_plan/evidence/QLT-04/api01-t1115/plan-replay
```

The output retains null for an unavailable evaluator hash, reports the 3/5 wide
deficit, and is explicitly metadata-only with `signable:false` and `Q1:NOT_RUN`.
