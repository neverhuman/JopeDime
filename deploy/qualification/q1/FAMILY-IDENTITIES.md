# Q1 frozen family identity resolution — T1005

[identity-table.json](identity-table.json) resolves all **36** previously unbound
roster records to **24 pre-existing owner family IDs**. Its file SHA-256 is
`43de8a81ed1c509bd10b461c7ad1692b45fd0cbca9fa199189cb1823bc73d384`;
the canonical content digest is
`af9eeec7b186cca7b729819f23577795a592feefaef1e37382346a311c900826`.
The table is metadata only. It does not update a plan, qualify an evaluator or
authorize a Q1 start.

## Binding authority

The original owner roster (`roster_t0341.json`, SHA-256
`53b2a41cc913c3742d30f3eb46acf11dfc774109035897aa5df412fb4f4a43dc`)
already names these families and seals these dataset members. Its membership
projection is recoverable from the original proposal pinned by the H150 manifest:

- H150 manifest SHA-256:
  `0314599fe06515d3f3e976c0d435cbe4e9fbe26738014bd89470f525702f983b`.
- Original proposal SHA-256:
  `799d33eb305b64c0969aa0f4f2fc3e86d171fc58f1f295ded50944d561ab2efb`.
- P3 registry SHA-256:
  `1a4f1d419d781452a37711210e9e80cf4cda1d8f2095ac6a4f0430a01bda93ca`.

Every one of the 36 dataset IDs has exactly one membership in that frozen owner
family set. Thirteen bindings are independently corroborated by direct P3 member
dataset/canonical identities. The other bindings use the already pinned owner
membership; the table does not infer additional P3 registry members. The historical
proposal's `adopted:false` and `qualifying:false` flags remain recorded: authority
for this membership comes from the later owner roster, not from claiming that the
proposal qualified a study.

For Q1's existing hexadecimal syntax, `resolved_family_key` removes only the
literal `fam-` prefix from the existing `fam-<64hex>` identifier. Restoring that
prefix must recover the exact frozen ID. **No new family ID is minted**, and no
filename, source URL or dataset ID is hashed into a family label. Original catalog
fingerprints and frozen owner family IDs are distinct namespaces; this table does
not rewrite the 117 already bound catalog keys or claim a namespace crosswalk.

The catalog JSON (`e5f732fd...`) and Markdown (`bb3a1d20...`) identify the source
records. All 36 manifest/meta pairs were read through metadata-only S3 GETs: each
manifest ETag matches the pinned catalog, its dataset/canonical identity matches,
and its meta blob matches the content-addressed SHA-256. These records contain no
additional recorded family key. The table retains object versions/hashes and
source witnesses without fetching TRAIN, TEST or HOLDOUT row blobs.

Pinned P2 `family_keys.py` SHA-256
`9b688e9b8a36a32485524243dc1adea7d30ee7404df4027d2b4d17bd6a900f7e`
derives each exact `rdatasets_item` source key from its catalog provenance URL.
Those keys corroborate source identity; they cannot split or merge a frozen owner
family. Unreviewed `srcfile:<stem>` and URL-parent heuristics are not promoted to
authority. `master/family.toml` is pinned as coordination configuration; it has no
dataset-family mapping and supplies no family identity.

## Preserved constraints and exposed aliases

The original [153-cut roster packet](rosters/ab434d145b8cec60399b35c35a1ce9551a881843e8136b1c757e9c93ae62bd9a.json)
remains byte-identical (`ab434d14...`). All dataset/cut identities, existing family
keys, the training-closure pin, r782/DOPE, seeds, budgets, thresholds and R19 wide
reporting rule remain unchanged. No record is added, removed, duplicated or assigned
a replacement family.

Resolution reveals shared families: **36 records correspond to 24 owner families**;
the complete 153-record frozen membership corresponds to **97 owner families**.
Both alias inventories are explicit in the table. Do not manufacture distinct keys
for shared families or silently drop records to make validation pass. The existing
one-dataset-per-family and smallest-eligible-dataset-SHA selection requirements
still need reconciliation with the fixed roster before execution. This table does
not select a replacement cohort or waive either requirement.

Frozen owner family IDs and training parent-lineage IDs are different namespaces.
Different strings are not proof of zero training/exposure overlap. The complete
namespace-aware closure and actual source-bound evaluator/runtime qualification
remain separate requirements. GOV03 owns the evaluator/runtime handoff. QLT02 and
RES03 independently verify this table at the PR head. Q1 remains **NOT RUN** and
`signable:false`; no signing request, signature, FIFO ticket or ledger consumption
is produced here.

## Verification and custody

On the authoring host, replay the metadata verification with:

```sh
python3 -B /home/ubuntu/jope_plan/evidence/QLT-04/api01-t1005/verify_identity_table.py
```

The receipt checks every input and record hash, exact dataset/cut membership,
lossless family-key encoding, frozen owner membership, P2 source witnesses, S3
manifest/meta identities and alias counts. Corruption controls must be rejected.
Native `jope-records` source/test inventories, coverage, paper trace and repository
layout supplement this metadata proof. None is a Q1 execution result.

The original ephemeral proposal was copied byte-for-byte into durable
`jope_plan/evidence/QLT-04/api01-t1005/ORIGINAL-SEALED-ROSTER-PROPOSAL.json` before
using it. All exact source paths, hashes and roles are in `input_pins`; snapshots,
generator, verification program and receipts are retained in that evidence directory.
