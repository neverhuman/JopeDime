# H127 reproducibility bundle

Preregistration: `3e04ffa66fc0221456a94611e8df7ed757c6cf0f`; PR#1108 merged before scoring. Executed CPU wrapper is exact-hash frozen in the execution archive; shared source pins are in manifest.json.

- Manifest: `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H127/astra-t1105/manifest.json`, VersionId `eAFAm_AFDAxMeujJdwEujz4KZoDDoxG4`, SHA256 `9ebb1313671f485745c131d8548651dfecafbdf44f8f64ac0c7abc451344a428`.
- Archive: `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H127/astra-t1105/closeout-evidence.tar.gz`, VersionId `.9xeZC7L9GptfjmoSHQvTpPOpuU5.TIO`, SHA256 `96817b8e063313c0f760a34c37f538900e95edb45e9d83de77b8f858e642c980`, 38770479 bytes.
- Preserve the source/protocol, roster, regression, inner checkpoint metrics/adapters/loss curves, selection custody and final candidate metrics/adapters. Partial runs retain only actual artifacts. All member hashes are in closeout-inventory.json.
- Data: pinned real-S3 H50 DL-Train only, catalog version v246; seeds73–75; fold seed127073; inner bootstrap127001; final bootstrap127000. Frozen DOPE c51a56d0; original r882 SHA0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64.

Retrieve by the exact VersionId into custody outside any repository, verify the archive SHA and every member hash, then replay `select_steps` and `summarize` in science.py using the saved JSONs. `verify_outcomes.py` documents the complete-run checks. It needs the archived metadata and adapter snapshots, not dataset payloads or optimizer execution. Do not invoke monitor.py again: the registered wave is finite and forbids automatic retry after outcomes/resource stops.

The source archive binds the H125 schedule prefix, 50-parameter output-only scope and exact frozen-anchor caching. No serving or GPU admission follows from this bundle. Environment, resource samples and the payload audit are retained alongside the results. `input-locations.json` maps the 257 catalog manifest keys and all 1,219 local pool/table identities. Exact S3 recovery versions for the derived H50 inputs and original r882 were not verified by H127; preserve their existing local custody. Catalog admission alone does not establish local-row lineage.
