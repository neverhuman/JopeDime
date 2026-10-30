# Reference datasets v1

All included data comes from read-only objects in `s3://veox-jopedime-datasets-use2`. No external public or synthetic fixtures are used. `manifest.json` records SHA-256, size, license status, and source S3 key for each file; per-dataset READMEs provide the same receipts in human-readable form. Total file bytes excluding this manifest: 2698144 (limit: 20,000,000).

| Category | Fixture | Catalog id | Status |
|---|---|---|---|
| D2 regression | `d2-regression/` | `21fbfe8b53dbcd60` | Included, 1999 × 22 |
| Binary | `binary-loanapp/` | `02d7a557213d8979` | Included |
| Grouped | `grouped-apartment-rent/` | `c4b905cec711a618` | Included, native grouped split |
| Time-ordered | `time-ordered-prisonlf/` | `05f1f3e17ca435fb` | Included |
| Tiny table | `tiny-30-row-table/` | `b4c8807a75fc190f` | Included, 30 × 13 |
| Messy CSV-like | `messy-csv-like/` | `f803c49aca66290d` | Included; quoted-field and Unicode-header traits remain partial GAP |
| Multiclass | — | — | DEFERRED (t525u) |

Licenses are recorded as `unknown` where the S3 catalog does not provide a license. See each directory README and the root manifest for source keys and SHA-256 receipts.
