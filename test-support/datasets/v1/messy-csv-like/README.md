# messy-csv-like

Real pre-normalization S3 source fixture for catalog dataset `f803c49aca66290d` (35 columns). It retains the raw source header, spaces and punctuation in header fields, and missing values. The fixture contains the header plus the first 1000 data rows (1001 CSV records total), parsed and serialized as CSV.

- Source key: `s3://veox-jopedime-datasets-use2/full-source-registry/v1/sources/sha256/f803c49aca66290d31fd0832b10b72df9a627e057877b06e880ba90ce1284099.csv`
- Full source SHA-256: `f803c49aca66290d31fd0832b10b72df9a627e057877b06e880ba90ce1284099`; bytes: 1602007; ETag: `1694aac6284efd37bad473cec5409a0d`.
- Fixture SHA-256: `2bb5dee875d2a5c41d004c5083d17acc3bac38e86005c2149eb6b004448889d3`; bytes: 138263; file: `raw-source-subset.csv`.
- License: catalog status `unknown`; no further license claim is made.
- Partial GAP traits: quoted fields and Unicode headers (catalog inspection found neither in the source sample).

See `source-receipt.json` for the machine-readable source and subset receipt.
