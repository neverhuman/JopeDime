# Bundled demo datasets

These CSVs are embedded in the `jopedime` executable and appear in the JAIN TUI as virtual
`example/<name>.csv` files (`/demo`, `/open example/<name>.csv`). Together they stay below 1 MB
(`MAX_TOTAL_BYTES` in `src/terminal/demos.rs`, checked by its tests).

They are derived deterministically from the reference fixtures in `test-support/datasets/v1/`
(`SOURCES` in `src/terminal/demos.rs`): each demo keeps every k-th row of the fixture's
`train.csv`, prepends the catalog's recorded column headers (`meta.json` `output_headers`), and
keeps the values exactly as the corpus stores them (min-max scaled). A unit test re-derives
every byte from the fixtures. `manifest.json` records, per demo, the row selection, source
fixture SHA-256, catalog id, source URL and licence.

| Demo | Source | Rows | Licence (researched 2026-11-04) |
| --- | --- | ---: | --- |
| `regression_parkinsons` | UCI Parkinsons Telemonitoring (`d2-regression`) | 1000 | CC-BY-4.0 ([UCI 189](https://archive.ics.uci.edu/dataset/189/parkinsons+telemonitoring)) |
| `regression_apartment_rent` | UCI dataset 555, apartment rent (`grouped-apartment-rent`), every 5th row | 1600 | CC-BY-4.0 ([UCI 555](https://archive.ics.uci.edu/dataset/555/apartment+for+rent+classified)) |

**Only CC-BY-4.0 demos ship.** The dataset catalog still records these sources as `unknown`;
the licences above were researched afterwards and are recorded in `manifest.json` with
evidence URLs and attribution; the licence, its URL and the credit are printed when a user
selects a demo (`/demo NAME` or `/open example/NAME.csv`). Two earlier candidates were removed before
the first public release rather than ship copyleft or share-alike data inside the binary:
`binary_loan_approval` (wooldridge `loanapp`, GPL-3.0 package data) and `regression_tiny_soil`
(LimeSoDa `BB.30_2`, CC-BY-SA-4.0). See `docs/public-readiness-checklist.md` §3.
