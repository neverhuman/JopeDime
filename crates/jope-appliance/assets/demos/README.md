# Bundled demo datasets

These CSVs are embedded in the `jain` executable (the jAIn CLI) and appear in the interactive CLI as virtual
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
| `binary_loan_approval` | Rdatasets `wooldridge/loanapp` (`binary-loanapp`), every 2nd row | 796 | GPL-3.0-only via the R package [wooldridge](https://cran.r-project.org/package=wooldridge); no licence from the original authors (Hunter & Walker 1996) |
| `regression_apartment_rent` | UCI dataset 555, apartment rent (`grouped-apartment-rent`), every 5th row | 1600 | CC-BY-4.0 ([UCI 555](https://archive.ics.uci.edu/dataset/555/apartment+for+rent+classified)) |
| `regression_tiny_soil` | LimeSoDa `BB.30_2` (`tiny-30-row-table`, source file `limesoda-bb-30-2-dataset.csv`) | 25 | CC-BY-SA-4.0 ([LimeSoDa](https://github.com/JonasSchmidinger/LimeSoDa/blob/main/man/BB.30_2.Rd)) |

**Not cleared for a public release yet.** The dataset catalog still records every source as
`unknown`; the licences above were researched afterwards and are recorded in `manifest.json`
with evidence URLs and attribution. Before a public release ships these bytes, the owner must
accept GPL-3.0 data (`binary_loan_approval`) and CC BY-SA share-alike terms
(`regression_tiny_soil`) inside the binary or replace those demos, and the CC BY attributions
must ship with the executable (`/demo` prints them). See `docs/public-readiness-checklist.md` §3.
