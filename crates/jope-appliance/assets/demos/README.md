# Bundled demo datasets

These CSVs are embedded in the `jain` executable (the jAIn CLI) and appear in the interactive CLI as virtual
`example/<name>.csv` files (`/demo`, `/open example/<name>.csv`). Together they stay below 1 MB
(`MAX_TOTAL_BYTES` in `src/terminal/demos.rs`, checked by its tests).

They are generated deterministically from the reference fixtures in
`test-support/datasets/v1/` by `generate.py` (run from the repository root with
`python3 -I crates/jope-appliance/assets/demos/generate.py`). Each demo keeps every k-th row of
the fixture's `train.csv`, prepends the catalog's recorded column headers, and keeps the values
exactly as the corpus stores them (min-max scaled). `manifest.json` records, per demo, the row
selection, source fixture SHA-256, catalog id, source URL and license status.

| Demo | Source | Rows |
| --- | --- | ---: |
| `regression_parkinsons` | UCI Parkinsons Telemonitoring (`d2-regression`) | 1000 |
| `binary_loan_approval` | Rdatasets `wooldridge/loanapp` (`binary-loanapp`), every 2nd row | 796 |
| `regression_apartment_rent` | UCI dataset 555, apartment rent (`grouped-apartment-rent`), every 5th row | 1600 |
| `regression_tiny_soil` | 30-row corpus table (`tiny-30-row-table`), source not recorded | 25 |

**Licensing is not yet cleared for public redistribution.** The S3 catalog records every
source's license status as `unknown`, and this directory makes no further license claim.
Confirm each source's terms (and attribution wording) before a public release ships them.
