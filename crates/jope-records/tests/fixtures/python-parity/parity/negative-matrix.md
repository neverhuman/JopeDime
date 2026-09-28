# Capability parity matrix (negative-case fixture)

One `bin` and one `example`, both with rows, so the base metadata passes the check.

## Counts

Recomputed from the rows below and checked by `deploy/ci/parity-matrix.sh`: a tally that disagrees
with them fails the gate. Two lanes adding a row of the same kind each bump the same cell to the
same number, so git reports no conflict — update this table from the rows, not from the last value
you saw.

| kind | rows | moved | new | pending | dropped | lost |
|---|---:|---:|---:|---:|---:|---:|
| new-bin | 1 | 0 | 1 | 0 | 0 | 0 |
| new-example | 1 | 0 | 1 | 0 | 0 | 0 |
| **total** | **2** | **0** | **2** | **0** | **0** | **0** |

`moved`/`new` rows without a covering test: 2.

## Rows

| kind | old path/name | new home | test | status | notes |
|---|---|---|---|---|---|
| new-bin | (new) | crates/demo/src/main.rs (demo) | none | new | negative-case fixture |
| new-example | (new) | crates/demo/examples/ex.rs (ex) | none | new | negative-case fixture |
