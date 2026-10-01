# Kernel version-2 identity ledger

The v1 wire registry (0–52), program envelope (`FMKPRG\0\x01`), exact
`Func` tags (0–18), and exact `Var` tags (0–7) remain frozen. These new IDs
belong to a separate version-2 reservation table in
`crates/feat-math/src/kernel/wire_v2.rs`. They are not entries in the jain
registry, the frozen grammars, or a generator. Reservation gives a number a
stable owner; it does not make the operation executable or exportable.

| Domain | ID | Assigned name | State |
| --- | ---: | --- | --- |
| wire | 53 | symlog | reserved orphan stage |
| wire | 54 | select_corr | reserved orphan stage |
| wire | 55 | impute | reserved orphan stage |
| wire | 56 | pair_ratio | reserved orphan stage |
| wire | 57 | spline | reserved orphan stage |
| wire | 58 | savgol | reserved |
| wire | 59 | finite_diff | reserved |
| wire | 60 | cumint | reserved |
| wire | 61 | goertzel_bank | reserved |
| wire | 62 | acf_bank | reserved |
| wire | 63 | hampel | reserved |
| wire | 64 | realized_path_stats | reserved |
| wire | 65–74 | unassigned | reserved slots; proposed names in the breakthrough brief require later assignment |
| wire | 75 | timescale_nabla | reserved |
| Func | 19 | probit | reserved |
| Func | 20 | logit | reserved |
| Func | 21 | expm1 | reserved |
| Func | 22 | log1mexp | reserved |
| Func | 23–34 | unassigned | reserved slots; proposed names require later assignment |
| Var | 8 | row_median | reserved |
| Var | 9 | row_iqr | reserved |

The binary reservation envelope uses `FMKPRG\0\x02`, variant 0, minimum
reader version 2, and required feature bit 0. Variant 0 contains only a
canonical ordered list of assigned reservations. It carries no executable AST
or fit state. Future executable variants must receive a different variant tag
and a reader-first rollout. The new versioned reader accepts v1 through the
unchanged v1 reader, accepts the v2 reservation envelope, and reports an
explicit `UnsupportedVersion` error for an unknown envelope version, feature,
variant, domain, or ID. The old `Program::from_bytes` continues to reject v2
before evaluation with its frozen v1 error.

Versioned model JSON has a distinct tagged top-level envelope with
`format_version`, `min_reader_version`, `required_features`, `numerical_policy`,
grammar and emitter identities, output schema, the v2 reservation program,
and a validated legacy model payload. Its v2 form is deliberately not
convertible to an executable `Model`; direct legacy serde remains unchanged.
No v2 generation or numerical policy change is admitted by this reservation PR.

The checked capability manifest renders the table and marks every row
`generatable: false`. Tests pin contiguity, uniqueness, v1 byte preservation,
v2 round trips and refusals, and the unchanged jain registry hash. Later PRs
may assign the held slots or add executable v2 variants without reusing an ID
or changing the v1 tables.
