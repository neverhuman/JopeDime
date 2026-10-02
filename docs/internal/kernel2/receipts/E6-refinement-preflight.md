# Track E6 — refinement objective preflight

This focused correction closes PR157's post-merge independent P2 finding
5394321472. The owner merged that PR while the fresh independent review was
running; that child subsequently posted a blocking COMMENT and no approval.

`optimize_constants_v2_guarded` now refuses `LinearScaled` with elementwise
scope before consulting the trial guard. Both an allowing and a stopping guard
receive the same error as the existing optimizer, with zero guard calls and no
trial. Valid row objectives and elementwise callbacks retain their prior paths.
The original optimizer and numerical helpers are unchanged.

## Acceptance

```sh
cargo test -p feat-math --locked --test kernel2_refinement_control
```

The two new nonignored regressions fail before the correction: a false guard
hides the error, while a true guard reports it as a charged initial trial failure.
Both require the old error text and zero guard invocations. The existing seven
control, accounting and bit-parity tests remain present.

Fresh untouched origin/main `5043dd94f4491e0dd0d7127e42a20dbffc1ca893`
qualified before source edits: PR1967/29, main4047/66, Python231/2, golden16,
fitted ASan/UBSan18 and C++2018. Identical full13 after commands must add
exactly two tests: PR1969/29 and main4049/66. Existing selected, skipped,
ignored and excluded names are preserved. Amendment1 Clippy allows zero new
diagnostics; touched feat-math passes. Strict original O3 plus additional g++17
O2 emitter outputs, all71 frozen files and38 evaluator-identity inputs remain
byte-identical. No fixture regeneration, identity transition or dependency change.

This is a hard correction before E6 shipping search. Profile-aware shipping
offspring/generation integration, CLI flags, v4 receipts and fitted-C++ defaults
remain pending, followed by EEC clause(b) and B05. This library correction does
not qualify those capabilities. Lane-only audit, exact-head automatic/manual CI,
fresh independent native approval and guarded rebase merge are required.

Durable source/command/environment/time/exit/count/name/hash evidence, red
reproduction, after-gates and lifecycle readbacks:
`/home/ubuntu/jope-evidence/kernel2/bt/TRACK_E_06_PREFLIGHT_REPAIR/`.
