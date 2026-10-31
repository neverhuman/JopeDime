# Release executable exclusion check

QLT-03 requires the ordinary release build to omit `research`. The existing
`ordinary_build_has_no_research_help_or_dispatch` Rust test covers the parser.
This companion check executes Cargo's actual optimized, non-test executable.

The acceptance command uses default features, with no `research` feature:

```sh
export CARGO_BUILD_JOBS=4 CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only
nice -n 19 cargo build --release --locked -p jope-appliance --bin jopedime \
  --message-format=json > /path/to/evidence/release-build.jsonl
nice -n 19 python3 -B crates/jope-service/src/research/test_release_cli_check
nice -n 19 python3 -B crates/jope-service/src/research/release_cli_check \
  --cargo-messages /path/to/evidence/release-build.jsonl \
  --output /path/to/evidence/release-cli.json
```

Use an existing evidence directory outside the checkout and the lane's reused
target directory. The output receipt must be new. Retain the build command,
exit status and source SHA alongside it. Run only within the host's CPU budget.

The check rejects failed/ambiguous Cargo builds, test/debug artifacts and
research-enabled or non-default builds. Both help forms must omit `research`;
`research --help` and `research record --out ...` must exit 2 with Clap's unknown
subcommand diagnostic, before package/backend activation, and create no record.
The receipt contains the Cargo artifact, executable SHA256 and all four process
outputs. Negative controls ensure package/backend failures cannot masquerade as
command absence. These controls are software fixtures, not scientific data.

This closes only release-command exclusion for the tested feature/profile and
source. Explicit research-feature builds, signed distribution packaging and
positive corpus recording retain their separate acceptance requirements.
