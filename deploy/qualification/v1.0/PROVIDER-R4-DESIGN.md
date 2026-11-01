# REL-07 R4 reconnect provider

R4 requires a terminal closed during a run to leave the engine running, then replay the same durable events on reattach without gaps or duplicates. The provider runs the maintained native probe embedded in `crates/jope-appliance/tests/system/live_driver.rs`, with `--r4-only --capture-sse`, against the supplied candidate itself. It does not rebuild a replacement appliance or import historical PASS receipts. Only the R4 leg runs; the S2 and crash matrices remain their owners' gates.

The command is `python3 -B deploy/qualification/v1.0/providers/r4.py`, with the standard five `JOPE_REL07_*` bindings in [GATE-RUNNER.md](GATE-RUNNER.md). It verifies the full non-development package and its source pin through the shared `jope-dist inspect`/`verify` helper. Before and after execution it checks candidate bytes and the exact clean source checkout. Evidence and the result must be new and outside that checkout. Symlink/non-file candidates, existing outputs, missing or malformed reports, mismatched identities and failed assertions refuse PASS.

Three measured assertions count toward the existing `jope.rel07.gate-execution/v1` result: terminal closure leaves the run running and reattach completes; the retained MCP/CLI journal has one contiguous sequence through the final watermark for the same run; and the reattached terminal's actual SSE events equal that journal, with the same rendered champion. Report flags alone cannot pass: the provider also reads the retained event lists, terminal streams and screen, checks the candidate/source identities and requires clean probe cleanup. Counts are exactly three collected/executed/passed and zero failed/skipped/excluded only after all checks succeed. Nonzero probe exits preserve their actual exit and logs without publishing a passing result.

The probe uses the existing admitted 96-row real-S3 TRAIN fixture, pinned by the maintained driver to full TRAIN SHA-256 `d9a0b97e91ff9d04bc87fefa4671a49c40cee21dfac05be58a7fc7ce08045931`. Normal internal assessment comes from those TRAIN bytes; no external holdout, sealed data or new data download is involved. CUDA is disabled and math threads are one. Run under the controller's nice10/CPU admission and four-job environment; this provider adds no speed verdict or signing operation.

The candidate runs in an owned short directory below `~/.local/state/jope-r4/` to avoid Unix-socket path limits. The new `<result>.r4/` directory retains its location, driver, observer, command output, event/frame/PTY metadata and report. Runtime stores remain outside the checkout and are retained for the owning qualification lane; private daemon credentials are not copied into logs. Probe timeout/interruption stops only its newly created process group and refuses PASS.

Acceptance for the controller is data-free:

```sh
nice -n 10 python3 -B -m pytest -q -p no:cacheprovider deploy/qualification/v1.0/test_provider_r4
```

These tests simulate candidate inspection and probe responses, including identity drift, failed/empty/malformed/replayed observations, event gaps and duplicates, cleanup failures, evidence overwrite and timeout cleanup. Their synthetic candidate cannot qualify a release. A real R4 gate still requires an owner-approved full candidate and this command actually executed against those exact bytes. Existing INT-02 platform receipts remain separate source-pinned observations. No claim is made for the other 27 gates, learned guidance, native platform coverage beyond the executing host or release approval.

A source-only Rust fixture wrapper was rejected because it would not exercise the supplied binary. A new reconnect implementation was rejected because the existing live PTY/MCP/SSE probe already exercises the product path and has retained operational evidence. No public API, worker/model contract, common provider helper or runner schema changes are needed.
