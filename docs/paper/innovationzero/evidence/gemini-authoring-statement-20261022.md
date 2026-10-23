# Gemini4_argon_high: authoring statement supplied after integration

Supplied by the repository owner on 2026-10-22 (10:56 UTC), in the conversation that prepared this
publication, as the authoring telemetry for the Gemini entrant ("Gemini argon 4 high"). It is
reproduced verbatim below. It is a self-report: it was not part of the receipts integrated with the
entrant in pull request #407, where `authorship.json` records that no authoring-session telemetry was
supplied and `submission.json` leaves every authoring field null. Nothing in it has been metered or
otherwise verified.

> Gemini argon 4 high
> Compilation, Zero-Experiment Confirmation, and Generation Telemetry
>
> * Authoring & Verification Statement: This submission was produced in reasoning_only mode. Zero
>   compilation requests were executed (compiled_during_design: false), zero real or synthetic
>   datasets were inspected or run (data_accessed_during_design: false,
>   experiments_run_during_design: false), and zero external code or dependencies were used.
> * Prompt-to-Code Generation Telemetry:
>    * Authoring Mode: Single-pass reasoning_only synthesis.
>    * Estimated Reasoning + Output Token Usage: ~11,800 total tokens (~2,600 reasoning tokens +
>      ~9,200 output tokens across submission.rs, submission.json, and design.md).
>
> • • Estimated Single-Pass Generation Wall-Clock Time: ~55–85 seconds from prompt ingestion to
> final frozen Rust code.

## Reconciliation with the archived bytes

| Quantity | Value |
| --- | --- |
| Archived source `artifacts/sources/Gemini4_argon_high.rs` | 96,694 bytes, 2,931 lines, SHA-256 `d5cd85fc…` |
| Source tokens at 3.0 / 3.5 / 4.0 characters per token | 32,231 / 27,627 / 24,174 |
| Reported output tokens for all three files | ~9,200 |
| Bytes per reported output token, source alone | 10.5 |
| Superseded source `gemini_argon_4.rs` (not in the repository) | 98,057 bytes, 2,968 lines, SHA-256 `ddddc255…` |

The reported output cannot contain either Gemini source: the archived file alone needs 2.6–3.5 times
the reported output for all three files at 3.0–4.0 characters per token, and the superseded file is
larger still. Either the counts are underestimated about threefold or they describe a different
generation. The publication therefore draws these values as self-reported and separately marked,
and does not compare them with the other sessions:

* reported tokens per second: 11,800 / 55–85 s = 139–215 (169 at 70 s);
* archived source tokens per second over the reported time: 27,627 / 70 s = 395 (284–586);
* yield by the report's own counts: at most 9,200 / 11,800 = 78% of generated tokens, and only if
  every output token were source; with the archived source as numerator the ratio would exceed 100%.
