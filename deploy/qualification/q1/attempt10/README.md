# QLT-04 one infrastructure retry

Authority: owner nudge `qlt04_fix_attempt10_t2345.md`, SHA256
`b7493bf59d94a3855c487fd1453255a02398f01dd190f8a05e285a36ed326747`.
This directory does not authorize a launch by itself.

`controller.py` is the concrete caller of `q1_dispatch.dispatch_units`. It
persists a driver failure and aborts before dispatching another unit on the
first nonzero backend exit, timeout, cleanup failure or invalid response. Its
driver failure envelope is deliberately not a Q1 response, and no failed unit
receives an invented charge or score. The unchanged analyzer stays strict.

The retry preserves the original signed scientific plan, original backend
`8288835b…`, evaluator `4e9239a1…`/source `ccaf14b2…`, and request hashes.
Changing the plan's scientific source fails the unchanged evaluator binding;
changing the plan digest would prevent reuse of the 60 historical responses.
The repaired server is separately frozen in a runtime amendment at its actual
new source SHA. Its research-only `q1_infra_retry` example verifies and uses the
original numerical worker. The modified structured backend is maintenance code
for future versioned plans; it must not replace `8288835b…` under the old hash.

Before launch, the owner supplies an immutable
`jope.q1-attempt10-runtime-amendment/v1` JSON receipt with:

- `decision: ADMIT_ORIGINAL_IDENTITIES_RUNTIME_REPAIR`, `budget_reset: false`,
  `valid_unit_reruns: false`, actual `runtime_source_commit`, a fixed
  `fixed_deadline_unix`, and a new private `output` directory.
- `artifacts` references `{path, sha256}` for `authority`, `gov03_review`, `plan`,
  `signature`, `allowed_signers`, `backend`, `original_panel`, `selection`, and
  `engine_receipt`. GOV03's receipt must contain `reviewed_head` and
  `verdict: PASS`; a preliminary or compile-blocked review is insufficient.
- `engine_endpoint`, a private `local_daemon_home` containing the authenticated
  tunnel descriptor, and `monitor_status`. The pinned engine receipt records
  `status: ENGINE_READY`, `runtime_source_commit` and `evaluator_sha256`.
- A fresh live monitor status, atomically updated at least every 60 seconds,
  recording `sampled_unix`, `runtime_source_commit`, `remote_load` and
  `remote_free_bytes`. The launcher must verify the exact daemon PID/executable
  and original numerical worker hash, and retain ownership of daemon/tunnel
  cleanup on success or failure. Missing/stale monitor evidence stops dispatch.

The driver retains its FIFO lease after dispatch while it writes
`CLEANUP-REQUESTED.json`. The owning launcher must stop/drain its exact daemon,
workers and tunnel, then atomically publish `RUNTIME-CLEANUP.json` in the new
output directory with `amendment_sha256`, `engine_receipt_sha256`, `daemon_pid`,
`completed_unix`, `daemon_exited: true`, `workers_remaining: 0` and
`tunnel_exited: true`. It must derive those facts from live process readbacks;
they are not assumed from a completed client. Cleanup must precede the lease
handoff, including on fatal backend errors.

The controller takes the existing qualification FIFO lease and creates one
separate `BUNDLE.attempt10.started.json` marker without changing historical
markers. An existing retry marker is a refusal, not permission to resume or
reset anything. The 522 failed identities are selected from the immutable
original terminal panel `f874cae1…`; the 60 valid identities are never dispatched.
Original failures and unknown physical charges stay in custody.
Dispatch order is reconstructed from the signed plan, including reversed policy
order on odd seeds. The archived logical panel's sorted order is not a dispatch
schedule.

Only after all 522 new responses are durably returned does the controller
assemble a separate logical panel. It copies historical valid rows verbatim,
and replaces failed logical finishes with new receipts having the exact same
scientific request identity. Original raw ledgers are never edited. The result
is `TERMINAL_UNANALYZED`, not qualification; the unchanged analyzer and owner
default decision remain separate steps.

The owning launcher runs the source-reviewed controller with
`python3 -B deploy/qualification/q1/attempt10/controller.py --admission ABS_PATH`.
No launch or execution success is implied by the existence of this source.
