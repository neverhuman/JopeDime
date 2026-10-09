# Native successor v5, cold-start revision 2

This local-only correction preserves the failed original v5 source, receipts and PR revision. It adds one shared `native_build.py` verifier to `native_dispatch.validate_build` and `operations_launch.inputs`. Neither GP execution nor a runtime clock has been started by this packet.

The actual S1 control receipt is `jope.dl-s1-control-receipt/v1`; `source` contains `pin`, not `plan_sha256`. The separate six-field `build-plan.json` is uniquely located in `source_pins` by `source_build_plan_sha256`. Its original 1119-byte compact JSON has SHA `7b200996636f3b8eead135dee30ca90339e9b5ca384299f64bc45c95c917346c`. The verifier checks those raw bytes before parsing, rejects duplicate/unknown fields, and binds source commit and all five descriptors to the current packet. A current descriptor may add the byte count omitted by the historical plan, but cannot contradict any recorded value. The independently hash-checked original receipt binds the source pin, honest-worker SHA/size and canonicalizer proposal SHA/size. Receipt s1-control paths may differ from equivalent s1-prod binary copies; only the plan-to-config paths must match.

The captured fixture contains the actual parsed receipt and plan. Receipt serialization is NOT claimed to reproduce the original receipt SHA. The plan serialization DOES reproduce the independently observed original plan SHA. Production callers read the original receipt bytes via their existing bounded hash checks. No receipt field is fabricated.

`native_evaluate.py` is byte-identical to original v5: its result metadata already uses `self.config['source_build_plan_sha256']`. A new test evaluates that exact AST metadata expression without GP, Rust compilation or network activity. Numeric calculations, fixed 1e-9 parity, leases, permanent global claims, unknown handling and original global deadline remain unchanged.

Validation: 22 affected local CPU fixture identities passed, zero skips (9 new build tests, 12 launch tests using the corrected schema, and the one affected dispatcher/evaluator contract fixture). The other historical fixtures were not replayed.

From this revision2 directory:

```sh
python3 -B -m unittest test_native_build test_operations_launch test_native_successor.SuccessorTests.test_evaluator_config_contract_and_exact_build_plan -v
```

Stage only after independent review. Include `native_build.py` in every worker's exact source roster and beside `operations_launch.py`; pin both helper bytes. Preserve the existing prepared census and permanent registry identity. Use a fresh create-only staging namespace and rerun bounded cold validation before freeze; never rewrite the failed v5 dry-run artifacts, start its old clock, or fabricate plan metadata in the foreign receipt. Root owns staging and dispatch. This revision does not authorize or prove a live launch.
