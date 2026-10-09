# Trace recovery3 and cold input preparation: source integration boundary

This source-only addition preserves the existing687 wired fixture identities and adds34 author-qualified synthetic fixtures, for721 wired identities. The author ran only these34 fixtures. Integration does not claim a721-test execution, live archive qualification, new data admission or training improvement.

## Portable source and fixtures

`pipelines/telemetry/trace_recovery_v3/` includes six exact Python source files, the frozen template and source packet, synthetic tests, encoded historical test output/source delta and pin-only custody descriptors. Previous base and recovery2 sources remain byte-identical. Multi-megabyte `prior1-capture.json`, `prior2-capture.json` and the deployment `terminal-proof.json` bodies are deliberately absent. Their exact hashes/lengths remain in `custody-inputs.json`; root stages them separately for any authorized actual preflight. No credentials, raw training payloads, checkpoints or live logs are included.

The isolated runner copies this source into an owned temporary directory, enforces exact import origins and hashes and the exact34-name fixture roster, and requires assertions enabled. Only `time.time` is fixed to1791564000 inside that fixture process; `time.time_ns`, monotonic and CPU clocks stay real. Production sources and their original deadline are unchanged. `ColdJoin` injects an already-verified prior1 result to isolate the new join; this is not proof the full old verifier was rerun. Launcher subprocess/S3 clients are mocked.

To execute only the new packaged fixtures, from the repository root run:

```sh
rtk proxy python3 -B deploy/dl-flywheel/run_trace_v3_offline.py --output /tmp/trace-v3-new-proof
```

Use a new output directory. This emits a receipt with34 exact identities and a runtime log outside the tracked tree. The all-suite runner appends this suite only after the unchanged687 checks; old suites were not rerun during this integration.

## Archive execution remains separately qualified

Archive v3 retains all1213 historical PUT attempts,606 verified segments/1212 objects and permanently held unknown part153. It starts at607 with3787 remaining attempts; unchanged global deadline1791592146.380285 and total5000 PUT/2500-part ceiling apply. A new one-hour window cannot extend that global deadline. Actual source/capture readback, full prior verifier/raw chain replay, all old units physically closed, current resource authorization and one-use cold preflight are required. A failed or unknown attempt is preserved; it is never an automatic retry authorization.

Consult the exact package README for operational commands. Its initial status and SOURCE_PACKET are historical local-qualification records. A separate independent review receipt, when present, supplies later review status; neither proves a live launch or fresh cloud verification.

## Separate CPU/GET-only preparation helper

`pipelines/fresh_gp_v2/operations/cold_prepare_controller.py` is intentionally outside the unchanged fresh-GP revision3 runtime directory. Its independent source review is retained under `operations/provenance/`. It bounds a new cold-input preparation attempt to one CPU,512MiB RAM, no swap/GPU and at most600 seconds under the same global deadline. It checks exact sources/request/Python, current host pressure and exact own-unit identity before any allowed pressure retirement. It may perform exact-version GETs and read-only canonicalization; it does not fit GP, PUT objects, create claims/brokers/epochs or admit training data.

512MiB is a cap, not a memory qualification or host reservation: an honest OOM, resource stop or unknown launch preserves partial evidence and cannot become readiness. Successful process exit and verified prepared closure remain separate gates. This helper does not release native pool/resource HOLD, authorize the old4GiB scoring envelope or permit controls of foreign services.
