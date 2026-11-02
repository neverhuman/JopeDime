---
code: HLT-034-CI-BAD-BEHAVIOR
owner: ops
reason: >-
  The workflow-command detector reports self-hosted labels and receipt-validation code as
  privileged Docker or secret-bearing artifact paths. The admitted jobs run inside the non-root,
  read-only Jope runner container with all Linux capabilities dropped and no Docker socket mounted;
  deploy/runner/job-policy.py rejects workflows and events outside its fixed allowlist. The
  referenced artifacts are target/ci-* evidence containing source SHA, run identifiers, status,
  and test counts; they contain no credentials. These controls and exact-head receipt checks are
  retained unchanged.
expires: 2026-12-02
migration_plan: >-
  Keep the runner boundary and receipt validation. Re-review when the detector can distinguish
  source-bound receipt metadata from credentials and can inspect the runner's deployed isolation
  policy. Remove this record if a future detector proves a real exposure.
proof_lane: >-
  actionlint -color; python3 deploy/ci/test-runner.py; deploy/runner/run.sh and
  deploy/runner/job-policy.py security review; inspect .github/workflows/ci.yml and
  .github/workflows/full-main.yml receipt outputs and upload paths.
---

This is a documented detector exception for current findings in `.github/workflows/ci.yml`
and `.github/workflows/typescript.yml`. It does not change runner admission, triggers, required
checks, upload paths, or receipt contents. The workflow findings remain visible in the full
Jankurai report until the detector supports these controls.
