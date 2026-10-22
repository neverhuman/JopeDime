# Operations instructions

Read the [root contract](../AGENTS.md) before changing this directory. It owns the
checkout, claims, credentials, source review, CI, custody and merge rules.

## Owns

`ops/ci/` contains the shared CI entry points and full-source audit wrapper.
The command implementation and receipt validators live in `deploy/ci/`;
GitHub scheduling and transport actions live in `.github/`. Keep local and hosted
execution tied to those same commands. `ops/git-hooks/pre-push` is an optional local
hygiene check, not a substitute for repository protection or exact-head manual CI.

## Boundaries

Claim exact paths on the locked coordination board before writes. Preserve other
owners' source, live board appends and untracked files. Changes here do not authorize
installed runner/admission changes, host services, CI storage, credential provisioning,
cloud policies, foreign job cancellation, research execution or model promotion.
Do not edit generated evidence, frozen learners, numerical fixture identities,
`target/health/`, or retained audit inputs to improve a reported result.

Keep PR and trusted-main storage/resource domains separate. No PR job receives a
Docker socket, runtime sudo or long-lived cloud credentials. A skipped workflow is
not an executed audit. Missing tools and failed reports remain failures.

## Proof and handoff

Start with [testing.md](../docs/testing.md) and [CI replay](ci/README.md).
For audit changes, run `python3 -B deploy/ci/test-audit-lane.py`;
for trusted-main routing, also run `python3 -B deploy/ci/test-full-main.py`.
Use `bash -n`, ShellCheck and actionlint for changed shell/workflow source.
Run the pinned `just audit` only after committing product changes; preserve its
fresh report, original exit, source manifest and input hashes whether it passes or fails.

Full manual CI and head-bound source/security review are required before merge.
At handoff, record the actual source, commands, run IDs/attempts, results, retained
evidence and unresolved findings. The default audit floor stays 85. Input-integrity
validation, a policy pass, a merged change and live qualification are separate facts.
