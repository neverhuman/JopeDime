# Runbook: admit scoped JopeDime research roles

## Status

`deploy/iam/jope-research-role-candidates.json` is a review candidate matrix only. It is not an
AWS IAM policy document, is not deployable, and does not show that any listed role exists. As of
this task, candidate roles are `not_provisioned`; live positive and denied-operation tests are
`NOT_EXECUTED`. The dated repository inventory records that the available host profile resolves to
an administrator principal and that append-only writer and curator credentials are not provisioned.
This task did not inspect credential bytes or refresh the dated account observation.

No provisioning, IAM calls, S3 calls, sealed-key access, or live role test is part of this runbook's
current execution. External provisioning and operational admission remain blockers.

## Credential refresh behavior found in source

`crates/jope-datasets/src/credentials.rs` calls `aws configure export-credentials --format env`
when a selected profile is resolved. It retains the access key, secret key, and optional session
token, but not an expiration value. `crates/jope-datasets/src/store.rs` resolves the profile while
opening an S3 `Store` and attaches that tuple through `StaticCredentialProvider`.
Reopening a store resolves credentials again, while an already-open store has no refresh path in
this adapter. The AWS CLI call also has no process deadline in this source. The current behavior is
therefore a credential snapshot per opened store, not automatic refresh during store use. This was
a source review only; no profile or credential process was invoked.

The candidate contract proposes short-lived credentials delivered by a refreshing provider that
retains expiration metadata. Before expiry, refresh once with a bounded deadline; retry the
authorized request at most once. On refresh failure or expired credentials, fail closed without
reusing stale credentials or falling back to ambient credentials. Diagnostics may report the
candidate ID, expiry boundary, status, and request ID, but never credential values. This contract
is a design proposal, not implemented or operationally verified here.

## Candidate responsibility boundaries

| Candidate | Intended allow boundary | Must remain denied |
| --- | --- | --- |
| actor | Read owner-approved training inputs; create its own proposal artifacts | Judge/dconfirm/sealed rows; accepted pointers; deletes |
| evaluator | Read nonsealed training inputs; create evaluation/executable evidence | Judge rows; accepted pointers; deletes |
| trainer | Read approved training inputs; create run-scoped checkpoints | Judge rows; accepted pointers; deletes |
| judge | Read only admitted sealed/dconfirm rows; create verdict receipts | Accepted pointers; deletes |
| publisher | Read signature-verified verdict receipt; conditional pointer update using reviewed If-Match | Judge rows; deletes |
| curator | Read catalog metadata; create approved append-only curated records | Judge rows; accepted pointers; deletes |

The JSON records symbolic bucket/prefix names and conditions. An owner must replace them with
exact account, resource, source identity, and trust-claim bindings and independently review the
result before any role exists. The publisher is the only candidate with pointer-write capability;
the judge is the only candidate with judge-row read capability. Resource-policy explicit denies
and permission boundaries should preserve those separations even if a later identity policy is
broadened. No candidate receives object deletion.

## Admission checklist for a later owner-authorized change

1. Confirm the exact source principal, repository/workflow/ref trust claims, target account, bucket
   ARNs, prefix owners, and session ceilings. Do not infer these from a profile name or prior dated
   inventory.
2. Translate the candidate matrix into separately reviewed IAM trust and permission policies.
   Keep actor/evaluator/trainer/judge/publisher/curator policies separate; keep judge-row access
   restricted to judge; keep accepted-pointer writes restricted to conditional publisher updates.
3. Require explicit `If-Match` pointer writes tied to a verified decision receipt. Exercise
   create-only conditions for append-only artifacts and confirm delete actions remain denied.
4. Provision through the authorized owner process, record role ARN and policy/trust document
   digests, and independently verify the installed principal before use. This task does not grant
   permission to run those steps.
5. Add a production credential provider that retains expiration and refreshes within its declared
   bounded deadline, or reopen stores at a controlled boundary. Prove refresh failure and expiry
   fail closed; do not treat the current static store snapshot as auto-refreshing.
6. Run positive and denied-operation tests only after a separate operational admission. Record
   actual caller identity, operation, resource, request ID, result, and session expiry without
   recording secrets. Include denied actor attempts to read judge rows and write accepted pointers.
7. Keep live test status `NOT_EXECUTED` until steps 1–6 are independently accepted. Offline fixture
   tests validate the candidate contract only and cannot clear the provisioning blocker.

## Offline evidence boundary

`cargo test -p jope-learning --test research_roles` reads the candidate JSON and exercises a fake
credential-expiry and logical-capability model. It does not load AWS credentials, invoke the AWS
CLI, contact IAM/S3, prove policy simulator results, or establish that a candidate role is installed.
