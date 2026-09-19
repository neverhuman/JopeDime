# Runbook: rotate credentials

When: a token was exposed (pasted into a commit, a log, a shared host), a person with access
leaves, or on the schedule below. Two credentials are due **now** (public-readiness checklist §4):
the GitHub PAT that lived in `gh.env` and the static AWS key that lived in `aws.env` in the
historical tree. Rotation is an owner action; this runbook is what the owner runs.

Principles: revoke *before* you clean up (a burned secret is burned even if the commit never left
your machine); prefer short-lived credentials (OIDC, `gh auth login` device flow, SSO profiles)
over anything stored in a file; never commit the receipt of a rotation with the secret in it.

## 1. GitHub PAT (`gh.env`)

1. Revoke: <https://github.com/settings/tokens> → the token → *Delete* (fine-grained tokens:
   *Personal access tokens (Fine-grained)*). Do this first.
2. Replace the file: `gh` itself needs no PAT for interactive use —
   `gh auth login --hostname github.com --git-protocol https --web` (stored in `~/.config/gh/hosts.yml`,
   never in the repository). Verify: `gh auth status`.
3. If an automation genuinely needs a token, mint a **fine-grained** one scoped to
   `veox-ai/JopeDime` with the minimum permissions (`contents: read` for a mirror, `administration: write`
   only for the runner's `generate-jitconfig` — that one lives root-0600 on xbabe2 and is never
   mounted into a container), expiry ≤ 90 days, and record its *name and expiry* (not the value)
   in the checklist sign-off table.
4. Confirm nothing references the old file: `git grep -n 'gh\.env'` in the repository is empty
   (only ADR-0001 and this runbook mention the name); `ls ~/jain-split/jope-dime/gh.env` on xbabe2
   should not exist.

## 2. AWS static key (`aws.env`)

1. Identify the key id: `aws iam list-access-keys --user-name <user> --profile veox-jepson`.
2. Create the replacement *only if a static key is still needed*. It is not for CI (OIDC roles
   `JopeDimeCiRead`, `JopeDimeTrainer`, `JopeDimeRelease`) and not for humans (the
   `veox-jepson`/`veox-alton` profiles use `credential_process`). If it is: `aws iam create-access-key`,
   store it in the credential process, never in a file in a repository.
3. Deactivate then delete the old key: `aws iam update-access-key --access-key-id <old> --status Inactive`,
   wait for one CloudTrail window to confirm nothing breaks, then `aws iam delete-access-key --access-key-id <old>`.
4. Verify the OIDC path still works without any static key: trigger `main.yml` (or the WP0
   dry-run workflow) and check `aws sts get-caller-identity` shows the role ARN.
5. Check CloudTrail for the old key id over the exposure window
   (`aws cloudtrail lookup-events --lookup-attributes AttributeKey=AccessKeyId,AttributeValue=<old>`);
   anything unexpected is an incident, not a rotation.

## 3. Self-hosted runner registration token (WP8)

The runner registers with JIT config (`POST …/actions/runners/generate-jitconfig`) using a
fine-grained PAT with `administration: write` on the repository only, kept root-0600 in
`/etc/jope-runner/` on xbabe2. Rotate it every 90 days: mint the new token, replace the file
atomically (`install -m 0600`), restart `jope-runner@.service`, revoke the old token. The runner
itself is ephemeral (one job per registration), so no runner state carries the token.

## 4. OIDC role trust and bucket policies (no secret, but rotate the *binding*)

If a workflow file is renamed or a branch/tag pattern changes, the trust policies' `sub` claims
(`repo:veox-ai/JopeDime:ref:refs/heads/main`, `…:ref:refs/tags/v*`) must be updated in
`deploy/aws/` (landing with WP8/WP11) and applied with the `veox-jepson` profile; every call is
receipted. Roles never get `DeleteObject`/`DeleteObjectVersion` (bucket policy denies it to all
but `JopeDimeMaintainer`).

## 5. After any rotation

- Re-run the history scan on the current repository (ADR-0001 step 5 regex) and
  `gitleaks detect --redact --config .gitleaks.toml --source .`.
- If the secret was ever committed, even on a branch: it is in the reflog and possibly in a
  bundle — rotate (done above), then remove the commit from every branch, `git gc --prune=now`, and
  note the exposure in the checklist. Do not rewrite `main` history after the first push; the
  rotation is the fix, the history rewrite is cosmetic.
- Update the sign-off row in [public-readiness-checklist.md](../public-readiness-checklist.md)
  with the date and the credential's *name*.

Schedule: fine-grained PATs ≤ 90 days; static AWS keys — none should exist; runner token 90 days;
review the OIDC trust policies whenever a workflow's trigger or name changes.
