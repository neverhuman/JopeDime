# Handling duplicate code findings

Jankurai's HLT-043 findings identify exact files or same-named units with matching
normalized source. A finding is evidence of duplication; it does not establish that
the copies have the same deployment boundary, runtime inputs, or maintenance owner.

## Remediation

1. Use the finding's paths and fingerprint to inspect every copy at the audited
   commit. Confirm whether each path is executable source, a generated file, a
   vendored package member, a test fixture, or immutable research evidence.
2. For maintained code with the same runtime boundary, choose one canonical owner
   and have the other call it. Keep behavior and validation in that owner; do not
   replace a copy with a wrapper that silently weakens its checks.
3. For separately delivered packages, prove that the canonical module is present
   in every package's import closure. Update package builders, source manifests,
   offline receipts, and import-closure checks together. A repository-relative
   import that works only from the checkout is not a valid package boundary.
4. Preserve historical source snapshots and their recorded hashes. If a historical
   or pinned copy must remain, document the exact paths, source identity, and reason
   in a narrow, reviewable audit exception. Do not suppress a directory, language,
   or whole finding class to hide unrelated duplicates.
5. Verify behavior through the owning package checks and confirm each packaged
   runtime can import the canonical implementation from its staged closure. Record
   the source commit, commands, and results with the change.

## Scope of exceptions

An exception explains a known copy; it does not make the copy disappear or prove
that the copied behavior is correct. Keep current mutable production copies visible
until consolidation is implemented and verified. Keep historical or independently
shipped copies visible unless their exact provenance and immutable boundary support
a path-specific exception. Recheck findings against a fresh audit after each change.
