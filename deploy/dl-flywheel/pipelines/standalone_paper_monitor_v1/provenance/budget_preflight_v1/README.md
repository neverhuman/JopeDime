# Pure pre-I/O refusing budget validator

Source/fake-case qualification only. `preflight.project(plan_wire, ExternalPlanPin(sha256, bytes))` checks exact closed input bytes and calculates one finite packet's read/retention/transport reservations. It opens no file, invokes no callback, executes no historical verifier and issues no authority. Oversized plans refuse; a small plan returns `WITHIN_BUDGET_NOT_ADMISSION` with `admission_authorized=False`.

The plan explicitly charges each distinct original path, all declared repeated reads and retained/serialized copies, metadata, full transport envelope and future output capacity/readbacks. Future outputs have no invented physical SHA/inode before creation. Limits remain8MiB body/1MiB metadata/32MiB reads/32MiB retention/48MiB transport. Arithmetic verifies only the provided independently pinned plan. Complete independent historical membership/policy/source/clock/identity authority, actual bounds and producer implementation remain mandatory external gates. No actual ledger capability or durable cross-process producer is claimed.

`QUALIFICATION_PLAN.json` aba552d7 was frozen before a single invocation of `run_offline.py`. All15 new synthetic cases passed, zero failures/errors/skips/expected failures. Tests include the concrete six-root35,674,138-byte scalar lower bound, but load no old body/capture or actual authority. The runner rejects fixture-roster/source drift and expected/unexpected failures, pins local import origins and uses create-only attempt/receipt. Completed fixture invocation must not be repeated.

Run command already executed once (not a request to replay):

`rtk proxy env PYTHONDONTWRITEBYTECODE=1 python3 work/owner-three-objectives-20261010-1358/producer-implementation/preflight-validator-v1/run_offline.py --plan-sha256 aba552d7cdd72d0596e8467c6c0a638a39263c410034d759775dc06e6792a202`

Independent review should inspect the exact new sources/receipt and the parent concrete roster. No tests, actual producer, old RAW, RootSeal issuance or runtime adoption is needed to review. Future actual implementation cannot call a projection success sufficient admission; an incomplete or unauthorized supplied roster stays unqualified even if arithmetic fits.
