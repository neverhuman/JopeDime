# Local custody and future worktree retirement checklist

Task: **I53 — Plan safe local worktree retirement separately**. Intended repository path: `docs/plans/algorithmic-selfplay-20261017/LOCAL_CUSTODY_CHECKLIST.md`.

Source pin C=`1a8a66c502ca2cede77c197b4f5b5bbc24ff398e`; tree `1c6d5882718434ae6f109854bff4597896f1210f`. Exact direct dependencies: I01, I50. Their implementation/acceptance is not inferred from prepared packets. Completion criterion (verbatim tasks.json): Specific retain/remove evidence for future owner action; no deletion authority created by a task checkbox.

This is the complete staged document contract/plan for the assigned scope. Source facts are tied to the verified C citations below; proposed behavior, owner bindings and future proof are explicitly distinguished. No Git worktree, source, Git state, live service or canonical ledger is changed by this document; no Cargo command or product experiment is executed.

## Standing dispositions

| Subject | Disposition | Evidence boundary |
| --- | --- | --- |
| Retired exp-next-02 | HOLD / UNPROBED | No existence check, lstat, traversal, recreate, restore-in-place, build or process probe is authorized by this checklist. Preserve its prior retirement record. |
| `/home/ubuntu/jopedime-paper-wt` | HOLD | The old paper tree remains protected; no inspection, release, move or retirement is inferred from its age or a merged commit. |
| Any symlink component, folder symlink or contained protected link such as .venv | HOLD | Do not traverse the link to discover a deletion target. Use only an already admitted inventory; unresolved link identity is itself a hold. |
| Active, dirty, private, ignored, foreign, locked or ambiguously owned contents | RETAIN | Process inactivity, clean tracked status and directory age do not release ownership or filesystem obligations. |
| A merged source commit | SOURCE SUCCESSOR ONLY | Git reachability and full filesystem recovery/consumer closure are separate proofs. |
| target/ and apparent Cargo caches | RETAIN UNTIL CLASSIFIED | Reports, receipts and evidence can occur at any depth, including debug, release and tmp. Cargo markers are not disposal authority. |

These are owner constraints and document decisions, not the result of a new filesystem survey. This staging task does not probe either protected tree or any candidate tree.

## Decision rule

A proposed removal is eligible for a future owner decision only if every gate below has independently verified evidence, all identities remain current, and the owner admits the exact concrete action. Any UNKNOWN, symlink, protected content, stale generation, missing recovery object, unresolved consumer or missing owner release yields HOLD. “Eligible” does not execute or authorize removal.

| Gate | Required record | Refusal |
| --- | --- | --- |
| Ownership | Stopped full HEAD/tree/index/dirty state, checkout owner, branch, lease release and exact path scope under the existing claim protocol. | An idle session or old claim is not release; overlapping owners remain held. |
| Safe path identity | Admitted no-follow inventory of every component and contained link, mount, hardlink class, device/inode and supplied versus resolved path. | Any symlink component or protected contained link, including .venv, is HOLD. Never substitute the link destination. |
| Specific successor | Exact merged successor and remote readback, with the required source/evidence mapped to it. | A merged branch name or a matching source tree does not prove all local state is saved. |
| Git-only uniqueness | Future read-only git status, git stash list and git log --all --not --remotes; compare local-only tags with git ls-remote --tags origin before calling tagged commits unsaved. Include detached HEAD, refs/reflog obligations, staged index and untracked/ignored source. | The branch-only log form misses other refs. A --all bundle can omit a detached HEAD; verify each required commit explicitly. |
| Git preservation | Only when the exact candidate is actually proposed for removal: preserve required refs/objects in a named custody bundle, hash it and prove the required commits are included. | A checksum or git bundle verify alone does not establish that the desired commit exists in the archive. Do not bundle retained trees merely to consume custody space. |
| Whole filesystem preservation | Preserve tracked, untracked, ignored, private and administrative files, index/locks/config, permissions/ownership, symlink text, xattrs/ACLs, hardlink equivalence, sparse-file and mount-dependent obligations, plus every retained report/receipt. | A Git bundle cannot preserve this closure. Unknown special files, credentials/private material or inaccessible contents stay held with restricted custody, not copied into public logs. |
| Independent restoration | Restore exact Git objects/refs into the existing independently provisioned bare verification facility and restore filesystem obligations into a separate admitted custody location with the original unavailable. Check connectivity, commits/trees, bytes, metadata and declared runtime restoration; no alternates or dependence on the source. | Same bytes in the original tree or a backup dependent on it is not independent recovery. Do not create a new clone/worktree or overwrite a protected path to test recovery. |
| Current consumers | Owner-admitted fresh process/open-file, service/timer, mount/namespace, configuration, scheduled future, foreign-user and downstream path-consumer closure. Record explicit negative evidence and its time. | “No process now” does not rule out a configured future consumer. Unknown consumers block retirement. |
| Fresh revalidation | Recheck path identity, owner release, refs/index/dirty state, recovery manifests and all current consumers immediately before the approved operation. | Any change since capture invalidates eligibility. No automatic budget/deadline or approval reset. |
| Exact action | Owner records the one candidate, normal nonforce git worktree remove invocation, original response and resulting registration/state evidence. | No force, prune, reset, clean, manual directory deletion, prefix operation, symlink deletion or opportunistic adjacent cleanup. |

No commands in this checklist were run against a retirement candidate. The source contract's independent bare recovery procedure is a future custody step through the existing owner; it is not permission to create an extra working copy.

## Evidence capsule for a future owner decision

Record decision ID and UTC; owner/reviewer and stopped-head release; candidate's opaque admitted inventory reference; protected-link/mount status; Git objects/refs/index and filesystem inventory digests; specific successor and remote/tag readbacks; bundle membership and independent restoration receipts; metadata/hardlink/xattr checks; target/evidence classification; process/mount/configured-future consumers; unresolved gaps; exact proposed operation; approval scope and expiry; original execution/terminal receipt if separately admitted.

Keep public manifests limited to safe locators and hashes. Preserve private material under its existing access control. An unknown field is UNKNOWN with an owner/blocker, never a favorable zero. Failed/inconclusive recovery and old custody reports remain retained.

## Review fixtures and required outcomes

| Supplied documentary fixture | Required result |
| --- | --- |
| Clean tracked Git, but one path component is a symlink | HOLD, without following the link or deriving a removal target. |
| No component link, but inventory lists a contained .venv link | HOLD; do not delete a containing folder to evade the rule. |
| Successor commit merged, but private untracked report or index state lacks restoration | RETAIN; source merge does not close filesystem custody. |
| Bundle verifies, but the stopped detached commit is absent | HOLD; preserve and independently verify the missing commit before reconsideration. |
| Restored bytes match, but xattr/hardlink or future timer consumer is unknown | HOLD with the exact unresolved obligation. |
| A report mentions exp-next-02 or the old paper tree | Preserve UNPROBED/HOLD disposition; do not turn a report reference into a probe request. |

These are acceptance specifications for reviewing this document. No physical recovery, removal, host scan or live consumer proof is claimed. The report-only maintenance tool rejects --delete; its dated inventories and reclaimed-byte estimates cannot establish current deletability. Preserve the host rules: extra lanes only in the repo's .agent/worktrees, one writer each, at most four under this repository's tighter rule, and no worktrees/build output in /tmp.

## Authoritative acceptance coverage

| Verbatim tasks.json acceptance case | Required evidence / refusal | Current evidence ceiling |
| --- | --- | --- |
| Symlink component/contained .venv link is HOLD | A supplied inventory containing a symlink component or contained .venv link must yield HOLD without traversal, deletion or replacement. | SPECIFIED; implementation/runtime NOT_EXECUTED |
| Merged commit alone does not prove filesystem disposable | A merged commit without independent full filesystem and consumer closure must remain RETAIN; a Git-only bundle cannot stand in for filesystem recovery. | SPECIFIED; implementation/runtime NOT_EXECUTED |
| Retired exp-next-02 remains unprobed; old paper tree remains held | Record exp-next-02 as unprobed and the old paper tree as held; checklist validation must not touch either path or claim release from historical status. | SPECIFIED; implementation/runtime NOT_EXECUTED |

The case table is a review/implementation contract, not a manufactured runtime result. Future proof retains exact source/input/argv/redacted environment/resource/UTC/exit and actual collected/executed/pass/fail/ignored/filtered counts at `/home/ubuntu/jope-evidence/spec-integration-20261017/work-packets/evidence/I53/<full-head>/`. Missing capabilities and unknown counts do not pass.

## Verified source citations

| Full-pin source citation | Verified boundary |
| --- | --- |
| `1a8a66c502ca2cede77c197b4f5b5bbc24ff398e:AGENTS.md:34` | Stopped-head ownership and handoff |
| `1a8a66c502ca2cede77c197b4f5b5bbc24ff398e:AGENTS.md:108` | Preservation, bundle membership and independent recovery |
| `1a8a66c502ca2cede77c197b4f5b5bbc24ff398e:AGENTS.md:121` | All-ref local-only commit inventory |
| `1a8a66c502ca2cede77c197b4f5b5bbc24ff398e:AGENTS.md:418` | Custody, managed lane closure and evidence preservation |
| `1a8a66c502ca2cede77c197b4f5b5bbc24ff398e:deploy/maintenance/README.md:3` | Report-only candidate selection, no symlink traversal, --delete refusal |
| `1a8a66c502ca2cede77c197b4f5b5bbc24ff398e:deploy/maintenance/README.md:12` | Evidence at every target depth; preservation and consumer gates |
| `1a8a66c502ca2cede77c197b4f5b5bbc24ff398e:MASTER_CLEAN_UP.md:70` | Historical symlink-path worktrees remain protected |
| `1a8a66c502ca2cede77c197b4f5b5bbc24ff398e:docs/plans/algorithmic-selfplay-20261017/tasks.json:2555` | Exact task dependencies, completion and acceptance cases |

These exact lines, line hashes and whole-blob SHA-256 values are recorded in `wave3-staged-docs-receipt.json`. Parent paths, feature/test declarations and new symbols proposed by the packet must be rebound to the actual accepted implementation head before use; C is an inspection pin, not an assertion that future code exists.

## Integration boundary

agy (`2830191.jopedime_agy`) owns the one integration PR, accepted dependencies, source/Cargo/schema/profile/root hooks, all seven identity/guard surfaces, coverage/inventory/README and claims. This stage adds only the requested document/configuration candidates. It moves no evaluator/lineage/frozen/witness/Clover/retention/discovery surface and grants no source scope, deployment, cleanup, campaign, serving or scientific authority. Preserve accepted readers/parent and every failed/rejected/partial/inconclusive receipt; disable an additive candidate rather than reset shared history or destroy evidence.

Execution grant (verbatim tasks.json): NONE; planning is not campaign admission
