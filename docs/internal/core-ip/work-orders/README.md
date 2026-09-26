# Regression remediation work orders: W1–W12

Planning baseline: `c98037d2b151622145d8c43613ce36b24196bdac`, inspected 2026-09-26.
Status: **planned work, not implemented or qualified by these documents**.

These work orders expand the owner's regression remediation plan into executable
assignments. They preserve the accepted lineage-B model and distinguish defects,
unverified safeguards, historical unknowns and research questions. A completed
document, schema or configuration never closes an operational requirement.

Read [the execution contract](EXECUTION-CONTRACT.md) first. It defines task states,
evidence, approval gates, budget accounting and the independent Codex review lifecycle.
Read [the validation matrix](VALIDATION-MATRIX.md) before selecting a test lane.
[Findings coverage](FINDINGS-COVERAGE.md) retains every original gap/improvement ID;
the authoritative original text remains in the [findings register](../findings/register.json).

| Order | Work order | Main handoff | Required result |
|---|---|---|---|
| 1 | [W1 — inputs](W01-input-boundaries.tasks.md) | Acquired bytes → restricted worker | Immutable regression release and demonstrated forbidden-input isolation |
| 2 | [W2 — semantics](W02-model-encoder-semantics.tasks.md) | Accepted assets → typed context/policy | Authenticated semantic assembly, unchanged accepted calculation |
| 3 | [W3 — admission](W03-provenance-metrics-hwm.tasks.md) | Context/predictions → teacher/HWM | Mandatory transitive provenance and reconstructable admission decisions |
| 4 | [W4 — generator](W04-generator-ranking.tasks.md) | Context/history → proposal/rank | Qualified autoregression, total bounded decoding and honest scorer dependencies |
| 5 | [W5 — GP](W05-gp-fitting-selection.tasks.md) | Proposal → selected fitted predictor | Immutable ancestry, independent selection replay and enforced work bounds |
| 6 | [W6 — training](W06-training-checkpoints.tasks.md) | Frozen corpus → committed update | Exact input consumption, explicit objectives and resumable complete checkpoints |
| 7 | [W7 — three nodes](W07-three-node-orchestration.tasks.md) | Controller → admitted participants | Actual CPU/GPU work on all three nodes and qualified distributed updates |
| 8 | [W8 — S3](W08-durable-s3-progress.tasks.md) | Local committed results → reusable remote state | Durable outbox, conditional publication, independent restoration and resume |
| 9 | [W9 — export](W09-fitted-export-release.tasks.md) | Fitted subject → installed package | Python/Rust/C++17 package identity and exact-bit qualified inference |
| 10 | [W10 — learning](W10-learning-transfer-studies.tasks.md) | Updated policy → subsequent search | Real loop witness and separately scoped, honest research/transfer results |
| 11 | [W11 — performance](W11-gpu-performance.tasks.md) | Qualified requests → timing claims | Complete timing inventory and controlled, semantics-preserving optimization |
| 12 | [W12 — claims](W12-claims-revocation-replay.tasks.md) | Original evidence → publication | Revocable, independently replayable claims without evidence upgrades |

## Scheduling and dependencies

One implementation PR is active at a time. The table is the default serial delivery
order, not permission to accumulate twelve simultaneous branches. A package can
contain several reviewed PRs; its live gate may remain blocked after its source PRs
merge. Independent review is a separate Codex process under the owner's explicit
Claude-quota override; changing account in the writer process is not independence.

Use these dependency edges for implementation readiness:

| Package | Required predecessors | Later operational dependencies |
|---|---|---|
| W1 | Applicable W0 reproductions and trustworthy test execution | W7 supplies managed live worker isolation proof |
| W2 | W1 scope/release contracts | Hardware-specific parity is qualified through W7/W11 |
| W3 | W1 and W2 trusted producer contracts | W8 proves remote recovery of committed decisions |
| W4 | W2 bundle/context contract | W10 measures proposal and ranker utility |
| W5 | W1 and W4 candidate semantics | W7 proves distributed containment; W9 proves serving export |
| W6 | W2, W3 and W5 corpus/program identities | W7 supplies multi-rank proof; W8 supplies remote restore |
| W7 | W1, W5 and W6 execution contracts | W10 supplies the joined real loop witness |
| W8 | W3, W6 and W7 artifact/commit contracts | Scoped publisher credential and approved live storage budget |
| W9 | W1, W2 and W5 fitted subject contract | W10 links a real selected subject to its released package |
| W10 | W3–W9 relevant gates | Transfer requires separately approved costed manifests and exposure evidence |
| W11 | W1, W2, W4, W5, W7 and relevant W9 backends | Each changed optimization re-enters semantic qualification |
| W12 | W3, W8, W9, W10 and W11 qualified records | Historical unknowns can remain explicitly unresolved |

The later dependencies are integration tests of earlier contracts, not circular
prerequisites to authoring those contracts. For example, W1 can merge fixture-proven
isolation before W7 demonstrates it on all nodes; W1's live attestation stays empty
until that later execution. Do not substitute fixture success for that gate.

## W0 prerequisite register

PR109 delivered the 195-entry source inventory only. Before dependent behavioral
work, establish current reproductions for fit-authority and view-contract failures,
private-corpus fixtures, scratch failures/timeouts and native-neural SDK discovery.
The previous full diagnostic remains failed. The newly reproduced SDK check fails
inside the pinned runner image before compilation; repairing discovery is not a
native runtime qualification. Preserve classification failures in the broad diagnostic.

W0 must also supply the maintained regression test inventory and repair receiving
test authority through legitimate issuers. Never enable production fitting by copying
test-only activation logic into a public path. Records `S01-G2` and `S01-K2` retain
their W0 primary ownership in the coverage crosswalk.

## How to take an assignment

1. Select one proposed PR slice from one work order; refresh its source anchors.
2. Record its task IDs, source head, owner, receiving entrypoint, input scope and
   verification budget under the repository claim protocol.
3. Collect the applicable tests and reproduce the defect, or document that the
   question is a missing-evidence/research task rather than an observed defect.
4. Implement and verify through the actual consumer; keep valid and invalid cases.
5. Complete independent review, exact-head CI, guarded merge and remote verification.
6. Update task evidence links and each affected finding's scoped disposition through
   a reviewed register-schema change. Leave every unearned attestation empty.

All checkboxes start unchecked. No schedule date, live allocation, model promotion,
IAM change or positive scientific outcome is granted by these files.
