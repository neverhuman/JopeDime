# Phase 2 follow-up (plan only)

The current loop computes exact CPU landmarks, frozen DOPE representations and GPU
adapter costs, then fits proposed GP programs in the native CPU evaluator. It does
not use the fused Phase 2 code generation path. GP DSL tokens are not Rust, other
language or reasoning emission tokens. Those three rates remain unavailable.

1. Pin a merged Phase 2 source and artifact identity. Establish parity on the same
   allowlisted DL-Train tables, programs, vocabulary and two-fold training cuts.
   Exclude DL-Val from optimizer and mutation inputs, and exclude DL-Holdout/sealed
   data from the worker entirely. Preserve dataset labels and family assignments.
2. Implement a Rust-only code generation gate and bounded fit queue. Benchmark
   each phase separately, retaining actual generated Rust bytes, compilation and
   execution status, optimizer updates, transfer bytes and wall-time denominators.
   Compare end-to-end wall against the same native-evaluation budget. Keep CPU85,
   max64 native evaluations, max2 adapter updates, GPU90%, and round lock leases.
3. Prove frozen DOPE and anchor tensors bit identical, exact pool/promotion replay,
   resumed optimizer state and the same-cut evaluator outputs. A faster fixture
   alone does not qualify a live replacement.
4. Qualify the joint regression/binary path with task-correct candidate pools and
   heads. The approved 1219/91 corpus is entirely regression according to current
   S3 manifests; it cannot demonstrate binary learning. Use reviewed family keys
   and an authoritative sealed exclusion roster before admitting wider data;
   catalog-derived source-file family keys remain unreviewed.
5. Run the JRB-01 equal-budget joint/reg-only/bin-only study, three seeds, with
   separate regression and binary panel Elo and matched historical percentiles.
   Retain one joint model unless the specified two-evaluation >25 Elo interference
   threshold with confidence intervals supports splitting.
6. Only after reviewed parity and operational qualification may a controlled
   checkpoint-boundary rollout enable Phase 2. Record rollback checkpoint, source,
   pool journal prefix, resource headroom and task-separated quality curves.

No Phase 2 code, new split objects, benchmark fits or serving promotion is enabled
by this document. Golden45/panel Elo remains unavailable in this loop until a
same-protocol panel evaluation fits the approved resource budget.

Owner update t528u (regfirst_2250): train, select and judge regression only for now.
Headline regression validation, regression Elo and per-dataset/split percentile
trend; binary/JRB-01 work is deferred until sustained regression validation gains.
This supersedes the immediate joint-training priority above. No binary blend.
