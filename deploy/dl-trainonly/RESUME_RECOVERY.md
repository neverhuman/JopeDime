# Completed-context resume and recovery

A completed context (`.done`) is immutable. The T1, L7a/L7b and raw builders compare the recomputed
replacement record (`epoch0_replaced`, including built, admitted and refused folds) as well as the
existing identity and file bindings before counting it as already done. The dataset shortcut also
checks the completed contexts against its recorded replacement set. A missing context invokes the
normal build; stale metadata raises `done_context_replacement_differs` before any pending context
is written. The original completed contexts and their bytes stay intact.

The four-fold trees produced before the PR #251 membership correction can contain four completed
contexts that each advertise five replacements. They are historical, unpublishable trees. Do not
edit their metadata, remove `.done`, or splice them into a corrected publication tree.

To recover through the existing source-controlled builders:

1. Preserve the stale tree and its dataset failure receipt. Record its source SHA, plan digest and
   input bindings. Do not replace an existing manifest or reuse its publication success claim.
2. Create a **new, empty output tree outside Git**, in the owning lane's approved custody location.
   Copy the original frozen `plan.json` and any selected supplement plan byte for byte into it,
   using create-only writes, and verify their digests against the retained originals. Keep the
   same explicit split version, tree/scope, shard count, shard selection and dataset selection.
   Do not run the planner again against newer ambient inputs.
3. Use the corrected, pinned source to run the original builder invocation with only `--out`
   changed to that new tree. The three entrypoints are `tools/v2_rowsel_build.py`,
   `tools/v2_build_local.py` and `tools/v2_rowsel_raw.py`. The original immutable input loaders,
   train-only fold gates, counterfactuals, and per-context refusals all execute again. A changed
   plan/input or newly refused context is a new result, never a metadata repair of the old one.
4. Summarize the new tree through its owning builder. Build its publication copy through the
   existing payload-strip/publisher path, then run `tools/v2_rowsel_raw_manifest.py` with the new
   tree as `--source` and its pinned summary. Require the admitted fold set to be complete and
   source-bound. Keep the stale tree, its manifest refusal, and the new evidence as separate records.

Recovery uses the original campaign grants, work accounting and deadline; this procedure grants
no new compute or S3 writes. It never changes production artifacts in place. Matching completed
contexts remain byte-identical and publishable. An incomplete admitted set and a tree holding a
refused fold remain publication refusals.

The synthetic acceptance path is `tests.test_v2_partial_refusal_manifest.Resume`: it runs all three
real builders, retains the stale-tree refusal, rebuilds the same inputs into a new tree, and checks
that the unchanged manifest publishes exactly folds 0, 2, 3 and 4. It also covers matching resumes,
incomplete delivery and refusal before pending writes. Fake verified ledgers are fixture evidence,
not a live publication claim.
