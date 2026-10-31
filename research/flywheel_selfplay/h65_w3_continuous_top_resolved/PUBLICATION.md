# W3 publication preparation — ASTRA t2155

H65 is already reserved and implemented by API06 (PR #796). This supplement adds the [publication manifest](PUBLICATION_MANIFEST.json); it does not duplicate the CPU runner, select a W1 winner, or claim an executed W3 study.

The manifest freezes the existing W3 arms, two primary contrasts, seeds701/702/703, five family folds,40 terminal updates, original utility semantics and2GPU-hour/zero-native-fit ceiling. It keeps W1 winner configuration, H59 outcome disposition and completed-status-clean W3 input export explicitly unresolved. A configured runner and a training-ready study are separate states. No model/data-payload work was performed for this supplement.

ASTRA independently checked the W1 source-manifest SHA and metadata consistency:1,219 distinct contexts and205 distinct families partitioned into five disjoint held folds, with pilot contexts disjoint from held contexts and fit+held counts consistent. Physical-table alias detection is the upstream manifest's reported result, not an independently rerun dataset audit. The metadata may be reused for H65 preparation; it does not establish admissibility of missing outcome rows or inherited model assets.

The existing catalog-v246 train-blob smoke is plumbing evidence only. It supplies neither W3 completed-status provenance nor a winner configuration. No new dataset key is admitted here. `PUBLICATION_MANIFEST.json` is not suitable as `--w1-winner-config`; that argument must bind an actual accepted winning recipe once one exists.

Following H62's corrected result (PR #799), every matched W1/W3 arm and continuity control retains current live slate-centering, with exact slate sizes frozen at execution. This records an accepted measured dependence, not an invariance claim. An anchor-reference intervention belongs to a separate matched comparison.

Before a result PR, bind exact input keys/versions/hashes, original row and evaluator identities, the accepted W1 configuration, all intended/completed/failed cells, cumulative budgets and one terminal checkpoint per cell. Report family Elo, raw `val_reg_inner`, and verified U4 on its separate track. Keep the common adjusted family gate; no auxiliary-loss gain or post-hoc endpoint switch establishes advancement. Preserve outputs create-only in the campaign W3 prefix and verify downloaded SHA256s.

This supplement contains no new scientific outcome. All outcome metrics remain null with a reason. No GPU lock, native fit, HWM candidate, serving change or new holdout access occurred.
