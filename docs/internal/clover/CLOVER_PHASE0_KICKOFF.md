# CLOVER Phase 0 kickoff (coding-agent task)

Source of direction: `docs/internal/clover/CLOVER_VNEXT_FINAL.md` (Jailgun flywheel final revise, run e7a8a34d) and `docs/internal/clover/CLOVER_VNEXT_SUMMARY.md`. Read both first.

## Direction in one paragraph
CLOVER-Discover (ordinary GP program discovery + verification) is the foundation and must work with **no neural model**. CLOVER-Learn (learned controller) is an experiment only. Hypotheses: H1 GP finds useful programs beyond the 6,640-program catalog; H2 a learned selector beats simple selectors on fixed pools; H3 fresh feedback beats frozen and restart-only controls. Deferred: DPO on the old catalog, MCTS, self-play leagues — do not start them.

## Your task: Phase 0 — establish comparability
Work on a new branch `clover/phase0-foundations` from up-to-date `origin/main`, in a separate git worktree (e.g. `~/JopeDime-clover`), never in the shared `~/JopeDime` checkout. Implement, with tests that run offline on CPU in CI:

1. **Experiment-identity contract** — one canonical, hashed identity record covering full source, initial state/checkpoint, evaluator, support (program set), objective, data roster, seed, budget, and guard identities. Reject conflicting lane labels and any run whose identity is incomplete.
2. **Evaluator quantile audit** — independent quantile fixtures and a versioned parity contract that checks evaluator semantics against them.
3. **Label-leakage audit** — encoder/label visibility audit proving no label or future data reaches features/encoders across task-role boundaries.
4. **HOLD / recovery fault tests** — fault-injected tests showing fail-closed behavior: persistent HOLD, recovery, terminalization of unknown states, and immutable evidence publication.

Storage rules for any evidence you design: immutable, content-addressed objects; the manifest is written **last**; plan to the spec cap of 25 GB/day.

## Hard limits
- Do NOT touch S3 deletion, lifecycle rules/policies, bucket settings, or run any delete against S3.
- Do NOT stop, restart, modify, or launch training runs, and don't touch other screen sessions or their worktrees.
- No force-push to `main`; no admin merges. Do not merge your own PR.
- Keep changes small and reviewable; split into multiple PRs if one gets large.

## Done when
CI is green and a PR (title prefix `clover: phase0`) is open against `main` describing each item, its tests, and how it maps to the P0 rows of §12 in `docs/internal/clover/CLOVER_VNEXT_FINAL.md`. Then stop and wait.

Note: these files live under docs/internal/ because deploy/ci/ip-scrub.sh forbids search-method vocabulary (GP, genetic, mutation, ...) in public prose (README.md, docs/*.md, runbooks, crate descriptions). Keep any new CLOVER prose under docs/internal/ or in code.
