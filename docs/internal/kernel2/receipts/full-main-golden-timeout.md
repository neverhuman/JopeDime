# Full-main scratch golden timeout repair

Scope: CI/test only. The `feat-math::golden fast_profile_portfolio_head` test stays selected in both full-default and full-scratch with identical source, assertions, inputs, and nextest per-test deadline (600s × 2). No skip, ignore, xfail, tolerance, golden update, or runtime change.

Run 37287624697/1 on main14a7ce05b recorded full-default golden PASS in731.644s. Its full-scratch family reached the outer900s GNU timeout while that same test had run647.475s, after252.525s of collection/build/other setup. This produced SIGTERM and family exit124, with2667/2668 tests passing. Untouched46b shared main also passed the golden in670.679s. These are execution-time observations, not a golden mismatch.

Raise only the full-scratch family deadline900→1500s: preserve its existing1200s per-test hard deadline plus300s setup allowance. The full-default1500s family is unchanged. All other family commands and test selection stay unchanged. Nextest can still fail a genuinely stalled test at its original deadline, and GNU timeout still bounds the family. Increase the workflow deadline120→130min so the maximum sum of11 family budgets plus15s kill grace each (7365s,122.75min) leaves435s for evidence/cleanup. The regression reads the actual workflow deadline and reserves at least5min for uploads.

Acceptance: `python3 deploy/ci/test-full-main.py` must reproduce the inconsistent budget before correction and pass afterward. The complete scratch family and PR tier must remain nonzero and green; results and source/command/time/resource identities are retained outside the checkout under `/home/ubuntu/jope-evidence/kernel2/bt/GOLDEN_FULL_MAIN_TIMEOUT/`. Exact-head hosted CI and independent neverhumanbot approval are required before guarded rebase merge. A local CI change does not establish post-merge hosted full-main success.


Local cold-target scratch attempt at `0cfb163838281e07238016aa7717df05d3b4810b`
executed2691 tests:2686 passed,5 failed,36 skipped. The unchanged portfolio
golden passed in677.290s. All five failures reported missing `target/debug/scqd`:
this standalone local invocation omitted the workspace daemon build normally
supplied by the preceding full-default CI family. This attempt is retained as
failed, not called a passing scratch suite. Supply that unchanged build
prerequisite, then rerun affected fixtures and the bounded full-scratch family;
source/command/counts and the final outcome stay external and source-bound.
