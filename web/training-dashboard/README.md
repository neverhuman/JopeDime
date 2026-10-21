# Training observatory browser checks

`npm test` runs the existing Vitest evidence-semantics suite. `npm run test:rendered`
runs Chromium against the local Vite frontend at `127.0.0.1:4173`. It starts its own
server and refuses to reuse another process. The API is intercepted with artificial
snapshot/program records; these checks never query a live observer or prove real
training, scientific utility, custody, or serving admission.

## Run and retain evidence

Use the Node version declared in `package.json`, then run:

```sh
npm ci
PLAYWRIGHT_SKIP_BROWSER_GC=1 PLAYWRIGHT_BROWSERS_PATH="$PWD/../../target/ci-evidence/rendered-ux/browser-cache" npx playwright install chromium
PLAYWRIGHT_SKIP_BROWSER_GC=1 PLAYWRIGHT_BROWSERS_PATH="$PWD/../../target/ci-evidence/rendered-ux/browser-cache" npm run test:rendered
```

The runner image must already contain the Chromium OS dependencies. The browser
install runs without privileged package installation. Keep the browser cache
inside the job's writable evidence area and disable Playwright cache garbage
collection so the installer cannot reclaim another user's browser revisions.

Set `JOPE_UX_EVIDENCE` to an absolute job-specific directory to select the evidence
root (default `target/ci-evidence/rendered-ux` from repository root). This contains:

- `junit.xml` and `report.json`, with nonzero test counts and all attachments;
- `html/`, the portable browser report;
- `results/`, failure screenshots/traces when a test fails.

The rendered receipt helper's v2 format binds the tested source, browser checkout,
workflow event, run and attempt separately. Hosted creation passes `--checkout-sha`
from `git rev-parse HEAD` after checking it equals `JOPE_UI_SOURCE_SHA`, and
`--event-sha` from `GITHUB_SHA`. Aggregate verification expects the producer's
source checkout, even when its own checkout is a synthetic PR merge. Local helper
calls default both identities to `--source-sha`. Known Playwright CI/Git annotations
remain in the raw report and are checked against those identities; they cannot
replace the mandatory source/run/attempt metadata. Earlier v1 receipts must be
regenerated from their original evidence with explicitly checked identities.

The two viewport projects use desktop 1440×1000 and mobile 390×844. Every section
captures a screenshot, measured geometry and an axe WCAG A/AA result. Tests fail
on document overflow, clipped panels, inaccessible scroll regions or axe violations.
A keyboard test verifies initial modal focus, forward/backward focus containment,
Escape dismissal and return to the opener, including React StrictMode. A simulated
HTTP 503 checks that old observations remain visible with interruption disclosed.

Screenshots are review evidence, not automatically approved visual baselines.
Inspect desktop/mobile captures when changing layout, content or design tokens.
The automated checks cover these synthetic states; they do not establish complete
WCAG conformance, touch-device coverage or live-observer correctness.
