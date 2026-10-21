import { defineConfig } from "@playwright/test";
import { resolve } from "node:path";

// One bounded browser worker and a loopback-only synthetic observer. Never reuse
// an unknown running service or fetch live fleet/training/campaign telemetry.
const evidence = resolve(process.env.JOPE_UX_EVIDENCE ?? "../../target/ci-evidence/rendered-ux");
export default defineConfig({
  testDir: "./e2e",
  metadata: {
    source_sha: process.env.JOPE_UI_SOURCE_SHA,
    run_id: process.env.GITHUB_RUN_ID,
    attempt: process.env.GITHUB_RUN_ATTEMPT,
  },
  fullyParallel: false,
  workers: 1,
  retries: 0,
  forbidOnly: !!process.env.CI,
  timeout: 45_000,
  expect: { timeout: 5_000 },
  outputDir: resolve(evidence, "results"),
  reporter: [
    ["list"],
    ["junit", { outputFile: resolve(evidence, "junit.xml") }],
    ["json", { outputFile: resolve(evidence, "report.json") }],
    ["html", { outputFolder: resolve(evidence, "html"), open: "never" }],
  ],
  use: {
    browserName: "chromium",
    baseURL: "http://127.0.0.1:4173",
    locale: "en-US",
    timezoneId: "UTC",
    reducedMotion: "reduce",
    screenshot: "only-on-failure",
    trace: "retain-on-failure",
  },
  projects: [
    { name: "desktop", use: { viewport: { width: 1440, height: 1000 } } },
    { name: "mobile", use: { viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true } },
  ],
  webServer: {
    command: "npm run dev -- --port 4173 --strictPort",
    url: "http://127.0.0.1:4173",
    reuseExistingServer: false,
    timeout: 30_000,
  },
});
