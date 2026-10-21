import { expect, test } from "@playwright/test";
import type { Page, TestInfo } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";
import { observedAt, observerFixture } from "./observer-fixture";

async function evidence(page: Page, testInfo: TestInfo, name: string) {
  const geometry = await page.evaluate(() => {
    const rect = (selector: string) => {
      const element = document.querySelector(selector);
      if (!element) throw new Error(`Missing layout landmark: ${selector}`);
      const r = element.getBoundingClientRect();
      return { x: r.x, y: r.y, width: r.width, height: r.height, right: r.right };
    };
    const dialog = document.querySelector('[role="dialog"]')?.getBoundingClientRect();
    return { viewport: innerWidth, viewportHeight: innerHeight,
      dialog: dialog ? { x: dialog.x, y: dialog.y, right: dialog.right, bottom: dialog.bottom } : null,
      documentWidth: document.documentElement.scrollWidth,
      main: rect("main"), heading: rect("h1"), navigation: rect("nav"),
      panels: [...document.querySelectorAll(".panel")].map((element) => {
        const r = element.getBoundingClientRect();
        return { x: r.x, right: r.right, width: r.width, height: r.height };
      }) };
  });
  await testInfo.attach(`${name}-geometry.json`, { body: JSON.stringify(geometry, null, 2), contentType: "application/json" });
  expect(geometry.documentWidth, "The document must fit the viewport; tables/nav may scroll internally").toBeLessThanOrEqual(geometry.viewport + 1);
  expect(geometry.heading.width).toBeGreaterThan(0);
  for (const panel of geometry.panels) {
    expect(panel.width).toBeGreaterThan(0);
    expect(panel.height).toBeGreaterThan(0);
    expect(panel.x).toBeGreaterThanOrEqual(-1);
    expect(panel.right).toBeLessThanOrEqual(geometry.viewport + 1);
  }
  if (geometry.dialog) {
    expect(geometry.dialog.x).toBeGreaterThanOrEqual(0);
    expect(geometry.dialog.y).toBeGreaterThanOrEqual(0);
    expect(geometry.dialog.right).toBeLessThanOrEqual(geometry.viewport);
    expect(geometry.dialog.bottom).toBeLessThanOrEqual(geometry.viewportHeight);
  }
  const accessibility = await new AxeBuilder({ page }).withTags(["wcag2a", "wcag2aa", "wcag21aa"]).analyze();
  await testInfo.attach(`${name}-accessibility.json`, { body: JSON.stringify(accessibility, null, 2), contentType: "application/json" });
  expect(accessibility.violations.map(({ id, nodes }) => ({ id, targets: nodes.map((node) => node.target) })),
    "No axe WCAG A/AA violations in the rendered synthetic state").toEqual([]);
  await testInfo.attach(`${name}.png`, { body: await page.screenshot({ fullPage: !geometry.dialog, animations: "disabled" }), contentType: "image/png" });
}

test.beforeEach(async ({ page }) => {
  await page.clock.setFixedTime(new Date(observedAt * 1000));
  await page.route("**/api/**", async (route) => {
    const path = new URL(route.request().url()).pathname;
    const snapshot = observerFixture();
    if (path === "/api/v1/snapshot") await route.fulfill({ json: snapshot });
    else if (path === "/api/v1/gp") await route.fulfill({ json: snapshot.gp });
    else await route.fulfill({ status: 404, json: { error: "No live API access in rendered tests" } });
  });
});

test("all six sections retain bounded layout, screenshots and accessible semantics", async ({ page }, testInfo) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto("/");
  await expect(page.getByRole("heading", { name: "synthetic-rendered-ui-only", exact: true })).toBeVisible();
  for (const section of ["Overview", "Training", "Fleet", "Storage & datasets", "GP library", "Results & lineage"]) {
    const button = page.getByRole("navigation").getByRole("button", { name: section, exact: true });
    await button.click();
    await expect(button).toHaveAttribute("aria-current", "page");
    await expect(page.getByRole("heading", { level: 1 })).toHaveText(section === "Overview" ? "Training observatory" : section);
    if (section === "GP library") await expect(page.getByRole("button", { name: "Inspect program" })).toBeVisible();
    await evidence(page, testInfo, section.toLowerCase().replaceAll(/[^a-z]+/g, "-"));
  }
  expect(errors).toEqual([]);
});

test("keyboard opens details, traps focus, dismisses and restores the opener", async ({ page }, testInfo) => {
  await page.goto("/");
  const training = page.getByRole("navigation").getByRole("button", { name: "Training", exact: true });
  await training.focus();
  await page.keyboard.press("Enter");
  const opener = page.getByRole("button", { name: "Inspect", exact: true });
  await opener.focus();
  await page.keyboard.press("Enter");
  const dialog = page.getByRole("dialog", { name: "Checkpoint synthetic-step-2" });
  await expect(dialog).toBeVisible();
  const close = dialog.getByRole("button", { name: "Close details" });
  await expect(close).toBeFocused();
  await page.keyboard.press("Tab");
  await expect(close).toBeFocused();
  await page.keyboard.press("Shift+Tab");
  await expect(close).toBeFocused();
  await evidence(page, testInfo, "checkpoint-dialog");
  await page.keyboard.press("Escape");
  await expect(dialog).toHaveCount(0);
  await expect(opener).toBeFocused();
});

test("failed polls retain the previous observation and visibly disclose interruption", async ({ page }, testInfo) => {
  await page.goto("/");
  await expect(page.getByRole("heading", { name: "synthetic-rendered-ui-only", exact: true })).toBeVisible();
  await page.route("**/api/v1/snapshot", (route) => route.fulfill({ status: 503, json: { error: "synthetic outage" } }));
  await expect(page.getByRole("alert")).toContainText("Retaining the last observation", { timeout: 12_000 });
  await expect(page.getByRole("heading", { name: "synthetic-rendered-ui-only", exact: true })).toBeVisible();
  await expect(page.locator(".connection")).toContainText("Feed interrupted");
  await evidence(page, testInfo, "interrupted-feed");
});
