#!/usr/bin/env node
/**
 * Deterministic Headless Chromium high-DPI (2x Retina) screenshot & A4 PDF renderer
 * for `carbon-engineering-testing-spec.html` -> `carbon-engineering-testing-spec.pdf`.
 *
 * Uses `@playwright/test` locked in `web/training-dashboard/package-lock.json` with zero
 * network access and zero extra npm packages.
 */
import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { chromium } from "@playwright/test";

function parseArgs(argv) {
  let specDir = "";
  for (let i = 2; i < argv.length; i += 1) {
    if (argv[i] === "--spec-dir" && i + 1 < argv.length) {
      specDir = argv[i + 1];
      i += 1;
    } else {
      throw new Error(`Unknown or incomplete argument: ${argv[i]}`);
    }
  }
  if (!specDir) {
    throw new Error("Usage: node render-carbon-spec-pdf.mjs --spec-dir <DIR>");
  }
  return path.resolve(specDir);
}

function sha256Hex(buf) {
  return crypto.createHash("sha256").update(buf).digest("hex");
}

function readPngDimensions(buf) {
  if (buf.length < 24 || buf[0] !== 0x89 || buf[1] !== 0x50 || buf[2] !== 0x4e || buf[3] !== 0x47) {
    throw new Error("Invalid PNG header");
  }
  return {
    width: buf.readUInt32BE(16),
    height: buf.readUInt32BE(20),
  };
}

function countPdfPages(buf) {
  const text = buf.toString("latin1");
  const matches = text.match(/\/Type\s*\/Page(?![sS])/g);
  return matches ? matches.length : 0;
}

function resolveContainedLeaf(baseDir, rawLeaf) {
  const leaf = String(rawLeaf ?? "");
  if (!/^[a-z0-9][a-z0-9.-]*\.(?:png|html|pdf|json)$/.test(leaf) || leaf.includes("..")) {
    throw new Error(`Refusing unsafe relative asset leaf: ${leaf}`);
  }
  const resolvedBase = path.resolve(baseDir);
  const candidate = path.resolve(resolvedBase, leaf);
  const rel = path.relative(resolvedBase, candidate);
  if (!rel || rel.startsWith("..") || path.isAbsolute(rel)) {
    throw new Error(`Resolved asset path escaped base directory: ${leaf}`);
  }
  return { leaf, candidate };
}

async function main() {
  const specDir = parseArgs(process.argv);
  const manifestPath = resolveContainedLeaf(specDir, "spec-bundle-manifest.json").candidate;
  const manifest = JSON.parse(fs.readFileSync(manifestPath, "utf8"));
  const htmlPath = resolveContainedLeaf(specDir, manifest.html_file).candidate;
  const shotsDir = path.resolve(specDir, "screenshots");
  fs.mkdirSync(shotsDir, { recursive: true });

  const consoleErrors = [];
  const browser = await chromium.launch({ headless: true });
  try {
    const context = await browser.newContext({
      viewport: { width: 1280, height: 960 },
      deviceScaleFactor: 2,
    });
    const page = await context.newPage();
    page.on("pageerror", (err) => consoleErrors.push(String(err)));
    page.on("console", (msg) => {
      if (msg.type() === "error") {
        consoleErrors.push(msg.text());
      }
    });

    await page.goto(pathToFileURL(htmlPath).href, { waitUntil: "load" });

    // 1. Capture all 8 Retina (2x) PNG screenshots from `#capture-stage`
    const screenshotEntries = [];
    for (const [shotId, rawShotLeaf, caption] of manifest.required_screenshots) {
      if (!/^shot-0[1-8]$/.test(String(shotId))) {
        throw new Error(`Refusing unexpected screenshot DOM selector id: ${shotId}`);
      }
      const { leaf: safeShotLeaf, candidate: outPng } = resolveContainedLeaf(shotsDir, rawShotLeaf);
      const loc = page.locator(`#${shotId}`);
      await loc.waitFor({ state: "visible", timeout: 10000 });
      await loc.screenshot({ path: outPng, type: "png" });
      const raw = fs.readFileSync(outPng);
      const dims = readPngDimensions(raw);
      screenshotEntries.push({
        id: shotId,
        file_name: safeShotLeaf,
        caption,
        width: dims.width,
        height: dims.height,
        bytes: raw.length,
        sha256: sha256Hex(raw),
      });
    }

    // 2. Hide `#capture-stage` and refresh `#embedded-screenshot-gallery` `<img>` tags
    await page.evaluate(() => {
      const stage = document.getElementById("capture-stage");
      if (stage) {
        stage.style.display = "none";
      }
      for (const img of document.querySelectorAll("img.spec-screenshot-img")) {
        const fn = img.getAttribute("data-file-name");
        if (fn && /^[0-9]{2}-[a-z0-9-]+\.png$/.test(fn)) {
          img.setAttribute("src", `screenshots/${fn}?v=${Date.now()}`);
        }
      }
    });

    await page.waitForFunction(() => {
      const imgs = Array.from(document.querySelectorAll("img.spec-screenshot-img"));
      return imgs.length === 8 && imgs.every((img) => img.complete && img.naturalWidth >= 1200);
    });

    // 3. Measure horizontal overflow in print/A4 container width
    await page.setViewportSize({ width: 1120, height: 960 });
    const overflowPx = await page.evaluate(() => {
      const docEl = document.documentElement;
      return Math.max(0, Math.round(docEl.scrollWidth - docEl.clientWidth));
    });

    if (consoleErrors.length > 0) {
      throw new Error(`Browser console/page errors during spec render: ${consoleErrors.join("; ")}`);
    }

    // 4. Render publication-grade A4 PDF with header & page-numbered footer
    const { leaf: pdfOutputLeaf, candidate: pdfPath } = resolveContainedLeaf(
      specDir,
      "carbon-engineering-testing-spec.pdf"
    );
    await page.pdf({
      path: pdfPath,
      format: "A4",
      printBackground: true,
      preferCSSPageSize: true,
      displayHeaderFooter: true,
      headerTemplate: `<div style="width:100%;font-size:7.3pt;color:#64748b;padding:0 11mm;font-family:'Liberation Sans',sans-serif;display:flex;justify-content:space-between;"><span>JopeDime Carbon v${manifest.binary_version} (${manifest.release_tag}) — Zero-Knowledge S3 CAS &amp; Multi-Region Engineering Specification</span><span>SHA: ${String(manifest.source_sha).slice(0, 12)}</span></div>`,
      footerTemplate: `<div style="width:100%;font-size:7.3pt;color:#64748b;padding:0 11mm;font-family:'Liberation Sans',sans-serif;display:flex;justify-content:space-between;"><span>Confidentiality: CUEN v1 + CUDK v1 + S3 AES256 | Automated CI Qualification Bundle</span><span>Page <span class="pageNumber"></span> of <span class="totalPages"></span></span></div>`,
    });

    const pdfRaw = fs.readFileSync(pdfPath);
    const pdfPageCount = countPdfPages(pdfRaw);
    const receipt = {
      schema: "jopedime.carbon.spec-pdf-receipt.v1",
      rendered_epoch_ms: Date.now(),
      source_sha: manifest.source_sha,
      binary_version: manifest.binary_version,
      release_tag: manifest.release_tag,
      all_ci_tests_passed: Boolean(manifest.all_ci_tests_passed),
      all_prod_evidence_verified: Boolean(manifest.all_prod_evidence_verified),
      pdf_file: pdfOutputLeaf,
      pdf_bytes: pdfRaw.length,
      pdf_sha256: sha256Hex(pdfRaw),
      pdf_page_count: pdfPageCount,
      html_sha256: manifest.html_sha256,
      ci_qualification_sha256: manifest.ci_qualification_sha256,
      horizontal_overflow_px: overflowPx,
      screenshots: screenshotEntries,
      evidence_ledger: manifest.evidence_ledger,
    };
    const receiptPath = resolveContainedLeaf(specDir, "spec-pdf-receipt.json").candidate;
    fs.writeFileSync(receiptPath, `${JSON.stringify(receipt, null, 2)}\n`, "utf8");
    process.stdout.write(
      `${JSON.stringify({
        pdf_file: pdfOutputLeaf,
        pdf_bytes: pdfRaw.length,
        pdf_page_count: pdfPageCount,
        screenshots: screenshotEntries.length,
        horizontal_overflow_px: overflowPx,
      })}\n`
    );
  } finally {
    await browser.close();
  }
}

main().catch((err) => {
  process.stderr.write(`render-carbon-spec-pdf failed: ${err?.stack || err}\n`);
  process.exit(1);
});
