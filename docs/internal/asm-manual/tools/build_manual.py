#!/usr/bin/env python3
"""Render the private maintainer book from local Markdown and figure inputs only."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
from urllib.parse import unquote, urlparse

from bs4 import BeautifulSoup
import markdown
from weasyprint import CSS, HTML, default_url_fetcher

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT.parents[1]
FETCH_FAILURES: list[str] = []


def local_only(url: str, *args, **kwargs):
    try:
        parsed = urlparse(url)
        if parsed.scheme not in {"file", "data"}:
            raise ValueError(f"Rendering cannot fetch a remote resource: {parsed.scheme}")
        if parsed.scheme == "file":
            if parsed.netloc:
                raise ValueError("Rendering cannot fetch a file URL with a host")
            path = Path(unquote(parsed.path)).resolve()
            if not path.is_relative_to(ROOT):
                raise ValueError(f"Rendering resource outside manual: {path}")
        return default_url_fetcher(url, *args, **kwargs)
    except Exception as error:
        # WeasyPrint logs and swallows fetch errors; retain them for the final gate.
        FETCH_FAILURES.append(str(error))
        raise


def main() -> None:
    FETCH_FAILURES.clear()
    # fontTools otherwise stamps subset fonts with the rendering clock.
    os.environ.setdefault("SOURCE_DATE_EPOCH", "1790121600")  # 2026-09-23 UTC
    def inputs():
        paths = [p for p in sorted(ROOT.rglob("*")) if p.is_file() and p.suffix in {".md", ".json", ".jsonl", ".csv", ".py", ".css", ".png", ".svg", ".plist"} and p.name not in {"render-receipt.json", "MANUAL.md"}]
        paths.append(DOCS / "asm-maintainer-status.plist")
        return [{"path": str(p.relative_to(DOCS)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths]

    before = inputs()
    sections = sorted((ROOT / "sections").glob("*.md"))
    if len(sections) != 17:
        raise SystemExit("Incomplete manual: expected seventeen section files")
    chapter_links = {path.name: f"chapter-{index:02}" for index, path in enumerate(sections, 1)}
    nav, content, combined = [], [], []
    for index, path in enumerate(sections, 1):
        source = path.read_text()
        page = BeautifulSoup(markdown.markdown(source, extensions=["tables", "fenced_code", "sane_lists"]), "html.parser")
        title = page.find("h1")
        if title is None:
            raise SystemExit(f"Missing chapter title: {path}")
        chapter_id = f"chapter-{index:02}"
        title["id"] = chapter_id
        title["data-chapter"] = str(index)
        nav.append(f'<li><a href="#{chapter_id}">{index:02}. {title.get_text()}</a></li>')
        for number, heading in enumerate(page.find_all(["h2", "h3"]), 1):
            heading["id"] = f"{chapter_id}-{number}"
        for link in page.find_all("a", href=True):
            if link["href"] in chapter_links:
                link["href"] = "#" + chapter_links[link["href"]]
        for table in page.find_all("table"):
            cells = table.find("tr").find_all(["th", "td"], recursive=False)
            table["class"] = ["dense" if len(cells) >= 8 else "standard"]
        for image in page.find_all("img"):
            # Markdown chapters also work on their own in sections/.
            image["src"] = image.get("src", "").removeprefix("../")
            image_path = (ROOT / image["src"]).resolve()
            if not image_path.is_relative_to(ROOT) or not image_path.is_file():
                raise SystemExit(f"Missing or nonlocal figure: {image['src']}")
        content.append(f'<section class="chapter">{page}</section>')
        book_source = source.replace("](../", "](")
        for filename, anchor in chapter_links.items():
            book_source = book_source.replace("](" + filename + ")", "](#" + anchor + ")")
        combined.append(f'<a id="{chapter_id}"></a>\n\n' + book_source)
    cover = """<section class="cover"><div class="eyebrow">JOPE DIME / PRIVATE MAINTAINER EDITION</div>
    <h1>Algorithm<br>Synthesis Model</h1><p class="subtitle">Technology, operations<br>and the evidence behind the claims</p>
    <div class="rule"></div><p class="definition">Learns to generate and optimize<br>executable algorithms from data.</p>
    <div class="cover-bottom"><b>User &amp; maintainer manual</b><br>23 September 2026<br><br>
    Implementation: reviewed PR83 / main a75e543d2166<br>30 frozen datasets · 522 feature records · full gap checklist<br>
    <span>Historical rankings and current measurements are identified separately.</span></div></section>"""
    front = """<section class="front"><h1>How to use this book</h1>
    <p>Start with the status chapter to distinguish what has shipped, what has been demonstrated, and what remains pending. Operators can then use installation and training procedures; maintainers can follow the architecture, language matrix, benchmark contract, detailed roadmap and evidence index.</p>
    <p>For actual learning progress, start with chapters 14–17: historical optimizer and target trajectories, binary packing, model versions, measurable KPIs, continuation commands and the remaining flywheel gaps. The plots distinguish real step logs, sampled endpoints and changing evaluation populations.</p>
    <p>The appendix contains development statistics for all thirty frozen datasets. The adjacent data directory supplies complete machine-readable statistics, historical ranking inputs, metadata inventories and source pins. The property list supplies 43 primary acceptance items and 24 separately marked coordinated proposals.</p>
    <p><strong>Snapshot:</strong> implementation reviewed at 68f56404c3e199e06413bfd645d3fe0803d7b96f; merged main a75e543d2166017cb0139e0a251d51d9a0c98e08. Both have tree 176b75bc36d76c7ce4eaeb7234677ec013468889. Individual observations retain their own timestamps.</p>
    <p>This private manual explains internal methods at the user's request. Public inference bundle allowlists remain separate. No sealed label bodies, model weights or credentials are embedded in this book.</p>
    <h2>Evidence vocabulary</h2><p><strong>Implemented</strong> means source exists. <strong>Qualified</strong> means a named execution was measured. <strong>Pending</strong> means acceptance remains. <strong>Unknown</strong> is not zero. Historical results do not establish current capability.</p>
    </section>"""
    html = '<!doctype html><html lang="en"><head><meta charset="utf-8"><title>ASM — Maintainer and User Manual</title><meta name="author" content="JopeDime maintainers"></head><body>' + cover + front + '<nav class="contents"><h1>Contents</h1><ol>' + ''.join(nav) + '</ol></nav>' + ''.join(content) + '</body></html>'
    html_path = ROOT / "manual.html"
    html_path.write_text(html)
    (ROOT / "MANUAL.md").write_text("# ASM maintainer and user manual\n\nPrivate edition · 23 September 2026\n\n" + "\n\n---\n\n".join(combined))
    document = HTML(string=html, base_url=str(ROOT), url_fetcher=local_only).render(stylesheets=[CSS(filename=str(ROOT / "tools/manual.css"), url_fetcher=local_only)])
    if FETCH_FAILURES:
        raise SystemExit(f"Incomplete rendering: {FETCH_FAILURES}")
    pdf = DOCS / "asm-maintainer-manual.pdf"
    document.write_pdf(pdf, pdf_tags=True)
    if FETCH_FAILURES:
        raise SystemExit(f"Incomplete PDF output: {FETCH_FAILURES}")
    if before != inputs():
        raise SystemExit("Manual inputs changed during rendering; PDF is not qualified")
    fonts = []
    for family in ["DejaVu Sans", "DejaVu Sans:style=Bold", "DejaVu Sans:style=Oblique", "DejaVu Sans:style=Bold Oblique", "DejaVu Sans Mono"]:
        font = Path(subprocess.check_output(["fc-match", "-f", "%{file}", family], text=True))
        fonts.append({"selection": family, "path": str(font), "sha256": hashlib.sha256(font.read_bytes()).hexdigest()})
    runtime = {"python": sys.version, "source_date_epoch": os.environ["SOURCE_DATE_EPOCH"], "libraries": {name: importlib.metadata.version(name) for name in ["weasyprint", "Markdown", "beautifulsoup4", "pydyf", "fonttools"]}, "fonts": fonts}
    receipt = {"schema": "jope.manual-render/v1", "sections": len(sections), "pages": len(document.pages), "pdf_sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(), "pdf_bytes": pdf.stat().st_size, "network_resources": False, "runtime": runtime, "inputs": before}
    (ROOT / "render-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({k: v for k, v in receipt.items() if k not in {"inputs", "runtime"}}))


if __name__ == "__main__":
    main()
