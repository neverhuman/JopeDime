#!/usr/bin/env python3
"""Render the private maintainer book from local Markdown and figure inputs only.

book.json orders the book: numbered chapters (optionally grouped into parts), then lettered
appendices, and it can publish appendices as a separate volume so every PDF stays under the
repository's 5 MiB file limit. Headings in the chapter files are written without numbers; this
renderer numbers them (chapter N, section N.M, subsection N.M.K, sub-subsection N.M.K.L; appendix
A, A.1, ...) and builds each volume's two-level contents, a three-level contents list at the top of
every chapter, four-level PDF bookmarks, running chapter and section headers, and figure numbers.
It refuses to render a malformed hierarchy, an unresolved internal link or an oversized PDF.

Anchors do not depend on chapter position: a chapter file is named by its topic, and an anchor is
`c-<file stem>--<GitHub heading slug>`. Cross-references therefore also work when a chapter is read
on its own: `[text](training.md#resuming-a-run)`, or `[text](#resuming-a-run)` within a chapter; a
link whose text is exactly `§` is filled in with the target's number and title. aliases.json keeps
older destinations (edition-1 PDF anchors and earlier file names) pointing at their new homes.
publication.json supplies the publication record printed in the front matter.
"""
from __future__ import annotations

import hashlib
from datetime import datetime
from html import escape
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import unquote, urlparse

from bs4 import BeautifulSoup
import markdown
import pydyf
from weasyprint import CSS, HTML, default_url_fetcher

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT.parents[1]
FETCH_FAILURES: list[str] = []
GENERATED = {"render-receipt.json", "MANUAL.md", "structure.json"}
EXTENSIONS = ["tables", "fenced_code", "sane_lists"]
HEADINGS = ["h1", "h2", "h3", "h4", "h5", "h6"]
LEVEL_NAMES = {1: "chapter", 2: "section", 3: "subsection", 4: "sub-subsection"}
FENCE = re.compile(r"^(```|~~~)")
MD_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
PDF_LIMIT = 5 * 1024 * 1024


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


def relative_links(pdf_names: list[str], counts: dict[str, int]):
    """A write_pdf finisher. WeasyPrint writes every non-internal link as an absolute file:// URI,
    which names the build directory and breaks once the PDFs move. A link into another volume
    becomes a GoToR action naming that file and its named destination; any other local file link
    becomes a URI relative to docs/, where the PDFs are published."""
    def finish(document, pdf):
        for obj in pdf.objects:
            if not isinstance(obj, pydyf.Dictionary) or not isinstance(obj.get("A"), pydyf.Dictionary):
                continue
            action = obj["A"]
            if action.get("S") != "/URI" or not action["URI"].string.startswith("file:"):
                continue
            parsed = urlparse(action["URI"].string)
            path, fragment = Path(unquote(parsed.path)), parsed.fragment
            if path.name in pdf_names and fragment:
                obj["A"] = pydyf.Dictionary({"Type": "/Action", "S": "/GoToR", "F": pydyf.String(path.name), "D": pydyf.String(fragment), "NewWindow": "false"})
                label = f"{path.name}#{fragment}"
                counts["cross_volume"] += 1
            else:
                if not path.resolve().is_relative_to(DOCS):
                    raise SystemExit(f"A link leaves the published docs directory: {path}")
                label = os.path.relpath(path, DOCS) + (f"#{fragment}" if fragment else "")
                action["URI"] = pydyf.String(label)
                counts["local_relative"] += 1
            if "Contents" in obj:
                obj["Contents"] = pydyf.String(label)
    return finish


def github_slug(text: str, seen: dict[str, int]) -> str:
    """GitHub's heading anchor: lowercase, keep letters, digits, spaces, '-' and '_', spaces to
    hyphens, then -1, -2, ... for repeats within one file."""
    base = re.sub(r"[^\w\- ]", "", text.strip().lower()).replace(" ", "-")
    count = seen.get(base, 0)
    seen[base] = count + 1
    return base if count == 0 else f"{base}-{count}"


def fail(problems: list[str], what: str) -> None:
    if problems:
        raise SystemExit(f"{what}:\n  " + "\n  ".join(problems))


def load_book() -> dict:
    book = json.loads((ROOT / "book.json").read_text())
    if book.get("schema") != "asm-manual.book/v1":
        raise SystemExit("book.json: expected schema asm-manual.book/v1")
    if "parts" in book:
        book["chapters"] = [{"file": name, "part": p["number"]} for p in book["parts"] for name in p["chapters"]]
    listed = [c["file"] for c in book["chapters"]] + [a["file"] for a in book["appendices"]]
    on_disk = sorted(p.name for p in (ROOT / "sections").glob("*.md"))
    problems = [f"listed twice: {n}" for n in sorted({n for n in listed if listed.count(n) > 1})]
    problems += [f"not in book.json: {n}" for n in on_disk if n not in listed]
    problems += [f"missing file: {n}" for n in listed if n not in on_disk]
    letters = [a["letter"] for a in book["appendices"]]
    if letters != [chr(ord("A") + i) for i in range(len(letters))]:
        problems.append(f"appendix letters must run A, B, C, ...: {letters}")
    volumes = book.get("volumes", {})
    problems += [f"{a['file']}: unknown volume {a['volume']}" for a in book["appendices"] if a.get("volume", "main") not in volumes | {"main": None}]
    fail(problems, "Book layout is inconsistent")
    return book


def main() -> None:
    FETCH_FAILURES.clear()
    pub = json.loads((ROOT / "publication.json").read_text())
    if pub.get("schema") != "asm-manual.publication/v1":
        raise SystemExit("Invalid publication schema")
    def timestamp(value):
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
            raise SystemExit("Publication timestamps must include UTC")
        return parsed
    build = timestamp(pub["build_utc"])
    if pub["edition_date"] != build.date().isoformat():
        raise SystemExit("Publication date differs from build timestamp")
    imp = pub["implementation_snapshot"]
    for key in ("source", "main", "tree"):
        if not re.fullmatch(r"[0-9a-f]{40}", imp[key]):
            raise SystemExit(f"Invalid implementation {key}")
    if timestamp(imp["merged_utc"]) > build or not pub["qualification_observations"]:
        raise SystemExit("Invalid publication observations")
    for observation in pub["qualification_observations"]:
        if timestamp(observation["observed_utc"]) > build or not observation["receipt"] or not observation["observation"]:
            raise SystemExit("Invalid dated qualification observation")
    # fontTools uses the declared build clock, not an inherited or hardcoded date.
    os.environ["SOURCE_DATE_EPOCH"] = str(int(build.timestamp()))

    def inputs():
        paths = [p for p in sorted(ROOT.rglob("*")) if p.is_file() and p.suffix in {".md", ".json", ".jsonl", ".csv", ".py", ".css", ".png", ".svg", ".plist", ".gz", ".txt"} and p.name not in GENERATED]
        paths.append(DOCS / "asm-maintainer-status.plist")
        return [{"path": str(p.relative_to(DOCS)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths]

    before = inputs()
    book = load_book()

    # Every chapter in reading order, with its display number: 1, 2, ... then A, B, ...
    chapters = [{"file": c["file"], "label": str(n), "kind": "Chapter", "part": c.get("part"), "volume": "main"} for n, c in enumerate(book["chapters"], 1)]
    chapters += [{"file": a["file"], "label": a["letter"], "kind": "Appendix", "part": "appendices", "volume": a.get("volume", "main")} for a in book["appendices"]]
    stems = {c["file"]: Path(c["file"]).stem for c in chapters}
    volume_specs = {"main": {"pdf": book.get("pdf", "asm-maintainer-manual.pdf"), "title": book["title"]}} | book.get("volumes", {})
    pdf_names = {key: spec["pdf"] for key, spec in volume_specs.items()}
    if sorted(pub["pdfs"]) != sorted(pdf_names.values()) or len(set(pub["pdfs"])) != len(pub["pdfs"]):
        raise SystemExit("Publication PDF inventory differs from book")
    if any(Path(name).name != name or not name.endswith(".pdf") for name in pub["pdfs"]):
        raise SystemExit("Publication PDF names must be local basenames")
    if any(str(p.relative_to(DOCS)) not in pub["pdfs"] for p in DOCS.rglob("*.pdf")):
        raise SystemExit("Undeclared PDF under docs/")
    roadmap = json.loads((ROOT / "data/roadmap.json").read_text())
    import plistlib
    if roadmap != plistlib.loads((DOCS / "asm-maintainer-status.plist").read_bytes()):
        raise SystemExit("Roadmap JSON and plist differ")
    book["edition_line"] = f'Edition {pub["edition"]} · {pub["edition_date"]} · implementation {imp["main"][:12]}'
    book["cover"]["bottom_lines"] = [book["edition_line"], f'{len(roadmap["items"])} primary acceptance items',
        "Qualification observations and pending acceptance are recorded separately.", "See ASM at a glance in chapter 1."]
    volume_of = {c["file"]: c["volume"] for c in chapters}

    # Pass 1: parse, validate the hierarchy, number every heading and assign stable anchors.
    problems: list[str] = []
    warnings: list[str] = []
    structure = []
    anchors: dict[str, dict] = {}
    aliases: dict[str, str] = {}
    for chapter in chapters:
        chapter["source"] = (ROOT / "sections" / chapter["file"]).read_text()
        page = BeautifulSoup(markdown.markdown(chapter["source"], extensions=EXTENSIONS), "html.parser")
        chapter["page"] = page
        stem = stems[chapter["file"]]
        heads = page.find_all(HEADINGS)
        if not heads or heads[0].name != "h1":
            problems.append(f"{chapter['file']}: must begin with one '#' chapter title")
            continue
        if sum(1 for h in heads if h.name == "h1") != 1:
            problems.append(f"{chapter['file']}: more than one '#' chapter title")
        counters = [0, 0, 0]
        previous = 1
        seen: dict[str, int] = {}
        children: dict[str, list[str]] = {}
        parents = [f"c-{stem}"]  # parents[k] is the open heading at level k + 1
        for heading in heads:
            level = int(heading.name[1])
            title = heading.get_text(" ", strip=True)
            if level > 4:
                problems.append(f"{chapter['file']}: '{title}' is deeper than a sub-subsection ({heading.name})")
                continue
            if level > previous + 1:
                problems.append(f"{chapter['file']}: '{title}' skips from h{previous} to h{level}")
            previous = level
            if level == 1:
                display = chapter["label"]
                anchor = f"c-{stem}"
                # GitHub also anchors the chapter title; accept `file.md#title-slug` as the chapter.
                aliases[f"c-{stem}--{github_slug(title, seen)}"] = anchor
            else:
                counters[level - 2] += 1
                for deeper in range(level - 1, 3):
                    counters[deeper] = 0
                display = ".".join([chapter["label"]] + [str(c) for c in counters[: level - 1]])
                anchor = f"c-{stem}--{github_slug(title, seen)}"
                parents = parents[: level - 1]
                siblings = children.setdefault(parents[-1], [])
                if title in siblings:
                    problems.append(f"{chapter['file']}: duplicate heading '{title}' under the same parent")
                siblings.append(title)
                parents.append(anchor)
            heading["id"] = anchor
            heading["data-number"] = display
            heading["data-running"] = f"{display}  {title}" if level > 1 else f"{chapter['kind']} {display} · {title}"
            anchors[anchor] = {"number": display, "title": title, "level": level, "file": chapter["file"]}
            structure.append({"number": display, "level": level, "kind": LEVEL_NAMES[level], "title": title, "id": anchor, "file": chapter["file"], "volume": chapter["volume"]})
        # Empty headings and single-child subdivisions make a hierarchy harder to read.
        elements = [e for e in page.children if getattr(e, "name", None)]
        for index, element in enumerate(elements):
            if element.name in HEADINGS[:4]:
                level = int(element.name[1])
                following = elements[index + 1] if index + 1 < len(elements) else None
                if following is None or (following.name in HEADINGS and int(following.name[1]) <= level):
                    problems.append(f"{chapter['file']}: '{element.get_text(strip=True)}' has no content")
        for parent, kids in children.items():
            if len(kids) == 1:
                problems.append(f"{chapter['file']}: '{anchors.get(parent, {}).get('title', parent)}' has a single subdivision '{kids[0]}'")
        if len(children.get(f"c-{stem}", [])) < 2:
            warnings.append(f"{chapter['file']}: fewer than two sections")
    fail(problems, "Heading hierarchy is malformed")

    # Older destinations (edition-1 PDF anchors, earlier file names) keep resolving.
    alias_path = ROOT / "aliases.json"
    alias_record = json.loads(alias_path.read_text())
    published_aliases = alias_record["aliases"]
    if alias_record.get("unmapped") or len(published_aliases) != 266:
        raise SystemExit("Expected all 266 published legacy destinations")
    for old, new in published_aliases.items():
        new = aliases.get(new, new)
        if new not in anchors:
            problems.append(f"aliases.json: {old} -> {new} names no heading")
        elif old in anchors:
            problems.append(f"aliases.json: {old} is already a heading anchor")
        else:
            aliases[old] = new
    fail(problems, "Anchor aliases are inconsistent")

    # Pass 2: resolve links, add numbers to headings and figures, build the contents.
    unresolved = []
    figures: list[dict] = []

    def href_to(target: str, from_volume: str) -> str:
        """An internal link, or a link into the other PDF when the target lives there. A link to
        an appendix title in another volume stays in this volume, whose signpost carries it."""
        volume = volume_of[anchors[target]["file"]]
        if volume == from_volume or (anchors[target]["level"] == 1 and from_volume == "main"):
            return "#" + target
        return f"{pdf_names[volume]}#{target}"

    def resolve_links(page, where: str, stem: str | None, from_volume: str) -> None:
        """Point chapter links at anchors; fill `§` links with the target's number and title."""
        for link in page.find_all("a", href=True):
            href = link["href"]
            if href.startswith(("http://", "https://", "mailto:")):
                continue
            target = None
            if href.startswith("#"):
                target = href[1:] if href[1:] in aliases or href[1:] in anchors else (f"c-{stem}--{href[1:]}" if stem else None)
            else:
                name, _, fragment = href.partition("#")
                name = name.removeprefix("sections/")  # front matter links from the manual root
                if name in stems:
                    target = f"c-{stems[name]}" + (f"--{fragment}" if fragment else "")
            if target is None:
                # A local data or figure path. Chapter files sit in sections/, so their links start
                # with ../; the book resolves paths from the manual root, as it does for images.
                if stem is not None and href.startswith("../"):
                    link["href"] = href[3:]
                if not (ROOT / unquote(link["href"].partition("#")[0])).resolve().exists():
                    unresolved.append(f"{where}: {href} (no such local file)")
                continue
            target = aliases.get(target, target)
            if target not in anchors:
                unresolved.append(f"{where}: {href}")
                continue
            link["href"] = href_to(target, from_volume)
            if link.get_text(strip=True) == "§":
                link.string = f"§{anchors[target]['number']} {anchors[target]['title']}"

    for chapter in chapters:
        page = chapter["page"]
        stem = stems[chapter["file"]]
        resolve_links(page, chapter["file"], stem, chapter["volume"])
        for heading in page.find_all(["h1", "h2", "h3", "h4"]):
            if heading.name == "h1":
                eyebrow = page.new_tag("span", attrs={"class": "eyebrow"})
                eyebrow.string = f"{chapter['kind']} {chapter['label']}"
                heading.insert(0, eyebrow)
            else:
                number_span = page.new_tag("span", attrs={"class": "secnum"})
                number_span.string = heading["data-number"]
                heading.insert(0, number_span)
            for old in sorted(a for a, t in aliases.items() if t == heading["id"] and not a.startswith(f"{heading['id']}--")):
                heading.insert(0, page.new_tag("a", attrs={"id": old, "class": "alias"}))
        for table in page.find_all("table"):
            cells = table.find("tr").find_all(["th", "td"], recursive=False)
            table["class"] = ["dense" if len(cells) >= 8 else "standard"]
        for figure_number, image in enumerate(page.find_all("img"), 1):
            # Markdown chapters also work on their own in sections/.
            image["src"] = image.get("src", "").removeprefix("../")
            image_path = (ROOT / image["src"]).resolve()
            if not image_path.is_relative_to(ROOT) or not image_path.is_file():
                raise SystemExit(f"Missing or nonlocal figure: {image['src']}")
            # Number every figure within its chapter and print its alt text as the caption.
            label = f"Figure {chapter['label']}.{figure_number}"
            holder = image.parent if image.parent.name == "p" and len([c for c in image.parent.contents if str(c).strip()]) == 1 else image
            figure = page.new_tag("figure", attrs={"class": "fig", "id": f"fig-{stem}-{figure_number}"})
            caption = page.new_tag("figcaption")
            number_span = page.new_tag("span", attrs={"class": "fignum"})
            number_span.string = label
            caption.append(number_span)
            caption.append(" " + image.get("alt", ""))
            holder.replace_with(figure)
            figure.append(image)
            figure.append(caption)
            figures.append({"number": label, "file": chapter["file"], "src": image["src"], "caption": image.get("alt", "")})
        # A contents list at the top of each chapter, one level deeper than the volume contents.
        entries = [s for s in structure if s["file"] == chapter["file"] and s["level"] in (2, 3, 4)]
        if len(entries) >= 3:
            items = "".join(f'<li class="l{s["level"]}"><a href="#{s["id"]}"><span class="n">{s["number"]}</span> {escape(s["title"])}</a></li>' for s in entries)
            mini = BeautifulSoup(f'<nav class="minitoc"><div class="minitoc-title">In this {chapter["kind"].lower()}</div><ol>{items}</ol></nav>', "html.parser")
            page.find("h1").insert_after(mini)

    front_page = None
    front_path = ROOT / "front.md"
    if front_path.exists():
        front_page = BeautifulSoup(markdown.markdown(front_path.read_text(), extensions=EXTENSIONS), "html.parser")
        resolve_links(front_page, "front.md", None, "main")
    fail(unresolved, "Unresolved internal links")

    def contents_entry(entry, cls, from_volume="main"):
        href = href_to(entry["id"], from_volume)
        marker = "" if href.startswith("#") else ' class="ext"'  # page numbers cannot cross files
        return f'<li class="{cls}"><a href="{href}"{marker}><span class="n">{entry["number"]}</span> {escape(entry["title"])}</a></li>'

    def title_of(name):
        return next(s for s in structure if s["file"] == name and s["level"] == 1)

    by_file = {c["file"]: c for c in chapters}
    groups = []  # (id, label, title, summary, files) for each part; a flat book has one unnamed group
    if "parts" in book:
        groups = [(f"part-{p['number']}", f"Part {p['number']}", p["title"], p.get("summary", ""), p["chapters"]) for p in book["parts"]]
    else:
        groups = [(None, None, None, None, [c["file"] for c in book["chapters"]])]
    main_appendices = [a["file"] for a in book["appendices"] if a.get("volume", "main") == "main"]
    if main_appendices:
        groups.append(("part-appendices", "Appendices", book.get("appendices_title", "Reference material"), book.get("appendices_summary", ""), main_appendices))

    nav, content = [], []
    for part_id, label, title, summary, files in groups:
        if part_id:
            nav.append(f'<li class="toc-part"><a href="#{part_id}">{label} · {escape(title)}</a></li>')
            listing = "".join(f'<li><a href="#{title_of(n)["id"]}"><span class="n">{by_file[n]["kind"]} {title_of(n)["number"]}</span> {escape(title_of(n)["title"])}</a></li>' for n in files)
            content.append(f'<section class="part" id="{part_id}"><div class="part-label">{label}</div><h1 class="part-title" data-running="{escape(label)} · {escape(title)}">{escape(title)}</h1><p class="part-summary">{escape(summary)}</p><ol class="part-chapters">{listing}</ol></section>')
        for name in files:
            nav.append(contents_entry(title_of(name), "toc-ch"))
            nav.extend(contents_entry(s, "toc-sec") for s in structure if s["file"] == name and s["level"] == 2)
            content.append(f'<section class="chapter">{by_file[name]["page"]}</section>')

    # Appendices published in another volume keep their letters and titles here, on one signpost page.
    other_volumes = {}
    for a in book["appendices"]:
        if a.get("volume", "main") != "main":
            other_volumes.setdefault(a["volume"], []).append(a["file"])
    for key, names in other_volumes.items():
        spec = volume_specs[key]
        nav.append(f'<li class="toc-part"><a href="#volume-{key}">{escape(spec["label"])} · {escape(spec["title"])}</a></li>')
        nav.extend(contents_entry(title_of(n), "toc-ch") for n in names)
        rows = "".join(
            f'<div class="companion-entry" id="{title_of(n)["id"]}"><h2 class="companion-title" data-running="{by_file[n]["kind"]} {title_of(n)["number"]} · {escape(title_of(n)["title"])}"><span class="secnum">{by_file[n]["kind"]} {title_of(n)["number"]}</span> {escape(title_of(n)["title"])}</h2>'
            f'<ol class="companion-list">{"".join(contents_entry(s, "toc-sec") for s in structure if s["file"] == n and s["level"] == 2)}</ol></div>'
            for n in names)
        content.append(f'<section class="chapter companion" id="volume-{key}"><h1 class="volume-title" data-running="{escape(spec["label"])} · {escape(spec["title"])}"><span class="eyebrow">{escape(spec["label"])}</span>{escape(spec["title"])}</h1>'
                       f'<p class="companion-note">These appendices are published as <code>docs/{spec["pdf"]}</code>, with the same numbering and anchors; the entries link into that file. '
                       f'{escape(spec.get("note", ""))}</p>{rows}</section>')
        # Keep every old Volume-1 URL reachable after content moves to Volume 2.
        # Each landing has an explicit link to the exact semantic destination.
        forwards = []
        for old, target in sorted(published_aliases.items()):
            if volume_of[anchors[target]["file"]] == key:
                entry = anchors[target]
                forwards.append(f'<li id="{escape(old)}"><a href="{spec["pdf"]}#{target}">{escape(old)} → {entry["number"]} {escape(entry["title"])}</a></li>')
        if forwards:
            content.append('<section class="chapter"><h1>Legacy destination index</h1><p>Published edition-1 links to evidence now land here. Follow the entry to Volume 2.</p><ul>' + ''.join(forwards) + '</ul></section>')

    # The publication record separates the three identities a reader must not conflate.
    record = ""
    publication_path = ROOT / "publication.json"
    if publication_path.exists():
        pub = json.loads(publication_path.read_text())
        imp = pub["implementation_snapshot"]
        observations = "".join(f'<tr><td>{escape(o["observation"])}</td><td>{escape(o["observed_utc"])}</td><td><code>{escape(o["receipt"])}</code></td></tr>' for o in pub["qualification_observations"])
        inputs_digest = hashlib.sha256("".join(f'{i["path"]}\0{i["sha256"]}\n' for i in before).encode()).hexdigest()
        record = (
            '<h2>Publication record</h2>'
            '<p>Three identities describe this book. They change independently, and none of them implies the others.</p>'
            '<h3>Implementation snapshot</h3>'
            f'<p>Source <code>{escape(imp["source"])}</code>, merged to main as <code>{escape(imp["main"])}</code> at {escape(imp["merged_utc"])}; Git tree <code>{escape(imp["tree"])}</code>. {escape(imp.get("note", ""))}</p>'
            '<h3>Qualification observations</h3>'
            f'<p>{escape(pub.get("observations_note", ""))}</p>'
            f'<table><thead><tr><th>Observation</th><th>Observed (UTC)</th><th>Receipt</th></tr></thead><tbody>{observations}</tbody></table>'
            '<h3>Document build</h3>'
            f'<p>Edition {escape(str(pub["edition"]))}, {escape(pub["edition_date"])}. Rendered by <code>docs/internal/asm-manual/tools/build_manual.py</code> with SOURCE_DATE_EPOCH {escape(os.environ["SOURCE_DATE_EPOCH"])} from {len(before)} pinned inputs '
            f'(digest of their path and SHA-256 list: <code>{inputs_digest}</code>). Each volume\'s PDF hash, size and page count are recorded in <code>render-receipt.json</code>, because a PDF cannot contain its own hash. {escape(pub.get("build_note", ""))}</p>')

    c = book["cover"]
    lines = lambda key: "<br>".join(escape(x) for x in c[key])
    cover = (f'<section class="cover"><div class="eyebrow">{escape(c["eyebrow"])}</div><h1>{lines("title_lines")}</h1>'
             f'<p class="subtitle">{lines("subtitle_lines")}</p><div class="rule"></div><p class="definition">{lines("definition_lines")}</p>'
             f'<div class="cover-bottom"><b>{escape(c["bottom_title"])}</b><br>{lines("bottom_lines")}<br><span>{escape(c["note"])}</span></div></section>')
    head_html = lambda title: '<!doctype html><html lang="en"><head><meta charset="utf-8"><title>' + escape(title) + '</title><meta name="author" content="JopeDime maintainers"></head><body>'
    html = (head_html(book["title"]) + cover + f'<section class="front">{front_page or ""}{record}</section>'
            + '<nav class="contents"><h1>Contents</h1><ol>' + "".join(nav) + "</ol></nav>" + "".join(content) + "</body></html>")
    (ROOT / "manual.html").write_text(html)
    # Other volumes: a title page, their own two-level contents, then the full appendices.
    volume_html = {}
    for key, names in other_volumes.items():
        spec = volume_specs[key]
        rows = "".join(contents_entry(s, {1: "toc-ch", 2: "toc-sec"}[s["level"]], key) for n in names for s in structure if s["file"] == n and s["level"] in (1, 2))
        pages = "".join(f'<section class="chapter">{by_file[n]["page"]}</section>' for n in names)
        volume_html[key] = (head_html(f'{book["title"]} — {spec["title"]}') + f'<section class="cover"><div class="eyebrow">{escape(c["eyebrow"])}</div><h1>{escape(spec["title"])}</h1>'
                            f'<p class="subtitle">{escape(spec["label"])} of {escape(book["title"])}</p><div class="rule"></div><div class="cover-bottom">{escape(book["edition_line"])}</div></section>'
                            + f'<nav class="contents"><h1>Contents</h1><ol>{rows}</ol></nav>' + pages + "</body></html>")

    # The combined Markdown carries the same numbers, with anchors for every link target.
    targets = {link["href"].partition("#")[2] for chapter in chapters for link in chapter["page"].find_all("a", href=True) if "#c-" in link["href"]}
    combined = []
    order = [(label, title, summary, files) for _, label, title, summary, files in groups] + [(volume_specs[k]["label"], volume_specs[k]["title"], "", names) for k, names in other_volumes.items()]
    for label, title, summary, files in order:
        if label:
            combined.append(f"# {label} — {title}\n\n{summary or ''}".rstrip())
        for name in files:
            chapter = by_file[name]
            stem = stems[name]
            entries = [s for s in structure if s["file"] == name]
            numbered = iter(entries)
            out, fenced, matched = [], False, 0
            for line in chapter["source"].replace("](../", "](").split("\n"):
                if FENCE.match(line):
                    fenced = not fenced
                match = None if fenced else MD_HEADING.match(line)
                if match and len(match.group(1)) <= 4:
                    matched += 1
                    entry = next(numbered, None)
                    if entry is None or entry["level"] != len(match.group(1)):
                        raise SystemExit(f"{name}: Markdown heading lines do not match the parsed headings near '{line}'")
                    if entry["id"] in targets:
                        out.append(f'<a id="{entry["id"]}"></a>\n')
                    prefix = f"{chapter['kind']} {entry['number']} — " if entry["level"] == 1 else f"{entry['number']} "
                    line = f"{match.group(1)} {prefix}{match.group(2)}"
                out.append(line)
            if matched != len(entries):
                raise SystemExit(f"{name}: {matched} Markdown heading lines but {len(entries)} parsed headings")
            text = "\n".join(out)
            text = re.sub(r"\]\(#([^)]+)\)", lambda m: f"](#{aliases.get(m.group(1), m.group(1)) if m.group(1) in aliases or m.group(1) in anchors else aliases.get(f'c-{stem}--{m.group(1)}', f'c-{stem}--{m.group(1)}')})", text)
            text = re.sub(r"\]\(([^)/#]+\.md)(#[^)]*)?\)", lambda m: f"](#{aliases.get(t := f'c-{stems[m.group(1)]}' + ('--' + m.group(2)[1:] if m.group(2) else ''), t)})" if m.group(1) in stems else m.group(0), text)
            combined.append(text)
    (ROOT / "MANUAL.md").write_text((f"# {book['title']}\n\n{book['edition_line']}\n\n" + "\n\n---\n\n".join(combined)).rstrip() + "\n")
    (ROOT / "structure.json").write_text(json.dumps({"schema": "asm-manual.structure/v2", "warnings": warnings, "headings": structure, "figures": figures, "aliases": aliases}, indent=1) + "\n")

    footer = CSS(string='@page { @bottom-left { content: "' + book["edition_line"].replace('"', "'") + '"; } }', url_fetcher=local_only)
    stylesheets = [CSS(filename=str(ROOT / "tools/manual.css"), url_fetcher=local_only), footer]
    written = []
    for key, source in [("main", html)] + sorted(volume_html.items()):
        document = HTML(string=source, base_url=str(ROOT), url_fetcher=local_only).render(stylesheets=stylesheets)
        if FETCH_FAILURES:
            raise SystemExit(f"Incomplete rendering: {FETCH_FAILURES}")
        pdf = DOCS / pdf_names[key]
        links = {"cross_volume": 0, "local_relative": 0}
        document.write_pdf(pdf, pdf_tags=True, finisher=relative_links(list(pdf_names.values()), links))
        if FETCH_FAILURES:
            raise SystemExit(f"Incomplete PDF output: {FETCH_FAILURES}")
        if pdf.stat().st_size > PDF_LIMIT:
            raise SystemExit(f"{pdf.name} is {pdf.stat().st_size} bytes, above the repository's 5 MiB file limit")
        written.append({"volume": key, "pdf": pdf.name, "pages": len(document.pages), "pdf_sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(), "pdf_bytes": pdf.stat().st_size, "rewritten_links": links})
    if before != inputs():
        raise SystemExit("Manual inputs changed during rendering; PDF is not qualified")
    fonts = []
    for family in ["DejaVu Sans", "DejaVu Sans:style=Bold", "DejaVu Sans:style=Oblique", "DejaVu Sans:style=Bold Oblique", "DejaVu Sans Mono"]:
        font = Path(subprocess.check_output(["fc-match", "-f", "%{file}", family], text=True))
        fonts.append({"selection": family, "path": str(font), "sha256": hashlib.sha256(font.read_bytes()).hexdigest()})
    runtime = {"python": sys.version, "source_date_epoch": os.environ["SOURCE_DATE_EPOCH"], "libraries": {name: importlib.metadata.version(name) for name in ["weasyprint", "Markdown", "beautifulsoup4", "pydyf", "fonttools", "matplotlib", "numpy", "Pillow"]}, "fonts": fonts}
    levels = {name: sum(1 for s in structure if s["kind"] == name) for name in LEVEL_NAMES.values()}
    main_pdf = written[0]
    receipt = {"schema": "jope.manual-render/v2", "chapters": len(book["chapters"]), "appendices": len(book["appendices"]), "sections": len(chapters), "headings": levels, "aliases": len(aliases), "warnings": warnings,
               "pages": main_pdf["pages"], "pdf_sha256": main_pdf["pdf_sha256"], "pdf_bytes": main_pdf["pdf_bytes"], "volumes": written, "network_resources": False, "runtime": runtime, "inputs": before}
    (ROOT / "render-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({k: v for k, v in receipt.items() if k not in {"inputs", "runtime"}}))


if __name__ == "__main__":
    main()
