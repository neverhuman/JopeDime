#!/usr/bin/env python3
"""Verify the publication's complete input/output closure without rerendering."""
import hashlib
import json
from pathlib import Path
import plistlib
import re

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT.parents[1]


def verify():
    from build_manual import GENERATED

    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    pub = json.loads((ROOT / "publication.json").read_text())
    receipt = json.loads((ROOT / "render-receipt.json").read_text())
    expected = {str(p.relative_to(DOCS)) for p in ROOT.rglob("*")
                if p.is_file() and p.suffix in {".md", ".json", ".jsonl", ".csv", ".py", ".css", ".png", ".svg", ".plist", ".gz", ".txt"}
                and p.name not in GENERATED} | {"asm-maintainer-status.plist", "asm-maintainer-manual.md"}
    assert {i["path"] for i in receipt["inputs"]} == expected, "Incomplete render input inventory"
    assert len(receipt["inputs"]) == len(expected), "Duplicate render input"
    for item in receipt["inputs"]:
        assert sha(DOCS / item["path"]) == item["sha256"], f"Stale input: {item['path']}"
    landing = (DOCS / "asm-maintainer-manual.md").read_text()
    for target in re.findall(r"\]\(([^)]+)\)", landing):
        assert (DOCS / target.split("#")[0]).is_file(), f"Broken manual entrypoint link: {target}"
    book = json.loads((ROOT / "book.json").read_text())
    for chapter in book["chapters"] + book["appendices"]:
        assert f"(internal/asm-manual/sections/{chapter['file']})" in landing, "Missing manual entrypoint chapter"
    assert set(pub["pdfs"]) == {str(p.relative_to(DOCS)) for p in DOCS.rglob("*.pdf")}, "Undeclared PDF"
    assert set(pub["pdfs"]) == {v["pdf"] for v in receipt["volumes"]}, "Missing volume receipt"
    for volume in receipt["volumes"]:
        pdf = DOCS / volume["pdf"]
        assert sha(pdf) == volume["pdf_sha256"], f"Stale PDF: {pdf}"
        assert pdf.stat().st_size == volume["pdf_bytes"] <= 5 * 1024 * 1024, "PDF size violation"
    roadmap = json.loads((ROOT / "data/roadmap.json").read_text())
    assert roadmap == plistlib.loads((DOCS / "asm-maintainer-status.plist").read_bytes()), "JSON/plist mismatch"
    plan = (ROOT / "sections/plan.md").read_text()
    catalogue = (ROOT / "sections/catalogue.md").read_text()
    for item in roadmap["items"]:
        rows = [line for line in plan.splitlines() if line.startswith(f"| {item['id']} |")]
        assert len(rows) == 1 and rows[0].split("|")[2].strip() == item["priority"], f"Task index mismatch: {item['id']}"
        assert rows[0].split("|")[5].strip() == item["status"], f"Task index status mismatch: {item['id']}"
        card = re.search(r"^#### " + re.escape(item["id"]) + r" .*?(?=^#### |^### |\Z)", catalogue, re.M | re.S)
        assert card and f"**Priority:** {item['priority']}." in card[0], f"Task card mismatch: {item['id']}"
        status = re.search(r"\*\*Status:\*\* (.+?)(?:\n|$)", card[0])
        assert status and status[1].rstrip(".") == item["status"].rstrip("."), f"Task card status mismatch: {item['id']}"
    inventory = json.loads((ROOT / "data/dataset-inventory.json").read_text())
    assert len(inventory["datasets"]) == 30
    assert sum(len(d["features"]) for d in inventory["datasets"]) == 522
    return {"inputs": len(expected), "volumes": len(pub["pdfs"]), "tasks": len(roadmap["items"]), "datasets": 30, "feature_records": 522}


if __name__ == "__main__":
    print(json.dumps(verify()))
