#!/usr/bin/env python3
"""Create tall, complete Rust page assets from the six verified native exports.

The existing code-page schema/chunk reconstructor is unchanged. This standalone
process temporarily selects a presentation palette and lexical type coloring;
it never rewrites source or edits an existing raster image.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import re
import struct
import sys

import code_pages

PALETTE = {
    "background": "#080D17", "border": "#1A2536", "text": "#D7DEE8",
    "keyword": "#CF8FE7", "type": "#76C9E8", "punctuation": "#A5B5C7",
    "string": "#DFC285", "number": "#F3C668", "comment": "#85B664",
    "gutter": "#657387",
}
TYPE_IDENTIFIER = re.compile(r"(?:[A-Z][A-Za-z0-9_]*|[ui](?:8|16|32|64|128)|f(?:32|64)|usize|isize|bool|char|str)")
ORIGINAL_SPANS = code_pages._spans


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rust_spans(source: str):
    spans, lexer = ORIGINAL_SPANS(source)
    colored = []
    for first, last, category in spans:
        token = source[first:last]
        # RustLexer treats //! and /// documentation as String.Doc; visually
        # they remain source comments, including their unmodified delimiters.
        if category == "string" and token.startswith(("//", "/*")):
            category = "comment"
        elif category in {"text", "keyword"} and TYPE_IDENTIFIER.fullmatch(token):
            category = "type"
        colored.append((first, last, category))
    return colored, lexer + " + Rust doc-comment and lexical type coloring"


@contextmanager
def presentation():
    # This CLI owns its process; restore both module bindings even on refusal.
    previous_palette, previous_spans = code_pages.PALETTE, code_pages._spans
    code_pages.PALETTE, code_pages._spans = PALETTE, rust_spans
    try:
        yield
    finally:
        code_pages.PALETTE, code_pages._spans = previous_palette, previous_spans


def inside(root: Path, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute():
        raise ValueError("artifact reference must be relative")
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ValueError("artifact reference escapes source root or is absent")
    return resolved


def make_catalog(source_root: Path, output_dir: Path, *, font_path: Path | None = None,
                 font_size: int = 22, columns: int = 66, rows: int = 60) -> dict:
    """Generate fresh opaque pages; the source controller is the identity input."""
    source_root, output_dir = Path(source_root).resolve(), Path(output_dir).resolve()
    worktree = Path(__file__).resolve().parents[2]
    if output_dir.exists():
        raise FileExistsError("refuse to replace an existing page catalog")
    if output_dir.is_relative_to(worktree):
        raise ValueError("generated source/image assets must remain outside Git")
    controller_path = source_root / "controller.receipt.json"
    controller_sha = sha(controller_path)
    controller = json.loads(controller_path.read_text())
    if controller.get("schema") != "jain.h256-gp-source-controller/v1" or controller.get("status") != "PASS":
        raise ValueError("a passing native source controller receipt is required")
    cases = controller["cases"]
    if not 1 <= len(cases) <= 16 or len({case["case_id"] for case in cases}) != len(cases):
        raise ValueError("require 1..16 distinct verified cases")
    inputs = []
    for case in cases:
        cid, selection = case["case_id"], case["selection"]
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", cid):
            raise ValueError("unsafe source case name")
        source = inside(source_root, selection["rust_path"])
        proof_path = inside(source_root, selection["proof_path"])
        if sha(proof_path) != selection["proof_sha256"]:
            raise ValueError("native source proof bytes changed")
        proof = json.loads(proof_path.read_text())
        expected = selection["rust_sha256"]
        if (sha(source) != expected or proof["source_sha256"]["rust"] != expected
                or proof["state"] != "pass" or proof.get("native_build_profile") != "release"
                or proof.get("source_unchanged") is not True):
            raise ValueError("source is not the verified unchanged release export")
        inputs.append((cid, source, expected, selection))
    output_dir.mkdir(parents=True, exist_ok=False)
    results = []
    with presentation():
        for cid, source, expected, selection in inputs:
            manifest = code_pages.make_pages(source, output_dir / cid, font_path,
                                             font_size, columns, rows)
            raw = code_pages.reconstruct_source(manifest)
            if hashlib.sha256(raw).hexdigest() != expected or raw != source.read_bytes():
                raise ValueError("complete page reconstruction differs from exact source")
            for page in manifest["pages"]:
                image = (output_dir / cid / page["file"]).read_bytes()
                if (hashlib.sha256(image).hexdigest() != page["sha256"]
                        or image[:8] != b"\x89PNG\r\n\x1a\n"
                        or struct.unpack(">II", image[16:24]) != (manifest["width"], manifest["height"])):
                    raise ValueError("page raster identity/dimensions differ")
            result = {"case_id": cid, "source_path": str(source), "source_sha256": expected,
                      "source_bytes": len(raw), "proof_sha256": selection["proof_sha256"],
                      "manifest_path": str(output_dir / cid / "manifest.json"),
                      "manifest_sha256": sha(output_dir / cid / "manifest.json"),
                      "pages": manifest["page_count"], "width": manifest["width"],
                      "height": manifest["height"], "all_source_bytes_reconstructed": True,
                      "all_png_hashes_dimensions_verified": True}
            results.append(result)
            print(json.dumps(result), flush=True)
    if sha(controller_path) != controller_sha:
        raise ValueError("native source controller changed during rasterization")
    receipt = {"schema": "jain.cinematic-code-page-catalog/v1", "status": "PASS",
               "generator_sha256": sha(Path(__file__)), "code_pages_sha256": sha(Path(code_pages.__file__)),
               "source_controller_path": str(controller_path), "source_controller_sha256": controller_sha,
               "output_directory": str(output_dir), "font_size": font_size, "columns": columns,
               "rows_per_page": rows, "palette": PALETTE,
               "type_color_scope": "lexical primitive/UpperCamel identifiers; no semantic type inference",
               "chrome_scope": "opaque page with internal gutter/title; rounded frame, glow and traffic lights are scene chrome",
               "source_coverage": "complete exact original Rust files; no excerpt, ellipsis or rejoined tokens",
               "page_count": sum(case["pages"] for case in results), "cases": results,
               "argv": sys.argv}
    with (output_dir / "catalog.receipt.json").open("x") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--font", type=Path)
    parser.add_argument("--font-size", type=int, default=22)
    parser.add_argument("--columns", type=int, default=66)
    parser.add_argument("--rows", type=int, default=60)
    args = parser.parse_args()
    receipt = make_catalog(args.source_root, args.output_dir, font_path=args.font,
                           font_size=args.font_size, columns=args.columns, rows=args.rows)
    print(json.dumps({"status": receipt["status"], "pages": receipt["page_count"],
                      "receipt": str(args.output_dir.resolve() / "catalog.receipt.json")}))


if __name__ == "__main__":
    main()
