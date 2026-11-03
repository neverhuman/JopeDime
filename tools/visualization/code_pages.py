"""Rasterize complete, byte-bound Rust source into opaque review HUD pages.

This creates new text assets, not edited screenshots. Soft wrapping never changes
the source. Manifest chunks retain every original character, including tabs and
CRLF, and ``reconstruct_source`` proves the original UTF-8 SHA-256. Gutters,
continuation marks, and the page title are separate presentation chrome.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import unicodedata
from pathlib import Path

MAX_SOURCE_BYTES = 4 * 1024 * 1024
MAX_PAGES = 4096
SCHEMA = "jain.code-pages/v1"
PALETTE = {
    "background": "#0B1119", "border": "#263140", "text": "#E8EEF5",
    "keyword": "#8FA3BF", "punctuation": "#78849A", "string": "#C8B98F",
    "number": "#B6B7A6", "comment": "#738092", "gutter": "#667384",
}
KEYWORDS = frozenset("as async await break const continue crate dyn else enum extern false fn for if impl in let loop match mod move mut pub ref return self Self static struct super trait true type unsafe use where while abstract become box do final macro override priv typeof unsized virtual yield try union".split())
FONT_CANDIDATES = (
    "/System/Library/Fonts/Menlo.ttc",
    "/System/Library/Fonts/Monaco.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationMono-Regular.ttf",
)
RAW_STRING_START = re.compile(r'(?:br|cr|r)(#*)"')
CHAR_LITERAL = re.compile(r"'(?:\\(?:x[0-9A-Fa-f]{2}|u\{[0-9A-Fa-f_]+\}|.)|[^'\\\r\n])'")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _cells(char: str, logical_column: int) -> int:
    if char == "\t":
        return 4 - logical_column % 4
    if unicodedata.combining(char):
        return 0
    return 2 if unicodedata.east_asian_width(char) in {"W", "F"} else 1


def _wrap(source: str, columns: int, maximum_rows: int) -> list[dict]:
    """Contiguous, lossless source chunks; columns measure visual cells."""
    chunks: list[dict] = []
    start = cursor = byte_start = byte_cursor = 0
    line = 1
    logical_column = start_column = visual_column = 0
    continuation = False

    def flush(end: int, byte_end: int, ending: str = "") -> None:
        nonlocal start, byte_start, start_column, visual_column, continuation
        if len(chunks) >= maximum_rows:
            raise ValueError(f"source exceeds {MAX_PAGES} page limit")
        chunks.append({
            "text": source[start:end], "source_char_start": start,
            "source_char_end": end, "source_byte_start": byte_start,
            "source_byte_end": byte_end, "source_line": line,
            "source_column_start": start_column, "continuation": continuation,
            "line_ending": ending, "visual_cells": visual_column,
        })
        start, byte_start = end, byte_end
        start_column, visual_column, continuation = logical_column, 0, True

    while cursor < len(source):
        char = source[cursor]
        if char in "\r\n":
            ending = "\r\n" if source.startswith("\r\n", cursor) else char
            cursor += len(ending)
            byte_cursor += len(ending.encode("utf-8"))
            flush(cursor, byte_cursor, ending)
            line += 1
            logical_column = start_column = 0
            continuation = False
            continue
        width = _cells(char, logical_column)
        if visual_column + width > columns and cursor > start:
            flush(cursor, byte_cursor)
        visual_column += width
        logical_column += width
        byte_cursor += len(char.encode("utf-8"))
        cursor += 1
    if start < len(source):
        flush(len(source), byte_cursor)
    return chunks


def _fallback_spans(source: str) -> list[tuple[int, int, str]]:
    """Explicit Rust lexical coloring; offsets never alter text."""
    result = []
    i = 0
    while i < len(source):
        begin, category = i, "text"
        if source.startswith("//", i):
            end = source.find("\n", i)
            i, category = (len(source) if end < 0 else end), "comment"
        elif source.startswith("/*", i):
            i, depth = i + 2, 1
            while i < len(source) and depth:
                if source.startswith("/*", i):
                    depth, i = depth + 1, i + 2
                elif source.startswith("*/", i):
                    depth, i = depth - 1, i + 2
                else:
                    i += 1
            category = "comment"
        else:
            raw = RAW_STRING_START.match(source, i)
            if raw:
                close = '"' + raw.group(1)
                end = source.find(close, i + len(raw.group(0)))
                i = len(source) if end < 0 else end + len(close)
                category = "string"
            elif source[i] == '"' or (source[i] in "bc" and source[i:i + 2].endswith('"')):
                i += 1 if source[i] == '"' else 2
                while i < len(source):
                    if source[i] == "\\":
                        i = min(len(source), i + 2)
                    elif source[i] == '"':
                        i += 1
                        break
                    else:
                        i += 1
                category = "string"
            elif source[i] == "'" and CHAR_LITERAL.match(source, i):
                match = CHAR_LITERAL.match(source, i)
                i += len(match.group(0))
                category = "string"
            elif source[i].isdigit():
                i += 1
                while i < len(source) and (source[i].isalnum() or source[i] in "_."):
                    i += 1
                category = "number"
            elif source[i].isalpha() or source[i] == "_":
                i += 1
                while i < len(source) and (source[i].isalnum() or source[i] == "_"):
                    i += 1
                category = "keyword" if source[begin:i] in KEYWORDS else "text"
            elif source[i].isspace():
                i += 1
                while i < len(source) and source[i].isspace():
                    i += 1
            else:
                i += 1
                category = "punctuation"
        result.append((begin, i, category))
    return result


def _spans(source: str) -> tuple[list[tuple[int, int, str]], str]:
    try:
        import pygments
        from pygments.lexers import RustLexer
        from pygments.token import Comment, Keyword, Number, Operator, Punctuation, String
    except ImportError:
        return _fallback_spans(source), "explicit-rust-lex/v1"
    lexer = RustLexer(stripnl=False, ensurenl=False, tabsize=0)
    spans, cursor = [], 0
    for offset, token, text in lexer.get_tokens_unprocessed(source):
        if not text:
            continue
        if offset != cursor or source[offset:offset + len(text)] != text:
            raise ValueError("syntax lexer changed source text or offsets")
        category = "text"
        for family, name in ((Comment, "comment"), (Keyword, "keyword"), (Number, "number"),
                             (String, "string"), (Punctuation, "punctuation"), (Operator, "punctuation")):
            if token in family:
                category = name
                break
        cursor += len(text)
        spans.append((offset, cursor, category))
    if cursor != len(source):
        raise ValueError("syntax lexer omitted source characters")
    return spans, f"Pygments {pygments.__version__} RustLexer"


def reconstruct_source(manifest: dict) -> bytes:
    """Reconstruct and verify every UTF-8 byte from ordered manifest chunks."""
    if manifest.get("schema") != SCHEMA:
        raise ValueError("unsupported code-page manifest schema")
    chars = byte_count = 0
    parts = []
    pages = manifest["pages"]
    if len(pages) != manifest["page_count"] or not pages:
        raise ValueError("code-page count differs")
    for number, page in enumerate(pages, 1):
        if page["page"] != number or not page["rows"]:
            raise ValueError("unordered or empty code page")
        for row in page["rows"]:
            text = row["text"]
            raw = text.encode("utf-8")
            if (row["source_char_start"] != chars or row["source_byte_start"] != byte_count
                    or row["source_char_end"] != chars + len(text)
                    or row["source_byte_end"] != byte_count + len(raw)):
                raise ValueError("noncontiguous source chunk coverage")
            parts.append(text)
            chars += len(text)
            byte_count += len(raw)
    raw = "".join(parts).encode("utf-8")
    if (len(raw) != manifest["source_bytes"] or chars != manifest["source_characters"]
            or _sha(raw) != manifest["source_sha256"]):
        raise ValueError("reconstructed source identity differs")
    return raw


def make_pages(source_path: Path, output_dir: Path, font_path: Path | None = None,
               font_size: int = 21, columns: int = 48, rows: int = 24) -> dict:
    """Create complete-source PNG pages and a lossless source-chunk manifest.

    Output directories must not exist. ``manifest.json`` is written last; an
    interrupted directory without that manifest is incomplete and never reused.
    ``columns`` counts display cells; tabs retain their source byte and advance
    to four-column stops, without inserting spaces into the stored source.
    """
    from PIL import Image, ImageDraw, ImageFont
    source_path, output_dir = Path(source_path).resolve(), Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError("refuse to replace an existing code-page output directory")
    if not (8 <= font_size <= 96 and 8 <= columns <= 160 and 2 <= rows <= 100):
        raise ValueError("font_size 8..96, columns 8..160, rows 2..100 required")
    with source_path.open("rb") as handle:
        raw = handle.read(MAX_SOURCE_BYTES + 1)
    if not raw or len(raw) > MAX_SOURCE_BYTES:
        raise ValueError("nonempty UTF-8 source of at most 4 MiB required")
    source = raw.decode("utf-8")
    font_path = Path(font_path) if font_path else next((Path(p) for p in FONT_CANDIDATES if Path(p).is_file()), None)
    if font_path is None:
        raise FileNotFoundError("no monospace font found; provide --font")
    font_path = font_path.resolve()
    font = ImageFont.truetype(str(font_path), font_size)
    if abs(font.getlength("M") - font.getlength("i")) > .05:
        raise ValueError("font must be monospace")
    chrome_font = ImageFont.truetype(str(font_path), max(8, round(font_size * .57)))
    cell = font.getlength("M")
    line_height = math.ceil(font_size * 1.23)
    left, top, right, bottom = 62, 22, 14, 14
    width = 4 * math.ceil((left + cell * columns + right) / 4)
    height = top + line_height * rows + bottom
    if width > 4096 or height > 4096:
        raise ValueError("code-page dimensions exceed 4096 pixels")
    chunks = _wrap(source, columns, MAX_PAGES * rows)
    spans, lexer = _spans(source)
    page_count = math.ceil(len(chunks) / rows)
    if not 1 <= page_count <= MAX_PAGES:
        raise ValueError("source exceeds page budget")
    manifest = {
        "schema": SCHEMA, "source_path": str(source_path), "source_name": source_path.name,
        "source_sha256": _sha(raw), "source_bytes": len(raw), "source_characters": len(source),
        "source_lines": chunks[-1]["source_line"], "encoding": "UTF-8; original line endings preserved",
        "output_directory": str(output_dir), "page_count": page_count,
        "font_path": str(font_path), "font_sha256": _sha(font_path.read_bytes()),
        "font_size": font_size, "columns": columns, "rows_per_page": rows,
        "width": width, "height": height, "line_height": line_height,
        "code_origin": [left, top], "tab_stop": 4, "palette": PALETTE, "lexer": lexer,
        "lexer_scope": "syntax color only; never the source-emission token counter",
        "source_coverage": "all original characters in contiguous ordered chunks; chrome is separate",
        "gp_semantic_head": None, "semantic_scope": "exact source only; no invented program annotations",
        "presentation": "opaque screen-space HUD pages; page switches CONSTANT or explicit vertical scrolling",
        "pages": [],
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir()  # Exclusive creation; never edit or replace existing images.
    span_index = 0
    for page_index in range(page_count):
        page_rows = chunks[page_index * rows:(page_index + 1) * rows]
        image = Image.new("RGBA", (width, height), PALETTE["background"])
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, width - 1, height - 1), outline=PALETTE["border"], width=1)
        draw.text((14, 4), f"{source_path.name}  {page_index + 1}/{page_count}", font=chrome_font, fill=PALETTE["gutter"])
        for row_index, row in enumerate(page_rows):
            y = top + row_index * line_height
            gutter = ">" if row["continuation"] else str(row["source_line"])
            draw.text((left - 10, y + 4), gutter, anchor="ra", font=chrome_font, fill=PALETTE["gutter"])
            end = row["source_char_end"] - len(row["line_ending"])
            char_index, logical_column, visual_column = row["source_char_start"], row["source_column_start"], 0
            run, run_color, run_column = "", None, 0

            def flush_run() -> None:
                if run:
                    draw.text((left + run_column * cell, y), run, font=font, fill=PALETTE[run_color])

            while char_index < end:
                while span_index < len(spans) and spans[span_index][1] <= char_index:
                    span_index += 1
                category = spans[span_index][2]
                char = source[char_index]
                advance = _cells(char, logical_column)
                if char == "\t" or category != run_color:
                    flush_run()
                    run, run_color, run_column = "", category, visual_column
                if char != "\t":
                    if not run:
                        run_column = visual_column
                    run += char
                visual_column += advance
                logical_column += advance
                char_index += 1
            flush_run()
        filename = f"page-{page_index + 1:04d}.png"
        image.save(output_dir / filename, format="PNG")
        manifest["pages"].append({
            "page": page_index + 1, "file": filename,
            "sha256": _sha((output_dir / filename).read_bytes()),
            "source_char_start": page_rows[0]["source_char_start"],
            "source_char_end": page_rows[-1]["source_char_end"],
            "source_byte_start": page_rows[0]["source_byte_start"],
            "source_byte_end": page_rows[-1]["source_byte_end"],
            "source_line_start": page_rows[0]["source_line"], "source_line_end": page_rows[-1]["source_line"],
            "first_visual_row": page_index * rows + 1, "last_visual_row": page_index * rows + len(page_rows),
            "rows": page_rows,
        })
    if reconstruct_source(manifest) != raw:
        raise ValueError("code-page reconstruction failed")
    partial = output_dir / "manifest.json.partial"
    partial.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    partial.rename(output_dir / "manifest.json")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--font", type=Path)
    parser.add_argument("--font-size", type=int, default=21)
    parser.add_argument("--columns", type=int, default=48)
    parser.add_argument("--rows", type=int, default=24)
    args = parser.parse_args()
    result = make_pages(args.source, args.output, args.font, args.font_size, args.columns, args.rows)
    print(json.dumps({"manifest": str(args.output.resolve() / "manifest.json"),
                      "source_sha256": result["source_sha256"], "pages": result["page_count"],
                      "width": result["width"], "height": result["height"]}))


if __name__ == "__main__":
    main()
