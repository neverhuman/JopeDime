#!/usr/bin/env python3
"""Validate the publication's import, citation and immutable-artifact contracts."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
import tempfile
from pathlib import Path

PAPER = Path(__file__).resolve().parent.parent
DOCS = ("main", "supplement", "listings")
OUTPUTS = ("InnovationZero", "InnovationZero-Supplement", "InnovationZero-Archive")
ARCHIVE_IDS = (
    ("challenge-master", "Complete organizer master"),
    ("challenge-participant", "Exact participant packet"),
    ("opus-source", "Opus5p5 Extra: complete source"),
    ("astra-source", "GPT6 Astra Max: complete source"),
    ("sol-source", "GPT6p1 Sol Max: complete source"),
    ("gemini-source", "Gemini4 argon high: complete source"),
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def local_file(relative: str) -> Path:
    path = PAPER / relative
    require(not Path(relative).is_absolute(), f"Absolute source path: {relative}")
    require(path.resolve().is_relative_to(PAPER.resolve()), f"Escaping source path: {relative}")
    require(path.is_file() and not path.is_symlink(), f"Missing regular source: {relative}")
    return path


def read_artifacts() -> dict:
    manifest = json.loads(local_file("evidence/artifact-manifest.json").read_text())
    items = manifest.get("artifacts")
    require(isinstance(items, list) and items, "No archival artifacts declared")
    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    for item in items:
        identity, relative = item["id"], item["path"]
        require(identity not in seen_ids, f"Duplicate artifact id: {identity}")
        require(relative not in seen_paths, f"Duplicate artifact path: {relative}")
        seen_ids.add(identity)
        seen_paths.add(relative)
        require(relative.startswith("artifacts/"), f"Artifact outside archive: {relative}")
        payload = local_file(relative).read_bytes()
        require(len(payload) == item["bytes"], f"Byte count mismatch: {relative}")
        require(hashlib.sha256(payload).hexdigest() == item["sha256"], f"SHA-256 mismatch: {relative}")
        require(bool(item.get("source_commit")), f"Missing snapshot provenance: {relative}")
    require(set(key for key, _ in ARCHIVE_IDS) <= seen_ids, "Complete challenge/source set is missing")
    return manifest



def check_elo(manifest: dict) -> None:
    """Bind the numerical plot summary to the checked historical result receipt."""
    summary = json.loads(local_file("evidence/elo-summary.json").read_text())
    result_item = next((item for item in manifest["artifacts"]
                        if item["path"] == "artifacts/benchmark/RESULTS.json"), None)
    require(result_item is not None, "No archived Elo result receipt")
    result = json.loads(local_file(result_item["path"]).read_text())
    require(summary["source"]["revision"] == result_item["source_commit"], "Elo source revision drift")
    require(summary["source"]["path"] == result_item["source_path"], "Elo source path drift")
    original = local_file(result_item["path"]).read_bytes()
    # The archival framing retains one additional LF in this result receipt.
    # Either identity is acceptable only if the source metadata records it exactly.
    require(any(hashlib.sha256(candidate).hexdigest() == summary["source"]["sha256"]
                and len(candidate) == summary["source"]["bytes"]
                for candidate in (original, original[:-1] if original.endswith(b"\n") else original)),
            "Elo source bytes/hash drift")
    ratings = summary["ratings"]
    require(len(ratings) == len(result["elo"]), "Elo entrant count drift")
    seen = set()
    for item in ratings:
        identity = item["artifact"]
        require(identity not in seen and identity in result["elo"], f"Unexpected/repeated Elo arm: {identity}")
        seen.add(identity)
        reference = result["elo"][identity]
        require(item["rating"] == reference["rating"] and item["ci95"] == reference["ci95"],
                f"Elo estimate changed: {identity}")
        lo, hi = item["ci95"]
        require(all(isinstance(value, (int, float)) and math.isfinite(value)
                    for value in (lo, item["rating"], hi)) and lo <= item["rating"] <= hi,
                f"Nonfinite/reversed Elo interval: {identity}")
    require(summary["pairwise"] == result["pairwise"], "Elo pairwise result drift")
    require(summary["panel"]["innovationzero_k1_entrant"] is False,
            "Exploratory panel incorrectly includes a qualified InnovationZero arm")
    require(summary["evidence_status"] == "repository-reported exploratory", "Elo evidence scope drift")
    require(summary["panel"]["families"] == 64 and summary["panel"]["folds"] == 3
            and summary["panel"]["seeds"] == [41, 73, 109]
            and summary["panel"]["cells_per_entrant"] == 576,
            "Exploratory panel identity drift")
    oriented = next(pair for pair in result["pairwise"]
                    if pair["a"] == "GPT6p1_Sol_Max" and pair["b"] == "Gemini4_argon_high")
    derived = summary["derived_interpretation"]
    ratio = oriented["geometric_rmse_ratio"]
    require(derived["sol_over_gemini_geometric_rmse_ratio"] == ratio
            and math.isclose(derived["gemini_relative_error_increase_point_estimate"], 1 / ratio - 1),
            "Oriented error-ratio interpretation drift")


def check_tex() -> None:
    bibliography = local_file("references.bib").read_text()
    bibkeys = re.findall(r"@\w+\s*\{\s*([^,\s]+)\s*,", bibliography)
    require(len(bibkeys) == len(set(bibkeys)), "Duplicate bibliography key")
    keys = set(bibkeys)
    require(keys, "Bibliography is empty")
    files = [PAPER / f"{name}.tex" for name in DOCS]
    files += sorted((PAPER / "sections").glob("*.tex"))
    files += sorted((PAPER / "appendices").glob("*.tex"))
    inputs: set[str] = set()
    cited: set[str] = set()
    for path in files:
        require(path.is_file(), f"Missing manuscript: {path.name}")
        text = path.read_text()
        # Comments may explain a deliberately absent experiment; only live prose is scanned.
        text = re.sub(r"(?<!\\)%[^\n]*", "", text)
        require(not re.search(r"\b(?:TODO|TBD|TBC|PLACEHOLDER)\b|\{\{[^}]+\}\}", text),
                f"Unresolved editorial placeholder: {path.relative_to(PAPER)}")
        require("\\write18" not in text, f"Shell escape in manuscript: {path.name}")
        for relative in re.findall(r"\\(?:input|include)\{([^}]+)\}", text):
            if relative == "archive-index":
                continue  # Generated exclusively from the checked archival manifest.
            inputs.add(relative)
            local_file(relative if Path(relative).suffix else relative + ".tex")
        for relative in re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", text):
            local_file(relative)
        for group in re.findall(r"\\cite\w*\*?(?:\[[^]]*\]){0,2}\{([^}]+)\}", text):
            cited.update(key.strip() for key in group.split(","))
    require(not (cited - keys), f"Unknown bibliography keys: {sorted(cited - keys)}")
    for folder in ("sections", "appendices"):
        for path in (PAPER / folder).glob("*.tex"):
            relative = str(path.relative_to(PAPER).with_suffix(""))
            require(relative in inputs or relative + ".tex" in inputs, f"Unincluded section: {relative}")


def latex_escape(value: str) -> str:
    replacements = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
                    "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
                    "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(replacements.get(c, c) for c in value)


def prepare_archive(manifest: dict, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    artifacts = {item["id"]: item for item in manifest["artifacts"]}
    lines = ["% Generated from checked artifact-manifest.json; do not edit.", r"\appendix"]
    for identity, title in ARCHIVE_IDS:
        item = artifacts[identity]
        lines += [r"\clearpage", r"\section{" + title + "}",
                  r"\noindent\textbf{Archive identity.} \nolinkurl{" + item["path"] + "}",
                  r"\par\noindent\textbf{Size.} " + f'{item["bytes"]:,}' + " bytes.",
                  r"\par\noindent\textbf{SHA-256.} \nolinkurl{" + item["sha256"] + "}",
                  r"\par\noindent\textbf{Snapshot.} \nolinkurl{" + str(item["source_commit"]) + "}",
                  r"\par\vspace{8pt}",
                  r"\VerbatimInput[fontsize=\footnotesize,breaklines=true,breakanywhere=true,"
                  r"breaksymbolleft={},breaksymbolright={},numbers=left,numbersep=5pt,"
                  r"frame=lines,framerule=0.2pt,xleftmargin=2em,tabsize=4]{" + item["path"] + "}"]
    (output / "archive-index.tex").write_text("\n".join(lines) + "\n")


def check_output(directory: Path) -> None:
    for name in OUTPUTS:
        pdf, log = directory / f"{name}.pdf", directory / f"{name}.log"
        require(pdf.is_file() and pdf.stat().st_size > 1000, f"Missing PDF: {name}")
        require(log.is_file(), f"Missing TeX log: {name}")
        logtext = log.read_text(errors="replace")
        defects = re.findall(r"^.*(?:Overfull \\[hv]box|undefined references|undefined citations|"
                             r"Citation .* undefined|Reference .* undefined|Missing character:|"
                             r"multiply defined|Package balance Warning:|LaTeX Font Warning: Font shape|"
                             r"LaTeX Error:|Package .* Error:).*$", logtext, re.M)
        require(not defects, f"PDF layout/reference defects in {name}:\n" + "\n".join(defects[:15]))
        metadata = subprocess.check_output(["pdfinfo", str(pdf)], text=True)
        require(re.search(r"Pages:\s+[1-9]\d*", metadata) is not None, f"Empty PDF: {name}")
        extracted = subprocess.check_output(["pdftotext", str(pdf), "-"], text=True)
        require(len(extracted.strip()) > 100, f"No extractable text: {name}")
        fonts = subprocess.check_output(["pdffonts", str(pdf)], text=True)
        for row in fonts.splitlines()[2:]:
            require("Type 3" not in row, f"Bitmap font in {name}: {row}")
            columns = row.split()
            # pdffonts table tail: emb sub uni object ID; names can have spaces.
            require(len(columns) >= 5 and columns[-5] == "yes", f"Unembedded font in {name}: {row}")
    print("PDFs: 3 built, references resolved, fonts embedded, no overfull boxes/missing glyphs")



def self_test() -> None:
    """Exercise refusal paths against disposable fixtures, never the live archive."""
    global PAPER
    original = PAPER
    build = original.parents[2] / "target" / "innovationzero-paper"
    build.mkdir(parents=True, exist_ok=True)
    cases = ("bytes", "digest", "duplicate", "escape", "citation", "input")
    try:
        with tempfile.TemporaryDirectory(prefix="validator-", dir=build) as tmp:
            PAPER = Path(tmp)
            (PAPER / "evidence").mkdir()
            (PAPER / "artifacts").mkdir()
            (PAPER / "sections").mkdir()
            (PAPER / "appendices").mkdir()
            items = []
            for identity, _ in ARCHIVE_IDS:
                relative = f"artifacts/{identity}.txt"
                (PAPER / relative).write_bytes(b"immutable\n")
                items.append({"id": identity, "path": relative, "bytes": 10,
                              "sha256": hashlib.sha256(b"immutable\n").hexdigest(),
                              "source_commit": "0" * 40})
            base = {"artifacts": items}
            (PAPER / "references.bib").write_text("@misc{known, title={Fixture}}\n")
            for document in DOCS:
                (PAPER / f"{document}.tex").write_text(r"\citep{known}")
            manifest_path = PAPER / "evidence/artifact-manifest.json"
            manifest_path.write_text(json.dumps(base))
            read_artifacts()
            check_tex()
            for case in cases:
                changed = json.loads(json.dumps(base))
                (PAPER / "main.tex").write_text(r"\citep{known}")
                if case == "bytes":
                    changed["artifacts"][0]["bytes"] += 1
                elif case == "digest":
                    changed["artifacts"][0]["sha256"] = "0" * 64
                elif case == "duplicate":
                    changed["artifacts"].append(changed["artifacts"][0].copy())
                elif case == "escape":
                    changed["artifacts"][0]["path"] = "artifacts/../../outside.txt"
                elif case == "citation":
                    (PAPER / "main.tex").write_text(r"\citep{unresolved}")
                elif case == "input":
                    (PAPER / "main.tex").write_text(r"\input{sections/missing}")
                manifest_path.write_text(json.dumps(changed))
                try:
                    read_artifacts()
                    check_tex()
                except ValueError:
                    continue
                raise ValueError(f"Validator accepted invalid {case} fixture")
            live_manifest = json.loads((original / "evidence/artifact-manifest.json").read_text())
            receipt = next(item for item in live_manifest["artifacts"]
                           if item["path"] == "artifacts/benchmark/RESULTS.json")
            destination = PAPER / receipt["path"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes((original / receipt["path"]).read_bytes())
            summary = json.loads((original / "evidence/elo-summary.json").read_text())
            summary_path = PAPER / "evidence/elo-summary.json"
            summary_path.write_text(json.dumps(summary))
            check_elo({"artifacts": [receipt]})
            for case in ("rating drift", "qualified arm"):
                changed = json.loads(json.dumps(summary))
                if case == "rating drift":
                    changed["ratings"][0]["rating"] += 1
                else:
                    changed["panel"]["innovationzero_k1_entrant"] = True
                summary_path.write_text(json.dumps(changed))
                try:
                    check_elo({"artifacts": [receipt]})
                except ValueError:
                    continue
                raise ValueError(f"Validator accepted {case}")
    finally:
        PAPER = original
    print("Validator self-test: 8 refusal cases passed (size, digest, duplicate, path escape, citation, input, rating drift, qualified arm)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="Exercise isolated validator refusal fixtures")
    parser.add_argument("--prepare", type=Path, help="Generate archival TeX under the build directory")
    parser.add_argument("--pdf-dir", type=Path, help="Also inspect completed PDF logs, text and fonts")
    args = parser.parse_args()
    try:
        if args.self_test:
            self_test()
            return 0
        manifest = read_artifacts()
        check_elo(manifest)
        check_tex()
        if args.prepare:
            prepare_archive(manifest, args.prepare.resolve())
        if args.pdf_dir:
            check_output(args.pdf_dir.resolve())
        print(f'Publication sources: imports/citations verified; {len(manifest["artifacts"])} exact archival artifacts verified')
    except (ValueError, KeyError, OSError, json.JSONDecodeError, subprocess.CalledProcessError) as error:
        print(f"publication check: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
