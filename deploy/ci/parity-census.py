#!/usr/bin/env python3
"""Capability-parity census and guard (plan: red-team pass 2, items 6/7; WP10).

Two modes, standard library only:

  census  --old <pre-port checkout> --new <this repo> [-o docs/parity/capability-matrix.md]
          [--nextest-list <cargo nextest list --message-format json>] [--pytest-ids <file>]
          [--merge <existing matrix>]
      Enumerates every entry point of the pre-port tree (jope-dime c9513e82: the 88 clap
      subcommands of parallel/src/main.rs, the engine binaries and examples, the 55 root
      src/bin programs, the jope-prime CLI commands, every parallel/tools and scripts tool,
      the Justfile recipes, every systemd unit, the AGENT_CHAT wires and the lineage-A state
      the plan declares lost), resolves each one against this tree and writes the matrix.
      `--merge` carries hand-edited cells (new home / test / status / notes) forward from an
      existing matrix, so the census can be re-run without losing annotations.

  check   --matrix docs/parity/capability-matrix.md --root <this repo> [--help-text <file>]
      The CI guard behind deploy/ci/parity-matrix.sh: (i) every row has a target or a
      dropped/lost/pending status (dropped rows name a reason and the archive bundle);
      (ii) every binary, clap subcommand (source enum plus, when given, the parsed `--help`),
      example, Python tool (python/src/jope/{tools,research}/*.{py,sh}), jope-py command and
      systemd unit present in this tree has a row; (iii) `pending-WP<N>` has N in 4..12;
      (iv) a `moved`/`new` row's path-like target exists. Prints counts by status and kind.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

ARCHIVE = "s3://veox-jopedime-archive-use2/git/20260918-jope-dime-all-refs.bundle"
COLUMNS = ["kind", "old path/name", "new home", "test", "status", "notes"]
STATUS_WORDS = ("moved", "new", "lost")
PENDING_RE = re.compile(r"^pending-WP(\d+)$")
WP_RANGE = range(4, 13)

# --------------------------------------------------------------------------------------
# parsers shared by both modes
# --------------------------------------------------------------------------------------


def kebab(name: str) -> str:
    """clap's default subcommand naming (heck to_kebab_case) for a CamelCase variant."""
    out = []
    for i, ch in enumerate(name):
        if i and ch.isupper():
            prev = name[i - 1]
            nxt = name[i + 1] if i + 1 < len(name) else ""
            if prev.islower() or prev.isdigit() or (prev.isupper() and nxt.islower()):
                out.append("-")
        out.append(ch.lower())
    return "".join(out)


def parse_clap_enum(path: Path, enum: str = "Cmd") -> list[dict]:
    """Top-level variants of `enum <enum> { ... }` with their outer attributes."""
    src = path.read_text(errors="replace")
    m = re.search(r"enum %s \{\n(.*?)\n\}\n" % re.escape(enum), src, re.S)
    if not m:
        raise SystemExit(f"{path}: enum {enum} not found")
    rows, attrs = [], []
    for line in m.group(1).splitlines():
        if re.match(r"^    #\[", line):
            attrs.append(line.strip())
            continue
        mm = re.match(r"^    ([A-Z][A-Za-z0-9]*)\s*(\{|,|\()", line)
        if mm:
            variant = mm.group(1)
            name = kebab(variant)
            for a in attrs:
                mname = re.search(r'name\s*=\s*"([^"]+)"', a)
                if a.startswith("#[command") and mname:
                    name = mname.group(1)
            rows.append(
                {
                    "variant": variant,
                    "name": name,
                    "neural": any('feature = "neural"' in a for a in attrs),
                    "hidden": any("hide = true" in a for a in attrs),
                }
            )
            attrs = []
    return rows


def parse_help_commands(text: str) -> list[str]:
    """Subcommand names from clap `--help` output (the `Commands:` block)."""
    names, inside = [], False
    for line in text.splitlines():
        if line.strip() == "Commands:":
            inside = True
            continue
        if inside:
            if not line.strip() or line.startswith("Options"):
                if names:
                    break
                continue
            mm = re.match(r"^  ([a-z0-9][a-z0-9-]*)(\s|$)", line)
            if mm and mm.group(1) != "help":
                names.append(mm.group(1))
    return names


def parse_argparse_commands(path: Path) -> list[str]:
    """Command paths (`data build`, `train`, ...) from an argparse CLI built with add_parser."""
    src = path.read_text(errors="replace")
    sub_owner: dict[str, str | None] = {}  # subparsers var -> parent parser var
    parser_cmd: dict[str, str] = {}  # parser var -> full command path
    for m in re.finditer(r"(\w+)\s*=\s*(\w+)\.add_subparsers\(", src):
        sub_owner[m.group(1)] = m.group(2)
    out = []
    for m in re.finditer(r"(\w+)\s*=\s*(\w+)\.add_parser\(\s*\"([^\"]+)\"", src):
        var, subs, name = m.groups()
        parent = sub_owner.get(subs)
        prefix = parser_cmd.get(parent, "") if parent else ""
        full = f"{prefix} {name}".strip()
        parser_cmd[var] = full
        out.append(full)
    return out


def cargo_bins(root: Path) -> list[tuple[str, str, str, str]]:
    """(package, kind, name, src path) for every bin/example target, via cargo metadata."""
    try:
        meta = subprocess.run(
            ["cargo", "metadata", "--locked", "--no-deps", "--format-version", "1"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"warning: cargo metadata unavailable ({exc}); falling back to src/bin listings", file=sys.stderr)
        return fallback_bins(root)
    out = []
    for pkg in json.loads(meta)["packages"]:
        for t in pkg["targets"]:
            kind = t["kind"][0]
            if kind in ("bin", "example"):
                rel = os.path.relpath(t["src_path"], root)
                out.append((pkg["name"], kind, t["name"], rel))
    return sorted(out)


def fallback_bins(root: Path) -> list[tuple[str, str, str, str]]:
    out = []
    for cargo in sorted(root.glob("crates/**/Cargo.toml")):
        crate = cargo.parent
        text = cargo.read_text(errors="replace")
        mp = re.search(r'^\[package\]\s*\nname\s*=\s*"([^"]+)"', text, re.M)
        if not mp:
            continue
        pkg = mp.group(1)
        if (crate / "src" / "main.rs").exists():
            out.append((pkg, "bin", pkg, str((crate / "src" / "main.rs").relative_to(root))))
        for b in sorted((crate / "src" / "bin").glob("*.rs")):
            out.append((pkg, "bin", b.stem, str(b.relative_to(root))))
        for e in sorted((crate / "examples").glob("*.rs")):
            out.append((pkg, "example", e.stem, str(e.relative_to(root))))
        for m in re.finditer(r'\[\[bin\]\]\s*\nname\s*=\s*"([^"]+)"\s*\npath\s*=\s*"([^"]+)"', text):
            out.append((pkg, "bin", m.group(1), str((crate / m.group(2)).relative_to(root))))
    return sorted(set(out))


def python_tools(root: Path) -> list[str]:
    out = []
    for sub in ("python/src/jope/tools", "python/src/jope/research"):
        d = root / sub
        if not d.is_dir():
            continue
        for p in sorted(d.rglob("*")):
            if p.is_file() and p.suffix in (".py", ".sh") and p.name != "__init__.py":
                out.append(str(p.relative_to(root)))
    return out


def systemd_units(root: Path, base: str = "deploy/systemd") -> list[str]:
    d = root / base
    if not d.is_dir():
        return []
    return sorted(str(p.relative_to(root)) for p in d.rglob("*") if p.is_file() and p.suffix in (".service", ".timer"))


def count_test_fns(path: Path) -> int:
    return len(re.findall(r"^\s*#\[(?:tokio::)?test\]", path.read_text(errors="replace"), re.M))


def pytest_static_ids(root: Path, tests_dir: str = "python/tests") -> dict[str, list[str]]:
    """pytest node ids (relative to python/) of test functions found by AST, per file."""
    out: dict[str, list[str]] = {}
    base = root / tests_dir
    for p in sorted(base.rglob("test_*.py")):
        rel = str(p.relative_to(root / "python"))
        try:
            tree = ast.parse(p.read_text(errors="replace"))
        except SyntaxError as exc:
            print(f"warning: {rel}: {exc}", file=sys.stderr)
            continue
        ids = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test"):
                ids.append(f"{rel}::{node.name}")
            elif isinstance(node, ast.ClassDef) and (node.name.startswith("Test") or any("TestCase" in ast.unparse(b) for b in node.bases)):
                for sub in node.body:
                    if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)) and sub.name.startswith("test"):
                        ids.append(f"{rel}::{node.name}::{sub.name}")
        out[rel] = sorted(ids)
    return out


def py_test_fn_count(path: Path) -> int:
    try:
        tree = ast.parse(path.read_text(errors="replace"))
    except SyntaxError:
        return 0
    n = 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"):
            n += 1
    return n


# --------------------------------------------------------------------------------------
# matrix I/O
# --------------------------------------------------------------------------------------


def esc(cell: str) -> str:
    return cell.replace("|", "\\|").replace("\n", " ").strip()


def unesc(cell: str) -> str:
    return cell.replace("\\|", "|").strip()


def split_row(line: str) -> list[str]:
    parts, cur, i = [], [], 0
    while i < len(line):
        ch = line[i]
        if ch == "\\" and i + 1 < len(line) and line[i + 1] == "|":
            cur.append("|")
            i += 2
            continue
        if ch == "|":
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
        i += 1
    parts.append("".join(cur))
    return [p.strip() for p in parts[1:-1]]


def read_matrix(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        if not line.startswith("| ") or line.startswith("| kind ") or line.startswith("|---"):
            continue
        cells = split_row(line)
        if len(cells) != len(COLUMNS):
            continue
        rows.append(dict(zip(["kind", "old", "new", "test", "status", "notes"], cells)))
    return rows


def status_class(status: str) -> str:
    if status in STATUS_WORDS:
        return status
    if PENDING_RE.match(status):
        return "pending"
    if status.startswith("dropped:"):
        return "dropped"
    return "invalid"


def render_matrix(rows: list[dict], old_sha: str, new_sha: str) -> str:
    by_kind_status: dict[str, Counter] = defaultdict(Counter)
    statuses = Counter()
    no_test = 0
    for r in rows:
        cls = status_class(r["status"])
        by_kind_status[r["kind"]][cls] += 1
        statuses[cls] += 1
        if r["test"] == "none" and cls in ("moved", "new"):
            no_test += 1
    out = []
    out.append("# Capability parity matrix")
    out.append("")
    out.append(
        "One row per entry point of the pre-port tree (`jope-dime` `main` @ `%s`), resolved against this tree "
        "(census base `%s`). Generated by `deploy/ci/parity-census.py census` from the read-only pre-port "
        "checkout, then hand-annotated; `deploy/ci/parity-matrix.sh` (hygiene lane) fails when a row has no "
        "target and no `dropped`/`lost`/`pending-WP<N>` status, when a binary, clap subcommand, example, Python "
        "tool, `jope-py` command or systemd unit exists in this tree without a row, when a `pending-WP<N>` "
        "number is outside 4..12, or when a `moved` row points at a path that does not exist. Re-run the census "
        "with `--merge docs/parity/capability-matrix.md` to keep the hand edits." % (old_sha, new_sha)
    )
    out.append("")
    out.append("Legend — `status`: `moved` (present here, target + test recorded), `new` (no pre-port counterpart), "
               "`pending-WP<N>` (owned by that work package of `FABLE_GITHUB_PORT_PLAN.md`), `dropped: <reason> "
               "(archive: <key>)` (only where the plan names it dropped; every dropped row cites the all-refs bundle "
               "`%s`), `lost` (lineage-A state the plan declares unrecoverable, with the surviving archive receipts "
               "in `notes`). `test`: a nextest id `<package>::<binary> <name>` / a nextest binary id with its test "
               "count, a pytest node id relative to `python/`, or `none`. `new home` is a path in this repo, "
               "optionally with the command in parentheses, or `dropped` / `lost` / `pending`." % ARCHIVE)
    out.append("")
    out.append("## Counts")
    out.append("")
    classes = ["moved", "new", "pending", "dropped", "lost"]
    out.append("| kind | rows | " + " | ".join(classes) + " |")
    out.append("|---|---:|" + "---:|" * len(classes))
    for kind in sorted(by_kind_status):
        c = by_kind_status[kind]
        out.append(f"| {kind} | {sum(c.values())} | " + " | ".join(str(c[s]) for s in classes) + " |")
    out.append(f"| **total** | **{len(rows)}** | " + " | ".join(f"**{statuses[s]}**" for s in classes) + " |")
    out.append("")
    out.append(f"`moved`/`new` rows without a covering test: {no_test}.")
    out.append("")
    out.append("## Rows")
    out.append("")
    out.append("| " + " | ".join(COLUMNS) + " |")
    out.append("|" + "---|" * len(COLUMNS))
    for r in rows:
        out.append("| " + " | ".join(esc(r[k]) for k in ("kind", "old", "new", "test", "status", "notes")) + " |")
    out.append("")
    return "\n".join(out)


# --------------------------------------------------------------------------------------
# census
# --------------------------------------------------------------------------------------


class Census:
    def __init__(self, old: Path, new: Path, nextest: dict | None, pytest_ids: dict[str, list[str]]):
        self.old, self.new = old, new
        self.rows: list[dict] = []
        # nextest: binary id -> {package, kind, tests: set}
        self.suites: dict[str, dict] = {}
        self.test_ids: set[str] = set()
        if nextest:
            for bid, s in nextest.get("rust-suites", {}).items():
                names = set(s.get("testcases", {}).keys())
                self.suites[bid] = {"package": s.get("package-name"), "kind": s.get("kind"), "tests": names}
                self.test_ids.update(f"{bid} {n}" for n in names)
        self.pytest_ids = pytest_ids
        self.pytest_files = {f: ids for f, ids in pytest_ids.items()}
        self.rs_literal_index = self._index_rust_test_literals()

    # ---- helpers -------------------------------------------------------------------
    def add(self, kind, old, new, test, status, notes=""):
        self.rows.append({"kind": kind, "old": old, "new": new, "test": test, "status": status, "notes": notes})

    def exists(self, rel: str) -> bool:
        return (self.new / rel).exists()

    def dropped(self, reason: str) -> str:
        return f"dropped: {reason} (archive: {ARCHIVE})"

    def _index_rust_test_literals(self) -> dict[str, set[str]]:
        """literal string -> nextest ids of the tests whose body contains it, in files that spawn the engine CLI."""
        index: dict[str, set[str]] = defaultdict(set)
        if not self.test_ids:
            return index
        crates = {"crates/jope-engine": "jope-parallel", "crates/jope-campaigns": "jope-dime-regressor"}
        for crate, pkg in crates.items():
            for p in sorted((self.new / crate).rglob("*.rs")):
                text = p.read_text(errors="replace")
                if "CARGO_BIN_EXE_jope-parallel" not in text:
                    continue  # only tests that spawn the engine CLI count as subcommand coverage
                rel = p.relative_to(self.new / crate).as_posix()
                if rel.startswith("tests/"):
                    if "/" in rel[len("tests/"):]:
                        continue  # support modules
                    bid = f"{pkg}::{p.stem}"
                    mod_prefix = ""
                elif rel.startswith("src/bin/"):
                    bid = f"{pkg}::bin/{p.stem}"
                    mod_prefix = ""
                elif rel.startswith("src/"):
                    bid = pkg
                    parts = rel[len("src/"):-3].split("/")
                    if parts[-1] in ("lib", "main", "mod"):
                        parts = parts[:-1]
                    mod_prefix = "::".join(parts)
                elif rel.startswith("examples/"):
                    continue
                else:
                    continue
                if bid not in self.suites:
                    continue
                suite = self.suites[bid]["tests"]
                current, pending_test = None, False
                for line in text.splitlines():
                    s = line.strip()
                    if re.match(r"#\[(?:tokio::)?test", s):
                        pending_test = True
                        continue
                    mf = re.match(r"(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?fn\s+([A-Za-z0-9_]+)", s)
                    if mf:
                        if pending_test:
                            fn = mf.group(1)
                            # resolve to an existing nextest name (module path unknown for unit tests)
                            cands = [n for n in suite if n == fn or n.endswith("::" + fn)]
                            if mod_prefix:
                                pref = [n for n in cands if n.startswith(mod_prefix)]
                                cands = pref or cands
                            current = f"{bid} {cands[0]}" if len(cands) >= 1 else None
                        else:
                            current = None
                        pending_test = False
                        continue
                    if current:
                        for lit in re.findall(r'"([a-z][a-z0-9-]{2,})"', s):
                            index[lit].add(current)
        return index

    def rust_tests_for_literal(self, lit: str) -> str:
        ids = sorted(self.rs_literal_index.get(lit, ()))
        if not ids:
            return "none"
        if len(ids) <= 2:
            return "; ".join(ids)
        return f"{ids[0]} (+{len(ids) - 1} more)"

    def bin_tests(self, pkg: str, name: str, src_rel: str) -> str:
        bid = f"{pkg}::bin/{name}"
        if bid in self.suites and self.suites[bid]["tests"]:
            return f"{bid} ({len(self.suites[bid]['tests'])} tests)"
        if src_rel.endswith("src/main.rs") and pkg in self.suites and self.suites[pkg]["tests"]:
            pass
        # integration tests spawning the binary
        hits = []
        env = f"CARGO_BIN_EXE_{name}"
        for p in sorted(self.new.glob("crates/*/tests/*.rs")):
            if env in p.read_text(errors="replace"):
                bid2 = f"{pkg}::{p.stem}"
                if bid2 in self.suites:
                    hits.append(f"{bid2} ({len(self.suites[bid2]['tests'])} tests)")
        if hits:
            return "; ".join(hits[:3])
        n = count_test_fns(self.new / src_rel) if self.exists(src_rel) else 0
        if n:
            return f"{bid} ({n} tests, not in the nextest list given)"
        return "none"

    def py_sibling_note(self, new_rel: str) -> str:
        p = self.new / new_rel
        sib = p.with_name("test_" + p.name)
        if sib.exists():
            n = py_test_fn_count(sib)
            return f"sibling {sib.relative_to(self.new).as_posix()} ({n} test functions) is not collected (pytest testpaths=tests)"
        return ""

    # ---- (a) engine subcommands -------------------------------------------------------
    def engine_subcommands(self):
        old_cmds = parse_clap_enum(self.old / "parallel/src/main.rs")
        new_cmds = {c["variant"]: c for c in parse_clap_enum(self.new / "crates/jope-engine/src/main.rs")}
        for c in old_cmds:
            flags = []
            if c["neural"]:
                flags.append("`neural` feature (self-hosted only)")
            if c["hidden"]:
                flags.append("hidden from --help")
            nc = new_cmds.get(c["variant"])
            if nc:
                new = f"crates/jope-engine/src/main.rs (jope-parallel {nc['name']})"
                status = "moved"
                note = "; ".join(flags + ["`jope engine <sub>` umbrella is WP5"])
            else:
                new, status, note = "pending", "pending-WP5", "; ".join(flags + ["variant absent from crates/jope-engine"])
            self.add("engine-subcommand", f"jope-parallel {c['name']}", new, self.rust_tests_for_literal(c["name"]), status, note)

    # ---- (b) engine binaries + examples -----------------------------------------------
    def engine_binaries(self):
        self.add(
            "engine-binary",
            "parallel/src/main.rs (jope-parallel)",
            "crates/jope-engine/src/main.rs (jope-parallel)",
            self.bin_tests("jope-parallel", "jope-parallel", "crates/jope-engine/src/main.rs"),
            "moved",
            "88 subcommands, one row each above",
        )
        for p in sorted((self.old / "parallel/src/bin").glob("*.rs")):
            rel = f"crates/jope-engine/src/bin/{p.name}"
            if p.stem == "jope-witness-ssh":
                self.add("engine-binary", f"parallel/src/bin/{p.name}", rel if self.exists(rel) else "pending",
                         self.bin_tests("jope-parallel", p.stem, rel), "pending-WP5",
                         "plan lists it with the dropped ssh/node scripts (forced-OpenSSH witness entrypoint); still compiled here, removal lands with the WP5 CLI consolidation")
                continue
            if self.exists(rel):
                self.add("engine-binary", f"parallel/src/bin/{p.name}", rel, self.bin_tests("jope-parallel", p.stem, rel), "moved", "")
            else:
                self.add("engine-binary", f"parallel/src/bin/{p.name}", "pending", "none", "pending-WP5", "binary absent from crates/jope-engine")
        for p in sorted((self.old / "parallel/examples").glob("*.rs")):
            rel = f"crates/jope-engine/examples/{p.name}"
            if self.exists(rel):
                self.add("engine-example", f"parallel/examples/{p.name}", rel, "none", "moved", f"`cargo run -p jope-parallel --example {p.stem}`")
            else:
                self.add("engine-example", f"parallel/examples/{p.name}", "pending", "none", "pending-WP5", "example absent from crates/jope-engine")

    # ---- (c) root binaries --------------------------------------------------------------
    SELFPLAY_PREFIXES = ("jope_selfplay_", "jope_task_learning_controller", "jope_training_", "jope_queue_control",
                         "jope_ledger_reconciler", "jope_flywheel_liveness")

    def root_binaries(self):
        self.add("root-binary", "src/main.rs (jope-dime-regressor)", "crates/jope-campaigns/src/main.rs (jope-dime-regressor)",
                 self.bin_tests("jope-dime-regressor", "jope-dime-regressor", "crates/jope-campaigns/src/main.rs"), "moved",
                 "root crate main; WP5 makes it the argv[0] dispatcher")
        for p in sorted((self.old / "src/bin").glob("*.rs")):
            rel = f"crates/jope-campaigns/src/bin/{p.name}"
            lane = "jope-selfplay" if p.stem.startswith(self.SELFPLAY_PREFIXES) else "jope-campaigns"
            note = f"WP5 split pending: becomes a subcommand of `{lane}` (argv[0] symlink keeps the name)"
            if self.exists(rel):
                self.add("root-binary", f"src/bin/{p.name}", rel, self.bin_tests("jope-dime-regressor", p.stem, rel), "moved", note)
            else:
                self.add("root-binary", f"src/bin/{p.name}", "pending", "none", "pending-WP5", note + "; file absent")

    # ---- (d) jope-prime CLI -------------------------------------------------------------
    def jope_prime_cli(self):
        old_cli = self.old / "parallel/assets/source/jope-prime/src/jope/cli.py"
        new_cli = self.new / "python/src/jope/cli.py"
        new_cmds = set(parse_argparse_commands(new_cli)) if new_cli.exists() else set()
        for cmd in parse_argparse_commands(old_cli):
            if cmd in new_cmds:
                self.add("jope-prime-command", f"jope {cmd}", f"python/src/jope/cli.py (jope-py {cmd})", "none", "moved",
                         "console script `jope-py` (python/pyproject.toml); no CLI-level test in python/tests")
            else:
                self.add("jope-prime-command", f"jope {cmd}", "pending", "none", "pending-WP6", "command absent from python/src/jope/cli.py")

    # ---- (e) tools and scripts ----------------------------------------------------------
    DROPPED_SH = {
        "code_publish.sh": "node-side code publisher over ssh",
        "lane-publish.sh": "node lane publisher over ssh",
        "node-mirror.sh": "node mirror pull over ssh",
        "tui-remote.sh": "remote TUI over ssh (a frozen copy remains inside deploy/kernel-v1/source/tools/)",
        "tui-recovery.sh": "remote TUI recovery over ssh",
        "research-readouts.sh": "xbabe2 research-readouts cron wrapper",
        "run_all_regression_evolution_xbabes.sh": "xbabe fleet ssh fan-out",
    }

    def tools(self):
        base = self.old / "parallel/tools"
        entries = sorted(base.glob("*.py")) + sorted(base.glob("seed_dispatch/*.py")) + sorted(base.glob("*.sh"))
        for p in entries:
            rel_old = p.relative_to(self.old).as_posix()
            sub = p.relative_to(base).as_posix()
            if p.suffix == ".sh":
                if p.name in self.DROPPED_SH:
                    self.add("tool-sh", rel_old, "dropped", "none", self.dropped(self.DROPPED_SH[p.name] + "; plan: ssh/node scripts dropped in the history filter"), "")
                elif self.exists(f"python/src/jope/tools/{sub}"):
                    note = ("sudo-installs two engine units from a /home/ubuntu checkout; superseded by the templated deploy/systemd install (WP4)"
                            if p.name == "install-services.sh" else "ops shell script kept beside the tools")
                    self.add("tool-sh", rel_old, f"python/src/jope/tools/{sub}", "none", "moved", note)
                else:
                    self.add("tool-sh", rel_old, "pending", "none", "pending-WP4", "no counterpart in this tree")
                continue
            kind = "tool-test" if p.name.startswith("test_") and py_test_fn_count(p) else "tool-py"
            new_rel = f"python/src/jope/tools/{sub}"
            if p.name == "family_growth_policy.py":
                new_rel = "crates/jope-engine/policy/family_growth_policy.py"
            if self.exists(new_rel):
                if kind == "tool-test":
                    n = py_test_fn_count(self.new / new_rel)
                    self.add(kind, rel_old, new_rel, "none", "moved",
                             f"{n} test functions travel with the tool but are not collected (pytest testpaths=tests); collection decision is WP6 triage")
                else:
                    note = self.py_sibling_note(new_rel)
                    if p.name == "family_growth_policy.py":
                        note = "kept crate-local (read by jope-engine); " + self.py_sibling_note("python/src/jope/tools/family_growth_policy.py")
                    self.add(kind, rel_old, new_rel, "none", "moved", note)
            else:
                self.add(kind, rel_old, "pending", "none", "pending-WP6", "no counterpart in this tree")
        # the Rust dev tool that lived under parallel/tools
        if self.exists("xtools/registry-dump-legacy/src/main.rs"):
            self.add("tool-dir", "parallel/tools/jain-math-registry-dump/", "xtools/registry-dump-legacy/", "none", "moved",
                     "own Cargo workspace (dev-only regenerator of jain-math-executable-9fa594e2.json); byte-identical main.rs")
        else:
            self.add("tool-dir", "parallel/tools/jain-math-registry-dump/", "pending", "none", "pending-WP10", "")
        for wire in ("parallel/tools/AGENT_CHAT.md", "parallel/tools/AGENT_CHAT.md.lock", "parallel/AGENT_CHAT.md", "AGENT_CHAT.md", "REALTIME_AGENT_CHAT.md"):
            if (self.old / wire).exists():
                self.add("wire", wire, "dropped", "none", self.dropped("agent coordination wire, not a capability; plan: AGENT_CHAT wires dropped"), "")

    def scripts(self):
        base = self.old / "scripts"
        for p in sorted(base.glob("*.py")) + sorted(base.glob("*.sh")):
            rel_old = p.relative_to(self.old).as_posix()
            new_rel = f"python/src/jope/research/{p.name}"
            if p.suffix == ".sh":
                if p.name in self.DROPPED_SH:
                    self.add("script-sh", rel_old, "dropped", "none", self.dropped(self.DROPPED_SH[p.name] + "; plan: ssh/node scripts dropped in the history filter"), "")
                elif self.exists(new_rel):
                    self.add("script-sh", rel_old, new_rel, "none", "moved", "ops shell script; plan home is xtools/ops-sh/ (kept beside the research drivers for now)")
                else:
                    self.add("script-sh", rel_old, "pending", "none", "pending-WP6", "no counterpart in this tree")
                continue
            ntests = py_test_fn_count(p)
            kind = "script-test" if p.name.startswith("test_") and ntests and not re.search(r"^def main\(|__main__", p.read_text(errors="replace"), re.M) else "script-py"
            if not self.exists(new_rel):
                self.add(kind, rel_old, "pending", "none", "pending-WP6", "no counterpart in this tree")
                continue
            # a python/tests module that exercises this driver
            cover = self.pytest_files.get(f"tests/test_{p.stem}.py") or self.pytest_files.get(f"tests/{p.name}")
            test = "none"
            notes = []
            if cover:
                test = f"tests/test_{p.stem}.py ({len(cover)} tests)" if f"tests/test_{p.stem}.py" in self.pytest_files else f"tests/{p.name} ({len(cover)} tests)"
            if ntests:
                notes.append(f"{ntests} in-file test functions not collected (pytest testpaths=tests)")
            if kind == "script-py":
                notes.append("research driver (no CI beyond import)")
            self.add(kind, rel_old, new_rel, test, "moved", "; ".join(notes))
        for p in sorted((base / "review").glob("*")):
            rel_old = p.relative_to(self.old).as_posix()
            new_rel = f"xtools/ops-sh/review/{p.name}"
            if self.exists(new_rel):
                self.add("script-review", rel_old, new_rel, "none", "moved", "review-epoch shell tooling; shellcheck-linted by deploy/ci/check.sh")
            else:
                self.add("script-review", rel_old, "pending", "none", "pending-WP6", "no counterpart in this tree")
        for p in sorted((self.old / "ops/ci").glob("*.sh")):
            rel_old = p.relative_to(self.old).as_posix()
            new_rel = f"deploy/ci/{p.name}"
            if self.exists(new_rel):
                self.add("ops-sh", rel_old, new_rel, "none", "moved", "")
            else:
                self.add("ops-sh", rel_old, "pending", "none", "pending-WP8", "no counterpart in this tree")

    # ---- (f) Justfile ---------------------------------------------------------------------
    JUST = {
        "required": ("justfile (required = ci-pr)", "moved", "alias kept; hosted gate is `just ci-pr` = hygiene + fmt + clippy + test-pr + python-check"),
        "fast": ("justfile (clippy)", "moved", "`cargo check --all-targets` is subsumed by `clippy --workspace --all-targets`"),
        "format": ("justfile (fmt)", "moved", ""),
        "clippy": ("justfile (clippy)", "moved", "now workspace-wide"),
        "test": ("justfile (test-main)", "moved", "workspace-wide nextest `main` profile; `test-pr` is the hosted tier"),
        "kernel-v1-product-build": ("deploy/ci/kernel-v1-product-build.sh", "moved", "no just recipe yet; the script is the entry point (plan item 1: builds the current engine by default)"),
    }

    def justfile(self):
        text = (self.old / "Justfile").read_text()
        newjust = (self.new / "justfile").read_text() if self.exists("justfile") else ""
        recipes = re.findall(r"^([A-Za-z_][A-Za-z0-9_-]*)(?:\s+[^:\n]*)?:\s*$", text, re.M)
        new_recipes = set(re.findall(r"^([A-Za-z_][A-Za-z0-9_-]*)(?:\s+[^:\n]*)?:", newjust, re.M))
        for r in recipes:
            new, status, note = self.JUST.get(r, ("pending", "pending-WP9", "no counterpart recipe"))
            m = re.match(r"justfile \((.+)\)", new)
            if m and m.group(1).split()[0] not in new_recipes:
                new, status, note = "pending", "pending-WP9", f"recipe `{m.group(1)}` missing from justfile"
            self.add("justfile-recipe", f"Justfile:{r}", new, "none", status, note)

    # ---- (g) systemd -------------------------------------------------------------------------
    def systemd(self):
        unit_test = "none"
        b8 = ("jope-dime-regressor::systemd_unit_paths (2 tests) exists but panics on the missing crates/jope-campaigns/systemd "
              "dir (B8, timing run 4235db6); re-pointed at deploy/systemd in WP4")
        for p in sorted((self.old / "systemd").glob("*")):
            rel = f"deploy/systemd/selfplay/{p.name}"
            if self.exists(rel):
                self.add("systemd-unit", f"systemd/{p.name}", rel, unit_test, "moved", b8 + "; templating (@JOPE_BUNDLE@ …) is WP4")
            else:
                self.add("systemd-unit", f"systemd/{p.name}", "pending", "none", "pending-WP4", "unit absent from deploy/systemd/selfplay")
        for p in sorted((self.old / "parallel/systemd").glob("*")):
            if p.is_dir():
                continue
            rel = f"deploy/systemd/engine/{p.name}"
            if self.exists(rel):
                self.add("systemd-unit", f"parallel/systemd/{p.name}", rel, "none", "moved", "engine unit; templating is WP4")
            else:
                self.add("systemd-unit", f"parallel/systemd/{p.name}", "pending", "none", "pending-WP4", "unit absent from deploy/systemd/engine")
        for p in sorted((self.old / "parallel/systemd/xbabe2").glob("*")):
            self.add("systemd-unit", f"parallel/systemd/xbabe2/{p.name}", "dropped", "none",
                     self.dropped("xbabe2-only node unit (lane publisher / node mirror / research readouts); plan B12"), "")
        for p in sorted((self.old / "reference/systemd-pending").glob("*")):
            rel = f"deploy/systemd/pending/{p.name}"
            if self.exists(rel):
                self.add("systemd-unit", f"reference/systemd-pending/{p.name}", rel, "none", "moved", "never-activated unit kept for reference")
            else:
                self.add("systemd-unit", f"reference/systemd-pending/{p.name}", "pending", "none", "pending-WP4", "")

    # ---- lineage-A lost state --------------------------------------------------------------
    LOST = [
        ("test_exact_gen0/corpus/strict-test-exact.sqlite", "label authority of the xbabe2 self-play flywheel"),
        ("test_exact_gen0/ops/pair-registry.sqlite", "pair identities"),
        ("test_exact_gen0/training/immutable/checkpoints/*", "checkpoint descendants of fbcd5459… (the 109 MB historical warm start survives in the assets class)"),
        ("test_exact_gen0/training/{learning,accepted,milestone,accepted-by-task}.json", "promotion pointers"),
        ("test_exact_gen0/sealed-panel-pool.v2.json + consumption receipts", "sealed holdout panel"),
        ("test_exact_gen0/ops/task-elo.v1.json", "lineage-A Elo ladders (cannot be recomputed)"),
        ("test_exact_gen0/catalog/current.json", "catalog pointer"),
    ]
    LOST_RECEIPTS = ("surviving receipts: /home/ubuntu/preserved-archives/20260908-F-D-E/{D-runtime.tar.zst 48.9 GB (2.6 M xbabe2 task files), "
                     "D-recovery.tar.zst 19.5 GB, SHA256SUMS} (+ copy /mnt/fast-scratch/storage-offload-stage-20260908-1420), the four corpus "
                     "snapshots snapshot-004754860 … snapshot-009915070, genebank.db (best_scores 4,948 / observations 2.5 M); rebuild order in the "
                     "plan's Lost-state contingency; no lineage-A checkpoint can be retroactively accepted")

    def lost_state(self):
        for path, what in self.LOST:
            self.add("state", path, "lost", "none", "lost", f"{what}; {self.LOST_RECEIPTS}")

    # ---- entries that exist only in this tree -------------------------------------------------
    def new_only(self):
        seen_new = {r["new"].split(" (")[0] for r in self.rows}
        origins = {
            "jope-cli": ("(new) umbrella CLI", "new", "plan: `jope` umbrella (models/datasets/learn/engine verbs land in WP5/WP7/WP11)"),
            "starforge": ("jain-core/crates/starforge (import, WP3)", "moved", "vendored encoder; tests gated on the starforge weights"),
            "jain-smartcluster-daemon": ("jain-smartcluster@715128f", "moved", "vendored smartcluster crate"),
            "jain-worker": ("jain-smartcluster@715128f", "moved", "vendored smartcluster crate"),
            "jainnode": ("jain-smartcluster@715128f", "moved", "vendored smartcluster crate"),
            "scq": ("jain-smartcluster@715128f", "moved", "vendored smartcluster crate (the SCQ evaluator the flywheel calls)"),
            "feat-math": ("vendor/feat-math", "moved", "crates/feat-math example"),
        }
        for pkg, kind, name, src in cargo_bins(self.new):
            if src in seen_new:
                continue
            origin, status, note = origins.get(pkg, ("(new)", "new", ""))
            old = f"{origin}: {name}" if not origin.startswith("(new)") else origin
            if pkg == "feat-math":
                old = f"vendor/feat-math/examples/{Path(src).name}"
            test = "none"
            if kind == "bin":
                test = self.bin_tests(pkg, name, src)
            elif pkg in self.suites and self.suites[pkg]["tests"]:
                test = "none"
            self.add(f"{'vendored' if status == 'moved' else 'new'}-{kind}", old, f"{src} ({name})", test, status, note)
            seen_new.add(src)
        for rel in python_tools(self.new):
            if rel not in seen_new:
                self.add("new-tool", "(new)", rel, "none", "new", "")
        for rel in systemd_units(self.new):
            if rel not in seen_new:
                self.add("new-unit", "(new)", rel, "none", "new", "")

    def run(self):
        self.engine_subcommands()
        self.engine_binaries()
        self.root_binaries()
        self.jope_prime_cli()
        self.tools()
        self.scripts()
        self.justfile()
        self.systemd()
        self.lost_state()
        self.new_only()
        return self.rows


def git_sha(path: Path) -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short=10", "HEAD"], cwd=path, check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def merge_rows(rows: list[dict], existing: list[dict]) -> list[dict]:
    """Hand-edited cells win for rows that already exist (keyed by kind + old)."""
    by_key = {(r["kind"], r["old"]): r for r in existing}
    merged = []
    for r in rows:
        e = by_key.pop((r["kind"], r["old"]), None)
        if e:
            r = dict(r)
            for k in ("new", "test", "status", "notes"):
                if e[k] != r[k]:
                    r[k] = e[k]
        merged.append(r)
    for key, e in by_key.items():
        print(f"warning: row no longer produced by the census, kept verbatim: {key}", file=sys.stderr)
        merged.append(e)
    return merged


def cmd_census(a):
    old, new = Path(a.old).resolve(), Path(a.new).resolve()
    nextest = json.load(open(a.nextest_list)) if a.nextest_list else None
    pytest_ids = pytest_static_ids(new)
    if a.pytest_ids:
        pytest_ids = defaultdict(list)
        for line in Path(a.pytest_ids).read_text().splitlines():
            if "::" in line:
                pytest_ids[line.split("::")[0]].append(line.strip())
    rows = Census(old, new, nextest, pytest_ids).run()
    if a.merge and Path(a.merge).exists():
        rows = merge_rows(rows, read_matrix(Path(a.merge)))
    text = render_matrix(rows, git_sha(old), git_sha(new))
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(text)
    print(f"wrote {a.output}: {len(rows)} rows")
    return 0


# --------------------------------------------------------------------------------------
# check
# --------------------------------------------------------------------------------------


def cmd_check(a):
    root = Path(a.root).resolve()
    rows = read_matrix(Path(a.matrix))
    problems: list[str] = []
    targets: set[str] = set()
    parens: set[str] = set()
    for i, r in enumerate(rows, 1):
        where = f"row {i} [{r['kind']} | {r['old']}]"
        cls = status_class(r["status"])
        if cls == "invalid":
            problems.append(f"{where}: invalid status `{r['status']}`")
            continue
        new = r["new"]
        head = new.split(" (")[0].strip()
        for m in re.finditer(r"\(([^)]*)\)", new):
            parens.add(m.group(1).strip())
        if head:
            targets.add(head)
        if cls == "pending":
            n = int(PENDING_RE.match(r["status"]).group(1))
            if n not in WP_RANGE:
                problems.append(f"{where}: pending-WP{n} is outside WP4..WP12")
        elif cls == "dropped":
            if ARCHIVE.split("/git/")[0] not in r["status"]:
                problems.append(f"{where}: dropped without the archive location")
            if len(r["status"]) < len("dropped: x (archive: )") + 20:
                problems.append(f"{where}: dropped without a reason")
            if new not in ("dropped", "-", ""):
                problems.append(f"{where}: dropped row still names a target `{new}`")
        elif cls == "lost":
            if new != "lost":
                problems.append(f"{where}: lost row must have new home `lost`")
        else:  # moved / new
            if not head or head in ("dropped", "lost", "pending", "-", "none"):
                problems.append(f"{where}: status `{r['status']}` but no target")
            elif "/" in head and not (root / head).exists():
                problems.append(f"{where}: target `{head}` does not exist")
    # (ii) everything in this tree must have a row
    missing: list[str] = []
    main_rs = root / "crates/jope-engine/src/main.rs"
    if main_rs.exists():
        for c in parse_clap_enum(main_rs):
            if f"jope-parallel {c['name']}" not in parens:
                missing.append(f"clap subcommand: jope-parallel {c['name']}")
    if a.help_text:
        for name in parse_help_commands(Path(a.help_text).read_text()):
            if f"jope-parallel {name}" not in parens:
                missing.append(f"clap subcommand (from --help): jope-parallel {name}")
    bins = cargo_bins(root)
    for pkg, kind, name, src in bins:
        if src not in targets:
            missing.append(f"{kind} {pkg}/{name}: {src}")
    for rel in python_tools(root):
        if rel not in targets:
            missing.append(f"python tool: {rel}")
    cli = root / "python/src/jope/cli.py"
    if cli.exists():
        for cmd in parse_argparse_commands(cli):
            if f"jope-py {cmd}" not in parens:
                missing.append(f"jope-py command: {cmd}")
    for rel in systemd_units(root):
        if rel not in targets:
            missing.append(f"systemd unit: {rel}")
    for m in missing:
        problems.append(f"no row for {m}")
    # report
    by_status = Counter(status_class(r["status"]) for r in rows)
    by_kind = Counter(r["kind"] for r in rows)
    print(f"capability matrix: {len(rows)} rows")
    print("by status: " + ", ".join(f"{k}={v}" for k, v in sorted(by_status.items())))
    print("by kind: " + ", ".join(f"{k}={v}" for k, v in sorted(by_kind.items())))
    print(f"this tree: {sum(1 for b in bins if b[1] == 'bin')} binaries, {sum(1 for b in bins if b[1] == 'example')} examples, "
          f"{len(parse_clap_enum(main_rs)) if main_rs.exists() else 0} clap subcommands, {len(python_tools(root))} python tools, "
          f"{len(systemd_units(root))} systemd units")
    if problems:
        print(f"\n{len(problems)} problem(s):")
        for p in problems:
            print("  - " + p)
        return 1
    print("ok: every row resolved and every entry point of this tree has a row")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    c = sub.add_parser("census")
    c.add_argument("--old", required=True)
    c.add_argument("--new", default=".")
    c.add_argument("--nextest-list")
    c.add_argument("--pytest-ids")
    c.add_argument("--merge")
    c.add_argument("-o", "--output", default="docs/parity/capability-matrix.md")
    c.set_defaults(fn=cmd_census)
    k = sub.add_parser("check")
    k.add_argument("--matrix", default="docs/parity/capability-matrix.md")
    k.add_argument("--root", default=".")
    k.add_argument("--help-text")
    k.set_defaults(fn=cmd_check)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
