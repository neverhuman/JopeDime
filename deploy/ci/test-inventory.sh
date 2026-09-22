#!/usr/bin/env bash
# deploy/ci/test-inventory.sh — test-inventory guard (plan: red-team pass 2 item 7, WP10).
#
# Records every test id per crate/package into docs/parity/test-inventory.json and fails when a
# test that the committed inventory lists is no longer present, unless the inventory is updated in
# the same change (`--update` rewrites the file). Added tests are reported (run `--update`). The
# committed floors (Rust >= 966, Python >= 214, from the plan's Verification item 9) are enforced.
#
#   Rust:   cargo nextest list --workspace --locked -P main --message-format json
#           (falls back to --config-file deploy/ci/nextest-inventory.toml when the tree's
#           .config/nextest.toml does not load; JOPE_BUILD_JOBS caps --build-jobs)
#   Python: `uv sync --frozen --extra dev --extra cpu && uv run pytest --collect-only -q` when
#           python/uv.lock is committed and uv is installed; only with --static-python use
#           a stdlib AST scan of python/tests for `def test_*` — the JSON records which mode
#           produced it, and parametrized ids are compared with their `[...]` suffix stripped so the
#           two modes agree.
#
# Options: --update  rewrite docs/parity/test-inventory.json
#          --no-rust / --no-python  skip one half (the other half of the file is kept verbatim)
#          --static-python  never invoke uv/pytest (hygiene lane)
set -euo pipefail
cd "$(dirname "$0")/../.."

inventory=docs/parity/test-inventory.json
update=0 rust=1 python=1 static_python=0
for arg in "$@"; do
  case "$arg" in
    --update) update=1 ;;
    --no-rust) rust=0 ;;
    --no-python) python=0 ;;
    --static-python) static_python=1 ;;
    -h|--help) sed -n '2,22p' "$0"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

rust_cmd_used=""
if [ "$rust" = 1 ]; then
  cmd=(cargo nextest list --workspace --locked -P main --message-format json)
  if [ -n "${JOPE_BUILD_JOBS:-}" ]; then cmd+=(--build-jobs "$JOPE_BUILD_JOBS"); fi
  if ! "${cmd[@]}" > "$tmp/nextest.json" 2> "$tmp/nextest.err"; then
    if grep -q 'failed to parse profile' "$tmp/nextest.err"; then
      echo "note: the tree's .config/nextest.toml does not load ($(grep -m1 -o 'failed to parse profile[^ ]*' "$tmp/nextest.err")); listing with deploy/ci/nextest-inventory.toml"
      cmd+=(--config-file deploy/ci/nextest-inventory.toml)
      if ! "${cmd[@]}" > "$tmp/nextest.json" 2> "$tmp/nextest.err"; then
        tail -40 "$tmp/nextest.err" >&2
        exit 1
      fi
    else
      tail -40 "$tmp/nextest.err" >&2
      exit 1
    fi
  fi
  rust_cmd_used="${cmd[*]}"
fi

py_mode=""
if [ "$python" = 1 ]; then
  py_mode=static-scan
  if [ "$static_python" = 0 ]; then
    [ -f python/uv.lock ] || { echo 'Python collection requires python/uv.lock' >&2; exit 1; }
    command -v uv >/dev/null || { echo 'Python collection requires uv' >&2; exit 127; }
    (cd python && uv sync --frozen --extra dev --extra cpu) > "$tmp/uv-sync.log" 2>&1 || {
      rc=$?; cat "$tmp/uv-sync.log" >&2; exit "$rc";
    }
    (cd python && uv run --no-sync pytest --collect-only -q -p no:cacheprovider) > "$tmp/pytest.txt" 2> "$tmp/pytest.err" || {
      rc=$?; cat "$tmp/pytest.txt" "$tmp/pytest.err" >&2; exit "$rc";
    }
    py_mode=pytest-collect
  fi
fi

INVENTORY="$inventory" UPDATE="$update" RUST="$rust" PYTHON="$python" PY_MODE="$py_mode" \
RUST_CMD="$rust_cmd_used" TMP="$tmp" python3 - <<'PY'
import ast, json, os, re, sys
from pathlib import Path

inventory = Path(os.environ["INVENTORY"])
update = os.environ["UPDATE"] == "1"
do_rust = os.environ["RUST"] == "1"
do_python = os.environ["PYTHON"] == "1"
py_mode = os.environ["PY_MODE"]
tmp = Path(os.environ["TMP"])
FLOORS = {"rust": 966, "python": 214}

committed = json.loads(inventory.read_text()) if inventory.exists() else {}
fresh = {"schema": "jope.test-inventory/v1", "floors": committed.get("floors", FLOORS)}

# ---- Rust ---------------------------------------------------------------------------------------
if do_rust:
    listing = json.loads((tmp / "nextest.json").read_text())
    packages = {}
    total = ignored = 0
    for bid, suite in sorted(listing["rust-suites"].items()):
        pkg = packages.setdefault(suite["package-name"], {"count": 0, "ignored": 0, "binaries": {}})
        tests = sorted(suite.get("testcases", {}))
        n_ign = sum(1 for t in suite.get("testcases", {}).values() if t.get("ignored"))
        pkg["binaries"][bid] = {"kind": suite["kind"], "count": len(tests), "ignored": n_ign, "tests": tests}
        pkg["count"] += len(tests)
        pkg["ignored"] += n_ign
        total += len(tests)
        ignored += n_ign
    fresh["rust"] = {"command": os.environ["RUST_CMD"], "count": total, "ignored": ignored, "packages": dict(sorted(packages.items()))}
else:
    fresh["rust"] = committed.get("rust", {"command": "", "count": 0, "ignored": 0, "packages": {}})

# ---- Python -------------------------------------------------------------------------------------
def static_ids():
    out = {}
    base = Path("python/tests")
    for p in sorted(base.rglob("test_*.py")):
        rel = p.relative_to("python").as_posix()
        try:
            tree = ast.parse(p.read_text(errors="replace"))
        except SyntaxError as exc:
            raise SystemExit(f"static inventory cannot parse {rel}: {exc}") from exc
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

def collected_ids():
    out = {}
    for line in (tmp / "pytest.txt").read_text().splitlines():
        line = line.strip()
        if "::" in line and not line.startswith(("=", "warning", "ERROR")):
            out.setdefault(line.split("::")[0], []).append(line)
    return {k: sorted(v) for k, v in sorted(out.items())}

if do_python:
    files = collected_ids() if py_mode == "pytest-collect" else static_ids()
    command = ("cd python && uv sync --frozen --extra dev --extra cpu && uv run pytest --collect-only -q"
               if py_mode == "pytest-collect" else "explicit --static-python: stdlib AST scan of python/tests/**/test_*.py; not collection or execution")
    total = sum(len(v) for v in files.values())
    fresh["python"] = {"mode": py_mode, "command": command, "count": total,
                       "packages": {"jope": {"count": total, "files": {f: {"count": len(v), "tests": v} for f, v in files.items()}}}}
else:
    fresh["python"] = committed.get("python", {"mode": "", "command": "", "count": 0, "packages": {}})

fresh["totals"] = {"rust": fresh["rust"]["count"], "python": fresh["python"]["count"]}

# ---- compare ------------------------------------------------------------------------------------
def rust_ids(doc):
    return {f"{bid} {t}" for pkg in doc.get("rust", {}).get("packages", {}).values()
            for bid, b in pkg.get("binaries", {}).items() for t in b.get("tests", [])}

def py_ids(doc, strip):
    ids = {t for pkg in doc.get("python", {}).get("packages", {}).values()
           for f in pkg.get("files", {}).values() for t in (f.get("tests", []) if isinstance(f, dict) else f)}
    return {re.sub(r"\[.*\]$", "", t) for t in ids} if strip else ids

problems = []
for side, name in (("rust", "Rust"), ("python", "Python")):
    if (side == "rust" and not do_rust) or (side == "python" and not do_python):
        print(f"{name}: skipped (committed half kept: {committed.get(side, {}).get('count', 0)} tests)")
        continue
    if side == "rust":
        old, new = rust_ids(committed), rust_ids(fresh)
    else:
        strip = "static-scan" in (committed.get("python", {}).get("mode", ""), py_mode)
        old, new = py_ids(committed, strip), py_ids(fresh, strip)
    removed, added = sorted(old - new), sorted(new - old)
    print(f"{name}: {fresh[side]['count']} tests listed" + (f" ({fresh[side]['ignored']} ignored)" if side == "rust" else f" ({py_mode})")
          + f"; committed {committed.get(side, {}).get('count', 0)}; removed {len(removed)}, added {len(added)}")
    for t in removed[:50]:
        print(f"  - removed: {t}")
    for t in added[:50]:
        print(f"  + added: {t}")
    if removed and not update:
        problems.append(f"{name}: {len(removed)} test(s) in {inventory} no longer exist; update the inventory in the same change (deploy/ci/test-inventory.sh --update)")
    if added and not update:
        print(f"  note: {len(added)} new {name} test(s) not yet recorded; run deploy/ci/test-inventory.sh --update")
    floor = fresh["floors"].get(side, 0)
    if fresh[side]["count"] < floor:
        problems.append(f"{name}: {fresh[side]['count']} tests is below the committed floor {floor}")

if update:
    inventory.parent.mkdir(parents=True, exist_ok=True)
    inventory.write_text(json.dumps(fresh, indent=1, sort_keys=False) + "\n")
    print(f"wrote {inventory}: rust {fresh['totals']['rust']}, python {fresh['totals']['python']}")

if problems:
    print("\n" + "\n".join("error: " + p for p in problems))
    sys.exit(1)
print("ok: test inventory consistent")
PY
