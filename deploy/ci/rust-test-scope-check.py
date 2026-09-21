#!/usr/bin/env python3
"""Assert deploy/ci/rust-test-scope.sh partitions the workspace exactly.

A crate added later must not land in neither list: that fails OPEN — it would be gated by no lane
and flagged by nothing. jope-arena did exactly that once, between two pushes of the lane itself,
which is why this check exists rather than a comment asking people to remember.
"""

import json
import re
import subprocess
import sys
from pathlib import Path

here = Path(__file__).resolve().parent
script = (here / "rust-test-scope.sh").read_text()


def names(var: str) -> list[str]:
    """The array's entries AS WRITTEN, in order and with repeats kept.

    Returning a set here would make a name written twice indistinguishable from a name written
    once, so the counts printed below ("20 = 11 + 9") could not be trusted and a typo'd repeat
    would pass review as a partition. Multiplicity is checked before any set arithmetic.
    """
    m = re.search(rf"^{var}=\(([^)]*)\)", script, re.M)
    if not m:
        sys.exit(f"rust-test-scope.sh: no {var}=( ... ) array found")
    return m.group(1).split()


meta = json.loads(
    subprocess.run(
        ["cargo", "metadata", "--no-deps", "--locked", "--format-version", "1"],
        capture_output=True,
        text=True,
        check=True,
        cwd=here.parent.parent,
    ).stdout
)
workspace = {p["name"] for p in meta["packages"]}
product_list, deferred_list = names("product"), names("deferred")

problems = []
for var, listed in (("product", product_list), ("deferred", deferred_list)):
    if repeats := sorted({n for n in listed if listed.count(n) > 1}):
        problems.append(f"listed more than once in {var}: " + ", ".join(repeats))

product, deferred = set(product_list), set(deferred_list)
if unplaced := sorted(workspace - (product | deferred)):
    problems.append(
        "in the workspace but in neither list (they would be gated by nothing): "
        + ", ".join(unplaced)
    )
if ghosts := sorted((product | deferred) - workspace):
    problems.append("listed but not in the workspace: " + ", ".join(ghosts))
if both := sorted(product & deferred):
    problems.append("in both lists: " + ", ".join(both))

if problems:
    print("rust-test scope does not partition the workspace:")
    for p in problems:
        print(f"  - {p}")
    print("  fix deploy/ci/rust-test-scope.sh (product = gated on pull requests, deferred = self-hosted)")
    sys.exit(1)

print(
    f"ok: {len(workspace)} workspace packages = {len(product_list)} product + "
    f"{len(deferred_list)} deferred (no repeats, no overlap, none unplaced)"
)
