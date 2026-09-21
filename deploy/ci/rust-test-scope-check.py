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
script = here / "rust-test-scope.sh"


def emitted(*args: str) -> list[str]:
    """The package names the script ACTUALLY EMITS for this invocation.

    Reading the `product=( ... )` / `deferred=( ... )` arrays out of the source instead would check
    something the lane never runs: a second assignment later in the file wins in bash but loses to
    a first-match regex, and the `case` block that chooses between the two arrays would not be
    checked at all. Both divergences were demonstrated against the earlier regex version of this
    file. Running the script is the only reading of it that cannot drift from the lane.

    Repeats are kept: the script emits `-p X -p X` for a name written twice, `cargo` accepts it,
    and turning that into a set here would make the counts printed below disagree with the file.
    """
    out = subprocess.run(
        ["bash", str(script), *args], capture_output=True, text=True, cwd=here.parent.parent
    )
    if out.returncode != 0:
        sys.exit(f"rust-test-scope.sh {' '.join(args)}: exit {out.returncode}\n{out.stderr.strip()}")
    tokens = out.stdout.split()
    if not tokens or len(tokens) % 2 or tokens[0::2] != ["-p"] * (len(tokens) // 2):
        sys.exit(
            f"rust-test-scope.sh {' '.join(args)}: expected `-p <name>` pairs, got: {out.stdout!r}"
        )
    return tokens[1::2]


def declared(var: str) -> list[str]:
    """The `<var>=( ... )` array as a reader of the file sees it: first match, repeats kept.

    This is deliberately a SECOND, independent reading. Where it disagrees with what the script
    emits, the file and the lane disagree, and one of the two is lying to whoever reads it — a
    later duplicate assignment (bash takes the last, a reader takes the first) and a swapped
    `case` arm (the lane runs the other scope entirely) are both invisible to either reading alone.
    """
    m = re.search(rf"^{var}=\(([^)]*)\)", script.read_text(), re.M)
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
product_list, deferred_list = emitted(), emitted("--deferred")

problems = []
for var, listed in (("product", product_list), ("deferred", deferred_list)):
    if sorted(listed) != sorted(written := declared(var)):
        problems.append(
            f"the {var} scope emits {sorted(listed)} but the {var}=( ... ) array reads "
            f"{sorted(written)} — the file and the lane disagree"
        )
    if repeats := sorted({n for n in listed if listed.count(n) > 1}):
        problems.append(f"emitted more than once for the {var} scope: " + ", ".join(repeats))

product, deferred = set(product_list), set(deferred_list)
if unplaced := sorted(workspace - (product | deferred)):
    problems.append(
        "in the workspace but emitted by neither scope (they would be gated by nothing): "
        + ", ".join(unplaced)
    )
if ghosts := sorted((product | deferred) - workspace):
    problems.append("emitted but not in the workspace: " + ", ".join(ghosts))
if both := sorted(product & deferred):
    problems.append("emitted by both scopes: " + ", ".join(both))

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
