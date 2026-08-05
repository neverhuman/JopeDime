#!/usr/bin/env python3
"""Audit and repair symlinks under the foundation-expert tree.

Why this exists: the tree was moved from /home/ubuntu/fable_god* to
/home/ubuntu/remote_super/fable_god and every symlink was left pointing at the
vanished root. Eleven links across five chimera trees dangled, including model
weights. A dangling link is worse than a missing file because it reads as
present to anything that only checks the name.

Symlinks are banned here by owner decision. `repair` therefore does not repoint
them -- it REPLACES each with real content: a hard link for large files (same
filesystem, so no duplication, and a hard link is a real directory entry) and a
copy for directories, which cannot be hard-linked.

Safety: a link whose real target cannot be located is REPORTED, never removed.
A dangling link is at least evidence of intent; deleting it destroys that.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

TREE = Path("/home/ubuntu/remote_super/fable_god")

# Where the vanished roots actually live now. Ordered: first match wins.
RELOCATIONS: tuple[tuple[str, str], ...] = (
    # regressor_source / classifier_deps moved under provenance/
    ("/home/ubuntu/fable_god_run/code/regressor_source",
     "/home/ubuntu/remote_super/fable_god/provenance/regressor_source"),
    ("/home/ubuntu/fable_god/provenance/", "/home/ubuntu/remote_super/fable_god/provenance/"),
    ("/home/ubuntu/fable_god/weights/", "/home/ubuntu/remote_super/fable_god/weights/"),
    ("/home/ubuntu/fable_god/python/", "/home/ubuntu/remote_super/fable_god/python/"),
    ("/home/ubuntu/fable_god/", "/home/ubuntu/remote_super/fable_god/"),
)


def relocate(target: str) -> Path | None:
    """Map a stale target onto its real location, if one exists."""

    for stale, current in RELOCATIONS:
        if target.startswith(stale):
            candidate = Path(target.replace(stale, current, 1))
            if candidate.exists():
                return candidate
    # The name may simply have moved elsewhere in the tree; look for a basename
    # match rather than guessing, and only accept an unambiguous single hit.
    matches = [p for p in TREE.rglob(Path(target).name) if not p.is_symlink()]
    return matches[0] if len(matches) == 1 else None


def survey(tree: Path) -> list[tuple[Path, str, bool, Path | None]]:
    rows = []
    for path in sorted(tree.rglob("*")):
        if not path.is_symlink():
            continue
        target = os.readlink(path)
        intact = path.exists()
        rows.append((path, target, intact, None if intact else relocate(target)))
    return rows


def repair(rows, apply_changes: bool) -> int:
    failures = 0
    for path, target, intact, replacement in rows:
        label = str(path).replace(str(TREE) + "/", "")
        if intact:
            # Resolve through the link so we copy CONTENT, not another link.
            replacement = path.resolve()
        if replacement is None:
            print(f"  UNRESOLVED {label} -> {target}")
            print("     left in place: no real content found; this needs a human")
            failures += 1
            continue
        kind = "dir-copy" if replacement.is_dir() else "hard-link"
        print(f"  {kind:9s} {label} <- {replacement}")
        if not apply_changes:
            continue
        path.unlink()
        try:
            if replacement.is_dir():
                shutil.copytree(replacement, path, symlinks=False)
            else:
                os.link(replacement, path)
        except OSError as error:
            # Never leave a hole where a link used to be.
            print(f"     FAILED ({error}); restoring the original link")
            os.symlink(target, path)
            failures += 1
    return failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tree", type=Path, default=TREE)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="replace links with real content (default: report only)",
    )
    args = parser.parse_args()

    rows = survey(args.tree)
    broken = [row for row in rows if not row[2]]
    print(f"symlinks: {len(rows)}   dangling: {len(broken)}")
    if not rows:
        print("clean: no symlinks remain")
        return
    failures = repair(rows, args.apply)
    if args.apply:
        remaining = [p for p in args.tree.rglob("*") if p.is_symlink()]
        print(f"after: {len(remaining)} symlinks remain, {failures} unresolved")
        sys.exit(1 if remaining and failures else 0)
    print(f"(dry run; {failures} unresolved) re-run with --apply to replace")


if __name__ == "__main__":
    main()
