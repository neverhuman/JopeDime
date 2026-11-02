"""ADDITIVE helper shards for a running raw-route build (tools/v2_rowsel_raw.py build). PREPARED; launch only when told.

A helper builds datasets of an owner shard's queue from the TAIL, with exactly the owner's code and plan, and publishes
them into the owner's shard directory so the owner skips them when it gets there (tools/v2_rowsel_raw.build checks
`dataset_done(out_shard, h, codes)` immediately before starting each dataset). It never signals, stops, restarts or edits
a running shard, never deletes anything, and never touches a dataset that is done or that the owner is working on.

Why it is safe without the owners knowing about it:
  code      the helper imports the same tools/v2_rowsel_raw.py and dltrain modules (this file is not one of CODE_FILES)
            and refuses to start unless its code_sha256 equals the code_sha256 of the owners' existing done records; a
            helper-built record therefore satisfies the owner's dataset_done() check.
  output    run_dataset is deterministic: encoder/teacher JTT1, A0 and the identity/view_id are the owner's bytes;
            context.json differs only in utc_us, the dataset record only in timing fields (tests/test_v2_rowsel_raw_helper.py).
  claims    every dataset is claimed with O_CREAT|O_EXCL (OUT/_claims/<h>.claim) before a helper starts it: two helpers
            never build the same dataset. A claim is never removed (a crashed helper's dataset is left to the owner).
  distance  the owner works from the head of its queue and does not read claims, so a helper only takes dataset j when
            the owner's frontier (its first not-done dataset, the one it is on) is at least --min-gap unclaimed datasets
            and --margin x est(j) of unclaimed estimated work away. A collision is then unlikely, and harmless anyway:
  staging   the helper runs run_dataset into its own OUT/_helpers/<id>/shard-<i>/ (with a per-process temporary file name
            for the shared OUT/_match receipts, whose content is deterministic), then publishes context by context:
              - target with .done and the same identity and files: already there, nothing written
              - target non-empty without .done (the owner is writing it), or another identity: LOST RACE, the helper
                stops publishing this dataset and writes nothing more; the owner finishes it itself
              - otherwise os.rename(staged dir without .done -> target): atomic; it fails on a non-empty target and
                only replaces an EMPTY one (an owner that has not written a file yet then writes identical bytes into it)
            then the .done markers (rename), then the dataset record create-only (O_EXCL; an owner record already there
            is kept). A helper that dies mid-publish leaves dirs without .done, which the owner (or a rerun) completes.
Usage (inside the lane slice):
  v2_rowsel_raw_helper.py --route l7a|nonnumeric --split-version rowsel-raw-split-v2 --helper-id ID [--owners 0,1,2]
                          [--margin 4] [--min-gap 2] [--max-datasets N] [--out DIR] [--of 3] [--dry-run HELPERS]
"""
import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

from dltrain import cpuguard, scale_check

_spec = importlib.util.spec_from_file_location("v2_rowsel_raw", Path(__file__).resolve().parent / "v2_rowsel_raw.py")
tool = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tool)

SCHEMA = "jope.dl-v2-rowsel-raw-helper/v1"


def unique_atomic(path: Path, data: bytes):
    """tool.atomic with a per-process temporary name: never shares a .tmp path with an owner or another helper."""
    tmp = path.with_name(f".{path.name}.{os.getpid()}.helper.tmp")
    with open(tmp, "wb") as f:
        f.write(data); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def create_only(path: Path, data: bytes) -> bool:
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        return False
    with os.fdopen(fd, "wb") as f:
        f.write(data); f.flush(); os.fsync(f.fileno())
    return True


def owner_queue(pl: dict, shard: int) -> list:
    """The owner's exact processing order (tools/v2_rowsel_raw.build)."""
    mine = [e for e in pl["datasets"] if e.get("disposition", "build") == "build" and e["shard"] == shard]
    return sorted(mine, key=lambda e: -e["est_cpu_s"])


def claimed(out: Path, h: str, extra=()) -> bool:
    return h in extra or (out / "_claims" / f"{h}.claim").exists()


def claim(out: Path, h: str, helper_id: str) -> bool:
    (out / "_claims").mkdir(parents=True, exist_ok=True)
    return create_only(out / "_claims" / f"{h}.claim", tool.jdump({"dataset_hash": h, "helper": helper_id, "pid": os.getpid(),
                                                                   "utc_us": int(time.time() * 1e6)}))


def select(out: Path, pl: dict, owners, codes, margin: float, min_gap: int, extra_claimed=()):
    """(entry, owner shard, diagnostics) of the next dataset to take, or None. Per owner: its frontier is the first
    not-done dataset of its queue (the one it is on); a candidate j (walked from the tail) must be not done, not claimed,
    and have >= min_gap unclaimed not-done datasets with >= margin * est(j) estimated work between the frontier and j.
    Across owners, the one with the most unclaimed remaining work is served first."""
    best = None
    for i in owners:
        q = owner_queue(pl, i)
        sd = out / f"shard-{i}"
        done = [tool.dataset_done(sd, e["dataset_hash"], codes) for e in q]
        free = [not d and not claimed(out, e["dataset_hash"], extra_claimed) for e, d in zip(q, done)]
        if all(done):
            continue
        frontier = done.index(False)
        remaining = sum(e["est_cpu_s"] for k, e in enumerate(q) if k > frontier and free[k])
        for j in range(len(q) - 1, frontier, -1):
            if not free[j]:
                continue
            between = [k for k in range(frontier + 1, j) if free[k]]
            work = sum(q[k]["est_cpu_s"] for k in between)
            if len(between) >= min_gap and work >= margin * q[j]["est_cpu_s"]:
                cand = (remaining, q[j], i, {"owner": i, "frontier": frontier, "frontier_dataset": q[frontier]["dataset_hash"],
                                             "position": j, "queue": len(q), "gap_datasets": len(between),
                                             "gap_est_cpu_s": round(work, 1), "est_cpu_s": q[j]["est_cpu_s"]})
                if best is None or cand[0] > best[0]:
                    best = cand
            break                       # only the tail-most free candidate of each owner is considered
    return None if best is None else best[1:]


def _same_context(target: Path, staged: Path) -> bool:
    try:
        a = json.loads((target / "context.json").read_bytes())
        b = json.loads((staged / "context.json").read_bytes())
    except (OSError, ValueError):
        return False
    if a.get("rowsel", {}).get("identity") != b.get("rowsel", {}).get("identity") or a.get("files") != b.get("files"):
        return False
    return all(tool.sha_file(target / n) == m["sha256"] for n, m in a["files"].items() if (target / n).exists())


def publish(rec: dict, staging_shard: Path, owner_shard: Path) -> dict:
    """Move the staged contexts of one done dataset into the owner's shard dir (see the module docstring)."""
    h = rec["dataset_hash"]
    out = {"dataset_hash": h, "already_done": [], "published": [], "lost_race": None}
    markers = staging_shard / "_markers"
    markers.mkdir(parents=True, exist_ok=True)
    todo = []
    for c in rec["contexts"]:
        t, s = owner_shard / c["view_id"], staging_shard / c["view_id"]
        if (t / ".done").exists():
            if not _same_context(t, s):
                out["lost_race"] = f"{c['view_id']}: done with another identity or files"
                return out
            out["already_done"].append(c["view_id"])
        elif t.exists() and any(t.iterdir()):
            out["lost_race"] = f"{c['view_id']}: the owner is writing it"
            return out
        else:
            todo.append((c["view_id"], s, t))
    for v, s, t in todo:
        if (s / ".done").exists():
            os.rename(s / ".done", markers / f"{v}.done")
        try:
            os.rename(s, t)                                   # atomic; refuses a non-empty target
        except OSError as err:
            out["lost_race"] = f"{v}: rename refused ({type(err).__name__})"
            return out
        out["published"].append(v)
    for v, _, t in todo:                                      # .done last, after every context of the dataset is in place
        os.rename(markers / f"{v}.done", t / ".done")
    (owner_shard / "_datasets").mkdir(parents=True, exist_ok=True)
    out["record_written"] = create_only(owner_shard / "_datasets" / f"{h}.json",
                                        (staging_shard / "_datasets" / f"{h}.json").read_bytes())
    return out


def owner_codes_match(out: Path, codes) -> dict:
    """The owners' existing done records must carry exactly this process's code_sha256."""
    seen = {}
    for p in sorted(out.glob("shard-*/_datasets/*.json")):
        r = json.loads(p.read_bytes())
        if r.get("status") == "done":
            seen[json.dumps(r.get("code_sha256"), sort_keys=True)] = str(p)
    return {"records_with_code": len(seen), "equal": list(seen) == [json.dumps(codes, sort_keys=True)]}


def _print(line: str):
    print(line, flush=True)


def dry_run(a, helpers: int) -> list:
    """Which datasets `helpers` helpers would take now, claims simulated in memory; writes nothing."""
    out = Path(a.out)
    pl, _ = tool.load_plan(out)
    codes = tool.code_shas()
    owners = [int(x) for x in a.owners.split(",")]
    taken, picks = set(), []
    for k in range(helpers):
        sel = select(out, pl, owners, codes, a.margin, a.min_gap, taken)
        if sel is None:
            break
        e, i, diag = sel
        taken.add(e["dataset_hash"])
        picks.append({"helper": k, "dataset_hash": e["dataset_hash"], **diag})
    return picks


def run(a, src=None, cache=None, inv=None, held=None, log=_print):
    out = Path(a.out)
    pl, plan_sha = tool.load_plan(out)
    tool.plan_split_version(pl, a.split_version)
    codes = tool.code_shas()
    chk = owner_codes_match(out, codes)
    if not chk["equal"]:
        raise SystemExit(f"helper code differs from the owners' done records (or none yet): {chk}")
    tool.atomic = unique_atomic                                # this process only; same bytes, private tmp names
    owners = [int(x) for x in a.owners.split(",")]
    staging = out / "_helpers" / a.helper_id
    staging.mkdir(parents=True, exist_ok=True)
    if src is None:
        cpuguard.set_threads(2)
        src = scale_check.Sources(load_index=False)
        cache = scale_check.DataCache(max_items=2)
        inv = json.loads(tool.DATASETS_INV.read_text())
        held = tool.held_set()[0]
    guard = cpuguard.Guard(journal_path=staging / "guard.jsonl") if not getattr(a, "no_guard", False) else _NoGuard()
    hooks = tool.route_hooks(a, out, codes)
    t0, c0 = time.time(), tool.cpu_s()
    events = []
    while a.max_datasets is None or sum(1 for x in events if x.get("status") == "done") < a.max_datasets:
        guard.wait()
        sel = select(out, pl, owners, codes, a.margin, a.min_gap)
        if sel is None:
            break
        e, i, diag = sel
        h = e["dataset_hash"]
        if not claim(out, h, a.helper_id):
            continue
        owner_shard = out / f"shard-{i}"
        if tool.dataset_done(owner_shard, h, codes):
            events.append({"dataset_hash": h, "status": "done_before_start", **diag})
            continue
        sd = staging / f"shard-{i}"
        sd.mkdir(parents=True, exist_ok=True)
        rec = tool.run_dataset(dict(e), sd, codes, src, cache, inv, held, guard, log=lambda s: None, split_version=a.split_version, **hooks)
        ev = {"dataset_hash": h, "status": rec["status"], "reason": rec.get("reason"), "cpu_s": rec["cpu_s"], **diag}
        if rec["status"] == "done":
            ev["publish"] = publish(rec, sd, owner_shard)
        events.append(ev)
        log(json.dumps({"helper": a.helper_id, **{k: ev.get(k) for k in ("dataset_hash", "status", "reason", "cpu_s", "owner", "position")},
                        "lost_race": (ev.get("publish") or {}).get("lost_race")}))
    doc = {"schema": SCHEMA, "helper": a.helper_id, "out": str(out), "plan_sha256": plan_sha, "split_version": a.split_version,
           "route": a.route, "owners": owners, "margin": a.margin, "min_gap": a.min_gap, "code_sha256": codes,
           "helper_sha256": tool.sha_file(Path(__file__)), "owner_code_check": chk, "events": events,
           "built": sum(1 for x in events if x.get("status") == "done" and not (x.get("publish") or {}).get("lost_race")),
           "lost_races": sum(1 for x in events if (x.get("publish") or {}).get("lost_race")),
           "cpu_s": round(tool.cpu_s() - c0, 1), "wall_s": round(time.time() - t0, 1), "maxrss_mb": tool.maxrss_mb(),
           "published": False, "utc_us": int(time.time() * 1e6)}
    rp = staging / f"helper-{int(time.time())}.receipt.json"
    create_only(rp, (json.dumps(doc, indent=1, sort_keys=True) + "\n").encode())
    log(json.dumps({k: doc[k] for k in ("helper", "built", "lost_races", "cpu_s", "wall_s")}))
    return doc


class _NoGuard:
    def wait(self):
        return 0.0

    def summary(self):
        return {}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--route", choices=("l7a", "nonnumeric"), required=True)
    ap.add_argument("--split-version", choices=tuple(tool.SPLIT_VERSIONS), required=True)
    ap.add_argument("--helper-id", required=True)
    ap.add_argument("--owners", default="0,1,2")
    ap.add_argument("--of", type=int, default=3)
    ap.add_argument("--out")
    ap.add_argument("--margin", type=float, default=4.0)
    ap.add_argument("--min-gap", type=int, default=2)
    ap.add_argument("--max-datasets", type=int)
    ap.add_argument("--dry-run", type=int, metavar="HELPERS", help="print the datasets HELPERS helpers would take now; write nothing")
    a = ap.parse_args(argv)
    if a.out is None:
        a.out = str(tool.default_out(a.route, a.split_version))
    if a.dry_run:
        print(json.dumps(dry_run(a, a.dry_run), indent=1))
        return 0
    run(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
