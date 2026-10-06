"""L7a: re-fetch the original T1 sources (owner-typed 2026-10-05T18:34Z, receipt owner-decisions-20261005T183423Z.json).

Reads the fetch plan (l7a-fetch-plan-*.jsonl, rows with action "fetch"), and for each dataset downloads its source_url:
read-only GET, public URLs only (no credentials, cookies or auth headers), at most 2 concurrent requests and at least
1 s between requests to one host, per-file cap 400 MB, total cap 6 GB. The body streams to <dest>.part and is renamed
when complete; its size is compared with the declared source_size_bytes and recorded (the declared size is often the
staged copy). A source is accepted for use only by the content-match step; otherwise the dataset stays on L7b. One attempt plus one retry on a transport error. Resumable: a dataset with .done is skipped.
Every packet step pauses on the lane CPU guard. Writes only under RUN/inputs/t1-raw/ and one receipt.

Usage (inside the lane slice): l7a_fetch.py --plan PLAN.jsonl --out RUN/inputs/t1-raw [--limit N]
"""
import argparse
import hashlib
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

from dltrain import cpuguard

PER_FILE_CAP = 400_000_000
TOTAL_CAP = 6_000_000_000
HOST_GAP_S = 1.0
UA = "jopedime-dl-research/1.0 (read-only dataset re-fetch; owner-admitted L7a)"

_host_lock = threading.Lock()
_host_next = {}
_total_lock = threading.Lock()
_total = {"bytes": 0}


def wait_host(host):
    while True:
        with _host_lock:
            now = time.monotonic()
            t = _host_next.get(host, 0.0)
            if now >= t:
                _host_next[host] = now + HOST_GAP_S
                return
        time.sleep(min(1.0, t - now))


def guard():
    ok, reasons, _ = cpuguard.check()
    while not ok:
        time.sleep(15)
        ok, reasons, _ = cpuguard.check()


def fetch_one(row, out: Path):
    h = row["dataset_hash"]
    d = out / h
    if (d / ".done").exists():
        return {"dataset_hash": h, "status": "skipped_done"}
    url, want = row.get("url"), row.get("declared_bytes")
    if not url or urlparse(url).scheme not in ("http", "https"):
        return {"dataset_hash": h, "status": "no_public_url"}
    if isinstance(want, int) and want > PER_FILE_CAP:
        return {"dataset_hash": h, "status": "over_per_file_cap", "declared_bytes": want}
    with _total_lock:
        if _total["bytes"] + (want or 0) > TOTAL_CAP:
            return {"dataset_hash": h, "status": "total_cap_reached"}
    d.mkdir(parents=True, exist_ok=True)
    name = os.path.basename(urlparse(url).path) or "source"
    part, final = d / (name + ".part"), d / name
    last_err = None
    for attempt in (1, 2):
        guard()
        wait_host(urlparse(url).netloc)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            h256, n = hashlib.sha256(), 0
            with urllib.request.urlopen(req, timeout=120) as resp, open(part, "wb") as f:
                final_url = resp.geturl()
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    n += len(chunk)
                    if n > PER_FILE_CAP:
                        raise ValueError("per-file cap exceeded while streaming")
                    h256.update(chunk); f.write(chunk)
                f.flush(); os.fsync(f.fileno())
            # The declared size is often the staged, cleaned copy, and live APIs change: size is recorded, not trusted.
            # A source is accepted only by the later content match (T1 rows rebuilt from the raw rows via meta.json).
            status = "fetched_size_matches" if isinstance(want, int) and n == want else (
                "fetched_size_differs" if isinstance(want, int) else "fetched_no_declared_size")
            os.replace(part, final)
            os.chmod(final, 0o444)
            with _total_lock:
                _total["bytes"] += n
            rec = {"dataset_hash": h, "status": status, "declared_bytes": want, "bytes": n, "sha256": h256.hexdigest(), "file": str(final), "url": url,
                   "final_url": final_url, "attempts": attempt, "utc_us": int(time.time() * 1e6)}
            (d / "fetch.json").write_text(json.dumps(rec, sort_keys=True) + "\n")
            (d / ".done").write_text("ok\n")
            return rec
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError, ValueError) as e:
            last_err = f"{type(e).__name__}: {e}"[:300]
            part.unlink(missing_ok=True)
            if isinstance(e, (urllib.error.HTTPError, ValueError)):
                break                                  # an HTTP status or cap breach is not retried
    return {"dataset_hash": h, "status": "error", "error": last_err, "url": url}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    plan_b = Path(a.plan).read_bytes()
    rows = [json.loads(l) for l in plan_b.decode().splitlines() if l.strip()]
    todo = [r for r in rows if r.get("action") == "fetch"]
    if a.limit:
        todo = todo[: a.limit]
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=2) as ex:
        results = list(ex.map(lambda r: fetch_one(r, out), todo))
    counts = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    rec = {"schema": "jope.dl-l7a-fetch-run/v1", "utc_us": int(time.time() * 1e6), "elapsed_s": round(time.time() - t0, 1),
           "plan": {"path": a.plan, "sha256": hashlib.sha256(plan_b).hexdigest()}, "attempted": len(todo), "counts": counts,
           "bytes_fetched": sum(r.get("bytes", 0) for r in results if r["status"].startswith("fetched")), "results": results,
           "owner_decision": "owner-decisions-20261005T183423Z.json (L7a)",
           "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    p = out / f"fetch-run-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json"
    p.write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: rec[k] for k in ("attempted", "counts", "bytes_fetched", "elapsed_s")}), p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
