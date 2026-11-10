#!/usr/bin/env python3
"""Stage TRAIN-only H317 neural inputs from xbabe2 to the GPU hosts, with sha256 verification of every copied file.

extract   (xbabe2) H58 TRAIN-prefix extracts via the H308 pread loader inside `mixed_prefix_read()`:
          $H317_DATA/neural/h58train/{context_features_train1219.npz, pool_priors_train1219.npz} (no DL-Val byte is read
          or written; the extracts hold only the 1,219 TRAIN cids, in the H58 order).
manifest  (xbabe2) sha256 list of every file to stage: the 1,219 TRAIN tables + pools (sha checked against
          selected.json), selected.json, champions.json (TRAIN-only, checked), cpu_exact_fused.py, the two extracts,
          and the H317 code directory. Refuses if any DL-Val cid appears in a staged path.
push HOST (xbabe2) rsync --files-from the manifest to HOST:/home/ubuntu/jope-data/h317-cl/{res01-train,neural/h58train,code}
          then `sha256sum -c` on HOST; writes a staging receipt.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common_h317 as C  # noqa: E402

REMOTE_ROOT = os.environ.get("H317_REMOTE_DATA", "/home/ubuntu/jope-data/h317-cl")
NEURAL_ROOT = C.DATA_ROOT / "neural"
STAGE_DIR = NEURAL_ROOT / "staging"
H58_TRAIN_DIR = NEURAL_ROOT / "h58train"
CTX_OUT = "context_features_train1219.npz"
POOL_OUT = "pool_priors_train1219.npz"


class PreadAudit:
    """Records every os.pread(fd, n, offset) made by the H308 loader, keyed by the opened path, so each feature member's
    byte range can be asserted to be exactly its TRAIN prefix [data_off, data_off + n_train_rows * row_bytes)."""

    def __init__(self):
        self.calls = []
        self._fd_path = {}

    def __enter__(self):
        self._open, self._pread = os.open, os.pread

        def _open(path, flags, *a, **k):
            fd = self._open(path, flags, *a, **k)
            self._fd_path[fd] = os.fspath(path)
            return fd

        def _pread(fd, n, off):
            buf = self._pread(fd, n, off)
            self.calls.append((self._fd_path.get(fd, "?"), int(off), len(buf)))
            return buf

        os.open, os.pread = _open, _pread
        return self

    def __exit__(self, *exc):
        os.open, os.pread = self._open, self._pread
        return False


def _member_ranges(TL, path: Path, n_rows_by_member: dict) -> dict:
    out = {}
    for name, n_rows in n_rows_by_member.items():
        data_off, shape, dtype = TL._member_layout(path, name + ".npy")
        row_bytes = (
            int(np.prod(shape[1:], dtype=np.int64)) * dtype.itemsize
            if len(shape) > 1
            else dtype.itemsize
        )
        out[name] = {
            "data_offset": data_off,
            "rows_total": int(shape[0]),
            "rows_prefix": int(n_rows),
            "row_bytes": row_bytes,
            "prefix_bytes": int(n_rows) * row_bytes,
            "member_bytes": int(shape[0]) * row_bytes,
        }
    return out


def _check_reads(calls, path: Path, ranges: dict, feature_members) -> dict:
    """Every pread on `path` must fall inside exactly one member; feature members must be read as exactly their prefix."""
    res = {}
    mine = [
        (off, n) for p, off, n in calls if os.path.abspath(p) == os.path.abspath(path)
    ]
    for name, r in ranges.items():
        lo, hi_prefix, hi_member = (
            r["data_offset"],
            r["data_offset"] + r["prefix_bytes"],
            r["data_offset"] + r["member_bytes"],
        )
        reads = [(off, n) for off, n in mine if lo <= off < hi_member]
        nbytes = sum(n for _, n in reads)
        beyond = sum(
            max(0, min(off + n, hi_member) - max(off, hi_prefix)) for off, n in reads
        )
        res[name] = {
            **r,
            "bytes_read": nbytes,
            "bytes_read_beyond_train_prefix": beyond,
            "n_preads": len(reads),
        }
        if name in feature_members and (
            beyond != 0 or nbytes % r["prefix_bytes"] != 0 or nbytes == 0
        ):
            raise RuntimeError(
                f"{path.name}:{name} read outside / not exactly its TRAIN prefix: {res[name]}"
            )
    stray = [
        (off, n)
        for off, n in mine
        if not any(
            r["data_offset"] <= off < r["data_offset"] + r["member_bytes"]
            for r in ranges.values()
        )
    ]
    if stray:
        raise RuntimeError(
            f"{path.name}: preads outside every known member: {stray[:3]}"
        )
    return res


def cmd_extract(args) -> None:
    """Extract (or, with --verify-only, re-read and compare against the existing extract files) the H58 TRAIN prefix,
    recording a per-member byte-range audit: every feature member read == exactly its TRAIN-prefix bytes, 0 bytes
    beyond it. Id members (cids, offsets) are read whole by design (identifiers, H308)."""
    C.install_guard()
    import trainonly_loader as TL

    rows = C.train_rows()
    train = {r["context_sha256"] for r in rows}
    out = {}
    ctx_members = ("tokens_10x32", "mask_10", "dope629", "lm48_z", "ctx_feat_584")
    pool_members = ("base_prior_cost", "sos_prior", "proj16")
    ctx_path, pool_path = (
        C.H58_CACHE / "context_features.npz",
        C.H58_CACHE / "pool_priors.npz",
    )
    with C.mixed_prefix_read(), PreadAudit() as audit:
        with TL.TrainOnlyNpz(ctx_path, train) as z:
            ctx = {k: z[k] for k in ctx_members}
            ctx["cids"] = z["cids"][: z.n_train]
            n_ctx = z.n_train
        with TL.TrainOnlyNpz(pool_path, train) as z:
            pool = {k: z[k] for k in pool_members}
            pool["cids"] = z["cids"][: z.n_train]
            pool["offsets"] = z["offsets"][: z.n_train + 1]
            pool["scale"] = TL.read_rows(z.path, "scale.npy", z.n_train)
            n_pool_rows, n_pool_ctx = z.row_prefix, z.n_train
        ctx_ranges = _member_ranges(
            TL, ctx_path, {**{k: n_ctx for k in ctx_members}, "cids": 1310}
        )
        pool_ranges = _member_ranges(
            TL,
            pool_path,
            {
                **{k: n_pool_rows for k in pool_members},
                "scale": n_pool_ctx,
                "cids": 1310,
                "offsets": 1311,
            },
        )
    byte_audit = {
        "context_features.npz": _check_reads(
            audit.calls, ctx_path, ctx_ranges, ctx_members
        ),
        "pool_priors.npz": _check_reads(
            audit.calls, pool_path, pool_ranges, pool_members + ("scale",)
        ),
    }
    for d in (ctx, pool):
        if set(map(str, d["cids"])) != train or len(d["cids"]) != len(train):
            raise RuntimeError("extract cids are not exactly the TRAIN set")
    H58_TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    compare = {}
    for name, arrays in ((CTX_OUT, ctx), (POOL_OUT, pool)):
        p = H58_TRAIN_DIR / name
        if args.verify_only:
            with np.load(p, allow_pickle=False) as z:
                compare[name] = {
                    k: bool(np.array_equal(z[k], v)) for k, v in arrays.items()
                }
            if not all(compare[name].values()):
                raise RuntimeError(
                    f"existing extract {name} differs from a fresh prefix read: {compare[name]}"
                )
        else:
            np.savez(p, **arrays)
        out[name] = C.sha256_file(p)
    rec = {
        "schema": "jope.h317.h58-train-prefix-extract/v2",
        "mode": "verify-only (existing extract files compared array-by-array)"
        if args.verify_only
        else "extract",
        "source_sha256": {
            "context_features.npz": "75704e8279c19fee8a7843ccbea801e3cd4b67e0ee08d282b070ddae7ab82faf",
            "pool_priors.npz": C.H58_POOL_PRIORS_SHA256,
        },
        "loader": "trainonly_loader.py (H308, vendored)",
        "loader_counters_note": "trainonly_loader.COUNTERS['dlval_feature_rows_read'] is a constant 0; the byte_audit below is the real check",
        "loader_counters": dict(TL.COUNTERS),
        "byte_audit": byte_audit,
        "dlval_feature_bytes_read_total": int(
            sum(
                m["bytes_read_beyond_train_prefix"]
                for f in byte_audit.values()
                for k, m in f.items()
                if k not in ("cids", "offsets")
            )
        ),
        "compare_with_existing": compare,
        "pool_rows": int(pool["offsets"][-1]),
        "outputs_sha256": out,
        "guard": C.guard_report(),
    }
    C.write_json(
        H58_TRAIN_DIR
        / (
            "extract_verify_receipt.json"
            if args.verify_only
            else "extract_receipt.json"
        ),
        rec,
    )
    print(json.dumps({k: v for k, v in rec.items() if k != "byte_audit"}, indent=1))


def _files(rows):
    """(source path, remote relative path) for every staged file."""
    items = [
        (C.RES01 / "selected.json", "res01-train/selected.json"),
        (C.RES01 / "champions.json", "res01-train/champions.json"),
        (C.RES01 / "cpu_exact_fused.py", "res01-train/cpu_exact_fused.py"),
    ]
    for r in rows:
        cid = r["context_sha256"]
        items.append((C.table_path(cid), f"res01-train/tables/{cid}.jtt1"))
        items.append((C.pool_path(cid), f"res01-train/pools/{cid}.npz"))
    for name in (CTX_OUT, POOL_OUT, "extract_receipt.json"):
        items.append((H58_TRAIN_DIR / name, f"neural/h58train/{name}"))
    r882 = NEURAL_ROOT / "r882"
    for p in sorted(r882.glob("B-costs-fold*-seed189101-R882.json")) + [
        r882 / "SHA256SUMS"
    ]:
        items.append((p, f"neural/r882/{p.name}"))
    for p in sorted(HERE.iterdir()):
        if p.is_file() and p.suffix in (".py", ".json", ".md"):
            items.append((p, f"code/{C.SLUG}/{p.name}"))
    return items


def cmd_manifest(args) -> None:
    C.install_guard()
    rows = C.train_rows()
    by_cid = {r["context_sha256"]: r for r in rows}
    champs = json.loads((C.RES01 / "champions.json").read_text())
    if len(champs) != 1219 or any(
        c["context_sha256"] not in by_cid or c["dl_split"] != "DL-Train" for c in champs
    ):
        raise RuntimeError("champions.json is not exactly the 1,219 TRAIN champions")
    dev = C._STATE["dev_cids"]
    lines, n = [], 0
    for src, rel in _files(rows):
        if any(d in rel for d in dev):
            raise RuntimeError(f"DL-Val cid in staged path {rel}")
        sha = C.sha256_file(src)
        stem = Path(rel).stem.split(".")[0]
        if (
            rel.startswith("res01-train/tables/")
            and sha != by_cid[stem]["train_only_jtt1_sha256"]
        ):
            raise RuntimeError(f"table sha mismatch {stem}")
        if rel.startswith("res01-train/pools/") and sha != by_cid[stem]["pool_sha256"]:
            raise RuntimeError(f"pool sha mismatch {stem}")
        lines.append(f"{sha}  {rel}")
        n += 1
    STAGE_DIR.mkdir(parents=True, exist_ok=True)
    (STAGE_DIR / "SHA256SUMS").write_text("\n".join(lines) + "\n")
    with open(STAGE_DIR / "sources.tsv", "w") as f:
        for src, rel in _files(rows):
            f.write(f"{src}\t{rel}\n")
    print(
        json.dumps(
            {
                "files": n,
                "sha256sums_sha256": C.sha256_file(STAGE_DIR / "SHA256SUMS"),
                "guard": C.guard_report(),
            }
        )
    )


def cmd_push(args) -> None:
    C.install_guard()
    host = args.host
    pairs = [
        ln.split("\t") for ln in (STAGE_DIR / "sources.tsv").read_text().splitlines()
    ]
    t0 = time.time()
    # Group by source directory so rsync keeps one --files-from per root.
    groups = {}
    for src, rel in pairs:
        srcp = Path(src)
        dst_dir = str(Path(rel).parent)
        groups.setdefault((str(srcp.parent), dst_dir), []).append(srcp.name)
    for (src_dir, dst_dir), names in sorted(groups.items()):
        lst = STAGE_DIR / "filelist.tmp"
        lst.write_text("\n".join(names) + "\n")
        subprocess.run(
            ["ssh", "-o", "BatchMode=yes", host, f"mkdir -p {REMOTE_ROOT}/{dst_dir}"],
            check=True,
        )
        subprocess.run(
            [
                "rsync",
                "-a",
                f"--files-from={lst}",
                src_dir + "/",
                f"{host}:{REMOTE_ROOT}/{dst_dir}/",
            ],
            check=True,
        )
    subprocess.run(
        [
            "rsync",
            "-a",
            str(STAGE_DIR / "SHA256SUMS"),
            f"{host}:{REMOTE_ROOT}/SHA256SUMS",
        ],
        check=True,
    )
    res = subprocess.run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            host,
            f"cd {REMOTE_ROOT} && sha256sum -c --quiet SHA256SUMS && echo ALL_OK; "
            f"wc -l < SHA256SUMS; du -sh {REMOTE_ROOT} | cut -f1",
        ],
        capture_output=True,
        text=True,
    )
    out = res.stdout.strip().splitlines()
    rec = {
        "schema": "jope.h317.staging/v1",
        "host": host,
        "files": len(pairs),
        "verified_all": res.returncode == 0 and "ALL_OK" in out,
        "remote_stdout": out,
        "remote_stderr": res.stderr.strip()[-2000:],
        "sha256sums_sha256": C.sha256_file(STAGE_DIR / "SHA256SUMS"),
        "elapsed_s": round(time.time() - t0, 1),
        "guard": C.guard_report(),
    }
    C.write_json(NEURAL_ROOT / "receipts" / f"staging_{host}.json", rec)
    print(json.dumps(rec, indent=1))
    if not rec["verified_all"]:
        raise SystemExit("remote sha256 verification failed")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    x = sub.add_parser("extract")
    x.add_argument("--verify-only", action="store_true")
    sub.add_parser("manifest")
    p = sub.add_parser("push")
    p.add_argument("host")
    args = ap.parse_args()
    {"extract": cmd_extract, "manifest": cmd_manifest, "push": cmd_push}[args.cmd](args)


if __name__ == "__main__":
    main()
