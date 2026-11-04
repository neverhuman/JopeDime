"""H317 shared core: read guard, pinned TRAIN manifests, folds, pilot shard, SEARCH/AUDIT envelopes, utilities, stats.

Import this module FIRST in every H317 entry point. It installs a process-wide audit hook that refuses to open:
  * any path naming a DL-Val (dev91) context id, unless Stage V is unlocked (see `unlock_stage_v`);
  * the mixed Gemini H256/H257 caches under ~/jope-data/h256 and ~/jope-data/h257 (they hold DL-Val rows);
  * the mixed H58 npz caches, except inside `mixed_prefix_read()` (H308 prefix pread loader only);
  * sealed / holdout / meta_test / fresh-cohort / DL-Test paths.
Every refused or allowed sensitive open is counted in GUARD_COUNTERS and written to metrics.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import struct
import sys
import threading
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
WAVE = "H317"
SLUG = "h317_island10_multiscale_curriculum"

RES01 = Path(os.environ.get("H317_RES01", "/home/ubuntu/jope-data/res-01/input"))
H58_CACHE = Path(os.environ.get("H317_H58_CACHE", "/home/ubuntu/jope-data/h58/cache"))
DATA_ROOT = Path(os.environ.get("H317_DATA", "/home/ubuntu/jope-data/h317-cl"))
MICRO = Path(os.environ.get("H317_MICRO", "/home/ubuntu/JopeDimeMicro"))
WORKER_BIN = MICRO / "bin" / "bootstrap.s1main"

SELECTED_SHA256 = "981960ad628f5029f8c54552c7b1cb1d440072e7e9dcffecdea52c6a3493a0f1"
# TRAIN-only sandbox view (h317-cl/res01-train-view/selected.json): identical train rows (verified with jq -cS), dev rows
# reduced to {context_sha256, family_id, dataset_id, dl_split} so no DL-Val-derived count (n_gate_pass…) is readable.
SELECTED_REDACTED_SHA256 = (
    "7d64774ad9dacb5fec79190edbe0c674e778556db84dcf29c474e239b68e8c79"
)
DOPE_SHA256 = "c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814"
H58_POOL_PRIORS_SHA256 = (
    "6db567ef2c10d2c67c375f7500305ec693a700c4e71eb23003b4cd312ee51cbe"
)

FOLD_SEED = 317140
PILOT_SEED = 317141
AUDIT_SEED = 317200
TRAIN_SEEDS = (317101, 317102, 317103)
SEARCH_SEEDS = (317101, 317102)
BOOT_SEED_GATE_H0 = 317150
BOOT_SEED_STAGE_V = 317160
N_BOOT = 10_000
N_FOLDS = 5
PILOT_N = 205  # all families, one context each (owner compute waiver 2026-11-04 ~13:16Z); P96 = exact subset
AUDIT_FRACTION = 0.30
GATE_FAIL_UTILITY = -2.0

# ----------------------------------------------------------------------------------------------------------------------
# Read guard
# ----------------------------------------------------------------------------------------------------------------------
# v2 (2026-11-04 ~12:30Z) after the leakage red team (redteam/leak/guard_patch_proposal.py): realpath + inode refusal,
# dev-identifier hex-run matching (cid 12-prefix, dataset_id, family hash), a denylist of DL-Val/holdout-bearing trees
# with no dev id in their names, path checks on rename/link/symlink/copy/chdir/glob/sqlite and exec argv, a frame check
# inside mixed_prefix_read, a cross-process refusal log, automatic install at import, and a pinned one-shot Stage V
# unlock. The in-process hook cannot see native readers or child processes reading paths; every Gate H0 process on
# xbabe2 therefore also runs inside the deny-by-default bubblewrap sandbox (sandbox_h317.sh) with a TRAIN-only view.
GUARD_COUNTERS: Dict[str, int] = {
    "refused_dlval_path": 0,
    "refused_gemini_mixed_cache": 0,
    "refused_mixed_h58_direct": 0,
    "refused_sealed_path": 0,
    "refused_denylisted_tree": 0,
    "allowed_mixed_h58_prefix_opens": 0,
    "allowed_dlval_stage_v_opens": 0,
}
_TLS = threading.local()
_STATE = {
    "dev_cids": frozenset(),
    "dev_p12": frozenset(),
    "dev_ds": frozenset(),
    "dev_fam12": frozenset(),
    "dev_inodes": frozenset(),
    "mixed_inodes": frozenset(),
    "stage_v": False,
    "installed": False,
}
_FORBIDDEN_DIRS = (
    str(Path("/home/ubuntu/jope-data/h256")) + os.sep,
    str(Path("/home/ubuntu/jope-data/h257")) + os.sep,
)
# DL-Val / holdout-bearing locations whose names need not carry a dev id (leakage red team inventory, 2026-11-04).
_DENY_PRE_V = (
    "/home/ubuntu/jope-data/res-01/full-packet.tar",
    "/home/ubuntu/JopeDimeMicro/assets/benchmark_data/",
    "/home/ubuntu/jope-evidence/",
    "/home/ubuntu/jope-data/gov-03/",
    "/home/ubuntu/jope-data/flywheel-redteam-",
    "/home/ubuntu/flywheel/data/",
    "/home/ubuntu/scratch/",
    "/home/ubuntu/jope-data/h317-cl/redteam/",
    # Stage V agent: other waves' evidence/inputs (e.g. H258 inputs/r882/best_dev_checkpoint.json holds DL-Val
    # outcomes) and shared result stores. H317's own evidence dir is exempt (see _ALLOW_PRE_V).
    "/home/ubuntu/jope_plan/evidence/",
    "/home/ubuntu/jope_plan/ns/results/",
    "/home/ubuntu/jope-data/h317-stagev/",
)
_ALLOW_PRE_V = ("/home/ubuntu/jope_plan/evidence/H317/",)
_DENY_SUBSTR_PRE_V = ("/research/flywheel_selfplay/custody/files/",)
_MIXED_H58 = ("pool_priors.npz", "context_features.npz")
_SEALED_TOKENS = (
    "sealed",
    "holdout",
    "meta_test",
    "meta-test",
    "fresh-cohort",
    "fresh_cohort",
    "dl-test",
    "dl_test",
)
_DEV_TOKENS = (
    "dev91",
    "dl-val",
    "dl_val",
    "dlval",
    "hold_out",
    "hold-out",
    "outer_test",
    "outer-test",
)
_HEX = re.compile(r"[0-9a-f]{12,64}")
LOCK_RECEIPT = Path("/home/ubuntu/jope_plan/ns/receipts/H317_LOCK_sha256.txt")


class GuardViolation(PermissionError):
    pass


def _guard_log_path() -> Path:
    """Production log: DATA_ROOT/guard_refusals.log. Tests set H317_GUARD_LOG so probe refusals stay out of it."""
    return Path(os.environ.get("H317_GUARD_LOG", str(DATA_ROOT / "guard_refusals.log")))


def _log_refusal(kind: str, path: str) -> None:
    """Append every refusal to a cross-process log (forked/spawned children's counters never reach the parent)."""
    if getattr(_TLS, "logging", False):
        return
    _TLS.logging = True
    try:
        fd = os.open(
            str(_guard_log_path()), os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644
        )
        try:
            os.write(fd, f"{os.getpid()}\t{kind}\t{path}\n".encode(errors="replace"))
        finally:
            os.close(fd)
    except OSError:
        pass
    finally:
        _TLS.logging = False


def _refuse(counter: str, msg: str, path: str):
    GUARD_COUNTERS[counter] += 1
    _log_refusal(counter, path)
    raise GuardViolation(f"H317 guard: {msg}: {path}")


def _dev_named(low: str) -> bool:
    for m in _HEX.finditer(low):
        s = m.group(0)
        if (
            s[:12] in _STATE["dev_p12"]
            or s[:16] in _STATE["dev_ds"]
            or s[:12] in _STATE["dev_fam12"]
        ):
            return True
    return False


def _from_prefix_loader() -> bool:
    f = sys._getframe(1)
    ok = False
    while f is not None:
        fn = f.f_code.co_filename
        if (
            "npyio" in fn
        ):  # np.load of a mixed file = whole-member decode: never allowed pre-lock (numpy 1.x and 2.x)
            return False
        if fn.endswith("trainonly_loader.py"):
            ok = True
        f = f.f_back
    return ok


def _check_path(raw) -> None:
    if raw is None or isinstance(raw, int):
        return
    try:
        raw = os.fsdecode(raw)
    except TypeError:
        return
    if not raw:
        return
    a = os.path.abspath(raw)
    r = os.path.realpath(raw)
    stage_v = _STATE["stage_v"]
    for p in {a, r}:
        low = p.lower()
        if any(p.startswith(d) or p == d.rstrip(os.sep) for d in _FORBIDDEN_DIRS):
            _refuse("refused_gemini_mixed_cache", "mixed Gemini cache is off-limits", p)
        if any(tok in low for tok in _SEALED_TOKENS):
            _refuse("refused_sealed_path", "sealed/holdout path refused", p)
        if not stage_v:
            if (
                any(p.startswith(d) or p == d.rstrip(os.sep) for d in _DENY_PRE_V)
                and not any(p.startswith(a) for a in _ALLOW_PRE_V)
            ) or any(s in p for s in _DENY_SUBSTR_PRE_V):
                _refuse(
                    "refused_denylisted_tree",
                    "DL-Val/holdout-bearing tree before Stage V",
                    p,
                )
            if any(t in low for t in _DEV_TOKENS) or _dev_named(low):
                _refuse(
                    "refused_dlval_path", "DL-Val identifier in path before Stage V", p
                )
    try:
        st = os.stat(r)
    except OSError:
        return
    key = (st.st_dev, st.st_ino)
    if key in _STATE["mixed_inodes"]:
        if getattr(_TLS, "mixed_prefix", False) and _from_prefix_loader():
            GUARD_COUNTERS["allowed_mixed_h58_prefix_opens"] += 1
        elif stage_v:
            # Stage V only: DL-Val r882 base_prior_cost lives only in the mixed pool_priors.npz (dev rows = suffix).
            GUARD_COUNTERS["allowed_dlval_stage_v_opens"] += 1
        else:
            _refuse(
                "refused_mixed_h58_direct",
                "mixed H58 file outside the H308 prefix loader",
                raw,
            )
    if key in _STATE["dev_inodes"]:
        if stage_v:
            GUARD_COUNTERS["allowed_dlval_stage_v_opens"] += 1
        else:
            _refuse(
                "refused_dlval_path", "DL-Val file by inode (link/rename/symlink)", raw
            )


_PATH_EVENTS = {
    "open": (0,),
    "os.listdir": (0,),
    "os.scandir": (0,),
    "os.chdir": (0,),
    "glob.glob": (0,),
    "glob.glob/2": (0, 2),
    "os.rename": (0, 1),
    "os.link": (0, 1),
    "os.symlink": (0, 1),
    "shutil.copyfile": (0, 1),
    "shutil.copytree": (0, 1),
    "shutil.move": (0, 1),
    "sqlite3.connect": (0,),
    "os.truncate": (0,),
}
_EXEC_EVENTS = {"subprocess.Popen": 1, "os.exec": 1, "os.posix_spawn": 1, "os.spawn": 2}


def _audit(event: str, args) -> None:
    idx = _PATH_EVENTS.get(event)
    if idx is not None:
        for i in idx:
            if i < len(args):
                _check_path(args[i])
        return
    if event in _EXEC_EVENTS or event == "os.system":
        argv = (
            [args[0]] if event == "os.system" else list(args[_EXEC_EVENTS[event]] or [])
        )
        for tok_src in argv:
            s = (
                os.fsdecode(tok_src)
                if isinstance(tok_src, (bytes, str, os.PathLike))
                else ""
            )
            for tok in re.split(r"[\s=:,;'\"]+", s):
                if "/" in tok or _HEX.search(tok.lower()):
                    _check_path(tok)


class mixed_prefix_read:
    """Allow the mixed H58 npz files ONLY for the H308 TRAIN-prefix pread loader (frame-checked; np.load refused)."""

    def __enter__(self):
        _TLS.mixed_prefix = True
        return self

    def __exit__(self, *exc):
        _TLS.mixed_prefix = False
        return False


def _stat_key(p) -> tuple | None:
    try:
        st = os.stat(p)
        return (st.st_dev, st.st_ino)
    except OSError:
        return None


def install_guard() -> None:
    if _STATE["installed"]:
        return
    sel = load_selected()
    dev = sel["dev"]
    _STATE["dev_cids"] = frozenset(r["context_sha256"] for r in dev)
    _STATE["dev_p12"] = frozenset(r["context_sha256"][:12] for r in dev)
    _STATE["dev_ds"] = frozenset(r["dataset_id"].lower() for r in dev)
    _STATE["dev_fam12"] = frozenset(r["family_id"].split("-", 1)[-1][:12] for r in dev)
    # Self-check: no TRAIN identifier may trigger the dev-identifier matcher (no false refusals of TRAIN files).
    for r in sel["train"]:
        for ident in (
            r["context_sha256"],
            r["dataset_id"].lower(),
            r["family_id"].split("-", 1)[-1],
        ):
            if _dev_named(ident):
                raise RuntimeError(
                    f"H317 guard self-check: TRAIN identifier collides with a dev identifier: {ident}"
                )
    keys = set()
    for r in dev:  # stat() is metadata only; no DL-Val byte is read
        for p in (
            RES01 / "tables" / f"{r['context_sha256']}.jtt1",
            RES01 / "pools" / f"{r['context_sha256']}.npz",
            Path("/home/ubuntu/jope-data/res-01/input/tables")
            / f"{r['context_sha256']}.jtt1",
            Path("/home/ubuntu/jope-data/res-01/input/pools")
            / f"{r['context_sha256']}.npz",
        ):
            k = _stat_key(p)
            if k:
                keys.add(k)
    k = _stat_key("/home/ubuntu/jope-data/res-01/full-packet.tar")
    if k:
        keys.add(k)
    _STATE["dev_inodes"] = frozenset(keys)
    _STATE["mixed_inodes"] = frozenset(
        k for k in (_stat_key(H58_CACHE / n) for n in _MIXED_H58) if k
    )
    sys.addaudithook(_audit)
    _STATE["installed"] = True


def _require_guard() -> None:
    if not _STATE["installed"]:
        raise GuardViolation("H317 guard not installed")


def unlock_stage_v(*_ignored) -> None:
    """Stage V unlock, pinned: no caller-supplied sha. LOCK.json at DATA_ROOT/stage_v/LOCK.json must match the
    create-once receipt ~/jope_plan/ns/receipts/H317_LOCK_sha256.txt (also posted on comms); every source/worker/model
    hash listed in LOCK['code_sha256'] is rechecked; a create-once STAGE_V_STARTED marker makes it one-shot."""
    _require_guard()
    lock_path = DATA_ROOT / "stage_v" / "LOCK.json"
    posted = LOCK_RECEIPT.read_text().split()[0]
    got = sha256_file(lock_path)
    if got != posted:
        raise GuardViolation(f"LOCK.json sha {got} != posted receipt {posted}")
    lock = json.loads(lock_path.read_text())
    for rel, sha in lock["code_sha256"].items():
        p = Path(rel) if rel.startswith("/") else HERE / rel
        if sha256_file(p) != sha:
            raise GuardViolation(f"file changed after LOCK: {rel}")
    marker = lock_path.parent / "STAGE_V_STARTED"
    fd = os.open(
        marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o444
    )  # FileExistsError on a second attempt
    os.write(fd, f"{os.getpid()} {got}\n".encode())
    os.fsync(fd)
    os.close(fd)
    _STATE["stage_v"] = True


def resume_stage_v() -> None:
    """Re-enter Stage V in a later process (e.g. the xbabe0 audit phase, or a resume after an infrastructure crash):
    the same LOCK/receipt/code-hash checks as unlock_stage_v, but the STAGE_V_STARTED marker must ALREADY exist and
    name this LOCK's sha; no new marker is created. Whether a unit may run is decided by the Stage V driver's
    outcome records (no retry of a recorded unit)."""
    _require_guard()
    lock_path = DATA_ROOT / "stage_v" / "LOCK.json"
    posted = LOCK_RECEIPT.read_text().split()[0]
    got = sha256_file(lock_path)
    if got != posted:
        raise GuardViolation(f"LOCK.json sha {got} != posted receipt {posted}")
    lock = json.loads(lock_path.read_text())
    for rel, sha in lock["code_sha256"].items():
        p = Path(rel) if rel.startswith("/") else HERE / rel
        if sha256_file(p) != sha:
            raise GuardViolation(f"file changed after LOCK: {rel}")
    marker = lock_path.parent / "STAGE_V_STARTED"
    try:
        named = marker.read_text().split()[1]
    except (OSError, IndexError):
        raise GuardViolation(
            "resume_stage_v: no STAGE_V_STARTED marker (run unlock_stage_v first)"
        )
    if named != got:
        raise GuardViolation(
            f"resume_stage_v: marker names LOCK {named}, current LOCK is {got}"
        )
    _STATE["stage_v"] = True


def assert_allowed(path: Path | str) -> None:
    """Check a path we hand to a subprocess (the Rust worker cannot be audited in-process)."""
    _check_path(os.fspath(path))


def guard_report() -> Dict[str, int | bool]:
    n_log = 0
    try:
        with open(_guard_log_path(), "rb") as f:
            n_log = sum(1 for _ in f)
    except OSError:
        pass
    return {
        **GUARD_COUNTERS,
        "stage_v_unlocked": bool(_STATE["stage_v"]),
        "installed": bool(_STATE["installed"]),
        "cross_process_refusal_log_lines": n_log,
        "guard_version": 2,
    }


# ----------------------------------------------------------------------------------------------------------------------
# Manifests
# ----------------------------------------------------------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


_SELECTED_CACHE: Dict[str, dict] = {}


def load_selected() -> dict:
    if "doc" not in _SELECTED_CACHE:
        p = RES01 / "selected.json"
        got = sha256_file(p)
        if got not in (SELECTED_SHA256, SELECTED_REDACTED_SHA256):
            raise RuntimeError(
                f"selected.json sha {got} is neither the pinned original nor the pinned redacted view"
            )
        _SELECTED_CACHE["doc"] = json.loads(p.read_text())
    return _SELECTED_CACHE["doc"]


def train_rows() -> List[dict]:
    rows = [r for r in load_selected()["train"] if r["dl_split"] == "DL-Train"]
    if len(rows) != 1219 or len({r["family_id"] for r in rows}) != 205:
        raise RuntimeError("TRAIN manifest is not 1,219 contexts / 205 families")
    return sorted(rows, key=lambda r: r["context_sha256"])


def dev_rows_for_stage_v() -> List[dict]:
    if not _STATE["stage_v"]:
        raise GuardViolation("dev rows requested before Stage V unlock")
    rows = [r for r in load_selected()["dev"]]
    if len(rows) != 91 or len({r["family_id"] for r in rows}) != 19:
        raise RuntimeError("DL-Val manifest is not 91 contexts / 19 families")
    return sorted(rows, key=lambda r: r["context_sha256"])


def table_path(cid: str) -> Path:
    _require_guard()
    return RES01 / "tables" / f"{cid}.jtt1"


def pool_path(cid: str) -> Path:
    _require_guard()
    return RES01 / "pools" / f"{cid}.npz"


def stream_seed(*parts) -> int:
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(h[:8], "little") & ((1 << 63) - 1)


def family_folds(
    rows: Sequence[dict], seed: int = FOLD_SEED, k: int = N_FOLDS
) -> Dict[str, int]:
    """Family-grouped k-fold: families sorted by (size desc, seeded key) are greedily placed in the smallest fold."""
    by_fam: Dict[str, int] = {}
    for r in rows:
        by_fam[r["family_id"]] = by_fam.get(r["family_id"], 0) + 1
    fams = sorted(by_fam, key=lambda f: (-by_fam[f], stream_seed(seed, f)))
    load = [0] * k
    fam_fold: Dict[str, int] = {}
    for f in fams:
        j = min(range(k), key=lambda i: (load[i], i))
        fam_fold[f] = j
        load[j] += by_fam[f]
    return {r["context_sha256"]: fam_fold[r["family_id"]] for r in rows}


# ----------------------------------------------------------------------------------------------------------------------
# JTT1 envelopes
# ----------------------------------------------------------------------------------------------------------------------
class Jtt1:
    __slots__ = ("magic", "nt", "nq", "width", "flag", "folds", "mat")

    def __init__(self, magic, nt, nq, width, flag, folds, mat):
        self.magic, self.nt, self.nq, self.width, self.flag, self.folds, self.mat = (
            magic,
            nt,
            nq,
            width,
            flag,
            folds,
            mat,
        )

    @property
    def y(self) -> np.ndarray:
        return self.mat[:, -1]

    def to_bytes(self) -> bytes:
        assert self.mat.shape == (self.nt, self.width + 1) and self.folds.shape == (
            self.nt,
        )
        return (
            self.magic
            + struct.pack("<III", self.nt, self.nq, self.width)
            + bytes([self.flag])
            + self.folds.astype(np.uint8).tobytes()
            + np.ascontiguousarray(self.mat, dtype="<f8").tobytes()
        )

    def to_worker_bytes(self) -> bytes:
        """Train-only envelope + 10 all-zero query records: `automatic_promotion.training_cv_envelope` (main) exactly.

        The pinned S1 parser checks nq >= 10 but never decodes query cells; no outer rows exist anywhere in H317.
        """
        raw = bytearray(self.to_bytes())
        struct.pack_into("<I", raw, 8, 10)
        raw.extend(bytes(10 * (self.width + 1) * 8))
        return bytes(raw)


def read_jtt1(path: Path) -> Jtt1:
    _require_guard()
    raw = Path(path).read_bytes()
    magic = raw[:4]
    if magic != b"JTT1":
        raise ValueError(f"bad JTT1 magic in {path}")
    nt, nq, width = struct.unpack_from("<III", raw, 4)
    if nq != 0:
        raise ValueError(
            f"{path}: query rows present (nq={nq}); H317 expects train-only envelopes"
        )
    flag = raw[16]
    folds = np.frombuffer(raw, dtype=np.uint8, count=nt, offset=17).copy()
    mat = (
        np.frombuffer(raw, dtype="<f8", count=nt * (width + 1), offset=17 + nt)
        .reshape(nt, width + 1)
        .copy()
    )
    if len(raw) != 17 + nt + 8 * nt * (width + 1):
        raise ValueError(f"{path}: unexpected size")
    return Jtt1(magic, nt, nq, width, flag, folds, mat)


def audit_mask(
    cid: str, y: np.ndarray, fraction: float = AUDIT_FRACTION, seed: int = AUDIT_SEED
) -> np.ndarray:
    """Target-quantile stratified AUDIT rows: sort by y, blocks of 10, 3 seeded picks per block (30%)."""
    n = len(y)
    rng = np.random.default_rng(stream_seed(seed, cid))
    order = np.argsort(y, kind="stable")
    mask = np.zeros(n, dtype=bool)
    per_block = int(round(10 * fraction))
    for s in range(0, n, 10):
        blk = order[s : s + 10]
        take = min(len(blk), max(0, int(round(per_block * len(blk) / 10))))
        if take:
            mask[rng.choice(blk, size=take, replace=False)] = True
    return mask


def search_and_audit_envelopes(cid: str, env: Jtt1) -> Tuple[Jtt1, Jtt1, np.ndarray]:
    """SEARCH = non-AUDIT rows keeping their native 2-fold labels; AUDIT envelope = fold0 SEARCH rows, fold1 AUDIT rows.

    `--fit-s1` on the AUDIT envelope gives inner_cv[1] = model fit on all SEARCH rows, predicting AUDIT rows, with the
    baseline = mean of SEARCH targets. Only that record is used for AUDIT utility (inner_cv[0] is ignored).
    """
    m = audit_mask(cid, env.y)
    keep = ~m
    s_folds = env.folds[keep]
    if len(np.unique(s_folds)) != 2:
        raise ValueError(f"{cid}: SEARCH lost a native fold")
    search = Jtt1(
        env.magic, int(keep.sum()), 0, env.width, env.flag, s_folds, env.mat[keep]
    )
    a_folds = np.where(m, 1, 0).astype(np.uint8)
    order = np.concatenate([np.flatnonzero(keep), np.flatnonzero(m)])
    audit = Jtt1(
        env.magic, env.nt, 0, env.width, env.flag, a_folds[order], env.mat[order]
    )
    return search, audit, m


# ----------------------------------------------------------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------------------------------------------------------
def native_utility(fitness: np.ndarray, gate: np.ndarray) -> np.ndarray:
    """Native91 / H258 utility: -fitness when the 2-fold gate passed, else -2.0."""
    fitness = np.asarray(fitness, dtype=np.float64)
    gate = np.asarray(gate, dtype=bool)
    return np.where(gate, -fitness, GATE_FAIL_UTILITY)


def s1_from_inner_cv(dsl: str, inner_cv: list) -> dict:
    """Byte-for-byte the JopeDimeMicro `compute_s1_metrics_from_inner_cv` formula plus utility."""
    if len(inner_cv) != 2:
        return {
            "status": "failed",
            "fitness": 100.0,
            "gate": False,
            "utility": GATE_FAIL_UTILITY,
        }
    ratios, wins = [], 0
    for rec in inner_cv:
        m = (rec.get("metrics") or {}).get("rmse")
        b = (rec.get("baseline_metrics") or {}).get("rmse")
        if (
            m is None
            or b is None
            or not math.isfinite(m)
            or not math.isfinite(b)
            or b <= 1e-12
        ):
            return {
                "status": "failed",
                "fitness": 100.0,
                "gate": False,
                "utility": GATE_FAIL_UTILITY,
            }
        r = max(float(m) / float(b), 1e-12)
        ratios.append(r)
        wins += int(r <= 1.0)
    lr0, lr1 = math.log(ratios[0]), math.log(ratios[1])
    mean_lr, tail_lr, sd_lr = 0.5 * (lr0 + lr1), max(lr0, lr1), 0.5 * abs(lr0 - lr1)
    comp = dsl.count("(")
    gate = wins == 2 and math.exp(mean_lr) <= 0.90
    if gate:
        fitness = 0.8 * mean_lr + 0.1 * tail_lr + 0.1 * sd_lr + 0.0005 * comp
    else:
        fitness = 100.0 + max(mean_lr, 0.0) + max(tail_lr, 0.0) + max(sd_lr, 0.0)
    return {
        "status": "completed",
        "fitness": float(fitness),
        "gate": bool(gate),
        "utility": float(-fitness) if gate else GATE_FAIL_UTILITY,
        "lr0": lr0,
        "lr1": lr1,
        "mean_lr": mean_lr,
        "tail_lr": tail_lr,
        "sd_lr": sd_lr,
        "complexity": comp,
    }


def audit_utility_from_rmse(
    model_rmse: float | None, baseline_rmse: float | None
) -> float:
    """AUDIT utility = -log(RMSE_audit(model fit on SEARCH) / RMSE_audit(mean of SEARCH y)); failure -> -2.0."""
    if model_rmse is None or baseline_rmse is None:
        return GATE_FAIL_UTILITY
    if (
        not (math.isfinite(model_rmse) and math.isfinite(baseline_rmse))
        or baseline_rmse <= 1e-12
    ):
        return GATE_FAIL_UTILITY
    return float(-math.log(max(model_rmse / baseline_rmse, 1e-12)))


# ----------------------------------------------------------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------------------------------------------------------
def family_means(values: Dict[str, float], fam_of: Dict[str, str]) -> Dict[str, float]:
    acc: Dict[str, List[float]] = {}
    for cid, v in values.items():
        acc.setdefault(fam_of[cid], []).append(float(v))
    return {f: float(np.mean(v)) for f, v in acc.items()}


def family_equal_mean(values: Dict[str, float], fam_of: Dict[str, str]) -> float:
    fm = family_means(values, fam_of)
    return float(np.mean(list(fm.values())))


def paired_family_bootstrap(
    arm: Dict[str, float],
    ref: Dict[str, float],
    fam_of: Dict[str, str],
    seed: int,
    n_boot: int = N_BOOT,
) -> dict:
    """Family-equal mean paired gain (arm - ref) with a whole-family percentile bootstrap."""
    cids = sorted(set(arm) & set(ref))
    if not cids:
        raise ValueError("no paired contexts")
    diff = {c: float(arm[c]) - float(ref[c]) for c in cids}
    fm = family_means(diff, fam_of)
    fams = sorted(fm)
    d = np.array([fm[f] for f in fams])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boots = d[idx].mean(axis=1)
    lo, hi = np.quantile(boots, [0.025, 0.975])
    wins = int((d > 1e-4).sum())
    losses = int((d < -1e-4).sum())
    return {
        "gain_family_mean": float(d.mean()),
        "gain_family_median": float(np.median(d)),
        "ci95": [float(lo), float(hi)],
        "lcb95": float(lo),
        "families": len(d),
        "contexts": len(cids),
        "wtl": [wins, len(d) - wins - losses, losses],
    }


def _family_diffs(
    arm: Dict[str, float], ref: Dict[str, float], fam_of: Dict[str, str]
) -> np.ndarray:
    cids = sorted(set(arm) & set(ref))
    fm = family_means({c: float(arm[c]) - float(ref[c]) for c in cids}, fam_of)
    return np.array([fm[f] for f in sorted(fm)])


def signflip_pvalue(
    arm: Dict[str, float],
    ref: Dict[str, float],
    fam_of: Dict[str, str],
    seed: int,
    n_perm: int = N_BOOT,
) -> float:
    """One-sided (arm > ref) family-level sign-flip permutation p-value of the family-equal mean gain."""
    d = _family_diffs(arm, ref, fam_of)
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_perm, len(d)))
    null = (signs * d).mean(axis=1)
    return float((1 + np.sum(null >= d.mean() - 1e-15)) / (n_perm + 1))


def robustness_summary(
    arm: Dict[str, float], ref: Dict[str, float], fam_of: Dict[str, str], seed: int
) -> dict:
    """Bootstrap + sign-flip + leave-one-family-out + drop-top-k + harm share (repo common experimental contract)."""
    boot = paired_family_bootstrap(arm, ref, fam_of, seed)
    d = _family_diffs(arm, ref, fam_of)
    order = np.sort(d)[::-1]
    lofo = (
        [float(np.delete(d, i).mean()) for i in range(len(d))]
        if len(d) > 1
        else [float(d.mean())]
    )
    drop = {
        f"drop_top_{k}": (float(order[k:].mean()) if len(order) > k else None)
        for k in (1, 3, 5, 10)
    }
    return {
        **boot,
        "signflip_p_one_sided": signflip_pvalue(arm, ref, fam_of, seed),
        "lofo_min_gain": float(min(lofo)),
        "harmed_share_gt_0p01": float(np.mean(d < -0.01)),
        **drop,
    }


def holm_adjusted_lcb_alpha(m: int, rank: int, alpha: float = 0.05) -> float:
    """Two-sided alpha for the `rank`-th smallest p (0-based) under Holm step-down over m contrasts."""
    return alpha / (m - rank)


def contract_advance_gate(
    summary: dict,
    lcb_key: str = "lcb95",
    margin: float = 0.001,
    signflip_alpha: float = 0.025,
    signflip_p: float | None = None,
) -> dict:
    """Repo common contract: adjusted family-mean LCB > +0.001, family median >= 0, <= 10% families losing > 0.01;
    H317 adds: one-sided sign-flip p < alpha and leave-one-family-out minimum gain > 0.

    Holm (GP code red team MEDIUM-1): callers testing m challengers pass either `signflip_alpha` = the rank's Holm
    one-sided level (e.g. 0.025/(m-rank)), or `signflip_p` = an already Holm-adjusted p (compared with 0.025)."""
    p = summary["signflip_p_one_sided"] if signflip_p is None else signflip_p
    checks = {
        "lcb_gt_margin": summary[lcb_key] > margin,
        "median_ge_0": summary["gain_family_median"] >= 0.0,
        "harm_le_10pct": summary["harmed_share_gt_0p01"] <= 0.10,
        "signflip_p_lt_alpha": p < signflip_alpha,
        "lofo_min_gt_0": summary["lofo_min_gain"] > 0.0,
    }
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "signflip_alpha": signflip_alpha,
        "signflip_p_used": p,
    }


def paired_family_bootstrap_alpha(
    arm: Dict[str, float],
    ref: Dict[str, float],
    fam_of: Dict[str, str],
    seed: int,
    alpha: float,
    n_boot: int = N_BOOT,
) -> dict:
    """Percentile interval at an arbitrary (e.g. Holm-adjusted) two-sided alpha."""
    d = _family_diffs(arm, ref, fam_of)
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return {
        "alpha": alpha,
        "lcb": float(lo),
        "ucb": float(hi),
        "gain_family_mean": float(d.mean()),
    }


def write_json(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, sort_keys=True, indent=1, default=_json_default)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def pilot_shard(
    rows: Sequence[dict],
    folds: Dict[str, int],
    n: int = PILOT_N,
    seed: int = PILOT_SEED,
) -> List[str]:
    """n contexts from n distinct families, ~equal per outer fold, one seeded context per chosen family.

    n >= number of families (or n <= 0) selects ALL families, one seeded context each. The per-family context choice
    does not depend on n, so every smaller pilot (P24/P48/P96) is an exact subset of the all-family pilot (P205).
    """
    by_fold: Dict[int, List[str]] = {}
    fam_ctx: Dict[str, List[str]] = {}
    for r in rows:
        fam_ctx.setdefault(r["family_id"], []).append(r["context_sha256"])
    fam_fold = {r["family_id"]: folds[r["context_sha256"]] for r in rows}
    for f, j in fam_fold.items():
        by_fold.setdefault(j, []).append(f)
    if n <= 0 or n >= len(fam_fold):
        quota = [len(by_fold[j]) for j in range(N_FOLDS)]
    else:
        quota = [n // N_FOLDS + (1 if j < n % N_FOLDS else 0) for j in range(N_FOLDS)]
    chosen: List[str] = []
    for j in range(N_FOLDS):
        fams = sorted(by_fold[j], key=lambda f: stream_seed(seed, "fam", f))[: quota[j]]
        for f in fams:
            ctxs = sorted(fam_ctx[f], key=lambda c: stream_seed(seed, "ctx", c))
            chosen.append(ctxs[0])
    return sorted(chosen)


# The guard is installed by importing this module (leakage red team HIGH-2: an opt-in guard protects nothing).
install_guard()


if __name__ == "__main__":
    rows = train_rows()
    folds = family_folds(rows)
    pilot = pilot_shard(rows, folds)
    sizes = [sum(1 for v in folds.values() if v == j) for j in range(N_FOLDS)]
    print(
        json.dumps(
            {
                "train": len(rows),
                "fold_sizes": sizes,
                "pilot": len(pilot),
                "guard": guard_report(),
            }
        )
    )
