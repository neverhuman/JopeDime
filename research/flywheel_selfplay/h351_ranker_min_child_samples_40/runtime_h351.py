"""Pinned modules and physically restricted TRAIN-only packet construction."""

from pathlib import Path
import os, sys, json, hashlib, datetime, zipfile
import numpy as np

P = Path(__file__).resolve().parent.parent
D = Path("/home/ubuntu/jope-data/h351-gov03")
V = P / "vendor"
for name in [
    "h303_ensvote2_fully_nested_hparams",
    "h302_ensvote2_trainseed_nested_cv",
    "h308_ensvote2_trainonly_repartition",
]:
    sys.path.insert(0, str(V / name))
import run_h303 as r303
import run_h302 as r302
import analysis_h302 as a302
import trainonly_loader as tl

PARTS = ("P0", "P1", "P2")
SEEDS = (30001, 30002, 30003)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def durable(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".new")
    with tmp.open("w") as f:
        json.dump(obj, f, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    fd = os.open(path.parent, os.O_DIRECTORY)
    os.fsync(fd)
    os.close(fd)


def fold(part, f):
    return (
        a302.outer_fold_of_family(f)
        if part == "P0"
        else hashlib.sha256(
            {"P1": b"H308-outer-v1\0", "P2": b"H308-outer-v2\0"}[part] + f.encode()
        ).digest()[-1]
        % 4
    )


def initialize(require_prereg=True):
    pins = json.loads((P / "source/manifest.json").read_text())
    for rel, digest in pins["sources"].items():
        if rel.endswith(".py") or Path(rel).name in [
            "manifest.json",
            "s3_manifest.json",
        ]:
            assert sha(V / rel) == digest, rel
    if require_prereg:assert json.loads((P / "prereg-published.json").read_text())["published"]
    r302.install_h302_audit_guard()
    core, topk, veto, bagged, ens = r302.load_pinned_h287_modules(
        D / "setup/pinned_h287_mods"
    )
    r302.MAX_RSS_BYTES = 6 * 2**30
    core.INPUT_POOLS_DIR = r302.POOLS
    # Default pause file must not point at another owner's lane.
    os.environ["H265_PAUSE_FILE"] = str(D / "h351.pause.json")
    selected_path = r302.POOLS / "selected.json"
    assert sha(selected_path) == r302.SELECTED_JSON_SHA256
    rows = json.loads(selected_path.read_text())["train"]
    assert len(rows) == 1219 and len({r["family_id"] for r in rows}) == 205
    assert all(
        r.get("has_meta_test") is not True and r["task"] == "regression" for r in rows
    )
    r303.G.update(
        r302=r302,
        a302=a302,
        core=core,
        topk=topk,
        veto=veto,
        bagged=bagged,
        ens=ens,
        train_rows=rows,
        fam_by_cid={r["context_sha256"]: r["family_id"] for r in rows},
    )
    return r303.G


class Scope:
    def __init__(self, allowed, all_train, fit=False):
        self.allowed = set(allowed)
        self.all_train = set(all_train)
        assert self.allowed <= self.all_train
        self.fit = fit
        self.bytes = 0
        self.contexts = set()
        self.ranges = []
        self.denied = 0
        self.active = True
        self.expected = {}

    def require(self, cid):
        if cid not in self.allowed:
            self.denied += 1
            raise PermissionError("Context outside worker input closure")

    def hook(self, event, args):
        if (
            not self.active
            or event != "open"
            or not args
            or not isinstance(args[0], (str, bytes, os.PathLike))
        ):
            return
        path = Path(os.fsdecode(args[0]))
        s = str(path)
        if "/h50/input/pools/" in s and path.suffix == ".npz":
            self.require(path.stem)
        if self.fit and any(x in s for x in ["/h351-gov03/base/","/h351-gov03/base_native/"]):
            raise PermissionError(
                "Historical outer outcome access forbidden during fit"
            )

    def read(self, path, key, lo, hi, cid):
        self.require(cid)
        if self.expected.get((str(path), key, cid)) != (lo, hi):
            self.denied += 1
            raise PermissionError("Byte range outside declared context")
        off, shape, dtype = tl._member_layout(path, key + ".npy")
        assert 0 <= lo <= hi <= shape[0]
        stride = (
            int(np.prod(shape[1:], dtype=np.int64)) * dtype.itemsize
            if len(shape) > 1
            else dtype.itemsize
        )
        fd = os.open(path, os.O_RDONLY)
        try:
            b = os.pread(fd, (hi - lo) * stride, off + lo * stride)
        finally:
            os.close(fd)
        assert len(b) == (hi - lo) * stride
        self.bytes += len(b)
        self.contexts.add(cid)
        self.ranges.append(
            dict(
                file=str(path),
                member=key,
                context=cid,
                lo=lo,
                hi=hi,
                bytes=len(b),
                sha256=hashlib.sha256(b).hexdigest(),
            )
        )
        return np.frombuffer(b, dtype=dtype).reshape((hi - lo,) + shape[1:]).copy()


def packets(rows, scope):
    G = r303.G
    cids = [r["context_sha256"] for r in rows]
    for c in cids:
        scope.require(c)
    costs = {}
    with np.load(r302.H276_INPUTS / "r882-train-costs.npz", allow_pickle=False) as z:
        for c in cids:
            costs[c] = z[c].astype(np.float32)
    ppath = r302.CACHE / "pool_priors.npz"
    cpath = r302.CACHE / "context_features.npz"
    pc = [str(c) for c in tl.read_all(ppath, "cids.npy")]
    offset = tl.read_all(ppath, "offsets.npy")
    cc = [str(c) for c in tl.read_all(cpath, "cids.npy")]
    n = len(scope.all_train)
    for ids in [pc, cc]:
        assert (
            len(set(ids[:n])) == n
            and set(ids[:n]) == scope.all_train
            and not scope.all_train & set(ids[n:])
        )
    pi = {c: i for i, c in enumerate(pc[:n])}
    ci = {c: i for i, c in enumerate(cc[:n])}
    prior = {}
    proj = {}
    ctx = {}
    for c in cids:
        i = pi[c]
        lo, hi = int(offset[i]), int(offset[i + 1])
        for key in ["base_prior_cost", "proj16"]:
            scope.expected[(str(ppath), key, c)] = (lo, hi)
        for key in ["lm48_z", "dope629"]:
            scope.expected[(str(cpath), key, c)] = (ci[c], ci[c] + 1)
        prior[c] = scope.read(ppath, "base_prior_cost", lo, hi, c).astype(np.float32)
        proj[c] = scope.read(ppath, "proj16", lo, hi, c).astype(np.float32)
        j = ci[c]
        ctx[c] = {
            key: scope.read(cpath, key, j, j + 1, c)[0].astype(np.float32)
            for key in ["lm48_z", "dope629"]
        }
    out = G["topk"].load_packets(rows, costs, prior, proj, ctx, G["ens"].K)
    scope.packet_hashes = {
        c: hashlib.sha256(
            b"".join(
                np.ascontiguousarray(out[c][k]).tobytes()
                for k in ["X", "utility_k", "rel_k", "gate_k"]
            )
        ).hexdigest()
        for c in cids
    }
    return out


def score_models(directory, members, packets, cids, chosen):
    G = r303.G
    X = np.concatenate([packets[c]["X"] for c in cids])
    outs = r303.score_members(directory, members, X, len(cids), G["ens"])
    ml = [
        r303.member_picks(
            o, chosen["tau"], chosen["k"], G["veto"], G["core"], G["bagged"]
        )
        for o in outs
    ]
    return np.asarray(r303.ens_picks(ml, chosen["a"], G["ens"]), dtype=np.int32)
