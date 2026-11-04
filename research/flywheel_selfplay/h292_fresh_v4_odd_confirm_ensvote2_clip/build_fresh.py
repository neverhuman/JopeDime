"""H283 fresh-context builder (xbabe2 CPU, hw_t2305): the corpus's own construction, applied to fresh S1-prod contexts.

Per context (one per cohort family):
  table  = the S1 context's teacher.jtt1.gz with the test rows stripped (byte-identical to the corpus tables);
  pool   = the S1 summary.tar.gz accepts, extracted exactly as research/flywheel_selfplay/dataset_builder.py
           extract_outer_pool_and_champions does (constrained_fitness, sha256(dsl) order, rank_prepare 412D features);
  feats  = H58 prepare_cache.py's extract_single_table_features / compute_candidate_priors on CPU with the frozen
           network (dope629, lm48_z, tokens, base_prior_cost, sos_prior, proj16, scale).
--regression rebuilds N corpus DL-Val contexts and requires table bytes, pool arrays and H58 cache arrays to be equal."""

import argparse
import gzip
import hashlib
import importlib.util
import json
import math
import struct
import sys
import tarfile
from pathlib import Path

import numpy as np

CORRECTED = Path("/home/ubuntu/jope-evidence/dope-lift-20261021/corrected-study")
FEAT_NAMES = CORRECTED / "program-feature-names.json"
S1 = Path("/home/ubuntu/scratch/jopedime-dl-training-v1/receipts/trainonly-v2/s1-prod")
ADMITTED = Path(
    "/home/ubuntu/jope-evidence/dope-lift-20261021/deep-learning-dl03/admitted-input-contexts.json"
)
H50 = Path("/home/ubuntu/jope-data/h50/input")
H58 = Path("/home/ubuntu/jope-data/h58/cache")
H58_SRC = Path(__file__).resolve().parent.parent / "h58" / "prepare_cache.py"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def train_only_table(teacher_gz):
    raw = gzip.decompress(Path(teacher_gz).read_bytes())
    nt, nq, w = struct.unpack_from("<III", raw, 4)
    if raw[:4] != b"JTT1" or raw[16] != 2:
        raise RuntimeError("teacher JTT1 header")
    body = raw[17 + nt : 17 + nt + nt * (w + 1) * 8]
    return (
        raw[:4] + struct.pack("<III", nt, 0, w) + raw[16:17] + raw[17 : 17 + nt] + body
    )


def constrained_fitness(r):
    """dataset_builder.py constrained_fitness, verbatim."""
    ratio = float(r.get("cv_geomean_rmse_ratio") or 1e9)
    mean = float(r.get("cv_mean_log_ratio") or math.log(max(ratio, 1e-12)))
    tail = float(r.get("cv_tail_log_ratio") or mean)
    sd = float(r.get("cv_log_ratio_sd") or 0.0)
    comp = float(r.get("complexity_open_parens") or str(r.get("dsl") or "").count("("))
    gate = bool(r.get("cv_gate_passed")) and math.isfinite(ratio) and ratio <= 0.90
    if not (math.isfinite(mean) and math.isfinite(tail) and math.isfinite(sd)):
        return 1e6, 30.0, False
    f = (
        (0.8 * mean + 0.1 * tail + 0.1 * sd + 0.0005 * comp)
        if gate
        else (100.0 + max(mean, 0.0) + max(tail, 0.0) + max(sd, 0.0))
    )
    return float(f), float(math.log(max(ratio, 1e-12))), gate


def extract_pool(tar_path, features, feat_names):
    """dataset_builder.py extract_outer_pool_and_champions pool branch, except that fitness is the GP record's own
    `fitness` when present (gate and logratio from constrained_fitness): this reproduces the corpus DL03 pools exactly
    (regression), whereas constrained_fitness differs on failed-gate candidates."""
    by_dsl = {}
    with tarfile.open(tar_path, "r:gz") as tf:
        for member in tf.getmembers():
            if not (
                member.name.endswith("/accepts.jsonl") or member.name == "accepts.jsonl"
            ):
                continue
            parts = member.name.split("/")
            gen = next(
                (int(p[3:]) for p in parts if p.startswith("gen") and p[3:].isdigit()),
                0,
            )
            fobj = tf.extractfile(member)
            if fobj is None:
                continue
            for line in fobj:
                if not line.strip():
                    continue
                raw_rec = json.loads(line)
                rec = raw_rec.get("result") or raw_rec
                prog_info = rec.get("program") or {}
                dsl = str(prog_info.get("dsl") or rec.get("dsl") or "").strip()
                if not dsl:
                    continue
                f, lr, gate = constrained_fitness(rec)
                if rec.get("fitness") is not None:
                    f = float(
                        rec["fitness"]
                    )  # the corpus (DL03) pools carry the GP record's own fitness
                seq = int(prog_info.get("birth_seq") or rec.get("seq") or 0)
                cur = by_dsl.get(dsl)
                if cur is None:
                    by_dsl[dsl] = dict(
                        dsl=dsl,
                        fitness=f,
                        logratio=lr,
                        gate=gate,
                        seq_min=seq,
                        seq_max=seq,
                        gen_mask=1 << gen,
                        count=1,
                    )
                else:
                    if f < cur["fitness"]:
                        cur["fitness"], cur["logratio"], cur["gate"] = f, lr, gate
                    cur["seq_min"] = min(cur["seq_min"], seq)
                    cur["seq_max"] = max(cur["seq_max"], seq)
                    cur["gen_mask"] |= 1 << gen
                    cur["count"] += 1
    items = sorted(
        by_dsl.values(), key=lambda x: hashlib.sha256(x["dsl"].encode()).hexdigest()
    )
    n = len(items)
    pool = dict(
        program=np.zeros((n, len(feat_names)), dtype=np.float32),
        fitness=np.zeros((n,), dtype=np.float64),
        logratio=np.zeros((n,), dtype=np.float64),
        gate=np.zeros((n,), dtype=np.bool_),
        program_sha256=np.zeros((n,), dtype="S64"),
        birth_seq_min=np.zeros((n,), dtype=np.int32),
        birth_seq_max=np.zeros((n,), dtype=np.int32),
        generation_mask=np.zeros((n,), dtype=np.int32),
        delivery_count=np.zeros((n,), dtype=np.int16),
    )
    for i, it in enumerate(items):
        d = features(it["dsl"])
        pool["program"][i] = [float(d.get(k, 0.0)) for k in feat_names]
        pool["fitness"][i] = it["fitness"]
        pool["logratio"][i] = it["logratio"]
        pool["gate"][i] = it["gate"]
        pool["program_sha256"][i] = (
            hashlib.sha256(it["dsl"].encode()).hexdigest().encode("ascii")
        )
        pool["birth_seq_min"][i] = it["seq_min"]
        pool["birth_seq_max"][i] = it["seq_max"]
        pool["generation_mask"][i] = it["gen_mask"]
        pool["delivery_count"][i] = min(it["count"], 32767)
    return pool


def load_h58():
    sys.path.insert(
        0, str(H58_SRC.parents[3] / "python/src/jope/north_star")
    )  # dope_freeze_guard_types
    spec = importlib.util.spec_from_file_location("h58_prepare_cache", H58_SRC)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def setup():
    sys.path.insert(0, str(CORRECTED))
    from rank_prepare import features

    feat_names = json.loads(FEAT_NAMES.read_text())
    h58 = load_h58()
    import torch

    torch.set_num_threads(4)
    fused_src = H50 / "cpu_exact_fused.py"
    if sha(fused_src) != h58.FUSED_SOURCE_SHA:
        raise RuntimeError("fused source")
    spec = importlib.util.spec_from_file_location("cpu_exact_fused", fused_src)
    fused = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fused)
    net = fused.FusedJopeNetwork(device=torch.device("cpu"), adapter_variant="h40b")
    net.eval()
    ws = h58.base.FusedLandmarkWorkspace(num_threads=4)
    return features, feat_names, h58, net, ws, torch


def context_features(h58, net, ws, torch, table_bytes, prog):
    raw = table_bytes
    nt, _, width = struct.unpack_from("<III", raw, 4)
    values = (
        np.frombuffer(raw, dtype="<f8", offset=17 + nt).reshape(nt, width + 1).copy()
    )
    tok, mask, d629, lm48_z, ctx584 = h58.extract_single_table_features(
        net, ws, values[:, :-1], values[:, -1], torch.device("cpu")
    )
    bp, sp = h58.compute_candidate_priors(net, prog, lm48_z, torch.device("cpu"))
    return dict(
        tokens_10x32=tok, mask_10=mask, dope629=d629, lm48_z=lm48_z, ctx_feat_584=ctx584,
        base_prior_cost=bp, sos_prior=sp, proj16=(prog.astype(np.float64) @ h58.base.evaluator.get_w1_sign_matrix() / np.sqrt(412)).astype(np.float32),
    )  # fmt: skip


def regression(n):
    features, feat_names, h58, net, ws, torch = setup()
    sel = json.loads((H50 / "selected.json").read_text())["dev"][:n]
    adm = {r["context_sha256"]: r for r in json.loads(ADMITTED.read_text())}
    zc, zp = np.load(H58 / "context_features.npz"), np.load(H58 / "pool_priors.npz")
    cids = [str(c) for c in zc["cids"]]
    out = []
    for r in sel:
        cid = r["context_sha256"]
        a = adm[cid]
        table = train_only_table(
            Path(a["context_json_path"]).parent / "teacher.jtt1.gz"
        )
        res = dict(
            cid=cid, table_equal=table == (H50 / "tables" / f"{cid}.jtt1").read_bytes()
        )
        tar = Path(a["final_path"]).parent / "summary.tar.gz"
        with np.load(H50 / "pools" / f"{cid}.npz") as ref:
            if tar.exists():
                pool = extract_pool(tar, features, feat_names)
                res["pool_equal"] = all(
                    np.array_equal(pool[k], ref[k])
                    for k in ("program", "fitness", "gate", "program_sha256")
                )
                res["pool_n"] = [len(pool["fitness"]), len(ref["fitness"])]
            else:
                res["pool_equal"] = None
            prog = ref["program"].astype(np.float32)
        f = context_features(h58, net, ws, torch, table, prog)
        i = cids.index(cid)
        lo, hi = int(zp["offsets"][i]), int(zp["offsets"][i + 1])
        res["ctx_equal"] = all(
            np.array_equal(f[k], zc[k][i])
            for k in ("dope629", "lm48_z", "tokens_10x32", "mask_10", "ctx_feat_584")
        )
        res["prior_equal"] = all(
            np.array_equal(f[k], zp[k][lo:hi])
            for k in ("base_prior_cost", "sos_prior", "proj16")
        )
        out.append(res)
        print(json.dumps(res), flush=True)
    return out


def build(cohort_path, out_dir, wave="H283"):
    features, feat_names, h58, net, ws, torch = setup()
    cohort = json.loads(Path(cohort_path).read_text())
    fams = cohort["assignment"][wave]["families"]
    by_key = {r["family_key"]: r for r in cohort["eligible"]}
    out = Path(out_dir)
    (out / "pools").mkdir(parents=True)
    (out / "tables").mkdir()
    rows, ctx, pri = [], {}, {}
    for k in fams:
        e = by_key[k]
        c = e["context"]
        cid = c["context_sha256"]
        teacher = Path(c["context_dir"]) / "teacher.jtt1.gz"
        cj = json.loads((Path(c["context_dir"]) / "context.json").read_text())
        if (
            hashlib.sha256(gzip.decompress(teacher.read_bytes())).hexdigest()
            != cj["files"]["teacher.jtt1.gz"]["raw_sha256"]
        ):
            raise RuntimeError("teacher hash " + cid)
        table = train_only_table(teacher)
        (out / "tables" / f"{cid}.jtt1").write_bytes(table)
        pool = extract_pool(
            S1 / "contexts" / cid / "summary.tar.gz", features, feat_names
        )
        np.savez_compressed(out / "pools" / f"{cid}.npz", **pool)
        row = dict(
            context_sha256=cid, dataset_id=e["dataset_id"], family_id="fresh:" + k, family_key=k,
            view_cut_id=c.get("cut_id"), source_split="train", dl_split="FRESH", task="regression",
            n_candidates=int(len(pool["fitness"])), n_gate_pass=int(pool["gate"].sum()),
            pool_sha256=sha(out / "pools" / f"{cid}.npz"), train_only_jtt1_sha256=hashlib.sha256(table).hexdigest(),
        )  # fmt: skip
        if row["n_candidates"] >= 8:
            f = context_features(h58, net, ws, torch, table, pool["program"])
            ctx[cid], pri[cid] = f, f
        rows.append(row)
        print(
            json.dumps(
                {
                    k2: row[k2]
                    for k2 in ("context_sha256", "n_candidates", "n_gate_pass")
                }
            ),
            flush=True,
        )
    built = [r for r in rows if r["n_candidates"] >= 8]
    ids = [r["context_sha256"] for r in built]
    np.savez_compressed(
        out / "context_features.npz", cids=np.asarray(ids),
        **{k: np.stack([ctx[c][k] for c in ids]) for k in ("tokens_10x32", "mask_10", "dope629", "lm48_z", "ctx_feat_584")},
    )  # fmt: skip
    offsets = np.cumsum([0] + [len(pri[c]["base_prior_cost"]) for c in ids])
    np.savez_compressed(
        out / "pool_priors.npz", cids=np.asarray(ids), offsets=offsets,
        **{k: np.concatenate([pri[c][k] for c in ids]) for k in ("base_prior_cost", "sos_prior", "proj16")},
    )  # fmt: skip
    (out / "selected.json").write_text(
        json.dumps(
            dict(
                protocol="fresh-cohort-v1/" + wave,
                fresh=built,
                unbuilt=[r for r in rows if r["n_candidates"] < 8],
            ),
            indent=1,
        )
        + "\n"
    )
    print(
        json.dumps(
            dict(
                built=len(built),
                unbuilt=len(rows) - len(built),
                families=len({r["family_key"] for r in built}),
            )
        )
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--regression", type=int)
    ap.add_argument("--cohort")
    ap.add_argument("--out")
    ap.add_argument("--wave", default="H283")  # H286: cohort v2 assignment key
    a = ap.parse_args()
    if a.regression:
        res = regression(a.regression)
        ok = all(
            r["table_equal"]
            and r["ctx_equal"]
            and r["prior_equal"]
            and r["pool_equal"] is not False
            for r in res
        )
        print("REGRESSION", "PASS" if ok else "FAIL")
        sys.exit(0 if ok else 1)
    build(a.cohort, a.out, a.wave)
