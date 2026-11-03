"""Approved corpus pool construction; no final.json or outer-test row decode."""

import gzip
import struct
import hashlib
import tarfile
import json
import math
import re
import numpy as np


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


def features(dsl):
    ops = re.findall(r"\(([^()]+)\)", dsl)
    result = {"steps": float(len(ops)), "dsl_characters": float(len(dsl))}
    for pos, content in enumerate(ops):
        tokens = content.split()
        op = tokens[0]
        result["count." + op] = result.get("count." + op, 0) + 1
        result[f"position{pos}.{op}"] = 1.0
        if pos == len(ops) - 1:
            result["estimator." + op] = 1.0
        for token in tokens[1:]:
            if "=" not in token:
                continue
            key, value = token.split("=", 1)
            if key == "seed":
                continue  # random IDs are not algorithm semantics
            try:
                value = float(value)
            except ValueError:
                if value not in ["true", "false"]:
                    continue
                value = float(value == "true")
            value = float(np.sign(value) * np.log1p(abs(value)))
            result[f"parameter.{op}.{key}.sum"] = (
                result.get(f"parameter.{op}.{key}.sum", 0) + value
            )
            result[f"parameter.{op}.{key}.count"] = (
                result.get(f"parameter.{op}.{key}.count", 0) + 1
            )
            result[f"position{pos}.{key}"] = value
    return result


def train_only_table(path):
    with gzip.open(path, "rb") as f:
        header = f.read(17)
        if len(header) != 17 or header[:4] != b"JTT1" or header[16] != 2:
            raise ValueError("teacher header")
        nt, nq, width = struct.unpack_from("<III", header, 4)
        folds = f.read(nt)
        body = f.read(nt * (width + 1) * 8)
        if len(folds) != nt or len(body) != nt * (width + 1) * 8:
            raise ValueError("short train prefix")
        if max(folds) >= 2:
            raise ValueError("inner fold labels")
    return header[:4] + struct.pack("<III", nt, 0, width) + header[16:] + folds + body
