#!/usr/bin/env python3
"""H447 cross-scale `R4-e8` (`0.60 * 104M + 0.40 * 24M`) `[X, S]` 5-fold stack column builder (`cols_r4_e8.py`).

Commands:
  cols_r4_e8.py selftest
  cols_r4_e8.py run --bank FILE --ckpt-l CKPT_L --sha-l HEX --ckpt-m CKPT_M --sha-m HEX
                    [--ckpt-m20 CKPT_M20 --sha-m20 HEX] --out DIR
                    [--device cpu|cuda] [--threads 1] [--shard k/n | --queue --worker-id W]
                    [--budget-s S] [--ctx-s 7200]

Emits create-once `OUT/<cid>.npz` containing:
  - `q_r4_l`, `bag_r4_l`, `ref_r4_l`: 4-member `104M` (`L`) `[X, S]` residual (`perms=4` when `F+d_S<=30`, else `4`
    deterministic subsets under `tag="R4E8_L"` retaining all `d_S` structural columns `S`)
  - `q_r4_m`, `bag_r4_m`, `ref_r4_m`: 4-member `24M` (`M`) `[X, S]` residual (`perms=4` when `F+d_S<=30`, else `4`
    distinct deterministic subsets under `tag="R4E8_M"` retaining all `d_S` structural columns `S`)
  - `q_r4_e8`, `bag_r4_e8`, `ref_r4_e8`: `0.60 * r4_l + 0.40 * r4_m` (`8`-member cross-scale ensemble)
  - `q_cw5_m20`, `bag_cw5_m20`, `ref_cw5_m20` (when `--ckpt-m20` is given): exact `CW-5 R4` baseline on `24M h020`

`SC.build_struct_cols` and `RES.residual_targets` are computed ONCE per fold strictly on context rows (`0` query/audit
target reads; query targets passed as zeros).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import signal
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
if os.environ.get("WAVE_BASE_CODE"):
    sys.path.append(os.environ["WAVE_BASE_CODE"])

import icl_diag as DG  # noqa: E402
import icl_resid as RES  # noqa: E402
import struct_cols as SC  # noqa: E402
from icl_model import ICL  # noqa: E402

WIDE = 30
SUB = 30
W_L = 0.60
W_M = 0.40
SEED_BASE = 445_700


class _Timeout(Exception):
    pass


def _alarm(_s, _f):
    raise _Timeout()


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def subsets_keep_s(cid: str, f_raw: int, d_s: int, k_bag: int, tag: str) -> list[torch.Tensor]:
    """Returns `k_bag` deterministic index tensors into `[X, S]` that ALWAYS retain all `d_s` columns of `S`."""
    s_idx = np.arange(f_raw, f_raw + d_s, dtype=np.int64)
    k_raw = max(1, SUB - d_s)
    if f_raw <= k_raw:
        return [torch.from_numpy(np.concatenate([np.arange(f_raw, dtype=np.int64), s_idx]))]
    out = []
    for j in range(k_bag):
        h = int.from_bytes(hashlib.sha256(f"{tag}\0{cid}\0{j}".encode()).digest()[:8], "big")
        raw_pick = np.sort(np.random.default_rng(h).choice(f_raw, k_raw, replace=False))
        out.append(torch.from_numpy(np.concatenate([raw_pick, s_idx])))
    return out


@torch.no_grad()
def predict_from_resid(
    model: ICL,
    x_feat: torch.Tensor,
    yc_res: torch.Tensor,
    use_base: bool,
    base_q: torch.Tensor,
    n_ctx: int,
    cid: str,
    f_raw: int,
    d_s: int,
    seed: int,
    perms: int,
    k_bag: int,
    tag: str,
) -> torch.Tensor:
    w_tot = int(x_feat.shape[1])
    if w_tot <= WIDE:
        res_q = DG.predict(model, x_feat, yc_res, n_ctx, perms=perms, seed=seed)
    else:
        subs = subsets_keep_s(cid, f_raw, d_s, k_bag=k_bag, tag=tag)
        acc = torch.zeros(int(x_feat.shape[0] - n_ctx), dtype=torch.float64)
        for s_idx in subs:
            acc += DG.predict(model, x_feat[:, s_idx], yc_res, n_ctx) / len(subs)
        res_q = acc
    return res_q + base_q if use_base else res_q


@torch.no_grad()
def eval_fold_models(
    model_l: ICL,
    model_m: ICL,
    model_m20: ICL | None,
    x_raw: torch.Tensor,
    y_ctx_only: torch.Tensor,
    n_ctx: int,
    cid: str,
    seed0: int,
) -> dict[str, torch.Tensor]:
    """Computes `r4_l`, `r4_m`, `r4_e8` (and optional `cw5_m20`) strictly from `(x_raw[:n_ctx], y_ctx_only[:n_ctx], x_raw[n_ctx:])`."""
    f_raw = int(x_raw.shape[1])
    n_q = int(x_raw.shape[0] - n_ctx)
    y_pad = torch.cat([y_ctx_only[:n_ctx], torch.zeros(n_q, dtype=y_ctx_only.dtype)])
    xc_np = x_raw[:n_ctx].double().numpy()
    yc_np = y_ctx_only[:n_ctx].double().numpy()
    xq_np = x_raw[n_ctx:].double().numpy()

    S_c, S_q, _ = SC.build_struct_cols(xc_np, yc_np, xq_np, seed=seed0 + 17)
    d_s = int(S_c.shape[1])
    if d_s > 0:
        S_all = torch.from_numpy(np.vstack([S_c, S_q]).astype(np.float32))
        x_feat = torch.cat([x_raw.float(), S_all], dim=1)
    else:
        x_feat = x_raw.float()

    yr, used, bq = RES.residual_targets(x_feat[None], y_pad[None], n_ctx, torch.Generator().manual_seed(seed0))
    yc_res = yr[0, :n_ctx]
    use_base = bool(used[0])
    base_q = bq[0].cpu().double() if use_base else torch.zeros(n_q, dtype=torch.float64)

    # Note: when d_s == 0, CW-5 R4 falls back to perms=0 on narrow tables; for R4-e8 we ALWAYS use 4 members
    # (perms=4 when w_tot <= 30, else k_bag=4 subsets) so R4-e8 is a uniform 8-member (4x104M + 4x24M) ensemble.
    p_l = predict_from_resid(
        model_l, x_feat, yc_res, use_base, base_q, n_ctx, cid, f_raw, d_s,
        seed=seed0 + 11, perms=4, k_bag=4, tag="R4E8_L",
    )
    p_m = predict_from_resid(
        model_m, x_feat, yc_res, use_base, base_q, n_ctx, cid, f_raw, d_s,
        seed=seed0 + 23, perms=4, k_bag=4, tag="R4E8_M",
    )
    p_e8 = W_L * p_l + W_M * p_m
    out = {"r4_l": p_l, "r4_m": p_m, "r4_e8": p_e8}

    if model_m20 is not None:
        perms_m20 = (4 if (f_raw + d_s) <= WIDE else 0) if d_s > 0 else 0
        p_m20 = predict_from_resid(
            model_m20, x_feat, yc_res, use_base, base_q, n_ctx, cid, f_raw, d_s,
            seed=seed0 + 11, perms=perms_m20, k_bag=8, tag="CW5",
        )
        out["cw5_m20"] = p_m20
    return out


def columns_one(
    model_l: ICL,
    model_m: ICL,
    model_m20: ICL | None,
    S: torch.Tensor,
    lab: torch.Tensor,
    XA: torch.Tensor,
    cid: str,
    idx: int,
) -> dict[str, np.ndarray]:
    n, na = int(S.shape[0]), int(XA.shape[0])
    X, y = S[:, :-1], S[:, -1]
    keys = ("r4_l", "r4_m", "r4_e8") + (("cw5_m20",) if model_m20 is not None else ())
    q_map = {k: torch.zeros(n, dtype=torch.float64) for k in keys}
    bag_map = {k: torch.zeros(na, dtype=torch.float64) for k in keys}

    for fk in range(5):
        ci = torch.nonzero(lab != fk).squeeze(1)
        qi = torch.nonzero(lab == fk).squeeze(1)
        nq = int(qi.numel())
        x_cat = torch.cat([X[ci], X[qi], XA], dim=0)
        res = eval_fold_models(model_l, model_m, model_m20, x_cat, y[ci], int(ci.numel()), cid, SEED_BASE + 10 * idx + fk)
        for k in keys:
            q_map[k][qi] = res[k][:nq]
            bag_map[k] += res[k][nq:] / 5.0

    x_ref = torch.cat([X, XA], dim=0)
    ref_res = eval_fold_models(model_l, model_m, model_m20, x_ref, y, n, cid, SEED_BASE + 10 * idx + 9)

    out: dict[str, np.ndarray] = {}
    for k in keys:
        out[f"q_{k}"] = q_map[k].numpy()
        out[f"bag_{k}"] = bag_map[k].numpy()
        out[f"ref_{k}"] = ref_res[k].numpy()
    return out


def _try_claim(out: Path, cid: str, worker_tag: str) -> bool:
    claim = out / f".{cid}.claim"
    try:
        fd = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        return False
    try:
        os.write(fd, f"{worker_tag} pid={os.getpid()} t={time.time():.3f}\n".encode())
    finally:
        os.close(fd)
    return True


def _load_icl(ckpt_path: str, expected_sha: str, size: str, dev: torch.device) -> ICL:
    actual = sha256_file(Path(ckpt_path))
    if actual != expected_sha:
        raise SystemExit(f"ckpt sha256 mismatch for {ckpt_path}: {actual} != {expected_sha}")
    ck = torch.load(ckpt_path, map_location=dev, weights_only=False)
    m = ICL(size).to(dev)
    m.load_state_dict(ck["ema"] if "ema" in ck else ck["model"], strict=True)
    m.eval()
    return m


def cmd_run(a) -> None:
    if a.device == "cpu" and a.threads > 1:
        raise SystemExit("single-thread per CPU worker required (--threads 1)")
    t0, c0 = time.time(), time.process_time()
    dev = torch.device(a.device)
    torch.set_num_threads(a.threads)
    random.seed(0)
    torch.manual_seed(0)

    model_l = _load_icl(a.ckpt_l, a.sha_l, a.size_l, dev)
    model_m = _load_icl(a.ckpt_m, a.sha_m, a.size_m, dev)
    model_m20 = _load_icl(a.ckpt_m20, a.sha_m20, "M", dev) if a.ckpt_m20 else None

    with np.load(a.bank, allow_pickle=False) as z:
        ss, sa = z["s_shape"].astype(np.int64), z["a_shape"].astype(np.int64)
        cids = [str(c) for c in z["cids"].tolist()]
        s_off = np.concatenate([[0], np.cumsum(ss[:, 0] * ss[:, 1])])
        a_off = np.concatenate([[0], np.cumsum(sa[:, 0] * sa[:, 1])])
        l_off = np.concatenate([[0], np.cumsum(ss[:, 0])])
        cells = (ss[:, 0] + sa[:, 0]) * (ss[:, 1] - 1)
        order = sorted(range(len(cids)), key=lambda j: (int(cells[j]), cids[j]))
        if a.queue:
            mine = list(reversed(order))  # LPT: largest contexts claimed first
            worker_tag = f"q{a.worker_id}"
            shard_desc = f"queue/{a.worker_id}"
        else:
            k, m = (int(v) for v in a.shard.split("/"))
            mine = order[k::m]
            worker_tag = f"s{k}of{m}"
            shard_desc = a.shard
        search_all = z["search"]
        audit_all = z["audit"]
        lab_all = z["lab"]
        items = []
        for i in mine:
            (r, c), (ra, ca) = ss[i].tolist(), sa[i].tolist()
            items.append((
                i,
                cids[i],
                torch.from_numpy(np.array(search_all[s_off[i] : s_off[i + 1]], np.float32, copy=True).reshape(r, c)),
                torch.from_numpy(np.array(audit_all[a_off[i] : a_off[i + 1]], np.float32, copy=True).reshape(ra, ca)),
                torch.from_numpy(np.array(lab_all[l_off[i] : l_off[i + 1]], np.int64, copy=True)),
            ))

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGALRM, _alarm)
    done = skipped = failed = 0
    for i, cid, S, XA, lab in items:
        p = out / f"{cid}.npz"
        if p.exists():
            skipped += 1
            continue
        if time.time() - t0 > a.budget_s:
            break
        if a.queue and not _try_claim(out, cid, worker_tag):
            skipped += 1
            continue
        t1 = time.time()
        signal.alarm(a.ctx_s)
        completed_write = False
        try:
            cols = columns_one(model_l, model_m, model_m20, S, lab, XA, cid, i)
            ok = all(bool(np.isfinite(v).all()) for v in cols.values())
            err = None
        except _Timeout:
            cols, ok, err = {}, False, "timeout"
        except Exception as e:
            cols, ok, err = {}, False, f"{type(e).__name__}: {e}"
        finally:
            signal.alarm(0)
        wall = time.time() - t1
        tmp = out / f".{cid}.{worker_tag}.tmp.npz"
        np.savez(tmp, ok=np.array(ok), wall_s=np.array(wall), **cols)
        (out / f"{cid}.json").write_text(json.dumps({
            "cid": cid, "i": int(i), "ok": ok, "error": err, "wall_s": round(wall, 3),
        }, sort_keys=True))
        tmp.rename(p)
        completed_write = True
        if a.queue and not completed_write and not p.exists():
            (out / f".{cid}.claim").unlink(missing_ok=True)
        done += 1
        failed += not ok

    st = {
        "schema": "jope.h447.r4_e8_cols/v1",
        "shard": shard_desc,
        "contexts": len(items),
        "written": done,
        "already_there": skipped,
        "failed": int(failed),
        "bank_sha256": sha256_file(Path(a.bank)),
        "sha_l": a.sha_l,
        "sha_m": a.sha_m,
        "sha_m20": a.sha_m20 or None,
        "threads": a.threads,
        "device": a.device,
        "host": os.uname().nodename,
        "wall_s": round(time.time() - t0, 2),
        "cpu_s": round(time.process_time() - c0, 2),
    }
    (out / f"cols_status_{worker_tag}_{int(t0)}.json").write_text(json.dumps(st, sort_keys=True))
    print(json.dumps(st), flush=True)


def cmd_selftest(_a) -> None:
    torch.manual_seed(447)
    m_l = ICL("S").eval()
    torch.manual_seed(448)
    m_m = ICL("S").eval()

    # 1. Narrow + structural columns + strict query-target invariance
    rng = np.random.default_rng(447)
    n_s, n_a, f = 45, 15, 5
    X = rng.normal(size=(n_s + n_a, f)).astype(np.float32)
    X[:, 0] = (np.arange(n_s + n_a) % 5).astype(np.float32)  # exact-key candidate
    y_s = (3.0 * X[:n_s, 0] + 1.5 * X[:n_s, 1] + 0.1 * rng.normal(size=n_s)).astype(np.float32)
    S_mat = torch.from_numpy(np.column_stack([X[:n_s], y_s]))
    XA_mat = torch.from_numpy(X[n_s:])
    lab = torch.from_numpy(np.arange(n_s, dtype=np.int64) % 5)

    c1 = columns_one(m_l, m_m, m_m, S_mat, lab, XA_mat, "cid_narrow_01", 0)
    c2 = columns_one(m_l, m_m, m_m, S_mat, lab, XA_mat, "cid_narrow_01", 0)
    for k in c1:
        assert np.array_equal(c1[k], c2[k]), f"non-deterministic {k}"
        assert np.all(np.isfinite(c1[k])), f"non-finite {k}"
    assert np.allclose(c1["ref_r4_e8"], 0.60 * c1["ref_r4_l"] + 0.40 * c1["ref_r4_m"], atol=1e-12)

    # 2. Wide table (F=42 > 30): verify 4 distinct subsets per scale always keeping S
    subs_l = subsets_keep_s("cid_wide_01", 42, 4, k_bag=4, tag="R4E8_L")
    subs_m = subsets_keep_s("cid_wide_01", 42, 4, k_bag=4, tag="R4E8_M")
    assert len(subs_l) == 4 and len(subs_m) == 4
    for s_idx in subs_l + subs_m:
        assert int(s_idx.numel()) == 30
        assert s_idx[-4:].tolist() == [42, 43, 44, 45], "structural columns S not retained"
    assert not torch.equal(subs_l[0], subs_m[0]), "R4E8_L and R4E8_M subsets collided"

    # 3. End-to-end bank CLI smoke
    with tempfile.TemporaryDirectory(dir="/home/ubuntu/jope-data/h447") as td:
        td_p = Path(td)
        ck_l = td_p / "l.pt"
        ck_m = td_p / "m.pt"
        torch.save({"ema": m_l.state_dict(), "cfg": m_l.cfg}, ck_l)
        torch.save({"ema": m_m.state_dict(), "cfg": m_m.cfg}, ck_m)
        bank_p = td_p / "bank.npz"
        s_arr = S_mat.numpy().ravel()
        a_arr = XA_mat.numpy().ravel()
        np.savez(
            bank_p,
            cids=np.array(["cid_smoke_0001"]),
            cohort=np.array(["FIT-SEL"]),
            search=s_arr,
            audit=a_arr,
            y64=y_s.astype(np.float64),
            lab=lab.numpy(),
            s_shape=np.array([[n_s, f + 1]], np.int64),
            a_shape=np.array([[n_a, f]], np.int64),
            ya_sha256=np.array(["0" * 64]),
        )
        ns = argparse.Namespace(
            bank=str(bank_p),
            ckpt_l=str(ck_l),
            sha_l=sha256_file(ck_l),
            size_l="S",
            ckpt_m=str(ck_m),
            sha_m=sha256_file(ck_m),
            size_m="S",
            ckpt_m20="",
            sha_m20="",
            out=str(td_p / "cols_out"),
            device="cpu",
            threads=1,
            shard="0/1",
            queue=False,
            worker_id=0,
            budget_s=600.0,
            ctx_s=120,
        )
        cmd_run(ns)
        z = np.load(td_p / "cols_out" / "cid_smoke_0001.npz", allow_pickle=False)
        assert bool(z["ok"]) and z["ref_r4_e8"].shape == (n_a,)

    print(json.dumps({"all_passed": True, "tests": 3}), flush=True)
    print("H447_COLS_R4E8_SELFTEST_OK", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest").set_defaults(fn=cmd_selftest)

    p = sub.add_parser("run")
    p.add_argument("--bank", required=True)
    p.add_argument("--ckpt-l", required=True)
    p.add_argument("--sha-l", required=True)
    p.add_argument("--size-l", default="L", choices=("S", "M", "L"))
    p.add_argument("--ckpt-m", required=True)
    p.add_argument("--sha-m", required=True)
    p.add_argument("--size-m", default="M", choices=("S", "M", "L"))
    p.add_argument("--ckpt-m20", default="")
    p.add_argument("--sha-m20", default="")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--shard", default="0/1")
    p.add_argument("--queue", action="store_true")
    p.add_argument("--worker-id", type=int, default=0)
    p.add_argument("--budget-s", type=float, default=1e12)
    p.add_argument("--ctx-s", type=int, default=7200)
    p.set_defaults(fn=cmd_run)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
