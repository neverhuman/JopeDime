#!/usr/bin/env python3
"""Conditional CW-5 5-fold stack column builder (`cols_cw5_dev.py`) in H424/H430 bank format.

Invoked IF AND ONLY IF `H442` `FROZEN.json` has `promote_to_joint_dev == true`:
  cols_cw5_dev.py --bank FILE --ckpt CKPT --ckpt-sha256 HEX --size M --frozen FILE --out DIR
                  [--threads 1] [--shard k/n | --queue --worker-id W] [--budget-s S] [--ctx-s 7200]

Emits create-once `OUT/<cid>.npz` with `q_cw5`, `bag_cw5`, `ref_cw5` using ONLY `SEARCH` targets
from the context folds (`AUDIT` targets are absent from `bank_DEV.npz`, query targets passed as zeros).
In `--queue` mode, workers iterate in Largest-Processing-Time-First (`LPT`, `reversed(order)`) order
and claim each context atomically via `O_CREAT | O_EXCL` on `OUT/.<cid>.claim`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import eval_cw5_fitsel as CW5  # noqa: E402
import icl_resid as RES  # noqa: E402
import struct_cols as SC  # noqa: E402
from icl_model import ICL  # noqa: E402

SEED = 445_700


class _Timeout(Exception):
    pass


def _alarm(_s, _f):
    raise _Timeout()


@torch.no_grad()
def eval_single_cw5_recipe(
    model,
    recipe: str,
    x_raw: torch.Tensor,
    y_full: torch.Tensor,
    n_ctx: int,
    cid: str,
    seed0: int,
) -> torch.Tensor:
    """Evaluates ONLY the single frozen CW-5 recipe (`R1..R5`) strictly on `(x_raw[:n_ctx], y_full[:n_ctx], x_raw[n_ctx:])`."""
    f_raw = int(x_raw.shape[1])
    n_q = int(x_raw.shape[0] - n_ctx)
    y_pad = torch.cat([y_full[:n_ctx], torch.zeros(n_q, dtype=y_full.dtype)])
    xc_np = x_raw[:n_ctx].double().numpy()
    yc_np = y_full[:n_ctx].double().numpy()
    xq_np = x_raw[n_ctx:].double().numpy()

    S_c, S_q, _ = SC.build_struct_cols(xc_np, yc_np, xq_np, seed=seed0 + 17)
    d_s = int(S_c.shape[1])
    if d_s == 0:
        return CW5._icl_residual_pass(model, x_raw, x_raw, y_pad, n_ctx, cid, f_raw, 0, seed0, perms=0)

    S_all = torch.from_numpy(np.vstack([S_c, S_q]).astype(np.float32))
    x_xs = torch.cat([x_raw.float(), S_all], dim=1)

    if recipe == "R1":
        return CW5._icl_residual_pass(model, x_raw, x_xs, y_pad, n_ctx, cid, f_raw, d_s, seed0, perms=0)
    if recipe == "R2":
        return CW5._icl_residual_pass(model, x_xs, x_xs, y_pad, n_ctx, cid, f_raw, d_s, seed0, perms=0)
    if recipe == "R4":
        perms = 4 if (f_raw + d_s) <= CW5.WIDE else 0
        return CW5._icl_residual_pass(model, x_xs, x_xs, y_pad, n_ctx, cid, f_raw, d_s, seed0, perms=perms)
    if recipe == "R5":
        oof_x, _ = RES.ridge_base(x_raw[:n_ctx][None], y_full[:n_ctx][None], n_ctx, torch.Generator().manual_seed(seed0 + 7))
        oof_xs, _ = RES.ridge_base(x_xs[:n_ctx][None], y_full[:n_ctx][None], n_ctx, torch.Generator().manual_seed(seed0 + 7))
        rm_x = float(torch.sqrt(((y_full[:n_ctx].double() - oof_x[0]) ** 2).mean()))
        rm_xs = float(torch.sqrt(((y_full[:n_ctx].double() - oof_xs[0]) ** 2).mean()))
        if rm_xs <= 0.995 * rm_x:
            return CW5._icl_residual_pass(model, x_xs, x_xs, y_pad, n_ctx, cid, f_raw, d_s, seed0, perms=0)
        return CW5._icl_residual_pass(model, x_raw, x_raw, y_pad, n_ctx, cid, f_raw, 0, seed0, perms=0)
    if recipe == "R3":
        p_r2 = CW5._icl_residual_pass(model, x_xs, x_xs, y_pad, n_ctx, cid, f_raw, d_s, seed0, perms=0)
        g_in = np.random.default_rng(seed0 + 1000)
        in_fold = g_in.permutation(n_ctx) % 4
        p1_oof = torch.zeros(n_ctx, dtype=torch.float64)
        p2_oof = torch.zeros(n_ctx, dtype=torch.float64)
        xc_t, yc_t = x_xs[:n_ctx], y_full[:n_ctx]
        for fk in range(4):
            ci = torch.from_numpy(np.nonzero(in_fold != fk)[0])
            vi = torch.from_numpy(np.nonzero(in_fold == fk)[0])
            sub_x = torch.cat([xc_t[ci], xc_t[vi]], dim=0)
            sub_y = torch.cat([yc_t[ci], torch.zeros(len(vi), dtype=yc_t.dtype)])
            p1_oof[vi] = CW5._icl_residual_pass(model, sub_x, sub_x, sub_y, int(len(ci)), cid, f_raw, d_s, seed0 + 30 + fk, perms=0)
        r1_c = (yc_t.double() - p1_oof).float()
        if float(r1_c.std()) <= 1e-4 * max(float(yc_t.double().std()), 1e-9):
            return p_r2
        x_aug_c = torch.cat([xc_t, p1_oof.float()[:, None]], dim=1)
        for fk in range(4):
            ci = torch.from_numpy(np.nonzero(in_fold != fk)[0])
            vi = torch.from_numpy(np.nonzero(in_fold == fk)[0])
            sub_xa = torch.cat([x_aug_c[ci], x_aug_c[vi]], dim=0)
            p2_oof[vi] = CW5._icl_residual_pass(
                model, sub_xa, sub_xa, torch.cat([r1_c[ci], torch.zeros(len(vi), dtype=r1_c.dtype)]),
                int(len(ci)), cid, f_raw, d_s + 1, seed0 + 60 + fk, perms=0,
            )
        yc_d = yc_t.double()
        best_eta, best_mse = 0.0, float(((yc_d - p1_oof) ** 2).mean())
        for eta in (0.35, 0.70):
            m_e = float(((yc_d - (p1_oof + eta * p2_oof)) ** 2).mean())
            if m_e < 0.995 * best_mse:
                best_mse = m_e
                best_eta = eta
        if best_eta <= 0.0:
            return p_r2
        x_aug_all = torch.cat([x_xs, torch.cat([p1_oof, p_r2]).float()[:, None]], dim=1)
        r2_q = CW5._icl_residual_pass(
            model, x_aug_all, x_aug_all, torch.cat([r1_c, torch.zeros(n_q, dtype=r1_c.dtype)]),
            n_ctx, cid, f_raw, d_s + 1, seed0 + 99, perms=0,
        )
        return p_r2 + best_eta * r2_q
    raise ValueError(f"unsupported CW-5 recipe: {recipe}")


def columns_one(model, recipe: str, S: torch.Tensor, lab: torch.Tensor, XA: torch.Tensor, cid: str, idx: int) -> dict[str, np.ndarray]:
    n, na = S.shape[0], XA.shape[0]
    X, y = S[:, :-1], S[:, -1]
    q = torch.zeros(n, dtype=torch.float64)
    bag = torch.zeros(na, dtype=torch.float64)
    zeros = lambda m: torch.zeros(m, dtype=y.dtype)
    for k in range(5):
        ci = torch.nonzero(lab != k).squeeze(1)
        qi = torch.nonzero(lab == k).squeeze(1)
        x_cat = torch.cat([X[ci], X[qi], XA], dim=0)
        y_cat = torch.cat([y[ci], zeros(int(qi.numel()) + na)], dim=0)
        p = eval_single_cw5_recipe(model, recipe, x_cat, y_cat, int(ci.numel()), cid, SEED + 10 * idx + k)
        q[qi] = p[: int(qi.numel())]
        bag += p[int(qi.numel()) :] / 5.0
    x_ref = torch.cat([X, XA], dim=0)
    y_ref = torch.cat([y, zeros(na)], dim=0)
    ref = eval_single_cw5_recipe(model, recipe, x_ref, y_ref, n, cid, SEED + 10 * idx + 9)
    return {"q_cw5": q.numpy(), "bag_cw5": bag.numpy(), "ref_cw5": ref.numpy()}


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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bank", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ckpt-sha256", required=True)
    ap.add_argument("--size", default="M")
    ap.add_argument("--frozen", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--queue", action="store_true", help="LPT dynamic atomic work-queue across all contexts")
    ap.add_argument("--worker-id", type=int, default=0)
    ap.add_argument("--budget-s", type=float, default=1e12)
    ap.add_argument("--ctx-s", type=int, default=7200)
    a = ap.parse_args()
    if a.threads > 1:
        raise SystemExit("single-thread per process required")

    frz = json.loads(Path(a.frozen).read_text())
    if not frz.get("promote_to_joint_dev") or frz.get("selected_recipe") not in ("R1", "R2", "R3", "R4", "R5"):
        raise SystemExit("FROZEN.json did not promote a CW-5 recipe; reuse stored H424 cols_DEV instead")
    recipe = str(frz["selected_recipe"])

    t0, c0 = time.time(), time.process_time()
    torch.set_num_threads(1)
    random.seed(0)
    torch.manual_seed(0)
    if hashlib.sha256(Path(a.ckpt).read_bytes()).hexdigest() != a.ckpt_sha256:
        raise SystemExit("ckpt sha256 mismatch")
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    model = ICL(a.size)
    model.load_state_dict(ck["ema"])
    model.eval()

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
            cols = columns_one(model, recipe, S, lab, XA, cid, i)
            ok = all(bool(np.isfinite(v).all()) for v in cols.values())
            err = None
            wall = time.time() - t1
            tmp = out / f".{cid}.{worker_tag}.tmp.npz"
            np.savez(tmp, ok=np.array(ok), wall_s=np.array(wall), recipe=np.array(recipe), **cols)
            (out / f"{cid}.json").write_text(json.dumps({
                "cid": cid, "i": int(i), "recipe": recipe, "ok": ok, "error": err, "wall_s": round(wall, 3),
            }, sort_keys=True))
            tmp.rename(p)
            completed_write = True
        except _Timeout:
            cols, ok, err = {}, False, "timeout"
            wall = time.time() - t1
            tmp = out / f".{cid}.{worker_tag}.tmp.npz"
            np.savez(tmp, ok=np.array(ok), wall_s=np.array(wall), recipe=np.array(recipe), **cols)
            (out / f"{cid}.json").write_text(json.dumps({
                "cid": cid, "i": int(i), "recipe": recipe, "ok": ok, "error": err, "wall_s": round(wall, 3),
            }, sort_keys=True))
            tmp.rename(p)
            completed_write = True
        except Exception as e:
            cols, ok, err = {}, False, f"{type(e).__name__}: {e}"
            wall = time.time() - t1
            tmp = out / f".{cid}.{worker_tag}.tmp.npz"
            np.savez(tmp, ok=np.array(ok), wall_s=np.array(wall), recipe=np.array(recipe), **cols)
            (out / f"{cid}.json").write_text(json.dumps({
                "cid": cid, "i": int(i), "recipe": recipe, "ok": ok, "error": err, "wall_s": round(wall, 3),
            }, sort_keys=True))
            tmp.rename(p)
            completed_write = True
        finally:
            signal.alarm(0)
            if a.queue and not completed_write and not p.exists():
                (out / f".{cid}.claim").unlink(missing_ok=True)
        done += 1
        failed += not ok

    st = {
        "schema": "jope.cw5.dev_cols/v1",
        "shard": shard_desc,
        "recipe": recipe,
        "contexts": len(items),
        "written": done,
        "already_there": skipped,
        "failed": int(failed),
        "bank_sha256": hashlib.sha256(Path(a.bank).read_bytes()).hexdigest(),
        "ckpt_sha256": a.ckpt_sha256,
        "threads": 1,
        "device": "cpu",
        "cuda_visible": bool(torch.cuda.is_available()),
        "wall_s": round(time.time() - t0, 2),
        "cpu_s": round(time.process_time() - c0, 2),
    }
    (out / f"cols_status_{worker_tag}_{int(t0)}.json").write_text(json.dumps(st, sort_keys=True))
    print(json.dumps(st), flush=True)


if __name__ == "__main__":
    main()
