#!/usr/bin/env python3
"""Assemble the DL_FINAL training bundle (labelled one-holdout GP-HWM views x clean features), purge-before-split.
Inputs (all read-only): views_labeled.json (one-holdout target after exclusion layers v4, selection_reached, denylist v5 and sealed
contexts removed), the role map (sealed families), GP-HWM evidence objects (S3 GET by VersionId, sha256-verified), clean encodings
(S3 main tree, key = encodings/<type>/<stem> where stem = <dataset>/f<fold>/<code12 stem>; refused object keys skipped), LV4 stats JSONs (stats_v2.1 seeded / v3 / v4, train rows only).
Outputs DIR/: blocks_<name>.npy (float32), targets.npz, meta.json, folds.npy, MANIFEST.json. GP labels go only to targets.npz.
Lineage groups: union-find over wave-10 lineage components, dataset_hash, family_id, catalog display name, and the wave-10
pre-registered stats_v2.1 fingerprint rule (tau = 0.5 * median cross-group NN RMS distance). Folds = group_folds(comp, 5, 10100+r)."""
import argparse, collections as Co, concurrent.futures as cf, io, json, os, re, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dlhwm.common import PFX, CLEAN_ENCODERS, get, log, sha256b, sha256f
ENC = list(CLEAN_ENCODERS)
if ENC != ["kumo_tabular_l", "kumo_tabular_s", "mitra_v2", "tabdpt13", "tabicl2", "a0", "dope", "hyperion_v6all"]: raise SystemExit("clean encoder list changed: review build_data")
Z_FROM = ["kumo_tabular_l", "mitra_v2", "tabdpt13", "tabicl2"]
LMC = ["lm_ridge", "lm_knn10", "lm_1nn", "lm_stump", "lm_tree3", "lm_tree6", "lm_best"]
def group_folds(groups, k, seed):   # identical to w4_lib.group_folds / w10_build
    rng = np.random.default_rng(seed); ug, cnt = np.unique(groups, return_counts=True); order = rng.permutation(len(ug))
    order = order[np.argsort(-cnt[order], kind="stable")]; load_ = np.zeros(k); gf = {}
    for gi in order:
        f = int(np.argmin(load_ + rng.random(k) * 1e-6)); gf[ug[gi]] = f; load_[f] += cnt[gi]
    return np.array([gf[g] for g in groups])
def fetch_type(t, stems, refused, threads):
    """All views for one encoder type (own process: boto3 is GIL-bound). stem already = <dataset>/f<fold>/<code12 stem>."""
    import dlhwm.common as C; C._S3 = None   # boto3 clients are not fork-safe: fresh client per process
    ext = ".f32" if t == "a0" else ".npy"
    def one(i):
        key = f"{PFX}encodings/{t}/{stems[i]}{ext}"
        if key in refused: return i, None, "refused"
        try: b = get(key)
        except Exception: return i, None, "missing"
        x = np.frombuffer(b, "<f4") if ext == ".f32" else np.load(io.BytesIO(b)); return i, np.asarray(x, np.float32).ravel(), sha256b(b)
    got, hh, mm = {}, {}, Co.Counter()
    with cf.ThreadPoolExecutor(threads) as ex:
        for i, x, h in ex.map(one, range(len(stems))):
            if x is None: mm[h] += 1
            else: got[i] = x; hh[i] = h
    return got, hh, dict(mm)
def walk(o, acc):
    if isinstance(o, dict):
        for k, v in o.items():
            if k == "context_sha256" and isinstance(v, str): acc["ctx"].add(v)
            if k == "family_id" and isinstance(v, str): acc["fam"].add(v)
            if k == "dataset_hash" and isinstance(v, str): acc["ds"].add(v)
            walk(v, acc)
    elif isinstance(o, list):
        for v in o: walk(v, acc)
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", required=True); ap.add_argument("--rolemap", required=True); ap.add_argument("--refused-keys", required=True)
    ap.add_argument("--stats-dirs", required=True, help="comma list of dirs each holding v21/ v3r/ v4r/ subdirs of <cid>.json")
    ap.add_argument("--cols", required=True, help="wave-10 cols.json (V21/V3R/V4R column order)")
    ap.add_argument("--w10-meta", required=True); ap.add_argument("--catalog", required=True); ap.add_argument("--tabdpt-overlap", required=True)
    ap.add_argument("--plan", default="", help="plan_full.jsonl (context_sha256 -> dataset_hash) so sealed/heldout cids also purge their whole dataset")
    ap.add_argument("--evidence-cache", required=True); ap.add_argument("--out", required=True); ap.add_argument("--threads", type=int, default=32)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    V = json.load(open(a.views)); refused = set(json.load(open(a.refused_keys)))
    rm = json.load(open(a.rolemap)); sealed = {"ctx": set(rm), "fam": set(), "ds": set()}   # meta/exclude_sealed.json: {cid: role}; every listed cid is out
    if isinstance(rm, dict) and "sealed" in rm: walk(rm["sealed"], sealed)
    hard = {c for c, role in (rm.items() if isinstance(rm, dict) else []) if isinstance(role, str) and role in ("sealed_excluded", "heldout", "sealed", "lockbox")}
    if hard and not a.plan: raise SystemExit(f"{len(hard)} sealed/heldout cids in the role map: --plan is required so their datasets are purged too")
    if a.plan:
        for l in open(a.plan):
            x = json.loads(l)
            if x.get("context_sha256") in hard and x.get("dataset_hash"): sealed["ds"].add(x["dataset_hash"])
    log("views", len(V), "sealed ctx/fam/ds", {k: len(v) for k, v in sealed.items()}, "hard-role cids", len(hard))
    # 1) evidence
    ev = {}
    want = {v["context_sha256"]: v["sha"] for v in V}
    if os.path.exists(a.evidence_cache):   # cached lines carry the sha256 of the verified S3 bytes; reused only if it matches this views file
        for l in open(a.evidence_cache):
            d = json.loads(l); c = d.get("context_sha256")
            if c in want and d.get("_evidence_sha256") == want[c]: ev[c] = d
    def get_ev(v):
        b = get(v["key"], v["vid"])
        if sha256b(b) != v["sha"]: raise RuntimeError(f"evidence sha256 mismatch {v['key']}")
        d = json.loads(b); d["_evidence_sha256"] = v["sha"]; return d
    pending_views = [v for v in V if v["context_sha256"] not in ev]
    with cf.ThreadPoolExecutor(a.threads) as ex, open(a.evidence_cache, "a") as fo:
        for i, d in enumerate(ex.map(get_ev, pending_views)):
            fo.write(json.dumps(d) + "\n"); ev[d["context_sha256"]] = d
            if i % 2000 == 0: log("evidence", i, len(pending_views))
    keep, why = [], Co.Counter()
    for v in V:
        d = ev[v["context_sha256"]]; fam = (d.get("family") or {}).get("family_id") or ("ds-" + d["dataset_hash"])
        sel = d.get("selection") or {}
        if d["context_sha256"] in sealed["ctx"] or d["dataset_hash"] in sealed["ds"] or fam in sealed["fam"]: why["sealed"] += 1; continue
        if sel.get("fitness") is None or not sel.get("cv_geomean_rmse_ratio"): why["no_fitness"] += 1; continue
        if d.get("selection_protocol") != "jope.s1-two-fold/v1": why["protocol"] += 1; continue
        v = dict(v, family_id=fam, fitness=float(sel["fitness"]), lr_gp=float(np.log(sel["cv_geomean_rmse_ratio"])),
                 program_sha256=sel["program"]["program_sha256"], program_dsl=sel["program"]["dsl"]); keep.append(v)
    log("kept", len(keep), dict(why)); V = keep; n = len(V); cids = [v["context_sha256"] for v in V]
    # 2) encodings
    E, ehash, miss = {}, {}, Co.Counter()
    with cf.ProcessPoolExecutor(len(ENC), mp_context=__import__("multiprocessing").get_context("fork")) as px:
        futs = {t: px.submit(fetch_type, t, [v["stem"] for v in V], refused, max(4, a.threads // 2)) for t in ENC}
        for t, fu in futs.items():
            got, hh, mm = fu.result(); E[t] = got; ehash[t] = {cids[i]: h for i, h in hh.items()}
            for k_, c_ in mm.items(): miss[(t, k_)] += c_
            log("enc", t, "have", len(got), "missing", dict(mm))
    dims = {}
    for t in ENC:
        dm = Co.Counter(len(x) for x in E[t].values()).most_common(1)[0][0]; A = np.full((n, dm), np.nan, np.float32)
        for i, x in E[t].items():
            if len(x) == dm: A[i] = x
        A[~np.isfinite(A)] = np.nan; np.save(f"{a.out}/blocks_{t}.npy", A); dims[t] = dm
    Zs = np.concatenate([np.load(f"{a.out}/blocks_{t}.npy")[:, -3:] for t in Z_FROM], 1)
    # 3) stats (LV4)
    cols = json.load(open(a.cols)); dirs = a.stats_dirs.split(",")
    def load_stats(sub, cl, vblock):
        A = np.full((n, len(cl)), np.nan); found = 0
        for i, c in enumerate(cids):
            r = None
            for d in dirs:
                p = f"{d}/{sub}/{c}.json"
                if os.path.exists(p): r = json.load(open(p)); break
            if r is None: continue
            found += 1
            if vblock and r.get("degenerate") != 0.0: continue
            A[i] = [float(r[k]) if isinstance(r.get(k), (int, float)) and not isinstance(r.get(k), bool) else np.nan for k in cl]
        A[~np.isfinite(A)] = np.nan; return A, found
    V21, f1 = load_stats("v21", cols["V21"], True); V3, f3 = load_stats("v3r", cols["V3R"], False); V4, f4 = load_stats("v4r", cols["V4R"], False)
    log("stats found v21/v3/v4", f1, f3, f4, "of", n)
    Lg = lambda r: 0.5 * np.log(np.clip(1 - r, 1e-8, None)); VC = cols["V21"]
    TL = np.column_stack([Lg(V21[:, VC.index(c)]) for c in LMC] + [np.log(np.clip(V21[:, VC.index("noise_delta_test")], 1e-8, None))])
    ZL = np.column_stack([Lg(np.clip(Zs[:, j], -10, 1)) for j in (0, 3, 6, 9)])
    L = np.column_stack([V21, TL, ZL, V3, V4, Zs]).astype(np.float32)
    Lcols = [f"V21:{c}" for c in VC] + [f"TL:{c}" for c in LMC + ["log_noise_delta_test"]] + [f"ZL:{j}" for j in range(4)] + [f"V3:{c}" for c in cols["V3R"]] + [f"V4:{c}" for c in cols["V4R"]] + [f"Z:{t}:{s}" for t in Z_FROM for s in ("qr2", "spread", "logspread")]
    np.save(f"{a.out}/blocks_L.npy", L)
    # 4) targets (wave-10 v2.1 definitions)
    lr_gp = np.array([v["lr_gp"] for v in V]); fit = np.array([v["fitness"] for v in V])
    cheap = np.min(np.column_stack([Lg(V21[:, VC.index(c)]) for c in LMC[:6]]), 1); gap = lr_gp - cheap
    c4 = cols["V4R"]; ch_sh = np.min(np.column_stack([V4[:, c4.index(c)] for c in ("v4_s_et", "v4_s_hgb")]), 1); ch_sh = np.where(np.isfinite(ch_sh), ch_sh, cheap)
    beat10 = np.where(np.isfinite(gap), (gap < np.log(0.9)).astype(float), np.nan); beat10sh = np.where(np.isfinite(lr_gp - ch_sh), ((lr_gp - ch_sh) < np.log(0.9)).astype(float), np.nan)
    np.savez(f"{a.out}/targets.npz", fitness=fit, lr_gp=lr_gp, cheap=cheap, gap=gap, beat10=beat10, beat10sh=beat10sh)
    # 5) lineage
    ds = [v["dataset_hash"] for v in V]; fam = [v["family_id"] for v in V]
    cat = {}
    for l in open(a.catalog):
        x = json.loads(l); cat[x["id"]] = x
    par = list(range(n))
    def fnd(x):
        while par[x] != x: par[x] = par[par[x]]; x = par[x]
        return x
    def uni(i, j): par[fnd(i)] = fnd(j)
    for key in (ds, fam, [re.sub(r"[^a-z]", "", (cat.get(d, {}).get("display_name") or "").lower()) or None for d in ds]):
        first = {}
        for i, k in enumerate(key):
            if k is None: continue
            if k in first: uni(i, first[k])
            else: first[k] = i
    wm = json.load(open(a.w10_meta)); wcomp = dict(zip(wm["cids"], wm["lineage_comp"])); first = {}
    for i, c in enumerate(cids):
        if c in wcomp:
            k = wcomp[c]
            if k in first: uni(i, first[k])
            else: first[k] = i
    pre = np.array([fnd(i) for i in range(n)])
    FC = [j for j, c in enumerate(VC) if c != "degenerate"]; A = V21[:, FC]
    med = np.nanmedian(A, 0); mad = 1.4826 * np.nanmedian(np.abs(A - med), 0); sd = np.nanstd(A, 0); sc = np.where(mad > 1e-9, mad, np.where(sd > 1e-9, sd, np.inf))
    F = np.clip((A - med) / sc, -10, 10); ok = np.isfinite(V21[:, FC]).any(1); F[~np.isfinite(F)] = 0.0; F = F.astype(np.float32)
    sq = (F ** 2).sum(1); nnx = np.full(n, np.inf); E2 = Co.Counter()
    for s in range(0, n, 512):
        Dm = np.sqrt(np.maximum((sq[s:s + 512, None] + sq[None] - 2 * F[s:s + 512] @ F.T) / F.shape[1], 0)); Dm[pre[s:s + 512, None] == pre[None]] = np.inf
        Dm[~ok[s:s + 512]] = np.inf; Dm[:, ~ok] = np.inf; nnx[s:s + 512] = Dm.min(1)
    tau = 0.5 * float(np.median(nnx[np.isfinite(nnx)]))
    for s in range(0, n, 512):
        Dm = np.sqrt(np.maximum((sq[s:s + 512, None] + sq[None] - 2 * F[s:s + 512] @ F.T) / F.shape[1], 0)); Dm[pre[s:s + 512, None] == pre[None]] = np.inf
        Dm[~ok[s:s + 512]] = np.inf; Dm[:, ~ok] = np.inf
        for x, y in zip(*np.where(Dm < tau)):
            i, j = s + x, y
            if i < j: E2[(min(pre[i], pre[j]), max(pre[i], pre[j]))] += 1
    gsz = Co.Counter(pre.tolist()); nl = 0
    for (g1, g2), cnt in E2.items():
        if cnt >= 2 or min(gsz[g1], gsz[g2]) < 3: uni(g1, g2); nl += 1
    comp = np.unique(np.array([fnd(i) for i in range(n)]), return_inverse=True)[1]
    FO = np.stack([group_folds(list(comp), 5, 10100 + r) for r in range(3)]); np.save(f"{a.out}/folds.npy", FO); np.save(f"{a.out}/gid.npy", comp)
    tdo = set(json.load(open(a.tabdpt_overlap)).keys()); tab_ov = np.array([d in tdo for d in ds])
    meta = {"schema": "dlhwm.data/v1", "n": n, "cids": cids, "view_id": [v["view_id"] for v in V], "dataset": ds, "fold": [v["fold"] for v in V], "family": fam,
            "universe": [v["universe"] for v in V], "rights": [v["rights"] for v in V], "program_sha256": [v["program_sha256"] for v in V],
            "tabdpt_overlap": tab_ov.tolist(), "L_cols": Lcols, "enc_dims": dims, "blocks": ["L"] + ENC, "lineage": {"tau": tau, "fp_links": nl, "n_components": int(comp.max() + 1),
            "largest": int(np.bincount(comp).max())}, "fold_sizes": {f"r{r}": np.bincount(FO[r]).tolist() for r in range(3)}, "drop_reasons": dict(why),
            "enc_missing": {f"{t}:{h}": c for (t, h), c in miss.items()}, "stats_found": [f1, f3, f4],
            "rates": {"beat10": float(np.nanmean(beat10)), "beat10sh": float(np.nanmean(beat10sh))}}
    json.dump(meta, open(f"{a.out}/meta.json", "w")); json.dump(ehash, open(f"{a.out}/encoding_sha256.json", "w"))
    json.dump({"programs": [{"cid": v["context_sha256"], "program_sha256": v["program_sha256"], "dsl": v["program_dsl"]} for v in V]}, open(f"{a.out}/hwm_programs.json", "w"))
    man = {f: sha256f(f"{a.out}/{f}") for f in sorted(os.listdir(a.out)) if f != "MANIFEST.json"}
    json.dump({"files": man}, open(f"{a.out}/MANIFEST.json", "w"), indent=1)
    log("DONE n", n, "components", int(comp.max() + 1), "tau", round(tau, 4), "dims", dims, "rates", meta["rates"])
if __name__ == "__main__": main()
