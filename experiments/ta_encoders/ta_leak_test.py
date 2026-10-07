#!/usr/bin/env python3
"""Leak tests for the TA data path (run before training; results -> leak_test.json).

T1 counterfactual test block: for K admitted s0 payloads, append a synthetic outer-test block (random rows AND random targets)
    and fix the header's n_test; ta_common.train_view(require_stripped=False) must return byte-identical X/y, and prep_context
    on the stripped original must equal the stored record (same train_rows_sha256).
T2 must-fail control: a deliberately wrong decoder that reads past the train range must see the synthetic test rows (differs).
T3 refusal: train_view(require_stripped=True) must refuse any payload with n_test != 0.
T4 query isolation (model level): changing ONLY query-row targets leaves every model output for the context (context
    embedding, support rows, and query predictions) bit-identical, for both TA_Hyperion_v8 and TA_Dope_V2.
"""
import gzip, json, os, struct, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ta_common as C


def main():
    data, out, K = sys.argv[1], sys.argv[2], 50
    idx = json.load(open(os.path.join(data, "index.json")))["contexts"]
    rng = np.random.default_rng(0); res = {"T1_identical": 0, "T2_wrong_decoder_differs": 0, "T3_refused": 0, "checked": 0}
    for r in [idx[i] for i in rng.choice(len(idx), K, replace=False)]:
        raw = gzip.decompress(open(os.path.join(data, "jtt1", r["gz_sha256"] + ".jtt1.gz"), "rb").read())
        n, nt, p, inner = C.jtt1_header(raw); assert nt == 0
        T = rng.normal(size=(17, p + 1)) * 1e3
        fake = b"JTT1" + C.HEADER.pack(n, 17, p, inner) + bytes(raw[C.HB:]) + T.astype("<f8").tobytes()
        X0, y0, _ = C.train_view(raw); X1, y1, _ = C.train_view(fake, require_stripped=False)
        res["T1_identical"] += int(np.array_equal(X0, X1) and np.array_equal(y0, y1))
        allblk = np.frombuffer(memoryview(fake)[C.HB + n:], dtype="<f8").reshape(n + 17, p + 1)
        res["T2_wrong_decoder_differs"] += int(allblk.shape[0] != X0.shape[0] or not np.array_equal(allblk[:, -1][-17:], y0[-17:]))
        try:
            C.train_view(fake); 
        except ValueError:
            res["T3_refused"] += 1
        _, rec = C.prep_context(raw, r["cid"]); assert rec["train_rows_sha256"] == r["train_rows_sha256"]
        res["checked"] += 1
    # T4 model-level query isolation
    import torch
    torch.manual_seed(0); dev = "cuda" if torch.cuda.is_available() else "cpu"
    from ta_hyperion_v8 import TAHyperionV8
    from ta_dope_v2 import TADopeV2
    B, N, P = 3, 96, 20
    sup = torch.ones(B, N, dtype=torch.bool, device=dev); sup[:, -24:] = False; valid = torch.ones_like(sup)
    y = torch.randn(B, N, device=dev); y2 = y.clone(); y2[:, -24:] = torch.randn(B, 24, device=dev) * 50
    ok = {}
    m = TAHyperionV8().to(dev).eval(); R = torch.rand(B, N, 512, device=dev)
    with torch.no_grad():
        a = m(R, y, sup, valid); b = m(R, y2, sup, valid)   # raw y: the model itself must ignore query targets
        ok["hv8_query_isolated"] = all(torch.equal(u, v) for u, v in zip(a, b))
        c = m(R, torch.randn_like(y), sup, valid)
        ok["hv8_support_y_changes_output(must-differ)"] = not torch.equal(a[1], c[1])
    md = TADopeV2().to(dev).eval(); X = torch.rand(B, N, P, device=dev); fv = torch.ones(B, P, dtype=torch.bool, device=dev); miss = torch.zeros(B, P, device=dev); ys = torch.zeros(B, 8, device=dev)
    with torch.no_grad():
        a = md(X, y, sup, valid, fv, miss, ys); b = md(X, y2, sup, valid, fv, miss, ys)
        ok["dv2_query_isolated"] = all(torch.equal(u, v) for u, v in zip(a, b))
        c = md(X, torch.randn_like(y), sup, valid, fv, miss, ys)
        ok["dv2_support_y_changes_output(must-differ)"] = not torch.equal(a[1], c[1])
    res.update(ok)
    res["pass"] = (res["T1_identical"] == res["checked"] == res["T2_wrong_decoder_differs"] == res["T3_refused"] == K) and all(ok.values())
    json.dump(res, open(out, "w"), indent=1); print(json.dumps(res))
    sys.exit(0 if res["pass"] else 1)


if __name__ == "__main__":
    main()
