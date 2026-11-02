"""Canonical H46/H47 token construction."""

def make_tokens(net, tables, device, base, torch, np):
    tokens, masks = {}, {}
    ws = base.FusedLandmarkWorkspace(num_threads=4)
    for i, (cid, (x, y)) in enumerate(tables.items(), 1):
        if i % 16 == 0:
            base.headroom()
        yd = y.astype(np.float64)
        yn = ((yd - yd.mean()) / max(float(yd.std()), 1e-8)).astype(np.float32)
        vec, cols, _ = ws.fit_transform_and_extract_bundle_high(
            base.landmark_float32(x), np.ascontiguousarray(yn)
        )
        colbuf = np.zeros((64, 17), np.float32)
        n = min(len(cols), 64)
        colbuf[:n, : min(cols.shape[1], 17)] = cols[:n, :17]
        with torch.no_grad():
            d, lm = net._run_dope_mlc_v3_and_lm48(
                torch.as_tensor(vec, device=device).reshape(1, 256),
                torch.as_tensor(colbuf, device=device).reshape(1, 64, 17),
                use_landmarks=True,
            )
            tok, mask, _ = net._project_stage1b_tokens_batched_gpu(lm, d)
        tokens[cid], masks[cid] = tok[0].detach(), mask[0].detach()
    return tokens, masks
