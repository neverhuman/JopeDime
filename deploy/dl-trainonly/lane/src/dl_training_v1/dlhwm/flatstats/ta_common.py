"""TA encoders (TA_Hyperion_v8, TA_Dope_V2): shared data path, leak rules and run guards.

Row data source: the train-only JTT1 encoder payloads published under the MAIN v2-trainonly tree
(inputs/jtt1-encoder-s0/<dataset>/<fold>/<view>.jtt1.gz; outer-test block stripped, n_test == 0).
Leak rules enforced here (hard):
  * Only the outer-train byte range [17 + n_train, 17 + n_train + 8*(p+1)*n_train) is ever decoded (train_view slices the
    bytes before any parse); a payload with n_test != 0 is refused outright (s0 payloads must be stripped).
  * Every reduction / scaler is fit on outer-train rows only: target-copy drop (same rule as the v7 encoder), XGBoost gain
    importance (fixed seed 20261007, outer-train rows) to cap width at PSTORE, fixed-seed row subsample to NSTORE.
  * G3 contexts (research-only rights, no row-data ruling) are excluded by context id even though none carry s0 inputs.
GP-HWM labels are never read by training code; only the checkpoint probe (ta_probe.py) reads them.
"""
import gzip, hashlib, io, json, os, struct, subprocess, time
import numpy as np

HEADER = struct.Struct("<IIIB"); HB = 4 + HEADER.size
PSTORE = 256      # max features stored per context (importance-reduced beyond this)
PHV = 64          # Hyperion v8 trunk width cap (first PHV of the importance-ordered stored features)
NSTORE = 2048     # max outer-train rows stored per context (fixed-seed subsample)
SEED = 20261007


def sha256(b):
    return hashlib.sha256(b).hexdigest()


def jtt1_header(raw):
    if bytes(raw[:4]) != b"JTT1":
        raise ValueError("only JTT1 is admitted")
    n_train, n_test, p, inner = HEADER.unpack(bytes(raw[4:HB]))
    if len(raw) != HB + n_train + 8 * (p + 1) * (n_train + n_test) or n_train < 1 or p < 1:
        raise ValueError("JTT1 length or shape mismatch")
    return n_train, n_test, p, inner


def train_view(raw, require_stripped=True):
    """(X f64[n_train,p], y f64[n_train], header). Slices the outer-train range BEFORE parsing; the test range is never decoded."""
    n_train, n_test, p, inner = jtt1_header(raw)
    if require_stripped and n_test != 0:
        raise ValueError(f"payload not stripped (n_test={n_test}); TA training only admits stripped s0 payloads")
    start = HB + n_train
    block = np.frombuffer(memoryview(raw)[start:start + 8 * (p + 1) * n_train], dtype="<f8").reshape(n_train, p + 1).copy()
    return block[:, :-1], block[:, -1], dict(n_train=n_train, n_test=n_test, p=p, inner_folds=inner)


def drop_target_copies(X, y):
    """Same rule as HYPERION-V7/tools/v2_gpu_encode.py drop_target_copies (exact affine copies of y on outer-train rows)."""
    removed = []
    if np.ptp(y) > 1e-12:
        c = y - y.mean(); den = float(c @ c)
        for i in range(X.shape[1]):
            v = X[:, i]; slope = float(c @ (v - v.mean())) / den
            if abs(slope) > 1e-12 and np.max(np.abs(v - (slope * y + (v.mean() - slope * y.mean())))) <= 1e-10 * max(1, float(np.max(np.abs(v)))):
                removed.append(i)
    kept = [i for i in range(X.shape[1]) if i not in removed]
    return kept, removed


def importance_order(X, y, seed=SEED):
    """XGBoost gain importance fit on the given (outer-train) rows, fixed seed; ties broken by variance then index."""
    import xgboost as xgb
    Xc = np.nan_to_num(X.astype(np.float32), nan=0.0)
    bst = xgb.train(dict(tree_method="hist", device="cpu", max_depth=4, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
                         seed=seed, nthread=2, max_bin=64, verbosity=0), xgb.DMatrix(Xc, label=y.astype(np.float32)), num_boost_round=60)
    g = bst.get_score(importance_type="gain"); score = np.zeros(X.shape[1])
    for k, v in g.items():
        score[int(k[1:])] = v
    var = np.nanvar(Xc, 0)
    return sorted(range(X.shape[1]), key=lambda j: (-score[j], -var[j], j)), score


def prep_context(raw, cid):
    """outer-train rows -> stored arrays + manifest record (all fits on outer-train rows only)."""
    X, y, h = train_view(raw)
    ok = np.isfinite(y); X, y = X[ok], y[ok]
    kept, removed = drop_target_copies(np.nan_to_num(X), y)
    X = X[:, kept]
    if X.shape[1] == 0 or X.shape[0] < 8 or np.ptp(y) <= 1e-12:
        return None, {"cid": cid, "skipped": "degenerate (no features / <8 rows / constant y)"}
    rng = np.random.default_rng(int(cid[:16], 16) ^ SEED)
    order = rng.permutation(X.shape[0])[:NSTORE]            # fixed-seed subsample + shuffle (row order carries no info)
    imp_rows = order
    reduced = None
    if X.shape[1] > PHV:
        feat_order, score = importance_order(X[imp_rows], y[imp_rows])
        reduced = {"method": "xgboost-gain", "seed": SEED, "rounds": 60, "fit_rows": int(len(imp_rows)), "p_in": int(X.shape[1])}
    else:
        feat_order = list(range(X.shape[1]))
    feat_order = feat_order[:PSTORE]
    Xs = X[order][:, feat_order].astype(np.float32); ys = y[order].astype(np.float32)
    miss = np.isnan(Xs).mean(0).astype(np.float32)
    rec = {"cid": cid, "n_train": h["n_train"], "n_test": h["n_test"], "p_raw": h["p"], "removed_target_copies": removed,
           "n": int(Xs.shape[0]), "p": int(Xs.shape[1]), "features": [int(kept[j]) for j in feat_order], "reduction": reduced,
           "binary": bool(set(np.unique(ys).tolist()) <= {0.0, 1.0}), "train_rows_sha256": sha256(memoryview(raw)[HB:HB + h["n_train"] + 8 * (h["p"] + 1) * h["n_train"]])}
    return (Xs, ys, miss), rec


class Guard:
    """Cooperative yield guard: pause while host RAM available < 17%, the GPU has < min_free_mib free, or another
    process (the encode queue) is busy on the GPU (then cap our duty cycle). Never signals any other process."""

    def __init__(self, log, min_free_mib=3000, duty_when_busy=0.35, siblings=()):
        """siblings: cmdline substrings of peer processes that do NOT count as 'busy' (e.g. the two v8b runs ignore each
        other). Everything else on the GPU — the encode queue and the already-running hv8r2 run — keeps priority."""
        self.log, self.min_free, self.duty, self.siblings = log, min_free_mib, duty_when_busy, tuple(siblings)
        self.t_check = 0; self.busy = False; self.paused_s = 0.0

    def mem_ok(self):
        m = {}
        for l in open("/proc/meminfo"):
            k, v = l.split(":"); m[k] = int(v.split()[0])
        return m["MemAvailable"] / m["MemTotal"] >= 0.17

    def gpu_state(self):
        try:
            q = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout.split(",")
            free = int(q[1]) - int(q[0])
            pm = subprocess.run(["nvidia-smi", "pmon", "-c", "1", "-s", "u"], capture_output=True, text=True, timeout=15).stdout
            me = os.getpid(); other = 0
            for l in pm.splitlines():
                f = l.split()
                if len(f) > 4 and f[1].isdigit() and int(f[1]) != me and f[3] not in ("-",):
                    if self.siblings:
                        try: cl = open(f"/proc/{f[1]}/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="ignore")
                        except OSError: cl = ""
                        if any(x in cl for x in self.siblings): continue
                    try: other = max(other, int(f[3]))
                    except ValueError: pass
            return free, other
        except Exception:
            return 99999, 0

    def step(self, step_s):
        now = time.time()
        if now - self.t_check > 30:
            self.t_check = now
            while True:
                free, other = self.gpu_state(); mem = self.mem_ok()
                self.busy = other >= 25
                if mem and free >= self.min_free:
                    break
                self.log(f"guard pause: ram_ok={mem} gpu_free_mib={free} other_sm={other}"); time.sleep(30); self.paused_s += 30
        if self.busy and step_s > 0:   # encode queue active on this GPU -> cap our duty cycle
            s = step_s * (1 / self.duty - 1); time.sleep(min(s, 2.0)); self.paused_s += min(s, 2.0)


def gpu_info():
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    except Exception:
        return "unknown"
