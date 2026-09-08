"""Frozen experts with a common interface. Each wrapper hides its model's calling
convention and hooks the 512-d per-test-row hidden state.

ExpertOut fields (per test row, batch B=1 dataset at a time):
  mean      (M,)        de-standardized point prediction (regression) or None
  logits    (M, K)      class logits (classification) or None
  quantiles (M, Q)      predictive quantiles (regression) or None
  hidden    (M, 512)    row-interactor / output_norm embedding
  conf      (M, Cc)     per-row confidence features (generalizable)
"""
from __future__ import annotations

import numpy as np
import torch
from dataclasses import dataclass

HID = 512


@dataclass
class ExpertOut:
    hidden: torch.Tensor            # (M, 512) float32 cpu
    conf: torch.Tensor             # (M, Cc)
    mean: torch.Tensor | None = None       # (M,)
    logits: torch.Tensor | None = None     # (M, K)
    quantiles: torch.Tensor | None = None  # (M, Q)


def _freeze(model):
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _zscore(X: np.ndarray, ref: np.ndarray, clip: float = 100.0):
    mu = ref.mean(0, keepdims=True); sd = ref.std(0, keepdims=True) + 1e-6
    return np.clip((X - mu) / sd, -clip, clip).astype("float32")


class Expert:
    name = "base"
    task = "regression"

    def infer(self, ds) -> ExpertOut:  # ds: fusion.data.QuantDataset
        raise NotImplementedError


# --------------------------------------------------------------------------
# TabPFN-v3 regressor (our rank-1 native regressor) — bar distribution head
# --------------------------------------------------------------------------
class TabPFNv3Reg(Expert):
    name = "tabpfn_v3_reg"
    task = "regression"

    MAX_TOTAL = 1024   # model caps context+query per forward (bf16 autocast)

    def __init__(self, device="cuda", full_kv=True):
        import sys
        sys.path.insert(0, "/home/ubuntu/fable_god")
        from python.elo_finetune.model_io import load_regressor, bucket_centres
        self.device = torch.device(device)
        self.model = _freeze(load_regressor(device=device))
        if full_kv:
            for m in self.model.modules():
                if hasattr(m, "num_kv_heads_test"):
                    m.num_kv_heads_test = None
        self.centres = bucket_centres(self.model).to(self.device)

    @torch.no_grad()
    def infer(self, ds) -> ExpertOut:
        cx = torch.from_numpy(ds.Xtr.astype("float32")).to(self.device)
        qx_all = ds.Xte.astype("float32")
        cy = ds.ytr.astype("float32")
        mu, sd = float(cy.mean()), float(cy.std()) + 1e-8
        cyz = torch.from_numpy((cy - mu) / sd).to(self.device).unsqueeze(1)
        C = cx.shape[0]; qc = max(8, min(qx_all.shape[0], self.MAX_TOTAL - C))
        logits_parts, hid_parts = [], []
        for s in range(0, qx_all.shape[0], qc):
            qx = torch.from_numpy(qx_all[s:s + qc]).to(self.device)
            x = torch.cat([cx, qx], 0).unsqueeze(1)
            out = self.model(x, cyz, only_return_standard_out=False)
            logits_parts.append(out["standard"].float().reshape(qx.shape[0], -1))
            hid_parts.append(out["test_embeddings"].float().reshape(qx.shape[0], HID))
        logits = torch.cat(logits_parts, 0)             # (M, 5000)
        hid = torch.cat(hid_parts, 0)                   # (M, 512)
        p = torch.softmax(logits, -1)
        mean_z = p @ self.centres
        mean = mean_z * sd + mu
        cdf = torch.cumsum(p, -1)
        qlev = torch.linspace(0.05, 0.95, 19, device=self.device)
        idx = torch.searchsorted(cdf, qlev.expand(qx_all.shape[0], -1)).clamp(0, self.centres.numel() - 1)
        quant = self.centres[idx] * sd + mu             # (M, 19)
        var = (p * (self.centres.unsqueeze(0) - mean_z.unsqueeze(1)) ** 2).sum(-1)
        std = var.clamp_min(0).sqrt() * sd
        ent = -(p.clamp_min(1e-9).log() * p).sum(-1)
        conf = torch.stack([std, ent, mean], -1).cpu()
        return ExpertOut(hidden=hid.cpu(), conf=conf, mean=mean.cpu(), quantiles=quant.cpu())


# --------------------------------------------------------------------------
# TabICL regressor — 999-quantile head, row_interactor 512-d hidden
# --------------------------------------------------------------------------
import os as _os


class TabICLReg(Expert):
    name = "tabicl_reg"
    task = "regression"
    W = _os.environ.get("TABICL_REG_WEIGHTS",
        "/home/ubuntu/jain_small/artifacts/foundation/tabicl-regressor-v2-20260212.safetensors")

    def __init__(self, device="cuda"):
        from tabicl._model.tabicl import TabICL
        from safetensors.torch import load_file
        self.device = torch.device(device)
        m = TabICL(max_classes=0, num_quantiles=999, bias_free_ln=True)
        m.load_state_dict(load_file(self.W), strict=True)
        self.model = _freeze(m).to(self.device)
        self._hid = {}
        self.model.row_interactor.register_forward_hook(
            lambda mod, i, o: self._hid.__setitem__("h", o.detach()))
        self.qlev = torch.linspace(0.0005, 0.9995, 999)

    @torch.no_grad()
    def infer(self, ds) -> ExpertOut:
        n = ds.n_train
        X = np.concatenate([ds.Xtr, ds.Xte], 0)
        Xz = _zscore(X, ds.Xtr)                          # z-score by train stats
        cy = ds.ytr.astype("float32"); mu, sd = float(cy.mean()), float(cy.std()) + 1e-8
        yz = (cy - mu) / sd
        Xt = torch.from_numpy(Xz).unsqueeze(0).to(self.device)
        yt = torch.from_numpy(yz).unsqueeze(0).to(self.device)
        q = self.model(Xt, yt).float()[0]               # (M, 999) raw quantiles (z-space)
        q, _ = torch.sort(q, -1)                         # enforce monotone
        quant_z = q
        quant = quant_z * sd + mu
        mean = quant.mean(-1)                            # mean of quantiles ~ E[y]
        # 19 evenly-spaced quantiles to match a common grid
        pick = torch.linspace(0, 998, 19).long()
        quant19 = quant[:, pick]
        spread = (quant[:, 899] - quant[:, 99])          # 90-10 spread (uncertainty)
        hid = self._hid["h"].float()[0, n:]              # test rows (M, 512)
        conf = torch.stack([spread * 0 + spread, spread * 0, mean], -1).cpu()
        return ExpertOut(hidden=hid.cpu(), conf=conf,
                         mean=mean.cpu(), quantiles=quant19.cpu())


# --------------------------------------------------------------------------
# Classification experts (binary) — 2-class logits + 512-d hidden
# --------------------------------------------------------------------------
def _cls_conf(p1):
    """per-row conf from P(class=1): [confidence, entropy, signed margin]."""
    p = torch.stack([1 - p1, p1], -1).clamp(1e-6, 1 - 1e-6)
    ent = -(p * p.log()).sum(-1)
    return torch.stack([p.max(-1).values, ent, p1 - (1 - p1)], -1)


class TabPFNv3ClsBinary(Expert):
    name = "tabpfn_v3_cls"
    task = "binary"
    CKPT = _os.environ.get("TABPFN_CLS_CKPT",
        "/home/ubuntu/tabpfn-v3-classifier-v3_20260417_binary.ckpt")
    DEPS = _os.environ.get("TABPFN_CLS_DEPS",
        "/home/ubuntu/fable_god/provenance/classifier_deps")
    MAX_TOTAL = 1024

    def __init__(self, device="cuda"):
        import sys
        sys.path.insert(0, self.DEPS)
        from tabpfn_v3_train import build_model_from_checkpoint
        self.device = torch.device(device)
        self.model = _freeze(build_model_from_checkpoint(
            self.CKPT, load_weights=True, strict=True, device=device))

    @torch.no_grad()
    def infer(self, ds) -> ExpertOut:
        cx = torch.from_numpy(ds.Xtr.astype("float32")).to(self.device)
        qx_all = ds.Xte.astype("float32")
        cy = torch.from_numpy(ds.ytr.astype("float32")).to(self.device).unsqueeze(1)
        C = cx.shape[0]; qc = max(8, min(qx_all.shape[0], self.MAX_TOTAL - C))
        lg_parts, hid_parts = [], []
        for s in range(0, qx_all.shape[0], qc):
            qx = torch.from_numpy(qx_all[s:s + qc]).to(self.device)
            x = torch.cat([cx, qx], 0).unsqueeze(1)
            out = self.model(x, cy, only_return_standard_out=False)
            lg_parts.append(out["standard"][..., :2].float().reshape(qx.shape[0], 2))
            hid_parts.append(out["test_embeddings"].float().reshape(qx.shape[0], HID))
        logits = torch.cat(lg_parts, 0)                 # (M,2) log-probs
        hid = torch.cat(hid_parts, 0)
        p1 = torch.softmax(logits, -1)[:, 1]
        return ExpertOut(hidden=hid.cpu(), conf=_cls_conf(p1).cpu(), logits=logits.cpu())

    @torch.no_grad()
    def infer_batch(self, tasks) -> list:
        """Batched over B same-shape tasks. x:[R,B,C], y:[N,B]; chunk queries."""
        C = tasks[0].Xtr.shape[0]; B = len(tasks)
        cx = torch.from_numpy(np.stack([t.Xtr for t in tasks], 1)).to(self.device)  # (C,B,F)
        qx_all = np.stack([t.Xte for t in tasks], 1)                                 # (Mtot,B,F)
        cy = torch.from_numpy(np.stack([t.ytr.astype("float32") for t in tasks], 1)).to(self.device)  # (C,B)
        M = qx_all.shape[0]; qc = max(8, min(M, self.MAX_TOTAL - C))
        lg_parts, hid_parts = [], []
        for s in range(0, M, qc):
            qx = torch.from_numpy(qx_all[s:s + qc]).to(self.device)
            x = torch.cat([cx, qx], 0)                    # (C+q, B, F)
            out = self.model(x, cy, only_return_standard_out=False)
            lg_parts.append(out["standard"][..., :2].float())      # (q,B,2)
            hid_parts.append(out["test_embeddings"].float())       # (q,B,512)
        logits = torch.cat(lg_parts, 0)                   # (M,B,2)
        hid = torch.cat(hid_parts, 0)                     # (M,B,512)
        outs = []
        for b in range(B):
            p1 = torch.softmax(logits[:, b], -1)[:, 1]
            outs.append(ExpertOut(hidden=hid[:, b].cpu(), conf=_cls_conf(p1).cpu(),
                                  logits=logits[:, b].cpu()))
        return outs


class TabICLCls(Expert):
    name = "tabicl_cls"
    task = "binary"
    W = _os.environ.get("TABICL_CLS_WEIGHTS",
        "/home/ubuntu/jain_small/artifacts/foundation/tabicl-classifier-v2-20260212.safetensors")

    def __init__(self, device="cuda"):
        from tabicl._model.tabicl import TabICL
        from safetensors.torch import load_file
        self.device = torch.device(device)
        m = TabICL(max_classes=10)
        m.load_state_dict(load_file(self.W), strict=True)
        self.model = _freeze(m).to(self.device)
        self._hid = {}
        self.model.row_interactor.register_forward_hook(
            lambda mod, i, o: self._hid.__setitem__("h", o.detach()))

    @torch.no_grad()
    def infer(self, ds) -> ExpertOut:
        n = ds.n_train
        Xz = _zscore(np.concatenate([ds.Xtr, ds.Xte], 0), ds.Xtr)
        X = torch.from_numpy(Xz).unsqueeze(0).to(self.device)
        y = torch.from_numpy(ds.ytr.astype("float32")).unsqueeze(0).to(self.device)
        lg = self.model(X, y).float()[0]                # (M, K>=2)
        logits = lg[:, :2] if lg.shape[1] >= 2 else torch.cat([lg, lg], -1)
        p1 = torch.softmax(logits, -1)[:, 1]
        hid = self._hid["h"].float()[0, n:]
        return ExpertOut(hidden=hid.cpu(), conf=_cls_conf(p1).cpu(), logits=logits.cpu())

    @torch.no_grad()
    def infer_batch(self, tasks) -> list:
        """Batched over B same-shape tasks -> one forward. tasks: list[PackedTask]."""
        n = tasks[0].Xtr.shape[0]
        Xb = np.stack([_zscore(np.concatenate([t.Xtr, t.Xte], 0), t.Xtr) for t in tasks])
        yb = np.stack([t.ytr.astype("float32") for t in tasks])
        X = torch.from_numpy(Xb).to(self.device); y = torch.from_numpy(yb).to(self.device)
        lg = self.model(X, y).float()                   # (B, M, K)
        H = self._hid["h"].float()                       # (B, T, 512)
        outs = []
        for b in range(len(tasks)):
            logits = lg[b][:, :2] if lg.shape[-1] >= 2 else torch.cat([lg[b], lg[b]], -1)
            p1 = torch.softmax(logits, -1)[:, 1]
            outs.append(ExpertOut(hidden=H[b, n:].cpu(), conf=_cls_conf(p1).cpu(),
                                  logits=logits.cpu()))
        return outs


# --------------------------------------------------------------------------
# GBDT classification expert — the DECORRELATED 3rd expert. A gradient-boosted
# tree ensemble (LightGBM) is a structurally different model class from the two
# transformers, so its errors are decorrelated → it widens the fusion oracle and
# gives the stacking head complementary signal (the decisive-win lever). Its VALUE
# is its prediction (logits/conf); it has no natural 512-d hidden (→ zeros, which
# the fusion adapter maps to 0; the head still sees its calibrated logits + conf).
# --------------------------------------------------------------------------
class GBDTCls(Expert):
    name = "gbdt_cls"
    task = "binary"

    def __init__(self, device="cuda", n_estimators=300, n_jobs=4):
        self.n_estimators = n_estimators; self.n_jobs = n_jobs

    @torch.no_grad()
    def infer(self, ds) -> ExpertOut:
        import lightgbm as lgb
        Xtr = np.asarray(ds.Xtr, dtype="float32"); ytr = np.asarray(ds.ytr).astype("int32")
        Xte = np.asarray(ds.Xte, dtype="float32"); M = Xte.shape[0]
        if np.unique(ytr).size < 2:                              # degenerate context → constant
            p1 = np.full(M, float(np.clip(ytr.mean(), 1e-3, 1 - 1e-3)), dtype="float32")
        else:
            n = len(ytr)
            m = lgb.LGBMClassifier(objective="binary", n_estimators=self.n_estimators,
                                   learning_rate=0.05, num_leaves=31,
                                   min_child_samples=max(5, n // 50), subsample=0.8,
                                   subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                                   n_jobs=self.n_jobs, verbosity=-1)
            m.fit(Xtr, ytr)
            p1 = m.predict_proba(Xte)[:, 1].astype("float32")
        p1 = np.clip(p1, 1e-6, 1 - 1e-6)
        p1t = torch.from_numpy(p1)
        logits = torch.stack([torch.log1p(-p1t), torch.log(p1t)], -1)   # (M,2) log-probs → softmax recovers p1
        return ExpertOut(hidden=torch.zeros(M, HID), conf=_cls_conf(p1t), logits=logits)

    def infer_batch(self, tasks) -> list:
        return [self.infer(t) for t in tasks]              # CPU per-task fit; batched TRAIN path needs this


def _sk_out(clf, ds):
    """Fit a sklearn binary classifier per dataset → ExpertOut (logits/conf; zero hidden)."""
    Xtr = np.asarray(ds.Xtr, dtype="float32"); ytr = np.asarray(ds.ytr).astype("int32")
    Xte = np.asarray(ds.Xte, dtype="float32"); M = Xte.shape[0]
    if np.unique(ytr).size < 2:
        p1 = np.full(M, float(np.clip(ytr.mean(), 1e-3, 1 - 1e-3)), dtype="float32")
    else:
        clf.fit(Xtr, ytr); p1 = clf.predict_proba(Xte)[:, 1].astype("float32")
    p1 = np.clip(p1, 1e-6, 1 - 1e-6); p1t = torch.from_numpy(p1)
    logits = torch.stack([torch.log1p(-p1t), torch.log(p1t)], -1)
    return ExpertOut(hidden=torch.zeros(M, HID), conf=_cls_conf(p1t), logits=logits)


class ExtraTreesCls(Expert):
    """Decorrelated 4th expert — bagged randomized trees (variance-reduction bias, very
    different from boosting + transformers)."""
    name = "extratrees_cls"; task = "binary"

    def __init__(self, device="cuda", n_estimators=400, n_jobs=4):
        from sklearn.ensemble import ExtraTreesClassifier
        self._mk = lambda: ExtraTreesClassifier(n_estimators, n_jobs=n_jobs)

    @torch.no_grad()
    def infer(self, ds) -> ExpertOut:
        return _sk_out(self._mk(), ds)

    def infer_batch(self, tasks) -> list:
        return [self.infer(t) for t in tasks]


class LogisticCls(Expert):
    """Decorrelated 5th expert — a LINEAR model (standardized logistic). Maximally
    different inductive bias; wins on ~linearly-separable datasets where trees/transformers
    over-fit, so it widens the fusion oracle."""
    name = "logistic_cls"; task = "binary"

    def __init__(self, device="cuda", C=1.0):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
        self._mk = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=C))

    @torch.no_grad()
    def infer(self, ds) -> ExpertOut:
        return _sk_out(self._mk(), ds)

    def infer_batch(self, tasks) -> list:
        return [self.infer(t) for t in tasks]
