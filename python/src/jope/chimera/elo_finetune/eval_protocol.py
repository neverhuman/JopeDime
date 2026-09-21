"""Shared measurement protocols.

Tier 1 (training-time proxy): identity view, seeded size-quantized row/column
subsets from the pack, single forward, differentiable soft utility
û = −sqrt(mse + 1e-12) decoded exactly like inference (softmax @ centres,
un-z-scored by context stats). Context mu/sd are computed per dataset in
numpy f32 (mirroring predict_reg and independent of batch composition).

Tier 2 (authoritative): full rows, ALL features, the 6-view ensemble median
with predict_reg's own `_transform`/`VIEWS` and its exact `_mean` chunking —
produces the report-protocol rmse in f64.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np
import torch

from .pack import Batch, DatasetView
from .paths import regressor_source_dir


def _predict_reg():
    source_dir = str(regressor_source_dir())
    if source_dir not in sys.path:
        sys.path.insert(0, source_dir)
    import predict_reg

    return predict_reg


@dataclass
class Tier1Output:
    u_soft: torch.Tensor  # (B,) differentiable, report units
    logits: torch.Tensor  # (qry, B, n_buckets)
    y_qry_z: torch.Tensor  # (qry, B) query targets in context z-space
    mu: torch.Tensor  # (B,)
    sd: torch.Tensor  # (B,)

    def utilities_f64(self) -> np.ndarray:
        return self.u_soft.detach().double().cpu().numpy()


class ProtocolRunner:
    def __init__(self, model, device: torch.device, autocast_dtype=torch.bfloat16):
        self.model = model
        self.device = device
        self.autocast_dtype = autocast_dtype
        borders = model.regression_borders.float()
        self.centres = (borders[:-1] + (borders[1:] - borders[:-1]) / 2).to(device)

    def _autocast(self):
        return torch.autocast(
            device_type=self.device.type,
            dtype=self.autocast_dtype,
            enabled=self.device.type == "cuda",
        )

    def context_stats(self, y_ctx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Per-dataset f32 stats exactly as predict_reg._mean computes them."""
        mu = np.array([float(y_ctx[:, b].mean()) for b in range(y_ctx.shape[1])])
        sd = np.array([float(y_ctx[:, b].std()) + 1e-8 for b in range(y_ctx.shape[1])])
        return mu, sd

    def tier1(self, batch: Batch, with_grad: bool) -> Tier1Output:
        mu_np, sd_np = self.context_stats(batch.y_ctx)
        x = torch.from_numpy(batch.x).to(self.device)
        y_ctx = torch.from_numpy(batch.y_ctx).to(self.device)
        y_qry = torch.from_numpy(batch.y_qry).to(self.device)
        mu = torch.from_numpy(mu_np).float().to(self.device)
        sd = torch.from_numpy(sd_np).float().to(self.device)
        y_ctx_z = (y_ctx - mu) / sd

        ctx = torch.enable_grad() if with_grad else torch.no_grad()
        with ctx, self._autocast():
            logits = self.model(x, y_ctx_z)
        logits = logits.float()
        p = torch.softmax(logits, dim=-1)
        yhat = (p @ self.centres) * sd + mu
        mse = ((yhat - y_qry) ** 2).mean(dim=0)
        u_soft = -torch.sqrt(mse + 1e-12)
        y_qry_z = (y_qry - mu) / sd
        return Tier1Output(u_soft=u_soft, logits=logits, y_qry_z=y_qry_z, mu=mu, sd=sd)

    @torch.no_grad()
    def _mean_single(
        self, cx: np.ndarray, cy: np.ndarray, qx: np.ndarray, max_total: int = 1024
    ) -> np.ndarray:
        """Bit-faithful port of predict_reg.Regressor._mean for a live model."""
        cx = np.array(cx, np.float32, copy=True)
        qx = np.array(qx, np.float32, copy=True)
        cy = np.array(cy, np.float32, copy=True)
        mu, sd = float(cy.mean()), float(cy.std()) + 1e-8
        cyz = (cy - mu) / sd
        C = cx.shape[0]
        out = []
        cxt = torch.from_numpy(cx).to(self.device)
        cyt = torch.from_numpy(cyz).to(self.device)
        qc = max(8, min(qx.shape[0], max_total - C))
        for s in range(0, qx.shape[0], qc):
            qxt = torch.from_numpy(qx[s : s + qc]).to(self.device)
            x = torch.cat([cxt, qxt], 0).unsqueeze(1)
            yt = cyt.unsqueeze(1)
            with self._autocast():
                lo = self.model(x, yt)
            mean_z = (torch.softmax(lo.float(), -1) @ self.centres).reshape(-1)
            out.append((mean_z * sd + mu).cpu().numpy())
        return np.concatenate(out)[: qx.shape[0]]

    @torch.no_grad()
    def tier2_rmse(self, view: DatasetView) -> float:
        """Report-protocol rmse: 6-view ensemble median on full rows/features."""
        predict_reg = _predict_reg()
        cx, cy, qx = view.Xtr, view.ytr, view.Xte
        preds = []
        for name in predict_reg.VIEWS:
            cxt, qxt = predict_reg._transform(name, cx, qx)
            preds.append(self._mean_single(cxt, cy, qxt))
        pred = np.median(np.stack(preds, 0), 0)
        err = pred.astype(np.float64) - view.yte.astype(np.float64)
        return float(np.sqrt(np.mean(err * err)))
