"""Frozen native engine; augmentation changes only tokens or slate ordering."""

import numpy as np
import torch
from parents import h192
from treatment import noisy_tokens, permutations, permute_kwargs


class Engine(h192.Engine):
    def check(self):
        super().check()
        if (
            self.rt.tensor_hash(h192.adapter_state(self.net, self.names))
            != self.initial_hash
        ):
            raise RuntimeError("adapter state drift")

    def batches(self, rows):
        groups = {}
        for row in rows:
            cid = row["context_sha256"]
            groups.setdefault(len(self.pools[cid]["fitness"]), []).append(cid)
        for k, cids in sorted(groups.items()):
            for start in range(0, len(cids), 8):
                batch = cids[start : start + 8]
                kwargs, tokens, masks, _, _ = self.batch(
                    batch, [np.arange(k) for _ in batch]
                )
                yield batch, kwargs, tokens, masks

    def native_forward(self, kwargs, tokens, masks):
        return self.net.adapter_10tok.predict_costs(
            **kwargs, mm_tokens_10x32=tokens, mm_mask_10=masks
        )["total_cost"].float()

    def verify_costs(self, cids, scores):
        expected = torch.as_tensor(np.stack([self.native_costs[c] for c in cids]))
        if not torch.equal(expected, scores):
            raise RuntimeError("native batching/cached cost identity failure")

    def noise(self, rows, scale, seed, cell):
        self.check()
        result = {}
        with torch.inference_mode():
            for cids, kwargs, tokens, masks in self.batches(rows):
                anchor = tuple(
                    torch.stack([self.cached[c][j] for c in cids]) for j in range(2)
                )
                self.verify_costs(cids, self.forward(kwargs, tokens, masks, anchor))
                draws = []
                for draw in range(16):
                    altered = torch.as_tensor(
                        np.stack(
                            [
                                noisy_tokens(
                                    self.tokens[c],
                                    self.masks[c],
                                    scale,
                                    seed,
                                    cell,
                                    c,
                                    draw,
                                )
                                for c in cids
                            ]
                        )
                    )
                    scores = self.forward(kwargs, altered, masks, anchor)
                    if not torch.isfinite(scores).all():
                        raise RuntimeError("nonfinite noisy costs")
                    draws.append(scores.numpy().copy())
                costs = np.stack(draws)
                for i, cid in enumerate(cids):
                    result[cid] = costs[:, i].copy()
        self.check()
        return result

    def permuted(self, rows, seed):
        self.check()
        result = {}
        with torch.inference_mode():
            for cids, kwargs, tokens, masks in self.batches(rows):
                self.verify_costs(cids, self.native_forward(kwargs, tokens, masks))
                plans = [
                    permutations(seed, c, kwargs["prog_412"].shape[1]) for c in cids
                ]
                draws = []
                for draw in range(8):
                    changed, inverse = permute_kwargs(
                        kwargs, np.stack([p[draw] for p in plans])
                    )
                    scores = self.native_forward(changed, tokens, masks).gather(
                        1, inverse
                    )
                    if not torch.isfinite(scores).all():
                        raise RuntimeError("nonfinite permuted costs")
                    draws.append(scores.numpy().copy())
                for i, cid in enumerate(cids):
                    result[cid] = np.stack(draws)[:, i].copy()
        self.check()
        return result

    def probe(self, rows, seed):
        """FIT-only first eight IDs; stop at first non-bitexact permutation."""
        self.check()
        cids = sorted(r["context_sha256"] for r in rows)[:8]
        compared, arrays = 0, {}
        with torch.inference_mode():
            for cid in cids:
                k = len(self.pools[cid]["fitness"])
                kwargs, tokens, masks, _, _ = self.batch([cid], [np.arange(k)])
                native = self.native_forward(kwargs, tokens, masks)
                # Same batch shape on both sides isolates candidate ordering.
                arrays[cid + "__native"] = native.numpy().copy()
                for draw, order in enumerate(permutations(seed, cid, k)):
                    changed, inverse = permute_kwargs(kwargs, order[None, :])
                    scores = self.native_forward(changed, tokens, masks).gather(
                        1, inverse
                    )
                    arrays[cid + f"__perm{draw}"] = scores.numpy().copy()
                    compared += 1
                    if not torch.equal(native, scores):
                        self.check()
                        return dict(
                            skip=False,
                            comparisons=compared,
                            planned_contexts=cids,
                            first_mismatch=dict(
                                context=cid,
                                draw=draw,
                                max_abs_delta=float((native - scores).abs().max()),
                            ),
                        ), arrays
        self.check()
        return dict(
            skip=True,
            comparisons=compared,
            planned_contexts=cids,
            first_mismatch=None,
            scope="64 FIT comparisons only; not global proof",
        ), arrays
