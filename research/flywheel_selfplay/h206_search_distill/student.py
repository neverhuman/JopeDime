"""Width128 residual student with the inherited live Fused DOPE/SVD path."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import torch
from jope.north_star.dope_freeze_guard import assert_research_optimizer_policy, configure_research_dope
from torch import nn

from .contracts import canonical, digest, guard_cpu, sha

ARCHITECTURE = {
    "schema": "h206-student-width128-v1",
    "input": 749,
    "hidden": [128, 128],
    "activation": "GELU",
    "output": 1,
    "output_init": "zero",
    "dropout": 0,
    "optimizer": "AdamW",
    "lr": 0.0001,
    "weight_decay": 0.0001,
    "teacher": "frozen r882 adapter_10tok; original tokens and native full kernel",
    "encoder": "inherited Fused DOPE router/col_gate + Stage1B SVD, live autograd",
}
ARCHITECTURE_SHA256 = digest(ARCHITECTURE)


def tensor_hash(value):
    a = value.detach().cpu().contiguous().numpy()
    return sha(canonical({"dtype": a.dtype.str, "shape": list(a.shape)}) + a.tobytes())


def state_hash(module):
    return digest({k: tensor_hash(v) for k, v in sorted(module.state_dict().items())})


class SearchStudent(nn.Module):
    """A widened adapter, initialized from r882 assets, never H204 fitted weights."""

    def __init__(
        self,
        fused,
        *,
        seed: int,
        run_id: str,
        launch_metadata: dict,
        encoder_state_sha256: str,
        anchor_state_sha256: str,
        encoder_source_sha256: str,
    ):
        super().__init__()
        if (
            state_hash(fused) != encoder_state_sha256
            or state_hash(fused.adapter_10tok) != anchor_state_sha256
        ):
            raise ValueError("frozen r882 encoder/teacher model content identity mismatch")
        source_file = inspect.getsourcefile(fused._run_dope_mlc_v3_and_lm48)
        if source_file is None or sha(Path(source_file).read_bytes()) != encoder_source_sha256:
            raise ValueError("inherited encoder source content identity mismatch")
        self.encoder_source_sha256 = encoder_source_sha256
        self.encoder_initial_sha256 = encoder_state_sha256
        self.encoder = fused
        self.dope_model = fused.dope_model
        self.gamenet_anchor = fused.adapter_10tok
        for p in self.parameters():
            p.requires_grad_(False)
            p.grad = None
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.residual = nn.Sequential(
                nn.Linear(749, 128), nn.GELU(), nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 1)
            )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)
        configure_research_dope(self, launch_metadata=launch_metadata, unfreeze=True, research_run_id=run_id)
        self.run_id = run_id
        self.initial_frozen = self.frozen_hashes()

    def frozen_hashes(self):
        return {
            "parameters": {k: tensor_hash(v) for k, v in self.named_parameters() if not v.requires_grad},
            "buffers": {k: tensor_hash(v) for k, v in self.named_buffers()},
        }

    def assert_frozen(self):
        expected = {id(p) for p in self.dope_model.parameters()} | {id(p) for p in self.residual.parameters()}
        actual = {id(p) for p in self.parameters() if p.requires_grad}
        if (
            actual != expected
            or self.frozen_hashes() != self.initial_frozen
            or any(p.grad is not None for p in self.parameters() if not p.requires_grad)
            or any(b.requires_grad for b in self.buffers())
        ):
            raise ValueError("research trainable closure/frozen anchor or normalizer changed")

    def forward(self, loaded):
        loaded.verify()
        if (
            loaded.context.encoder_sha256 != self.encoder_source_sha256
            or loaded.context.encoder_model_sha256 != self.encoder_initial_sha256
        ):
            raise ValueError("loaded encoder source identity mismatch")
        device = next(self.parameters()).device
        t = {k: torch.as_tensor(v, device=device) for k, v in loaded.tensors.items()}
        dope, lm48 = self.encoder._run_dope_mlc_v3_and_lm48(t["vec256"], t["col_lm"], use_landmarks=True)
        tokens, mask, _ = self.encoder._project_stage1b_tokens_batched_gpu(lm48, dope)
        if tokens.shape != (1, 10, 32) or not torch.equal(mask.bool(), t["mask_10"]):
            raise ValueError("live encoder token/mask contract mismatch")
        kwargs = {
            k: t[k]
            for k in (
                "ctx_feat",
                "scale",
                "base_prior_cost",
                "sos_prior",
                "revealed_indices",
                "revealed_raw_f",
                "revealed_gate",
            )
        }
        kwargs.update(
            prog_412=t["program"][None],
            proj_16=t["projection"][None],
            mm_tokens_10x32=t["tokens_10x32"],
            mm_mask_10=t["mask_10"],
        )
        with torch.no_grad():
            base = self.gamenet_anchor.predict_costs(**kwargs)["total_cost"]
        k = len(loaded.context.programs)
        live = (tokens * mask[..., None]).reshape(1, 1, 320).expand(1, k, 320)
        features = torch.cat([t["program"][None], t["projection"][None], live, base[..., None]], -1)
        scores = base + self.residual(features).squeeze(-1)
        if scores.shape != (1, k) or not torch.isfinite(scores).all():
            raise ValueError("student produced invalid costs")
        return scores[0]


class Learner:
    def __init__(self, model: SearchStudent):
        self.model = model
        self.optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=0.0001, weight_decay=0.0001
        )
        assert_research_optimizer_policy(self.optimizer, model, research_run_id=model.run_id)
        self.updates = 0
        self.before = {k: tensor_hash(v) for k, v in model.named_parameters()}
        self.gradient_squares = {"dope": 0.0, "residual": 0.0}
        self.witness = None
        self.scope_sha256 = None
        self.teacher_targets = {}
        self.examples = 0

    def state_hash(self):
        return state_hash(self.model)

    def bind_scope(self, loaded):
        if self.scope_sha256 is None:
            self.scope_sha256 = loaded.scope_sha256
        if loaded.scope_sha256 != self.scope_sha256:
            raise ValueError("learner role/exclusion scope changed across training/evaluation")

    def score(self, loaded):
        self.bind_scope(loaded)
        self.model.assert_frozen()
        self.model.eval()
        with torch.no_grad():
            return self.model(loaded).cpu().numpy().copy()

    def teacher_target(self, loaded, teacher_ref, store):
        key = (teacher_ref.bucket, teacher_ref.key, teacher_ref.version_id, teacher_ref.sha256)
        if key not in self.teacher_targets:
            teacher = json.loads(teacher_ref.read(store))
            if (
                teacher["schema"] != "h206-native-teacher-v1"
                or teacher["context_identity"] != loaded.context.identity
                or teacher["records_sha256"] != digest(teacher["records"])
            ):
                raise ValueError("learner teacher/context/label identity mismatch")
            identity = teacher["identity"]
            if (
                identity.get("context") != loaded.context.identity
                or identity.get("scope") != loaded.scope_sha256
                or identity.get("native_binary") != loaded.context.native_binary_sha256
                or identity.get("evaluator") != loaded.context.evaluator_sha256
            ):
                raise ValueError("learner native teacher full identity mismatch")
            records = teacher["records"]
            for record in records:
                if (
                    record["binary_sha256"] != loaded.context.native_binary_sha256
                    or record["original_train_sha256"] != loaded.context.train.sha256
                    or record["protocol_sha256"] != loaded.context.protocol_sha256
                    or record["environment_sha256"] != loaded.context.environment_sha256
                    or record["program_sha256"] != sha(loaded.context.programs[record["index"]].encode())
                ):
                    raise ValueError("learner native outcome/input binding mismatch")
            completed = [r for r in records if r["metrics"]["status"] == "completed"]
            if not completed:
                raise ValueError("native teacher has no completed candidate")
            selected = min(completed, key=lambda r: (r["metrics"]["fitness"], records.index(r)))["index"]
            target = teacher["target_index"]
            if (
                type(target) is not int
                or target != selected
                or not 0 <= target < len(loaded.context.programs)
            ):
                raise ValueError("native teacher target missing/invalid/disconnected")
            self.teacher_targets[key] = (loaded.context.identity, target)
        identity, target = self.teacher_targets[key]
        if identity != loaded.context.identity:
            raise ValueError("cached teacher/context identity mismatch")
        return target

    def update(self, loaded, teacher_ref, store):
        return self.update_batch([loaded], [teacher_ref], store)

    def update_batch(self, rows, teacher_refs, store):
        guard_cpu(Path(__file__).parent)
        if not rows or len(rows) != len(teacher_refs):
            raise ValueError("nonempty aligned training batch required")
        targets = []
        for loaded, teacher in zip(rows, teacher_refs, strict=True):
            self.bind_scope(loaded)
            loaded.verify()
            if loaded.context.role != "FIT":
                raise ValueError("held-out family cannot enter student update")
            targets.append(self.teacher_target(loaded, teacher, store))
        self.model.assert_frozen()
        if self.witness is None:
            self.witness = (rows[0], self.score(rows[0]))
        assert_research_optimizer_policy(self.optimizer, self.model, research_run_id=self.model.run_id)
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        loss_value = 0.0
        for loaded, target in zip(rows, targets, strict=True):
            scores = self.model(loaded)
            loss = torch.nn.functional.cross_entropy(
                -scores[None], torch.tensor([target], device=scores.device)
            ) / len(rows)
            if not torch.isfinite(loss):
                raise ValueError("nonfinite student loss")
            loss.backward()
            loss_value += float(loss.detach())
        for group, module in (("dope", self.model.dope_model), ("residual", self.model.residual)):
            grad = sum(
                float(p.grad.detach().square().sum()) for p in module.parameters() if p.grad is not None
            )
            if not np.isfinite(grad):
                raise ValueError("nonfinite research gradient")
            self.gradient_squares[group] += grad
        self.optimizer.step()
        self.updates += 1
        self.examples += len(rows)
        self.model.assert_frozen()
        return loss_value

    def proof(self):
        changed = [k for k, v in self.model.named_parameters() if tensor_hash(v) != self.before[k]]
        dope_ids = {id(p) for p in self.model.dope_model.parameters()}
        dope_names = {k for k, p in self.model.named_parameters() if id(p) in dope_ids}
        if (
            self.updates < 1
            or not (set(changed) & dope_names)
            or not any(k.startswith("residual.") for k in changed)
            or min(self.gradient_squares.values()) <= 0
            or self.witness is None
            or np.array_equal(self.witness[1], self.score(self.witness[0]))
        ):
            raise ValueError("zero updates/no intended DOPE+student parameter or gradient change")
        return {
            "updates": self.updates,
            "examples": self.examples,
            "changed_parameters": changed,
            "forward_changed": True,
            "gradient_squared_sums": self.gradient_squares,
            "state_sha256": self.state_hash(),
        }

    def round_snapshot(self, loaded):
        return {
            "parameters": {k: tensor_hash(p) for k, p in self.model.named_parameters()},
            "gradients": dict(self.gradient_squares),
            "updates": self.updates,
            "examples": self.examples,
            "forward": self.score(loaded),
        }

    def proof_since(self, snapshot, loaded):
        changed = [k for k, p in self.model.named_parameters() if tensor_hash(p) != snapshot["parameters"][k]]
        dope_ids = {id(p) for p in self.model.dope_model.parameters()}
        dope_names = {k for k, p in self.model.named_parameters() if id(p) in dope_ids}
        gradients = {k: self.gradient_squares[k] - snapshot["gradients"][k] for k in self.gradient_squares}
        if (
            self.updates <= snapshot["updates"]
            or not set(changed) & dope_names
            or not any(k.startswith("residual.") for k in changed)
            or min(gradients.values()) <= 0
            or np.array_equal(snapshot["forward"], self.score(loaded))
        ):
            raise ValueError("zero updates/no intended parameter or forward change in THIS round")
        return {
            "updates": self.updates,
            "updates_this_round": self.updates - snapshot["updates"],
            "examples_this_round": self.examples - snapshot["examples"],
            "changed_parameters": changed,
            "gradient_squared_sums": gradients,
            "forward_changed": True,
            "state_sha256": self.state_hash(),
        }
