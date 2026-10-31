"""Small, hash-bound definition closure; no historical module/constructor imports."""

from __future__ import annotations

import ast
import hashlib
import math
import re
import struct
from pathlib import Path

import numpy as np
import torch

MICRO = "JopeDimeMicro/python/"
FUSED = "cpu_exact_fused.py"


class Reference:
    def __init__(self, custody, manifest, state, feature_names):
        self.custody = Path(custody)
        self.pins = {row["path"]: row for row in manifest["import_files"]}
        self.source_hashes = {}
        self.env = dict(
            np=np,
            torch=torch,
            nn=torch.nn,
            F=torch.nn.functional,
            math=math,
            re=re,
            struct=struct,
            hashlib=hashlib,
        )
        self.define(MICRO + "rank_prepare.py", ["features"])
        self.env.update(dsl_features_dict=self.env["features"], _FEAT_NAMES=feature_names, _W1_SIGN_MAT=None)
        self.define(
            MICRO + "rust_evaluator.py",
            ["get_feat_names", "get_w1_sign_matrix", "featurize_dsl_412", "select_landmark_preferred_pools"],
            ["STAGE_POOL_BY_REGIME", "HEAD_POOL_BY_REGIME"],
        )
        self.featurize = self.env["featurize_dsl_412"]
        self.preferred_pools = self.env["select_landmark_preferred_pools"]
        self.define(
            MICRO + "model.py", ["extract_head_and_stage_signatures", "SwiGLUResBlock", "FlywheelGameNet"]
        )
        self.define(
            MICRO + "phase12_multimodal_model.py",
            [
                "compute_semantic_struct_sig_24d_torch",
                "transform_program_features_torch",
                "FlywheelBlockFormerAdapterNet",
            ],
            ["OPERATOR_VOCAB"],
        )
        self.define(MICRO + "research/levels/multiclass_head.py", ["InterLandmarkAttentionBlock"])
        self.define(
            MICRO + "research/levels/multilevel_model.py",
            ["LandmarkHyperRouter", "ConditionalHyperTargetNet"],
            ["MAX_COLS", "_LEVEL_INDEX"],
        )
        methods = [
            "_run_dope_mlc_v3_and_lm48",
            "_compute_lm_smart_80d_gpu",
            "_svd_slot_project",
            "_project_stage1b_tokens_batched_gpu",
            "_install_sync_free_mha_wrappers",
        ]
        self.define(
            FUSED,
            ["Stage1ADistilledBridge", *["FusedJopeNetwork." + x for x in methods]],
            ["STEP3_GATE_KEYS_14"],
        )
        self.define("frozen-source/research/flywheel_selfplay/run_fused_closed_loop.py", [])
        # Only constructors reviewed to allocate nn layers are called. In particular,
        # FusedJopeNetwork.__init__, bank registration and all torch.load paths are absent.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            anchor = self.env["FlywheelGameNet"]()
            anchor.load_state_dict(self.prefix(state, "gamenet_anchor."), strict=True)
            self.adapter = self.env["FlywheelBlockFormerAdapterNet"](
                anchor, state["adapter_10tok.prog_mean"].numpy(), state["adapter_10tok.prog_std"].numpy()
            )
            self.adapter.load_state_dict(self.prefix(state, "adapter_10tok."), strict=True)
            router = self.env["LandmarkHyperRouter"]()
            router.load_state_dict(self.prefix(state, "dope_model.router."), strict=True)
            target = self.env["ConditionalHyperTargetNet"]()
            target.load_state_dict(self.prefix(state, "dope_model.target_net."), strict=True)
            bridge = self.env["Stage1ADistilledBridge"]()
            bridge.load_state_dict(self.prefix(state, "landmark_bridge."), strict=True)
        for module in (self.adapter, router, target, bridge):
            module.eval().requires_grad_(False)
        self.net = torch.nn.Module()
        self.net.dope_model = torch.nn.Module()
        self.net.dope_model.router = router
        self.net.dope_model.target_net = target
        self.net.adapter_10tok = self.adapter
        self.net.landmark_bridge = bridge
        self.net.step3_gate_keys_14 = self.env["STEP3_GATE_KEYS_14"]
        for name, value in state.items():
            if "." not in name:
                setattr(self.net, name, value)
        for name in methods:
            setattr(self.net, name, self.env[name].__get__(self.net))
        # Match the frozen constructor's owned-MHA wrappers, without calling it.
        torch.backends.mha.set_fastpath_enabled(False)
        self.net._install_sync_free_mha_wrappers()
        self.state = state

    @staticmethod
    def prefix(state, prefix):
        return {name[len(prefix) :]: value for name, value in state.items() if name.startswith(prefix)}

    def define(self, suffix, names, literals=()):
        candidates = [path for path in self.pins if path.endswith("/" + suffix)]
        if len(candidates) != 1:
            raise ValueError("reference source must have one pinned match: " + suffix)
        original = candidates[0]
        pin = self.pins[original]
        path = self.custody / "files" / original.lstrip("/")
        if path.is_symlink() or not path.is_file():
            raise ValueError("missing regular pinned reference source: " + original)
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != pin["sha256"]:
            raise ValueError("reference source hash differs: " + original)
        self.source_hashes[original] = pin["sha256"]
        tree = ast.parse(data, filename=original)
        definitions = {
            node.name: node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
        }
        for node in tree.body:
            target = (
                node.target
                if isinstance(node, ast.AnnAssign)
                else (node.targets[0] if isinstance(node, ast.Assign) and len(node.targets) == 1 else None)
            )
            if isinstance(target, ast.Name) and target.id in literals:
                self.env[target.id] = ast.literal_eval(node.value)
        if any(name not in self.env for name in literals):
            raise ValueError("required source literal absent")
        selected = []
        for name in names:
            if "." in name:
                owner, method = name.split(".")
                node = next(
                    n for n in definitions[owner].body if isinstance(n, ast.FunctionDef) and n.name == method
                )
            else:
                node = definitions[name]
            selected.append(node)
        future = ast.parse("from __future__ import annotations").body[0]
        module = ast.Module(body=[future, *selected], type_ignores=[])

        class BindPinnedHelper(ast.NodeTransformer):
            def visit_ImportFrom(self, node):
                if node.module == "phase12_multimodal_model":
                    if [(item.name, item.asname) for item in node.names] != [
                        ("transform_program_features_torch", None)
                    ]:
                        raise ValueError("unexpected reference helper import")
                    return None  # Already bound above from the separately hash-verified source.
                return node

        module = ast.fix_missing_locations(BindPinnedHelper().visit(module))
        exec(compile(module, original, "exec"), self.env)

    def intermediates(self, vec256, col):
        with torch.inference_mode():
            v_array = np.asarray(vec256)
            c_array = np.asarray(col)
            if v_array.shape not in ((256,), (1, 256)) or c_array.shape not in ((64, 17), (1, 64, 17)):
                raise ValueError("exact landmark output shapes required")
            if not np.isfinite(v_array).all() or not np.isfinite(c_array).all():
                raise ValueError("finite landmark outputs required")
            v = torch.from_numpy(v_array.astype(np.float32)).reshape(1, 256)
            c = torch.from_numpy(c_array.astype(np.float32)).reshape(1, 64, 17)
            dope, lm48 = self.net._run_dope_mlc_v3_and_lm48(v, c, use_landmarks=True)
            tokens, mask, standardized = self.net._project_stage1b_tokens_batched_gpu(lm48, dope)
        return dict(v256=v, col=c, dope629=dope, lm48=lm48, tokens=tokens, mask=mask, lm48_z=standardized)

    def costs(self, intermediates, features, projections, revealed_f, revealed_gate):
        with torch.inference_mode():
            prog = torch.from_numpy(np.asarray(features, dtype=np.float32)).unsqueeze(0)
            proj = torch.from_numpy(np.asarray(projections, dtype=np.float32)).unsqueeze(0)
            # Frozen run_fused_closed_loop.py predict() uses zero base/SOS priors;
            # the separate GPU-fitter calibrated-prior route is not this h40b loop.
            prior = torch.zeros(prog.shape[:2], dtype=torch.float32)
            sos = torch.zeros((*prog.shape[:2], 2), dtype=torch.float32)
            inputs = dict(
                ctx_feat=torch.zeros((1, 584)),
                prog_412=prog,
                proj_16=proj,
                base_prior_cost=prior,
                scale=torch.tensor([0.4208]),
                sos_prior=sos,
                mm_tokens_10x32=intermediates["tokens"],
                mm_mask_10=intermediates["mask"],
                revealed_indices=torch.tensor([[0, 1, 2, 3]], dtype=torch.int64),
                revealed_raw_f=torch.tensor([revealed_f], dtype=torch.float32),
                revealed_gate=torch.tensor([revealed_gate], dtype=torch.float32),
            )
            outputs = self.adapter.predict_costs(**inputs)
        return {
            **{"input." + k: v for k, v in inputs.items()},
            **{"output." + k: v for k, v in outputs.items()},
        }
