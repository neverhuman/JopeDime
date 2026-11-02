"""Construct the original r882 Fused model from exact-version approved bytes."""

from __future__ import annotations

import contextlib
import importlib.metadata
import importlib.util
import io
import os
import sys
import threading
from pathlib import Path

import torch

from .contracts import ObjectRef, digest, guard_cpu, sha
from .student import Learner, SearchStudent, state_hash

_FACTORY_LOCK = threading.RLock()
_CODE_PREFIX = "research/flywheel_selfplay/custody/files"
_FUSED_PATH = _CODE_PREFIX + "/home/ubuntu/jope-data/res-01/input/cpu_exact_fused.py"
_MICRO = "/home/ubuntu/JopeDimeMicro/python/"
_ORDER = (
    ("rust_evaluator", "rust_evaluator.py"),
    ("model", "model.py"),
    ("phase12_multimodal_model", "phase12_multimodal_model.py"),
    ("research.levels", "research/levels/__init__.py"),
    ("research.levels.scaling", "research/levels/scaling.py"),
    ("research.levels.budget_planner", "research/levels/budget_planner.py"),
    ("research.levels.building_block_landmarks", "research/levels/building_block_landmarks.py"),
    ("research.levels.multiclass_head", "research/levels/multiclass_head.py"),
    ("research.levels.multilevel_model", "research/levels/multilevel_model.py"),
    ("_h206_original_fused", None),
)


def environment_identity():
    """Observed host package identity, never the historical version declaration."""
    return {
        "python": sys.version,
        "packages": {name: importlib.metadata.version(name) for name in ("numpy", "torch", "psutil")},
        "torch_cuda_build": torch.version.cuda,
        "torch_default_dtype": str(torch.get_default_dtype()),
        "torch_intraop_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }


class ModelCache:
    def __init__(self, root, repo, *, cap_bytes=40_000_000_000):
        self.root, self.repo = Path(root).resolve(), Path(repo).resolve()
        if self.root == Path("/tmp") or Path("/tmp") in self.root.parents:
            raise ValueError("model cache must be outside /tmp")
        if self.root == self.repo or self.repo in self.root.parents:
            raise ValueError("model cache must be outside checkout")
        self.root.mkdir(parents=True, exist_ok=True)
        self.cap = cap_bytes

    def fetch(self, ref: ObjectRef, store):
        guard_cpu(self.root)
        path = self.root / ref.sha256
        provenance = self.root / (ref.sha256 + "." + digest(ref.__dict__) + ".ref.json")
        if path.exists():
            if sha(path.read_bytes()) != ref.sha256:
                raise ValueError("model cache poison")
            if provenance.exists():
                import json

                if json.loads(provenance.read_text()) != ref.__dict__:
                    raise ValueError("model cache reference provenance poison")
                return path
        body = ref.read(store)  # A new bucket/key/VersionId is always actually verified.
        size = sum(p.stat().st_size for p in self.root.rglob("*") if p.is_file())
        if size + len(body) > self.cap:
            raise RuntimeError("model cache cap")
        if not path.exists():
            with path.open("xb") as out:
                out.write(body)
                out.flush()
                os.fsync(out.fileno())
        with provenance.open("xb") as out:
            from .contracts import canonical

            out.write(canonical(ref.__dict__))
            out.flush()
            os.fsync(out.fileno())
        return path

    def asset_layout(self, refs, store):
        layout = self.root / ("assets-" + digest({k: v.sha256 for k, v in refs.items()}))
        for name, ref in refs.items():
            cached = self.fetch(ref, store)
            subdir = "projections" if name.startswith("frozen_stage") else "models"
            target = layout / "assets" / subdir / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                os.link(cached, target)
            if sha(target.read_bytes()) != ref.sha256:
                raise ValueError("model asset alias poison")
        return layout


@contextlib.contextmanager
def pinned_original_modules(repo, source_pins):
    """Load verified original source; restore ambient imports and sys.path."""
    paths = {}
    for name, relative in _ORDER:
        path = (
            Path(repo) / _FUSED_PATH
            if relative is None
            else (Path(repo) / _CODE_PREFIX / (_MICRO + relative).lstrip("/"))
        )
        expected = source_pins.get(str(path.relative_to(repo)))
        if expected is None or sha(path.read_bytes()) != expected:
            raise ValueError("unbound or changed model source: " + str(path))
        paths[name] = path
    fused_path = paths["_h206_original_fused"]
    # These optional branches change the original constructor and are not in this pin.
    for name in ("expanded_gpu_bank_1766.pt", "distilled_stage1a_bridge_real_s3.pt"):
        if (fused_path.parent / name).exists():
            raise ValueError("unadmitted optional model asset: " + name)
    absent = object()
    saved = {name: sys.modules.get(name, absent) for name in paths}
    old_path = list(sys.path)
    try:
        for name, path in paths.items():
            spec = importlib.util.spec_from_file_location(
                name,
                path,
                submodule_search_locations=[str(path.parent)] if name == "research.levels" else None,
            )
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
        yield sys.modules["_h206_original_fused"]
    finally:
        sys.path[:] = old_path
        for name, value in saved.items():
            if value is absent:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


class PinnedR882Factory:
    def __init__(self, repo, bindings, *, bindings_sha256, source_pins, store, cache):
        if sha(bindings) != bindings_sha256:
            raise ValueError("model binding SHA mismatch")
        import json

        self.binding = json.loads(bindings)
        self.repo, self.source_pins = Path(repo).resolve(), source_pins
        self.store, self.cache = store, cache
        self.checkpoint = ObjectRef(**self.binding["initial_checkpoint"])
        self.assets = {k: ObjectRef(**v) for k, v in self.binding["parameter_assets"].items()}
        self.receipt = None

    def construct(self):
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
            raise RuntimeError("CPU model qualification requires CUDA hidden")
        return self._construct()

    def _construct(self):
        before = guard_cpu(self.cache.root)
        checkpoint_path = self.cache.fetch(self.checkpoint, self.store)
        checkpoint = torch.load(
            io.BytesIO(checkpoint_path.read_bytes()), map_location="cpu", weights_only=True
        )
        layout = self.cache.asset_layout(self.assets, self.store)
        loaded_paths = []
        original_load = torch.load
        approved = {
            (
                layout / "assets" / ("projections" if n.startswith("frozen_stage") else "models") / n
            ).resolve(): r
            for n, r in self.assets.items()
        }

        def checked_load(path, *args, **kwargs):
            resolved = Path(path).resolve()
            if resolved not in approved or sha(resolved.read_bytes()) != approved[resolved].sha256:
                raise ValueError("constructor attempted unapproved model payload")
            loaded_paths.append(str(resolved))
            # These exact-hash, original model parameter assets contain NumPy
            # scaler/SVD objects; retain the original constructor's decoder.
            return original_load(path, *args, **kwargs)

        with _FACTORY_LOCK, pinned_original_modules(self.repo, self.source_pins) as inherited:
            torch.load = checked_load
            try:
                fused = inherited.FusedJopeNetwork(
                    micro_root=layout, adapter_variant="h40b", device=torch.device("cpu")
                )
            finally:
                torch.load = original_load
            fused.load_state_dict(checkpoint["state_dict"], strict=True)
        for parameter in fused.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
        fused.eval()
        expected = self.binding["checkpoint_state"]
        actual = {
            "checkpoint_tensor_map_sha256": state_hash(fused),
            "anchor_tensor_map_sha256": state_hash(fused.adapter_10tok),
            "dope_tensor_map_sha256": state_hash(fused.dope_model),
        }
        if any(actual[k] != expected[k] for k in actual):
            raise ValueError("actual loaded r882 model/state mismatch")
        if len(set(loaded_paths)) != 4 or set(map(Path, loaded_paths)) != set(approved):
            raise ValueError("incomplete actual constructor asset reads")
        after = guard_cpu(self.cache.root)
        self.receipt = {
            "schema": "h206-actual-r882-factory-v1",
            "checkpoint": self.checkpoint.__dict__,
            "loaded_state": actual,
            "actual_constructor": "original pinned FusedJopeNetwork.__init__",
            "strict_state_load": True,
            "parameter_asset_reads": loaded_paths,
            "source_pins": self.source_pins,
            "environment": environment_identity(),
            "resource_before": before,
            "resource_after": after,
            "all_initial_parameters_frozen": all(not p.requires_grad for p in fused.parameters()),
            "gpu_or_scientific_forward": False,
        }
        return fused

    def learner(self, *, seed, run_id):
        return self._learner(self.construct(), seed=seed, run_id=run_id)

    def learner_for_phase(self, permit, *, seed, run_id):
        from .supervisor import GPUWorkerPermit

        if not isinstance(permit, GPUWorkerPermit):
            raise ValueError("actual admitted GPU phase permit required")
        permit.validate()
        learner = self._learner(self._construct(), seed=seed, run_id=run_id)
        learner.model.to(permit.device)
        learner.model.assert_frozen()
        self.receipt["gpu_phase"] = {
            "device": str(permit.device),
            "reservation": permit.token,
            "state_after_device_copy": learner.state_hash(),
        }
        return learner

    def _learner(self, fused, *, seed, run_id):
        metadata = {"entrypoint": "research", "research_run_id": run_id}
        state = self.receipt["loaded_state"]
        learner = Learner(
            SearchStudent(
                fused,
                seed=seed,
                run_id=run_id,
                launch_metadata=metadata,
                encoder_state_sha256=state["checkpoint_tensor_map_sha256"],
                anchor_state_sha256=state["anchor_tensor_map_sha256"],
                encoder_source_sha256=self.source_pins[_FUSED_PATH],
            )
        )
        learner.model.assert_frozen()
        trainable = {k: p.numel() for k, p in learner.model.named_parameters() if p.requires_grad}
        self.receipt["student"] = {
            "initial_state_sha256": learner.state_hash(),
            "seed": seed,
            "run_id": run_id,
            "trainable_parameters": trainable,
            "trainable_parameter_count": sum(trainable.values()),
            "residual_parameter_count": sum(p.numel() for p in learner.model.residual.parameters()),
            "dope_policy": metadata,
            "optimizer_updates": 0,
        }
        return learner


def native91_replay(rows, reference, *, score, expected_identities, persist):
    """Ordered singleton full-cost replay, not a mean-only or argmin gate.

    Caller admits exact role/body/tensor/source pins before giving rows to this
    function. score returns the original scorer's entire candidate-cost vector,
    selected DSL and native selected utility; persist is durable/create-only.
    """
    import json

    import numpy as np

    ref = json.loads(reference)
    contract = ref["evaluation_contract"]["contexts"]
    if len(rows) != 91 or [r["cid"] for r in rows] != [r["cid"] for r in contract]:
        raise ValueError("native91 ordered batch mismatch")
    records = []
    for i, (row, pinned) in enumerate(zip(rows, contract, strict=True)):
        if any(row[k] != pinned[k] for k in ("cid", "pool_sha256", "training_table_sha256")):
            raise ValueError("native91 approved input identity mismatch")
        result = score(row)  # Exactly one context; no cross-context batching.
        if result["identities"] != expected_identities:
            raise ValueError("native91 scorer/environment/model identity mismatch")
        actual = np.asarray(result["costs"])
        expected = np.asarray(row["reference_costs"])
        finite = bool(np.isfinite(actual).all())
        record = {
            "index": i,
            "cid": row["cid"],
            **result,
            "costs": actual.tolist() if finite else None,
            "cost_dtype": actual.dtype.str,
            "cost_shape": list(actual.shape),
            "cost_bytes_hex": actual.tobytes().hex(),
        }
        records.append(record)
        persist(record)  # Persist each available vector before any next forward.
        if (
            actual.dtype != np.dtype("float32")
            or expected.dtype != np.dtype("float32")
            or actual.shape != expected.shape
            or not finite
            or not np.array_equal(actual, expected)
            or result["selected_program"] != ref["utility"]["selected_programs"][i]
            or result["utility"] != ref["utility"]["per_context"][i]
        ):
            raise ValueError("native91 exact full-vector/pick/utility mismatch")
    values = np.asarray([r["utility"] for r in records], dtype=np.float64)
    if float(values.mean()) != 1.1377941898164525 or ref["utility"]["mean"] != 1.1377941898164525:
        raise ValueError("native91 exact context-mean mismatch")
    return {"status": "PASS", "contexts": 91, "identities": expected_identities, "records": records}


def qualify_historical_native91(factory, manifest, *, manifest_sha256, persist):
    """Actual original r882 singleton scorer on the approved historical91 only.

    This compatibility gate consumes prepared program/fitness/token artifacts,
    never raw outer/test tables or any H205 producer packet. It cannot admit
    scientific training inputs, release a budget, or allocate a GPU.
    """
    import json

    import numpy as np

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("native91 CPU qualification requires CUDA hidden")
    if sha(manifest) != manifest_sha256:
        raise ValueError("native91 input manifest SHA mismatch")
    doc = json.loads(manifest)
    if doc["schema"] != "h206-approved-native91-compat-inputs-v1":
        raise ValueError("named historical native91 compatibility inputs required")
    if factory.checkpoint.sha256 != "0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64":
        raise ValueError("original r882 model required, no override")
    history = json.loads(
        (factory.repo / "research/flywheel_selfplay/h170_onecycle/manifest.json").read_text()
    )
    named = {
        "selected": "/home/ubuntu/jope-data/h50/input/selected.json",
        "source_manifest": "/home/ubuntu/jope-data/h50/input/source_manifest.json",
        "feature_cache": "/home/ubuntu/jope-data/h58/cache/context_features.npz",
    }
    payloads = {}
    for name, original in named.items():
        ref = ObjectRef(**doc[name])
        if ref.sha256 != history["inputs"][original]:
            raise ValueError("native91 historical input identity changed")
        payloads[name] = ref.read(factory.store)
    selected = json.loads(payloads["selected"])
    source = json.loads(payloads["source_manifest"])
    if (
        source["outer_rows_delivered"] != 0
        or source["excluded_payloads_opened"] != 0
        or source["excluded_dl_holdout_count"] != 93
        or len(selected["train"]) != 1219
        or len(selected["dev"]) != 91
        or {r["family_id"] for r in selected["train"]} & {r["family_id"] for r in selected["dev"]}
    ):
        raise ValueError("historical unprotected baseline closure mismatch")
    reference_ref = ObjectRef(**doc["reference"])
    if dataclass_dict(reference_ref) != factory.binding["native91_reference"]["object_ref"]:
        raise ValueError("native91 original reference changed")
    reference = reference_ref.read(factory.store)
    ref_doc = json.loads(reference)
    contract = ref_doc["evaluation_contract"]["contexts"]
    metadata = {r["context_sha256"]: r for r in selected["dev"]}
    if set(metadata) != {r["cid"] for r in contract} or set(doc["pools"]) != set(metadata):
        raise ValueError("native91 exact permitted roster mismatch")
    costs_ref = ObjectRef(**doc["reference_costs"])
    if costs_ref.sha256 != "a5109aff14e9aed679752f9884059adf9ce580f1a2d266c2d0a88fe42194ee38":
        raise ValueError("original full-cost reference required; no replacement")
    with np.load(io.BytesIO(costs_ref.read(factory.store)), allow_pickle=False) as z:
        costs = {k: z[k].copy() for k in z.files}
    if set(costs) != set(metadata):
        raise ValueError("native91 full-cost vector reference roster mismatch")
    all_allowed = {r["context_sha256"] for r in selected["train"] + selected["dev"]}
    with np.load(io.BytesIO(payloads["feature_cache"]), allow_pickle=False) as z:
        cids = list(map(str, z["cids"]))
        if len(cids) != 1310 or set(cids) != all_allowed:
            raise ValueError("historical cache contains unapproved contexts")
        tokens = z["tokens_10x32"]
        masks = z["mask_10"]
        features = {cid: (tokens[cids.index(cid)].copy(), masks[cids.index(cid)].copy()) for cid in metadata}
    rows, pools, program_orders = [], {}, {}
    for original in contract:
        cid = original["cid"]
        meta = metadata[cid]
        ref = ObjectRef(**doc["pools"][cid])
        if (
            ref.sha256 != original["pool_sha256"]
            or meta["pool_sha256"] != ref.sha256
            or meta["train_only_jtt1_sha256"] != original["training_table_sha256"]
            or meta["dl_split"] != "DL-Val"
            or meta["source_split"] not in ("train", "inner")
        ):
            raise ValueError("native91 row/input role identity mismatch")
        with np.load(io.BytesIO(ref.read(factory.store)), allow_pickle=False) as z:
            pools[cid] = {k: z[k].copy() for k in ("program", "fitness", "gate", "program_sha256")}
        program_orders[cid] = sha(pools[cid]["program_sha256"].tobytes())
        rows.append({**original, "reference_costs": costs[cid]})
    fused = factory.construct()  # Actual original constructor + strict original r882 state.
    identities = {
        "model_state_sha256": state_hash(fused),
        "encoder_source_sha256": factory.source_pins[_FUSED_PATH],
        "scorer": "original pinned adapter_10tok.predict_costs",
        "scorer_source_sha256": history["assets"][
            "/home/ubuntu/JopeDimeMicro/python/phase12_multimodal_model.py"
        ],
        "batch_size": 1,
        "ordered_batch_sha256": digest(contract),
        "program_orders_sha256": digest(program_orders),
        "feature_cache_sha256": ObjectRef(**doc["feature_cache"]).sha256,
        "reference_sha256": reference_ref.sha256,
        "full_cost_reference_sha256": costs_ref.sha256,
        "environment": environment_identity(),
        "source": "historical TRAIN-only-prepared pools/tokens, no outer data",
    }
    projection = fused.w1_sign_matrix_412x16.detach().cpu().numpy().astype(np.float64) / np.sqrt(412.0)

    def score(row):
        guard_cpu(factory.cache.root)
        cid = row["cid"]
        pool = pools[cid]
        program = pool["program"].astype(np.float32)
        k = len(program)
        token, mask = features[cid]
        kwargs = {
            "ctx_feat": torch.zeros((1, 584)),
            "prog_412": torch.as_tensor(program)[None],
            "proj_16": torch.as_tensor((program.astype(np.float64) @ projection).astype(np.float32))[None],
            "base_prior_cost": torch.zeros((1, k)),
            "scale": torch.full((1,), 0.4208),
            "sos_prior": torch.zeros((1, k, 2)),
            "revealed_indices": torch.arange(4).reshape(1, 4),
            "revealed_raw_f": torch.as_tensor(pool["fitness"], dtype=torch.float32)[None, :4],
            "revealed_gate": torch.as_tensor(pool["gate"], dtype=torch.bool)[None, :4].float(),
            "mm_tokens_10x32": torch.as_tensor(token, dtype=torch.float32)[None],
            "mm_mask_10": torch.as_tensor(mask, dtype=torch.bool)[None],
        }
        with torch.no_grad():
            actual = fused.adapter_10tok.predict_costs(**kwargs)["total_cost"][0].float().numpy().copy()
        choice = int(np.argmin(actual))
        pid = pool["program_sha256"][choice]
        pid = pid.decode() if isinstance(pid, bytes) else str(pid)
        return {
            "identities": identities,
            "costs": actual,
            "selected_program": pid,
            "program_order_sha256": program_orders[cid],
            "utility": float(-pool["fitness"][choice]) if bool(pool["gate"][choice]) else -2.0,
        }

    return native91_replay(rows, reference, score=score, expected_identities=identities, persist=persist)


def dataclass_dict(value):
    import dataclasses

    return dataclasses.asdict(value)
