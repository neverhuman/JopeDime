"""CPU-only prereg/skeleton controls; no corpus, candidate forward or GPU probes."""

from __future__ import annotations

import ast
import gzip
import hashlib
import importlib.util
import io
import json
import struct
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from research.flywheel_selfplay.h206_search_distill.contracts import (
    ArraySpec,
    Context,
    ObjectRef,
    Scope,
    array_hash,
    canonical,
    digest,
    load_context,
    sha,
)
from research.flywheel_selfplay.h206_search_distill.executor import (
    BudgetLedger,
    evaluate_fixed,
    run_round,
)
from research.flywheel_selfplay.h206_search_distill.native import (
    OutcomeCache,
    RustS1Evaluator,
    S3Store,
    collect_teacher,
)
from research.flywheel_selfplay.h206_search_distill.student import (
    Learner,
    SearchStudent,
    state_hash,
)

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("h206_prereg_runner", HERE / "runner.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def report():
    return {
        "pid": 100,
        "environment": {k: "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
        "torch_intraop": 1,
        "native_pools": [],
    }


def admitted_fixture():
    return {
        "prereg_pr_open": True,
        "queue_release_received": True,
        "executor_reviewed": True,
        "immutable_pins_bound": True,
        "host": "xbabe3",
        "gpu": "NVIDIA GeForce RTX 3090",
        "gpu_claimed": False,
        "nice": 19,
        "cpu_percent": 10,
        "disk_total_bytes": 1_000_000_000_000,
        "disk_used_bytes": 880_000_000_000,
        "memory_available_bytes": 32 * 2**30,
        "rss_bytes": 1_000_000_000,
        "os_threads": 64,
    }


def test_plan_authority_policy_and_unmeasured_state():
    plan = runner.load_plan()
    assert plan["authority"] in (HERE / "PREREG.md").read_text()
    assert plan["control_k"] == 1 and plan["terminal_updates"] == 1800
    assert len(plan["arm_slots"]) == 4 and plan["gpu_hours_cap"] == 18
    assert plan["dope_policy"]["status"] == "PLANNED_NOT_APPLIED"
    metrics = json.loads((HERE / "metrics.json").read_text())
    assert metrics["verdict"] is None and metrics["native_parity"] is None
    assert metrics["scientific_optimizer_updates"] == 0


def test_cpu_plan_exclusive_output_and_run_identity(tmp_path):
    output = tmp_path / "launch.json"
    assert runner.main(["--plan", "--run-id", "H206-unit", "--output", str(output)]) == 0
    plan = json.loads(output.read_text())
    assert plan["authority"] == runner.AUTHORITY
    assert plan["dope_policy"]["research_run_id"] == plan["research_run_id"] == "H206-unit"
    assert plan["runtime_admission"] == "NOT_PERFORMED" and not plan["queue_released"]
    with pytest.raises(FileExistsError):
        runner.main(["--output", str(output)])
    with pytest.raises(SystemExit):
        runner.main(["--run-id", "../invalid"])


def test_even_admitted_fixture_cannot_execute_or_write_launch(tmp_path):
    assert runner.future_admission_reasons(admitted_fixture(), [report()]) == []
    output = tmp_path / "launch.json"
    with pytest.raises(SystemExit) as exc:
        runner.main(["--execute", "--output", str(output)])
    assert exc.value.code == 2 and not output.exists()
    assert runner.EXECUTOR_IMPLEMENTED is True
    assert runner.SCIENTIFIC_EXECUTION_ENABLED is False


@pytest.mark.parametrize(
    "key", ["prereg_pr_open", "queue_release_received", "executor_reviewed", "immutable_pins_bound"]
)
def test_missing_queue_and_source_prerequisites_refuse(key):
    snapshot = admitted_fixture()
    snapshot[key] = False
    assert any(key in v for v in runner.future_admission_reasons(snapshot, [report()]))


@pytest.mark.parametrize("used", [880_000_000_001, 900_000_000_000, 938_000_000_000])
def test_disk_88_gate_is_stricter_than_running_guard(used):
    snapshot = admitted_fixture()
    snapshot["disk_used_bytes"] = used
    assert any("pre-reservation disk" in v for v in runner.future_admission_reasons(snapshot, [report()]))


def test_reserve_cache_and_fifty_gb_headroom_independently_required():
    snapshot = admitted_fixture()
    snapshot.update(disk_total_bytes=500_000_000_000, disk_used_bytes=440_000_000_000)
    assert any("post-cache" in v for v in runner.future_admission_reasons(snapshot, [report()]))
    snapshot.update(disk_total_bytes=400_000_000_000, disk_used_bytes=352_000_000_000)
    assert any("headroom" in v for v in runner.future_admission_reasons(snapshot, [report()]))


@pytest.mark.parametrize(
    "change",
    [
        {"host": "xbabe1"},
        {"gpu_claimed": True},
        {"nice": 0},
        {"cpu_percent": 85.1},
        {"cpu_percent": float("nan")},
        {"disk_total_bytes": None},
        {"memory_available_bytes": 0},
        {"rss_bytes": 16 * 2**30},
    ],
)
def test_host_ownership_and_other_resource_limits(change):
    snapshot = admitted_fixture()
    snapshot.update(change)
    assert runner.future_admission_reasons(snapshot, [report()])


def test_driver_overhead_allowed_but_numeric_excess_and_missing_reports_refuse():
    assert runner.future_admission_reasons(admitted_fixture(), [report()]) == []
    bad = report()
    bad["torch_intraop"] = 5
    assert any("numeric" in v for v in runner.future_admission_reasons(admitted_fixture(), [bad]))
    assert any("numeric" in v for v in runner.future_admission_reasons(admitted_fixture(), []))


def test_research_optimizer_cannot_include_frozen_anchor(monkeypatch):
    # Persist #1726's independent review regression control for this research path.
    import torch
    from jope.north_star.dope_freeze_guard import (
        DopeFreezeViolationError,
        assert_research_optimizer_policy,
        configure_research_dope,
    )

    monkeypatch.delenv("JOPE_RESEARCH_DOPE_UNFREEZE", raising=False)
    net = torch.nn.Module()
    net.dope_model = torch.nn.Linear(2, 2)
    net.gamenet_anchor = torch.nn.Linear(2, 2)
    configure_research_dope(net, launch_metadata={}, unfreeze=True, research_run_id="H206-unit")
    opt = torch.optim.SGD(
        [
            {"params": net.dope_model.parameters()},
            {"params": net.gamenet_anchor.parameters()},
        ],
        lr=0.01,
    )
    with pytest.raises(DopeFreezeViolationError, match="includes the frozen anchor"):
        assert_research_optimizer_policy(opt, net, research_run_id="H206-unit")
    assert all(not p.requires_grad for p in net.gamenet_anchor.parameters())


def test_manifest_hashes_bind_all_prereg_sources():

    manifest = json.loads((HERE / "manifest.json").read_text())
    assert manifest["status"] == "QUEUED_NOT_RUN" and manifest["evidence"] == []
    for name, expected in manifest["source_sha256"].items():
        assert hashlib.sha256((HERE / name).read_bytes()).hexdigest() == expected, name


# Real H206 transport/native/learner integration; fixtures never qualify corpus science.


ROOT = HERE.parents[2]
FUSED_SOURCE = (
    ROOT / "research/flywheel_selfplay/custody/files/home/ubuntu/jope-data/res-01/input/cpu_exact_fused.py"
)
METRICS_SOURCE = ROOT / "research/flywheel_selfplay/rust_evaluator.py"


class _VersionedFixtureS3:
    def __init__(self):
        self.objects = {}
        self.trace = []
        self.corrupt = False

    def put_object(self, *, Bucket, Key, Body, IfNoneMatch):
        assert IfNoneMatch == "*"
        if any(b == Bucket and k == Key for b, k, _ in self.objects):
            raise FileExistsError("create-only collision")
        version = str(len(self.objects) + 1)
        self.objects[Bucket, Key, version] = bytes(Body)
        self.trace.append(("put", Key))
        return {"VersionId": version}

    def get_object(self, *, Bucket, Key, VersionId):
        self.trace.append(("get", Key))
        body = self.objects[Bucket, Key, VersionId]
        if self.corrupt and Key.endswith("/teacher.json"):
            body += b"corruption"
        return {"Body": io.BytesIO(body), "VersionId": VersionId}


class _FixtureRouter(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(256, 309)

    def forward(self, x, *, level):
        assert level == "high"
        y = self.linear(x)
        return {
            "task_latent": y[:, :128],
            "dist_latent": y[:, 128:256],
            "gates": {f"g{i}": y[:, 256 + i] for i in range(14)},
            "landmark_pred": y[:, 270:293],
            "bb_pred": y[:, 293:309],
        }


class _FixtureAnchor(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.01))

    def predict_costs(self, **kwargs):
        return {"total_cost": kwargs["prog_412"][..., 0] * self.weight}


def inherited_fused():
    # Execute the ORIGINAL tracked Fused methods, not a copied MockInheritedEngine.
    # Skip its asset-loading constructor; all state below is explicitly fixture-only.
    tree = ast.parse(FUSED_SOURCE.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "FusedJopeNetwork")
    wanted = {
        "_run_dope_mlc_v3_and_lm48",
        "_compute_lm_smart_80d_gpu",
        "_svd_slot_project",
        "_project_stage1b_tokens_batched_gpu",
    }
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    assert {n.name for n in methods} == wanted
    selected = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            ast.ClassDef(
                name="InheritedFused",
                bases=[ast.Attribute(value=ast.Name(id="nn", ctx=ast.Load()), attr="Module", ctx=ast.Load())],
                keywords=[],
                body=methods,
                decorator_list=[],
            ),
        ],
        type_ignores=[],
    )
    namespace = {"torch": torch, "nn": nn, "np": np}
    exec(compile(ast.fix_missing_locations(selected), str(FUSED_SOURCE), "exec"), namespace)
    net = namespace["InheritedFused"]()
    net.dope_model = nn.Module()
    net.dope_model.router = _FixtureRouter()
    net.dope_model.target_net = nn.Module()
    net.dope_model.target_net.col_gate_head = nn.Linear(145, 1)
    net.adapter_10tok = _FixtureAnchor()
    net.step3_gate_keys_14 = [f"g{i}" for i in range(14)]
    values = {
        "step2_lm_mean_1x256": torch.zeros(1, 256),
        "step2_lm_std_1x256": torch.ones(1, 256),
        "step2_svd_basis_256x48": torch.randn(256, 48) * 0.01,
        "lm80_mu": torch.zeros(1, 80),
        "lm80_sd": torch.ones(1, 80),
        "mu_lm48": torch.zeros(1, 48),
        "sd_lm48": torch.ones(1, 48),
        "mu_bb20": torch.zeros(1, 20),
        "sd_bb20": torch.ones(1, 20),
        "mu_dm24": torch.zeros(1, 24),
        "sd_dm24": torch.ones(1, 24),
        "mask_10_base": torch.ones(1, 10, dtype=torch.bool),
    }
    for name, value in values.items():
        net.register_buffer(name, value)
    for slot, dimension in [
        ("svd_slot0_m0", 184),
        ("svd_slot1_m1", 48),
        ("svd_slot2_m2", 629),
        ("svd_slot4_d629_mid", 629),
        ("svd_slot5_d629_tail", 629),
        ("svd_slot7_lm_fine", 184),
        ("svd_slot9_kron", 64),
    ]:
        for name, value in {
            "mu": torch.zeros(1, dimension),
            "sd_safe": torch.ones(1, dimension),
            "active_mask": torch.ones(1, dimension),
            "effective_proj_mat": torch.randn(dimension, 32) * 0.01,
        }.items():
            net.register_buffer(slot + "_" + name, value)
    return net


def fixture_ledger(path, used_cpu=0):
    path.write_bytes(
        canonical(
            {
                "campaign_id": "H206",
                "accounting_status": "RECONCILED",
                "purpose": "controlled data-free fixture, not the scientific campaign ledger",
                "cpu_cap_seconds": 400 * 3600,
                "gpu_cap_seconds": 18 * 3600,
                "cpu_used_seconds": used_cpu,
                "gpu_used_seconds": 0,
                "reservations": {},
                "entries": [],
                "stages": [],
            }
        )
    )
    return BudgetLedger(path)


@pytest.fixture
def integrated(tmp_path):
    torch.manual_seed(206101)
    fused = inherited_fused()
    metadata = {}
    learner = Learner(
        SearchStudent(
            fused,
            seed=206101,
            run_id="H206-controlled-fixture",
            launch_metadata=metadata,
            encoder_state_sha256=state_hash(fused),
            anchor_state_sha256=state_hash(fused.adapter_10tok),
            encoder_source_sha256=sha(FUSED_SOURCE.read_bytes()),
        )
    )
    initial_encoder_hash = learner.model.encoder_initial_sha256
    client = _VersionedFixtureS3()
    store = S3Store(client, "fixture-only", "H206-controlled")
    binary = tmp_path / "controlled-native"
    binary.write_text("""#!/usr/bin/python3
import sys, json, struct
from pathlib import Path
assert sys.argv[1] == "--fit-s1"
body=Path(sys.argv[2]).read_bytes()
assert body[:4] == b"JTT1" and struct.unpack_from("<I",body,8)[0] == 10
nt,nq,p=struct.unpack_from("<III",body,4)
import array
values=array.array("d",body[17+nt:])
assert all(values[(nt+i)*(p+1)+p] == 0 for i in range(10))
assert all(values[(nt+i)*(p+1)+j] == values[(i%nt)*(p+1)+j] for i in range(10) for j in range(p))
candidate=json.loads(Path(sys.argv[3]).read_bytes())
if "FAIL" in candidate["dsl"]: raise SystemExit(7)
ratio=0.4+(int(candidate["program_sha256"][:4],16)%100)/1000
Path(sys.argv[4]).write_text(json.dumps({"inner_cv":[{"metrics":{"rmse":ratio},"baseline_metrics":{"rmse":1.0}}]*2}))
""")
    binary.chmod(0o755)
    native = RustS1Evaluator(
        binary,
        binary_sha256=sha(binary.read_bytes()),
        evaluator_sha256=sha(METRICS_SOURCE.read_bytes()),
        scratch=tmp_path / "native",
    )
    cache = OutcomeCache(tmp_path / "cache")
    ledger = fixture_ledger(tmp_path / "ledger.json")

    def acquire(round_index, actor_hash, *, role="FIT", protected=False, bad_source=False):
        rng = np.random.default_rng(206101)
        vec = torch.tensor(rng.normal(size=(1, 256)), dtype=torch.float32)
        col = torch.tensor(rng.normal(size=(1, 64, 17)), dtype=torch.float32)
        with torch.no_grad():
            dope, lm = fused._run_dope_mlc_v3_and_lm48(vec, col, use_landmarks=True)
            tokens, mask, _ = fused._project_stage1b_tokens_batched_gpu(lm, dope)
        arrays = {
            "vec256": vec.numpy(),
            "col_lm": col.numpy(),
            "tokens_10x32": tokens.numpy(),
            "mask_10": mask.numpy(),
            "program": rng.normal(size=(256, 412)).astype("float32"),
            "projection": rng.normal(size=(256, 16)).astype("float32"),
            "ctx_feat": np.zeros((1, 584), "float32"),
            "scale": np.array([0.4208], "float32"),
            "base_prior_cost": np.zeros((1, 256), "float32"),
            "sos_prior": np.zeros((1, 256, 2), "float32"),
            "revealed_indices": np.array([[0, 1, 2, 3]], "int64"),
            "revealed_raw_f": np.zeros((1, 4), "float32"),
            "revealed_gate": np.ones((1, 4), "float32"),
        }
        folds = bytes([0, 1] * 32)
        raw = (
            struct.pack("<4sIIIB", b"JTT1", 64, 0, 2, 2)
            + folds
            + (np.arange(64 * 3).reshape(64, 3) / 100).astype("<f8").tobytes()
        )
        ref = store.put(f"input/{role}-{round_index}/train.jtt1.gz", gzip.compress(raw, mtime=0))
        buf = io.BytesIO()
        np.savez(buf, **arrays)
        tensor_ref = store.put(f"input/{role}-{round_index}/tensors.npz", buf.getvalue())
        context = Context(
            context_id=digest({"fixture": "new-unseen-CID", "role": role}),
            dataset_id="fixture-dataset-" + role,
            family_id="fixture-family-" + role,
            task="regression",
            role=role,
            train=ref,
            cut_sha256=sha(folds),
            tensors=tensor_ref,
            arrays={k: ArraySpec(v.shape, v.dtype.str, array_hash(v)) for k, v in arrays.items()},
            programs=tuple(f"(pipe X (standardize) (ridge {i + 1}.0))" for i in range(256)),
            seed=206101 + round_index,
            actor_sha256=actor_hash,
            proposal_source_sha256=digest({"fixture": "proposal"}),
            evaluator_sha256=sha(METRICS_SOURCE.read_bytes()),
            native_binary_sha256=native.binary_sha256,
            environment_sha256=native.environment_sha256,
            protocol_sha256=native.protocol_sha256,
            encoder_sha256=sha(FUSED_SOURCE.read_bytes()),
            encoder_model_sha256=initial_encoder_hash,
        )
        data = asdict(context)
        fields = ("context_id", "dataset_id", "family_id", "task", "role", "train", "cut_sha256")
        sources = {
            k: data[k]
            for k in (
                "proposal_source_sha256",
                "evaluator_sha256",
                "native_binary_sha256",
                "environment_sha256",
                "protocol_sha256",
                "encoder_sha256",
                "encoder_model_sha256",
            )
        }
        if bad_source:
            sources["evaluator_sha256"] = "f" * 64
        roles = canonical(
            {
                "schema": "h206-role-contract-v1",
                "reference_families": ["fixture-family-FIT"],
                "contexts": {context.context_id: {k: data[k] for k in fields}},
                "sources": sources,
            }
        )
        exclusion = canonical(
            {
                "schema": "h206-exclusion-contract-v1",
                "links": [
                    ["dataset:" + context.dataset_id, "family:" + context.family_id],
                    ["family:" + context.family_id, "family:alias-" + role],
                ],
                "protected": ["family:alias-" + role] if protected else [],
            }
        )
        scope = Scope(roles, exclusion, roles_sha256=sha(roles), exclusion_sha256=sha(exclusion))
        return [load_context(context, scope, store, role=role)]

    initial_rows = {role: acquire(1, learner.state_hash(), role=role)[0] for role in ("FIT", "B", "A")}
    fields = ("context_id", "dataset_id", "family_id", "task", "role", "train", "cut_sha256")
    sample = asdict(initial_rows["FIT"].context)
    source_fields = (
        "proposal_source_sha256",
        "evaluator_sha256",
        "native_binary_sha256",
        "environment_sha256",
        "protocol_sha256",
        "encoder_sha256",
        "encoder_model_sha256",
    )
    role_doc = {
        "schema": "h206-role-contract-v1",
        "reference_families": ["fixture-family-FIT"],
        "contexts": {
            r.context.context_id: {key: asdict(r.context)[key] for key in fields}
            for r in initial_rows.values()
        },
        "sources": {key: sample[key] for key in source_fields},
    }
    closure_doc = {
        "schema": "h206-exclusion-contract-v1",
        "links": [
            edge
            for r in initial_rows.values()
            for edge in (
                ["dataset:" + r.context.dataset_id, "family:" + r.context.family_id],
                ["family:" + r.context.family_id, "family:alias-" + r.context.role],
            )
        ],
        "protected": [],
    }

    def approved_acquire(round_index, actor_hash, *, role="FIT", protected=False, bad_source=False):
        roles = json.loads(canonical(role_doc))
        closure = json.loads(canonical(closure_doc))
        if bad_source:
            roles["sources"]["evaluator_sha256"] = "f" * 64
        if protected:
            closure["protected"] = ["family:alias-" + role]
        roles_bytes, closure_bytes = canonical(roles), canonical(closure)
        scope = Scope(
            roles_bytes, closure_bytes, roles_sha256=sha(roles_bytes), exclusion_sha256=sha(closure_bytes)
        )
        row = replace(initial_rows[role].context, actor_sha256=actor_hash, seed=206101 + round_index)
        return [load_context(row, scope, store, role=role)]

    approved_acquire.scope = Scope(
        canonical(role_doc),
        canonical(closure_doc),
        roles_sha256=sha(canonical(role_doc)),
        exclusion_sha256=sha(canonical(closure_doc)),
    )
    return learner, approved_acquire, native, cache, store, client, ledger


def test_real_round_consumes_native_labels_and_inherited_dope_updates(integrated):
    learner, acquire, native, cache, store, client, ledger = integrated
    before = learner.state_hash()
    trace = client.trace
    original_step = learner.optimizer.step

    def observed_step(*args, **kwargs):
        trace.append(("optimizer", "real AdamW"))
        return original_step(*args, **kwargs)

    learner.optimizer.step = observed_step
    prior = None
    for round_index in (1, 2, 3):
        prior = run_round(
            round_index=round_index,
            learner=learner,
            acquire=acquire,
            native=native,
            cache=cache,
            store=store,
            prefix=f"round-{round_index}",
            budget=ledger,
            cpu_reserve_seconds=120,
            k=1,
            updates=3,
            prior_round=prior,
        )
    result = json.loads(prior.read(store))
    assert learner.updates == 9 and native.calls == 3
    assert result["proof"]["forward_changed"] and learner.state_hash() != before
    assert min(result["proof"]["gradient_squared_sums"].values()) > 0
    assert any("dope_model" in n for n in result["proof"]["changed_parameters"])
    assert learner.model.gamenet_anchor.training is False
    first_update = next(i for i, v in enumerate(trace) if v[0] == "optimizer")
    assert any(v[0] == "get" and v[1].endswith("/teacher.json") for v in trace[:first_update])
    (ledger.path.parent / "integration-proof.json").write_bytes(
        canonical(
            {
                "scope": "controlled data-free fixture; not genuine r882/native-Rust qualification",
                "inherited_encoder_source_sha256": sha(FUSED_SOURCE.read_bytes()),
                "native_boundary": "real subprocess using controlled executable fixture",
                "native_calls": native.calls,
                "optimizer_updates": learner.updates,
                "examples": learner.examples,
                "proof": result["proof"],
                "new_independent_families_vs_reference": result["new_independent_families_vs_reference"],
                "candidate_endpoint_eligible": result["candidate_endpoint_eligible"],
            }
        )
    )
    ledger_state = json.loads(ledger.path.read_text())
    assert len(ledger_state["entries"]) == 3 and ledger_state["cpu_used_seconds"] > 0
    assert ledger_state["gpu_used_seconds"] == 0 and ledger_state["reservations"] == {}


def test_native_cache_reuse_and_semantic_identity_mismatch(integrated):
    learner, acquire, native, cache, store, _, ledger = integrated
    row = acquire(1, learner.state_hash())[0]
    with ledger.operation("controlled-search", cpu_reserve=120, gpu_reserve=0):
        ref = collect_teacher(row, learner.score(row), 16, native, cache, store, "search-1", ledger)
        collect_teacher(row, learner.score(row), 16, native, cache, store, "search-2", ledger)
    assert native.calls == 16 and cache.hits == 1
    assert len(json.loads(ref.read(store))["records"]) == 16
    cached = next(cache.root.glob("*.json"))
    doc = json.loads(cached.read_bytes())
    doc["identity"]["scope"] = "wrong"
    cached.write_bytes(canonical(doc))
    with ledger.operation("poisoned-search", cpu_reserve=120, gpu_reserve=0):
        with pytest.raises(ValueError, match="cached native outcome identity"):
            collect_teacher(row, learner.score(row), 16, native, cache, store, "search-3", ledger)


@pytest.mark.parametrize("fault", ["protected", "source"])
def test_transitive_family_and_source_refusal_before_payload_load(integrated, fault):
    learner, acquire, _, _, _, _, _ = integrated
    with pytest.raises(ValueError, match="protected|source/evaluator"):
        acquire(1, learner.state_hash(), protected=fault == "protected", bad_source=fault == "source")


def test_missing_tensors_task_role_and_identity_failures(integrated):
    learner, acquire, _, _, store, _, _ = integrated
    row = acquire(1, learner.state_hash())[0]
    with pytest.raises(ValueError, match="regression"):
        replace(row.context, task="binary")
    row.tensors.pop("vec256")
    with pytest.raises(ValueError, match="missing/unexpected"):
        learner.score(row)

    class WrongVersion:
        def get(self, ref):
            data, _ = store.get(ref)
            return data, "wrong-version"

    with pytest.raises(ValueError, match="version/body"):
        row.context.train.read(WrongVersion())


def test_s3_verification_failure_prevents_any_optimizer_update(integrated):
    learner, acquire, native, cache, store, client, ledger = integrated
    client.corrupt = True
    with pytest.raises(ValueError, match="version/body"):
        run_round(
            round_index=1,
            learner=learner,
            acquire=acquire,
            native=native,
            cache=cache,
            store=store,
            prefix="corrupt",
            budget=ledger,
            cpu_reserve_seconds=120,
            k=1,
            updates=3,
        )
    assert learner.updates == 0
    assert json.loads(ledger.path.read_text())["entries"][-1]["failed"]


def test_no_updates_and_frozen_anchor_failures(integrated):
    learner, _, _, _, _, _, _ = integrated
    with pytest.raises(ValueError, match="zero updates"):
        learner.proof()
    learner.model.gamenet_anchor.weight.requires_grad_(True)
    with pytest.raises(ValueError, match="closure/frozen"):
        learner.model.assert_frozen()


def test_heldout_learning_and_A_before_B_lock_refused(integrated):
    learner, acquire, _, _, store, _, _ = integrated
    row = acquire(1, learner.state_hash(), role="A")[0]
    with pytest.raises(ValueError, match="held-out"):
        learner.update(row, None, store)
    with pytest.raises(ValueError, match="before durable B"):
        evaluate_fixed(learner, [row], checkpoint_sha256=learner.state_hash(), store=store, prefix="A")
    with pytest.raises(ValueError, match="differs"):
        evaluate_fixed(learner, [row], checkpoint_sha256="f" * 64, store=store, prefix="A")


def test_cumulative_budget_survives_resume_and_refuses_near_exhaustion(tmp_path):
    path = tmp_path / "prior-ledger.json"
    fixture_ledger(path, used_cpu=400 * 3600 - 0.5)
    resumed = BudgetLedger(path)
    with pytest.raises(RuntimeError, match="cumulative campaign budget"):
        with resumed.operation("cannot-reset", cpu_reserve=1, gpu_reserve=0):
            raise AssertionError("must refuse before any work")
    assert json.loads(path.read_text())["cpu_used_seconds"] == 400 * 3600 - 0.5
    with pytest.raises(ValueError, match="existing cumulative"):
        BudgetLedger(tmp_path / "absent.json")


def test_wrong_native_binary_and_malformed_object_contract(integrated, tmp_path, monkeypatch):
    _, _, native, _, _, _, _ = integrated
    from research.flywheel_selfplay.h206_search_distill import native as native_module

    def forbidden_launch(*args, **kwargs):
        raise AssertionError("wrong native SHA must fail before any subprocess")

    monkeypatch.setattr(native_module.subprocess, "run", forbidden_launch)
    rejected_scratch = tmp_path / "wrong-binary-must-not-open"
    with pytest.raises(ValueError, match="^native evaluator binary SHA mismatch$"):
        RustS1Evaluator(
            native.binary,
            binary_sha256="f" * 64,
            scratch=rejected_scratch,
            evaluator_sha256=sha(METRICS_SOURCE.read_bytes()),
        )
    assert not rejected_scratch.exists()
    from research.flywheel_selfplay.h206_search_distill.contracts import ObjectRef

    with pytest.raises(ValueError, match="VersionId"):
        ObjectRef("bucket", "key", "null", "f" * 64)


def test_disconnected_or_nonchanging_optimizer_cannot_claim_training(integrated):
    learner, acquire, native, cache, store, _, ledger = integrated
    learner.optimizer.step = lambda *args, **kwargs: None
    with pytest.raises(ValueError, match="zero updates/no intended"):
        run_round(
            round_index=1,
            learner=learner,
            acquire=acquire,
            native=native,
            cache=cache,
            store=store,
            prefix="no-param-change",
            budget=ledger,
            cpu_reserve_seconds=120,
            k=1,
            updates=3,
        )
    assert learner.updates == 3  # Counter alone is explicitly not a training proof.


def test_optimizer_frozen_anchor_group_rejected_before_step(integrated):
    learner, acquire, native, cache, store, _, ledger = integrated
    learner.optimizer.add_param_group({"params": list(learner.model.gamenet_anchor.parameters())})
    with pytest.raises(RuntimeError, match="frozen anchor"):
        run_round(
            round_index=1,
            learner=learner,
            acquire=acquire,
            native=native,
            cache=cache,
            store=store,
            prefix="bad-optimizer",
            budget=ledger,
            cpu_reserve_seconds=120,
            k=1,
            updates=3,
        )
    assert learner.updates == 0


def test_cache_cost_must_equal_raw_native_output(integrated):
    learner, acquire, native, cache, store, _, ledger = integrated
    row = acquire(1, learner.state_hash())[0]
    with ledger.operation("cache-cost-proof", cpu_reserve=120, gpu_reserve=0):
        collect_teacher(row, learner.score(row), 1, native, cache, store, "raw-original", ledger)
        path = next(cache.root.glob("*.json"))
        doc = json.loads(path.read_bytes())
        doc["records"][0]["metrics"]["fitness"] += 1
        doc["records_sha256"] = digest(doc["records"])
        path.write_bytes(canonical(doc))
        with pytest.raises(ValueError, match="fitness differs from raw"):
            collect_teacher(row, learner.score(row), 1, native, cache, store, "raw-poisoned", ledger)


def test_durable_resume_keeps_optimizer_updates_scope_and_costs(integrated):
    from research.flywheel_selfplay.h206_search_distill.contracts import ObjectRef
    from research.flywheel_selfplay.h206_search_distill.executor import restore_learner

    learner, acquire, native, cache, store, _, ledger = integrated
    receipt = run_round(
        round_index=1,
        learner=learner,
        acquire=acquire,
        native=native,
        cache=cache,
        store=store,
        prefix="before-resume",
        budget=ledger,
        cpu_reserve_seconds=120,
        k=1,
        updates=3,
    )
    round_doc = json.loads(receipt.read(store))
    consumed = json.loads(ledger.path.read_text())["cpu_used_seconds"]
    learner.updates = 0  # Prove resume restores counts from verified bytes, not caller state.
    restore_learner(
        learner,
        ObjectRef(**round_doc["checkpoint"]),
        store,
        expected_state_sha256=round_doc["checkpoint_sha256"],
        budget=ledger,
        cpu_reserve_seconds=30,
    )
    assert learner.updates == 3 and learner.examples == 96
    assert json.loads(ledger.path.read_text())["cpu_used_seconds"] >= consumed
    receipt2 = run_round(
        round_index=2,
        learner=learner,
        acquire=acquire,
        native=native,
        cache=cache,
        store=store,
        prefix="after-resume",
        budget=ledger,
        cpu_reserve_seconds=120,
        k=1,
        updates=3,
        prior_round=receipt,
    )
    after = json.loads(receipt2.read(store))
    assert after["optimizer_updates_total"] == 6
    assert after["new_independent_families_vs_reference"] == []
    assert len(after["repeated_contexts"]) == 1 and after["new_contexts_to_campaign"] == []
    assert not after["candidate_endpoint_eligible"]


def test_no_early_B_scoring_even_after_actual_updates(integrated):
    learner, acquire, native, cache, store, _, ledger = integrated
    b_rows = acquire(1, learner.state_hash(), role="B")
    with pytest.raises(ValueError, match="B scoring forbidden"):
        run_round(
            round_index=1,
            learner=learner,
            acquire=acquire,
            native=native,
            cache=cache,
            store=store,
            prefix="early-B",
            budget=ledger,
            cpu_reserve_seconds=120,
            k=1,
            updates=3,
            evaluation_rows=b_rows,
        )
    assert learner.updates == 3


def test_checkpoint_cost_panel_is_durable_and_exact(integrated):
    learner, acquire, _, _, store, client, _ = integrated
    b_rows = acquire(1, learner.state_hash(), role="B")
    checkpoint = learner.state_hash()
    ref = evaluate_fixed(learner, b_rows, checkpoint_sha256=checkpoint, store=store, prefix="fixed-B")
    panel = json.loads(ref.read(store))
    assert panel["checkpoint_sha256"] == checkpoint
    assert np.array_equal(panel["panel"][b_rows[0].context.context_id], learner.score(b_rows[0]))
    assert client.trace[-1][0] == "get"


def test_model_content_pin_and_loaded_tensor_identity_cannot_drift(integrated):
    learner, acquire, _, _, _, _, _ = integrated
    with pytest.raises(ValueError, match="model content"):
        SearchStudent(
            inherited_fused(),
            seed=206101,
            run_id="H206-pin-failure",
            launch_metadata={},
            encoder_state_sha256="f" * 64,
            anchor_state_sha256="f" * 64,
            encoder_source_sha256=sha(FUSED_SOURCE.read_bytes()),
        )
    row = acquire(1, learner.state_hash())[0]
    row.train_bytes += b"bad"
    with pytest.raises(ValueError, match="TRAIN identity changed"):
        learner.score(row)


def test_real_inherited_proposer_source_and_cap():
    from research.flywheel_selfplay.h206_search_distill.native import propose_slate

    parents = [{"dsl": "(pipe X (standardize) (ridge_gcv))", "fitness": 0.0, "family_id": "fixture"}]
    with pytest.raises(ValueError, match="proposal source"):
        propose_slate(parents, np.zeros(48), [], "fixture", 206101, source_sha256="f" * 64)
    # Deterministic real GP generation, not a mock or an accept-either-outcome test.
    result = propose_slate(
        parents, np.zeros(48), [], "fixture", 206101, source_sha256=sha(METRICS_SOURCE.read_bytes())
    )
    assert len({p["dsl"] for p in result["programs"]}) == 256
    assert result["attempts"] == 256


def test_model_factory_uses_verified_checkpoint_bytes_and_budget(integrated):
    from research.flywheel_selfplay.h206_search_distill.executor import initialize_learner

    _, _, _, _, store, _, ledger = integrated
    fused = inherited_fused()
    checkpoint_bytes = b"controlled fixture initialization, not model payload"
    ref = store.put("fixture-initialization.json", checkpoint_bytes)
    called = []

    def factory(data):
        assert data == checkpoint_bytes
        called.append(True)
        return fused

    learner = initialize_learner(
        factory,
        ref,
        store,
        budget=ledger,
        cpu_reserve_seconds=30,
        expected_checkpoint_sha256=ref.sha256,
        seed=206101,
        run_id="H206-factory-fixture",
        launch_metadata={},
        encoder_state_sha256=state_hash(fused),
        anchor_state_sha256=state_hash(fused.adapter_10tok),
        encoder_source_sha256=sha(FUSED_SOURCE.read_bytes()),
    )
    assert called == [True] and learner.updates == 0
    assert json.loads(ledger.path.read_text())["entries"][-1]["stage"] == "model_setup_including_fetch"


def test_scope_rejects_same_component_in_different_roles():
    roles = canonical(
        {
            "schema": "h206-role-contract-v1",
            "reference_families": ["f"],
            "contexts": {
                "fit": {"family_id": "f", "role": "FIT"},
                "held": {"family_id": "alias", "role": "A"},
            },
        }
    )
    exclusion = canonical(
        {"schema": "h206-exclusion-contract-v1", "links": [["family:f", "family:alias"]], "protected": []}
    )
    with pytest.raises(ValueError, match="transitive family leakage"):
        Scope(roles, exclusion, roles_sha256=sha(roles), exclusion_sha256=sha(exclusion))


def test_prior_round_changes_do_not_mask_a_later_noop_optimizer(integrated):
    learner, acquire, native, cache, store, _, ledger = integrated
    prior = run_round(
        round_index=1,
        learner=learner,
        acquire=acquire,
        native=native,
        cache=cache,
        store=store,
        prefix="real-first",
        budget=ledger,
        cpu_reserve_seconds=120,
        k=1,
        updates=3,
    )
    learner.optimizer.step = lambda *args, **kwargs: None
    with pytest.raises(ValueError, match="THIS round"):
        run_round(
            round_index=2,
            learner=learner,
            acquire=acquire,
            native=native,
            cache=cache,
            store=store,
            prefix="noop-second",
            budget=ledger,
            cpu_reserve_seconds=120,
            k=1,
            updates=3,
            prior_round=prior,
        )


def test_student_refuses_teacher_from_a_different_native_binary(integrated):
    learner, acquire, native, cache, store, _, ledger = integrated
    row = acquire(1, learner.state_hash())[0]
    with ledger.operation("teacher-binding", cpu_reserve=30, gpu_reserve=0):
        ref = collect_teacher(row, learner.score(row), 1, native, cache, store, "bound-native", ledger)
        doc = json.loads(ref.read(store))
        doc["identity"]["native_binary"] = "f" * 64
        wrong = store.put("wrong-native-teacher.json", canonical(doc))
        with pytest.raises(ValueError, match="full identity"):
            learner.update(row, wrong, store)
    assert learner.updates == 0


@pytest.mark.parametrize("status", ["UNRECONCILED", None])
def test_unreconciled_ledger_refuses_before_reservation(tmp_path, status):
    path = tmp_path / "campaign-ledger.json"
    budget = fixture_ledger(path)
    state = json.loads(path.read_text())
    if status is None:
        state.pop("accounting_status")
    else:
        state["accounting_status"] = status
    state["cpu_used_seconds"] = None
    state["cpu_used_seconds_lower_bound"] = 206.771232
    state["conservative_unmeasured_cpu_reservation_seconds"] = 1440000 - 206.771232
    path.write_bytes(canonical(state))
    before = path.read_bytes()
    with pytest.raises(ValueError, match="^campaign accounting is not reconciled$"):
        with budget.operation("must-not-start", cpu_reserve=1, gpu_reserve=0):
            pytest.fail("unreconciled usage must not start an operation")
    assert path.read_bytes() == before
    assert not path.with_suffix(".next").exists()
    assert not budget.active


@pytest.mark.parametrize("update_hash", [False, True])
def test_recovered_binding_poison_is_refused(tmp_path, monkeypatch, update_hash):
    from research.flywheel_selfplay.h206_search_distill import runner

    for name in ("launch.json", "bindings.json"):
        (tmp_path / name).write_bytes((HERE / name).read_bytes())
    monkeypatch.setattr(runner, "HERE", tmp_path)
    assert runner.load_plan()["scientific_execution_enabled"] is False
    binding_path = tmp_path / "bindings.json"
    binding = json.loads(binding_path.read_text())
    binding["initial_checkpoint"]["sha256"] = "f" * 64
    binding_path.write_bytes(canonical(binding))
    if update_hash:
        plan_path = tmp_path / "launch.json"
        plan = json.loads(plan_path.read_text())
        plan["recovered_bindings_sha256"] = sha(binding_path.read_bytes())
        plan_path.write_bytes(canonical(plan))
    error = "recovered binding plan mismatch" if update_hash else "recovered binding manifest hash mismatch"
    with pytest.raises(ValueError, match=error):
        runner.load_plan()


def _study_rows():
    return [
        {"context_id": f"family-{i:03d}-context-{j}", "family_id": f"family-{i:03d}"}
        for i in range(205)
        for j in range(6 if i < 194 else 5)
    ]


def test_full_study_fixed_60_cells_and_exact_once_roles():
    from research.flywheel_selfplay.h206_search_distill.study import registered_cells

    rows = _study_rows()
    plan = registered_cells(rows)
    assert len(plan["cells"]) == 60
    assert plan["rounds"] == 3 and plan["updates_per_round"] == 600
    for seed in (206101, 206102, 206103):
        cells = [c for c in plan["cells"] if c["seed"] == seed and c["arm"] == "CTRL"]
        for role in ("B", "A"):
            assert sorted(cid for c in cells for cid in c["context_ids"][role]) == sorted(
                r["context_id"] for r in rows
            )
        for cell in cells:
            assert {r: len(f) for r, f in cell["family_ids"].items()} == {"A": 41, "B": 41, "FIT": 123}
    with pytest.raises(ValueError, match="population"):
        registered_cells(rows[:-1])


def test_selector_private_step_and_joint_harm_gate():
    from research.flywheel_selfplay.h177_pooled_selector import analysis as original
    from research.flywheel_selfplay.h206_search_distill.study import load_selector, pooled_select

    rows = _study_rows()
    before = dict(original.PRESPECIFIED_CANDIDATE_STEP)
    private = load_selector(HERE.parents[2])
    assert set(private.PRESPECIFIED_CANDIDATE_STEP.values()) == {1800}
    base = np.ones((3, len(rows)))
    panels = {
        a: base + gain for a, gain in zip(("CTRL", "ACC4", "ACC16", "OC15"), (0.002, 0.003, 0.004, 0.02))
    }
    harmed = {r["family_id"] for r in rows[:126]}
    assert len(harmed) == 21
    panels["OC15"][:, [i for i, r in enumerate(rows) if r["family_id"] in harmed]] = 0.9
    result = pooled_select(HERE.parents[2], rows, base, panels)
    assert result["selected_arm"] == "ACC16"
    assert result["decisions"]["OC15"]["harm_families"] == 21
    assert result["decisions"]["OC15"]["admitted_with_harm"] is False
    assert original.PRESPECIFIED_CANDIDATE_STEP == before


def test_phased_runtime_real_labels_and_per_step_durability(integrated, tmp_path):
    from research.flywheel_selfplay.h206_search_distill.study import PhaseRuntime

    learner, acquire, native, cache, store, client, ledger = integrated
    for round_index in (1, 2, 3):
        rows = acquire(round_index, learner.state_hash())
        runtime = PhaseRuntime(store, ledger, scientific=False, journal=tmp_path / f"journal-{round_index}")
        with ledger.operation(f"controlled-round-{round_index}", cpu_reserve=120, gpu_reserve=0):
            proposal = runtime.propose(learner, rows, f"phased/{round_index}/propose")
            teachers = runtime.teachers(
                proposal, acquire.scope, native, cache, 1, f"phased/{round_index}/teacher"
            )
            doc = json.loads(teachers.read(store))
            refs = [ObjectRef(**r["teacher"]) for r in doc["rows"]]
            result_ref = runtime.train(
                learner, rows, refs, f"phased/{round_index}/train", round_index=round_index, updates=3
            )
        result = json.loads(result_ref.read(store))
        assert len(result["steps"]) == 3 and result["proof"]["forward_changed"]
        assert min(result["proof"]["gradient_squared_sums"].values()) > 0
        step = json.loads(ObjectRef(**result["steps"][-1]).read(store))
        assert np.isfinite(step["loss_before"]) and np.isfinite(step["loss_after"])
        assert step["state_before"] != step["state_after"]
        assert step["changed_parameters"]
    assert learner.updates == 9 and native.calls == 3
    # Every update's remote readback precedes the next optimizer update.
    step_gets = [i for i, e in enumerate(client.trace) if e[0] == "get" and "/steps/" in e[1]]
    assert len(step_gets) >= 9


def test_incremental_proof_and_partial_model_survive_step_custody_failure(integrated, tmp_path, monkeypatch):
    from research.flywheel_selfplay.h206_search_distill.study import PhaseRuntime

    learner, acquire, native, cache, store, _, ledger = integrated
    rows = acquire(1, learner.state_hash())
    journal = tmp_path / "partial-journal"
    runtime = PhaseRuntime(store, ledger, scientific=False, journal=journal)
    with ledger.operation("controlled-failure", cpu_reserve=120, gpu_reserve=0):
        proposal = runtime.propose(learner, rows, "partial/propose")
        teachers = runtime.teachers(proposal, acquire.scope, native, cache, 1, "partial/teacher")
        refs = [ObjectRef(**r["teacher"]) for r in json.loads(teachers.read(store))["rows"]]
        put = store.put

        def stopped_put(name, body):
            if "/steps/000002.json" in name or name.endswith("partial-checkpoint.pt"):
                raise OSError("controlled persistent custody outage")
            return put(name, body)

        monkeypatch.setattr(store, "put", stopped_put)
        with pytest.raises(OSError, match="custody outage"):
            runtime.train(learner, rows, refs, "partial/train", round_index=1, updates=3)
    assert learner.updates == 2
    local = json.loads((journal / "update-000002.json").read_text())
    assert local["update"] == 2 and local["state_after"] == learner.state_hash()
    assert json.loads((journal / "INCOMPLETE.json").read_text())["updates"] == 2
    state = torch.load(journal / "partial-checkpoint.pt", weights_only=True)
    assert state["updates"] == 2 and state["examples"] == 64
    assert (journal / "custody-failure.json").is_file()


def test_parent_reservation_token_is_reused_without_budget_renewal(tmp_path):
    from research.flywheel_selfplay.h206_search_distill.supervisor import ReservedPhaseBudget

    ledger = fixture_ledger(tmp_path / "parent-ledger.json", used_cpu=123)
    with ledger.operation("parent-phase", cpu_reserve=10, gpu_reserve=0) as token:
        assert token
        child = ReservedPhaseBudget(ledger.path, token, "parent-phase")
        with child.measure("actual child stage"):
            sum(range(100))
        with pytest.raises(RuntimeError, match="cannot create another"):
            child.operation("bad nested", cpu_reserve=10, gpu_reserve=0)
        with pytest.raises(ValueError, match="missing parent"):
            ReservedPhaseBudget(ledger.path, "invented-token", "parent-phase")
    state = json.loads(ledger.path.read_text())
    assert state["cpu_used_seconds"] >= 123 and len(state["entries"]) == 1
    assert state["reservations"] == {}


def test_real_gpu_permit_refuses_before_any_cuda_probe(tmp_path, monkeypatch):
    from research.flywheel_selfplay.h206_search_distill.supervisor import GPUWorkerPermit

    def forbidden():
        raise AssertionError("unadmitted GPU probe")

    monkeypatch.setattr(torch.cuda, "device_count", forbidden)
    permit = GPUWorkerPermit(
        {"reservation": "unowned", "phase": "teacher", "supervisor_pid": -1}, tmp_path, None
    )
    with pytest.raises(ValueError, match="not the admitted"):
        permit.validate()


@pytest.mark.parametrize("used,free,launch", [(89, 11, True), (94, 6, False)])
def test_study_resource_guards_keep_disk_limits(used, free, launch):
    from research.flywheel_selfplay.h206_search_distill.supervisor import require_resources

    snapshot = {
        "nice": 19,
        "cpu_percent": 1,
        "available_bytes": 32 * 2**30,
        "tree_rss_bytes": 1,
        "disk_used_bytes": used * 10**10,
        "disk_available_bytes": free * 10**10,
        "effective_cores": 4,
    }
    with pytest.raises(RuntimeError):
        require_resources(snapshot, launching_gpu=launch)


def test_model_cache_new_version_requires_its_own_verified_get(tmp_path, integrated):
    from research.flywheel_selfplay.h206_search_distill.model import ModelCache

    _, _, _, _, store, client, _ = integrated
    body = b"same approved model bytes"
    first = store.put("model-one", body)
    second = store.put("model-two", body)
    cache = ModelCache(tmp_path / "actual-model-cache", HERE.parents[2])
    cache.fetch(first, store)
    before = len(client.trace)
    cache.fetch(second, store)
    assert any(e[0] == "get" and e[1] == second.key for e in client.trace[before:])
    before = len(client.trace)
    cache.fetch(second, store)
    assert len(client.trace) == before


def test_native91_rejects_nonargmin_vector_change_and_order_before_scoring():
    from research.flywheel_selfplay.h206_search_distill.model import native91_replay

    target = 1.1377941898164525
    values = np.zeros(91, dtype=np.float64)
    values[-1] = target * 91
    assert float(values.mean()) == target
    contract = [
        {"cid": str(i), "pool_sha256": "a" * 64, "training_table_sha256": "b" * 64} for i in range(91)
    ]
    rows = [{**c, "reference_costs": np.array([0.0, 1.0], dtype=np.float32)} for c in contract]
    reference = canonical(
        {
            "evaluation_contract": {"contexts": contract},
            "utility": {"selected_programs": ["p"] * 91, "per_context": values.tolist(), "mean": target},
        }
    )
    identities = {"scope": "controlled source fixture, never scientific qualification"}
    saved = []

    def score(row):
        return {
            "identities": identities,
            "costs": np.array([0.0, 1.0], dtype=np.float32),
            "selected_program": "p",
            "utility": float(values[int(row["cid"])]),
        }

    result = native91_replay(
        rows, reference, score=score, expected_identities=identities, persist=saved.append
    )
    assert result["contexts"] == 91 and len(saved) == 91

    def changed(row):
        out = score(row)
        out["costs"][1] = np.nextafter(np.float32(1), np.float32(2))
        return out

    saved.clear()
    with pytest.raises(ValueError, match="full-vector"):
        native91_replay(rows, reference, score=changed, expected_identities=identities, persist=saved.append)
    assert len(saved) == 1

    def forbidden(row):
        raise AssertionError("wrong row order must fail before any scored forward")

    with pytest.raises(ValueError, match="ordered batch"):
        native91_replay(
            rows[::-1], reference, score=forbidden, expected_identities=identities, persist=saved.append
        )


def test_native_scratch_custody_restores_before_owned_reclamation(integrated, monkeypatch):
    from research.flywheel_selfplay.h206_search_distill.study import preserve_native_scratch

    learner, acquire, native, cache, store, client, ledger = integrated
    row = acquire(1, learner.state_hash())[0]
    with ledger.operation("controlled-native-custody", cpu_reserve=30, gpu_reserve=0):
        teacher = collect_teacher(row, learner.score(row), 1, native, cache, store, "pack-teacher", ledger)
        doc = json.loads(teacher.read(store))
        jobs = [Path(r["scratch_job"]) for r in doc["records"]]
        original = Path.unlink

        def guarded_unlink(path, *args, **kwargs):
            assert any(
                e[0] == "get" and e[1].endswith("owned-native-pack-restoration.json") for e in client.trace
            )
            return original(path, *args, **kwargs)

        with monkeypatch.context() as scoped:
            scoped.setattr(Path, "unlink", guarded_unlink)
            proof = preserve_native_scratch(teacher, row, native, cache, store, "pack-proof")
    recovered = json.loads(proof.read(store))
    assert recovered["all_restored_sha256_match"] is True
    assert len(recovered["members"]) == 4
    assert all(not job.exists() for job in jobs)
    # Durable labels remain consumable after owned scratch/cache copies are gone.
    assert isinstance(learner.teacher_target(row, teacher, store), int)


@pytest.mark.parametrize("abort_fit", [False, True])
def test_full_controller_order_and_whole_attempt_abort(integrated, abort_fit):
    from contextlib import contextmanager

    from research.flywheel_selfplay.h206_search_distill.study import ARMS, StudyController, registered_cells

    _, _, _, _, store, _, _ = integrated
    rows = _study_rows()
    baseline = store.put(
        "macro/baseline.json",
        canonical({"context_ids": [r["context_id"] for r in rows], "values": [1.0] * len(rows)}),
    )
    inputs = {
        cell["cell_id"]: {
            "cpu_reserve_seconds": {p: 1 for p in ("propose", "teacher", "train", "score", "utility")},
            "phase_caps": {p: 1 for p in ("propose", "teacher", "score", "utility")},
        }
        for cell in registered_cells(rows)["cells"]
    }

    class ControlledSupervisor:
        """Metadata-only state-machine control; never a real controller/device."""

        def __init__(self):
            self.store, self.trace = store, []
            self.admission = {
                "source": {"fixture": "metadata control, no source/science qualification"},
                "study_input_payload_sha256": digest(
                    {"rows": rows, "inputs": inputs, "baseline": asdict(baseline)}
                ),
                "fold_manifest_sha256": digest(registered_cells(rows)),
            }

        @contextmanager
        def controller(self):
            yield self

        def phase(self, task, **kwargs):
            self.trace.append((task["round"], task["phase"], task["cell_id"], task.get("role")))
            if abort_fit and task["phase"] == "train":
                raise RuntimeError("controlled required fit failure")
            if task["phase"] == "train":
                doc = {"updates": 600 * task["round"], "state_sha256": digest(task["cell_id"])}
            elif task["phase"] == "utility":
                if task["role"] == "A":
                    lock = json.loads(ObjectRef(**task["b_lock"]).read(store))
                    assert lock["status"] == "B_LOCKED" and len(lock["checkpoint_sha256"]) == 60
                utility = 1.0 + (list(ARMS).index(task["arm"]) + 1) * 0.002
                doc = {
                    cid: {"role": task["role"], "student": {"utility": utility}}
                    for cid in task["context_ids"][task["role"]]
                }
            else:
                doc = {"scope": "metadata-only controller test"}
            return store.put("macro/results/" + str(len(self.trace)), canonical(doc))

    supervisor = ControlledSupervisor()
    controller = StudyController(supervisor, rows, inputs, baseline, repo=HERE.parents[2])
    if abort_fit:
        with pytest.raises(RuntimeError, match="required fit failure"):
            controller.run()
        assert not any(p in ("score", "utility") for _, p, _, _ in supervisor.trace)
        return
    ref = controller.run()
    result = json.loads(ref.read(store))
    assert result["selected_arm"] == "OC15" and result["confirmed_lift"] is False
    assert result["sealed_confirmation"] == "NOT_RUN"
    assert len([t for t in supervisor.trace if t[1] == "train"]) == 180
    assert len(supervisor.trace) == 780
    first_a = next(i for i, t in enumerate(supervisor.trace) if t[3] == "A")
    assert len([t for t in supervisor.trace[:first_a] if t[1] == "utility" and t[3] == "B"]) == 60


def test_protected_role_metadata_refuses_before_template_body(integrated, monkeypatch):
    from research.flywheel_selfplay.h206_search_distill.study import StudyInputs

    learner, acquire, _, _, store, _, _ = integrated
    scope = acquire.scope
    roles = scope.roles
    role_ref = store.put("protected-template/roles.json", canonical(roles))
    links = [["dataset:" + r["dataset_id"], "family:" + r["family_id"]] for r in roles["contexts"].values()]
    row = next(r for r in roles["contexts"].values() if r["role"] == "FIT")
    exclusion_ref = store.put(
        "protected-template/exclusion.json",
        canonical(
            {
                "schema": "h206-exclusion-contract-v1",
                "links": links,
                "protected": ["family:" + row["family_id"]],
            }
        ),
    )
    missing = ObjectRef("fixture-only", "forbidden-template-body", "v1", "a" * 64)
    task = {
        "scope": {"roles": asdict(role_ref), "exclusion": asdict(exclusion_ref)},
        "round": 1,
        "context_ids": {"FIT": [row["context_id"]]},
        "templates": {"1": {"FIT": asdict(missing)}},
    }
    inputs = StudyInputs(task, store)

    def forbidden(ref):
        raise AssertionError("protected metadata must refuse before ANY template/tensor/body GET")

    monkeypatch.setattr(store, "get", forbidden)
    with pytest.raises(ValueError, match="transitively protected"):
        inputs.acquire("FIT", learner.state_hash())


def test_factory_private_source_closure_replaces_ambient_optional_import(monkeypatch):
    import inspect
    import sys
    import types

    from research.flywheel_selfplay.h206_search_distill.model import pinned_original_modules

    repo = HERE.parents[2]
    manifest = json.loads((repo / "research/flywheel_selfplay/h170_onecycle/manifest.json").read_text())
    pins = {
        "research/flywheel_selfplay/custody/files" + p: value
        for p, value in manifest["assets"].items()
        if p.endswith(".py")
    }
    pins[str(FUSED_SOURCE.relative_to(repo))] = sha(FUSED_SOURCE.read_bytes())
    ambient = types.ModuleType("rank_prepare")
    monkeypatch.setitem(sys.modules, "rank_prepare", ambient)
    before_path = list(sys.path)
    with pinned_original_modules(repo, pins) as fused:
        feature_fn = fused.get_w1_sign_matrix.__globals__["dsl_features_dict"]
        assert feature_fn is not None
        path = Path(inspect.getsourcefile(feature_fn)).resolve()
        assert path.is_relative_to(repo / "research/flywheel_selfplay/custody/files")
        assert (
            sha(path.read_bytes()) == manifest["assets"]["/home/ubuntu/JopeDimeMicro/python/rank_prepare.py"]
        )
    assert sys.modules["rank_prepare"] is ambient and sys.path == before_path
