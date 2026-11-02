"""H215 single guarded CUDA attempt; source read from immutable Git objects."""

import argparse
from collections import OrderedDict
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import zipfile

import boto3
import numpy as np
import psutil
import torch

PREFIX = "research/flywheel_selfplay/h215_pess_disagreement/"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    raw = json.dumps(value, indent=2, allow_nan=False) + "\n"
    with Path(path).open("x") as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())


def source(repo, commit, path):
    return subprocess.check_output(
        ["git", "-C", str(repo), "show", commit + ":" + path]
    )


def load_module(repo, commit, path, name, expected):
    """Import a hash-pinned module from the exact checked-out Git revision."""
    repo = Path(repo).resolve()
    head = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    if head != commit:
        raise RuntimeError("checkout does not match pinned commit")
    module_path = (repo / path).resolve()
    if not module_path.is_relative_to(repo) or digest(module_path.read_bytes()) != expected:
        raise RuntimeError("source hash mismatch: " + path)
    spec = importlib.util.spec_from_file_location(name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import pinned module: " + path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def array_prefix(archive, name, count):
    """Decode only permitted revealed entries from a compressed numeric member."""
    with archive.open(name + ".npy") as f:
        version = np.lib.format.read_magic(f)
        if version != (1, 0):
            raise ValueError("unexpected pool header")
        shape, fortran, dtype = np.lib.format.read_array_header_1_0(f)
        if len(shape) != 1 or fortran or dtype.hasobject or count > shape[0]:
            raise ValueError("invalid reveal member")
        raw = f.read(count * dtype.itemsize)
        if len(raw) != count * dtype.itemsize:
            raise ValueError("short reveal read")
        return np.frombuffer(raw, dtype=dtype).copy()


class Pools:
    def __init__(self, rows, root):
        self.rows = {r["context_sha256"]: r for r in rows}
        self.root = Path(root)
        self.cache = OrderedDict()
        self.outcome_reads = []

    def blob(self, cid):
        path = self.root / (cid + ".npz")
        data = path.read_bytes()
        if digest(data) != self.rows[cid]["pool_sha256"]:
            raise RuntimeError("pool identity mismatch")
        return data

    def predictors(self, cid):
        if cid not in self.cache:
            raw = self.blob(cid)
            with np.load(io.BytesIO(raw), allow_pickle=False) as z:
                program = z["program"].astype(np.float32)
            with zipfile.ZipFile(io.BytesIO(raw)) as z:
                reveal_f = array_prefix(z, "fitness", 4)
                reveal_g = array_prefix(z, "gate", 4)
            if program.shape != (self.rows[cid]["n_candidates"], 412):
                raise ValueError("program shape mismatch")
            self.cache[cid] = (program, reveal_f, reveal_g)
            if len(self.cache) > 64:
                self.cache.popitem(last=False)
        self.cache.move_to_end(cid)
        return self.cache[cid]

    def utility(self, cid, allowed, role, locked=False):
        if (
            cid not in allowed
            or role not in ("NATIVE91", "FIT", "B", "A")
            or (role == "A" and not locked)
        ):
            raise RuntimeError("outcome role/lock refusal")
        with np.load(io.BytesIO(self.blob(cid)), allow_pickle=False) as z:
            fitness, gate = z["fitness"].astype(np.float64), z["gate"].astype(bool)
        self.outcome_reads.append((role, cid))
        utility = np.where(gate, -fitness, -2.0)
        if not np.isfinite(utility).all():
            raise RuntimeError("nonfinite utility")
        return utility


class Store:
    def __init__(self, output, prefix):
        self.output, self.prefix = output, prefix
        self.client = boto3.Session(profile_name="veox-jepson-prod").client(
            "s3", region_name="us-east-2"
        )
        self.receipts = {}

    def put(self, name, data):
        key = self.prefix + name
        bucket = "veox-jopedime-learning-use2"
        sha = digest(data)
        version = self.client.put_object(
            Bucket=bucket, Key=key, Body=data, IfNoneMatch="*", Metadata={"sha256": sha}
        )["VersionId"]
        response = self.client.get_object(Bucket=bucket, Key=key, VersionId=version)
        recovered = response["Body"].read()
        response["Body"].close()
        if response["VersionId"] != version or digest(recovered) != sha:
            raise RuntimeError("S3 exact-version recovery failure")
        receipt = dict(
            s3_uri="s3://" + bucket + "/" + key,
            version_id=version,
            sha256=sha,
            bytes=len(data),
        )
        self.receipts[name] = receipt
        save(self.output / ("receipt-" + name.replace("/", "_") + ".json"), receipt)
        return receipt

    def arrays(self, name, **arrays):
        b = io.BytesIO()
        np.savez_compressed(b, **arrays)
        return self.put(name + ".npz", b.getvalue())


def snapshot():
    disk = psutil.disk_usage("/home/ubuntu")
    process = psutil.Process()
    tree = [process] + process.children(recursive=True)
    return dict(
        load=os.getloadavg()[0],
        load5=os.getloadavg()[1],
        cores=len(os.sched_getaffinity(0)),
        cpu=psutil.cpu_percent(),
        available=psutil.virtual_memory().available,
        swap_free=psutil.swap_memory().free,
        disk_used_fraction=disk.used / (disk.used + disk.free),
        free=disk.free,
        rss=sum(p.memory_info().rss for p in tree if p.is_running()),
        threads=sum(p.num_threads() for p in tree if p.is_running()),
    )


class SustainedLoad:
    """Separate monotonic clocks: a recovery resets only its own signal."""

    def __init__(self):
        self.since = {"load": None, "cpu": None}
        self.elapsed = {"load": 0.0, "cpu": 0.0}

    def overloaded(self, load1, load5, cpu, cores, now):
        for key, over in (("load", load1 > 0.85 * cores), ("cpu", cpu > 85)):
            if not over:
                self.since[key] = None
            elif self.since[key] is None:
                self.since[key] = now
            self.elapsed[key] = (
                0.0 if self.since[key] is None else now - self.since[key]
            )
        return load5 > 0.85 * cores or any(t >= 300 for t in self.elapsed.values())


LOAD_STATE = SustainedLoad()


def guard(s, *, state=None, now=None, admission=False):
    state = LOAD_STATE if state is None else state
    overloaded = state.overloaded(
        s["load"],
        s["load5"],
        s["cpu"],
        s["cores"],
        time.monotonic() if now is None else now,
    )
    s.update(
        load_over_seconds=state.elapsed["load"], cpu_over_seconds=state.elapsed["cpu"]
    )
    if (
        overloaded
        or (admission and (s["load"] > 0.85 * s["cores"] or s["cpu"] > 85))
        or s["available"] < 16 * 2**30
        or s["disk_used_fraction"] > 0.938
        or s["free"] < 40_000_000_000
        or s["rss"] >= 6_000_000_000
    ):
        raise RuntimeError("resource refusal: " + repr(s))


def make_resource_check(resources, guard, snapshot, output, started, clock=time.monotonic):
    """Production resource sampler shared by timer and synchronous checks."""
    checking = False
    last_check = 0.0

    def check(*unused):
        nonlocal checking, last_check
        if checking or clock() - last_check < 1.0:
            return
        checking = True
        last_check = clock()
        try:
            sample = snapshot()
            resources.write(json.dumps(dict(sample, elapsed=clock() - started)) + "\n")
            resources.flush()
            guard(sample)
            if clock() - started > 5400 or sum(
                p.stat().st_size for p in output.rglob("*") if p.is_file()
            ) >= 300000000:
                raise RuntimeError("fixed wall/output budget exceeded")
        finally:
            checking = False

    return check


def run(args):
    os.nice(19)
    if os.uname().nodename != "xbabe2" or os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("xbabe2 GPU0 only")
    psutil.cpu_percent()
    time.sleep(1)
    guard(snapshot(), admission=True)
    if args.output.exists():
        raise RuntimeError("fresh output required")
    pins = json.loads(source(args.repo, args.commit, PREFIX + "manifest.json"))
    for kind in ("assets", "inputs"):
        for path, expected in pins[kind].items():
            if file_hash(path) != expected:
                raise RuntimeError("input/asset mismatch: " + path)
    for name, version in pins["packages"].items():
        if importlib.metadata.version(name) != version:
            raise RuntimeError("package mismatch: " + name)
    current_pins = dict(pins["source_sha256"])
    current_pins.update(pins.get("current_source_sha256", {}))
    for path, expected in current_pins.items():
        if digest(source(args.repo, args.commit, PREFIX + path)) != expected:
            raise RuntimeError("own source mismatch")
    args.output.mkdir(parents=True)
    started = time.monotonic()
    resources = (args.output / "resources.jsonl").open("x")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    sci = load_module(
        args.repo,
        args.commit,
        PREFIX + "science.py",
        "h215_science",
        pins["source_sha256"]["science.py"],
    )
    thread_path = "research/flywheel_selfplay/hwave_numeric_threads.py"
    numeric = load_module(
        args.repo,
        args.commit,
        thread_path,
        "h215_numeric",
        pins["parent_sources"][thread_path],
    )
    runtime_path = "research/flywheel_selfplay/h146_lookahead_exec/runtime.py"
    rt = load_module(
        args.repo,
        args.commit,
        runtime_path,
        "h215_runtime",
        pins["parent_sources"][runtime_path],
    )
    rt.HERE = (
        args.repo
    )  # Used only as Git cwd by the immutable historical guard loader.

    check = make_resource_check(resources, guard, snapshot, args.output, started)

    def numeric_check():
        report = numeric.capture_numeric_threads(torch)
        admitted = numeric.admit_numeric_threads(
            [report], os_threads=psutil.Process().num_threads()
        )
        if admitted["numeric_threads_upper_bound"] != 1:
            raise RuntimeError("H215 requires one numeric thread")
        return admitted

    def interrupted(signum, frame):
        raise RuntimeError("interrupted; no retry")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    signal.signal(signal.SIGALRM, check)
    signal.setitimer(signal.ITIMER_REAL, 2, 2)
    store = Store(args.output, pins["s3_prefix"] + "attempt1/")
    progress = dict(updates=0, locked=False, source_commit=args.commit)
    try:
        check()
        numeric_report = numeric_check()
        save(
            args.output / "launch.json",
            dict(
                wave="H215",
                source_commit=args.commit,
                manifest_sha256=digest(
                    source(args.repo, args.commit, PREFIX + "manifest.json")
                ),
                host=os.uname().nodename,
                device_uuid=pins["gpu_uuid"],
                numeric=numeric_report,
                start_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                deadline_seconds=5400,
                resource_limits=pins["resource_limits"],
                authority="hw_t0230_astra.md",
                optimizer_updates=0,
            ),
        )
        store.put("launch.json", (args.output / "launch.json").read_bytes())
        packet = json.loads(Path(pins["selected"]).read_text())
        train, val = packet["train"], packet["dev"]
        if (
            len(train) != 1219
            or len(val) != 91
            or len({r["family_id"] for r in train}) != 205
            or len({r["family_id"] for r in val}) != 19
            or {r["family_id"] for r in train} & {r["family_id"] for r in val}
        ):
            raise RuntimeError("cohort closure mismatch")
        for rows, split in ((train, "DL-Train"), (val, "DL-Val")):
            if any(
                r["dl_split"] != split
                or r["task"] != "regression"
                or r["source_split"] not in ("train", "inner")
                or r.get("has_meta_test", False)
                for r in rows
            ):
                raise RuntimeError("ineligible cohort row")
        catalog = json.loads(Path(pins["catalog"]).read_text())
        provenance = json.loads(
            source(args.repo, args.commit, PREFIX + "provenance.json")
        )
        selected_ids = {r["dataset_id"] for r in train + val}
        if (
            provenance["status"] != "PASS"
            or provenance["zero_synthetic"] is not True
            or selected_ids != set(provenance["dataset_ids"])
        ):
            raise RuntimeError("historical real-S3 provenance mismatch")
        eligible = {
            r["id"]
            for r in catalog["datasets"]
            if r["task"] == "regression"
            and r["s3_bucket"] == "veox-jopedime-datasets-use2"
        }
        source_manifest = json.loads(Path(pins["source_manifest"]).read_text())
        if (
            source_manifest["outer_rows_delivered"] != 0
            or source_manifest["excluded_payloads_opened"] != 0
            or source_manifest["legacy_test_metric_fields_delivered"]
        ):
            raise RuntimeError("protected source payload refusal")
        excluded = set(catalog["inventory"]["sealed_name_matches_excluded"])
        if any(
            r["dataset_id"] not in eligible or r["dataset_id"] in excluded
            for r in train + val
        ):
            raise RuntimeError("real catalog/protected row refusal")
        fold_manifest = json.loads(
            source(args.repo, args.commit, PREFIX + "fold_manifest.json")
        )
        for seed in sci.SEEDS:
            for outer in range(5):
                fit, b, a = sci.partition(train, seed, outer)
                if (
                    sorted(r["context_sha256"] for r in a)
                    != fold_manifest[str(seed)][outer]
                ):
                    raise RuntimeError("fold identity mismatch")
        allowed = (
            set(pins["inputs"])
            | set(pins["assets"])
            | {
                str(rt.INPUT / "pools" / (r["context_sha256"] + ".npz"))
                for r in train + val
            }
        )
        audit = rt.Audit(allowed, args.output)
        sys.addaudithook(audit.check)
        pools = Pools(train + val, rt.INPUT / "pools")
        with np.load(rt.CACHE, allow_pickle=False) as z:
            ids = list(map(str, z["cids"]))
            tokens, masks = z["tokens_10x32"], z["mask_10"]
        if len(ids) != 1310 or set(ids) != {r["context_sha256"] for r in train + val}:
            raise RuntimeError("feature cache closure mismatch")
        position = {cid: i for i, cid in enumerate(ids)}
        check()
        net, frozen_guard, names, frozen, fused = rt.build(pins["checkpoint"])
        net.to("cuda:0")
        numeric_check()
        for parameter in net.parameters():
            parameter.requires_grad_(False)
        all_frozen = rt.tensor_hash(net.state_dict())
        projection = fused.get_w1_sign_matrix() / np.sqrt(412.0)
        native = {}
        gpu_inputs = {}

        def native_forward(row):
            cid = row["context_sha256"]
            program, reveal_f, reveal_g = pools.predictors(cid)
            k, ix = len(program), position[cid]
            kwargs = dict(
                ctx_feat=torch.zeros((1, 584)),
                prog_412=torch.as_tensor(program[None]),
                proj_16=torch.as_tensor(
                    (program.astype(np.float64) @ projection).astype(np.float32)[None]
                ),
                base_prior_cost=torch.zeros((1, k)),
                scale=torch.full((1,), 0.4208),
                sos_prior=torch.zeros((1, k, 2)),
                revealed_indices=torch.arange(4)[None],
                revealed_raw_f=torch.as_tensor(reveal_f.astype(np.float32)[None]),
                revealed_gate=torch.as_tensor(reveal_g.astype(np.float32)[None]),
            )
            gpu_kwargs = {k: v.to("cuda:0") for k, v in kwargs.items()}
            tok = torch.as_tensor(tokens[ix][None], device="cuda:0")
            mask = torch.as_tensor(masks[ix][None], device="cuda:0")
            gpu_inputs[cid] = (gpu_kwargs, tok, mask)
            with torch.no_grad():
                out = (
                    net.adapter_10tok.predict_costs(
                        **gpu_kwargs, mm_tokens_10x32=tok, mm_mask_10=mask
                    )["total_cost"]
                    .float()[0]
                    .cpu()
                    .numpy()
                )
            if not np.isfinite(out).all():
                raise RuntimeError("nonfinite native score")
            return out

        val_allowed = {r["context_sha256"] for r in val}
        native_val = []
        for row in val:
            check()
            cid = row["context_sha256"]
            native[cid] = native_forward(row)
            native_val.append(
                float(
                    pools.utility(cid, val_allowed, "NATIVE91")[
                        int(native[cid].argmin())
                    ]
                )
            )
        reference = json.loads(Path(pins["reference"]).read_text())["utility"][
            "per_context"
        ]
        try:
            sci.require_native_reference(native_val, reference)
        except RuntimeError:
            save(
                args.output / "native-failure.json",
                dict(observed=native_val, expected=reference),
            )
            raise
        save(
            args.output / "baseline.json",
            dict(
                utilities=native_val,
                mean=float(np.mean(native_val)),
                bit_exact=True,
                contexts=[r["context_sha256"] for r in val],
            ),
        )
        for row in train:
            check()
            native[row["context_sha256"]] = native_forward(row)
        if rt.tensor_hash(net.state_dict()) != all_frozen:
            raise RuntimeError("frozen model drift")
        rt.check_frozen(net, frozen_guard, names, frozen)
        store.arrays("native-costs", **native)

        def parent_module(path, name):
            return load_module(
                args.repo, args.commit, path, name, pins["parent_sources"][path]
            )

        treatment = parent_module(
            "research/flywheel_selfplay/h177_pooled_selector/treatment.py",
            "h215_treatment",
        )
        selector = parent_module(
            "research/flywheel_selfplay/h177_pooled_selector/analysis.py",
            "h215_selector",
        )
        elo = parent_module(
            "research/flywheel_selfplay/h200_nested_blend/analysis.py", "h215_elo"
        )
        calibration_path = (
            "research/flywheel_selfplay/h178_pooled_calibration/h178_receipt.json"
        )
        calibration_raw = source(args.repo, args.commit, calibration_path)
        if digest(calibration_raw) != pins["parent_sources"][calibration_path]:
            raise RuntimeError("calibration receipt pin mismatch")
        calibration = json.loads(calibration_raw)
        if (
            calibration["h177_analysis_sha256"]
            != pins["parent_sources"][
                "research/flywheel_selfplay/h177_pooled_selector/analysis.py"
            ]
        ):
            raise RuntimeError("calibrated rule source mismatch")
        backend = load_module(
            args.repo,
            args.commit,
            PREFIX + "backend.py",
            "h215_backend",
            pins["source_sha256"]["backend.py"],
        )
        study = load_module(
            args.repo,
            args.commit,
            PREFIX + "study.py",
            "h215_study",
            pins["source_sha256"]["study.py"],
        )
        for name, parameter in net.named_parameters():
            parameter.requires_grad_(name in names)
        engine = backend.Engine(
            net, rt, treatment, pools, gpu_inputs, native, check, progress, store, names
        )
        for row in train:
            cid = row["context_sha256"]
            if not np.array_equal(engine.score(cid), native[cid]):
                raise RuntimeError(
                    "cached static r882 forward differs from native bits"
                )
        save(
            args.output / "static-parity.json",
            dict(contexts=len(train), bit_exact=True),
        )
        store.put(
            "static-parity.json", (args.output / "static-parity.json").read_bytes()
        )
        result = study.execute(
            engine,
            sci,
            selector,
            calibration,
            elo,
            train,
            store,
            rt,
            frozen_guard,
            frozen,
            progress,
            save,
            args.output,
            check,
        )
        # Persistent incumbent file and every pinned asset/input must remain unchanged.
        for kind in ("assets", "inputs"):
            for path, expected in pins[kind].items():
                if file_hash(path) != expected:
                    raise RuntimeError("postrun input/asset drift")
        result.update(
            source_commit=args.commit,
            elapsed_seconds=time.monotonic() - started,
            native_mean=float(np.mean(native_val)),
            baseline91_bitexact=True,
            frozen_hash_before_after=frozen,
            dope_sha256=pins["dope_composite_sha256"],
            numeric=numeric_check(),
            audit=audit.receipt(),
            remote=dict(store.receipts),
        )
        save(args.output / "metrics.json", result)
        store.put("metrics.json", (args.output / "metrics.json").read_bytes())
        print(
            json.dumps(
                {
                    k: result[k]
                    for k in (
                        "status",
                        "verdict",
                        "optimizer_updates",
                        "elapsed_seconds",
                    )
                }
            ),
            flush=True,
        )
    except BaseException as error:
        signal.setitimer(signal.ITIMER_REAL, 0)
        failure = dict(
            status="INCOMPLETE",
            verdict=None,
            error=repr(error),
            optimizer_updates=progress["updates"],
            elapsed_seconds=time.monotonic() - started,
            source_commit=args.commit,
            remote=store.receipts,
        )
        save(args.output / "failure.json", failure)
        try:
            store.put("failure.json", (args.output / "failure.json").read_bytes())
        except Exception:
            pass
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        resources.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
