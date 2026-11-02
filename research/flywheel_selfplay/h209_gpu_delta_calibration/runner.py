"""H209 single guarded CUDA attempt; source read from immutable Git objects."""

import argparse
from collections import OrderedDict
import gc
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import types
import zipfile

import boto3
import numpy as np
import psutil
import torch

PREFIX = "research/flywheel_selfplay/h209_gpu_delta_calibration/"


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
    data = source(repo, commit, path)
    if digest(data) != expected:
        raise RuntimeError("source hash mismatch: " + path)
    mod = types.ModuleType(name)
    mod.__file__ = str(Path(repo) / path)
    sys.modules[name] = mod
    exec(compile(data, mod.__file__, "exec"), mod.__dict__)
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
        cores=len(os.sched_getaffinity(0)),
        cpu=psutil.cpu_percent(),
        available=psutil.virtual_memory().available,
        disk_used_fraction=disk.used / (disk.used + disk.free),
        free=disk.free,
        rss=sum(p.memory_info().rss for p in tree if p.is_running()),
        threads=sum(p.num_threads() for p in tree if p.is_running()),
    )


def guard(s):
    if (
        s["load"] > 0.85 * s["cores"]
        or s["cpu"] > 85
        or s["available"] < 16 * 2**30
        or s["disk_used_fraction"] > 0.938
        or s["free"] < 40_000_000_000
        or s["rss"] >= 6_000_000_000
    ):
        raise RuntimeError("resource refusal: " + repr(s))


def run(args):
    os.nice(19)
    if os.uname().nodename != "xbabe2" or os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("xbabe2 GPU0 only")
    psutil.cpu_percent()
    time.sleep(1)
    guard(snapshot())
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
    for path, expected in pins["source_sha256"].items():
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
        "h209_science",
        pins["source_sha256"]["science.py"],
    )
    thread_path = "research/flywheel_selfplay/hwave_numeric_threads.py"
    numeric = load_module(
        args.repo,
        args.commit,
        thread_path,
        "h209_numeric",
        pins["parent_sources"][thread_path],
    )
    runtime_path = "research/flywheel_selfplay/h146_lookahead_exec/runtime.py"
    rt = load_module(
        args.repo,
        args.commit,
        runtime_path,
        "h209_runtime",
        pins["parent_sources"][runtime_path],
    )
    rt.HERE = (
        args.repo
    )  # Used only as Git cwd by the immutable historical guard loader.

    def check(*unused):
        sample = snapshot()
        resources.write(
            json.dumps(dict(sample, elapsed=time.monotonic() - started)) + "\n"
        )
        resources.flush()
        guard(sample)
        if (
            time.monotonic() - started > 3600
            or sum(p.stat().st_size for p in args.output.rglob("*") if p.is_file())
            >= 300_000_000
        ):
            raise RuntimeError("fixed wall/output budget exceeded")

    def numeric_check():
        report = numeric.capture_numeric_threads(torch)
        admitted = numeric.admit_numeric_threads(
            [report], os_threads=psutil.Process().num_threads()
        )
        if admitted["numeric_threads_upper_bound"] != 1:
            raise RuntimeError("H209 requires one numeric thread")
        return admitted

    def interrupted(signum, frame):
        raise RuntimeError("interrupted; no retry")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    signal.signal(signal.SIGALRM, check)
    signal.setitimer(signal.ITIMER_REAL, 2, 2)
    store = Store(args.output, pins["s3_prefix"] + "attempt1/")
    updates = 0
    try:
        check()
        numeric_report = numeric_check()
        save(
            args.output / "launch.json",
            dict(
                wave="H209",
                source_commit=args.commit,
                manifest_sha256=digest(
                    source(args.repo, args.commit, PREFIX + "manifest.json")
                ),
                host=os.uname().nodename,
                device_uuid=pins["gpu_uuid"],
                numeric=numeric_report,
                start_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                deadline_seconds=3600,
                resource_limits=pins["resource_limits"],
                authority="gbw_2236_rd_astra.md",
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
                mm_tokens_10x32=torch.as_tensor(tokens[ix : ix + 1]),
                mm_mask_10=torch.as_tensor(masks[ix : ix + 1]),
            )
            with torch.no_grad():
                out = (
                    net.adapter_10tok.predict_costs(
                        **{k: v.to("cuda:0") for k, v in kwargs.items()}
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
        net = None
        gc.collect()
        torch.cuda.empty_cache()
        store.put("baseline.json", (args.output / "baseline.json").read_bytes())
        states, locks, proofs = {}, [], []

        def x_for(cid):
            ix = position[cid]
            return torch.as_tensor(
                sci.features(
                    pools.predictors(cid)[0], tokens[ix], masks[ix], native[cid]
                ),
                device="cuda:0",
            )

        def score(head, cid):
            with torch.no_grad():
                return head(x_for(cid), int(native[cid].argmin())).cpu().numpy()

        for seed in sci.SEEDS:
            for outer in range(5):
                fit, b_rows, _ = sci.partition(train, seed, outer)
                fit_ids = {r["context_sha256"] for r in fit}
                b_ids = {r["context_sha256"] for r in b_rows}
                family_rows = {
                    f: sorted(r["context_sha256"] for r in fit if r["family_id"] == f)
                    for f in sorted({r["family_id"] for r in fit})
                }
                rng = np.random.default_rng(sci.stream(seed, outer, "plan"))
                families = sorted(family_rows)
                plan = [
                    [
                        str(rng.choice(family_rows[str(rng.choice(families))]))
                        for _ in range(4)
                    ]
                    for _ in range(128)
                ]
                for label in ("REAL", "SHUFFLE"):
                    numeric_check()
                    torch.manual_seed(sci.stream(seed, outer, "init") % (2**63 - 1))
                    head = sci.DeltaHead().to("cuda:0")
                    before = rt.tensor_hash(head.state_dict())
                    witness = plan[0][0]
                    before_score = score(head, witness)
                    optimizer = torch.optim.AdamW(
                        head.parameters(), lr=0.001, weight_decay=0.0001
                    )
                    losses, gradient_sum = [], 0.0
                    for step, cids in enumerate(plan):
                        check()
                        optimizer.zero_grad(set_to_none=True)
                        loss_value = 0.0
                        for cid in cids:
                            u = pools.utility(cid, fit_ids, "FIT")
                            if label == "SHUFFLE":
                                u = np.random.default_rng(
                                    sci.stream(seed, outer, cid, "shuffle")
                                ).permutation(u)
                            pred = head(x_for(cid), int(native[cid].argmin()))
                            loss = (
                                sci.balanced_loss(
                                    pred,
                                    torch.as_tensor(
                                        u, device="cuda:0", dtype=torch.float32
                                    ),
                                    int(native[cid].argmin()),
                                )
                                / 4
                            )
                            if not torch.isfinite(loss):
                                raise RuntimeError("nonfinite loss")
                            loss.backward()
                            loss_value += float(loss.detach())
                        grad = sum(
                            float(p.grad.square().sum())
                            for p in head.parameters()
                            if p.grad is not None
                        )
                        if not np.isfinite(grad):
                            raise RuntimeError("nonfinite gradient")
                        gradient_sum += grad
                        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                        optimizer.step()
                        updates += 1
                        losses.append(loss_value)
                    after = rt.tensor_hash(head.state_dict())
                    if (
                        before == after
                        or gradient_sum <= 0
                        or np.array_equal(before_score, score(head, witness))
                    ):
                        raise RuntimeError("no actual learned forward change")
                    proof = dict(
                        seed=seed,
                        outer=outer,
                        arm=label,
                        updates=128,
                        gradient_squared_sum=gradient_sum,
                        before_sha256=before,
                        after_sha256=after,
                        forward_changed=True,
                        actual_device=str(next(head.parameters()).device),
                        losses=losses,
                    )
                    proofs.append(proof)
                    state = {
                        k: v.detach().cpu().clone()
                        for k, v in head.state_dict().items()
                    }
                    states[(seed, outer, label)] = state
                    payload = io.BytesIO()
                    torch.save(state, payload)
                    store.put(f"heads/{seed}-{outer}-{label}.pt", payload.getvalue())
                    gains = {t: [] for t in sci.THRESHOLDS}
                    scores = {}
                    for row in b_rows:
                        cid = row["context_sha256"]
                        delta = score(head, cid)
                        scores[cid] = delta
                        ref = int(native[cid].argmin())
                        u = pools.utility(cid, b_ids, "B")
                        for t in sci.THRESHOLDS:
                            gains[t].append(float(u[sci.pick(delta, ref, t)] - u[ref]))
                    chosen, grid = sci.select_threshold(
                        gains, [r["family_id"] for r in b_rows]
                    )
                    store.arrays(f"B/{seed}-{outer}-{label}", **scores)
                    locks.append(
                        dict(
                            seed=seed,
                            outer=outer,
                            arm=label,
                            threshold=None if np.isinf(chosen) else chosen,
                            grid=grid,
                            head_sha256=after,
                        )
                    )
                    del optimizer, head
        lock = dict(
            cells=locks,
            proofs=proofs,
            source_commit=args.commit,
            remote=dict(store.receipts),
            updates=updates,
        )
        save(args.output / "selection-lock.json", lock)
        lock_hash = file_hash(args.output / "selection-lock.json")
        store.put(
            "selection-lock.json", (args.output / "selection-lock.json").read_bytes()
        )
        panels = {arm: np.full((3, len(train)), np.nan) for arm in sci.ARMS}
        index = {r["context_sha256"]: i for i, r in enumerate(train)}
        adoption = {arm: [0, 0, 0] for arm in sci.ARMS[1:]}
        for cell in locks:
            check()
            if file_hash(args.output / "selection-lock.json") != lock_hash:
                raise RuntimeError("selection lock mutated")
            seed, outer, label = cell["seed"], cell["outer"], cell["arm"]
            _, _, a_rows = sci.partition(train, seed, outer)
            a_ids = {r["context_sha256"] for r in a_rows}
            head = sci.DeltaHead().to("cuda:0")
            head.load_state_dict(states[(seed, outer, label)], strict=True)
            if rt.tensor_hash(head.state_dict()) != cell["head_sha256"]:
                raise RuntimeError("terminal head identity mismatch")
            tau = float("inf") if cell["threshold"] is None else cell["threshold"]
            saved = {}
            for row in a_rows:
                cid = row["context_sha256"]
                si, i = sci.SEEDS.index(seed), index[cid]
                delta = score(head, cid)
                saved[cid] = delta
                ref = int(native[cid].argmin())
                u = pools.utility(cid, a_ids, "A", locked=True)
                panels["R882"][si, i] = u[ref]
                names_and_thresholds = (
                    (("DELTA_RAW", 0.0), ("DELTA_SAFE", tau))
                    if label == "REAL"
                    else (("SHUF_SAFE", tau),)
                )
                for arm, t in names_and_thresholds:
                    selected = sci.pick(delta, ref, t)
                    panels[arm][si, i] = u[selected]
                    adoption[arm][si] += int(selected != ref)
            store.arrays(f"A/{seed}-{outer}-{label}", **saved)
        if any(not np.isfinite(p).all() for p in panels.values()) or updates != 3840:
            raise RuntimeError("incomplete endpoint/update coverage")
        families = [r["family_id"] for r in train]
        contrasts = {
            arm: sci.contrast(panels[arm], panels["R882"], families)
            for arm in sci.ARMS[1:]
        }
        elo_path = "research/flywheel_selfplay/h200_nested_blend/analysis.py"
        elo_module = load_module(
            args.repo,
            args.commit,
            elo_path,
            "h209_elo",
            pins["parent_sources"][elo_path],
        )
        elo = {
            arm: elo_module.elo_vs_reference(
                panels[arm], panels["R882"], families, seed=209151
            )
            for arm in sci.ARMS[1:]
        }
        store.arrays(
            "panels",
            **panels,
            context_ids=np.array([r["context_sha256"] for r in train]),
            family_ids=np.array(families),
        )
        result = dict(
            status="COMPLETE",
            verdict=sci.verdict(contrasts["DELTA_SAFE"]),
            source_commit=args.commit,
            contrasts=contrasts,
            elo_vs_r882=elo,
            adoption_counts=adoption,
            adoption_rates={
                k: [n / len(train) for n in v] for k, v in adoption.items()
            },
            val_reg_inner_oof={
                arm: dict(
                    context_mean_per_seed=panel.mean(1).tolist(),
                    equal_family_mean_per_seed=sci.family_mean(panel, families)
                    .mean(1)
                    .tolist(),
                )
                for arm, panel in panels.items()
            },
            U4=None,
            U4_status="not measured; no verified multi-proposal budget",
            contexts_per_seed=len(train),
            optimizer_updates=updates,
            baseline91_bitexact=True,
            native_mean=float(np.mean(native_val)),
            selection_lock_sha256=lock_hash,
            elapsed_seconds=time.monotonic() - started,
            frozen_hash=all_frozen,
            numeric=numeric_check(),
            audit=audit.receipt(),
            remote=store.receipts,
            serving_promotion=False,
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
            optimizer_updates=updates,
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
