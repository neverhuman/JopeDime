"""Current-source CPU measurement and strict four-clause M3 comparison."""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import signal
import struct
import subprocess
import time

from common import ROOT, ProviderError, sha256

CATALOG = "a16efb5c3a8bd09ea08caaaf030c199f5d2d733153874d112b928a3f1cd2bb86"
R0_EVALUATOR = "e3be938644722d14a1c1a0e82168518838c4d174cfe031076a339745c48870a8"
NAMES = "research/flywheel_selfplay/custody/files/home/ubuntu/JopeDimeMicro/assets/projections/program-feature-names.json"
NAMES_SHA = "c4e4373d1bc7e99bc38e44eb4acda22d0f9ecc9e332fa8cb9c6a0b463fe37962"
ENV = dict(
    CARGO_BUILD_JOBS="4",
    CARGO_INCREMENTAL="0",
    CARGO_PROFILE_DEV_DEBUG="line-tables-only",
    OMP_NUM_THREADS="1",
    OPENBLAS_NUM_THREADS="1",
    MKL_NUM_THREADS="1",
    RAYON_NUM_THREADS="1",
    CUDA_VISIBLE_DEVICES="",
)


# Trusted data pins from the accepted R5/RES-01 reconstruction, not caller assertions.
SELECTED_SHA = "981960ad628f5029f8c54552c7b1cb1d440072e7e9dcffecdea52c6a3493a0f1"
OPERATOR_FAMILIES = {
    "champion_retrieval",
    "edit_parameter",
    "edit_replace_head",
    "edit_insert_stage",
    "edit_delete_stage",
    "edit_crossover",
    "landmark_synthesis",
}
PROPOSER_PINS = {
    "admission": "52df3263b45b90fa39304ef45fd9a791eacc0554f4e44e844240a8a822164e8f",
    "bank": "e0cea61862ef6949f9b36c58f0631e48370733c8ff7086aa56b383305b04af58",
    "cohort": "c08ee9abe87ec72bd35e616f81d9035860e5e8298fc9eb13d604e080960bfe45",
    "native_reference": "7c3e211edaa5d5ead121a1c5b6563aad9b6d8e3bd1d6d3a46950db4d58334840",
    "requests": "a0bfcee8c110e2321f381533ab478b76aab22118d2aa799c937feb49df2a4043",
    "vectors": "c115ca08095f00b31e637a6a3fdf46593c90c8d31fbb9a939efa9692073aac18",
}


def require(ok, reason):
    if not ok:
        raise ProviderError("M3 full: " + reason)


def regular(path, cap=64 * 1024 * 1024):
    path = Path(path)
    require(
        path.is_absolute()
        and path.is_file()
        and not path.is_symlink()
        and not any(p.is_symlink() for p in path.parents),
        "absolute regular non-symlink file required",
    )
    require(path.stat().st_size <= cap, "input byte limit exceeded")
    return path


def pinned(spec):
    path = regular(spec["path"])
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        raw = stream.read(64 * 1024 * 1024 + 1)
        after = os.fstat(stream.fileno())
    require(len(raw) <= 64 * 1024 * 1024, "input byte limit exceeded")
    regular(path)
    current = path.stat()
    require(
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        == (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns),
        "input changed during read",
    )
    require(
        hashlib.sha256(raw).hexdigest() == spec["sha256"],
        "input hash differs: " + path.name,
    )
    return raw


def jsonlines(raw):
    return [json.loads(line) for line in raw.splitlines() if line]


def numeric(value):
    return type(value) in (int, float) and math.isfinite(value)


def tensor(raw, name):
    size = struct.unpack_from("<Q", raw)[0]
    require(0 < size <= len(raw) - 8, "invalid tensor header")
    header = json.loads(raw[8 : 8 + size])
    spec = header[name]
    require(spec["dtype"] == "F32", "F32 oracle required")
    start, end = spec["data_offsets"]
    require(
        0 <= start <= end <= len(raw) - 8 - size and (end - start) % 4 == 0,
        "invalid tensor offsets",
    )
    values = [
        x[0] for x in struct.iter_unpack("<f", raw[8 + size + start : 8 + size + end])
    ]
    require(len(values) == math.prod(spec["shape"]), "tensor shape differs")
    return spec["shape"], values


def reference_pins(cfg):
    require(
        cfg["selected"]["sha256"] == SELECTED_SHA,
        "approved R0 TRAIN roster pin required",
    )
    require(
        set(cfg["proposer"]) == set(PROPOSER_PINS)
        and all(
            cfg["proposer"][name]["sha256"] == pin
            for name, pin in PROPOSER_PINS.items()
        ),
        "accepted R5 proposer witness pins required",
    )


def load_inputs(path, pin, preflight):
    raw = pinned(dict(path=str(path), sha256=pin))
    cfg = json.loads(raw)
    require(
        cfg.get("schema") == "jope.rel07.m3-full-inputs/v1",
        "full input manifest schema required",
    )
    require(
        cfg["catalog_sha256"] == CATALOG, "owner-accepted complete R5 catalog required"
    )
    store = Path(cfg["store"])
    inventory = preflight(store, CATALOG)
    catalog = json.loads(pinned(dict(path=str(store / CATALOG), sha256=CATALOG)))

    def load(name):
        spec = catalog["files"][name]
        return pinned(dict(path=str(store / spec["sha256"]), sha256=spec["sha256"]))

    cohort = jsonlines(load("dsl/source-rows.jsonl"))
    programs = jsonlines(load("dsl/programs.jsonl"))
    cv = json.loads(load("sources/train-cv.json"))
    require(
        len(cohort) == 1024
        and len(programs) == 1137
        and len({r["context_sha256"] for r in cohort}) == 132,
        "complete accepted cohort required",
    )
    require(
        all(
            r["dl_split"] == "DL-Train"
            and hashlib.sha256(r["dsl"].encode()).hexdigest() == r["program_sha256"]
            for r in cohort
        ),
        "TRAIN/source identity differs",
    )
    require(
        cv["protected_metrics_decoded"] is False and len(cv["rows"]) == 1024,
        "complete TRAIN CV projection required",
    )
    scores = {(r["journal_sha256"], r["journal_line"]): r for r in cv["rows"]}
    require(len(scores) == 1024, "duplicate CV identity")
    expected = []
    reference_pins(cfg)
    selected = json.loads(pinned(cfg["selected"]))
    roster = {r["context_sha256"]: r for r in selected["train"]}
    tables = Path(cfg["tables_root"])
    require(
        tables.is_absolute() and tables.resolve() == tables,
        "canonical TRAIN tables root required",
    )
    table_specs = {}
    for row in cohort:
        key = (row["journal_sha256"], row["journal_line"])
        score = scores[key]
        require(
            score["program_sha256"] == row["program_sha256"]
            and score["status"] == "completed"
            and numeric(score["fitness"])
            and type(score["cv_gate_passed"]) is bool,
            "training measurement identity/value differs",
        )
        cid = row["context_sha256"]
        selected_row = roster[cid]
        require(
            selected_row["dl_split"] == "DL-Train"
            and selected_row["source_split"] in ("train", "inner")
            and selected_row["task"] == "regression"
            and selected_row["family_id"] == row["family_id"],
            "table not in admitted TRAIN roster",
        )
        spec = dict(
            path=str(tables / (cid + ".jtt1")),
            sha256=selected_row["train_only_jtt1_sha256"],
        )
        if cid not in table_specs:
            data = pinned(spec)
            n, q, p = struct.unpack_from("<III", data, 4)
            require(
                data[:4] == b"JTT1"
                and data[16] == 2
                and q == 0
                and 50 <= n <= 2048
                and 0 < p <= 512
                and len(data) == 17 + n + 8 * n * (p + 1)
                and set(data[17 : 17 + n]) == {0, 1},
                "complete full-TRAIN fold table required",
            )
            table_specs[cid] = spec
        expected.append(dict(row=row, score=score, table=spec))
    proposer = {
        name: jsonlines(pinned(cfg["proposer"][name]))
        for name in ("cohort", "requests", "bank", "vectors", "native_reference")
    }
    admission = json.loads(pinned(cfg["proposer"]["admission"]))
    require(
        proposer["cohort"] == cohort
        and len(proposer["requests"]) == 256
        and len(proposer["bank"]) == 1620
        and len(proposer["native_reference"]) == 44544
        and len(admission["results"]) == 3072,
        "incomplete R5 proposer reconstruction",
    )
    names_raw = (ROOT / NAMES).read_bytes()
    require(
        hashlib.sha256(names_raw).hexdigest() == NAMES_SHA, "frozen name order changed"
    )
    names = json.loads(names_raw)
    require(len(names) == 412, "412 names required")
    compatibility = json.loads(load("rnk05-compat/index.json"))
    require(
        compatibility["contexts"] == 50,
        "50-context approved training extraction required",
    )
    for name in (n for n in compatibility["files"] if n.startswith("surrogate/")):
        # The evaluator identity is taken from original TRAIN measurements, never a caller boolean.
        source = json.loads(load("rnk05-compat/" + name))
        require(
            source["evaluator_sha256"] == R0_EVALUATOR
            and source["recipe"].startswith("full_train_cv/"),
            "R0 evaluator/protocol differs",
        )
    return dict(
        cfg=cfg,
        catalog=catalog,
        inventory=inventory,
        load=load,
        programs=programs,
        names=names,
        expected=expected,
        proposer=proposer,
        admission=admission,
        compatibility=compatibility,
    )


def compare_features(measurement, inputs):
    require(measurement["names"] == inputs["names"], "native feature order differs")
    rows = measurement["programs"]
    ids = [r["program_sha256"] for r in inputs["programs"]]
    require(
        [r["program_sha256"] for r in rows] == ids
        and ids
        and len(set(ids)) == len(ids),
        "complete ordered program coverage required",
    )
    expected = []
    end = 0
    for spec in inputs["catalog"]["dsl_coverage"]["canonical_tensor_shards"]:
        require(
            spec["start"] == end and spec["end"] > end,
            "contiguous full feature shards required",
        )
        raw = inputs["load"](spec["path"])
        shape, values = tensor(raw, "feat412")
        require(shape == [spec["end"] - end, 412], "feature shape differs")
        expected.extend([values[i : i + 412] for i in range(0, len(values), 412)])
        end = spec["end"]
    require(end == len(ids), "missing historical programs")
    ops = {n.removeprefix("count.") for n in inputs["names"] if n.startswith("count.")}
    signs = [
        [
            1
            if hashlib.sha256(
                b"jope-context-cross-v1\0" + struct.pack("<II", i, k)
            ).digest()[0]
            & 1
            else -1
            for k in range(16)
        ]
        for i in range(412)
    ]
    mismatches = 0
    projection_misses = 0
    maximum = 0.0
    pmaximum = 0.0
    exact = 0
    for observed, oracle in zip(rows, expected):
        require(
            len(observed["feat412"]) == 412 and len(observed["proj16"]) == 16,
            "native feature/projection shape differs",
        )
        for name, a, b in zip(inputs["names"], observed["feat412"], oracle):
            require(numeric(a) and numeric(b), "nonfinite feature")
            count_flag = (
                name.startswith(("count.", "estimator."))
                or name.endswith((".count", ".append.sum"))
                or name in ("steps", "dsl_characters")
                or (
                    name.startswith("position")
                    and name.split(".")[-1] in ops | {"append"}
                )
            )
            error = abs(a - b)
            maximum = max(maximum, error)
            exact += count_flag
            mismatches += error > (0 if count_flag else 1e-6)
        for k, a in enumerate(observed["proj16"]):
            b = sum(v * signs[i][k] for i, v in enumerate(oracle)) / math.sqrt(412)
            require(numeric(a) and numeric(b), "nonfinite projection")
            error = abs(a - b)
            pmaximum = max(pmaximum, error)
            projection_misses += error > 1e-12
    return dict(
        passed=mismatches == projection_misses == 0,
        programs=len(ids),
        coordinates=len(ids) * 412,
        count_flag_comparisons=exact,
        feature_mismatches=mismatches,
        max_feature_error=maximum,
        projection_comparisons=len(ids) * 16,
        projection_mismatches=projection_misses,
        max_projection_error=pmaximum,
    )


def compare_landmarks(measurements, inputs):
    files = sorted(
        n for n in inputs["compatibility"]["files"] if n.startswith("landmarks/")
    )
    require(
        len(files) == 50 and len(measurements) == 50,
        "full 50-context landmark sample required",
    )
    mismatches = 0
    count = 0
    maximum = 0.0
    for row, name in zip(measurements, files):
        require(row["context_sha256"] == Path(name).stem, "landmark context differs")
        raw = inputs["load"]("rnk05-compat/" + name)
        for key in ("v256", "col_full", "col"):
            _, oracle = tensor(raw, key)
            observed = row[key]
            require(len(oracle) == len(observed), "landmark shape differs")
            for a, b in zip(observed, oracle):
                require(numeric(a) and numeric(b), "nonfinite landmark")
                error = abs(a - b)
                maximum = max(maximum, error)
                mismatches += error > 1e-5 + 1e-4 * abs(b)
                count += 1
    return dict(
        passed=mismatches == 0,
        contexts=50,
        comparisons=count,
        mismatches=mismatches,
        max_absolute_error=maximum,
    )


def compare_evaluator(measurements, expected):
    require(
        len(measurements) == len(expected) and expected,
        "complete evaluator coverage required",
    )
    misses = []
    maximum = 0.0
    for i, (actual, source) in enumerate(zip(measurements, expected)):
        row = source["row"]
        score = source["score"]
        require(
            (actual["context_sha256"], actual["program_sha256"])
            == (row["context_sha256"], row["program_sha256"]),
            "evaluator identity differs",
        )
        a, b = actual.get("fitness"), score["fitness"]
        if actual.get("failure") is not None or not numeric(a):
            misses.append(dict(index=i, reason=actual.get("failure") or "unavailable"))
            continue
        error = abs(a - b)
        relative = error / abs(b) if b else (0.0 if error == 0 else math.inf)
        maximum = max(maximum, relative)
        if error > 1e-9 * abs(b) or (a < 100) != score["cv_gate_passed"]:
            misses.append(
                dict(
                    index=i,
                    observed=a,
                    expected=b,
                    relative_error=relative if math.isfinite(relative) else None,
                )
            )
    return dict(
        passed=not misses,
        evaluations=len(expected),
        contexts=len({r["row"]["context_sha256"] for r in expected}),
        evaluator_sha256=R0_EVALUATOR,
        protocol="full_train_cv/two-fold",
        mismatches=len(misses),
        max_relative_error=maximum if math.isfinite(maximum) else None,
        first_mismatches=misses[:20],
    )


def compare_proposer(generated, inputs):
    reference = inputs["proposer"]["native_reference"]
    require(
        generated == reference,
        "fresh native proposer differs from pinned same-forward oracle; new admission measurement required",
    )
    requests = inputs["proposer"]["requests"]
    by_call = {}
    for row in generated:
        by_call.setdefault(row["native_call"], []).append(row)
    selected = []
    for call, result in enumerate(inputs["admission"]["results"]):
        require(
            result["native_call"] == call and call in by_call,
            "missing or reordered native calls",
        )
        source = requests[call % len(requests)]
        original = {r["program_sha256"]: r for r in by_call[call]}
        costs = result["costs"]
        require(
            len({r["program_sha256"] for r in costs}) == len(costs)
            and all(
                r["program_sha256"] in original and numeric(r["cost"]) for r in costs
            ),
            "invalid admission costs",
        )
        order = sorted(range(len(costs)), key=lambda i: (costs[i]["cost"], i))
        if source["exploration"] and len(order) > 2:
            chosen = order[:2] + random.Random(
                source["source_round_seed"] + source["context_index"]
            ).sample(order[2:], min(2, len(order) - 2))
        else:
            chosen = order[:4]
        actual = [costs[i]["program_sha256"] for i in chosen]
        require(
            actual == [r["program_sha256"] for r in result["selected"]],
            "same-forward admission selection differs",
        )
        for key in actual:
            selected.append(dict(original[key], exploration=source["exploration"]))
    require(len(selected) >= 10000, "10000 newly generated/admitted attempts required")
    selected = selected[:10000]
    ref = inputs["proposer"]["cohort"]
    require(
        {r["operator"] for r in ref} | {r["operator"] for r in selected}
        <= OPERATOR_FAMILIES,
        "unknown proposer operator family",
    )
    ops = sorted(
        {r["operator"] for r in ref}
        | {r["operator"] for r in selected}
        | {"landmark_synthesis"}
    )
    rows = []
    for mode in (None, False, True):
        a = Counter(
            r["operator"] for r in ref if mode is None or r["exploration"] == mode
        )
        b = Counter(
            r["operator"] for r in selected if mode is None or r["exploration"] == mode
        )
        na, nb = sum(a.values()), sum(b.values())
        require(na > 0 and nb > 0, "missing exploration stratum")
        values = {op: 100 * (b[op] / nb - a[op] / na) for op in ops}
        # Integer cross multiplication enforces exactly ±2 percentage points.
        passed = all(abs(b[op] * na - a[op] * nb) * 50 <= na * nb for op in ops)
        rows.append(
            dict(
                mode="all" if mode is None else mode,
                reference=na,
                native=nb,
                deltas_pp=values,
                reference_counts={op: a[op] for op in ops},
                native_counts={op: b[op] for op in ops},
                passed=passed,
            )
        )
    return dict(
        passed=all(r["passed"] for r in rows),
        newly_generated=len(generated),
        admitted=10000,
        strata=rows,
    )


def execute(args, *, env, timeout=1800):
    require(
        os.getloadavg()[0] <= min(85, 0.85 * (os.cpu_count() or 1)),
        "CPU admission unavailable",
    )
    with subprocess.Popen(
        ["nice", "-n", "19", *args],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    ) as child:
        try:
            stdout, stderr = child.communicate(timeout=timeout)
        except BaseException:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                child.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.communicate()
            raise
    return subprocess.CompletedProcess(args, child.returncode, stdout, stderr)


def build_probes(source, evidence, run):
    env = dict(
        os.environ,
        **ENV,
        JOPE_BUILD_SOURCE_SHA=source,
        TMPDIR=str(evidence / "runtime"),
    )
    (evidence / "runtime").mkdir()
    records = []

    def command(name, args, timeout=1800):
        start = time.monotonic()
        result = run(args, env=env, timeout=timeout)
        (evidence / (name + ".stdout")).write_text(result.stdout)
        (evidence / (name + ".stderr")).write_text(result.stderr)
        records.append(
            dict(
                name=name,
                command=args,
                exit=result.returncode,
                seconds=time.monotonic() - start,
            )
        )
        (evidence / "commands.json").write_text(json.dumps(records, indent=2) + "\n")
        require(result.returncode == 0, "measurement command failed: " + name)
        return result

    built = command(
        "build",
        [
            "cargo",
            "build",
            "--locked",
            "-p",
            "jope-service",
            "--lib",
            "--message-format=json",
        ],
    )
    deps = evidence / "dependencies"
    deps.mkdir()
    digests = {}
    libraries = {}
    for line in built.stdout.splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if row.get("reason") != "compiler-artifact":
            continue
        for name in row.get("filenames", []):
            path = Path(name)
            if path.suffix not in (".rlib", ".rmeta", ".so"):
                continue
            regular(path, 2**31)
            pin = sha256(path)
            dst = deps / path.name
            if dst.exists():
                require(digests[path.name] == pin, "ambiguous dependency name")
            else:
                with path.open("rb") as src, dst.open("xb") as out:
                    shutil.copyfileobj(src, out)
                require(
                    sha256(dst) == pin and sha256(path) == pin,
                    "dependency snapshot drift",
                )
                digests[path.name] = pin
            if path.suffix == ".rlib":
                libraries[row["target"]["name"]] = dst
    require("jope_service" in libraries, "normal current service library missing")
    (evidence / "dependencies.json").write_text(
        json.dumps(digests, sort_keys=True) + "\n"
    )
    bins = {}
    for name in ("m3_native", "m3_proposer"):
        binary = evidence / name
        cmd = [
            "rustc",
            "--edition=2021",
            "--crate-name=" + name,
            str(ROOT / "deploy/qualification/v1.0" / (name + ".rs")),
            "-L",
            "dependency=" + str(deps),
            "-o",
            str(binary),
        ]
        for crate in (
            "anyhow",
            "jope_service",
            "jope_runtime",
            "safetensors",
            "serde_json",
            "sha2",
            "feat_math",
        ):
            require(crate in libraries, "missing compiled dependency: " + crate)
            cmd += ["--extern", crate + "=" + str(libraries[crate])]
        command("compile-" + name, cmd)
        bins[name] = binary
    return bins, env, command, digests


def measure_full(source, path, pin, evidence, preflight, run=None):
    run = execute if run is None else run
    evidence = Path(evidence)
    require(
        evidence.is_absolute()
        and not evidence.exists()
        and not any(p.is_symlink() for p in evidence.parents),
        "new absolute evidence directory required",
    )
    require(
        not evidence.resolve().is_relative_to(ROOT.resolve()),
        "evidence must be outside source",
    )
    evidence.mkdir(parents=True, mode=0o700)
    inputs = load_inputs(path, pin, preflight)
    cfg = inputs["cfg"]
    bins, env, command, digests = build_probes(source, evidence, run)
    jobs = evidence / "jobs"
    jobs.mkdir()
    base = dict(store=cfg["store"], catalog_sha256=CATALOG)

    def native(mode, job, key):
        job_path = jobs / (key + ".json")
        job_path.write_text(json.dumps(job, sort_keys=True) + "\n")
        result = run(
            [str(bins["m3_native"]), mode, str(job_path)], env=env, timeout=120
        )
        (jobs / (key + ".stdout")).write_text(result.stdout)
        (jobs / (key + ".stderr")).write_text(result.stderr)
        require(result.returncode == 0, "native measurement failed: " + key)
        payload = json.loads(result.stdout)
        require(
            payload.get("schema") == "jope.rel07.m3-native-measurement/v1"
            and payload.get("source_sha") == source
            and payload.get("mode") == mode,
            "native source/schema identity differs",
        )
        return payload["measurement"]

    features = compare_features(native("features", base, "features"), inputs)
    (evidence / "features.json").write_text(json.dumps(features, sort_keys=True) + "\n")
    landmark_files = sorted(
        n for n in inputs["compatibility"]["files"] if n.startswith("landmarks/")
    )
    lm = []
    for i, name in enumerate(landmark_files):
        job = dict(base, fixture="rnk05-compat/" + name, context_sha256=Path(name).stem)
        lm.append(native("landmarks", job, f"landmark-{i:03}"))
    landmarks_result = compare_landmarks(lm, inputs)
    (evidence / "landmarks.json").write_text(
        json.dumps(landmarks_result, sort_keys=True) + "\n"
    )

    def evaluate(item):
        i, expected = item
        row = expected["row"]
        spec = expected["table"]
        return native(
            "evaluator",
            dict(
                table=spec["path"],
                table_sha256=spec["sha256"],
                context_sha256=row["context_sha256"],
                program_sha256=row["program_sha256"],
                dsl=row["dsl"],
            ),
            f"evaluator-{i:04}",
        )

    with ThreadPoolExecutor(max_workers=4) as pool:
        measured = list(pool.map(evaluate, enumerate(inputs["expected"])))
    evaluator_result = compare_evaluator(measured, inputs["expected"])
    (evidence / "evaluator.json").write_text(
        json.dumps(evaluator_result, sort_keys=True, allow_nan=False) + "\n"
    )
    staged = evidence / "proposer-input"
    staged.mkdir()
    out = evidence / "proposer-output"
    out.mkdir()
    for name, filename in (
        ("cohort", "cohort.jsonl"),
        ("requests", "requests-complete.jsonl"),
        ("bank", "bank-records.jsonl"),
        ("vectors", "raw-svd-all-train.jsonl"),
    ):
        (staged / filename).write_bytes(pinned(cfg["proposer"][name]))
    result = command("proposer", [str(bins["m3_proposer"]), str(staged), str(out)])
    diagnostic = json.loads(result.stdout)
    require(
        diagnostic["source_sha"] == source
        and diagnostic["native_calls"] == 3072
        and diagnostic["determinism_duplicate_calls_passed"] == 3072
        and diagnostic["excluded_operation_outputs"] == 0,
        "proposer replay incomplete",
    )
    generated = jsonlines((out / "native-admission-candidates.jsonl").read_bytes())
    proposer_result = compare_proposer(generated, inputs)
    (evidence / "proposer.json").write_text(
        json.dumps(proposer_result, sort_keys=True) + "\n"
    )
    for name, digest in digests.items():
        require(
            sha256(evidence / "dependencies" / name) == digest,
            "compiled dependency snapshot changed",
        )
    # Inputs remain bound at completion; no mutable witness may change under this run.
    load_inputs(path, pin, preflight)
    clauses = dict(
        feat412_proj16=features,
        training_landmarks=landmarks_result,
        full_train_cv=evaluator_result,
        r0_proposer=proposer_result,
    )
    report = dict(
        schema="jope.rel07.m3-full-measurement/v1",
        source_sha=source,
        inputs_sha256=pin,
        status="NUMERICAL_PASS"
        if all(v["passed"] for v in clauses.values())
        else "NUMERICAL_FAIL",
        scope="owner-accepted R5 reconstruction;2688 unlogged generated items excluded",
        original_historical_state_recovered=False,
        clauses=clauses,
        counts=dict(
            collected=4,
            executed=4,
            passed=sum(v["passed"] for v in clauses.values()),
            failed=sum(not v["passed"] for v in clauses.values()),
            skipped=0,
            excluded=0,
        ),
    )
    (evidence / "measurements.json").write_text(
        json.dumps(report, sort_keys=True, allow_nan=False) + "\n"
    )
    return report


def gate_payload(ctx, report, input_pin):
    """Account for measured assertions; reject flags contradicted by raw counts."""
    require(
        report.get("schema") == "jope.rel07.m3-full-measurement/v1"
        and report.get("source_sha") == ctx.source_sha
        and report.get("inputs_sha256") == input_pin,
        "measurement binding differs",
    )
    clauses = report.get("clauses")
    keys = {"feat412_proj16", "training_landmarks", "full_train_cv", "r0_proposer"}
    require(
        isinstance(clauses, dict) and set(clauses) == keys,
        "all four M3 clauses required",
    )
    for value in clauses.values():
        require(type(value.get("passed")) is bool, "strict clause boolean required")
    for key, fields in {
        "feat412_proj16": (
            "programs",
            "coordinates",
            "projection_comparisons",
            "count_flag_comparisons",
            "feature_mismatches",
            "projection_mismatches",
        ),
        "training_landmarks": ("contexts", "comparisons", "mismatches"),
        "full_train_cv": ("evaluations", "contexts", "mismatches"),
        "r0_proposer": ("newly_generated", "admitted"),
    }.items():
        require(
            all(
                type(clauses[key].get(field)) is int and clauses[key][field] >= 0
                for field in fields
            ),
            "strict nonnegative integer counts required",
        )
    feature = clauses["feat412_proj16"]
    landmark = clauses["training_landmarks"]
    evaluator = clauses["full_train_cv"]
    proposer = clauses["r0_proposer"]
    require(
        feature.get("programs") == 1137
        and feature.get("coordinates") == 1137 * 412
        and feature.get("projection_comparisons") == 1137 * 16
        and type(feature.get("count_flag_comparisons")) is int
        and feature["count_flag_comparisons"] > 0,
        "full historical feature/projection census required",
    )
    require(
        type(feature.get("feature_mismatches")) is int
        and type(feature.get("projection_mismatches")) is int
        and min(feature["feature_mismatches"], feature["projection_mismatches"]) >= 0,
        "invalid feature mismatch counts",
    )
    require(
        feature["passed"]
        == (feature["feature_mismatches"] == feature["projection_mismatches"] == 0),
        "feature pass/count contradiction",
    )
    if feature["passed"]:
        require(
            numeric(feature.get("max_feature_error"))
            and feature["max_feature_error"] <= 1e-6
            and numeric(feature.get("max_projection_error"))
            and feature["max_projection_error"] <= 1e-12,
            "feature/projection tolerance contradiction",
        )
    require(
        landmark.get("contexts") == 50
        and type(landmark.get("comparisons")) is int
        and landmark["comparisons"] > 0
        and type(landmark.get("mismatches")) is int
        and landmark["mismatches"] >= 0,
        "full landmark sample required",
    )
    require(
        landmark["passed"] == (landmark["mismatches"] == 0),
        "landmark pass/count contradiction",
    )
    require(
        evaluator.get("evaluations") == 1024
        and evaluator.get("contexts") == 132
        and evaluator.get("evaluator_sha256") == R0_EVALUATOR
        and evaluator.get("protocol") == "full_train_cv/two-fold"
        and type(evaluator.get("mismatches")) is int
        and evaluator["mismatches"] >= 0,
        "full identified evaluator replay required",
    )
    require(
        evaluator["passed"] == (evaluator["mismatches"] == 0),
        "evaluator pass/count contradiction",
    )
    if evaluator["passed"]:
        require(
            numeric(evaluator.get("max_relative_error"))
            and evaluator["max_relative_error"] <= 1e-9,
            "evaluator tolerance contradiction",
        )
    require(
        proposer.get("newly_generated") == 44544 and proposer.get("admitted") == 10000,
        "fresh comparable proposer population required",
    )
    strata = proposer.get("strata")
    require(
        isinstance(strata, list) and len(strata) == 3,
        "aggregate and both exploration strata required",
    )
    modes = [row.get("mode") for row in strata]
    require(
        modes == ["all", False, True] and all(type(x) is bool for x in modes[1:]),
        "strict unique exploration modes required",
    )
    derived = []
    for row in strata:
        a, b = row.get("reference_counts"), row.get("native_counts")
        require(
            isinstance(a, dict)
            and isinstance(b, dict)
            and set(a) == set(b) == OPERATOR_FAMILIES
            and all(type(x) is int and x >= 0 for x in [*a.values(), *b.values()]),
            "complete typed operator counts required",
        )
        require(
            type(row.get("reference")) is int and type(row.get("native")) is int,
            "strict population counts required",
        )
        na, nb = sum(a.values()), sum(b.values())
        require(
            na > 0
            and nb > 0
            and row.get("reference") == na
            and row.get("native") == nb,
            "operator population/count contradiction",
        )
        passed = all(abs(b[op] * na - a[op] * nb) * 50 <= na * nb for op in a)
        require(
            type(row.get("passed")) is bool and row["passed"] == passed,
            "operator pass/count contradiction",
        )
        derived.append(passed)
    require(
        strata[0]["reference"] == 1024 and strata[0]["native"] == 10000,
        "full proposer comparison census required",
    )
    for key in ("reference_counts", "native_counts"):
        require(
            all(
                strata[0][key][op]
                == strata[1][key].get(op, 0) + strata[2][key].get(op, 0)
                for op in strata[0][key]
            ),
            "exploration strata do not partition aggregate",
        )
    require(
        proposer["passed"] == all(derived),
        "pooled proposer pass cannot replace mode qualification",
    )
    passed = sum(value["passed"] for value in clauses.values())
    counts = dict(
        collected=4, executed=4, passed=passed, failed=4 - passed, skipped=0, excluded=0
    )
    require(
        isinstance(report.get("counts"), dict)
        and set(report["counts"]) == set(counts)
        and all(type(value) is int for value in report["counts"].values())
        and report["counts"] == counts
        and report.get("status")
        == ("NUMERICAL_PASS" if passed == 4 else "NUMERICAL_FAIL"),
        "report status/count contradiction",
    )
    return dict(
        schema="jope.rel07.gate-execution/v1",
        id="M3",
        status="PASS" if passed == 4 else "FAIL",
        source_sha=ctx.source_sha,
        artifact_sha256=ctx.artifact_sha256,
        counts=counts,
        inputs_sha256=input_pin,
        scope=report.get("scope"),
        clauses=clauses,
    )
