"""Read pinned H212/H139/H124 source from Git; construct frozen CPU inference only."""

from __future__ import annotations
import contextlib
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import types

REPO = Path("/home/ubuntu/JopeDime")
PREFIX = "research/flywheel_selfplay/h227_pool_headroom/"
SOURCE = os.environ.get("H227_SOURCE")
DATA = Path.home() / "jope-data/h227"
R882 = DATA / "input/round-882.pt"
REFERENCE_DIR = Path.home() / "jope-data/h195/inputs"
DOPE = "c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814"
R882_SHA = "0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64"


def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def blob(path, source=None):
    return subprocess.check_output(
        ["git", "show", (source or SOURCE) + ":" + path], cwd=REPO
    )


def module(path, name, source=None, expected=None):
    if not expected or len(expected) != 64:
        raise ValueError("source SHA-256 pin required: " + path)
    raw = blob(path, source)
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("pinned source changed: " + path)
    m = types.ModuleType(name)
    m.__file__ = str(REPO / path)
    sys.modules[name] = m
    exec(compile(raw, m.__file__, "exec"), m.__dict__)
    return m


def registration():
    return json.loads(blob(PREFIX + "manifest.json"))


def write(path, value):
    path = Path(path)
    temporary = Path(str(path) + ".new")
    with temporary.open("w") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    temporary.replace(path)


def relocate(h139, h124, input_root, checkpoint, reference):
    h139.INPUT, h139.ARM_DIR = input_root, reference
    h124.INPUT, h124.R882 = input_root, checkpoint
    h124.FILES[checkpoint] = R882_SHA


def ensure_scratch(path):
    Path(path).mkdir(parents=True, exist_ok=True)
    return Path(path)


def validate_envelope(raw, cached, expected, envelope_fn):
    if hashlib.sha256(raw).hexdigest() != expected or cached != envelope_fn(raw):
        raise ValueError("raw TRAIN identity or CV-envelope bytes differ")


def baseline_parity(values, reference):
    import numpy as np

    values = np.asarray(values, dtype=np.float64)
    expected = np.asarray(reference["utility"]["per_context"], dtype=np.float64)
    if (
        values.shape != (91,)
        or expected.shape != (91,)
        or values.tobytes() != expected.tobytes()
        or float(values.mean()) != 1.1377941898164525
    ):
        raise ValueError(
            "r882 requires all91 bit-exact and exact mean before proposals"
        )
    return dict(status="PASS", bit_exact_contexts=91, mean=float(values.mean()))


@contextlib.contextmanager
def single_thread_torch(torch):
    setter = torch.set_num_threads
    setter(1)
    torch.set_num_threads = lambda _n: setter(1)
    try:
        yield
    finally:
        torch.set_num_threads = setter
        setter(1)


def setup(root, phase):
    import numpy as np
    import torch

    doc = registration()
    parent = doc["source_parent"]
    for path, expected in doc["input_pins"].items():
        if digest(path) != expected:
            raise ValueError("input changed: " + path)
    if digest(R882) != R882_SHA:
        raise ValueError("owned r882 COPY differs")
    pins = doc["source_pins"]
    upstream_stat = module(
        "research/flywheel_selfplay/h212_pool_width/analysis.py",
        "h227_h212_analysis",
        parent,
        pins["research/flywheel_selfplay/h212_pool_width/analysis.py"],
    )
    previous = sys.modules.get("analysis")
    sys.modules["analysis"] = upstream_stat
    try:
        h212 = module(
            "research/flywheel_selfplay/h212_pool_width/runner.py",
            "h227_h212",
            parent,
            pins["research/flywheel_selfplay/h212_pool_width/runner.py"],
        )
    finally:
        if previous is None:
            sys.modules.pop("analysis", None)
        else:
            sys.modules["analysis"] = previous
    h139 = module(
        "research/flywheel_selfplay/h139_selection_audit/runner.py",
        "h227_h139",
        parent,
        h212.H139_SHA,
    )
    h124 = module(
        "research/flywheel_selfplay/h124_gbm_shrink_blend/runner.py",
        "h227_h124",
        parent,
        h212.H124_SHA,
    )
    relocate(h139, h124, h212.INPUT, R882, REFERENCE_DIR)
    train, val, reference = h139.roster(h124)
    if (
        len({r["family_id"] for r in train}) != 205
        or len({r["family_id"] for r in val}) != 19
        or {r["family_id"] for r in train} & {r["family_id"] for r in val}
    ):
        raise ValueError("approved disjoint TRAIN205 / Val19 families required")
    closure = json.loads((h212.INPUT / "source_manifest.json").read_text())
    if closure["excluded_payloads_opened"] != 0 or closure["outer_rows_delivered"] != 0:
        raise ValueError("read boundary changed")
    # H124's cache identity validator needs the full approved roster, even when only92 rows are scored.
    approved = [*val, *train]
    context, reads = h124.train_context(approved, [])
    projection = h124.train_projection(approved, [])
    selected = [*val, train[0]]
    pools = h139.pools_for(selected)
    with single_thread_torch(torch):
        costs, net = h124.baseline(root, selected, context, pools, projection)
    native = np.asarray(
        [
            h139.native(
                costs[r["context_sha256"]],
                pools[r["context_sha256"]]["fitness"],
                pools[r["context_sha256"]]["gate"],
            )
            for r in val
        ]
    )
    parity = baseline_parity(native, reference)
    write(root / "baseline-parity.json", parity)
    if (
        net.training
        or any(p.requires_grad for p in net.parameters())
        or torch.cuda.is_initialized()
    ):
        raise ValueError("frozen CPU inference required")
    generator = h212.load_source(
        h212.GENERATOR_PATH,
        "h227_generator",
        doc["external_source_pins"][str(h212.GENERATOR_PATH)],
    )
    if digest(generator.WORKER_BIN) != generator.WORKER_SHA256:
        raise ValueError("native worker pin differs")
    envelope = module(
        "research/flywheel_selfplay/automatic_promotion.py",
        "h227_envelope",
        parent,
        h212.H58_ENVELOPE_SHA,
    )
    champions = json.loads((h212.CACHE / "champions_featurized.json").read_text())
    if (
        len(champions) != 1219
        or {r["context_sha256"] for r in champions}
        != {r["context_sha256"] for r in train}
        or any(r["dl_split"] != "DL-Train" for r in champions)
    ):
        raise ValueError("complete TRAIN-only champion bank required")
    generator.EVAL_SCRATCH_ROOT = ensure_scratch(root / "rust_eval")
    write(
        root / "input-read-receipt.json",
        dict(
            approved_context_features=len(approved),
            baseline_contexts=91,
            treatment_split="DL-Train" if phase == "smoke" else "DL-Val",
            sealed_holdout_meta_test=0,
            outer_rows=0,
            context_reads=reads,
        ),
    )
    return dict(
        phase=phase,
        h212=h212,
        h139=h139,
        h124=h124,
        generator=generator,
        envelope=envelope,
        train=train,
        val=val,
        context=context,
        projection=projection,
        pools=pools,
        net=net,
        champions=champions,
        baseline=native,
    )


def freeze(runtime):
    import torch

    helper = runtime["h124"].source_module(
        "python/src/jope/north_star/dope_freeze_guard.py", "h227_freeze"
    )
    pin = helper.snapshot_dope_weights(runtime["net"]).composite_sha256
    if pin != DOPE or digest(R882) != R882_SHA or torch.cuda.is_initialized():
        raise ValueError("DOPE/r882/CPU freeze failed")
    return dict(
        dope_sha256=pin,
        r882_sha256=R882_SHA,
        optimizer_updates=0,
        CUDA_initialized=False,
    )
