"""Manual unscored model-construction check; no dataset, r882 or forward access."""

from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
import psutil
from monitor import sample, reason, ARROW_ALLOCATOR_CONF

SOURCE = Path("/home/ubuntu/jope-data/h50/input/cpu_exact_fused.py")
SOURCE_SHA = "b9f1468a61aa4dc888c8f2931f9c759f7d4f057793a77ccb05eed6300fb4f9ab"


def worker(out):
    if hashlib.sha256(SOURCE.read_bytes()).hexdigest() != SOURCE_SHA:
        raise RuntimeError("constructor source changed")

    def event(stage, **kw):
        value = dict(
            stage=stage, threads=psutil.Process().num_threads(), utc=time.time(), **kw
        )
        with (out / "events.jsonl").open("a") as f:
            f.write(json.dumps(value) + "\n")

    def audit(name, args):
        if name != "open" or not isinstance(args[0], (str, bytes)):
            return
        p = Path(os.fsdecode(args[0])).resolve()
        if p.is_relative_to(Path("/home/ubuntu/jope-data")) and p != SOURCE:
            raise RuntimeError("preflight forbids dataset and r882 input")
        if any(
            t in str(p).lower()
            for t in (
                "/sealed",
                "/holdout",
                "meta-test",
                "meta_test",
                "qualification.lock",
            )
        ):
            raise RuntimeError("protected input")

    sys.addaudithook(audit)
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    original = torch.load

    def load(*args, **kwargs):
        event("model_asset_before", path=str(args[0]))
        value = original(*args, **kwargs)
        event("model_asset_after", path=str(args[0]))
        return value

    torch.load = load
    spec = importlib.util.spec_from_file_location("h135_constructor_preflight", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    event("source_imported")
    net = module.FusedJopeNetwork(device=torch.device("cpu"), adapter_variant="h40b")
    device = str(next(net.parameters()).device)
    if device != "cpu":
        raise RuntimeError("CPU constructor required")
    event(
        "constructor_complete",
        device=device,
        parameters=sum(p.numel() for p in net.parameters()),
        predictions=0,
        optimizer_updates=0,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--worker", action="store_true")
    args = p.parse_args()
    out = args.output.resolve()
    if args.worker:
        worker(out)
        return
    if out.is_relative_to(Path(__file__).resolve().parents[3]) or out.is_relative_to(
        Path("/tmp")
    ):
        raise ValueError("output must be external owned custody")
    pre = sample(out)
    if reason(pre):
        raise RuntimeError(reason(pre))
    out.mkdir(exist_ok=False)
    env = dict(
        os.environ,
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
        NUMEXPR_MAX_THREADS="1",
        JE_ARROW_MALLOC_CONF=ARROW_ALLOCATOR_CONF,
        CUDA_VISIBLE_DEVICES="",
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONPYCACHEPREFIX=str(out / "bytecode"),
    )
    receipt = dict(
        kind="unscored model construction; pretrained assets read, no datasets or r882",
        source_sha256=SOURCE_SHA,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        samples=[],
        error=None,
    )
    with (out / "worker.log").open("w") as log:
        child = subprocess.Popen(
            [
                "nice",
                "-n",
                "19",
                sys.executable,
                __file__,
                "--worker",
                "--output",
                str(out),
            ],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        proc = psutil.Process(child.pid)
        try:
            while child.poll() is None:
                try:
                    value = sample(out, proc)
                except psutil.NoSuchProcess:
                    break
                receipt["samples"].append(value)
                bad = reason(value)
                if bad:
                    receipt["error"] = bad
                    os.killpg(child.pid, signal.SIGTERM)
                    child.wait(timeout=10)
                    break
                time.sleep(0.1)
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGKILL)
            receipt["exit_status"] = child.wait()
    events = (
        [json.loads(x) for x in (out / "events.jsonl").read_text().splitlines()]
        if (out / "events.jsonl").exists()
        else []
    )
    receipt["status"] = (
        "PASS"
        if receipt["exit_status"] == 0
        and not receipt["error"]
        and events
        and events[-1]["stage"] == "constructor_complete"
        else "FAIL"
    )
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    if receipt["status"] != "PASS":
        raise RuntimeError("constructor preflight failed")
    print("PASS", max(s["threads"] for s in receipt["samples"]), flush=True)


if __name__ == "__main__":
    main()
