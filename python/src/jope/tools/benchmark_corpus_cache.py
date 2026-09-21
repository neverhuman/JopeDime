#!/usr/bin/env python3
"""One-CPU corpus benchmark on pointer-only samples of completed GP ledgers."""

import argparse
import hashlib
import json
import os
import pathlib
import resource
import signal
import subprocess
import time


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=pathlib.Path, required=True)
    p.add_argument("--binary", type=pathlib.Path, required=True)
    p.add_argument("--phase", choices=["baseline", "baseline-cpu", "cached"], required=True)
    a = p.parse_args()
    root = a.root.resolve()
    os.chdir(root)
    sample = root / "runs/parity/corpus-cache-v1"
    frozen = root / "runs/regression-cycle-2/corpus-2/provenance.json"
    manifest = sample / "sample.json"
    if not manifest.exists():
        sources = json.loads(frozen.read_text())["sources"]
        legacy = [r for r in sources if r["path"].startswith("runs/gp-qualification/")][:16]
        fresh = [r for r in sources if r["path"].startswith("runs/regression-cycle-2/")][:16]
        assert len(legacy) == len(fresh) == 16
        selected = legacy + fresh + legacy[:4] + fresh[:4]
        rows = []
        for i, row in enumerate(selected):
            source = root / row["path"]
            assert sha(source) == row["sha256"]
            directory = sample / "ledgers" / f"{i:03d}"
            directory.mkdir(parents=True)
            for name in ["observations.json", "search.json"]:
                (directory / name).symlink_to(source.parent / name)
            rows.append(
                dict(
                    path=row["path"],
                    sha256=row["sha256"],
                    observations=row["observations"],
                    sample_path=str(directory.relative_to(root)),
                )
            )
        manifest.write_text(
            json.dumps(
                dict(
                    schema="parallel.corpus_benchmark_sample.v1",
                    frozen_provenance_sha256=sha(frozen),
                    inputs=rows,
                    ledger_payloads_copied=0,
                    final_tests_opened=False,
                ),
                indent=2,
            )
            + "\n"
        )
    info = json.loads(manifest.read_text())
    for row in info["inputs"]:
        assert sha(root / row["path"]) == row["sha256"]
    output = sample / a.phase
    assert not output.exists(), "Use an immutable new phase output"
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        RAYON_NUM_THREADS="1",
    )
    command = [
        "taskset",
        "-c",
        "60",
        str(a.binary),
        "--root",
        str(root),
        "build-corpus",
        "--contexts",
        str(root / "data/encodings/contexts-union-v2.json"),
        "--vocabulary",
        str(root / "runs/regression-cycle-2/corpus-2/vocabulary.json"),
        "--observations",
        str(sample / "ledgers"),
        "--output",
        str(output),
        "--qualification",
    ]
    began = time.perf_counter()
    with (sample / f"{a.phase}.log").open("w") as log:
        child = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True
        )
        try:
            code = child.wait(timeout=120)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
            raise
    elapsed = time.perf_counter() - began
    used = resource.getrusage(resource.RUSAGE_CHILDREN)
    assert code == 0, (sample / f"{a.phase}.log").read_text()
    outputs = {
        p.name: dict(sha256=sha(p), bytes=p.stat().st_size) for p in sorted(output.iterdir()) if p.is_file()
    }
    receipt = dict(
        schema="parallel.corpus_cache_benchmark.v1",
        phase=a.phase,
        sample_sha256=sha(manifest),
        executable_sha256=sha(a.binary),
        wall_seconds=elapsed,
        user_cpu_seconds=used.ru_utime,
        system_cpu_seconds=used.ru_stime,
        peak_rss_kib=used.ru_maxrss,
        cpu=60,
        threads=1,
        timeout_seconds=120,
        gpu_used=False,
        final_tests_opened=False,
        outputs=outputs,
    )
    if a.phase == "cached":
        baseline = json.loads((sample / "baseline-cpu-receipt.json").read_text())
        receipt["byte_identical_all_artifacts"] = outputs == baseline["outputs"]
        assert receipt["byte_identical_all_artifacts"], "Corpus artifacts changed"
        receipt["wall_speedup"] = baseline["wall_seconds"] / elapsed
        receipt["cpu_speedup"] = (baseline["user_cpu_seconds"] + baseline["system_cpu_seconds"]) / (
            used.ru_utime + used.ru_stime
        )
    (sample / f"{a.phase}-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
