"""Reviewed source for API01's one existing Q1 dispatcher.

Derived from api01-t0250/launch.py SHA256 68d8e8135ac7d029a5fefe5ffdc9acf02e496224cf5d1c819da4fee99a87ba91.
Install exact reviewed bytes at its original path; fixed state/ledger stay unchanged.
"""

from pathlib import Path
import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys
import time

E = Path("/home/ubuntu/jope_plan/evidence/QLT-04/api01-t0250")
READY = E / "handoff.ready.json"
BUNDLE = "94fd9587675c7806bd0a0582b6e97540cd2fdc7ebbc51958411b905185cad85c"
DOPE = "c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814"
LEDGER = Path("/home/ubuntu/jope_plan/ns/qualification/q1_ledger")
ENV = dict(
    os.environ,
    GIT_OPTIONAL_LOCKS="0",
    PYTHONDONTWRITEBYTECODE="1",
    CUDA_VISIBLE_DEVICES="",
    RAYON_NUM_THREADS="8",
    OMP_NUM_THREADS="1",
    OPENBLAS_NUM_THREADS="1",
    MKL_NUM_THREADS="1",
    AWS_EC2_METADATA_DISABLED="true",
)


class Pending(ValueError):
    pass


def sha(path):
    p = Path(path)
    if not p.is_file() or any(x.is_symlink() for x in (p, *p.parents)):
        raise Pending("required regular non-symlink file unavailable: " + str(p))
    with p.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def require(v, msg):
    if not v:
        raise ValueError(msg)


def git(root, *args):
    return subprocess.check_output(
        ["git", "-C", str(root), *args], env=ENV, text=True
    ).strip()


def original_ledger_identity():
    identity = json.loads(
        (E.parent / "api01-t0640/ORIGINAL-LEDGER-IDENTITY.json").read_text()
    )
    observed = {
        "driver_host": os.uname().nodename.split(".")[0],
        "ledger": str(LEDGER),
        "resolved": str(LEDGER.resolve()),
        "device": LEDGER.stat().st_dev,
        "inode": LEDGER.stat().st_ino,
    }
    require(
        observed == identity,
        "original driver/physical ledger differs; no fresh-host ledger",
    )


def check():
    if (E / "DISPATCH_DISABLED").exists():
        raise Pending("dispatch disabled by API01 steering marker")
    if not READY.is_file():
        raise Pending("producer handoff.ready.json absent; no valid signed plan yet")
    original_ledger_identity()
    handoff = json.loads(READY.read_text())
    require(
        handoff.get("status") == "QUALIFIED_SIGNED_READY",
        "producer has not declared qualified signed readiness",
    )
    root = Path(handoff["source_checkout"]).resolve(strict=True)
    registered = git(
        "/home/ubuntu/JopeDime", "worktree", "list", "--porcelain"
    ).splitlines()
    require(
        "worktree " + str(root) in registered, "registered source checkout required"
    )
    require(
        root.is_relative_to(Path("/home/ubuntu/JopeDime/.agent/worktrees")),
        "source checkout outside shared lane root",
    )
    prereg = root / "deploy/qualification/q1/PREREGISTRATION.md"
    harness = root / "deploy/qualification/q1/harness"
    manifest = json.loads((E / "PRESTAGE.json").read_text())
    for row in manifest["source_files"]:
        require(
            sha(root / row["path"]) == row["sha256"],
            "reviewed Q1 source input differs: " + row["path"],
        )
    sys.path.insert(0, str(harness))
    import q1

    plan = q1.load(handoff["plan"])
    q1.validate_plan(plan, prereg)
    require(
        plan["pins"]["bundle_sha256"] == BUNDLE and plan["pins"]["dope_sha256"] == DOPE,
        "frozen r782/DOPE differs",
    )
    require(
        str(LEDGER) == plan["ledger_root"] == str(LEDGER.resolve()),
        "original canonical ledger required",
    )
    require(sha(handoff["backend"]) == plan["backend_sha256"], "backend bytes differ")
    require(
        sha(handoff["backend"]) == sha(harness / "production_backend.py"),
        "backend is not the reviewed production adapter",
    )
    require(os.access(handoff["backend"], os.X_OK), "backend must be executable")
    for key in ["signature", "allowed_signers", "watcher_receipt"]:
        sha(handoff[key])
    q1.verify_owner(plan, handoff["signature"], handoff["allowed_signers"], prereg)
    receipt = json.loads(Path(handoff["watcher_receipt"]).read_text())
    require(
        receipt.get("kind") == "q1" and receipt.get("verify") == "PASS",
        "actual watcher Q1 PASS receipt required",
    )
    require(
        Path(receipt["plan"]).resolve() == Path(handoff["plan"]).resolve()
        and receipt["plan_sha256"] == sha(handoff["plan"]),
        "watcher plan binding differs",
    )
    require(
        Path(receipt["signature"]).resolve() == Path(handoff["signature"]).resolve(),
        "watcher signature path differs",
    )
    require(
        receipt["payload_sha256"]
        == hashlib.sha256(q1.approval_payload(plan)).hexdigest(),
        "watcher payload binding differs",
    )
    require(
        "Signed by ENG:watcher on behalf of Jepson" in receipt.get("signed_by", ""),
        "delegated watcher provenance absent",
    )
    qualified = handoff["qualification_inputs"]["source_bound_backend_receipt"]
    require(
        isinstance(qualified, dict)
        and qualified.get("status") in ("PASS", "PASS_WITH_WAIVER"),
        "qualified backend/input receipt handoff required",
    )
    require(
        sha(qualified["path"]) == qualified["sha256"],
        "qualified runtime receipt changed",
    )
    runtime = json.loads(Path(qualified["path"]).read_text())
    require(
        runtime["status"] == qualified["status"], "runtime disposition handoff differs"
    )
    pin = json.loads((harness.parent / "evaluator-pin.json").read_text())
    require(
        "metadata_source_commit" in plan or not pin.get("admission_inputs"),
        "Linux replacement requires a signed metadata source",
    )
    if "metadata_source_commit" in plan:
        import plan_publisher

        plan_publisher.validate_runtime_receipt(
            runtime, plan, pin, plan["metadata_source_commit"]
        )
    else:
        require(runtime["status"] == "PASS", "legacy admission cannot carry a waiver")
    binding = handoff["qualification_inputs"]["engine_protocol_manifest"]
    require(
        isinstance(binding, dict) and sha(binding["path"]) == binding["sha256"],
        "engine protocol manifest binding required",
    )
    engine_home = Path(handoff["engine_home"]).resolve(strict=True)
    expected = (
        engine_home / "q1/protocols" / (plan["pins"]["protocol_sha256"] + ".json")
    )
    require(
        Path(binding["path"]).resolve() == expected,
        "manifest is not the engine protocol lookup path",
    )
    protocol = json.loads(expected.read_text())
    require(
        protocol.get("schema") == "jope.q1-protocol-manifest/v1"
        and protocol.get("plan_sha256") == q1.digest(plan)
        and protocol.get("source_commit") == plan["source_commit"]
        and protocol.get("pins") == plan["pins"],
        "engine manifest identity differs",
    )
    require(
        set(protocol["datasets"]) == {d["id"] for d in plan["datasets"]},
        "engine manifest cohort differs",
    )
    for dataset in plan["datasets"]:
        require(
            all(
                protocol["datasets"][dataset["id"]].get(k) == v
                for k, v in dataset.items()
            ),
            "engine dataset metadata differs",
        )
    ENV.update(JOPEDIME_HOME=str(engine_home), JOPE_Q1_ENGINE_URL=handoff["engine_url"])
    import production_backend

    prior = os.environ.get("JOPEDIME_HOME")
    os.environ["JOPEDIME_HOME"] = str(engine_home)
    try:
        production_backend.read_daemon_token(
            handoff["engine_url"]
        )  # Verify privately; never log the token.
    finally:
        if prior is None:
            os.environ.pop("JOPEDIME_HOME", None)
        else:
            os.environ["JOPEDIME_HOME"] = prior
    import fifo_launch

    args = argparse.Namespace(
        plan=handoff["plan"],
        backend=handoff["backend"],
        signature=handoff["signature"],
        allowed_signers=handoff["allowed_signers"],
        ledger=str(LEDGER),
        run_dir=handoff["run_dir"],
        wait_seconds=10800,
        wall_seconds=6 * len(plan["datasets"]) * 600,
        authorized_launch=True,
    )
    fifo_launch.admit(
        args,
        fifo_launch.Paths(
            root=root,
            driver=harness / "q1.py",
            prereg=prereg,
            fifo=root / "research/flywheel_selfplay/qualification_lock_fifo.py",
            cohort=root / "deploy/qualification/q1/cohort.json",
            input_manifest=root / "deploy/qualification/q1/inputs-manifest.json",
        ),
    )
    argv = [
        sys.executable,
        "-B",
        str(harness / "fifo_launch.py"),
        "launch",
        "--authorized-launch",
    ]
    for key in [
        "plan",
        "backend",
        "signature",
        "allowed_signers",
        "ledger",
        "run_dir",
        "wait_seconds",
        "wall_seconds",
    ]:
        argv += ["--" + key.replace("_", "-"), str(getattr(args, key))]
    return {
        "schema": "jope.q1-api01-dispatch/v1",
        "utc": now(),
        "host": os.uname().nodename,
        "status": "QUALIFIED_SIGNED_READY",
        "argv": argv,
        "plan_digest": q1.digest(plan),
        "plan_bytes_sha256": sha(handoff["plan"]),
        "source_commit": plan["source_commit"],
        "bundle_sha256": BUNDLE,
        "handoff_sha256": sha(READY),
        "wait_cap_seconds": 10800,
        "whole_job_wall_cap_seconds": args.wall_seconds,
        "budget_basis": "unchanged signed24admissions/600s per policy; serial6N timeout upper bound; no reset or retry",
        "actual_run": "NOT_STARTED",
    }, handoff


def launch(proof):
    require(
        not (E / "dispatch.json").exists(),
        "dispatch already recorded; inspect original candidate ledger, never auto retry",
    )
    with (E / "dispatch.json").open("x") as f:
        f.write(json.dumps(proof, indent=2) + "\n")
    with (E / "launch.log").open("xb") as log:
        process = subprocess.Popen(
            proof["argv"], env=ENV, stdout=log, stderr=subprocess.STDOUT
        )
        (E / "RUNNER.json").write_text(
            json.dumps(
                {
                    "pid": process.pid,
                    "utc": now(),
                    "argv": proof["argv"],
                    "lock_owner": "maintained fifo_launch.py only; no nested flock",
                },
                indent=2,
            )
            + "\n"
        )
        code = process.wait()
    (E / "EXIT.json").write_text(
        json.dumps(
            {
                "utc": now(),
                "exit": code,
                "run_dir": proof["argv"][proof["argv"].index("--run-dir") + 1],
                "instruction": "retain original ledger/logs/full report; no automatic retry",
            },
            indent=2,
        )
        + "\n"
    )
    return code


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["check", "launch", "watch"])
    args = parser.parse_args()
    if os.getpriority(os.PRIO_PROCESS, 0) < 10:
        os.nice(10 - os.getpriority(os.PRIO_PROCESS, 0))
    if args.mode == "watch":
        end = time.monotonic() + 21600
        last = None
        while time.monotonic() < end:
            try:
                proof, _ = check()
                raise SystemExit(launch(proof))
            except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
                state = {
                    "utc": now(),
                    "status": "PRESTAGED_WAITING_FOR_QUALIFIED_SIGNED_HANDOFF",
                    "reason": str(error),
                    "Q1": "NOT_RUN",
                    "sealed_payloads_opened": 0,
                    "dispatch_enabled_after_actual_admission": True,
                }
                (E / "WATCH-STATE.json").write_text(json.dumps(state, indent=2) + "\n")
                if state["reason"] != last:
                    print(json.dumps(state), flush=True)
                    last = state["reason"]
            time.sleep(30)
        (E / "WATCH-EXPIRED.json").write_text(
            json.dumps(
                {
                    "utc": now(),
                    "status": "OBSERVATION_WINDOW_ENDED_WITHOUT_DISPATCH",
                    "Q1": "NOT_RUN",
                    "study_budget_reset": False,
                },
                indent=2,
            )
            + "\n"
        )
    else:
        try:
            proof, _ = check()
            print(json.dumps(proof))
            if args.mode == "launch":
                raise SystemExit(launch(proof))
        except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
            print(str(error), file=sys.stderr)
            raise SystemExit(3)
