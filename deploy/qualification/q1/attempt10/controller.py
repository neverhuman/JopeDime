#!/usr/bin/env python3
"""One admitted infrastructure retry, retaining the original scientific identities.

The owning launcher supplies an already authenticated, source-bound CPU daemon
and tunnel, and retains responsibility for their exact-PID cleanup. This driver
does not build, deploy, read dataset payloads, sign, or analyze partial results.
"""

import argparse
import datetime
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def harness_modules():
    sys.path.insert(0, str(HERE.parent / "harness"))
    import fifo_launch
    import q1
    import q1_dispatch

    return fifo_launch, q1, q1_dispatch


fifo_launch, q1, q1_dispatch = harness_modules()

AUTHORITY_SHA = "b7493bf59d94a3855c487fd1453255a02398f01dd190f8a05e285a36ed326747"
EVALUATOR_SHA = "4e9239a1126589a42c3d80b1742d463d1b2331def59e5a08796169b4f3cec177"
BACKEND_SHA = "8288835bd05199e92b3eac7cc59039f8e932fdfae552b5efe95f76c072beb911"
PANEL_SHA = "f874cae1087d716f3031a8a0dfa1cca1071b8f0cf6c7d97c13e4fa283cc1a9df"


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def pinned(config, key):
    reference = config["artifacts"][key]
    path = Path(reference["path"])
    q1.require(
        path.is_absolute() and path.is_file() and not path.is_symlink(),
        "regular absolute artifact required: " + key,
    )
    q1.require(
        q1.sha_file(path) == reference["sha256"], "artifact hash changed: " + key
    )
    return path


def key(row):
    return row["dataset"]["id"], row["seed"], row["guidance"]


def preflight(config):
    q1.require(
        config["schema"] == "jope.q1-attempt10-runtime-amendment/v1", "amendment schema"
    )
    q1.require(
        config["decision"] == "ADMIT_ORIGINAL_IDENTITIES_RUNTIME_REPAIR",
        "amendment not admitted",
    )
    q1.require(
        config["budget_reset"] is False and config["valid_unit_reruns"] is False,
        "retry must preserve historical budgets",
    )
    q1.require(
        q1.sha_file(pinned(config, "authority")) == AUTHORITY_SHA,
        "owner retry authority changed",
    )
    head = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    q1.require(
        head == config["runtime_source_commit"], "runtime source checkout mismatch"
    )
    q1.require(
        not subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain"]),
        "runtime source checkout is dirty",
    )
    review = q1.load(pinned(config, "gov03_review"))
    q1.require(
        review["reviewed_head"] == head and review["verdict"] == "PASS",
        "exact runtime/controller review is not PASS",
    )
    plan = q1.load(pinned(config, "plan"))
    q1.validate_plan(plan)
    q1.verify_owner(
        plan, pinned(config, "signature"), pinned(config, "allowed_signers")
    )
    q1.require(
        plan["pins"]["evaluator_sha256"] == EVALUATOR_SHA, "numerical evaluator changed"
    )
    q1.require(
        q1.sha_file(pinned(config, "backend")) == plan["backend_sha256"] == BACKEND_SHA,
        "original signed backend must remain byte-identical",
    )
    panel = pinned(config, "original_panel")
    q1.require(q1.sha_file(panel) == PANEL_SHA, "original terminal panel changed")
    original = [q1.decode(line) for line in panel.read_text().splitlines() if line]
    starts = {key(row): row for row in original if row["event"] == "started"}
    finished = [row for row in original if row["event"] == "finished"]
    valid = [row for row in finished if row["response"]["status"] == "complete"]
    failed = [row for row in finished if row["response"]["status"] == "failed"]
    q1.require(
        len(starts) == len(finished) == 582 and len(valid) == 60 and len(failed) == 522,
        "original panel count mismatch",
    )
    selection = q1.load(pinned(config, "selection"))
    selected = {tuple(row["key"]) for row in selection["retry_selection"]}
    q1.require(
        selected == {key(row) for row in failed} and len(selected) == 522,
        "retry selection must contain exactly the 522 failed identities",
    )
    for row in valid:
        q1.response_valid(
            row["response"],
            {
                "request_sha256": row["request_sha256"],
                "guidance": row["guidance"],
                "budget": row["budget"],
            },
        )
    q1.require(
        sum(row["response"]["charged"] for row in valid) == 1440,
        "historical charges changed",
    )
    q1.require(
        Path(config["output"]).is_absolute() and not Path(config["output"]).exists(),
        "new private output directory required",
    )
    q1.require(
        time.time() + plan["timeout_seconds"] + 5 < config["fixed_deadline_unix"],
        "insufficient admitted deadline remains",
    )
    return plan, original, failed


def requests(plan, failed, config, samples):
    """Yield only originally failed units, with the same signed request hash."""
    for old in failed:
        while True:
            q1.require(
                time.time() + plan["timeout_seconds"] + 5
                < config["fixed_deadline_unix"],
                "fixed retry deadline reached",
            )
            # The owning daemon monitor separately enforces the remote host
            # budget. Refuse a stale/dead monitor before another dispatch.
            engine = q1.load(pinned(config, "engine_receipt"))
            q1.require(
                engine["runtime_source_commit"] == config["runtime_source_commit"]
                and engine["evaluator_sha256"] == EVALUATOR_SHA
                and engine["status"] == "ENGINE_READY",
                "engine identity changed",
            )
            monitor = q1.load(Path(config["monitor_status"]))
            q1.require(
                0 <= time.time() - monitor["sampled_unix"] <= 90
                and monitor["runtime_source_commit"] == config["runtime_source_commit"],
                "remote daemon monitor is stale or unrelated",
            )
            local_load = os.getloadavg()[0]
            remote_load = monitor["remote_load"]
            q1.require(
                shutil.disk_usage(config["output"]).free > 100 * 1024**3
                and monitor["remote_free_bytes"] > 100 * 1024**3,
                "disk admission failed",
            )
            q1.append_receipt(
                samples,
                {"utc": utc(), "local_load": local_load, "remote_load": remote_load},
            )
            if local_load <= 85 and remote_load <= 85:
                break
            time.sleep(10)
        request = {"schema": "jope.q1-request/v1", "metric": "rmse"}
        request.update(
            {
                name: old[name]
                for name in (
                    "mode",
                    "plan_sha256",
                    "source_commit",
                    "pins",
                    "dataset",
                    "guidance",
                    "seed",
                    "budget",
                )
            }
        )
        request["request_sha256"] = q1.digest(request)
        q1.require(
            request["request_sha256"] == old["request_sha256"]
            and request["plan_sha256"] == q1.digest(plan)
            and request["source_commit"] == plan["source_commit"]
            and request["pins"] == plan["pins"],
            "scientific request identity changed",
        )
        yield request


def run(config, config_path):
    plan, original, failed = preflight(config)
    os.umask(0o077)
    os.setpriority(os.PRIO_PROCESS, 0, 19)
    q1.require(os.uname().nodename.split(".")[0] != "xbabe0", "forbidden host")
    output = Path(config["output"])
    output.mkdir()
    env = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith(("AWS_", "GH_", "GITHUB_", "JOPE_AWS_"))
    }
    env.update(
        CUDA_VISIBLE_DEVICES="",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        RAYON_NUM_THREADS="8",
        JOPE_Q1_ENGINE_URL=config["engine_endpoint"],
        JOPEDIME_HOME=config["local_daemon_home"],
    )
    paths = fifo_launch.Paths(root=ROOT)
    fifo = fifo_launch.fifo_module(paths)
    state = {
        "utc": utc(),
        "status": "NOT_STARTED",
        "amendment_sha256": q1.sha_file(config_path),
        "runtime_source_commit": config["runtime_source_commit"],
        "budget_reset": False,
        "prior_unknown_charges": None,
        "analysis_performed": False,
    }

    def await_owned_cleanup():
        # Keep the FIFO lease until the owning launcher has drained this exact
        # daemon and its workers, including after a fatal transport error.
        requested = time.time()
        with (output / "CLEANUP-REQUESTED.json").open("x") as stream:
            q1.append_receipt(
                stream,
                {
                    **state,
                    "status": "DISPATCH_EXITED_DRAIN_REQUIRED",
                    "requested_unix": requested,
                },
            )
        previous = {
            sig: signal.signal(sig, signal.SIG_IGN)
            for sig in (signal.SIGTERM, signal.SIGINT)
        }
        try:
            path = output / "RUNTIME-CLEANUP.json"
            while True:
                try:
                    cleanup = q1.load(path)
                    engine = q1.load(pinned(config, "engine_receipt"))
                    if (
                        not path.is_symlink()
                        and cleanup["amendment_sha256"] == state["amendment_sha256"]
                        and cleanup["engine_receipt_sha256"]
                        == q1.sha_file(pinned(config, "engine_receipt"))
                        and cleanup["daemon_pid"] == engine["pid"]
                        and cleanup["completed_unix"] >= requested
                        and cleanup["daemon_exited"] is True
                        and cleanup["workers_remaining"] == 0
                        and cleanup["tunnel_exited"] is True
                    ):
                        break
                except (OSError, ValueError, KeyError, TypeError):
                    # A missing/invalid drain receipt cannot release the lease.
                    pass
                time.sleep(2)
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)

    try:
        with fifo_launch.lease(paths, fifo, 10800, lambda: os.getloadavg()[0]):
            # This separate single-use marker never rewrites original markers.
            marker = Path(plan["ledger_root"]) / (
                plan["pins"]["bundle_sha256"] + ".attempt10.started.json"
            )
            with marker.open("x") as stream:
                q1.append_receipt(stream, state)
            state["status"] = "RUNNING"
            try:
                with (
                    (output / "receipts.jsonl").open("x") as stream,
                    (output / "load.jsonl").open("x") as samples,
                ):
                    q1_dispatch.dispatch_units(
                        requests(plan, failed, config, samples),
                        [str(pinned(config, "backend"))],
                        env,
                        plan["timeout_seconds"],
                        stream,
                    )
            finally:
                await_owned_cleanup()
            resumed = [
                q1.decode(line)
                for line in (output / "receipts.jsonl").read_text().splitlines()
            ]
            new = {key(row): row for row in resumed if row["event"] == "finished"}
            q1.require(len(new) == 522, "retry panel incomplete")
            with (output / "logical-panel.jsonl").open("x") as stream:
                for row in original:
                    if row["event"] == "finished" and key(row) in new:
                        row = new[key(row)]
                    q1.append_receipt(stream, row)
            state["status"] = "TERMINAL_UNANALYZED"
    except q1_dispatch.BackendFatal as error:
        state.update(status="STOPPED_FATAL_BACKEND", diagnostic=error.diagnostic)
        raise
    except BaseException as error:
        state.update(status="STOPPED_OR_INCOMPLETE", error_class=type(error).__name__)
        raise
    finally:
        state["utc"] = utc()
        with (output / "STATUS.json").open("x") as stream:
            q1.append_receipt(stream, state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--admission", type=Path, required=True)
    args = parser.parse_args()
    run(q1.load(args.admission), args.admission)
