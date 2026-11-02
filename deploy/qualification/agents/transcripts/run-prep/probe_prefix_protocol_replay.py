import re
import time

from verify_capture import require


def run_replay(rpc, dataset, target, key, results):
    """One bounded real run; protocol evidence is never named-client qualification."""
    from probe_prefix_protocol import outcome

    def call(step, name, arguments):
        ok, payload = outcome(rpc.tool(name, arguments))
        results.append(
            dict(
                step=step,
                tool=name,
                status="success" if ok else "error",
                payload=payload,
            )
        )
        require(ok and isinstance(payload, dict), "run replay refused at " + step)
        return payload

    spec = dict(
        dataset=dataset,
        target=target,
        task="regression",
        ignore_columns=[],
        test_dataset=None,
        split=dict(
            kind="random", holdout_fraction=0.2, group_column=None, time_column=None
        ),
        objective="rmse",
        effort="fast",
        budget=dict(evaluations="1", wall_seconds="5"),
        landmarks="none",
        guidance="heuristic",
        languages=["rust"],
        require_all_languages=True,
        device="cpu",
        seed="73",
        dry_run=True,
        plan_digest=None,
        idempotency_key=key + "-plan",
    )
    plan = call("run_plan", "run_start", spec)
    require(
        isinstance(plan.get("plan_digest"), str)
        and bool(plan["plan_digest"].strip())
        and isinstance(plan.get("placement"), dict)
        and plan["placement"].get("evaluation") == "cpu"
        and type(plan["placement"].get("workers")) is int
        and plan["placement"].get("workers") == 1,
        "plan must identify one CPU worker",
    )
    call("run_inventory_before", "run_status", {})
    start = dict(
        spec, dry_run=False, plan_digest=plan["plan_digest"], idempotency_key=key
    )
    first = call("run_create", "run_start", start)
    run = first.get("run_id")
    require(
        isinstance(run, str) and re.fullmatch(r"run_[0-7][0-9A-HJKMNP-TV-Z]{25}", run),
        "start returned no canonical run ID",
    )
    call("run_inventory_created", "run_status", {})
    retry = call("run_retry", "run_start", start)
    require(retry.get("run_id") == run, "retry changed run ID")
    call("run_inventory_retried", "run_status", {})
    cursor = 0
    last_kind = None
    while True:
        status = call(
            "run_poll",
            "run_status",
            dict(run_id=run, wait_seconds=0, after_event=str(cursor)),
        )
        require(status.get("run_id") == run, "status changed run ID")
        page = call(
            "run_events", "run_events", dict(run_id=run, after=str(cursor), limit=256)
        )
        events = page.get("events")
        require(isinstance(events, list) and len(events) <= 256, "invalid event page")
        for event in events:
            require(
                isinstance(event, dict)
                and event.get("run_id") == run
                and event.get("sequence") == str(cursor + 1),
                "event gap, duplicate or wrong run",
            )
            cursor += 1
            last_kind = event.get("kind")
            require(cursor <= 1024, "run replay event bound exceeded")
        require(page.get("next_after") == str(cursor), "event cursor differs")
        if status.get("state") in {"completed", "failed", "cancelled"}:
            require(status["state"] == "completed", "run did not complete successfully")
            if status.get("event_watermark") == str(cursor):
                require(
                    cursor > 0 and last_kind == "run.finished",
                    "completed run lacks its terminal event",
                )
                return dict(run_id=run, state="completed", event_watermark=str(cursor))
        require(time.monotonic() < rpc.deadline, "run replay deadline exceeded")
        time.sleep(0.25)
