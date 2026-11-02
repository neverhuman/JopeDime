from __future__ import annotations

import importlib.util
import json
import re
from collections import Counter
from pathlib import Path

from common import ROOT, ProviderError

TOOL_COUNTS = {
    "system_info": 1,
    "run_status": 5,
    "dataset_register": 2,
    "dataset_inspect": 1,
    "run_start": 3,
    "run_events": 1,
    "model_export": 2,
    "model_predict": 1,
}

_CAPTURE_PATH = (
    Path(__file__).resolve().parents[2]
    / "agents/transcripts/run-prep/verify_capture.py"
)
_CAPTURE_SPEC = importlib.util.spec_from_file_location(
    "a5_capture_verifier", _CAPTURE_PATH
)
if _CAPTURE_SPEC is None or _CAPTURE_SPEC.loader is None:
    raise RuntimeError("A5 capture verifier cannot be loaded")
_CAPTURE = importlib.util.module_from_spec(_CAPTURE_SPEC)
_CAPTURE_SPEC.loader.exec_module(_CAPTURE)

def validate_raw_flow(
    wire_raw: bytes, run_id: str, dataset_id: str, rows: int, receipt: dict
) -> None:
    """Recompute the claimed 16-call golden path from the paired raw JSON-RPC stream."""
    try:
        calls = sorted(_CAPTURE.tool_calls(wire_raw), key=lambda call: call["start"])
    except ValueError as error:
        raise ProviderError("A5 raw full-flow capture is invalid") from error
    names = [call.get("name") for call in calls]
    if (
        len(calls) != 16
        or any(not isinstance(name, str) for name in names)
        or Counter(names) != TOOL_COUNTS
        or any(not isinstance(call.get("data"), dict) for call in calls)
    ):
        raise ProviderError("A5 raw flow tool inventory is incomplete or failed")

    groups: dict[str, list[dict]] = {}
    for call in calls:
        groups.setdefault(call["name"], []).append(call)
    registrations = groups["dataset_register"]
    if any(call["data"].get("dataset_id") != dataset_id for call in registrations):
        raise ProviderError(
            "A5 raw registration does not match the claimed TRAIN dataset"
        )
    inspections = groups["dataset_inspect"]
    if not all(
        call["arguments"].get("dataset_id") == dataset_id
        and call["data"].get("dataset_id") == dataset_id
        and call["data"].get("rows") == str(rows)
        for call in inspections
    ):
        raise ProviderError(
            "A5 raw inspection does not match the claimed TRAIN dataset"
        )

    starts = groups["run_start"]
    plans = [call for call in starts if call["arguments"].get("dry_run") is True]
    creates = [call for call in starts if call["arguments"].get("dry_run") is False]
    if len(plans) != 1 or len(creates) != 2:
        raise ProviderError("A5 raw plan/create/retry path is incomplete")
    first, retry = creates
    plan_digest = plans[0]["data"].get("plan_digest")
    if (
        registrations[-1]["end"] >= first["start"]
        or inspections[0]["end"] >= first["start"]
        or any(call["arguments"].get("dataset") != dataset_id for call in starts)
        or first["data"] != retry["data"]
        or not any(
            plan["end"] < first["start"]
            and plan["data"].get("plan_digest") == first["arguments"].get("plan_digest")
            and isinstance(plan_digest, str)
            and re.fullmatch(r"[0-9a-f]{64}", plan_digest) is not None
            for plan in plans
        )
    ):
        raise ProviderError("A5 raw run creation does not match its plan and retry")
    placement = plans[0]["data"].get("placement")
    effort = plans[0]["data"].get("effort")
    if (
        not isinstance(placement, dict)
        or placement.get("evaluation") != "cpu"
        or type(placement.get("workers")) is not int
        or placement["workers"] != 1
        or placement.get("dope") != "none"
        or not isinstance(effort, dict)
        or effort.get("evaluations") != "1"
        or first["arguments"].get("device") != "cpu"
    ):
        raise ProviderError("A5 raw plan is outside the bounded CPU profile")

    terminal_rows = [
        call
        for call in groups["run_status"]
        if call["arguments"].get("run_id") == run_id
    ]
    if not terminal_rows:
        raise ProviderError("A5 raw completed run status is missing")
    terminal = terminal_rows[-1]
    terminal_data = terminal["data"]
    budget = terminal_data.get("budget_used")
    evaluations = budget.get("evaluations") if isinstance(budget, dict) else None
    if (
        terminal_data.get("run_id") != run_id
        or terminal_data.get("state") != "completed"
        or not isinstance(evaluations, str)
        or not evaluations.isdecimal()
        or int(evaluations) <= 0
        or evaluations != receipt.get("evaluations_used")
    ):
        raise ProviderError("A5 raw run did not complete an evaluation")

    events = []
    for call in groups["run_events"]:
        if call["arguments"].get("run_id") != run_id:
            raise ProviderError("A5 raw events refer to a different run")
        page = call["data"].get("events")
        if not isinstance(page, list):
            raise ProviderError("A5 raw event page is missing")
        events.extend(page)
    if (
        len(events) != receipt.get("event_count")
        or len(events) != 6
        or any(not isinstance(event, dict) for event in events)
        or [event.get("sequence") for event in events]
        != [str(index) for index in range(1, 7)]
        or any(event.get("run_id") != run_id for event in events)
        or not any(event.get("kind") == "experiment.finished" for event in events)
        or events[-1].get("kind") != "run.finished"
        or terminal_data.get("event_watermark") != "6"
    ):
        raise ProviderError("A5 raw events are incomplete or out of order")

    exports = groups["model_export"]
    if not all(
        call["arguments"].get("run_id") == run_id
        and call["data"].get("kind") == "bundle"
        and call["data"].get("retains_training_data") is False
        and isinstance(call["data"].get("languages"), list)
        and any(
            isinstance(language, dict)
            and language.get("language") == "rust"
            and language.get("status") == "exported"
            for language in call["data"]["languages"]
        )
        and terminal["end"] < call["start"]
        for call in exports
    ):
        raise ProviderError("A5 raw completed-model export is missing")
    if (
        not isinstance(exports[0]["arguments"].get("idempotency_key"), str)
        or not exports[0]["arguments"]["idempotency_key"].strip()
        or json.dumps(exports[0]["arguments"], sort_keys=True)
        != json.dumps(exports[1]["arguments"], sort_keys=True)
        or json.dumps(exports[0]["data"], sort_keys=True)
        != json.dumps(exports[1]["data"], sort_keys=True)
    ):
        raise ProviderError("A5 raw export retry changed its key or outcome")
    artifact_ids = {call["data"].get("artifact_id") for call in exports}
    best = terminal_data.get("best")
    if isinstance(best, dict):
        artifact_ids.add(best.get("model_artifact"))
    artifact_ids.discard(None)
    if not artifact_ids:
        raise ProviderError("A5 raw export identity is missing")

    prediction = groups["model_predict"][0]
    model = prediction["arguments"].get("model")
    prediction_input = prediction["arguments"].get("input")
    output = prediction["data"]
    if (
        not isinstance(model, dict)
        or not (
            model.get("run_id") == run_id or model.get("artifact_id") in artifact_ids
        )
        or not isinstance(prediction_input, dict)
        or prediction_input.get("dataset_id") != dataset_id
        or output.get("rows") != str(rows)
        or not isinstance(output.get("sha256"), str)
        or re.fullmatch(r"[0-9a-f]{64}", output["sha256"]) is None
        or output["sha256"] != receipt.get("predictions", {}).get("sha256")
        or exports[0]["end"] >= prediction["start"]
    ):
        raise ProviderError("A5 raw prediction is missing or mismatched")
