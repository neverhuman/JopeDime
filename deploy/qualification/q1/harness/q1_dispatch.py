"""Fatal dispatch boundary for the admitted Q1 infrastructure retry.

Driver failures are diagnostic envelopes, never successful engine responses.
The caller must append the envelope durably and stop on BackendFatal.
"""

import hashlib
import json
import subprocess
import time

import q1


class BackendFatal(RuntimeError):
    def __init__(self, diagnostic):
        super().__init__("fatal Q1 backend failure; stop dispatch")
        self.diagnostic = diagnostic


def redacted_stderr(raw):
    """Allowlisted diagnostics only; unexpected bytes remain hash-bound."""
    raw = raw or b""
    if isinstance(raw, str):
        raw = raw.encode()
    allowed = {
        "schema": {"jope.q1-backend-error/v1"},
        "phase": {"engine_http", "engine_transport", "request_or_receipt_validation"},
        "engine_code": {
            None,
            "IO_FAILURE",
            "NUMERICAL_FAILURE",
            "INVALID_INPUT",
            "CAPABILITY_UNAVAILABLE",
            "BACKEND_UNAVAILABLE",
            "MODEL_INTEGRITY_FAILED",
            "ENGINE_BUSY",
            "UNAUTHORIZED",
        },
        "error_class": {"ValueError", "KeyError", "TypeError", "JSONDecodeError"},
    }
    safe = {}
    if raw == b"production backend refused: production engine request failed\n":
        # The admitted retry retains the old signed adapter. Its one fixed
        # transport diagnostic contains no credentials or data values.
        safe = {
            "phase": "legacy_engine_request",
            "message": "production engine request failed",
        }
    try:
        if len(raw) > 4096:
            raise ValueError("diagnostic exceeds limit")
        value = q1.decode(raw)
        if (
            isinstance(value, dict)
            and value.get("schema") == "jope.q1-backend-error/v1"
        ):
            for key, choices in allowed.items():
                candidate = value.get(key)
                if isinstance(candidate, (str, type(None))) and candidate in choices:
                    safe[key] = candidate
            code = value.get("http_status")
            if type(code) is int and 100 <= code <= 599:
                safe["http_status"] = code
    except (ValueError, TypeError, UnicodeError):
        pass  # Unrecognized stderr is redacted in full, not copied into receipts.
    return {
        "stderr_redacted": json.dumps(safe, sort_keys=True),
        "stderr_bytes": len(raw),
        "stderr_sha256": hashlib.sha256(raw).hexdigest(),
        "stderr_content_omitted": not bool(safe),
    }


def invoke(request, backend, env, timeout, cleanup_grace=5):
    """Return one valid response or raise a fatal error with no implicit retry."""
    try:
        result = q1._invoke_backend(
            backend, q1.canonical(request), env, timeout, cleanup_grace
        )
    except (
        subprocess.TimeoutExpired,
        OSError,
        q1.BackendSpawnError,
        q1.BackendCleanupError,
    ) as error:
        raise BackendFatal(
            {
                "phase": "timeout"
                if isinstance(error, subprocess.TimeoutExpired)
                else "spawn_or_cleanup",
                "backend_returncode": None,
                "error_class": type(error).__name__,
                **redacted_stderr(getattr(error, "stderr", None)),
            }
        ) from error
    diagnostic = {
        "phase": "backend_exit" if result.returncode else "response_validation",
        "backend_returncode": result.returncode,
        **redacted_stderr(result.stderr),
    }
    if result.returncode != 0:
        raise BackendFatal(diagnostic)
    try:
        return q1.response_valid(q1.decode(result.stdout), request)
    except (ValueError, TypeError, KeyError, UnicodeError) as error:
        diagnostic["error_class"] = type(error).__name__
        raise BackendFatal(diagnostic) from error


def dispatch_units(requests, backend, env, timeout, stream):
    """Durably record a failed attempt and abort before the next dispatch."""
    for request in requests:
        identity = {
            key: request[key]
            for key in (
                "mode",
                "plan_sha256",
                "source_commit",
                "pins",
                "dataset",
                "guidance",
                "seed",
                "budget",
                "request_sha256",
            )
        }
        q1.append_receipt(
            stream, {**identity, "event": "started", "status": "outcome_unknown"}
        )
        started = time.monotonic()
        try:
            response = invoke(request, backend, env, timeout)
        except BackendFatal as error:
            q1.append_receipt(
                stream,
                {
                    **identity,
                    "event": "driver_failure",
                    "status": "outcome_unknown",
                    "diagnostic": error.diagnostic,
                    "wall_seconds": time.monotonic() - started,
                },
            )
            raise
        q1.append_receipt(
            stream,
            {
                **identity,
                "event": "finished",
                "response": response,
                "wall_seconds": time.monotonic() - started,
            },
        )
