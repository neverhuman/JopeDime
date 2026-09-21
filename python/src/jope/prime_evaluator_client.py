"""Blocking client for Prime's persistent ``battle-evald`` service.

The wire contract is owned by ``battle-eval-protocol``: protocol v4 JSON
envelopes inside unsigned-32-bit, big-endian length-prefixed frames.  A client
connection is handshaken before use and is closed permanently after any frame,
protocol, identity, or transport failure.

Cancellation is cooperative inside the evaluator.  If a job request times out,
the client makes one bounded best-effort cancellation request over a separate
connection; the running candidate may only observe it at its next cancellation
point.
"""

from __future__ import annotations

import json
import math
import re
import socket
import struct
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PROTOCOL_VERSION = 4
DEFAULT_MAX_FRAME_SIZE = 64 * 1024 * 1024
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_ERROR_CODES = {
    "invalid_request",
    "unsupported_protocol",
    "unknown_dataset",
    "unknown_fold_plan",
    "invalid_genome",
    "unsupported_operator",
    "unsupported_backend",
    "backend_unavailable",
    "evaluation_failed",
    "artifact_unsupported",
    "cancelled",
    "internal",
}


class EvaluatorClientError(RuntimeError):
    """Base error for evaluator transport, protocol, and remote failures."""


class EvaluatorTransportError(EvaluatorClientError):
    """The evaluator connection could not be opened or used."""


class EvaluatorProtocolError(EvaluatorClientError):
    """The peer violated the negotiated framing or protocol-v4 contract."""


class RegistryMismatchError(EvaluatorProtocolError):
    """The daemon does not own the registry expected by this client."""


class EvaluatorRemoteError(EvaluatorClientError):
    """Typed top-level error returned by the evaluator."""

    def __init__(
        self,
        *,
        code: str,
        message: str,
        retriable: bool,
        details: Mapping[str, str],
    ) -> None:
        super().__init__(f"evaluator returned {code}: {message}")
        self.code = code
        self.message = message
        self.retriable = retriable
        self.details = dict(details)


class UnknownDatasetError(EvaluatorRemoteError):
    """Dataset registration was lost and the caller may register it again."""

    def __init__(
        self,
        *,
        message: str,
        server_retriable: bool,
        details: Mapping[str, str],
    ) -> None:
        super().__init__(code="unknown_dataset", message=message, retriable=True, details=details)
        self.server_retriable = server_retriable


class EvaluatorTimeoutError(EvaluatorTransportError):
    """A bounded request timed out; cancellation details are attached."""

    def __init__(
        self,
        *,
        job_id: str | None,
        cancel_response: Mapping[str, Any] | None,
        cancel_error: str | None,
    ) -> None:
        message = "evaluator request timed out"
        if job_id is not None:
            message += f" for job {job_id!r}"
        super().__init__(message)
        self.job_id = job_id
        self.cancel_attempted = job_id is not None
        self.cancel_response = dict(cancel_response) if cancel_response else None
        self.cancel_error = cancel_error


def _positive_timeout(value: float, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return number


def _sha256(value: str, name: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value}")


def _strict_json(payload: bytes) -> Any:
    try:
        return json.loads(
            payload,
            object_pairs_hook=_object_without_duplicates,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise EvaluatorProtocolError(f"invalid evaluator frame JSON: {error}") from error


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise EvaluatorProtocolError(f"{name} must be a JSON object")
    return value


def _exact_fields(value: Mapping[str, Any], fields: set[str], name: str) -> None:
    actual = set(value)
    if actual != fields:
        raise EvaluatorProtocolError(
            f"{name} fields differ from protocol v4: expected {sorted(fields)}, received {sorted(actual)}"
        )


def _unsigned_integer(value: Any, name: str, *, bits: int, positive: bool = False) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise EvaluatorProtocolError(f"{name} must be an unsigned integer")
    minimum = 1 if positive else 0
    if not minimum <= value < (1 << bits):
        qualifier = "positive " if positive else ""
        raise EvaluatorProtocolError(f"{name} must be a {qualifier}u{bits} integer")
    return value


def _cuda_backend(value: Any, name: str) -> dict[str, Any]:
    backend = _mapping(value, name)
    _exact_fields(
        backend,
        {
            "kind",
            "device_ordinal",
            "device_name",
            "compute_capability",
            "driver_version",
            "execution_identity",
        },
        name,
    )
    if backend["kind"] != "cuda":
        raise EvaluatorProtocolError(f"{name}.kind must be 'cuda'")
    _unsigned_integer(backend["device_ordinal"], f"{name}.device_ordinal", bits=32)
    if not isinstance(backend["device_name"], str) or not backend["device_name"]:
        raise EvaluatorProtocolError(f"{name}.device_name must be a non-empty string")
    if backend["driver_version"] is not None and (
        not isinstance(backend["driver_version"], str) or not backend["driver_version"]
    ):
        raise EvaluatorProtocolError(f"{name}.driver_version must be a non-empty string or null")

    compute = _mapping(backend["compute_capability"], f"{name}.compute_capability")
    _exact_fields(compute, {"major", "minor"}, f"{name}.compute_capability")
    _unsigned_integer(
        compute["major"],
        f"{name}.compute_capability.major",
        bits=16,
        positive=True,
    )
    _unsigned_integer(compute["minor"], f"{name}.compute_capability.minor", bits=16)

    identity = _mapping(backend["execution_identity"], f"{name}.execution_identity")
    _exact_fields(
        identity,
        {
            "kernel_abi_version",
            "kernel_cache_key",
            "loaded_ptx_sha256",
            "cuda_driver_api_version",
            "nvrtc_version",
            "cuda_api_version",
            "cudarc_version",
            "fmad_enabled",
        },
        f"{name}.execution_identity",
    )
    _unsigned_integer(
        identity["kernel_abi_version"],
        f"{name}.execution_identity.kernel_abi_version",
        bits=32,
        positive=True,
    )
    for field in ("kernel_cache_key", "loaded_ptx_sha256"):
        try:
            _sha256(identity[field], f"{name}.execution_identity.{field}")
        except ValueError as error:
            raise EvaluatorProtocolError(str(error)) from error
    _unsigned_integer(
        identity["cuda_driver_api_version"],
        f"{name}.execution_identity.cuda_driver_api_version",
        bits=32,
        positive=True,
    )
    nvrtc = _mapping(identity["nvrtc_version"], f"{name}.execution_identity.nvrtc_version")
    _exact_fields(
        nvrtc,
        {"major", "minor"},
        f"{name}.execution_identity.nvrtc_version",
    )
    _unsigned_integer(
        nvrtc["major"],
        f"{name}.execution_identity.nvrtc_version.major",
        bits=16,
        positive=True,
    )
    _unsigned_integer(nvrtc["minor"], f"{name}.execution_identity.nvrtc_version.minor", bits=16)
    _unsigned_integer(
        identity["cuda_api_version"],
        f"{name}.execution_identity.cuda_api_version",
        bits=32,
        positive=True,
    )
    if not isinstance(identity["cudarc_version"], str) or not identity["cudarc_version"]:
        raise EvaluatorProtocolError(f"{name}.execution_identity.cudarc_version must be a non-empty string")
    if identity["fmad_enabled"] is not False:
        raise EvaluatorProtocolError(f"{name}.execution_identity.fmad_enabled must be false")
    return json.loads(json.dumps(backend))


class PrimeEvaluatorClient:
    """Synchronous, persistent AF_UNIX client for one trusted Prime daemon.

    Requests are serialized on the connection.  ``socket_path`` and
    ``prime_root`` must both be explicit absolute paths, and the socket must be
    below ``prime_root``; there is no legacy or ``/tmp`` fallback.
    """

    def __init__(
        self,
        socket_path: str | Path,
        *,
        prime_root: str | Path,
        expected_registry_hash: str,
        expected_registry_version: str | None = None,
        client_name: str = "jope-prime-python",
        client_version: str = "0.2.0",
        connect_timeout: float = 5.0,
        request_timeout: float = 300.0,
        cancel_timeout: float = 5.0,
        max_frame_size: int = DEFAULT_MAX_FRAME_SIZE,
    ) -> None:
        path = Path(socket_path).expanduser()
        root = Path(prime_root).expanduser()
        if not path.is_absolute() or not root.is_absolute():
            raise ValueError("socket_path and prime_root must be absolute")
        path = path.resolve(strict=False)
        root = root.resolve(strict=False)
        if not path.is_relative_to(root):
            raise ValueError("evaluator Unix socket must resolve below prime_root")
        if Path(socket_path).is_symlink():
            raise ValueError("evaluator Unix socket path must not be a symlink")
        if not isinstance(max_frame_size, int) or isinstance(max_frame_size, bool):
            raise ValueError("max_frame_size must be an integer")
        if not 0 < max_frame_size <= 0xFFFF_FFFF:
            raise ValueError("max_frame_size must be in 1..=u32::MAX")
        if not client_name or not client_version:
            raise ValueError("client_name and client_version must be non-empty")

        self._path = path
        self._expected_registry_hash = _sha256(expected_registry_hash, "expected_registry_hash")
        self._expected_registry_version = expected_registry_version
        self._client_name = client_name
        self._client_version = client_version
        self._connect_timeout = _positive_timeout(connect_timeout, "connect_timeout")
        self._request_timeout = _positive_timeout(request_timeout, "request_timeout")
        self._cancel_timeout = _positive_timeout(cancel_timeout, "cancel_timeout")
        self._max_frame_size = max_frame_size
        self._peer_max_frame_size = max_frame_size
        self._next_request_id = 1
        self._lock = threading.Lock()
        self._socket: socket.socket | None = None
        self._capabilities: dict[str, Any] | None = None
        self._fast_cuda_backend: dict[str, Any] | None = None
        self._handshake_complete = False

        try:
            self._socket = self._connect(self._connect_timeout)
            hello = self._request_locked(
                "hello",
                {
                    "client_name": self._client_name,
                    "client_version": self._client_version,
                    "supported_protocol_versions": [PROTOCOL_VERSION],
                },
                expected="hello",
                timeout=self._request_timeout,
                job_id=None,
                cancel_on_timeout=False,
            )
            self._accept_hello(hello)
        except Exception:
            self._close_socket()
            raise

    @property
    def capabilities(self) -> dict[str, Any]:
        if self._capabilities is None:
            raise EvaluatorTransportError("evaluator client is not handshaken")
        return json.loads(json.dumps(self._capabilities))

    @property
    def fast_cuda_backend(self) -> dict[str, Any] | None:
        """Return the protocol-v4 CUDA backend identity pinned at first hello."""
        if not self._handshake_complete:
            raise EvaluatorTransportError("evaluator client is not handshaken")
        if self._fast_cuda_backend is None:
            return None
        return json.loads(json.dumps(self._fast_cuda_backend))

    def __enter__(self) -> PrimeEvaluatorClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        with self._lock:
            self._close_socket()

    def reconnect(self) -> None:
        """Reconnect and repeat the strict registry/protocol handshake.

        Callers may use this only before replaying an idempotent registration
        or evaluation request. Artifact receipt consumption is deliberately
        not retried through this method.
        """
        with self._lock:
            self._close_socket()
            self._capabilities = None
            self._peer_max_frame_size = self._max_frame_size
            self._next_request_id = 1
            try:
                self._socket = self._connect(self._connect_timeout)
                hello = self._request_locked(
                    "hello",
                    {
                        "client_name": self._client_name,
                        "client_version": self._client_version,
                        "supported_protocol_versions": [PROTOCOL_VERSION],
                    },
                    expected="hello",
                    timeout=self._request_timeout,
                    job_id=None,
                    cancel_on_timeout=False,
                )
                self._accept_hello(hello)
            except Exception:
                self._close_socket()
                raise

    def register_dataset(
        self,
        *,
        task: str,
        rows: int,
        columns: int,
        features: Sequence[float],
        target: Sequence[float],
        claimed_content_hash: str | None = None,
        schema: Mapping[str, Any] | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        return self._request(
            "register_dataset",
            {
                "task": task,
                "rows": rows,
                "columns": columns,
                "features": list(features),
                "target": list(target),
                "claimed_content_hash": claimed_content_hash,
                "schema": dict(schema) if schema is not None else None,
            },
            timeout=timeout,
        )

    def register_fold_plan(
        self,
        *,
        dataset_id: str,
        k: int,
        assignments: Sequence[int] | None,
        seed: int,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        return self._request(
            "register_fold_plan",
            {
                "dataset_id": dataset_id,
                "k": k,
                "assignments": list(assignments) if assignments is not None else None,
                "seed": seed,
            },
            timeout=timeout,
        )

    def evaluate_batch(
        self,
        *,
        job_id: str,
        dataset_id: str,
        fold_plan_id: str,
        candidates: Sequence[Mapping[str, Any]],
        policy: Mapping[str, Any],
        timeout: float | None = None,
    ) -> dict[str, Any]:
        return self._request(
            "evaluate_batch",
            {
                "job_id": job_id,
                "dataset_id": dataset_id,
                "fold_plan_id": fold_plan_id,
                "candidates": [dict(candidate) for candidate in candidates],
                "policy": dict(policy),
            },
            timeout=timeout,
            job_id=job_id,
        )

    def promote_exact(
        self,
        *,
        job_id: str,
        source_job_id: str | None,
        dataset_id: str,
        fold_plan_id: str,
        candidates: Sequence[Mapping[str, Any]],
        seed: int,
        suite_hash: str | None,
        return_fold_scores: bool,
        compiled_only: bool = True,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        return self._request(
            "promote_exact",
            {
                "job_id": job_id,
                "source_job_id": source_job_id,
                "dataset_id": dataset_id,
                "fold_plan_id": fold_plan_id,
                "candidates": [dict(candidate) for candidate in candidates],
                "seed": seed,
                "compiled_only": compiled_only,
                "suite_hash": suite_hash,
                "return_fold_scores": return_fold_scores,
            },
            timeout=timeout,
            job_id=job_id,
        )

    def compile_artifact(
        self,
        *,
        job_id: str,
        candidate: Mapping[str, Any],
        source_promotion_job_id: str | None = None,
        artifact_kinds: Sequence[str],
        dataset_id: str | None,
        fold_plan_id: str | None,
        include_test_vectors: bool,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        return self._request(
            "compile_artifact",
            {
                "job_id": job_id,
                "candidate": dict(candidate),
                "source_promotion_job_id": source_promotion_job_id,
                "artifact_kinds": list(artifact_kinds),
                "dataset_id": dataset_id,
                "fold_plan_id": fold_plan_id,
                "include_test_vectors": include_test_vectors,
            },
            timeout=timeout,
            job_id=job_id,
        )

    def cancel_job(
        self,
        *,
        job_id: str,
        reason: str | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        return self._request("cancel_job", {"job_id": job_id, "reason": reason}, timeout=timeout)

    def stats(self, *, timeout: float | None = None) -> dict[str, Any]:
        return self._request("get_stats", None, timeout=timeout)

    def health(self, *, timeout: float | None = None) -> dict[str, Any]:
        response = self._request("health", None, timeout=timeout)
        if response.get("protocol_version") != PROTOCOL_VERSION:
            self._fail_protocol("health response protocol version mismatch")
        return response

    def _request(
        self,
        request_type: str,
        payload: Mapping[str, Any] | None,
        *,
        timeout: float | None,
        job_id: str | None = None,
    ) -> dict[str, Any]:
        request_timeout = self._request_timeout if timeout is None else _positive_timeout(timeout, "timeout")
        with self._lock:
            return self._request_locked(
                request_type,
                payload,
                expected=request_type,
                timeout=request_timeout,
                job_id=job_id,
                cancel_on_timeout=job_id is not None,
            )

    def _request_locked(
        self,
        request_type: str,
        payload: Mapping[str, Any] | None,
        *,
        expected: str,
        timeout: float,
        job_id: str | None,
        cancel_on_timeout: bool,
    ) -> dict[str, Any]:
        sock = self._socket
        if sock is None:
            raise EvaluatorTransportError("evaluator client connection is closed")
        request_id = self._take_request_id()
        request: dict[str, Any] = {"type": request_type}
        if payload is not None:
            request["payload"] = dict(payload)
        envelope = {
            "protocol_version": PROTOCOL_VERSION,
            "request_id": request_id,
            "request": request,
        }
        try:
            response = self._exchange(sock, envelope, timeout, self._peer_max_frame_size)
        except TimeoutError as error:
            self._close_socket()
            cancel_response = None
            cancel_error = None
            if cancel_on_timeout and job_id is not None:
                cancel_response, cancel_error = self._cancel_separately(job_id)
            raise EvaluatorTimeoutError(
                job_id=job_id,
                cancel_response=cancel_response,
                cancel_error=cancel_error,
            ) from error
        except EvaluatorProtocolError:
            self._close_socket()
            raise
        except OSError as error:
            self._close_socket()
            raise EvaluatorTransportError(f"evaluator socket error: {error}") from error
        try:
            return self._decode_response(response, request_id=request_id, expected=expected)
        except EvaluatorProtocolError:
            self._close_socket()
            raise

    def _exchange(
        self,
        sock: socket.socket,
        envelope: Mapping[str, Any],
        timeout: float,
        outgoing_limit: int,
    ) -> dict[str, Any]:
        try:
            payload = json.dumps(envelope, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode(
                "utf-8"
            )
        except (TypeError, ValueError) as error:
            raise ValueError(f"request is not valid finite JSON: {error}") from error
        if not payload or len(payload) > min(self._max_frame_size, outgoing_limit):
            raise EvaluatorProtocolError("outgoing evaluator frame exceeds negotiated limit")
        sock.settimeout(timeout)
        sock.sendall(struct.pack(">I", len(payload)) + payload)
        prefix = self._receive_exact(sock, 4)
        length = struct.unpack(">I", prefix)[0]
        if length == 0 or length > self._max_frame_size:
            raise EvaluatorProtocolError(
                f"invalid evaluator frame length {length}; maximum {self._max_frame_size}"
            )
        return _mapping(_strict_json(self._receive_exact(sock, length)), "response envelope")

    @staticmethod
    def _receive_exact(sock: socket.socket, length: int) -> bytes:
        chunks: list[bytes] = []
        remaining = length
        while remaining:
            chunk = sock.recv(remaining)
            if not chunk:
                raise OSError("evaluator closed the connection mid-frame")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def _decode_response(self, envelope: dict[str, Any], *, request_id: int, expected: str) -> dict[str, Any]:
        _exact_fields(
            envelope,
            {"protocol_version", "request_id", "response"},
            "response envelope",
        )
        if envelope["protocol_version"] != PROTOCOL_VERSION:
            self._fail_protocol("response protocol version mismatch")
        response_request_id = envelope["request_id"]
        if (
            not isinstance(response_request_id, int)
            or isinstance(response_request_id, bool)
            or response_request_id != request_id
        ):
            self._fail_protocol(f"response request id {response_request_id!r} did not match {request_id}")
        response = _mapping(envelope["response"], "response")
        _exact_fields(response, {"type", "payload"}, "response")
        response_type = response["type"]
        payload = _mapping(response["payload"], f"{response_type} payload")
        if response_type == "error":
            self._raise_remote(payload)
        if response_type != expected:
            self._fail_protocol(f"expected {expected!r} response, received {response_type!r}")
        return payload

    def _accept_hello(self, hello: dict[str, Any]) -> None:
        _exact_fields(
            hello,
            {"selected_protocol_version", "server_name", "server_version", "capabilities"},
            "hello payload",
        )
        if hello["selected_protocol_version"] != PROTOCOL_VERSION:
            raise EvaluatorProtocolError("server selected an unsupported protocol version")
        capabilities = _mapping(hello["capabilities"], "capabilities")
        required = {
            "tasks",
            "numerical_tiers",
            "backends",
            "artifact_kinds",
            "operator_registry_version",
            "operator_registry_hash",
            "emitter_version",
            "cuda_architectures",
            "max_frame_size",
            "max_batch_size",
            "supports_explicit_fold_assignments",
            "supports_cancellation",
            "supports_unix_socket",
            "supports_tcp",
        }
        _exact_fields(capabilities, required, "capabilities")
        backends = capabilities["backends"]
        if not isinstance(backends, list):
            raise EvaluatorProtocolError("capabilities.backends must be a list")
        fast_cuda_capabilities: list[tuple[int, dict[str, Any]]] = []
        for index, value in enumerate(backends):
            backend_capability = _mapping(value, f"capabilities.backends[{index}]")
            if backend_capability.get("selection") != "fast_cuda":
                continue
            _exact_fields(
                backend_capability,
                {"selection", "available", "backend", "diagnostic"},
                f"capabilities.backends[{index}]",
            )
            if not isinstance(backend_capability["available"], bool):
                raise EvaluatorProtocolError(f"capabilities.backends[{index}].available must be a boolean")
            fast_cuda_capabilities.append((index, backend_capability))
        if len(fast_cuda_capabilities) != 1:
            raise EvaluatorProtocolError("capabilities must advertise exactly one FastCuda backend")
        cuda_index, cuda_capability = fast_cuda_capabilities[0]
        if cuda_capability["available"]:
            if cuda_capability["diagnostic"] is not None:
                raise EvaluatorProtocolError(
                    f"capabilities.backends[{cuda_index}].diagnostic must be null when FastCuda is available"
                )
            fast_cuda_backend = _cuda_backend(
                cuda_capability["backend"],
                f"capabilities.backends[{cuda_index}].backend",
            )
        else:
            if cuda_capability["backend"] is not None:
                raise EvaluatorProtocolError(
                    f"capabilities.backends[{cuda_index}].backend must be null when FastCuda is unavailable"
                )
            fast_cuda_backend = None
        numerical_tiers = capabilities["numerical_tiers"]
        if not isinstance(numerical_tiers, list):
            raise EvaluatorProtocolError("capabilities.numerical_tiers must be a list")
        if ("fast_cuda" in numerical_tiers) != cuda_capability["available"]:
            raise EvaluatorProtocolError("FastCuda capability availability disagrees with numerical_tiers")
        cuda_architectures = capabilities["cuda_architectures"]
        if not isinstance(cuda_architectures, list):
            raise EvaluatorProtocolError("capabilities.cuda_architectures must be a list")
        if fast_cuda_backend is None:
            if cuda_architectures:
                raise EvaluatorProtocolError("unavailable FastCuda must not advertise CUDA architectures")
        else:
            if len(cuda_architectures) != 1:
                raise EvaluatorProtocolError(
                    "available FastCuda must advertise exactly one CUDA architecture"
                )
            architecture = _mapping(cuda_architectures[0], "capabilities.cuda_architectures[0]")
            _exact_fields(
                architecture,
                {"major", "minor"},
                "capabilities.cuda_architectures[0]",
            )
            _unsigned_integer(
                architecture["major"],
                "capabilities.cuda_architectures[0].major",
                bits=16,
                positive=True,
            )
            _unsigned_integer(
                architecture["minor"],
                "capabilities.cuda_architectures[0].minor",
                bits=16,
            )
            if architecture != fast_cuda_backend["compute_capability"]:
                raise EvaluatorProtocolError("CUDA architecture disagrees with the FastCuda backend")
        if self._handshake_complete and fast_cuda_backend != self._fast_cuda_backend:
            raise EvaluatorProtocolError("FastCuda backend identity changed across evaluator reconnect")
        received_hash = capabilities["operator_registry_hash"]
        if received_hash != self._expected_registry_hash:
            raise RegistryMismatchError(
                f"evaluator registry hash {received_hash!r} does not match {self._expected_registry_hash!r}"
            )
        if (
            self._expected_registry_version is not None
            and capabilities["operator_registry_version"] != self._expected_registry_version
        ):
            raise RegistryMismatchError("evaluator registry version mismatch")
        if capabilities["supports_unix_socket"] is not True:
            raise EvaluatorProtocolError("daemon does not advertise AF_UNIX support")
        peer_limit = capabilities["max_frame_size"]
        if not isinstance(peer_limit, int) or isinstance(peer_limit, bool) or peer_limit <= 0:
            raise EvaluatorProtocolError("invalid capability max_frame_size")
        self._peer_max_frame_size = min(peer_limit, self._max_frame_size)
        self._capabilities = capabilities
        self._fast_cuda_backend = fast_cuda_backend
        self._handshake_complete = True

    def _raise_remote(self, payload: dict[str, Any]) -> None:
        _exact_fields(payload, {"code", "message", "retriable", "details"}, "error payload")
        code = payload["code"]
        message = payload["message"]
        retriable = payload["retriable"]
        details = payload["details"]
        if code not in _ERROR_CODES or not isinstance(message, str):
            self._fail_protocol("invalid remote error code or message")
        if not isinstance(retriable, bool) or not isinstance(details, dict):
            self._fail_protocol("invalid remote error retry metadata")
        if not all(isinstance(key, str) and isinstance(value, str) for key, value in details.items()):
            self._fail_protocol("remote error details must map strings to strings")
        if code == "unknown_dataset":
            raise UnknownDatasetError(message=message, server_retriable=retriable, details=details)
        raise EvaluatorRemoteError(code=code, message=message, retriable=retriable, details=details)

    def _cancel_separately(self, job_id: str) -> tuple[dict[str, Any] | None, str | None]:
        sock: socket.socket | None = None
        try:
            sock = self._connect(self._cancel_timeout)
            hello_envelope = {
                "protocol_version": PROTOCOL_VERSION,
                "request_id": 1,
                "request": {
                    "type": "hello",
                    "payload": {
                        "client_name": self._client_name,
                        "client_version": self._client_version,
                        "supported_protocol_versions": [PROTOCOL_VERSION],
                    },
                },
            }
            hello_response = self._exchange(sock, hello_envelope, self._cancel_timeout, self._max_frame_size)
            hello = self._decode_response(hello_response, request_id=1, expected="hello")
            self._accept_hello(hello)
            cancel_envelope = {
                "protocol_version": PROTOCOL_VERSION,
                "request_id": 2,
                "request": {
                    "type": "cancel_job",
                    "payload": {
                        "job_id": job_id,
                        "reason": "client request timed out",
                    },
                },
            }
            response = self._exchange(sock, cancel_envelope, self._cancel_timeout, self._peer_max_frame_size)
            return (
                self._decode_response(response, request_id=2, expected="cancel_job"),
                None,
            )
        except Exception as error:  # best effort; preserve the original timeout
            return None, f"{type(error).__name__}: {error}"
        finally:
            if sock is not None:
                sock.close()

    def _connect(self, timeout: float) -> socket.socket:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            sock.settimeout(timeout)
            sock.connect(str(self._path))
            return sock
        except (OSError, TimeoutError) as error:
            sock.close()
            raise EvaluatorTransportError(
                f"could not connect to evaluator Unix socket {self._path}: {error}"
            ) from error

    def _take_request_id(self) -> int:
        request_id = self._next_request_id
        self._next_request_id = 1 if request_id == 0xFFFF_FFFF_FFFF_FFFF else request_id + 1
        return request_id

    def _close_socket(self) -> None:
        if self._socket is not None:
            self._socket.close()
            self._socket = None

    def _fail_protocol(self, message: str) -> None:
        self._close_socket()
        raise EvaluatorProtocolError(message)


__all__ = [
    "DEFAULT_MAX_FRAME_SIZE",
    "PROTOCOL_VERSION",
    "EvaluatorClientError",
    "EvaluatorProtocolError",
    "EvaluatorRemoteError",
    "EvaluatorTimeoutError",
    "EvaluatorTransportError",
    "PrimeEvaluatorClient",
    "RegistryMismatchError",
    "UnknownDatasetError",
]
