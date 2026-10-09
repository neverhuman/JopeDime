"""Versioned draw/slot JSONL with idempotent replay and a sealed byte manifest."""

import fcntl
import hashlib
import json
import math
import os
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

from .research_hwm import canonical_bytes, is_sha
from .trie_logprob import TrieLogProb


@dataclass(frozen=True)
class DecisionRecord:
    decision_id: str
    kind: str
    context_identity: str
    checkpoint_parameter_sha256: str
    trie_sha256: str
    normalizer_sha256: str
    tokenizer_sha256: str
    temperature: float
    seed: int
    tokens: tuple[int, ...]
    logprob: TrieLogProb
    dedup_group: str
    slate_id: str
    selection_rule: str
    propensity: float | None = None
    predicted: dict | None = None
    execution_key: str | None = None
    outcome_class: str | None = None
    game_spec_sha256: str | None = None

    def to_dict(self):
        if not isinstance(self.logprob, TrieLogProb):
            raise TypeError(
                "summed trie log-probability required; mean-token NLL refused"
            )
        if self.kind not in ("raw_draw", "selected_slot"):
            raise ValueError("one raw draw or selected slot per record")
        if not all(
            (
                self.decision_id,
                self.context_identity,
                self.dedup_group,
                self.slate_id,
                self.selection_rule,
            )
        ):
            raise ValueError(
                "decision/context/dedup/slate/selection identities required"
            )
        for pin in (
            self.checkpoint_parameter_sha256,
            self.trie_sha256,
            self.normalizer_sha256,
            self.tokenizer_sha256,
        ):
            if not is_sha(pin):
                raise ValueError(
                    "full parameter/support/normalizer/tokenizer pins required"
                )
        if type(self.seed) is not int or not 0 <= self.seed < 2**63:
            raise ValueError("bounded integer decoder seed required")
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("positive finite temperature required")
        if len(self.tokens) < 2 or any(
            type(x) is not int or x < 0 for x in self.tokens
        ):
            raise ValueError("BOS/EOS-inclusive integer tokens required")
        if len(self.logprob.per_token) != len(self.tokens) - 1:
            raise ValueError("one log-probability per token after BOS, including EOS")
        if self.propensity is not None and (
            not math.isfinite(self.propensity) or not 0 < self.propensity <= 1
        ):
            raise ValueError("propensity is null when unknown, otherwise in (0,1]")
        predicted = self.predicted or {}
        if set(predicted) - {"q", "p_complete", "runtime_seconds"}:
            raise ValueError(
                "predictions belong only to the dedicated predicted fields"
            )
        if any(
            not isinstance(x, (float, int)) or not math.isfinite(x)
            for x in predicted.values()
        ):
            raise ValueError("finite predictions required")
        if "p_complete" in predicted and not 0 <= predicted["p_complete"] <= 1:
            raise ValueError("predicted completion probability outside [0,1]")
        if "runtime_seconds" in predicted and predicted["runtime_seconds"] <= 0:
            raise ValueError("positive predicted runtime required")
        result = asdict(self)
        del result["logprob"]
        result.update(
            schema="jope.rlfw-decision/v2",
            tokens=list(self.tokens),
            per_token_logprob=list(self.logprob.per_token),
            total_logprob=self.logprob.total,
            logprob_support=self.logprob.support,
            logprob_aggregation=self.logprob.aggregation,
            predicted=predicted,
        )
        return result


class DecisionTrace:
    def __init__(self, directory, *, runs_root):
        self.directory = Path(directory).resolve()
        root = Path(runs_root).resolve()
        if root.name != "runs" or not self.directory.is_relative_to(root):
            raise ValueError(
                "decision traces must live beneath the explicit benchmark runs root"
            )
        if "flywheel_neural_continuous_20261009_v1" in self.directory.parts:
            raise ValueError("service writes are forbidden")
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "decisions.jsonl"
        self.manifest = self.directory / "decisions.manifest.json"
        self.lock = self.directory / "decisions.lock"

    def _sync_directory(self):
        fd = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def _recover_tail(self, body):
        if self.manifest.exists():
            raise ValueError("sealed trace has an incomplete tail")
        # Preserve and verify all interrupted bytes before repairing our trace.
        digest = hashlib.sha256(body).hexdigest()
        custody = self.directory / f"decisions.interrupted.{digest}.jsonl"
        if not custody.exists():
            with custody.open("xb") as stream:
                stream.write(body)
                stream.flush()
                os.fsync(stream.fileno())
            self._sync_directory()
        if hashlib.sha256(custody.read_bytes()).hexdigest() != digest:
            raise ValueError("interrupted trace custody mismatch")
        prefix = body[: body.rfind(b"\n") + 1]
        with self.path.open("wb") as stream:
            stream.write(prefix)
            stream.flush()
            os.fsync(stream.fileno())
        return prefix

    def append(self, record):
        if not isinstance(record, DecisionRecord):
            raise TypeError("validated DecisionRecord required")
        value = record.to_dict()
        raw = canonical_bytes(value) + b"\n"
        with self.lock.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            body = self.path.read_bytes() if self.path.exists() else b""
            if self.manifest.exists():
                manifest = json.loads(self.manifest.read_bytes())
                pin = manifest.pop("manifest_sha256")
                if (
                    hashlib.sha256(canonical_bytes(manifest)).hexdigest() != pin
                    or hashlib.sha256(body).hexdigest() != manifest["sha256"]
                ):
                    raise ValueError("sealed trace digest mismatch")
            if body and not body.endswith(b"\n"):
                body = self._recover_tail(body)
            for line in body.splitlines():
                existing = json.loads(line)
                if existing["decision_id"] == record.decision_id:
                    if canonical_bytes(existing) + b"\n" != raw:
                        raise ValueError("conflicting replay of decision identity")
                    return hashlib.sha256(raw).hexdigest()
            if self.manifest.exists():
                raise ValueError("sealed trace cannot accept new decisions")
            with self.path.open("ab") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            self._sync_directory()
        return hashlib.sha256(raw).hexdigest()

    def seal(self):
        with self.lock.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            body = self.path.read_bytes()
            if not body.endswith(b"\n"):
                raise ValueError("complete nonempty JSONL required")
            records = [json.loads(line) for line in body.splitlines()]
            value = {
                "schema": "jope.rlfw-decision-manifest/v2",
                "file": self.path.name,
                "sha256": hashlib.sha256(body).hexdigest(),
                "records": len(records),
            }
            value["manifest_sha256"] = hashlib.sha256(
                canonical_bytes(value)
            ).hexdigest()
            raw = canonical_bytes(value) + b"\n"
            if self.manifest.exists():
                if self.manifest.read_bytes() != raw:
                    raise ValueError("sealed trace digest mismatch")
            else:
                staged = (
                    self.directory / f"decisions.manifest.{uuid.uuid4().hex}.staging"
                )
                with staged.open("xb") as stream:
                    stream.write(raw)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(staged, self.manifest)
                self._sync_directory()
            return value
