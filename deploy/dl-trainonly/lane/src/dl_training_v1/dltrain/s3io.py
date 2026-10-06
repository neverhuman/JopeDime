"""Guarded S3 access for the dltrain lane (PLAN §2 credentials, §6.6 publish protocol, §5 P0 probe).

The principal behind AWS_PROFILE=veox-jepson-prod has AdministratorAccess, so the guard below is the only thing between
a coding mistake and an irreversible S3 action. Every client built by make_client():
- talks only to the pinned AWS endpoints (configured endpoint overrides are ignored, and the final URL is checked);
- refuses presigning (instance presigners shadowed; any query or POST-policy signature refused at before-sign);
- runs the guard first on the deepest wildcard nodes (before-parameter-build.*.*, before-call.*.*, before-sign.*.*,
  before-send.*.*), so the guard runs ahead of any handler registered later;
- allows only GetObject, HeadObject, bounded ListObjectsV2 and (on write clients only) PutObject;
- requires a PutObject to sit inside PREFIX and to carry exactly the allowlisted parameters. It must also have
  ChecksumAlgorithm=SHA256 with a matching ChecksumSHA256, and exactly one condition: IfNoneMatch='*' (create-only), or
  IfMatch=<strong ETag> on a CAS pointer key (…/HEAD.json) or a probe key;
- refuses reads outside PREFIX unless they pin an exact VersionId. The one exception is the recorded-only HEAD of the
  incoming latest.json pointer (PLAN P1).
Write clients and the lifecycle-once client never retry automatically (total_max_attempts=1). Read clients make up to 3
attempts and cannot write.

The G18 'lifecycle-once' client allows exactly one GetBucketLifecycleConfiguration on the lane bucket (one HTTP
attempt) and nothing else.

v2 train-only outputs (owner decision L6(e)) live under V2_PREFIX and are written only by a client made with
make_client(writes=True, v2=True, ledger_path=...). That writer:
- puts only inside V2_PREFIX (never v1 PREFIX, never anywhere else); the default writer still refuses every v2 key;
- is strictly create-only: IfNoneMatch='*' and no IfMatch (no pointer CAS in v2), plus the same checksum and parameter
  allowlist as every lane PUT;
- reserves each PUT's bytes in an append-only, hash-chained local ledger (V2Ledger) before the request is built and
  refuses the PUT that would take cumulative v2 bytes above V2_BUDGET_BYTES (40e9). The ledger survives restarts and is
  shared across processes under flock.
A v2-flagged client may also read unpinned inside V2_PREFIX (create-only keys hold one object). TabPFN 3.5 outputs go
only under V2_RESEARCH_ONLY_KEY; production loaders call assert_production_loadable / production_get_exact, which refuse
every research-only key.
"""
import argparse
import base64
import contextlib
import datetime
import fcntl
import hashlib
import json
import os
import re
import sys
import threading
from pathlib import Path
from urllib.parse import unquote
from urllib.parse import urlsplit

import boto3
import botocore
from botocore.config import Config
from botocore.exceptions import ClientError

from dltrain import common

BUCKET = "veox-jopedime-learning-use2"
PREFIX_KEY = "experiments/gp-policy-dl/v1/"
PREFIX = f"s3://{BUCKET}/{PREFIX_KEY}"
PROFILE = "veox-jepson-prod"
REGION = "us-east-2"
S3_HOST = f"s3.{REGION}.amazonaws.com"
STS_ENDPOINT = f"https://sts.{REGION}.amazonaws.com"
READ_OPS = {"GetObject", "HeadObject", "ListObjectsV2", "ListObjectVersions"}
ALLOWED_OPS = READ_OPS | {"PutObject"}
OP_METHOD = {"GetObject": "GET", "HeadObject": "HEAD", "ListObjectsV2": "GET", "ListObjectVersions": "GET", "PutObject": "PUT",
             "GetBucketLifecycleConfiguration": "GET"}
HISTORY_PARAMS = {"Bucket", "Prefix", "MaxKeys"}
HISTORY_MAX = 100
PUT_PARAMS = {"Bucket", "Key", "Body", "ContentType", "IfNoneMatch", "IfMatch", "ChecksumAlgorithm", "ChecksumSHA256"}
ETAG_RE = re.compile(r'"[0-9a-f]{32}(-[0-9]+)?"')
CAS_KEY_RE = re.compile(r"(^|/)HEAD\.json$")
UNPINNED_HEAD_ALLOWED = {("veox-jopedime-datasets-use2", "research/regression-2k/incoming/v1/latest.json")}
CAS_BUCKETS = {"veox-jopedime-datasets-use2", "veox-jopedime-models-use2"}
CAS_KEY_FULL_RE = re.compile(r"blobs/sha256/[0-9a-f]{64}")
LIST_MAX = 1000
HARMFUL_LIFECYCLE_ACTIONS = ("Transitions", "NoncurrentVersionTransitions", "NoncurrentVersionExpiration")
RULE_KEYS = {"Expiration", "ID", "Prefix", "Filter", "Status", "Transitions", "NoncurrentVersionTransitions",
             "NoncurrentVersionExpiration", "AbortIncompleteMultipartUpload"}
EXPIRATION_KEYS = {"Days", "Date", "ExpiredObjectDeleteMarker"}
REQUIRED_STEPS = ("1_create_only", "2_pointer_cas", "3_race", "4_ifmatch_missing_key", "5_head_flags", "6_exact_gets")
# v2 train-only (owner decision L6(e)): create-only, <= 40e9 bytes in total, never overwrite or delete; v1 untouched.
V2_PREFIX_KEY = "experiments/gp-policy-dl/v2-trainonly/"
V2_PREFIX = f"s3://{BUCKET}/{V2_PREFIX_KEY}"
V2_RESEARCH_ONLY_KEY = V2_PREFIX_KEY + "research-only/"
V2_BUDGET_BYTES = 40_000_000_000
V2_LEDGER_PATH = Path("/home/ubuntu/scratch/jopedime-dl-training-v1/receipts/trainonly-v2/s3-v2-ledger.jsonl")
V2_LEDGER_SCHEMA = "jope.dl-s3-v2-ledger/v1"
V2_LEDGER_GENESIS = "0" * 64
V2_LEDGER_FIELDS = {"schema", "event", "seq", "prev", "key", "sha256", "bytes", "used_after", "cap", "utc_us", "pid"}
V2_SEGMENT_RE = re.compile(r"[A-Za-z0-9._=+-]+")
SHA256_RE = re.compile(r"[0-9a-f]{64}")


class S3Refused(RuntimeError):
    """The lane guard refused an S3 call before any request was sent."""


class BudgetExceeded(S3Refused):
    """A v2 PUT would take cumulative v2 bytes above V2_BUDGET_BYTES."""


class ResearchOnlyRefused(S3Refused):
    """A production loader was asked for a research-only key."""


class LedgerCorrupt(RuntimeError):
    """The v2 byte-budget ledger is not a well-formed append-only chain; every v2 PUT fails closed."""


class PoisonedKey(RuntimeError):
    """A create-only key already holds different bytes."""


class ReadbackMismatch(RuntimeError):
    """An exact-version readback did not match the bytes written."""


def _in_prefix(bucket, key) -> bool:
    return bucket == BUCKET and isinstance(key, str) and key.startswith(PREFIX_KEY) and ".." not in key.split("/")


def _in_v2_prefix(bucket, key) -> bool:
    """A full object key strictly below V2_PREFIX_KEY: plain segments only (no empty, '.' or '..' segment)."""
    if bucket != BUCKET or not isinstance(key, str) or not key.startswith(V2_PREFIX_KEY):
        return False
    segs = key[len(V2_PREFIX_KEY):].split("/")
    return all(V2_SEGMENT_RE.fullmatch(s) and s not in (".", "..") for s in segs)


def _url_key(url):
    """The object key a before-call request URL addresses on the lane bucket (virtual-hosted or path style), else None."""
    if not isinstance(url, str):
        return None
    parts = urlsplit(url)
    if parts.hostname == f"{BUCKET}.{S3_HOST}":
        return unquote(parts.path[1:])
    if parts.hostname == S3_HOST and parts.path.startswith(f"/{BUCKET}/"):
        return unquote(parts.path[len(BUCKET) + 2:])
    return None


def is_research_only(key) -> bool:
    """True for a key no production loader may read: under V2_RESEARCH_ONLY_KEY, or any 'research-only' or 'tabpfn*'
    path segment anywhere (so '//' or case variants cannot smuggle one past the prefix test). Non-strings count as True."""
    if not isinstance(key, str) or key.startswith(V2_RESEARCH_ONLY_KEY):
        return True
    return any(s == "research-only" or s.startswith("tabpfn") for s in key.lower().split("/"))


def assert_production_loadable(key, bucket: str = BUCKET) -> str:
    if is_research_only(key):
        raise ResearchOnlyRefused(f"production loader refuses research-only key s3://{bucket}/{key}")
    return key


def production_get_exact(client, key: str, version_id: str, bucket: str = BUCKET) -> tuple[bytes, dict]:
    """The production loader's exact-version GET: refuses research-only keys before any request."""
    assert_production_loadable(key, bucket)
    return get_exact(client, key, version_id, bucket=bucket)


class V2Ledger:
    """Append-only, hash-chained JSONL of every v2 PutObject byte reservation (owner decision L6(e): <= 40e9 bytes).

    The guard reserves a PUT's bytes before the request is built; a reservation stays counted whatever the outcome (a
    timed-out PUT may have written). Usage is the sum over distinct (key, sha256): a create-only key holds at most one
    object, so a retry of the same bytes on the same key is not counted twice, while a different body is counted again.
    Every read-check-append runs under flock on the ledger file, so concurrent processes share one budget. The file is
    never truncated or rewritten; a shrunk file, a partial line, a broken chain or an unknown record raises LedgerCorrupt.
    """

    def __init__(self, path):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._offset, self._prev, self._seq, self._used, self._seen = 0, V2_LEDGER_GENESIS, 0, 0, set()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        new = not self.path.exists()
        with self._lock, self._locked() as fd:
            self._sync(fd)
        if new:
            dfd = os.open(str(self.path.parent), os.O_RDONLY)
            try:
                os.fsync(dfd)
            finally:
                os.close(dfd)

    @contextlib.contextmanager
    def _locked(self):
        fd = os.open(str(self.path), os.O_RDWR | os.O_APPEND | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield fd
        finally:
            os.close(fd)  # releases the flock

    def _sync(self, fd):
        size = os.fstat(fd).st_size
        if size < self._offset:
            raise LedgerCorrupt(f"{self.path} shrank from {self._offset} to {size} bytes")
        if size == self._offset:
            return
        chunks, pos = [], self._offset
        while pos < size:
            got = os.pread(fd, size - pos, pos)
            if not got:
                raise LedgerCorrupt(f"{self.path}: short read at {pos}")
            chunks.append(got)
            pos += len(got)
        data = b"".join(chunks)
        if not data.endswith(b"\n"):
            raise LedgerCorrupt(f"{self.path} ends in a partial line")
        for line in data[:-1].split(b"\n"):
            self._apply(line)
        self._offset = size

    def _apply(self, line: bytes):
        try:
            rec = json.loads(line)
        except ValueError as err:
            raise LedgerCorrupt(f"{self.path} line {self._seq}: not JSON") from err
        if not isinstance(rec, dict) or set(rec) != V2_LEDGER_FIELDS or rec["schema"] != V2_LEDGER_SCHEMA or rec["event"] != "reserve":
            raise LedgerCorrupt(f"{self.path} line {self._seq}: unknown record shape")
        if type(rec["seq"]) is not int or rec["seq"] != self._seq or rec["prev"] != self._prev:
            raise LedgerCorrupt(f"{self.path} line {self._seq}: broken chain")
        key, sha, n = rec["key"], rec["sha256"], rec["bytes"]
        if not _in_v2_prefix(BUCKET, key) or not isinstance(sha, str) or not SHA256_RE.fullmatch(sha) or type(n) is not int or n < 0:
            raise LedgerCorrupt(f"{self.path} line {self._seq}: bad key, sha256 or bytes")
        if (key, sha) not in self._seen:
            self._seen.add((key, sha))
            self._used += n
        if rec["used_after"] != self._used:
            raise LedgerCorrupt(f"{self.path} line {self._seq}: used_after {rec['used_after']} != recomputed {self._used}")
        self._prev = hashlib.sha256(line).hexdigest()
        self._seq += 1

    def usage(self) -> dict:
        with self._lock, self._locked() as fd:
            self._sync(fd)
            return {"path": str(self.path), "used_bytes": self._used, "cap_bytes": V2_BUDGET_BYTES,
                    "remaining_bytes": V2_BUDGET_BYTES - self._used, "reservations": self._seq, "distinct_objects": len(self._seen)}

    def reserve(self, key: str, sha256: str, nbytes: int) -> dict:
        if not _in_v2_prefix(BUCKET, key):
            raise S3Refused(f"v2 ledger refuses a key outside V2_PREFIX: {key!r}")
        if not isinstance(sha256, str) or not SHA256_RE.fullmatch(sha256) or type(nbytes) is not int or nbytes < 0:
            raise ValueError("reserve needs a lowercase sha256 hex digest and a non-negative int byte count")
        with self._lock, self._locked() as fd:
            self._sync(fd)
            add = 0 if (key, sha256) in self._seen else nbytes
            if self._used + add > V2_BUDGET_BYTES:
                raise BudgetExceeded(f"v2 byte budget: {self._used} used + {add} new > {V2_BUDGET_BYTES} cap; PUT of {key} refused")
            rec = {"schema": V2_LEDGER_SCHEMA, "event": "reserve", "seq": self._seq, "prev": self._prev, "key": key, "sha256": sha256,
                   "bytes": nbytes, "used_after": self._used + add, "cap": V2_BUDGET_BYTES, "utc_us": common.utc_us(), "pid": os.getpid()}
            line = common.canonical(rec)
            data = line + b"\n"
            written = 0
            while written < len(data):
                written += os.write(fd, data[written:])
            os.fsync(fd)
            self._apply(line)
            self._offset += len(data)
            return {"key": key, "sha256": sha256, "bytes": nbytes, "counted_bytes": add, "used_after": self._used, "seq": rec["seq"]}


def _checksum_b64(body: bytes) -> str:
    return base64.b64encode(hashlib.sha256(body).digest()).decode()


class Guard:
    """Per-client guard state. mode is 'lane' or 'lifecycle-once'; writes enables PutObject (write clients only).
    v2=True (lane mode only) switches the write scope from PREFIX to V2_PREFIX, makes PUTs strictly create-only and,
    on a writer, charges every PUT to the v2 byte-budget ledger (required)."""

    def __init__(self, mode: str = "lane", writes: bool = False, v2: bool = False, ledger=None):
        if mode not in ("lane", "lifecycle-once", "cas-read"):
            raise ValueError(mode)
        if mode in ("lifecycle-once", "cas-read") and writes:
            raise ValueError(f"{mode} clients never write")
        if v2 and mode != "lane":
            raise ValueError("v2 is a lane-client flag")
        if v2 and writes and ledger is None:
            raise ValueError("a v2 writer needs the persisted v2 byte-budget ledger")
        if ledger is not None and not (v2 and writes):
            raise ValueError("only a v2 writer takes a byte-budget ledger")
        self.mode, self.writes, self.v2 = mode, writes, bool(v2)
        self.ledger = ledger if ledger is None or isinstance(ledger, V2Ledger) else V2Ledger(ledger)
        self.lifecycle_used = False
        self.lock = threading.Lock()

    def check_params(self, params, model, **_):
        op = model.name
        bucket, key = params.get("Bucket"), params.get("Key")
        if self.mode == "lifecycle-once":
            with self.lock:
                if op == "GetBucketLifecycleConfiguration" and bucket == BUCKET and set(params) <= {"Bucket"} and not self.lifecycle_used:
                    self.lifecycle_used = True
                    return
            raise S3Refused(f"lifecycle-once client refuses {op} (used={self.lifecycle_used})")
        if self.mode == "cas-read":
            if op == "GetObject" and bucket in CAS_BUCKETS and isinstance(key, str) and CAS_KEY_FULL_RE.fullmatch(key) \
                    and set(params) <= {"Bucket", "Key"}:
                return
            raise S3Refused(f"cas-read client allows only GetObject of blobs/sha256/<64 hex> in {sorted(CAS_BUCKETS)}")
        if op not in ALLOWED_OPS:
            raise S3Refused(f"operation {op} is not allowed in this lane")
        if op == "PutObject":
            self._check_put(params, bucket, key)
            return
        if op == "ListObjectVersions":
            # P3 LP-1: the version history of exactly one key inside PREFIX (callers filter Key == prefix), bounded.
            prefix, mk = params.get("Prefix"), params.get("MaxKeys")
            if set(params) != HISTORY_PARAMS or bucket != BUCKET:
                raise S3Refused("ListObjectVersions: only Bucket, Prefix and MaxKeys on the lane bucket")
            if not (_in_prefix(bucket, prefix) and not prefix.endswith("/")):
                raise S3Refused("ListObjectVersions: Prefix must be one full key inside PREFIX")
            if not isinstance(mk, int) or isinstance(mk, bool) or not 1 <= mk <= HISTORY_MAX:
                raise S3Refused(f"ListObjectVersions: 1 <= MaxKeys <= {HISTORY_MAX}")
            return
        if op == "ListObjectsV2":
            mk = params.get("MaxKeys")
            if not isinstance(mk, int) or not 1 <= mk <= LIST_MAX or not params.get("Prefix"):
                raise S3Refused("ListObjectsV2 must carry a Prefix and 1 <= MaxKeys <= 1000")
            return
        if params.get("VersionId") == "null":
            raise S3Refused(f"{op} with VersionId 'null' refused (versioned buckets only)")
        if _in_prefix(bucket, key) or (self.v2 and _in_v2_prefix(bucket, key)) or params.get("VersionId"):
            return
        if op == "HeadObject" and (bucket, key) in UNPINNED_HEAD_ALLOWED:
            return
        raise S3Refused(f"{op} outside PREFIX without an exact VersionId: s3://{bucket}/{key}")

    def _check_put(self, params, bucket, key):
        if not self.writes:
            raise S3Refused("read-only client refuses PutObject")
        extra = set(params) - PUT_PARAMS
        if extra:
            raise S3Refused(f"PutObject parameters outside the allowlist refused: {sorted(extra)}")
        in_scope = _in_v2_prefix if self.v2 else _in_prefix
        if not in_scope(bucket, key):
            raise S3Refused(f"PutObject outside {'V2_PREFIX' if self.v2 else 'PREFIX'}: s3://{bucket}/{key}")
        inm, im = params.get("IfNoneMatch"), params.get("IfMatch")
        if self.v2 and (inm != "*" or im is not None):
            raise S3Refused("v2 PutObject must be create-only: IfNoneMatch='*' and no IfMatch")
        if (inm is None) == (im is None):
            raise S3Refused("PutObject needs exactly one condition: IfNoneMatch='*' or IfMatch=<ETag>")
        if inm is not None and inm != "*":
            raise S3Refused("IfNoneMatch must be '*'")
        if im is not None:
            if not ETAG_RE.fullmatch(im):
                raise S3Refused("IfMatch must be a quoted strong ETag")
            if not (CAS_KEY_RE.search(key) or key.startswith(PREFIX_KEY + "probes/")):
                raise S3Refused("IfMatch is allowed only on CAS pointer keys (…/HEAD.json) and probe keys")
        body = params.get("Body")
        if not isinstance(body, (bytes, bytearray)):
            raise S3Refused("PutObject Body must be bytes (single-request PUT)")
        if params.get("ChecksumAlgorithm") != "SHA256" or params.get("ChecksumSHA256") != _checksum_b64(bytes(body)):
            raise S3Refused("PutObject needs ChecksumAlgorithm=SHA256 and a matching ChecksumSHA256")
        if self.v2:  # last: only a PUT that passed every rule is charged to the budget
            self.ledger.reserve(key, common.sha256_bytes(bytes(body)), len(body))

    def check_request(self, model, params, **_):
        op = model.name
        if self.mode == "cas-read":
            if op != "GetObject":
                raise S3Refused(f"cas-read client refuses {op} (request stage)")
            return
        if self.mode == "lifecycle-once":
            if op != "GetBucketLifecycleConfiguration":
                raise S3Refused(f"lifecycle-once client refuses {op} (request stage)")
            return
        if op not in ALLOWED_OPS:
            raise S3Refused(f"operation {op} is not allowed in this lane (request stage)")
        if op == "PutObject":
            if not self.writes:
                raise S3Refused("read-only client refuses PutObject (request stage)")
            headers = {k.lower(): v for k, v in (params.get("headers") or {}).items()}
            if "if-none-match" not in headers and "if-match" not in headers:
                raise S3Refused("PutObject request without If-None-Match/If-Match header refused")
            url_key = _url_key(params.get("url"))
            if self.v2:
                if headers.get("if-none-match") != "*" or "if-match" in headers:
                    raise S3Refused("v2 PutObject request needs If-None-Match: * and no If-Match (request stage)")
                if url_key is None or not _in_v2_prefix(BUCKET, url_key):
                    raise S3Refused("v2 PutObject request URL is not a V2_PREFIX key on the lane bucket (request stage)")
            elif url_key is not None and url_key.startswith(V2_PREFIX_KEY):
                raise S3Refused("the default writer refuses a V2_PREFIX key (request stage)")

    def check_sign(self, request, operation_name, signature_version=None, **_):
        sv = str(signature_version or "")
        if sv.endswith(("-query", "-presign-post")):
            raise S3Refused(f"presigned signature {sv} refused")
        expected = OP_METHOD.get(operation_name)
        if expected is None or getattr(request, "method", None) != expected:
            raise S3Refused(f"signing {operation_name} as {getattr(request, 'method', None)} refused")

    def check_send(self, request, **_):
        parts = urlsplit(request.url)
        host = parts.hostname or ""
        if parts.scheme != "https" or not (host == S3_HOST or host.endswith("." + S3_HOST)):
            raise S3Refused(f"request to an unpinned endpoint refused: {parts.scheme}://{host}")


def _refuse_presign(*_a, **_k):
    raise S3Refused("presigning is refused in this lane")


def make_client(mode: str = "lane", writes: bool = False, session=None, v2: bool = False, ledger_path=None):
    """v2=True with writes=True is the only client that may PUT under V2_PREFIX; it needs an explicit ledger_path
    (the real one is V2_LEDGER_PATH). The default writer keeps PREFIX-only writes."""
    if v2 and writes and ledger_path is None:
        raise ValueError("make_client(writes=True, v2=True) needs an explicit ledger_path (V2_LEDGER_PATH for real writes)")
    guard = Guard(mode, writes, v2=v2, ledger=V2Ledger(ledger_path) if ledger_path is not None else None)
    session = session or boto3.session.Session(profile_name=PROFILE, region_name=REGION)
    single = writes or mode == "lifecycle-once"  # cas-read keeps read retries (the body is hash-checked)
    # botocore: total_max_attempts counts the initial call (max_attempts would count retries only).
    cfg = Config(region_name=REGION, retries={"total_max_attempts": 1 if single else 3, "mode": "standard"},
                 connect_timeout=10, read_timeout=60, ignore_configured_endpoint_urls=True)
    client = session.client("s3", config=cfg)
    if urlsplit(client.meta.endpoint_url).hostname != S3_HOST:
        raise S3Refused(f"unexpected S3 endpoint {client.meta.endpoint_url}")
    ev = client.meta.events
    # The deepest wildcard nodes run first in botocore's emitter, and first-section handlers run in registration order.
    ev.register_first("before-parameter-build.*.*", guard.check_params, unique_id="dltrain-guard-params")
    ev.register_first("before-call.*.*", guard.check_request, unique_id="dltrain-guard-request")
    ev.register_first("before-sign.*.*", guard.check_sign, unique_id="dltrain-guard-sign")
    ev.register_first("before-send.*.*", guard.check_send, unique_id="dltrain-guard-send")
    client.generate_presigned_url = _refuse_presign
    client.generate_presigned_post = _refuse_presign
    client._dltrain_guard = guard  # noqa: SLF001 (tests inspect it)
    return client


def sts_client(session=None):
    session = session or boto3.session.Session(profile_name=PROFILE, region_name=REGION)
    c = session.client("sts", endpoint_url=STS_ENDPOINT, config=Config(region_name=REGION, ignore_configured_endpoint_urls=True))
    if c.meta.endpoint_url != STS_ENDPOINT:
        raise S3Refused(f"unexpected STS endpoint {c.meta.endpoint_url}")
    return c


def _status(err) -> int:
    return int(getattr(err, "response", {}).get("ResponseMetadata", {}).get("HTTPStatusCode") or 0)


def _code(err) -> str:
    return str(getattr(err, "response", {}).get("Error", {}).get("Code") or "")


def safe_err(err: BaseException) -> dict:
    """Allowlisted error summary: never free text from botocore or credential providers (SH-1)."""
    out = {"type": type(err).__name__}
    if isinstance(err, ClientError):
        out.update(code=_code(err), status=_status(err), operation=getattr(err, "operation_name", None))
    elif isinstance(err, (S3Refused, ReadbackMismatch, PoisonedKey, LedgerCorrupt, common.ImmutableConflict)):
        out["message"] = str(err)[:500]
    elif isinstance(err, botocore.exceptions.CredentialRetrievalError):
        out["provider"] = getattr(err, "kwargs", {}).get("provider")
    elif isinstance(err, (botocore.exceptions.EndpointConnectionError, botocore.exceptions.ConnectTimeoutError,
                          botocore.exceptions.ReadTimeoutError, botocore.exceptions.ConnectionClosedError)):
        out["endpoint_url"] = getattr(err, "kwargs", {}).get("endpoint_url")
    return out


def get_exact(client, key: str, version_id: str, bucket: str = BUCKET) -> tuple[bytes, dict]:
    resp = client.get_object(Bucket=bucket, Key=key, VersionId=version_id)
    body = resp["Body"].read()
    return body, {k: resp.get(k) for k in ("VersionId", "ETag", "ContentLength")}


def get_content_addressed(client, bucket: str, key: str) -> tuple[bytes, dict]:
    """GET a content-addressed blob (key blobs/sha256/<sha>) with a cas-read client; the body must hash to the key."""
    if client._dltrain_guard.mode != "cas-read":  # noqa: SLF001
        raise S3Refused("get_content_addressed needs a cas-read client")
    resp = client.get_object(Bucket=bucket, Key=key)
    body = resp["Body"].read()
    if common.sha256_bytes(body) != key.rsplit("/", 1)[-1]:
        raise ReadbackMismatch(f"s3://{bucket}/{key}: body sha256 does not match the content-addressed key")
    return body, {"VersionId": resp.get("VersionId"), "ETag": resp.get("ETag"), "ContentLength": resp.get("ContentLength")}


def verify_readback(client, key: str, version_id: str, body: bytes) -> dict:
    got, meta = get_exact(client, key, version_id)
    if got != body:
        raise ReadbackMismatch(f"s3://{BUCKET}/{key}@{version_id}: {common.sha256_bytes(got)} != {common.sha256_bytes(body)}")
    return {"key": key, "version_id": version_id, "sha256": common.sha256_bytes(got), "bytes": len(got), "etag": meta.get("ETag")}


def put_create_only(client, key: str, body: bytes, readback_client, content_type: str = "application/json") -> dict:
    """Create-only PUT (IfNoneMatch='*'), then an exact-version readback with a read client.
    A 412 whose current bytes are equal is success (our own earlier write); different bytes raise PoisonedKey."""
    try:
        resp = client.put_object(Bucket=BUCKET, Key=key, Body=body, IfNoneMatch="*", ContentType=content_type,
                                 ChecksumAlgorithm="SHA256", ChecksumSHA256=_checksum_b64(body))
    except ClientError as err:
        if _status(err) == 412:
            cur = readback_client.get_object(Bucket=BUCKET, Key=key)
            existing = cur["Body"].read()
            if existing == body:
                return {"key": key, "version_id": cur.get("VersionId"), "sha256": common.sha256_bytes(body), "bytes": len(body),
                        "outcome": "already_present_equal_bytes"}
            raise PoisonedKey(f"s3://{BUCKET}/{key} holds different bytes") from err
        raise
    vid = resp.get("VersionId")
    if not vid or vid == "null":
        raise RuntimeError(f"PutObject returned no usable VersionId for {key}: {vid!r}")
    out = verify_readback(readback_client, key, vid, body)
    out.update(outcome="created", etag_put=resp.get("ETag"))
    return out


def put_if_match(client, key: str, body: bytes, etag: str, content_type: str = "application/json") -> dict:
    """CAS PUT on a pointer: succeeds only if the current ETag equals etag. The caller handles 412/409 by re-reading."""
    resp = client.put_object(Bucket=BUCKET, Key=key, Body=body, IfMatch=etag, ContentType=content_type,
                             ChecksumAlgorithm="SHA256", ChecksumSHA256=_checksum_b64(body))
    return {"key": key, "version_id": resp.get("VersionId"), "etag": resp.get("ETag")}


# ----------------------------------------------------------------------------------------------- G18 lifecycle
def _json_safe(o):
    if isinstance(o, datetime.datetime):
        return o.astimezone(datetime.timezone.utc).isoformat().replace("+00:00", "Z")
    raise TypeError(type(o).__name__)


def normalize_rules(rules) -> list[dict]:
    return json.loads(json.dumps(rules, default=_json_safe))


def _prefix_compatible(p: str) -> bool:
    return PREFIX_KEY.startswith(p) or p.startswith(PREFIX_KEY)


def _filter_can_match(rule: dict) -> tuple[bool, str]:
    """True when the rule's filter can select objects under PREFIX_KEY (tag and size filters count conservatively)."""
    if "Filter" not in rule:
        p = rule.get("Prefix")
        if p is None:
            return True, "no filter (counted as can-match)"
        return _prefix_compatible(p or ""), f"legacy prefix {p!r}"
    f = rule.get("Filter") or {}
    if not f:
        return True, "empty filter"
    if "Prefix" in f and len(f) == 1:
        p = f.get("Prefix") or ""
        return _prefix_compatible(p), f"prefix {p!r}"
    if "And" in f and len(f) == 1:
        a = f["And"] or {}
        p = a.get("Prefix")
        if p is not None and not _prefix_compatible(p):
            return False, f"and-filter prefix {p!r} disjoint"
        return True, "and-filter (prefix compatible or absent)"
    return True, f"filter {sorted(f)} (counted as can-match)"


def _harmful_actions(rule: dict) -> list[str]:
    acts = [a for a in HARMFUL_LIFECYCLE_ACTIONS if rule.get(a)]
    exp = rule.get("Expiration") or {}
    if exp.get("Days") is not None or exp.get("Date") is not None:
        acts.append("Expiration(Days/Date)")
    acts += [f"unrecognized rule key {k}" for k in sorted(set(rule) - RULE_KEYS)]
    acts += [f"unrecognized expiration key {k}" for k in sorted(set(exp) - EXPIRATION_KEYS)]
    return acts


def evaluate_lifecycle(rules: list[dict]) -> dict:
    """PLAN §6.6 G18. A matching harmful rule blocks (FAIL) unless Disabled; a Disabled one needs OWNER_REVIEW."""
    rows, blocking, review = [], [], []
    for r in rules:
        can, why = _filter_can_match(r)
        harmful = _harmful_actions(r)
        disabled = r.get("Status") == "Disabled"
        row = {"id": r.get("ID"), "status": r.get("Status"), "filter_can_match_prefix": can, "filter_reason": why,
               "harmful_actions": harmful, "blocking": bool(can and harmful and not disabled),
               "owner_review": bool(can and harmful and disabled)}
        rows.append(row)
        if row["blocking"]:
            blocking.append(row["id"])
        if row["owner_review"]:
            review.append(row["id"])
    verdict = "FAIL" if blocking else ("OWNER_REVIEW" if review else "PASS")
    return {"prefix": PREFIX, "rules": rows, "blocking_rules": blocking, "owner_review_rules": review, "verdict": verdict}


def _http_meta(meta: dict) -> dict:
    return {"status": meta.get("HTTPStatusCode"), "request_id": meta.get("RequestId"), "host_id": meta.get("HostId"),
            "retry_attempts": meta.get("RetryAttempts")}


def lifecycle_readback(session=None, client=None) -> dict:
    client = client or make_client(mode="lifecycle-once", session=session)
    captured = {}

    def capture(response_dict, **_):
        body = response_dict.get("body") or b""
        captured["raw_xml"] = body.decode("utf-8", "replace") if isinstance(body, (bytes, bytearray)) else str(body)

    client.meta.events.register("before-parse.s3.GetBucketLifecycleConfiguration", capture, unique_id="dltrain-g18-capture")
    try:
        resp = client.get_bucket_lifecycle_configuration(Bucket=BUCKET)
        rules, raw_status, meta = normalize_rules(resp.get("Rules", [])), "rules_present", resp.get("ResponseMetadata", {})
    except ClientError as err:
        meta = err.response.get("ResponseMetadata", {})
        if _code(err) != "NoSuchLifecycleConfiguration":
            return {"verdict": "UNKNOWN", "error": safe_err(err), "http": _http_meta(meta), "bucket": BUCKET}
        rules, raw_status = [], "no_lifecycle_configuration"
    out = evaluate_lifecycle(rules)
    raw = captured.get("raw_xml")
    out.update(bucket=BUCKET, raw_status=raw_status, raw_rules=rules, http=_http_meta(meta),
               raw_xml=raw, raw_xml_sha256=common.sha256_bytes(raw.encode()) if raw is not None else None)
    return out


# ----------------------------------------------------------------------------------------------- P0 live probe
def attempt_prefix(reader, claim_ts: str) -> tuple[str, int]:
    base = f"{PREFIX_KEY}probes/{claim_ts}/"
    resp = reader.list_objects_v2(Bucket=BUCKET, Prefix=base, Delimiter="/", MaxKeys=LIST_MAX)
    if resp.get("IsTruncated"):
        raise RuntimeError("attempt listing truncated; refusing to guess the next attempt")
    used = []
    for cp in resp.get("CommonPrefixes", []) or []:
        tail = cp["Prefix"][len(base):].strip("/")
        if tail.startswith("attempt-") and tail[8:].isdigit():
            used.append(int(tail[8:]))
    n = max(used, default=0) + 1
    return f"{base}attempt-{n}/", n


class Journal:
    """Write-ahead ledger of every PUT: an intent line before the send, an outcome line after (local JSONL, fsynced)."""

    def __init__(self, path=None):
        self.entries, self.path, self.lock = [], Path(path) if path else None, threading.Lock()
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            os.close(os.open(str(self.path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644))

    def _append(self, rec):
        with self.lock:
            self.entries.append(rec)
            if self.path:
                common.assert_no_secrets_obj(rec)
                line = json.dumps(rec, sort_keys=True) + "\n"
                with open(self.path, "a") as f:
                    f.write(line)
                    f.flush()
                    os.fsync(f.fileno())

    def put(self, step, client, key, body, **cond):
        rec = {"event": "intent", "step": step, "key": key, "cond": sorted(cond), "body_sha256": common.sha256_bytes(body),
               "bytes": len(body), "t_send_us": common.utc_us()}
        self._append(rec)
        out = {"event": "outcome", "step": step, "key": key, "body_sha256": rec["body_sha256"], "bytes": len(body),
               "t_send_us": rec["t_send_us"]}
        try:
            resp = client.put_object(Bucket=BUCKET, Key=key, Body=body, ContentType="application/json", ChecksumAlgorithm="SHA256",
                                     ChecksumSHA256=_checksum_b64(body), **cond)
            vid = resp.get("VersionId")
            out.update(outcome="written", status=resp.get("ResponseMetadata", {}).get("HTTPStatusCode", 200), version_id=vid, etag=resp.get("ETag"),
                       version_ok=bool(vid) and vid != "null")
        except ClientError as err:
            out.update(outcome="refused", status=_status(err), code=_code(err))
        except S3Refused as err:
            out.update(outcome="not_sent", status=None, error=safe_err(err))
        except BaseException as err:  # noqa: BLE001 (unknown outcome: an object may exist)
            out.update(outcome="unknown", status=None, error=safe_err(err))
        out["t_done_us"] = common.utc_us()
        self._append(out)
        return out

    def written(self):
        outs = [e for e in self.entries if e["event"] == "outcome"]
        ok = [e for e in outs if e["outcome"] == "written"]
        unknown = [e for e in outs if e["outcome"] == "unknown"]
        return {"keys": len({e["key"] for e in ok}), "versions": len(ok), "bytes": sum(e["bytes"] for e in ok),
                "unknown_outcomes": len(unknown), "possible_unrecorded_bytes": sum(e["bytes"] for e in unknown)}


def _head_flags(reader, key, vid) -> dict:
    h = reader.head_object(Bucket=BUCKET, Key=key, VersionId=vid)
    hdrs = h.get("ResponseMetadata", {}).get("HTTPHeaders", {})
    flags = {x: hdrs.get(x) for x in ("x-amz-storage-class", "x-amz-expiration", "x-amz-replication-status")}
    return {"key": key, "version_id": vid, **flags, "ok": not any(flags.values())}


def run_probe(claim_ts: str, attempt: str, n: int, clients: dict, journal: Journal) -> dict:
    """PLAN §5 P0 step 5. clients = {'writer', 'reader', 'racers': [8 write clients]}, all built before the first PUT."""
    res = {"schema": "jope.dl-s3-probe/v1", "started_utc_us": common.utc_us(), "attempt_prefix": f"s3://{BUCKET}/{attempt}",
           "attempt": n, "steps": {}, "current_step": None}
    try:
        _probe_steps(res, attempt, claim_ts, clients, journal)
    except BaseException as err:  # noqa: BLE001 (recorded; the caller still writes the receipt)
        res["exception"] = safe_err(err) | {"step": res.get("current_step")}
    res["finished_utc_us"] = common.utc_us()
    res["objects_written"] = journal.written()
    res["writes"] = [e for e in journal.entries if e["event"] == "outcome"]
    steps = res["steps"]
    res["verdict"] = "PASS" if "exception" not in res and all(steps.get(s, {}).get("pass") is True for s in REQUIRED_STEPS) else "FAIL"
    return res


def _probe_steps(res, attempt, claim_ts, clients, journal):
    writer, reader, racers = clients["writer"], clients["reader"], clients["racers"]
    nonce = os.urandom(8).hex()
    key_a, key_p, key_q = attempt + "A.json", attempt + "P.json", attempt + "never-written.json"
    # (1) create-only A; an identical repeat must be 412 and leave A's version unchanged
    res["current_step"] = "1_create_only"
    s1 = res["steps"]["1_create_only"] = {}
    body_a = common.canonical({"probe": "A", "claim_ts": claim_ts, "nonce": nonce})
    a1 = s1["put"] = journal.put("1", writer, key_a, body_a, IfNoneMatch="*")
    a2 = s1["repeat"] = journal.put("1-repeat", writer, key_a, body_a, IfNoneMatch="*")
    head_a = reader.head_object(Bucket=BUCKET, Key=key_a)
    s1["head_version_unchanged"] = head_a.get("VersionId") == a1.get("version_id")
    s1["pass"] = (a1.get("status") == 200 and a1.get("version_ok") is True and a2.get("status") == 412 and s1["head_version_unchanged"])
    # (2) pointer P: create (v1); IfMatch(v1) -> v2; IfMatch(v1) again -> 412
    res["current_step"] = "2_pointer_cas"
    s2 = res["steps"]["2_pointer_cas"] = {}
    b1, b2 = common.canonical({"probe": "P", "v": 1, "nonce": nonce}), common.canonical({"probe": "P", "v": 2, "nonce": nonce})
    p1 = s2["v1"] = journal.put("2-v1", writer, key_p, b1, IfNoneMatch="*")
    if not p1.get("etag"):
        raise RuntimeError("pointer v1 not written; stopping before CAS steps")
    p2 = s2["v2"] = journal.put("2-v2", writer, key_p, b2, IfMatch=p1["etag"])
    stale = s2["stale"] = journal.put("2-stale", writer, key_p, common.canonical({"probe": "P", "v": "stale", "nonce": nonce}),
                                      IfMatch=p1["etag"])
    s2["pass"] = p1.get("status") == 200 and p2.get("status") == 200 and stale.get("status") == 412
    if not p2.get("etag"):
        raise RuntimeError("pointer v2 not written; stopping before the race")
    # (3) race: 8 pre-built, pre-warmed write clients PUT distinct bodies with IfMatch=ETag(v2); exactly one may win
    res["current_step"] = "3_race"
    s3 = res["steps"]["3_race"] = {}
    race_bodies = [common.canonical({"probe": "P", "v": f"race-{i}", "nonce": nonce}) for i in range(8)]
    for c in racers:  # pre-warm: TLS outside the race window (an allowed HEAD inside PREFIX)
        c.head_object(Bucket=BUCKET, Key=key_p)
    outcomes, barrier = [None] * 8, threading.Barrier(8, timeout=120)

    def racer(i):
        rec = {"i": i}
        try:
            barrier.wait()
            rec["t_release_us"] = common.utc_us()
            rec.update(journal.put(f"3-race-{i}", racers[i], key_p, race_bodies[i], IfMatch=p2["etag"]))
        except BaseException as err:  # noqa: BLE001
            rec.update(outcome="not_sent", status=None, error=safe_err(err))
        finally:
            outcomes[i] = rec

    threads = [threading.Thread(target=racer, args=(i,), daemon=True) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=180)
    snap = [dict(o) if o else {"outcome": "unknown", "status": None, "thread_alive": True} for o in outcomes]
    winners = [o for o in snap if o.get("status") == 200]
    losers = [o for o in snap if o.get("status") in (409, 412)]
    unknown = [o for o in snap if o.get("status") is None]
    cur_body = reader.get_object(Bucket=BUCKET, Key=key_p)["Body"].read()
    sends = [o["t_send_us"] for o in snap if o.get("t_send_us")]
    win_done = winners[0].get("t_done_us") if winners else None
    s3.update(outcomes=snap, winners=len(winners), losers_409_or_412=len(losers), unknown=len(unknown),
              conflicts_409=sum(1 for o in snap if o.get("status") == 409),
              current_equals_winner=bool(winners) and cur_body == race_bodies[winners[0]["i"]],
              send_spread_us=(max(sends) - min(sends)) if sends else None,
              overlap=bool(sends) and win_done is not None and max(sends) < win_done)
    s3["race_concurrency"] = "overlapping" if s3["overlap"] else "unproven"
    s3["pass"] = len(snap) == 8 and not unknown and len(winners) == 1 and len(losers) == 7 and s3["current_equals_winner"]
    # (4) IfMatch on a never-written key: status recorded; nothing may be written
    res["current_step"] = "4_ifmatch_missing_key"
    s4 = res["steps"]["4_ifmatch_missing_key"] = {}
    q = s4["put"] = journal.put("4", writer, key_q, common.canonical({"probe": "Q", "nonce": nonce}), IfMatch='"' + "0" * 32 + '"')
    try:
        reader.head_object(Bucket=BUCKET, Key=key_q)
        q_absent = False
    except ClientError as err:
        q_absent = _status(err) == 404
    s4.update(absent_after=q_absent, recorded=True)
    s4["pass"] = q.get("outcome") == "refused" and q_absent
    # (5) HEAD flags and (6) exact-version GETs of every success
    res["current_step"] = "5_head_flags"
    successes = [(key_a, a1.get("version_id"), body_a), (key_p, p1.get("version_id"), b1), (key_p, p2.get("version_id"), b2)]
    successes += [(key_p, w.get("version_id"), race_bodies[w["i"]]) for w in winners]
    heads = [_head_flags(reader, k, v) for k, v, _ in successes if v]
    res["steps"]["5_head_flags"] = {"heads": heads, "pass": len(heads) == len(successes) and all(h["ok"] for h in heads)}
    res["current_step"] = "6_exact_gets"
    gets = []
    for k, v, b in successes:
        try:
            gets.append(verify_readback(reader, k, v, b) | {"match": True})
        except BaseException as err:  # noqa: BLE001
            gets.append({"key": k, "version_id": v, "match": False, "error": safe_err(err)})
    res["steps"]["6_exact_gets"] = {"gets": gets, "pass": bool(gets) and all(g["match"] for g in gets)}
    res["current_step"] = None


def caller_arn(session=None) -> str:
    return sts_client(session).get_caller_identity()["Arn"]


def write_receipt_or_fallback(path: Path, doc: dict) -> dict:
    try:
        return common.write_receipt(path, doc) | {"fallback": False}
    except BaseException as err:  # noqa: BLE001
        minimal = {k: doc.get(k) for k in ("schema", "attempt_prefix", "attempt", "verdict", "objects_written")}
        minimal["writes"] = [{k: w.get(k) for k in ("step", "key", "outcome", "status", "version_id")} for w in doc.get("writes", []) or []]
        minimal["receipt_error"] = safe_err(err)
        fb = Path(path).with_name(Path(path).name + f".fallback-{common.utc_us()}.json")
        res = common.write_receipt(fb, minimal)
        print(json.dumps({"fallback_receipt": res["path"], "objects_written": doc.get("objects_written")}), file=sys.stderr)
        return res | {"fallback": True}


def _main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="dltrain.s3io")
    sub = ap.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("probe")
    pr.add_argument("--claim-ts", required=True)
    pr.add_argument("--receipt-dir", required=True, help="receipt is <dir>/s3-probe.attempt-<n>.json; journal alongside")
    lc = sub.add_parser("lifecycle")
    lc.add_argument("--receipt", required=True)
    wh = sub.add_parser("whoami")
    wh.add_argument("--receipt", required=True)
    args = ap.parse_args(argv)
    meta = {"argv": common.redact_argv(sys.argv), "boto3": boto3.__version__, "botocore": botocore.__version__}
    doc, res = {}, {}
    try:
        if args.cmd == "probe":
            session = boto3.session.Session(profile_name=PROFILE, region_name=REGION)
            clients = {"writer": make_client(writes=True, session=session), "reader": make_client(session=session),
                       "racers": [make_client(writes=True, session=session) for _ in range(8)]}
            attempt, n = attempt_prefix(clients["reader"], args.claim_ts)
            receipt = Path(args.receipt_dir) / f"s3-probe.attempt-{n}.json"
            journal_path = Path(args.receipt_dir) / f"s3-probe.attempt-{n}.journal.jsonl"
            if receipt.exists() or journal_path.exists():
                raise S3Refused(f"receipt or journal for attempt {n} already exists; refusing before any write")
            journal = Journal(journal_path)
            doc = run_probe(args.claim_ts, attempt, n, clients, journal) | {"meta": meta | {"utc_us": common.utc_us()},
                                                                             "journal": str(journal_path)}
            res = write_receipt_or_fallback(receipt, doc)
        elif args.cmd == "lifecycle":
            receipt = Path(args.receipt)
            if receipt.exists():
                raise S3Refused("G18 receipt already exists; refusing a second lifecycle call")
            doc = {"schema": "jope.dl-g18-lifecycle/v1"} | lifecycle_readback() | {"meta": meta | {"utc_us": common.utc_us()}}
            res = write_receipt_or_fallback(receipt, doc)
        else:
            receipt = Path(args.receipt)
            if receipt.exists():
                raise S3Refused("caller-identity receipt already exists")
            doc = {"schema": "jope.dl-caller-identity/v1", "arn": caller_arn(), "verdict": "PASS", "meta": meta | {"utc_us": common.utc_us()}}
            res = write_receipt_or_fallback(receipt, doc)
    except BaseException as err:  # noqa: BLE001
        print(json.dumps({"error": safe_err(err)}), file=sys.stderr)
        return 3
    print(json.dumps({k: res.get(k) for k in ("path", "sha256", "bytes", "fallback")} | {"verdict": doc.get("verdict"),
                                                                                       "objects_written": doc.get("objects_written")}))
    return 0 if doc.get("verdict") == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(_main())
