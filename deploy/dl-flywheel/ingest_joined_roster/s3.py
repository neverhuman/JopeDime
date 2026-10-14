"""Bounded S3 LIST/HEAD/GET adapter; no remote mutation interface."""

import hashlib
import math
import threading
import time


class ReadOnlyS3:
    def __init__(self, client, *, bucket, prefix, deadline, max_requests, max_bytes):
        if (
            not bucket
            or not prefix.endswith("/")
            or type(max_requests) is not int
            or max_requests <= 0
            or type(max_bytes) is not int
            or max_bytes <= 0
        ):
            raise ValueError("bounded bucket/prefix required")
        if type(deadline) not in (int, float) or not math.isfinite(deadline):
            raise ValueError("finite monotonic deadline required")
        self._client = client
        self.bucket = bucket
        self.prefix = prefix
        self.deadline = deadline
        self.max_requests = max_requests
        self.max_bytes = max_bytes
        self.requests = []
        self.read_bytes = 0
        self.verified_read_bytes = 0
        self._lock = threading.Lock()

    def _request(self, operation, key=None):
        if operation not in {"ListObjectsV2", "HeadObject", "GetObject"}:
            raise ValueError("read-only operation required")
        if key is not None and not key.startswith(self.prefix):
            raise ValueError("S3 namespace escape")
        with self._lock:
            if time.monotonic() >= self.deadline:
                raise TimeoutError("S3 capture deadline")
            if len(self.requests) >= self.max_requests:
                raise ValueError("S3 request cap")
            self.requests.append({"operation": operation, "bucket": self.bucket, "key": key})

    def list_prefixes(self):
        self._request("ListObjectsV2")
        response = self._client.list_objects_v2(
            Bucket=self.bucket, Prefix=self.prefix, Delimiter="/", MaxKeys=1000
        )
        if response.get("IsTruncated") is not False:
            raise ValueError("incomplete prefix listing")
        values = sorted({item["Prefix"] for item in response.get("CommonPrefixes", [])})
        if any(not value.startswith(self.prefix) or not value.endswith("/") for value in values):
            raise ValueError("prefix namespace escape")
        return values

    def list_objects(self):
        token = None
        seen_tokens = set()
        objects = {}
        pages = []
        while True:
            self._request("ListObjectsV2")
            args = {"Bucket": self.bucket, "Prefix": self.prefix, "MaxKeys": 1000}
            if token is not None:
                args["ContinuationToken"] = token
            response = self._client.list_objects_v2(**args)
            page = []
            for obj in response.get("Contents", []):
                key = obj["Key"]
                if not isinstance(key, str) or not key.startswith(self.prefix):
                    raise ValueError("listing namespace escape")
                if type(obj["Size"]) is not int or obj["Size"] < 0:
                    raise ValueError("listing size identity")
                row = {
                    "key": key,
                    "bytes": obj["Size"],
                    "etag": obj.get("ETag"),
                    "last_modified": obj["LastModified"].isoformat(),
                    "storage_class": obj.get("StorageClass"),
                }
                if key in objects and objects[key] != row:
                    raise ValueError("object changed across listing pages")
                objects[key] = row
                page.append(row)
            pages.append({"objects": page, "is_truncated": response.get("IsTruncated")})
            if response.get("IsTruncated") is False:
                break
            token = response.get("NextContinuationToken")
            if not isinstance(token, str) or not token or token in seen_tokens:
                raise ValueError("incomplete or repeated listing cursor")
            seen_tokens.add(token)
        return {
            "bucket": self.bucket,
            "prefix": self.prefix,
            "complete": True,
            "objects": [objects[key] for key in sorted(objects)],
            "pages": pages,
            "snapshot_scope": "Sequential LIST pages; not an atomic cross-object snapshot",
        }

    def read_current(self, key, *, limit):
        self._request("HeadObject", key)
        head = self._client.head_object(Bucket=self.bucket, Key=key)
        version = head.get("VersionId")
        length = head.get("ContentLength")
        if not isinstance(version, str) or version in ("", "null"):
            raise ValueError("immutable S3 VersionId required")
        if type(length) is not int or not 0 < length <= limit:
            raise ValueError("S3 object size cap")
        return self.read_ref(
            {
                "bucket": self.bucket,
                "key": key,
                "version_id": version,
                "bytes": length,
                "sha256": head.get("Metadata", {}).get("sha256"),
            },
            limit=limit,
        )

    def read_ref(self, ref, *, limit):
        if ref.get("bucket") != self.bucket or not ref.get("key", "").startswith(self.prefix):
            raise ValueError("S3 namespace escape")
        key, version, length = ref["key"], ref.get("version_id"), ref.get("bytes")
        if not isinstance(version, str) or version in ("", "null"):
            raise ValueError("immutable S3 VersionId required")
        if type(length) is not int or not 0 < length <= limit:
            raise ValueError("S3 object size cap")
        with self._lock:
            if self.read_bytes + length > self.max_bytes:
                raise ValueError("S3 byte cap")
            self.read_bytes += length
        self._request("GetObject", key)
        response = self._client.get_object(Bucket=self.bucket, Key=key, VersionId=version)
        if response.get("VersionId") != version or response.get("ContentLength") != length:
            response["Body"].close()
            raise ValueError("S3 exact version/length mismatch")
        with response["Body"] as stream:
            body = stream.read(length + 1)
        if time.monotonic() >= self.deadline:
            raise TimeoutError("S3 capture deadline")
        if len(body) != length:
            raise ValueError("S3 body length mismatch")
        digest = hashlib.sha256(body).hexdigest()
        expected = ref.get("sha256")
        if expected is not None and expected != digest:
            raise ValueError("S3 metadata SHA mismatch")
        with self._lock:
            self.verified_read_bytes += len(body)
        return body, {
            "bucket": self.bucket,
            "key": key,
            "version_id": version,
            "sha256": digest,
            "bytes": length,
        }

    def receipt(self):
        return {
            "schema": "jope.ingest-read-only-s3/v1",
            "requests": list(self.requests),
            "request_count": len(self.requests),
            "reserved_get_bytes": self.read_bytes,
            "verified_get_bytes": self.verified_read_bytes,
            "allowed_operations": ["ListObjectsV2", "HeadObject", "GetObject"],
            "s3_writes": 0,
        }
