"""A bounded S3 reader with a before-send guard against remote mutation."""

from __future__ import annotations

import math
import time
from collections import Counter
from datetime import datetime, timezone

from .retention import identity, integer, need

OPERATIONS = {"list_object_versions": "ListObjectVersions", "get_object": "GetObject"}


class ReadOnlyS3:
    def __init__(
        self,
        client,
        bucket,
        prefix,
        *,
        max_requests=1500,
        max_metadata_bytes=32 * 1024**2,
        max_object_bytes=64 * 1024**2,
        seconds=900,
    ):
        need(
            type(bucket) is str
            and bucket
            and type(prefix) is str
            and prefix.endswith("/"),
            "explicit bucket/prefix",
        )
        self.client, self.bucket, self.prefix = client, bucket, prefix
        self.max_requests = integer(max_requests, "request cap")
        self.max_metadata_bytes = integer(max_metadata_bytes, "metadata byte cap")
        self.max_object_bytes = integer(max_object_bytes, "object byte cap")
        need(
            type(seconds) in (int, float) and math.isfinite(seconds) and seconds > 0,
            "finite work deadline",
        )
        self.end = time.monotonic() + seconds
        self.requests, self.metadata_bytes, self.object_bytes = [], 0, 0
        self.active = None
        client.meta.events.register("before-send.s3", self.before_send)
        client.meta.events.register("after-call.s3", self.after_call)

    def check(self):
        if time.monotonic() >= self.end:
            raise TimeoutError("original S3 read window expired")

    def before_send(self, request, event_name, **kwargs):
        self.check()
        op = event_name.rsplit(".", 1)[-1]
        need(
            op in OPERATIONS.values() and op == self.active,
            "remote mutation or unowned request refused before dispatch",
        )
        need(len(self.requests) < self.max_requests, "physical HTTP request cap")
        self.requests.append(
            {"operation": op, "started_at": datetime.now(timezone.utc).isoformat()}
        )

    def after_call(self, http_response, parsed, model, **kwargs):
        event = self.requests[-1]
        need(event["operation"] == model.name, "HTTP receipt joins request")
        event.update(
            {
                "http_status": http_response.status_code,
                "request_id": parsed.get("ResponseMetadata", {}).get("RequestId"),
                "finished_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        if model.name == "ListObjectVersions":
            n = len(http_response.content)
            self.metadata_bytes += n
            event["response_bytes"] = n
            need(self.metadata_bytes <= self.max_metadata_bytes, "metadata byte cap")

    def call(self, operation, **params):
        self.check()
        need(operation in OPERATIONS, "read-only operation allowlist")
        need(params.get("Bucket") == self.bucket, "bucket scope")
        scoped = (
            params.get("Prefix")
            if operation == "list_object_versions"
            else params.get("Key")
        )
        need(type(scoped) is str and scoped.startswith(self.prefix), "key/prefix scope")
        if operation == "get_object":
            need(
                type(params.get("VersionId")) is str
                and params["VersionId"] not in ("", "null"),
                "exact immutable GET version",
            )
        need(self.active is None, "one reader operation at a time")
        self.active = OPERATIONS[operation]
        try:
            return getattr(self.client, operation)(**params)
        finally:
            self.active = None

    def census(self):
        rows, markers, seen, tokens = [], [], set(), set()
        params = {"Bucket": self.bucket, "Prefix": self.prefix, "MaxKeys": 1000}
        while True:
            response = self.call("list_object_versions", **params)
            for item in response.get("Versions", []):
                need(item["Key"].startswith(self.prefix), "listed key outside prefix")
                row = {
                    "bucket": self.bucket,
                    "key": item["Key"],
                    "version_id": item["VersionId"],
                    "bytes": integer(item["Size"], "listed bytes"),
                    "storage_class": item["StorageClass"],
                    "is_latest": item["IsLatest"],
                    "last_modified": item["LastModified"].isoformat(),
                    "etag": item["ETag"],
                    "sha256": None,
                }
                ref = identity(row)
                need(
                    type(row["is_latest"]) is bool and ref not in seen,
                    "duplicate or invalid listed version",
                )
                seen.add(ref)
                rows.append(row)
            for item in response.get("DeleteMarkers", []):
                need(item["Key"].startswith(self.prefix), "marker outside prefix")
                marker = {
                    "bucket": self.bucket,
                    "key": item["Key"],
                    "version_id": item["VersionId"],
                    "is_latest": item["IsLatest"],
                    "last_modified": item["LastModified"].isoformat(),
                }
                need(
                    type(marker["is_latest"]) is bool and identity(marker) not in seen,
                    "duplicate or invalid marker/version identity",
                )
                seen.add(identity(marker))
                markers.append(marker)
            need(
                type(response["IsTruncated"]) is bool, "explicit pagination completion"
            )
            if not response["IsTruncated"]:
                break
            token = (response.get("NextKeyMarker"), response.get("NextVersionIdMarker"))
            need(
                type(token[0]) is str and token[0] and token not in tokens,
                "missing or repeated pagination cursor",
            )
            tokens.add(token)
            params = dict(params, KeyMarker=token[0])
            if token[1] is not None:
                params["VersionIdMarker"] = token[1]
            else:
                params.pop("VersionIdMarker", None)
        self.check()
        return {
            "rows": sorted(rows, key=identity),
            "delete_markers": sorted(markers, key=identity),
            "complete_listing": True,
            "scope": "all versions returned under the exact prefix in this listing interval",
            "point_in_time_snapshot": False,
            "qualified": False,
        }

    def capture(self, row):
        need(
            row["bucket"] == self.bucket and row["storage_class"] == "STANDARD",
            "standard scoped capture",
        )
        expected = integer(row["bytes"], "capture bytes")
        need(
            self.object_bytes + expected <= self.max_object_bytes,
            "body byte admission cap",
        )
        response = self.call(
            "get_object",
            Bucket=self.bucket,
            Key=row["key"],
            VersionId=row["version_id"],
        )
        stream = response["Body"]
        try:
            need(
                response.get("VersionId") == row["version_id"]
                and response.get("ContentLength") == expected,
                "GET identity/size joins listed immutable version",
            )
            parts, count = [], 0
            while True:
                self.check()
                part = stream.read(min(64 * 1024, expected - count + 1))
                if not part:
                    break
                count += len(part)
                self.object_bytes += len(part)
                need(
                    count <= expected and self.object_bytes <= self.max_object_bytes,
                    "body size/read cap",
                )
                parts.append(part)
            need(count == expected, "short object body")
            self.requests[-1]["response_bytes"] = count
            return b"".join(parts)
        finally:
            stream.close()

    def receipt(self):
        return {
            "http_requests": self.requests,
            "physical_request_count": len(self.requests),
            "metadata_response_bytes": self.metadata_bytes,
            "object_body_bytes": self.object_bytes,
            "remote_mutation_calls": 0,
            "operations": dict(Counter(r["operation"] for r in self.requests)),
            "etag_is_sha256": False,
            "qualified": False,
        }
