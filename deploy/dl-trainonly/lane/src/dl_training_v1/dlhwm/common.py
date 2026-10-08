"""Shared constants and the create-only S3 publisher (If-None-Match:* + GET-back sha256 verify; never overwrites or deletes)."""
import hashlib, io, json, os, socket, time
BUCKET = "veox-jopedime-learning-use2"; PFX = "experiments/gp-policy-dl/v2-trainonly/"
PROFILE = os.environ.get("DLHWM_AWS_PROFILE", "veox-jepson-prod")
# Clean-license policy (DL_FINAL_DESIGN.md section 1). Anything else is refused at load time.
CLEAN_ENCODERS = ("kumo_tabular_l", "kumo_tabular_s", "mitra_v2", "tabdpt13", "tabicl2", "a0", "dope", "hyperion_v6all")
REFUSED = ("foundation", "tabpfn35", "target", "context2048", "hyperion_v7", "ta_dope_v2", "ta_hyperion_v8", "ta_hyperion_v8braw")
def assert_clean(blocks):
    bad = [b for b in blocks if b in REFUSED or (b not in CLEAN_ENCODERS and b not in ("L",))]
    if bad: raise SystemExit(f"REFUSED non-clean feature blocks: {bad}")
def sha256b(b): return hashlib.sha256(b).hexdigest()
def sha256f(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""): h.update(ch)
    return h.hexdigest()
def md5f(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""): h.update(ch)
    return h.hexdigest()
def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)
_S3 = None
def s3():
    global _S3
    if _S3 is None:
        import boto3, botocore
        _S3 = boto3.Session(profile_name=PROFILE).client("s3", config=botocore.config.Config(max_pool_connections=48, retries={"max_attempts": 10, "mode": "adaptive"}))
    return _S3
def get(key, vid=None):
    kw = dict(Bucket=BUCKET, Key=key)
    if vid: kw["VersionId"] = vid
    return s3().get_object(**kw)["Body"].read()
def put_create_only(key, body):
    """Create-only upload. Returns {key, bytes, sha256, md5, version_id, status}. An existing identical object counts as verified;
    an existing different object is a conflict and is never overwritten."""
    import botocore
    sha = sha256b(body); md5 = hashlib.md5(body).hexdigest()
    try:
        r = s3().put_object(Bucket=BUCKET, Key=key, Body=body, IfNoneMatch="*", Metadata={"sha256": sha})
        vid = r.get("VersionId"); status = "created"
    except botocore.exceptions.ClientError as e:
        code = e.response.get("Error", {}).get("Code")
        if code not in ("PreconditionFailed", "ConditionalRequestConflict", "412"): raise
        vid = None; status = "existed"
    back = get(key, vid)
    if sha256b(back) != sha:
        raise RuntimeError(f"S3 conflict/readback mismatch at {key}: {status}")
    return {"key": key, "bytes": len(body), "sha256": sha, "md5": md5, "version_id": vid, "status": status, "verified_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
def host(): return socket.gethostname()
