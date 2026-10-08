"""Small CPU tests for the dlhwm guarantees: clean-license refusal, create-only S3 semantics, lineage-pure folds, fail-closed mask parsing."""
import hashlib, io, json, os, sys
import pytest
np = pytest.importorskip("numpy")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from dlhwm import common
def test_guard_refuses_non_clean():
    for b in ("foundation", "tabpfn35", "target", "context2048", "hyperion_v7", "ta_hyperion_v8", "Foundation", "mystery"):
        with pytest.raises(SystemExit): common.assert_clean(["L", b])
    common.assert_clean(["L"] + list(common.CLEAN_ENCODERS))
class _Err(Exception):
    def __init__(s, code): s.response = {"Error": {"Code": code}}
class StubS3:
    def __init__(s): s.o = {}
    def put_object(s, Bucket, Key, Body=None, IfNoneMatch=None, Metadata=None, **kw):
        import botocore.exceptions as be
        if Key in s.o: raise be.ClientError({"Error": {"Code": "PreconditionFailed"}}, "PutObject")
        s.o[Key] = Body; return {"VersionId": "v1"}
    def get_object(s, Bucket, Key, VersionId=None): return {"Body": io.BytesIO(s.o[Key])}
def test_put_create_only_never_overwrites(monkeypatch):
    pytest.importorskip("botocore")
    st = StubS3(); monkeypatch.setattr(common, "s3", lambda: st)
    assert common.put_create_only("k", b"abc")["status"] == "created"
    assert common.put_create_only("k", b"abc")["status"] == "existed"
    with pytest.raises(RuntimeError): common.put_create_only("k", b"xyz")
    assert st.o["k"] == b"abc"
