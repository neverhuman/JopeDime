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
def test_group_folds_lineage_pure():
    from dlhwm.lgb_baseline import group_folds
    g = np.repeat(np.arange(200), np.random.default_rng(0).integers(1, 9, 200)); f = group_folds(g, 5, 10100)
    for u in np.unique(g): assert len(set(f[g == u])) == 1
    assert set(f) == set(range(5))
def test_mask_parser_fails_closed(tmp_path):
    pytest.importorskip("torch")
    from dlhwm.train import parse_mask
    blocks = ["kumo_tabular_l", "mitra_v2"]
    p = tmp_path / "m.json"
    p.write_text(json.dumps({"encoders": {"kumo_tabular_l": {"block_weight": 0.5, "mode": "pcs", "n_pcs": 32, "kept": [0, 3, 31]}, "mitra_v2": {"block_weight": 0}}}))
    m = parse_mask(str(p), blocks); assert m["kumo_tabular_l"]["kept"] == [0, 3, 31] and m["mitra_v2"]["weight"] == 0
    p.write_text(json.dumps({"encoders": {"kumo_tabular_l": {"mode": "pcs", "n_pcs": 32, "kept": [1]}}}))
    assert parse_mask(str(p), blocks)["_unlisted"] == {"policy": "keep", "encoders": ["mitra_v2"]}
    assert parse_mask(str(p), blocks, unlisted_policy="drop")["mitra_v2"]["weight"] == 0.0
    for bad in ({"encoders": {"foundation": {"kept": [1]}}}, {"encoders": {"mitra_v2": {"mode": "magic", "kept": [1]}}},
                {"encoders": {"mitra_v2": {"mode": "pcs", "n_pcs": 32, "kept": [32]}}}, {"encoders": {"mitra_v2": {"kept": [1], "surprise": 1}}}, {"nothing": {}}, {"encoders": {"mitra_v2": {"kept": [1]}}, "blocks": {}}):
        p.write_text(json.dumps(bad))
        with pytest.raises(SystemExit): parse_mask(str(p), blocks)

def test_anchor_refuses_misaligned_folds(tmp_path):
    pytest.importorskip("torch")
    from dlhwm.train import load_anchor
    n = 50; FO = np.array([np.arange(n) % 5]); Y = {t: np.zeros(n) for t in ("fitness", "gap", "beat10")}
    for t in Y:
        te = np.where(FO[0] == 0)[0]; ti = np.where(FO[0] != 0)[0]
        np.savez(tmp_path / f"{t}_r0_f0.npz", te=te, ti=ti, P=np.zeros((1, len(te))), OOF=np.zeros((1, len(ti))), bundle_sha256="abc")
    AN, prov = load_anchor(str(tmp_path), 0, 0, n, FO, Y, "abc"); assert np.isfinite(AN).all()
    with pytest.raises(SystemExit): load_anchor(str(tmp_path), 0, 0, n, FO, Y, "other-bundle")
    with pytest.raises(SystemExit): load_anchor(str(tmp_path), 0, 0, n, np.array([(np.arange(n) + 1) % 5]), Y, "abc")
