"""H474 pre-registration checks (static; no model, no data). Run: pytest research/flywheel_selfplay/h474_pgap_dose_accum."""
import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
H474 = HERE / "h474_trainfold_cv.py"
H473 = HERE.parent / "h473_pair_cost_sensitive" / "h473_trainfold_cv.py"
WAITER = HERE / "waiter.sh"
METRICS = HERE / "metrics.json"


def _src(p):
    return p.read_text()


def _body(src):
    """Code after the module docstring."""
    return src.split('"""', 2)[2]


def _block(src, start, end):
    i = src.index(start)
    return src[i:src.index(end, i)]


def test_metrics_json_and_code_sha():
    m = json.loads(_src(METRICS))
    assert m["wave"] == "H474" and m["control"] == "pairnf_z40"
    assert m["arms"] == ["pairnf_z40", "pgap_z40", "pgap2_z40", "pgapa2_z40"]
    assert m["dope_pin"].startswith("c51a56d0")
    assert m["sealed_or_holdout_reads"] == 0
    assert hashlib.sha256(H474.read_bytes()).hexdigest() == m["script_sha256"]


def test_new_arms_registered():
    s = _src(H474)
    assert '"pairnf_pgap2": _pn | dict(pgap=2.0, rng_arm="pairnf")' in s
    assert '"pairnf_pgapa2": _pn | dict(pgap=1.0, accum=2, rng_arm="pairnf")' in s
    assert '"pgap2_z40": _d | dict(base="pairnf_pgap2", subs=1, zmin=0.40, ens=False, agree=False)' in s
    assert '"pgapa2_z40": _d | dict(base="pairnf_pgapa2", subs=1, zmin=0.40, ens=False, agree=False)' in s
    assert 'default="pairnf_z40,pgap_z40,pgap2_z40,pgapa2_z40"' in s


def test_control_and_pgap_paths_unchanged_vs_h473():
    a, b = _src(H473), _src(H474)
    for start, end in [("    BASE = dict(", "\n"), ("    _pn = BASE", "\n"),
                       ('"pairnf_pgap": _pn', ","), ("    def train_candidate(", "    SUBCACHE = {}"),
                       ("    def build_candidate(", "    def elo("), ("    def utility(", "\n    def ")]:
        if start not in a:
            continue
        assert _block(a, start, end) == _block(b, start, end), start


def test_diff_vs_h473_is_limited():
    a = _body(_src(H473)).splitlines()
    b = _body(_src(H474)).splitlines()
    extra = [x for x in b if x not in a]
    missing = [x for x in a if x not in b]
    allowed = re.compile(r'H474_ROOT|--arms|# H474|pairnf_pgap2|pairnf_pgapa2|pgap2_z40|pgapa2_z40|pairnf_pm25|pm25_z40|"wave": "H474"')
    assert all(allowed.search(x) for x in extra), [x for x in extra if not allowed.search(x)]
    assert all(re.search(r'H473_ROOT|--arms|pairnf_pm25|pm25_z40|"wave": "H473"', x) for x in missing), missing


def test_no_forbidden_hosts_or_sealed_paths():
    for p in (H474, WAITER):
        t = _src(p).lower()
        for host in ("xbabe" + "0", "xbabe" + "3"):  # spelled split so repo-wide host greps stay clean
            assert host not in t
    t = _src(H474).lower()
    assert not re.search(r"sealed|holdout|meta.test", t)


def test_waiter_targets_h474_with_memory_gates():
    w = _src(WAITER)
    assert "cd ~/jope-data/hwave-h474" in w
    assert "sha256sum -c h474_trainfold_cv.py.sha256" in w
    assert 'grep -q "h474_trainfold_cv.py --seeds $2 "' in w
    assert "MEMGATE=${MEMGATE:-30}" in w and "${MAXALIVE:-2}" in w
    assert "h473" not in w
    assert "if m < 15.0:" in _src(H474)
