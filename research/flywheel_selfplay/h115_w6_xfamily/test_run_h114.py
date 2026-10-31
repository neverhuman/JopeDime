"""H114 structural tests on synthetic fixtures (no data payloads, models or native calls)."""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_h114 as h  # noqa: E402

H106 = h.load_h106()
H81 = H106.load_h81()
S = lambda t: hashlib.sha256(t.encode()).hexdigest()


def test_template_masks_numeric_literals_only():
    assert h.template("(add x3 1.5e-2 (mul x0 -7))") == h.template("(add x3 4 (mul x0 2.25))")
    assert h.template("(add x3 1)") != h.template("(sub x3 1)")


class FakeProp:
    def __init__(self):
        self.idx = {f"c{i}": i for i in range(60)}
        self.fam = {f"c{i}": f"f{i // 2}" for i in range(60)}
        self.roster = [f"c{i}" for i in range(10)]


def test_panel_is_one_context_per_non_roster_family_and_skips_dry_run_contexts():
    prop = FakeProp()
    p = h.panel(prop, n=12)
    fams = [prop.fam[c] for c in p]
    assert len(p) == 12 and len(set(fams)) == 12 and not set(fams) & {prop.fam[c] for c in prop.roster}
    assert p == h.panel(prop, n=12)
    pool = [c for c in prop.idx if prop.fam[c] not in {prop.fam[c] for c in prop.roster}]
    for salt in h.EXCLUDED_DRY_RUN_SALTS:
        dry = sorted(pool, key=lambda c: hashlib.sha256(f"{salt}|{c}".encode()).hexdigest())[:3]
        assert not set(dry) & set(h.panel(prop, n=25))


def test_context_shuffle_is_a_deterministic_permutation():
    cs = [f"c{i}" for i in range(20)]
    p = h.context_permutation(cs)
    assert sorted(p) == sorted(p.values()) == sorted(cs) and p == h.context_permutation(list(reversed(cs)))
    assert sum(a != b for a, b in p.items()) > 10


def test_picks_respect_the_unseen_subset_and_random_is_seeded():
    rng = np.random.default_rng(0)
    ids = [S(str(i)) for i in range(16)]
    scores = {a: rng.normal(size=16) for a in h.ARMS if a != "random4"}
    sub = [1, 3, 5, 7, 9, 11]
    p = h.picks_for(scores, ids, H81.ranked, 9, sub)
    assert all(set(v) <= set(sub) and len(set(v)) == 4 for v in p.values())
    assert p == h.picks_for(scores, ids, H81.ranked, 9, sub)
    assert h.picks_for(scores, ids, H81.ranked, 9)["X_SCORER"] == H81.ranked(scores["X_SCORER"], ids)[:4]


def rows(dx=0.0, unseen_dx=0.0, n=38, shuf_dx=0.0, unseen_n=None):
    out = []
    for i in range(n):
        for pname in ("all", "unseen"):
            if pname == "unseen" and i >= (n if unseen_n is None else unseen_n):
                continue
            for a in h.ARMS:
                v = 1.0 + 0.01 * i
                v += (dx if pname == "all" else unseen_dx) if a == "X_SCORER" else 0.0
                v += shuf_dx if a == "X_SHUF" else 0.0
                out.append(dict(tag="H114", context=f"c{i}", family=f"f{i}", arm=a, panel=pname, status="completed", best_of_four=v))
    return out


PANEL = [f"c{i}" for i in range(38)]


def test_advance_requires_both_primaries_placebo_and_unseen_template_check():
    assert h.analyze(H81, rows(0.03, 0.01), PANEL)["verdict"] == "ADVANCE"
    assert h.analyze(H81, rows(0.03, -0.01), PANEL)["verdict"] == "NO_ADVANCE"  # benefit disappears on new templates
    assert h.analyze(H81, rows(0.03, 0.01, shuf_dx=0.05), PANEL)["verdict"] == "NO_ADVANCE"  # shuffle matches
    assert h.analyze(H81, rows(0.03, 0.01, unseen_n=5), PANEL)["verdict"].startswith("WITHHELD")  # sparse coverage
    assert h.analyze(H81, rows(0.0), PANEL)["verdict"] == "NO_ADVANCE"
    short = [r for r in rows(0.03, 0.01) if r["context"] != "c0"]
    assert h.analyze(H81, short, PANEL)["verdict"].startswith("INCOMPLETE")
    with pytest.raises(SystemExit):
        h.analyze(H81, [dict(r, tag="DRY_RUN") for r in rows()], PANEL)


def test_h106_loader_refuses_changed_bytes(tmp_path, monkeypatch):
    fake = tmp_path / "run_h106.py"; fake.write_text("# not the merged runner\n")
    monkeypatch.setattr(h, "H106_PATH", fake)
    with pytest.raises(SystemExit):
        h.load_h106()
