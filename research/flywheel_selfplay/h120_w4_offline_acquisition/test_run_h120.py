"""H120 structural tests on synthetic batches (no data payloads, models or native calls)."""

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_h120 as h  # noqa: E402

H112 = h.load_h112()
H81 = H112.load_h106().load_h81()


def sha_hex(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def batch(seed=0, n=16):
    rng = np.random.default_rng(seed)
    return (
        rng.normal(size=n),
        np.abs(rng.normal(size=n)) * 0.3,
        rng.normal(size=(n, 16)),
        [sha_hex(f"p{i}") for i in range(n)],
        list(rng.normal(size=n)),
    )


def test_every_rule_reveals_only_its_own_picks_and_returns_best_of_four():
    for seed in range(15):
        mu, sd, x, ids, u = batch(seed)
        res = h.screen_batch(H112, H81, mu, sd, x, ids, 1.0, u, 7)
        assert set(res) == set(h.RULES) | {"ORACLE4"}
        assert res["ORACLE4"] == max(u) and all(
            res[r] <= res["ORACLE4"] for r in h.RULES
        )
        assert res["S_TOP4"] == max(u[i] for i in H81.ranked(mu, ids)[:4])


def test_one_plus_three_and_ucb_shapes():
    mu, sd, x, ids, u = batch(3)
    seen = []
    p = h.one_plus_three(
        H81,
        mu,
        x,
        ids,
        1.0,
        lambda sel: (seen.append(list(sel)), [u[i] for i in sel])[1],
    )
    assert (
        seen == [H81.ranked(mu, ids)[:1]]
        and len(set(p)) == 4
        and p[0] == H81.ranked(mu, ids)[0]
    )
    q = h.ucb_two_plus_two(
        H112, H81, mu, sd, x, ids, 1.0, lambda sel: [u[i] for i in sel]
    )
    assert q[:2] == H81.ranked(mu + sd, ids)[:2] and len(set(q)) == 4
    assert h.one_plus_three(H81, mu, x, ids, 1.0, lambda sel: [None]) is None


def test_zero_uncertainty_ucb_matches_gp_2plus2():
    mu, _, x, ids, u = batch(5)
    q = h.ucb_two_plus_two(
        H112, H81, mu, np.zeros_like(mu), x, ids, 1.0, lambda sel: [u[i] for i in sel]
    )
    base = H112.select(H81, mu, x, ids, 1.0, 7, lambda sel: [u[i] for i in sel])
    assert q == base["s_2plus2"]


def test_integrity_matches_and_flags_a_mismatch():
    rows = [dict(context="c1", seed=901, status="screened", S_TOP4=1.0, GP_2PLUS2=1.2)]
    pub = [
        dict(
            context="c1", seed=901, arm="s_top4", status="completed", best_of_four=1.0
        ),
        dict(
            context="c1", seed=901, arm="s_2plus2", status="completed", best_of_four=1.2
        ),
    ]
    assert h.integrity(rows, pub)["passed"]
    pub[1]["best_of_four"] = 1.3
    assert (
        not h.integrity(rows, pub)["passed"] and h.integrity([], pub)["passed"] is False
    )


def test_integrity_rejects_one_missing_or_duplicate_h116_batch():
    screen_rows, published = [], []
    for c in range(50):
        for seed in (901, 902, 903):
            screen_rows.append(
                dict(
                    context=f"c{c}",
                    seed=seed,
                    status="screened",
                    S_TOP4=1.0,
                    GP_2PLUS2=1.0,
                )
            )
            published.extend(
                [
                    dict(
                        context=f"c{c}",
                        seed=seed,
                        arm=a,
                        status="completed",
                        best_of_four=1.0,
                    )
                    for a in ("s_top4", "s_2plus2")
                ]
            )
    assert h.integrity(screen_rows, published)["checked"] == 300
    assert h.integrity(screen_rows, published)["passed"] is True
    missing_row = list(screen_rows)
    missing_row[-1] = dict(context="c49", seed=903, status="unlabelled", missing=1)
    assert h.integrity(missing_row, published)["passed"] is False
    assert h.integrity(screen_rows[:-1], published)["passed"] is False
    assert h.integrity(screen_rows + [screen_rows[0]], published)["passed"] is False


def rows(delta=0.0, n=60):
    out = []
    for i in range(n):
        for seed in (801, 802, 803, 901, 902, 903):
            v = 1.0 + 0.01 * i
            out.append(
                dict(
                    context=f"c{i}",
                    family=f"f{i % 30}",
                    seed=seed,
                    status="screened",
                    S_TOP4=v,
                    GP_2PLUS2=v,
                    GP_1PLUS3=v + delta,
                    UCB_2PLUS2=v,
                    random4=v - 0.1,
                    ORACLE4=v + 0.5,
                )
            )
    return out


def test_analyze_candidate_or_close():
    assert (
        h.analyze(H81, rows(0.03))["verdict"].startswith("CANDIDATE")
        and "GP_1PLUS3" in h.analyze(H81, rows(0.03))["verdict"]
    )
    assert h.analyze(H81, rows(0.0))["verdict"].startswith("CLOSE")


def test_analyze_refuses_missing_paired_seed_instead_of_admitting_candidate():
    incomplete = rows(0.03, n=50)
    incomplete[-1] = dict(context="c49", seed=903, status="unlabelled", missing=1)
    result = h.analyze(H81, incomplete)
    assert result["verdict"].startswith(("INCOMPLETE", "REFUSE"))
    assert "primary" in result
    assert h.analyze(H81, rows(0.03, n=50)[:-1])["verdict"].startswith(
        ("INCOMPLETE", "REFUSE")
    )


def test_h112_loader_refuses_changed_bytes(tmp_path, monkeypatch):
    fake = tmp_path / "run_h112.py"
    fake.write_text("# not the merged runner\n")
    monkeypatch.setattr(h, "H112_PATH", fake)
    with pytest.raises(SystemExit):
        h.load_h112()
