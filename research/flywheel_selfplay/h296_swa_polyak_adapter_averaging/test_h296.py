"""Data-free H296 controls: the strict primary rule (3/3 seeds; margin over the stored best; above the paired control;
family-bootstrap LCB > 0; not concentrated; ties never win), Elo anchoring, the trajectory schedule and the pins."""

import ast
import math
from pathlib import Path

import numpy as np

import analysis as stat

HERE = Path(__file__).resolve().parent
FAMS = [f"f{i % 19:02}" for i in range(91)]
BASE = [1.0] * 91


def result(name, gain, n_pos=91):
    pc = [1.0 + (gain if i < n_pos else 0.0) for i in range(91)]
    return dict(
        name=name,
        mean=float(np.mean(pc)) + stat.STORED - 1.0,
        per_context=pc,
        selected_programs=[str(i) for i in range(91)],
    )


def test_rule_pass_and_failures():
    rs = []
    for s in ("29601", "29602", "29603"):
        rs += [result(f"SWA_CYCLE-{s}", 0.01), result(f"CONTROL-{s}", 0.0)]
    assert stat.evaluate(rs, BASE, FAMS, [str(i) for i in range(91)])["verdict"] == "DEV_ADVANCE"
    rs[0] = result("SWA_CYCLE-29601", 0.0005)  # below the +0.001 margin
    assert stat.evaluate(rs, BASE, FAMS, [str(i) for i in range(91)])["verdict"] == "NOT_ADVANCED"
    conc = [result("SWA_CYCLE-29601", 0.5, n_pos=3)] + rs[1:]  # gain from 3 contexts only
    ev = stat.evaluate(conc, BASE, FAMS, [str(i) for i in range(91)])
    assert not ev["candidates"]["SWA_CYCLE-29601"]["criteria"]["not_concentrated"] and ev["verdict"] == "NOT_ADVANCED"
    tie = [result(f"SWA_CYCLE-{s}", 0.0) for s in ("29601", "29602", "29603")] + [
        result(f"CONTROL-{s}", 0.0) for s in ("29601", "29602", "29603")
    ]
    assert stat.evaluate(tie, BASE, FAMS, [str(i) for i in range(91)])["verdict"] == "NOT_ADVANCED"


def test_elo_and_constants():
    assert abs(stat.bt_two(0.0, 0.0) - 1500) < 1e-9 and stat.bt_two(1.0, 9.0) > 1500
    assert stat.BOOT_SEED == 296150 and stat.DRAWS == 10000 and stat.MARGIN == 0.001


def test_schedule_and_pins():
    tree = ast.parse((HERE / "h296.py").read_text())
    consts = {
        t.id: n.value for n in tree.body if isinstance(n, ast.Assign) for t in n.targets if isinstance(t, ast.Name)
    }
    assert ast.literal_eval(consts["SEEDS"]) == (29601, 29602, 29603)
    lrs = [0.1 + 0.5 * 0.9 * (1 + math.cos(math.pi * p / 3)) for p in range(4)]
    assert abs(lrs[0] - 1.0) < 1e-12 and abs(lrs[3] - 0.1) < 1e-12
    src = (HERE / "h296.py").read_text()
    assert (
        "N_UPDATES, PER_CYCLE, LR_MIN_FRAC, EMA_DECAY = 16, 4, 0.1, 0.8" in src
        and '"1d6ea900ebed8d2a05c3b3f37576adc15aadd2b53c585bfd5cb71a103e9ff97b"' in src
    )
