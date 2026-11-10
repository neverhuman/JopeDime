"""H335 core: a thin layer over the FROZEN H317 modules, imported unmodified from the H317 source dir and sha-pinned.

Import this module FIRST in every H335 entry point. Before any H317 module is imported, it:
  * pins every H317 source file it may import (sha256 == origin/v1/next 97a1ffce, the H317 closeout);
  * selects the grammar for this process: H335_GROUP=AGG installs the LEARNERS-V1 grammar port (xtrees / rforest /
    oblivious heads; byte-identical to the H317 exploratory run's port) as `gp_grammar_h317`; H335_GROUP=BASE keeps the
    pinned H317 base grammar. A process never mixes the two (the port is installed through sys.modules at import);
  * sets H317_DATA (default /home/ubuntu/jope-data/h335-cl) so every H317 data path resolves inside the H335 data root;
  * installs the H317 read guard v2 (by importing common_h317) and extends it for H335: the H335 evidence dir is
    allowed, the H335 Stage V root is denied before the unlock, and the LOCK receipt is H335's.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "RAYON_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

HERE = Path(__file__).resolve().parent
WAVE = "H335"
SLUG = "h335_aggressive_locked_elo"
EVIDENCE = Path("/home/ubuntu/jope_plan/evidence/H335")
H317_SRC = Path(
    os.environ.get(
        "H335_H317_SRC",
        "/home/ubuntu/jope_plan/evidence/H317/source/research/flywheel_selfplay/h317_island10_multiscale_curriculum",
    )
)
os.environ.setdefault("H317_DATA", "/home/ubuntu/jope-data/h335-cl")

# sha256 of every frozen H317 module (origin/v1/next 97a1ffce = the H317 closeout; also == the xbabe0 stage).
H317_PINS: Dict[str, str] = {
    "common_h317.py": "1242bae25379dc1adb7cbaf253fb7066dcf783b579258f98302b4a3c31c1c3d4",
    "distributed_10gen_island_gp.py": "2de29588d61807db4fc8d7cbafcef07c13989c1bc810a934d94dd20db7497587",
    "gp_engine_h317.py": "43ec25e9c67bcf86d4be83229fee9c37eb738fe521e20edf762a869bcd171b36",
    "gp_grammar_h317.py": "d26fd2385eafc0e5340eb88f7b29f7269298a09531a1240e60e4d83216856a5c",
    "gp_lab.py": "8a87f46cf8e69ac653b789052a752cd59a5d024e1a90a682a08529db54565362",
    "screener_h317.py": "2e3f2fa04c88765a132b82bdcd50f6c97a0fb9077e6ab513281e9109ab59d7c3",
    "stacking_h317.py": "c2f5c753ea423f125d0dea05cb3972cca3be211ed39f4637228dfc1791f567ee",
    "crop_shuffle_synthetic_views.py": "eb7dee6bbfc1924b96baf2dc24d60b7fb704e1ec22c29e93f3795c754449bedb",
    "rapid_hypothesis_lab.py": "31161becd922ecd25784f1f08b16b445c81f8f0a1d3940f06f41e2fd6a805e25",
    "multiscale_axial_bridge.py": "f99726e42cfec84875d15cd5ae202e2674b8638315c2ddb5afb583900d13b248",
    "curriculum_trainer.py": "5453c8e25424bdc64ab55e84bcafbc570fc60aa22e72a059de6523d7585ae9c8",
    "neural_data.py": "a453dc090db7643a10331cd4c3a0986533cfc48826a65bc02d61e6a08e480f75",
    "trainonly_loader.py": "eef47f4f9ce4248edbfd90f0a1b4e2946e99013dda6e5c5c779cef94843fc3c1",
    "stage_v_h317.py": "ec207f1e13b436d64b15d516c3bef357edfb3a28ad4f9d2712fbf70e8091c5d8",
    "dope_freeze_guard.py": "2d8920839cccdc14324eab8eba47251f91e6bbd3ef7f46e66c585e20ecf10ee6",
    "dope_freeze_guard_types.py": "3e377ff8afe1fc7957937920c607ed021122880cf8d945776650d980520ff1b1",
    "dope_freeze_guard_optimizer.py": "94c5a75bbd7e3be5e614ae1363705dd8f0da119034400e2c7acf382d58f850ae",
    "encode_search_views.py": "621e8c3b6aab8591b225e7d64b2b2d430a24cae439a5b3d9940b1cdb6867a98f",
}
# The LEARNERS-V1 grammar port (H317 exploratory run, explore config grammar_port_sha256) and the superset worker
# (parity-verified vs the pinned bootstrap.s1main e3be9386... on 2,010 program x context pairs).
GRAMMAR_PORT = HERE / "gp_grammar_h317_learners_v1.py"
GRAMMAR_PORT_SHA256 = "3d6ec975a805b83d5b6dc693fa0d54704d6748bee04c6978635daa9ae3cd76dd"
SUPERSET_SHA256 = "2a848c8f8085a39210c9714b3d3b80448fea66afecb54a951345f2926c26d7a2"
PINNED_WORKER_SHA256 = "e3be938644722d14a1c1a0e82168518838c4d174cfe031076a339745c48870a8"


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_pins() -> Dict[str, str]:
    bad = {n: s for n, s in H317_PINS.items() if _sha(H317_SRC / n) != s}
    if bad:
        raise RuntimeError(f"H335: frozen H317 module changed: {sorted(bad)}")
    if _sha(GRAMMAR_PORT) != GRAMMAR_PORT_SHA256:
        raise RuntimeError("H335: LEARNERS-V1 grammar port sha mismatch")
    return dict(H317_PINS)


verify_pins()

GROUP = os.environ.get("H335_GROUP", "BASE")
if GROUP not in ("AGG", "BASE"):
    raise RuntimeError(f"H335_GROUP must be AGG or BASE, got {GROUP!r}")
if GROUP == "AGG":
    os.environ["H317_LEARNERS_V1"] = "1"  # read by the port at import
elif os.environ.get("H317_LEARNERS_V1"):
    raise RuntimeError("H335_GROUP=BASE with H317_LEARNERS_V1 set")

sys.path.insert(0, str(H317_SRC))
import common_h317 as C  # noqa: E402  (installs guard v2)

C.install_guard()
C._ALLOW_PRE_V = tuple(C._ALLOW_PRE_V) + (str(EVIDENCE) + os.sep,)
C._DENY_PRE_V = tuple(C._DENY_PRE_V) + (
    "/home/ubuntu/jope-data/h335-stagev/",
    "/home/ubuntu/jope-data/h335-cl/redteam/",
)
C.LOCK_RECEIPT = Path("/home/ubuntu/jope_plan/ns/receipts/H335_LOCK_sha256.txt")

if GROUP == "AGG":
    _spec = importlib.util.spec_from_file_location("gp_grammar_h317", GRAMMAR_PORT)
    G = importlib.util.module_from_spec(_spec)
    sys.modules["gp_grammar_h317"] = G
    _spec.loader.exec_module(G)
    assert G.LEARNERS_V1
else:
    import gp_grammar_h317 as G  # noqa: E402

    assert not getattr(G, "LEARNERS_V1", False)

import distributed_10gen_island_gp as D  # noqa: E402
import gp_engine_h317 as E  # noqa: E402
import gp_lab as L  # noqa: E402
import screener_h317 as S  # noqa: E402
import stacking_h317 as ST  # noqa: E402

assert D.G is G and L.G is G and ST.G is G
GRAMMAR_SHA256 = GRAMMAR_PORT_SHA256 if GROUP == "AGG" else H317_PINS["gp_grammar_h317.py"]

# ----------------------------------------------------------------------------------------------------------------------
# Locked design constants (PREREG §2-§4)
# ----------------------------------------------------------------------------------------------------------------------
SEEDS_T = (335101,)
SEEDS_V = (335101, 335102)
N_PROP = 48  # screened proposals per generation (H317 Evo A)
N_LIVE = 6  # live SEARCH evals per generation (H317 Evo A)
G_MAX = 107  # AGG: 22 + 6 * 107 = 664 live evals
CHECKPOINTS = (10, 25, 50, 75, 107)
OLD_OFFSET = 2  # OLD_MATCHED at gen gc + 2: 8 + 6 (gc + 2) = 6 gc + 20 vs AGG 6 gc + 22 (|delta| = 2)
H317_BASE_GEN = 10  # OLD_MATCHED's g10 snapshot = H317's adopted pipeline (68 evals)
STACK = "CV3@12"
LIVE_MATCH_TOL = 2
CLIP = 0.5
ALPHA = 0.025
TIE_TOL = 1e-4
BOOT_SEED = 335150
ELO_ANCHOR = "H257_REPLICA"
CPU_LIMIT_FACTOR = 1.0  # RLIMIT_CPU = the H317 per-kind timeout (60 s base, x4 portfolios, x2 AUDIT)
WALL_BACKSTOP_FACTOR = 3.0

# Seed F (H317 exploratory run, declared before any outcome): 6 new-head presets interleaved with Seed E.
NEW_HEAD_PRESETS = (
    ("xtrees_default", "(pipe X (xtrees))"),
    ("rforest_default", "(pipe X (rforest))"),
    ("oblivious_default", "(pipe X (oblivious))"),
    ("cheb3_oblivious", "(pipe X (cheb append=true degree=3) (oblivious))"),
    ("rank_xtrees", "(pipe X (rank) (xtrees max_features=0.7 min_leaf=3 trees=150))"),
    ("rforest_mf033", "(pipe X (rforest max_features=0.33 min_leaf=2 trees=200))"),
)


def seed_F(g: "D.GPContext", seed_e: "D.SeedRun", scope: str = "H335.seedF") -> "D.SeedRun":
    """Identical to explore_learners_h317.seed_F (only the selection-scope label differs)."""
    if GROUP != "AGG":
        raise RuntimeError("Seed F needs the LEARNERS-V1 grammar (H335_GROUP=AGG)")
    with D.selection_scope(scope):
        new = [G.Candidate.from_dsl(d, f"learners_preset:{n}") for n, d in NEW_HEAD_PRESETS]
        order = []
        for k in range(max(len(new), len(seed_e.order))):
            if k < len(new):
                order.append(new[k])
            if k < len(seed_e.order):
                order.append(seed_e.order[k])
        order = D.unique_by_sha(order)
        res = g.evaluate(order)
    return D.SeedRun("F", order, res, info={"new_head_presets": [n for n, _ in NEW_HEAD_PRESETS]})


def new_head(dsl: str) -> bool:
    if GROUP != "AGG":
        return False
    gg = G.parse(dsl)
    pipes = gg.members if isinstance(gg, G.Portfolio) else (gg,)
    return any(p.head is not None and p.head.op in G.NEW_HEADS for p in pipes)


# ----------------------------------------------------------------------------------------------------------------------
# Panels
# ----------------------------------------------------------------------------------------------------------------------
def _ctx_order(fam_ctx: Dict[str, List[str]]) -> Dict[str, List[str]]:
    return {f: sorted(c, key=lambda x: C.stream_seed(C.PILOT_SEED, "ctx", x)) for f, c in fam_ctx.items()}


def panel_p201b(rows: Sequence[dict]) -> List[str]:
    """The SECOND seeded context of every TRAIN family with >= 2 contexts. P205 (H317) took the first one of every
    family with the same seeded order (common_h317.pilot_shard), so P201b is disjoint from P205 by construction."""
    fam_ctx: Dict[str, List[str]] = {}
    for r in rows:
        fam_ctx.setdefault(r["family_id"], []).append(r["context_sha256"])
    out = sorted(c[1] for c in _ctx_order(fam_ctx).values() if len(c) >= 2)
    p205 = set(C.pilot_shard(rows, C.family_folds(rows)))
    if p205 & set(out):
        raise AssertionError("P201b overlaps P205")
    return out


def table_size(cid: str) -> int:
    """nt * (width + 1) from the 17-byte JTT1 header (used only to schedule the largest units first)."""
    with open(C.table_path(cid), "rb") as f:
        h = f.read(17)
    import struct

    nt, _nq, width = struct.unpack_from("<III", h, 4)
    return int(nt) * (int(width) + 1)


# ----------------------------------------------------------------------------------------------------------------------
# Worker runner: the H317 runner with a CPU-time limit (RLIMIT_CPU) and a wall backstop
# ----------------------------------------------------------------------------------------------------------------------
def rust_runner_cpu(worker_bin: Path, timeout_s: float, nice: int = 19):
    """E.rust_runner with one change: the per-kind H317 timeout (E.eval_timeout) is enforced as a CPU-time limit
    (RLIMIT_CPU via util-linux `prlimit`, SIGXCPU; no core dumps), so results do not depend on host load; a wall
    backstop of 3x stops a stalled process. A CPU-limit kill
    is reported as "timeout", exactly like a wall timeout in H317 (the engine retries it once, then records it)."""
    worker_bin = Path(worker_bin)
    if _sha(worker_bin) != SUPERSET_SHA256:
        raise RuntimeError("H335 runs every GP arm on the superset worker 2a848c8f...")

    def run(ctx, kind: str, cand, core: str, slot: Path):
        env_path = ctx.search_path if kind == "search" else ctx.audit_path if kind == "audit" else ctx.alts[kind].path
        C.assert_allowed(env_path)
        cand_path, out_path = slot / "candidate.json", slot / "out.json"
        cand_path.write_text(json.dumps({"dsl": cand.dsl, "program_sha256": cand.sha, "birth_seq": 0,
                                         "parents": list(cand.parents), "operator": cand.operator}, sort_keys=True))
        open(out_path, "w").close()  # truncate in place (no deletion)
        lim = max(1, int(round(E.eval_timeout(timeout_s, kind, cand.dsl) * CPU_LIMIT_FACTOR)))
        # prlimit execs the command (no preexec_fn: the engine forks from a thread pool)
        cmd = ["prlimit", f"--cpu={lim}:{lim + 2}", "--core=0", "--", "taskset", "-c", core, "nice", "-n", str(nice),
               str(worker_bin), "--fit-s1", str(env_path), str(cand_path), str(out_path)]
        env = {"PATH": "/usr/bin:/bin", "RAYON_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "HOME": str(slot),
               "TMPDIR": str(slot)}
        t0 = time.monotonic()
        try:
            p = subprocess.run(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                               timeout=lim * WALL_BACKSTOP_FACTOR)
            rc, err = p.returncode, p.stderr.decode(errors="replace")[-300:].strip()
        except subprocess.TimeoutExpired:
            return None, time.monotonic() - t0, "timeout"
        wall = time.monotonic() - t0
        if rc in (-signal.SIGXCPU, -signal.SIGKILL, 128 + signal.SIGXCPU):
            return None, wall, "timeout"
        if rc in (126, 127):  # prlimit/taskset could not exec the worker: infrastructure, never a program failure
            raise OSError(f"worker exec failed rc={rc}: {err}")
        if rc != 0:
            return None, wall, f"rc={rc}:{err}"
        try:
            doc = json.loads(out_path.read_text())
        except Exception as e:  # noqa: BLE001
            return None, wall, f"unparsable:{type(e).__name__}"
        if doc.get("status") != "completed":
            return None, wall, f"status={doc.get('status')}"
        return doc, wall, ""

    return run


def code_manifest(names: Sequence[str] = ()) -> Dict[str, str]:
    """sha256 of the frozen H317 modules + the named H335 files (default: every H335 source file), absolute paths
    (LOCK['code_sha256'] format; common_h317.unlock_stage_v resolves absolute keys as is)."""
    out = {str(H317_SRC / n): s for n, s in H317_PINS.items()}
    for p in sorted(HERE.iterdir()):
        if names and p.name not in names:
            continue
        if p.is_file() and p.suffix in (".py", ".rs", ".sh", ".md", ".json") and not p.name.startswith("."):
            out[str(p)] = _sha(p)
    if names and len(out) != len(H317_PINS) + len(set(names)):
        raise RuntimeError("code_manifest: a named H335 file is missing")
    return out
