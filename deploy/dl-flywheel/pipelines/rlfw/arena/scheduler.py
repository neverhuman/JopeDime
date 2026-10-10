"""Context-major, durable isolated outbox; slates precede all score disclosure."""

import fcntl
import json
import os
import random
import sqlite3
import tempfile
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .arms import Candidate, decision_value, draw_seed
from .game_spec import canonical, digest, local_path, R5_KPI_VERSION
from .kpi import SlotResult


def context_order(spec):
    order = list(spec.contexts)
    random.Random(spec.order_seed).shuffle(order)
    if spec.schedule == "one-per-lineage-then-all":
        first, rest, seen = [], [], set()
        for c in order:
            (rest if c.lineage_id in seen else first).append(c)
            seen.add(c.lineage_id)
        order = first + rest
    return order


def candidate_value(c):
    return c.to_dict()


def load_candidate(value):
    return Candidate.from_dict(value)


class ArenaScheduler:
    def __init__(self, spec, arms, oracle, decision_sink, *, workers=4):
        if type(workers) is not int or not 1 <= workers <= 4:
            raise ValueError(
                "arena starts with at most four isolated evaluator workers"
            )
        self.spec, self.arms, self.oracle, self.decision_sink = (
            spec,
            arms,
            oracle,
            decision_sink,
        )
        self.workers = workers
        spec.freeze()
        self.path = local_path(spec.bench_root, spec.state_path, "state")
        for suffix in ("-wal", "-shm", ".lock"):
            local_path(spec.bench_root, Path(str(self.path) + suffix), "state")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS slates(context TEXT, arm TEXT, seed INTEGER, payload TEXT NOT NULL,
                PRIMARY KEY(context,arm,seed));
            CREATE TABLE IF NOT EXISTS slots(context TEXT, arm TEXT, seed INTEGER, slot INTEGER,
                payload TEXT NOT NULL, charged INTEGER NOT NULL, execution_key TEXT NOT NULL, evidence TEXT NOT NULL,
                PRIMARY KEY(context,arm,seed,slot));
            CREATE TABLE IF NOT EXISTS closed_contexts(context TEXT PRIMARY KEY);
            CREATE TABLE IF NOT EXISTS attempts(context TEXT, arm TEXT, seed INTEGER, slot INTEGER, number INTEGER,
                result TEXT, PRIMARY KEY(context,arm,seed,slot,number));
            CREATE TABLE IF NOT EXISTS slate_costs(context TEXT, arm TEXT, seed INTEGER, wall_seconds REAL NOT NULL,
                PRIMARY KEY(context,arm,seed));
            CREATE TABLE IF NOT EXISTS prepared_slates(context TEXT, arm TEXT, seed INTEGER, payload TEXT NOT NULL,
                PRIMARY KEY(context,arm,seed));
        """)
        self.mutex = threading.RLock()
        with self.db:
            existing = self.db.execute(
                "SELECT value FROM metadata WHERE key='spec_sha256'"
            ).fetchone()
            if existing and existing[0] != spec.sha256:
                raise ValueError("outbox belongs to a different GameSpec")
            self.db.execute(
                "INSERT OR IGNORE INTO metadata VALUES('spec_sha256',?)", (spec.sha256,)
            )
        for key, evidence in self.db.execute(
            "SELECT execution_key,evidence FROM slots WHERE charged=1"
        ):
            self.oracle.own_results[key] = json.loads(evidence)
        if spec.kpi_version == R5_KPI_VERSION:
            self.db.executescript("""
                CREATE TABLE IF NOT EXISTS program_fates(
                    execution_key TEXT PRIMARY KEY, evidence TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS terminal_attempts(
                    context TEXT, arm TEXT, seed INTEGER, slot INTEGER,
                    payload TEXT NOT NULL, PRIMARY KEY(context,arm,seed,slot));
            """)
            for key, evidence in self.db.execute(
                "SELECT execution_key,evidence FROM program_fates"
            ):
                self.oracle.own_results[key] = json.loads(evidence)

    def close(self):
        self.db.close()

    def rows(self):
        return [
            SlotResult(**json.loads(row[0]))
            for row in self.db.execute(
                "SELECT payload FROM slots ORDER BY context,arm,seed,slot"
            )
        ]

    def closed_contexts(self):
        return [
            row[0]
            for row in self.db.execute(
                "SELECT context FROM closed_contexts ORDER BY context"
            )
        ]

    def committed(self, row):
        with self.mutex:
            value = self.db.execute(
                "SELECT payload,charged FROM slots WHERE context=? AND arm=? AND seed=? AND slot=?",
                (row.context_id, row.arm, row.seed, row.slot),
            ).fetchone()
        return (
            value is not None
            and json.loads(value[0]) == row.to_dict()
            and value[1] == int(row.charged)
        )

    def decode_costs(self):
        return [
            {
                "context_id": c,
                "arm": a,
                "seed": s,
                "wall_seconds": seconds,
                "spec_sha256": self.spec.sha256,
            }
            for c, a, s, seconds in self.db.execute(
                "SELECT context,arm,seed,wall_seconds FROM slate_costs ORDER BY context,arm,seed"
            )
        ]

    def commit(self, row, key, result):
        values = (
            row.context_id,
            row.arm,
            row.seed,
            row.slot,
            canonical(row.to_dict()),
            int(row.charged),
            key,
            canonical(result),
        )
        with self.mutex, self.db:
            old = self.db.execute(
                "SELECT payload FROM slots WHERE context=? AND arm=? AND seed=? AND slot=?",
                values[:4],
            ).fetchone()
            if old and old[0] != values[4]:
                raise ValueError("conflicting disclosure replay")
            self.db.execute(
                "INSERT OR IGNORE INTO slots VALUES(?,?,?,?,?,?,?,?)", values
            )
            self.db.execute(
                "UPDATE attempts SET result=? WHERE context=? AND arm=? AND seed=? AND slot=? AND number=?",
                (
                    canonical(result),
                    row.context_id,
                    row.arm,
                    row.seed,
                    row.slot,
                    row.attempts,
                ),
            )

    def attempt_history(self, key):
        with self.mutex:
            return [
                {"number": n, "result": json.loads(r) if r is not None else None}
                for n, r in self.db.execute(
                    "SELECT number,result FROM attempts WHERE context=? AND arm=? AND seed=? AND slot=? ORDER BY number",
                    key,
                )
            ]

    def reserve_attempt(self, key):
        with self.mutex, self.db:
            count = self.db.execute(
                "SELECT count(*) FROM attempts WHERE context=? AND arm=? AND seed=? AND slot=?",
                key,
            ).fetchone()[0]
            if count >= 3:
                raise ValueError("persisted per-slot retry ceiling exhausted")
            self.db.execute(
                "INSERT INTO attempts VALUES(?,?,?,?,?,NULL)", (*key, count + 1)
            )
            return count + 1

    def record_unknown_attempt(self, key, number, result):
        if (
            result["outcome_class"] not in ("parity_custody", "infra_unknown")
            or result.get("q_grid") is not None
        ):
            raise ValueError("unknown attempt records cannot carry a score")
        with self.mutex, self.db:
            self.db.execute(
                "UPDATE attempts SET result=? WHERE context=? AND arm=? AND seed=? AND slot=? AND number=?",
                (canonical(result), *key, number),
            )

    def terminal_attempt(self, key):
        if self.spec.kpi_version != R5_KPI_VERSION:
            return None
        with self.mutex:
            row = self.db.execute(
                "SELECT payload FROM terminal_attempts WHERE context=? AND arm=? AND seed=? AND slot=?",
                key,
            ).fetchone()
        return json.loads(row[0]) if row else None

    def record_program_fate(self, key, number, execution_key, result, *, cached):
        if (
            self.spec.kpi_version != R5_KPI_VERSION
            or result["outcome_class"] != "parity_unqualified"
            or result.get("qualified") is not False
            or any(result.get(name) is not None for name in ("q", "q_opt", "q_grid"))
        ):
            raise ValueError("terminal program fates require the R5 parity outcome")
        payload = canonical(
            {
                "number": number,
                "result": result,
                "cache_hit": cached,
                "wall_seconds": 0.0 if cached else result["wall_seconds"],
                "cpu_seconds": 0.0 if cached else result["cpu_seconds"],
                "unknown_cost": False
                if cached
                else bool(result.get("cost_unknown", False)),
            }
        )
        evidence = canonical(result)
        with self.mutex, self.db:
            reserved = self.db.execute(
                "SELECT 1 FROM attempts WHERE context=? AND arm=? AND seed=? AND slot=? AND number=?",
                (*key, number),
            ).fetchone()
            if reserved is None:
                raise ValueError(
                    "terminal fate requires its durable attempt reservation"
                )
            old = self.db.execute(
                "SELECT evidence FROM program_fates WHERE execution_key=?",
                (execution_key,),
            ).fetchone()
            if old and old[0] != evidence:
                raise ValueError("conflicting permanent program fate")
            previous = self.db.execute(
                "SELECT payload FROM terminal_attempts WHERE context=? AND arm=? AND seed=? AND slot=?",
                key,
            ).fetchone()
            if previous and previous[0] != payload:
                raise ValueError("conflicting terminal attempt replay")
            self.db.execute(
                "INSERT OR IGNORE INTO program_fates VALUES(?,?)",
                (execution_key, evidence),
            )
            self.db.execute(
                "INSERT OR IGNORE INTO terminal_attempts VALUES(?,?,?,?,?)",
                (*key, payload),
            )

    def slate(self, context, arm, seed):
        key = (context.context_id, arm.name, seed)
        if self.prepares(arm):
            return self.prepared_slate(context, arm, seed)
        value = self.db.execute(
            "SELECT payload FROM slates WHERE context=? AND arm=? AND seed=?", key
        ).fetchone()
        if value:
            return tuple(load_candidate(v) for v in json.loads(value[0]))
        start = time.monotonic()
        slate = self.arms.slate(context, arm, seed, self.spec.budget)
        elapsed = time.monotonic() - start
        if len(slate) != (
            int(context.hwm_program_sha256 is not None)
            if arm.name == "hwm"
            else self.spec.budget
        ):
            raise ValueError("arm did not precommit the required fixed number of slots")
        with self.db:
            self.db.execute(
                "INSERT INTO slates VALUES(?,?,?,?)",
                (*key, canonical([candidate_value(c) for c in slate])),
            )
            self.db.execute("INSERT INTO slate_costs VALUES(?,?,?,?)", (*key, elapsed))
        return slate

    def prepares(self, arm):
        return arm.name.startswith("rerank32:")

    def load_packet(self, value):
        from ..slate import PreparedSlate

        return PreparedSlate.from_dict(value)

    def prepared_slate(self, context, arm, seed):
        if self.spec.slate_k != 32:
            raise ValueError("rerank GameSpec requires fixed K=32")
        key = (context.context_id, arm.name, seed)
        value = self.db.execute(
            "SELECT payload FROM prepared_slates WHERE context=? AND arm=? AND seed=?",
            key,
        ).fetchone()
        if value:
            packet = self.load_packet(json.loads(value[0]))
        else:
            start = time.monotonic()
            packet = self.arms.prepare_slate(context, arm, seed)
            elapsed = time.monotonic() - start
            # Save raw draws, all predictions and shortfalls even when B is unavailable.
            with self.db:
                self.db.execute(
                    "INSERT INTO prepared_slates VALUES(?,?,?,?)",
                    (*key, canonical(packet.to_dict())),
                )
                self.db.execute(
                    "INSERT INTO slates VALUES(?,?,?,?)",
                    (*key, canonical([candidate_value(c) for c in packet.selected])),
                )
                self.db.execute(
                    "INSERT INTO slate_costs VALUES(?,?,?,?)", (*key, elapsed)
                )
        # The inspectable packet and all raw decisions precede score disclosure.
        payload = (
            canonical(
                {
                    "spec_sha256": self.spec.sha256,
                    "context_id": context.context_id,
                    "arm": arm.name,
                    "seed": seed,
                    "packet": packet.to_dict(),
                }
            )
            + "\n"
        )
        path = local_path(
            self.spec.bench_root,
            Path(self.spec.out_dir) / "slates" / (digest(key) + ".json"),
            "runs",
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        staged = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", dir=path.parent, prefix=".slate-", delete=False
            ) as f:
                staged = Path(f.name)
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())
            try:
                os.link(staged, path)
            except FileExistsError:
                if path.read_text() != payload:
                    raise ValueError("conflicting prepared slate replay")
            directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            if staged is not None:
                staged.unlink()
        for index, c in enumerate(packet.raw_draws, 1):
            if self.arms.by_sha.get(c.program.program_sha256) != c.program:
                raise ValueError("replayed raw draw is not an exact trie leaf")
            if c.draw_seed != draw_seed(context.context_id, arm.name, seed, index):
                raise ValueError("replayed raw draw seed identity changed")
            self.decision_sink(
                decision_value(self.spec, context, arm, seed, index, c, "raw_draw")
            )
        if len(packet.selected) != self.spec.budget:
            raise ValueError(
                "512 raw draws yielded fewer than eight distinct programs; no evaluations authorized"
            )
        return packet.selected

    def _run_context(self, context):
        self.oracle.check_role(context)
        jobs = []
        # Freeze every arm/seed slate before the first evaluation in this context.
        for arm in self.spec.arms:
            seeds = self.spec.seeds[:1] if arm.name == "hwm" else self.spec.seeds
            for seed in seeds:
                slate = self.slate(context, arm, seed)
                seen = set()
                for slot, candidate in enumerate(slate, 1):
                    duplicate = candidate.program.program_sha256 in seen
                    seen.add(candidate.program.program_sha256)
                    jobs.append((arm, seed, slot, candidate, duplicate))
                    if arm.name != "hwm" and not self.prepares(arm):
                        self.decision_sink(
                            decision_value(
                                self.spec,
                                context,
                                arm,
                                seed,
                                slot,
                                candidate,
                                "raw_draw",
                            )
                        )
        # One task per exact key, in fixed first-disclosure order. Shared repeats
        # are executed once and still charged for every logical arm/seed slot.
        pending, existing = defaultdict(list), {}
        for job in jobs:
            arm, seed, slot, candidate, duplicate = job
            old = self.db.execute(
                "SELECT payload FROM slots WHERE context=? AND arm=? AND seed=? AND slot=?",
                (context.context_id, arm.name, seed, slot),
            ).fetchone()
            if old:
                existing[(arm.name, seed, slot)] = SlotResult(**json.loads(old[0]))
            else:
                pending[self.oracle.execution_key(context, candidate.program)].append(
                    job
                )

        def resolve_group(group):
            return [
                self.oracle.resolve(
                    context,
                    arm,
                    seed,
                    slot,
                    candidate,
                    duplicate=duplicate,
                    commit=self.commit,
                    attempt_store=self,
                )
                for arm, seed, slot, candidate, duplicate in group
            ]

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = [pool.submit(resolve_group, group) for group in pending.values()]
            resolved = dict(existing)
            for future in futures:
                for row in future.result():
                    resolved[(row.arm, row.seed, row.slot)] = row
            # Result/trace ordering is deterministic even when executions finish out of order.
            for arm, seed, slot, candidate, duplicate in jobs:
                row = resolved[(arm.name, seed, slot)]
                if not self.committed(row):
                    raise RuntimeError(
                        "score disclosed before the durable charge committed"
                    )
                if arm.name != "hwm":
                    value = decision_value(
                        self.spec, context, arm, seed, slot, candidate, "selected_slot"
                    )
                    value["execution_key"] = self.oracle.execution_key(
                        context, candidate.program
                    )
                    value["outcome_class"] = row.outcome
                    self.decision_sink(value)
        self.spec.verify_frozen()
        self.oracle.check_role(context)
        with self.db:
            self.db.execute(
                "INSERT OR IGNORE INTO closed_contexts VALUES(?)", (context.context_id,)
            )

    def run(self, *, max_contexts=None, on_context=None):
        if max_contexts is not None and (
            type(max_contexts) is not int or max_contexts < 1
        ):
            raise ValueError("positive context limit required")
        with Path(str(self.path) + ".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            closed = set(self.closed_contexts())
            done = 0
            for context in context_order(self.spec):
                if context.context_id in closed:
                    continue
                if (Path(self.spec.bench_root) / "state/STOP.json").exists():
                    raise RuntimeError("campaign stop marker is present")
                self.spec.verify_frozen()
                self._run_context(context)
                done += 1
                if on_context:
                    on_context(self)
                if max_contexts is not None and done >= max_contexts:
                    break
        return self.rows()

    def hwm_values(self):
        hwm = {}
        for row in self.rows():
            if row.arm == "hwm":
                hwm[row.context_id] = row.q
                hwm[(row.context_id, "program_sha256")] = row.program_sha256
        return hwm
