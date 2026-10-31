"""Physical read-only SQLite backup; only the private copy receives indexes.

The frozen Outbox/Queue verifiers run unchanged. A SELECT fixes a WAL transaction
before backup. Snapshot preparation grants neither a lease nor execution rights.
"""
from pathlib import Path
import sqlite3
import time

_SNAPSHOT_PROBE_SQL = {
    'outbox': 'SELECT * FROM config LIMIT 1',
    'queue': 'SELECT * FROM jobs LIMIT 1',
}


def need(value, message):
    if not value:
        raise ValueError(message)


def source_path(path):
    path = Path(path)
    need(path.is_absolute() and '..' not in path.parts and path.is_file() and
         all(not p.is_symlink() for p in (path, *path.parents)), 'absolute existing nonsymlink SQLite source')
    for suffix in ('-wal', '-shm'):
        need(not Path(str(path) + suffix).is_symlink(), 'SQLite source sidecar symlink')
    return path


def readonly_snapshot(path, *, deadline, kind='outbox', max_bytes=256 * 1024**2):
    """Return an owned memory connection; caller must close it.

    max_bytes bounds database pages before and after backup. Python/SQLite index
    overhead is additional; the external 1GiB process ceiling remains mandatory.
    """
    path = source_path(path)
    need(type(kind) is str and kind in _SNAPSHOT_PROBE_SQL and type(max_bytes) is int and
         1024 <= max_bytes <= 256 * 1024**2, 'bounded snapshot kind/pages')
    start = time.monotonic()
    remaining = deadline - time.time()
    need(0 < remaining <= 86400, 'finite future snapshot deadline')
    def tick(*_):
        need(time.time() < deadline and time.monotonic() - start < remaining,
             'snapshot deadline')
    source = sqlite3.connect(path.as_uri() + '?mode=ro', uri=True,
                             isolation_level=None, timeout=min(2, remaining))
    target = sqlite3.connect(':memory:', isolation_level=None)
    target.row_factory = sqlite3.Row
    target.set_progress_handler(lambda: int(time.time() >= deadline or
                                            time.monotonic() - start >= remaining), 1000)
    try:
        source.execute('PRAGMA query_only=ON')
        source.execute('BEGIN')
        source.execute(_SNAPSHOT_PROBE_SQL[kind]).fetchone()
        pages = source.execute('PRAGMA page_count').fetchone()[0]
        size = source.execute('PRAGMA page_size').fetchone()[0]
        need(pages * size <= max_bytes, 'snapshot byte ceiling')
        source.backup(target, pages=256, progress=tick, sleep=0.01)
        need(target.execute('PRAGMA page_count').fetchone()[0] * size <= max_bytes,
             'copied snapshot byte ceiling')
        if kind == 'outbox':
            target.execute('CREATE INDEX native_private_batch_lookup ON emissions(batch_id)')
            target.execute("CREATE INDEX native_private_context_lookup ON emissions(json_extract(body,'$.emission.context_id'))")
        else:
            target.execute('CREATE INDEX native_private_transition_lookup ON transitions(job)')
        tick()
        return target
    except BaseException:
        target.close()
        raise
    finally:
        source.close()


def verified_snapshot(path, *, deadline, kind='outbox', expected_config=None,
                      max_bytes=256 * 1024**2):
    start = time.monotonic()
    remaining = deadline - time.time()
    db = readonly_snapshot(path, deadline=deadline, kind=kind, max_bytes=max_bytes)
    try:
        db.set_progress_handler(lambda: int(time.time() >= deadline or
                                            time.monotonic() - start >= remaining), 1000)
        if kind == 'outbox':
            import json
            from neural_outbox import Outbox
            checker = Outbox.__new__(Outbox)
            checker.db = db
            proof = checker.verify()
            if expected_config is not None:
                need(json.loads(db.execute('SELECT body FROM config').fetchone()[0]) == expected_config,
                     'exact snapshot config')
        else:
            from jope_flywheel.queue import Queue
            checker = Queue.__new__(Queue)
            checker.db = db
            proof = checker.verify()
        need(time.time() < deadline and time.monotonic() - start < remaining, 'full verifier deadline')
        return db, proof
    except BaseException:
        db.close()
        raise
