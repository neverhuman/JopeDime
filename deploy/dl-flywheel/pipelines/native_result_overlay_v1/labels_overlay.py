"""Overlay labels: the frozen ``label_context`` on the UNION of shard rows and cumulative overlay rows.

Invariants:
- The loss is unchanged (L6): every overlay record has ``ce_eligible=False`` and ``source_kind`` SOURCE_KIND; q,
  feasibility and runtime follow the frozen ``NativeContextLoader.get`` conversion exactly.
- Every overlay row carries its context's ANCHORED baseline pair, so the frozen bit-equality rule of
  ``label_context`` holds across shard and overlay rows.
- Old base labels are kept: for every shard row the union labels equal the shard-only labels on every field a
  NativeRecord uses; a context where they would differ is refused (OLD_LABELS_WOULD_CHANGE). The frozen base loader
  still labels its own window from shard rows only.
- Direct recomputation (q_from_detail / clip_q / q_grid from y_train, the anchored pair and the verified RMSE) must
  equal the label: exact ints and grids, floats within 1e-12 relative.
"""
import math
from dataclasses import asdict

from dltrain import labels as L
from jope_flywheel import training_snapshot_native as N

from .common import RECOMPUTE_REL, SOURCE_KIND, need

RECORD_FIELDS = tuple(N.NativeRecord.__dataclass_fields__)
OLD_LABEL_FIELDS = ('q', 'q_opt', 'q_grid', 'q_status', 'feas_class', 'feas_valid', 'rt_target', 'rt_censored',
                    'rt_valid')


def label_row(cid, program_sha256, tokens, skeleton_sha256, rmse, anchored, wall_us):
    """One detailed, completed, local-backend row in the frozen S1 row shape consumed by ``label_context``."""
    return {'context_sha256': cid, 'program_sha256': program_sha256, 'skeleton_sha256': skeleton_sha256,
            'tokens': list(tokens), 'length': len(tokens), 'outcome': 'completed', 'rmse_j': list(rmse),
            'baseline_rmse_j': list(anchored), 'wall_us': wall_us, 'backend': 'local', 'shared_seed': False}


def overlay_record(lab, tokens, skeleton_sha256):
    """NativeRecord exactly as the frozen loader converts a detailed row, except CE is always False."""
    qok = lab.q is not None
    rt = math.expm1(lab.rt_target) if lab.rt_valid else None
    return N.NativeRecord(lab.program_sha256, tuple(tokens), lab.outcome, lab.q if qok else None,
                          lab.q_opt if qok else None, lab.q_grid if qok else None,
                          lab.feas_class if lab.feas_valid else (0 if lab.outcome == 'completed' else None),
                          rt, lab.rt_censored, False, qok, qok, rt is not None, SOURCE_KIND, skeleton_sha256)


def record_json(record):
    out = asdict(record)
    out['tokens'] = list(record.tokens)
    return out


def record_from_json(doc):
    need(type(doc) is dict and set(doc) == set(RECORD_FIELDS), 'closed NativeRecord fields')
    return N.NativeRecord(**{**doc, 'tokens': tuple(doc['tokens'])})


def union_labels(context_doc, shard_rows, overlay_rows):
    """Frozen ``label_context`` over shard rows + overlay rows; returns (ContextLabel, {program: RecordLabel})."""
    context_label, labs = L.label_context(context_doc, list(shard_rows) + list(overlay_rows))
    return context_label, {lab.program_sha256: lab for lab in labs}


def old_labels_equal(shard_only, union):
    """True when every shard row keeps every NativeRecord-relevant label field under the union."""
    return all(all(getattr(shard_only[p], f) == getattr(union[p], f) for f in OLD_LABEL_FIELDS) for p in shard_only)


def _close(a, b):
    if a is None or b is None:
        return a is None and b is None
    return a == b or math.isclose(a, b, rel_tol=RECOMPUTE_REL, abs_tol=0.0)


def direct_recomputation(y_train, rmse, anchored, wall_us, record):
    """Independent recomputation of one overlay record's targets; returns the comparison (``equal`` must be True)."""
    s_c = L.robust_scale(y_train)
    eps = L.epsilon(y_train, s_c)
    if any(b < eps for b in anchored):
        q = None
    else:
        q = L.q_from_detail(rmse, anchored, eps)
    q_opt = None if q is None else L.clip_q(q)
    grid = None if q is None else L.q_grid(q)
    runtime = None if wall_us is None else math.expm1(math.log1p(wall_us / 1e6))
    checks = {'q': _close(q, record.q), 'q_opt': _close(q_opt, record.q_opt),
              'q_grid': grid == record.q_grid and (grid is None or type(record.q_grid) is int),
              'feasibility': record.feasibility == 0,
              'runtime_seconds': _close(runtime, record.runtime_seconds),
              'quality_eligible': record.quality_eligible == (q is not None) == record.pairwise_eligible,
              'runtime_eligible': record.runtime_eligible == (runtime is not None),
              'ce_eligible': record.ce_eligible is False}
    return {'program_sha256': record.program_sha256, 'q_direct': q, 'q_label': record.q, 'q_grid_direct': grid,
            'q_grid_label': record.q_grid, 'eps_c': eps, 'checks': checks, 'equal': all(checks.values())}
