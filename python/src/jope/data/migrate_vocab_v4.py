"""Migrate a frozen active_vocab.json to schema v4.

Old snapshots (pre-v4) have two defects that silently corrupt training:
1. No ``tasks``/``generate`` fields → rehydration made every op task-universal and
   sampleable (regression-only ``gpr`` leaked into binary; quarantined parse-only ops
   became proposable).
2. Stale newmath ``map`` expression macro bins (``sigmoid_col0`` …) that the source
   grammar replaced with emittable elementwise forms (``sigmoid(x)`` …) — training on the
   frozen vocab taught the model tokens the emitter cannot compile.

Migration = re-serialize each engine grammar with the CURRENT ``OpSpec.to_json`` (which
writes tasks+generate), preserving the frozen snapshot's learned numeric bins (corpus rows
must still encode) but replacing expr-kind bins with the source grammar's emittable set.
Rows using stale expr bins (~1%) will no longer encode and are skipped from policy
training (counted, not crashed).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any

from ..grammars import all_grammars
from ..grammars.spec import GrammarSpec
from ..grammars.vocab import grammars_from_vocab_v3

VOCAB_SCHEMA_VERSION = 4


def grammar_structure_hash(specs: list[GrammarSpec]) -> str:
    """Hash of the structural surface that must match between source and frozen vocab:
    op names, roles, param names/kinds, tasks, generate, and EXPR-param bins (the drift
    class that bit us). Numeric bins are excluded — learned bins legitimately extend them.
    """
    parts: list[str] = []
    for spec in sorted(specs, key=lambda s: s.engine):
        for role, group in (("t", spec.transforms), ("h", spec.heads)):
            for name in sorted(group):
                op = group[name]
                expr_bins = [
                    "|".join(str(b) for b in p.bins) for p in op.params if p.kind == "expr"
                ]
                parts.append(
                    f"{spec.engine}:{role}:{name}:params={','.join(p.name + ':' + p.kind for p in op.params)}"
                    f":tasks={','.join(sorted(op.tasks))}:gen={int(op.generate)}:expr={';'.join(expr_bins)}"
                )
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:16]


def check_vocab_drift(vocab: dict[str, Any]) -> tuple[bool, str, str]:
    """(ok, source_hash, frozen_hash). ok=False → training must refuse (migrate first)."""
    source = sorted(all_grammars(), key=lambda s: s.engine)
    frozen = grammars_from_vocab_v3(vocab)
    frozen_list = [frozen[e] for e in sorted(frozen)]
    sh = grammar_structure_hash(source)
    fh = grammar_structure_hash(frozen_list)
    return sh == fh, sh, fh


def migrate_vocab_v4(vocab_path: str | Path, *, write: bool = True) -> dict[str, Any]:
    """Rewrite the frozen vocab to schema v4. Returns the new vocab dict + a report."""
    vocab_path = Path(vocab_path)
    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    source = {spec.engine: spec for spec in all_grammars()}
    frozen = grammars_from_vocab_v3(vocab)  # already patches tasks/generate from source

    engines_out: dict[str, Any] = {}
    replaced_bins: dict[str, dict[str, list[str]]] = {}
    for engine, spec in frozen.items():
        src = source.get(engine)
        if src is None:
            engines_out[engine] = spec.to_json()
            continue
        patched: dict[str, dict] = {"transforms": {}, "heads": {}}
        for field, group, src_group in (
            ("transforms", spec.transforms, src.transforms),
            ("heads", spec.heads, src.heads),
        ):
            for name, op in group.items():
                src_op = src_group.get(name)
                if src_op is not None and any(p.kind == "expr" for p in op.params):
                    # replace EXPR bins with the source grammar's emittable set
                    new_params = []
                    for p in op.params:
                        if p.kind == "expr":
                            src_p = next((sp for sp in src_op.params if sp.name == p.name), None)
                            if src_p is not None and tuple(p.bins) != tuple(src_p.bins):
                                replaced_bins.setdefault(engine, {})[f"{name}.{p.name}"] = [
                                    f"{list(p.bins)} -> {list(src_p.bins)}"
                                ]
                                p = dataclasses.replace(p, bins=tuple(src_p.bins))
                        new_params.append(p)
                    op = dataclasses.replace(op, params=tuple(new_params))
                patched[field][name] = op
        spec = dataclasses.replace(spec, transforms=patched["transforms"], heads=patched["heads"])
        engines_out[engine] = spec.to_json()

    out = dict(vocab)
    out["engines"] = engines_out
    out["schema_version"] = VOCAB_SCHEMA_VERSION
    ok, sh, fh = check_vocab_drift(out)
    out["grammar_structure_hash"] = fh
    report = {
        "schema_version": VOCAB_SCHEMA_VERSION,
        "drift_resolved": ok,
        "source_hash": sh,
        "frozen_hash": fh,
        "replaced_expr_bins": replaced_bins,
    }
    if write:
        backup = vocab_path.with_suffix(".v3.bak.json")
        if not backup.exists():
            backup.write_text(json.dumps(vocab), encoding="utf-8")
        vocab_path.write_text(json.dumps(out), encoding="utf-8")
    return {"vocab": out, "report": report}
