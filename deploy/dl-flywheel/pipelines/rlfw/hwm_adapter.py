"""Read-only original incumbent selection; no historical numeric score is exposed."""

import hashlib
from dataclasses import dataclass


@dataclass(frozen=True)
class OriginalHWMProgram:
    context_identity: str
    native_view_id: str
    dataset_version: str
    cut: str
    outer_fold: str | int
    parent_lineage: str
    program_sha256: str
    dsl: str
    tokens: tuple[int, ...] | None
    native_descriptor: dict


def original_hwm_program(context, selection):
    """Preserve native identities; the returned program must be re-executed for q."""
    aliases = {
        "context_identity": "context_id",
        "native_view_id": "view_id",
        "dataset_version": "dataset_hash",
        "cut": "cut_id",
        "outer_fold": "fold",
        "parent_lineage": "parent_lineage_sha256",
    }
    identity = {
        field: context.get(field, context.get(alias))
        for field, alias in aliases.items()
    }
    if any(value is None or value == "" for value in identity.values()):
        raise ValueError("original native context identity is incomplete")
    cid = identity["context_identity"]
    if cid in selection:
        # The deployed read-only cache is a context-id -> original selection map.
        selection = selection[cid]
        selected_cid = selection.get(
            "context_identity", selection.get("context_id", cid)
        )
    else:
        selected_cid = selection.get("context_identity", selection.get("context_id"))
    if selected_cid != cid:
        raise ValueError("incumbent belongs to another context")
    dsl = selection.get("dsl")
    if not isinstance(dsl, str) or not dsl:
        raise ValueError("missing original incumbent program")
    program_sha256 = hashlib.sha256(dsl.encode()).hexdigest()
    if program_sha256 != selection.get("program_sha256"):
        raise ValueError("original incumbent program bytes do not match SHA")
    tokens = selection.get("tokens")
    if tokens is not None:
        tokens = tuple(tokens)
        if (
            len(tokens) < 2
            or tokens[0] != 1
            or tokens[-1] != 2
            or any(type(t) is not int for t in tokens)
        ):
            raise ValueError("original BOS/EOS-inclusive tokens required")
    return OriginalHWMProgram(
        **identity,
        program_sha256=program_sha256,
        dsl=dsl,
        tokens=tokens,
        native_descriptor=dict(context),
    )
