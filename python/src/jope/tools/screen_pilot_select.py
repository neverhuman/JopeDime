"""Freeze metadata-only H05 populations before outcome reads. Never fits."""

import argparse
import math
from pathlib import Path

from head_cost_census import program_head
from scientific_metadata import (
    CONTEXT_FIELDS,
    canonical,
    context,
    digest,
    finite,
    immutable,
    inside,
    program,
    read_with_digest,
    reference,
    sha256_value,
)
from screen_fixture import AXES, GATES

SCHEMA = "parallel.screen_pilot_selection.v2"
N_FAMILIES, CUTS, PROGRAMS, RETAINED, AUDITS = 8, (0, 1), 64, 16, 5


def refuse_payloads(path):
    if "data" in Path(path).parts:
        raise ValueError("refusing to open data/ payloads")
    return True


def load_observations(root):
    raise ValueError("outcome-led selection retired; supply a metadata-only inventory")


def _inventory(metadata):
    if (
        not isinstance(metadata, dict)
        or set(metadata) != {"schema", "contexts"}
        or metadata["schema"] != "parallel.screen_metadata.v1"
        or not isinstance(metadata["contexts"], list)
    ):
        raise ValueError("metadata-only inventory required")
    groups = {}
    for row in metadata["contexts"]:
        if not isinstance(row, dict) or set(row) != set(CONTEXT_FIELDS) | {"candidates"}:
            raise ValueError("metadata context has outcomes or unknown fields")
        bound = context(row)
        if bound["cut"] not in CUTS:
            continue
        key = (bound["family"], bound["dataset"], bound["cut"])
        if key in groups:
            raise ValueError("duplicate or conflicting metadata context")
        candidates = {}
        if not isinstance(row["candidates"], list):
            raise ValueError("metadata candidate list required")
        for candidate in row["candidates"]:
            if (
                not isinstance(candidate, dict)
                or not {"genome", "program_sha256"} <= set(candidate)
                or set(candidate) - {"genome", "program_sha256", "original_reference"}
            ):
                raise ValueError("candidate metadata has outcomes or unknown fields")
            p = program(candidate)
            if "original_reference" in candidate:
                ref = candidate["original_reference"]
                if (
                    not isinstance(ref, dict)
                    or set(ref) != {"path", "sha256"}
                    or not isinstance(ref["path"], str)
                    or not sha256_value(ref["sha256"])
                ):
                    raise ValueError("original reference pointer invalid")
                p["original_reference"] = ref  # Pointer only; not opened here.
            if p["program_sha256"] in candidates:
                raise ValueError("duplicate metadata candidate")
            candidates[p["program_sha256"]] = p
        groups[key] = (bound, candidates)
    return groups


def freeze_contexts(metadata, n_families=N_FAMILIES):
    if n_families != N_FAMILIES:
        raise ValueError("frozen eight-family pilot cannot be reduced")
    groups = _inventory(metadata)
    chosen = {}
    for family, dataset in sorted({key[:2] for key in groups}):
        if all(
            (family, dataset, cut) in groups and len(groups[family, dataset, cut][1]) >= PROGRAMS
            for cut in CUTS
        ):
            chosen.setdefault(family, dataset)
    if len(chosen) < N_FAMILIES:
        raise ValueError("metadata lacks eight training families with 64 candidates on both cuts")
    return [groups[family, chosen[family], cut][0] for family in sorted(chosen)[:N_FAMILIES] for cut in CUTS]


def select_programs(metadata, contexts):
    groups = _inventory(metadata)
    result = []
    for bound in contexts:
        saved, candidates = groups[bound["family"], bound["dataset"], bound["cut"]]
        if saved != bound:
            raise ValueError("frozen context changed")
        result.extend(dict(candidates[sha], context=bound) for sha in sorted(candidates)[:PROGRAMS])
    return result


def _scores(selection, selection_sha, scores):
    if (
        not isinstance(scores, dict)
        or set(scores) != {"schema", "selection_sha256", "axis", "rows"}
        or scores["schema"] != "parallel.screen_scores.v1"
        or scores["selection_sha256"] != selection_sha
        or scores["axis"] not in {a["name"] for a in AXES}
    ):
        raise ValueError("screen scores do not bind frozen selection/axis")
    expected = {(digest(p["context"]), p["program_sha256"]): p for p in selection["programs"]}
    rows = {}
    for row in scores["rows"]:
        fields = {
            "context",
            "program_sha256",
            "proxy_genome",
            "proxy_program_sha256",
            "proxy_contract_sha256",
            "status",
            "score",
        }
        if not isinstance(row, dict) or set(row) != fields:
            raise ValueError("complete original/proxy screen pair required")
        bound = context(row["context"])
        if bound != row["context"]:
            raise ValueError("screen context contains unbound fields")
        key = (digest(bound), row["program_sha256"])
        if key not in expected or key in rows:
            raise ValueError("extra, cross-context or duplicate screen candidate")
        proxy = program(dict(genome=row["proxy_genome"], program_sha256=row["proxy_program_sha256"]))
        if not sha256_value(row["proxy_contract_sha256"]):
            raise ValueError("proxy numerical contract missing")
        if row["status"] not in ("ok", "failed", "unknown"):
            raise ValueError("screen status unavailable")
        if (row["status"] == "ok" and not finite(row["score"])) or (
            row["status"] != "ok" and row["score"] is not None
        ):
            raise ValueError("screen failure/score conflict")
        if scores["axis"] == "same_row_f32" and proxy["program_sha256"] != row["program_sha256"]:
            raise ValueError("same-row precision screen changed canonical program")
        if scores["axis"] == "histgbm_proxy" and (
            program_head(expected[key]["genome"]) != "histgbm2"
            or program_head(proxy["genome"]) != "histgbm"
            or expected[key]["genome"].replace("(histgbm2", "(histgbm", 1) != proxy["genome"]
        ):
            raise ValueError("histogram original/proxy heads differ from frozen axis")
        rows[key] = dict(
            row,
            pair_identity=digest(
                dict(
                    context=bound,
                    original_program_sha256=row["program_sha256"],
                    proxy=proxy,
                    proxy_contract_sha256=row["proxy_contract_sha256"],
                    axis=scores["axis"],
                )
            ),
        )
    if set(rows) != set(expected):
        raise ValueError("missing screen candidates; failures cannot be dropped")
    return rows


def rejection_audits(selection, selection_sha, scores):
    """Best16 plus five deterministic members of rejected48, all from frozen64."""
    rows = _scores(selection, selection_sha, scores)
    shortlist, audits = [], []
    for bound in selection["contexts"]:
        population = [r for (c, _), r in rows.items() if c == digest(bound)]
        ranked = sorted(
            population,
            key=lambda r: (
                r["status"] != "ok",
                r["score"] if r["status"] == "ok" else math.inf,
                r["program_sha256"],
            ),
        )
        if sum(r["status"] == "ok" for r in ranked) < RETAINED:
            raise ValueError("fewer than sixteen completed screen candidates")
        shortlist.extend(ranked[:RETAINED])
        rejected = sorted(
            ranked[RETAINED:],
            key=lambda r: digest(
                dict(
                    selection_sha256=selection_sha,
                    context=bound,
                    original_program_sha256=r["program_sha256"],
                    purpose="rejected48_audit5_v1",
                )
            ),
        )
        audits.extend(rejected[:AUDITS])
    return shortlist, audits


def spearman(xs, ys):
    if len(xs) != len(ys) or len(xs) < 2 or not all(finite(x) for x in [*xs, *ys]):
        return None

    def ranks(values):
        order = sorted(range(len(values)), key=lambda i: values[i])
        result = [0.0] * len(values)
        start = 0
        while start < len(order):
            stop = start + 1
            while stop < len(order) and values[order[stop]] == values[order[start]]:
                stop += 1
            for i in order[start:stop]:
                result[i] = (start + stop - 1) / 2
            start = stop
        return result

    left, right = ranks(xs), ranks(ys)
    mean = (len(xs) - 1) / 2
    a, b = [v - mean for v in left], [v - mean for v in right]
    denominator = math.sqrt(sum(v * v for v in a) * sum(v * v for v in b))
    return sum(x * y for x, y in zip(a, b)) / denominator if denominator else None


def reference_status(root, candidates):
    """Check pointers only; self-hashed outcomes cannot grant original query access."""
    rows = []
    for candidate in candidates:
        reason = "original reference pointer unavailable"
        ref = candidate.get("original_reference")
        if ref is not None:
            try:
                value = reference(root, ref)
                if (
                    not isinstance(value, dict)
                    or value.get("context") != candidate["context"]
                    or value.get("program_sha256") != candidate["program_sha256"]
                ):
                    raise ValueError("original reference identity conflict")
                reason = (
                    "original native query/access verification unavailable; hashes are not execution evidence"
                )
            except (OSError, ValueError, TypeError, KeyError) as error:
                reason = str(error)
        rows.append(
            dict(
                context=candidate["context"],
                program_sha256=candidate["program_sha256"],
                original_reference=ref,
                verified=False,
                unavailable_reason=reason,
            )
        )
    return rows


def histgbm_replay(*_args, **_kwargs):
    raise ValueError("head-SHA intersection retired; use original/proxy pairs bound to frozen candidates")


def select(root, output, *, metadata=None, screen=None):
    if metadata is None:
        raise ValueError("metadata-only inventory required before reference outcome access")
    root, output = Path(root).resolve(strict=True), Path(output)
    if not output.is_absolute():
        output = root / output
    if ".." in output.parts or not output.is_relative_to(root):
        raise ValueError("screen output escapes workspace")
    metadata_path = inside(root, metadata)
    refuse_payloads(metadata_path)
    inventory, metadata_sha = read_with_digest(metadata_path)
    contexts = freeze_contexts(inventory)
    programs = select_programs(inventory, contexts)
    selection = dict(
        schema=SCHEMA,
        metadata_sha256=metadata_sha,
        contexts=contexts,
        programs=programs,
        gates=GATES,
        axes=list(AXES),
        selection_rule="lexical eligible metadata family/dataset; program digest order",
        qualification_ready=False,
        new_fits=0,
        teaching_admission=False,
    )
    # This durable barrier must precede screen and reference file access.
    selection_sha = immutable(output / "selection.json", selection)
    result = dict(
        schema="parallel.screen_pilot_diagnostic.v2",
        selection_sha256=selection_sha,
        n_contexts=len(contexts),
        n_programs=len(programs),
        qualification_ready=False,
        teaching_admission=False,
        new_fits=0,
        prospective_total_cost=None,
        retrospective_accuracy=None,
        original_reference_verification=False,
        unavailable_requirements=[
            "independent original query/access verifier",
            "funded prospective full-cost screening experiment",
        ],
        shortlist=[],
        rejection_audits=[],
    )
    if screen is not None:
        score_path = inside(root, screen)
        refuse_payloads(score_path)
        scores, score_sha = read_with_digest(score_path)
        shortlist, audits = rejection_audits(selection, selection_sha, scores)
        result.update(
            screen_metadata_sha256=score_sha,
            shortlist=shortlist,
            rejection_audits=audits,
            pairing="explicit original/proxy identities; unverified outcomes",
        )
    result["references"] = reference_status(root, programs)
    immutable(output / ("screen-diagnostic.json" if screen is not None else "reference-status.json"), result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "output", "metadata"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--screen", type=Path)
    args = parser.parse_args()
    result = select(args.root, args.output, metadata=args.metadata, screen=args.screen)
    print(
        canonical(
            {k: result[k] for k in ("schema", "n_contexts", "n_programs", "qualification_ready")}
        ).decode()
    )
