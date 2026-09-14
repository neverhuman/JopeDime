"""Inspect replay evidence without manufacturing discovery or training authority."""
import argparse
from collections import defaultdict
from pathlib import Path

from scientific_metadata import (canonical, context, digest, finite, immutable, metadata_bytes,
                                 program, read_with_digest, reference, sha256_value)

SCHEMA = "parallel.replay_plan.v2"
TEACHER_ORIGINS = ("incumbent_cut_teacher", "retained_cut_teacher")
GENERIC_ORIGINS = ("generic_comparison",)


def file_sha(path):
    return metadata_bytes(path)[1]


def refuse_payloads(path):
    if "data" in Path(path).parts:
        raise ValueError("refusing to open data/ payloads")
    return True


def original_rmse(row):
    metrics = row.get("metrics")
    if not isinstance(metrics, dict):
        return None
    score = metrics.get("validation_rmse", metrics.get("rmse"))
    if not finite(score):
        return None
    if metrics.get("units") == "original_target":
        return score
    scale = row.get("target_scale")
    if metrics.get("units") == "cut_normalized_target" and finite(scale) and scale > 0:
        return score * scale if finite(score * scale) else None
    return None


def is_teacher(row):
    return row.get("origin") in TEACHER_ORIGINS


def is_generic(row):
    return row.get("origin") in GENERIC_ORIGINS


def load_observations(root):
    root = Path(root)
    paths = sorted(root.rglob("observations.json"))
    sources, rows = [], []
    for path in paths:
        refuse_payloads(path)
        # Unreadable, malformed and nonfinite sources fail explicitly. No silent census loss.
        data, source_sha = read_with_digest(path)
        sources.append(dict(path=str(path), sha256=source_sha))
        if not isinstance(data, list):
            raise ValueError("original observation source is not a list: " + str(path))
        for number, row in enumerate(data):
            if not isinstance(row, dict):
                raise ValueError("observation row is not an object")
            if any(key.startswith("_") for key in row):
                raise ValueError("observation cannot supply internal source bindings")
            rows.append(dict(row, _source=str(path), _source_sha256=source_sha, _row_index=number))
    return sources, rows


def group_key(row):
    return digest(context(row))


def _bound(row):
    bound = context(row)
    p = program(row)
    evaluation = row.get("evaluation_identity_sha256")
    if not sha256_value(evaluation):
        raise ValueError("original computation identity missing")
    if row.get("independent_evaluation") is not None and row.get("independent_evaluation") is not False:
        raise ValueError("caller cannot declare independent evaluation")
    if not sha256_value(row.get("_source_sha256")) or type(row.get("_row_index")) is not int:
        raise ValueError("original source coordinates missing")
    return bound, p, evaluation


def beating_set(rows):
    groups = defaultdict(list)
    unavailable, failed, reused = [], [], []
    identities, conflicts = {}, set()
    for row in rows:
        coordinates = {k: row.get(k) for k in ("_source", "_source_sha256", "_row_index")}
        try:
            bound, p, evaluation = _bound(row)
            score = original_rmse(row)
            # A computation cannot be relabeled as a teacher or attached to another
            # original query by changing fields omitted from a partial fingerprint.
            payload = canonical({key: value for key, value in row.items() if not key.startswith("_")})
            if evaluation in identities and identities[evaluation] != payload:
                conflicts.add(evaluation)
            identities[evaluation] = payload
            if row.get("failure") is not None or score is None:
                failed.append(dict(coordinates, evaluation_identity_sha256=evaluation,
                                   failure=row.get("failure"), score_available=score is not None))
                continue
            if not is_teacher(row) and row.get("evaluation_origin") != "execution":
                reused.append(dict(coordinates, evaluation_identity_sha256=evaluation))
                continue
            groups[digest(bound)].append(dict(row, _context=bound, _score=score, _program=p))
        except (ValueError, TypeError, KeyError) as error:
            unavailable.append(dict(coordinates, reason=str(error)))
    beating, generic_beating = [], []
    teacher_groups = generic_groups = 0
    for _, items in sorted(groups.items()):
        items = [r for r in items if r["evaluation_identity_sha256"] not in conflicts]
        # Retrospective descriptive comparison only; no pre-outcome incumbent is implied.
        seen = {}
        program_conflicts = set()
        for row in items:
            key = row["program_sha256"]
            if key in seen and (seen[key]["_score"], seen[key].get("origin")) != (row["_score"], row.get("origin")):
                program_conflicts.add(key)
            seen.setdefault(key, row)
        clean = [row for key, row in seen.items() if key not in program_conflicts]
        teachers = [r["_score"] for r in clean if is_teacher(r)]
        generics = [r["_score"] for r in clean if is_generic(r)]
        teacher_groups += bool(teachers)
        generic_groups += bool(generics)
        for key, row in sorted(seen.items()):
            if key in program_conflicts or is_teacher(row):
                continue
            entry = dict(context=row["_context"], **row["_program"], rmse=row["_score"],
                         units="original_target", origin=row.get("origin"),
                         evaluation_identity_sha256=row["evaluation_identity_sha256"],
                         source_path=row["_source"], source_sha256=row["_source_sha256"],
                         row_index=row["_row_index"], original_query=row.get("original_query"),
                         independent_evaluation=False, original_verified=False)
            if teachers and row["_score"] < min(teachers):
                beating.append(entry)
            if generics and not is_generic(row) and row["_score"] < min(generics):
                generic_beating.append(entry)
        if program_conflicts:
            unavailable.append(dict(reason="conflicting repeated program outcomes",
                                    program_sha256=sorted(program_conflicts)))
    if conflicts:
        unavailable.append(dict(reason="conflicting original computation identities",
                                evaluation_identity_sha256=sorted(conflicts)))
    return dict(inspected_groups=len(groups), teacher_groups=teacher_groups,
                generic_groups=generic_groups, diagnostic_teacher_beating=beating,
                diagnostic_generic_beating=generic_beating, unavailable=unavailable,
                failed_observations=failed, reuse_or_unknown_origins=reused)


def argv(*_args, **_kwargs):
    raise ValueError("recover original selected condition/seed manifests; invented reset ablation retired")


def plan(root, dest=None, corpus=None, baseline=None, checkpoint=None, block_identity="0",
         selected_experiment=None):
    sources, rows = load_observations(root)
    found = beating_set(rows)
    originals = []
    for candidate in found["diagnostic_teacher_beating"]:
        ref = candidate.get("original_query")
        reason = "original query reference unavailable"
        if ref is not None:
            try:
                value = reference(Path(root).resolve(), ref)
                if (not isinstance(value, dict) or value.get("context") != candidate["context"]
                        or value.get("program_sha256") != candidate["program_sha256"]
                        or value.get("evaluation_identity_sha256") != candidate["evaluation_identity_sha256"]):
                    raise ValueError("original query identity conflict")
                reason = "independent original query/access verifier unavailable; hashes are not execution"
            except (OSError, ValueError, TypeError, KeyError) as error:
                reason = str(error)
        originals.append(dict(program_sha256=candidate["program_sha256"], original_query=ref,
                              verified=False, unavailable_reason=reason))
    selected = None
    if selected_experiment is not None:
        # Preserve the recovered original manifest as evidence; never invent its conditions.
        _, selected_sha = read_with_digest(selected_experiment)
        selected = dict(path=str(selected_experiment), sha256=selected_sha)
    result = dict(schema=SCHEMA, observation_files=len(sources), observation_rows=len(rows),
                  source_files=sources,
                  teacher_beating_n=None, generic_beating_n=None,
                  diagnostic_teacher_beating_n=len(found["diagnostic_teacher_beating"]),
                  diagnostic_generic_beating_n=len(found["diagnostic_generic_beating"]),
                  diagnostics=found, original_verification=originals,
                  selected_experiment=selected, replay_argv=None,
                  requested_inputs={k: str(v) if v is not None else None for k, v in
                                    (("corpus", corpus), ("baseline", baseline), ("checkpoint", checkpoint))},
                  unavailable_requirements=["original admitted computation and query/access verification",
                                            "incumbent snapshot frozen before each original discovery",
                                            "original selected condition/seed manifest qualification"],
                  importance_sampling="unavailable; rank_log_weight and legacy policy_logprob are a ranking weight, not behavior propensity",
                  freeze_ran=False, train_launched=False, materialize_view=False,
                  launch_train_allowed=False, serving_written=False, shuffled_minted=False,
                  qualification_ready=False, independent_evaluation=False)
    if dest is not None:
        immutable(Path(dest) / "plan.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    for name in ("corpus", "baseline", "checkpoint", "selected-experiment"):
        parser.add_argument("--" + name, type=Path)
    args = parser.parse_args()
    result = plan(args.observations, args.output, corpus=args.corpus, baseline=args.baseline,
                  checkpoint=args.checkpoint, selected_experiment=args.selected_experiment)
    print(canonical({k: result[k] for k in ("schema", "teacher_beating_n", "train_launched",
                                          "qualification_ready")}).decode())
