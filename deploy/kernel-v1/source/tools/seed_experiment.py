#!/usr/bin/env python3
"""Frozen seed qualification. All result writes are confined to one experiment.

The six rankings are sealed before the first held-out fit. Failed candidates are
terminal observations; cancelled candidates may resume. This is qualification
within previously used families, never an independent final-test estimate.
"""
import argparse
import collections
import ctypes
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import re
import time
import experiment_runtime as runtime

ARMS = ("current", "larger_pool", "earlier_checkpoint", "shuffled_context", "random", "generic")
SAMPLER = "family_uniform_champion_plus_seven_quantiles_v2"
MASK = (1 << 64) - 1


def read(path):
    return json.loads(Path(path).read_text())


def compact(value):
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def sha(value):
    return hashlib.sha256(value).hexdigest()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


write = runtime.write
inside = runtime.inside


def relative(root, path):
    return str(inside(root, path).relative_to(root))


def mix(x):
    x = (x + 0x9e3779b97f4a7c15) & MASK
    x = ((x ^ (x >> 30)) * 0xbf58476d1ce4e5b9) & MASK
    x = ((x ^ (x >> 27)) * 0x94d049bb133111eb) & MASK
    return x ^ (x >> 31)


def validation_panel(index, world, seed):
    """Exact translation of training::validation_panel, checked against its SHA."""
    families = collections.defaultdict(list)
    for i, group in enumerate(index["groups"]):
        if group["partition"] == "meta_validation":
            if len(group["records"]) < 8:
                raise ValueError("validation group lacks eight distinct comparators")
            families[group["family"]].append(i)
    if not families or world not in (1, 2, 8):
        raise ValueError("unsupported validation panel")
    families = [families[name] for name in sorted(families)]
    seed ^= 0x76616c
    panel = []
    for rank in range(world):
        records = []
        for g in range(rank * 32, (rank + 1) * 32):
            family = (g + mix(seed ^ (g // len(families)))) % len(families)
            groups = families[family]
            h = mix(seed ^ g)
            group = index["groups"][groups[h % len(groups)]]["records"]
            records.append(group[0])
            for band in range(7):
                lo = 1 + band * (len(group) - 1) // 7
                hi = 1 + (band + 1) * (len(group) - 1) // 7
                records.append(group[lo + mix(h ^ ((band + 1) * 7919)) % (hi - lo)])
        panel.append(records)
    return panel


def panel_cuts(index, state):
    if state.get("sampler_id") != SAMPLER:
        raise ValueError("cannot reconstruct unknown selection sampler")
    panel = validation_panel(index, state["world"], state["seed"])
    if sha(compact(panel)) != state["validation"]["panel_sha256"]:
        raise ValueError("reconstructed checkpoint-selection panel SHA mismatch")
    selected = set(x for rank in panel for x in rank)
    contexts = {g["context"] for g in index["groups"] if selected.intersection(g["records"])}
    return panel, [index["contexts"][i] for i in sorted(contexts)]


def choose_views(index, excluded, families=8):
    available = collections.defaultdict(list)
    for row in index["contexts"]:
        if row["partition"] == "meta_validation":
            available[row["family"]].append(row)
    if len(available) != families:
        raise ValueError(f"expected {families} validation families, found {len(available)}")
    result = []
    for family, rows in sorted(available.items()):
        selected = []
        memberships = set()
        for row in sorted(rows, key=lambda r: (r["cut"], r["dataset"])):
            if (row["dataset"], row["cut"]) in excluded:
                continue
            # Source index rows are unique cut references; training SHA is checked
            # separately against the exact prepared membership at freeze time.
            key = (row["prepared_sha256"], row["cut"])
            if key in memberships:
                continue
            memberships.add(key)
            selected.append(row)
            if len(selected) == 2:
                break
        if len(selected) != 2:
            raise ValueError(f"insufficient unselected decision cuts for {family}")
        result.extend(selected)
    return result


def snapshot_file(root, source, destination, expected=None):
    source, destination = inside(root, source), inside(root, destination)
    source_sha = digest(source)
    if expected is not None and source_sha != expected:
        raise ValueError(f"input hash mismatch: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if digest(destination) != source_sha:
            raise ValueError(f"existing frozen artifact differs: {destination}")
    else:
        shutil.copyfile(source, destination)
    if digest(destination) != source_sha or digest(source) != source_sha:
        raise ValueError(f"input changed during freeze: {source}")
    return {"path": relative(root, destination), "sha256": source_sha,
            "source": relative(root, source)}


def freeze(args):
    root = args.root.resolve()
    output = runtime.output_root(root, args.output)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        verify(root, manifest_path)
        return read(manifest_path)
    campaign_path = inside(root, args.campaign)
    campaign = read(campaign_path)
    campaign_output = inside(root, campaign["output"])
    if output == campaign_output or output.is_relative_to(campaign_output):
        raise ValueError("experiment output must be separate from production campaign")
    output.mkdir(parents=True, exist_ok=True)
    artifacts = []
    lineage = []
    artifacts.append(snapshot_file(root, campaign_path, output / "frozen/campaign.json"))
    artifacts.append(snapshot_file(root, campaign_output / "schedule.json", output / "frozen/schedule.json"))
    # Capture the durable selection pointer before following mutable run metadata.
    selection_path = inside(root, args.selection) if args.selection else root / "runs/auxiliary-training.json"
    frozen_selection = output / "frozen/selection.json"
    selection_doc = read(frozen_selection) if frozen_selection.exists() else read(selection_path)
    selection = selection_doc.get("selection", selection_doc.get("best", selection_doc))
    if "checkpoint" not in selection:
        raise ValueError("selection receipt contains no selected checkpoint")
    if not frozen_selection.exists():
        write(frozen_selection, selection_doc)
    write(output / "progress.json", {"phase": "freezing checkpoint, corpus, and selection lineage",
                                     "selected_checkpoint": selection["checkpoint"], "timestamp": int(time.time()),
                                     "completed": 0, "total": 16})
    artifacts.append({"path": relative(root, output / "frozen/selection.json"),
                      "sha256": digest(output / "frozen/selection.json"), "source": relative(root, selection_path)})
    selected = inside(root, selection["checkpoint"])
    if digest(selected / "state.json") != selection["state_sha256"]:
        raise ValueError("durably selected checkpoint state changed")
    initial = inside(root, campaign["initial_corpus"])
    corpus_paths = {digest(p): p.parent for p in [initial / "index.json", *sorted(campaign_output.glob("cycle-*/corpus-*/index.json"))]}
    states = []
    path = selected
    seen = set()
    while True:
        state_path = path / "state.json"
        state_sha = digest(state_path)
        if state_sha in seen:
            raise ValueError("cycle in checkpoint initialization lineage")
        seen.add(state_sha)
        state = read(state_path)
        if state.get("final_tests_opened") is not False:
            raise ValueError("checkpoint has no sealed-final-test binding")
        states.append((path, state))
        artifacts.append(snapshot_file(root, state_path, output / f"frozen/lineage/state-{len(states):03}.json", state_sha))
        initialization = state["initialization"]
        lineage.append({"checkpoint": relative(root, path), "state_sha256": state_sha,
                        "corpus_sha256": state["corpus_sha256"], "initialization": initialization})
        if initialization["kind"] == "fresh":
            break
        path = inside(root, initialization["checkpoint"])
        if digest(path / "state.json") != initialization["state_sha256"]:
            raise ValueError("checkpoint parent identity changed")
    # First-cycle model is the first mandatory train-1 checkpoint, not the
    # ancestor reached after auxiliary checkpoint selection.
    first_receipts = sorted(campaign_output.glob("cycle-*/train-1/latest.json"))
    if not first_receipts:
        raise ValueError("first-cycle checkpoint receipt missing")
    earlier_receipt = read(first_receipts[0])
    earlier = inside(root, earlier_receipt["checkpoint"])
    if digest(earlier / "state.json") != earlier_receipt["state_sha256"]:
        raise ValueError("first-cycle checkpoint state changed")
    if all(p != earlier for p, _ in states):
        states.append((earlier, read(earlier / "state.json")))
    for name, source in (("checkpoint", selected), ("earlier-checkpoint", earlier)):
        state = read(source / "state.json")
        for filename, field in (("state.json", None), ("config.json", "config_sha256"), ("model.ot", "model_sha256")):
            artifacts.append(snapshot_file(root, source / filename, output / "frozen" / name / filename,
                                           state[field] if field else None))
    selected_state = states[0][1]
    corpus = corpus_paths.get(selected_state["corpus_sha256"])
    if corpus is None:
        raise ValueError("selected checkpoint exact corpus is unavailable")
    frozen_corpus = output / "frozen/corpus"
    for name in ("index.json", "vocabulary.json", "records.bin", "labels.jsonl", "provenance.json"):
        artifacts.append(snapshot_file(root, corpus / name, frozen_corpus / name))
    index = read(frozen_corpus / "index.json")
    if (digest(frozen_corpus / "vocabulary.json") != selected_state["vocabulary_sha256"] or
            digest(frozen_corpus / "records.bin") != index["records_sha256"]):
        raise ValueError("frozen corpus payload identity mismatch")
    # Every selection panel in the ancestry is reconstructed. Auxiliary rejected
    # checkpoints use the same corpus/seed/world panel, so deduplicate by that SHA.
    excluded = set()
    panels = []
    panel_hashes = set()
    for path, state in states:
        panel_sha = state["validation"]["panel_sha256"]
        panel_identity = (state["corpus_sha256"], panel_sha)
        if panel_identity in panel_hashes:
            continue
        source = corpus_paths.get(state["corpus_sha256"])
        if source is None:
            raise ValueError(f"lineage corpus missing: {state['corpus_sha256']}")
        source_index = read(source / "index.json")
        panel, rows = panel_cuts(source_index, state)
        excluded.update((r["dataset"], r["cut"]) for r in rows)
        panels.append({"checkpoint": relative(root, path), "corpus_sha256": state["corpus_sha256"],
                       "panel_sha256": panel_sha, "records_by_rank": panel,
                       "cuts": [{k: r[k] for k in ("dataset", "family", "cut")} for r in rows]})
        panel_hashes.add(panel_identity)
        artifacts.append(snapshot_file(root, source / "index.json", output / f"frozen/lineage/corpus-{len(panels):03}.json", state["corpus_sha256"]))
    anchors = []
    for path in sorted(campaign_output.glob("cycle-*/cycle.json")):
        spec = read(path)
        first = spec.get("cut_start", 0)
        last = first + spec["cuts_each_dataset"]
        anchors.append({"path": relative(root, path), "sha256": digest(path), "cut_start": first, "cut_end": last})
        excluded.update((r["dataset"], r["cut"]) for r in index["contexts"] if first <= r["cut"] < last)
        artifacts.append(snapshot_file(root, path, output / "frozen/anchors" / path.parent.name / "cycle.json"))
    if not anchors:
        raise ValueError("campaign anchor history missing")
    views = choose_views(index, excluded)
    bound = {a["path"] for a in artifacts}
    def bind(path, expected):
        path = inside(root, path)
        key = relative(root, path)
        if key not in bound:
            if digest(path) != expected:
                raise ValueError(f"frozen input hash mismatch: {path}")
            artifacts.append({"path": key, "sha256": expected})
            bound.add(key)
    # All training contexts can be selected by the shuffled-context control.
    for row in index["contexts"]:
        bind(row["path"], row["sha256"])
        bind(inside(root, row["prepared"]) / "prepared.json", row["prepared_sha256"])
    frozen_views = []
    memberships = collections.defaultdict(set)
    for row in views:
        prepared = read(inside(root, row["prepared"]) / "prepared.json")
        if prepared.get("storage"):
            bind(prepared["storage"]["source"], prepared["storage"]["source_sha256"])
        cut = next(c for c in prepared["cuts"] if c["index"] == row["cut"])
        member_sha = sha(compact([cut["train_indices"], cut["validation_indices"]]))
        if member_sha in memberships[row["family"]]:
            raise ValueError("decision cuts repeat exact row membership")
        memberships[row["family"]].add(member_sha)
        preprocessing = inside(root, row["prepared"]) / f"cut-{row['cut']}/preprocessing.json"
        bind(preprocessing, cut["preprocessing_sha256"])
        frozen_views.append({k: row[k] for k in ("dataset", "family", "cut", "prepared", "prepared_sha256") } |
                            {"context_path": row["path"], "context_sha256": row["sha256"],
                             "training_sha256": cut["training_sha256"], "validation_sha256": cut["validation_sha256"],
                             "membership_sha256": member_sha, "preprocessing_sha256": cut["preprocessing_sha256"], "target_scale": read(preprocessing)["target_scale"]})
    write(output / "frozen/selection-lineage.json", {"lineage": lineage, "panels": panels, "campaign_anchors": anchors})
    bind(output / "frozen/selection-lineage.json", digest(output / "frozen/selection-lineage.json"))
    write(output / "frozen/decision-cuts.json", {"views": frozen_views, "selection_panels": len(panels),
                                               "lowest_available_unselected_cut_ids": True})
    bind(output / "frozen/decision-cuts.json", digest(output / "frozen/decision-cuts.json"))
    evaluator = inside(root, args.evaluator)
    bind(evaluator, digest(evaluator))
    manifest = {"schema": "parallel.seed_experiment.v2", "root": str(root), "output": relative(root, output),
                "corpus": relative(root, frozen_corpus), "checkpoint": relative(root, output / "frozen/checkpoint"),
                "earlier_checkpoint": relative(root, output / "frozen/earlier-checkpoint"),
                "selected_checkpoint_source": relative(root, selected), "evaluator": relative(root, evaluator),
                "evaluator_sha256": digest(evaluator), "artifacts": artifacts, "views": frozen_views,
                "seed": 20260908, "corpus_records": index["records"], "selection_panels": len(panels),
                "frozen_at": int(time.time()), "qualification": True, "independent_evaluation": False,
                "final_tests_opened": False, "production_champion_writes": False,
                "candidate_timeout_seconds": 30, "worker_memory_gib": 8, "maximum_spare_cpus": 16,
                "downstream_evaluations": 200, "downstream_seconds": 180,
                "meeting_gate": {"best8_ratio_at_most": .98, "minimum_family_wins": 5,
                                 "baselines": ["current", "generic"], "downstream_ratio_at_most": 1.0}}
    manifest["runtime_dependencies"] = executable_dependencies(evaluator)
    manifest["evaluation_policy"] = evaluation_policy(manifest["evaluator_sha256"], manifest["runtime_dependencies"])
    manifest["meeting_gate"] = manifest["evaluation_policy"]["meeting_gate"]
    for view in manifest["views"]:
        view["evaluation_policy_sha256"] = sha(compact(manifest["evaluation_policy"]))
    write(manifest_path, manifest)
    write(output / "progress.json", {"phase": "frozen; awaiting six proposal rankings", "selected_checkpoint": manifest["selected_checkpoint_source"],
                                    "corpus_records": index["records"], "views": len(frozen_views), "timestamp": int(time.time())})
    return manifest


def verify(root, manifest_path):
    manifest = read(manifest_path)
    if manifest["schema"] not in ("parallel.seed_experiment.v1", "parallel.seed_experiment.v2") or manifest.get("production_champion_writes") is not False:
        raise ValueError("not an isolated seed experiment")
    output = runtime.output_root(root, manifest["output"])
    if inside(root, manifest_path) != output / "manifest.json":
        raise ValueError("experiment manifest must live inside its output directory")
    if manifest["schema"] == "parallel.seed_experiment.v2":
        policy = evaluation_policy(manifest["evaluator_sha256"], manifest.get("runtime_dependencies"))
        if manifest.get("evaluation_policy") != policy or manifest.get("meeting_gate") != policy["meeting_gate"]:
            raise ValueError("unsupported or changed evaluation policy")
        for view in manifest["views"]:
            if view.get("evaluation_policy_sha256") != sha(compact(policy)):
                raise ValueError("view evaluation policy binding changed")
        if manifest.get("runtime_dependencies") != executable_dependencies(inside(root, manifest["evaluator"])):
            raise ValueError("evaluator runtime dependency binding changed")
    for view in manifest["views"]:
        if not isinstance(view["dataset"], str) or Path(view["dataset"]).name != view["dataset"] or view["dataset"] in (".", "..") or type(view["cut"]) is not int or view["cut"] < 0:
            raise ValueError("invalid derived view path")
        if not finite_nonnegative(view["target_scale"]) or view["target_scale"] == 0:
            raise ValueError("invalid target scale")
        validate_view(root, view)
    if len({(v["dataset"], v["cut"]) for v in manifest["views"]}) != len(manifest["views"]):
        raise ValueError("duplicate experiment view")
    for artifact in manifest["artifacts"]:
        if digest(inside(root, artifact["path"])) != artifact["sha256"]:
            raise ValueError(f"frozen artifact changed: {artifact['path']}")
    return manifest


def validate_view(root, view):
    path = inside(root, view["prepared"])
    if digest(path / "prepared.json") != view["prepared_sha256"]:
        raise ValueError("prepared view hash changed")
    prepared = read(path / "prepared.json")
    if prepared["dataset"] != view["dataset"] or prepared["task"] != "regression":
        raise ValueError("prepared view dataset or task mismatch")
    cuts = [c for c in prepared["cuts"] if c["index"] == view["cut"]]
    if len(cuts) != 1:
        raise ValueError("prepared view cut missing or duplicated")
    cut = cuts[0]
    for field in ("training_sha256", "validation_sha256"):
        if view[field] != cut[field]:
            raise ValueError("view membership binding changed")
    membership = sha(compact([cut["train_indices"], cut["validation_indices"]]))
    if view.get("membership_sha256", membership) != membership:
        raise ValueError("view membership identity mismatch")
    recipe = path / f"cut-{view['cut']}/preprocessing.json"
    if (digest(recipe) != cut["preprocessing_sha256"] or
            view.get("preprocessing_sha256", cut["preprocessing_sha256"]) != cut["preprocessing_sha256"] or
            read(recipe)["target_scale"] != view["target_scale"]):
        raise ValueError("view preprocessing or target scale changed")
    source = prepared["storage"]
    if digest(inside(root, source["source"])) != source["source_sha256"]:
        raise ValueError("bound source data changed")


def seal_rankings(root, manifest, manifest_path):
    output = runtime.output_root(root, manifest["output"])
    rows, bindings = [], []
    for view in manifest["views"]:
        path = output / f"proposals/{view['dataset']}/cut-{view['cut']}/rankings.json"
        data = read(path)
        if any(data[k] != view[k] for k in ("dataset", "family", "cut")):
            raise ValueError("ranking query identity mismatch")
        if data.get("training_context_only") is not True or any(data.get(field) is not False for field in
                ("retained_teachers_consulted", "heldout_labels_read", "gp_outcomes_read")):
            raise ValueError("rankings do not bind training-only pure prediction")
        rankings = data["rankings"]
        complete_path = path.parent / "complete.json"
        complete = read(complete_path)
        if data.get("manifest_sha256") != digest(manifest_path) or complete.get("manifest_sha256") != digest(manifest_path) or complete.get("rankings_sha256") != digest(path):
            raise ValueError("ranking completion or manifest binding changed")
        if set(rankings) != set(ARMS):
            raise ValueError("all six control rankings must be frozen before evaluation")
        for arm, programs in rankings.items():
            if len(programs) < 64 or len(set(programs)) != len(programs) or not all(isinstance(p, str) and p for p in programs):
                raise ValueError(f"incomplete or duplicate frozen ranking: {arm}")
        if any(set(rankings[arm]) != set(rankings["larger_pool"]) for arm in ("earlier_checkpoint", "shuffled_context", "random")):
            raise ValueError("larger-pool controls do not share the exact candidate set")
        rows.append((view, rankings))
        bindings.append({"path": relative(root, path), "sha256": digest(path)})
    receipt = {"schema": "parallel.seed_rankings_frozen.v1", "manifest_sha256": digest(manifest_path), "rankings": bindings,
               "arms": list(ARMS), "all_rankings_precede_any_evaluation": True, "heldout_labels_read": False}
    path = output / "rankings-frozen.json"
    if path.exists():
        if read(path) != receipt:
            raise ValueError("frozen rankings changed after outcome access")
    else:
        if (output / "evaluations").exists() or (output / "downstream-results").exists():
            raise ValueError("outcomes exist without prior frozen-ranking receipt")
        write(path, receipt)
    return rows


def evaluation_key(view, program, evaluator_sha):
    identity = [view["prepared_sha256"], view["cut"], view["training_sha256"], view["validation_sha256"], evaluator_sha, program]
    if view.get("evaluation_policy_sha256"):
        identity += [view["preprocessing_sha256"], view["membership_sha256"], view["evaluation_policy_sha256"]]
    return sha(compact(identity))


def evaluation_policy(evaluator_sha, dependencies=None):
    return {"schema": "parallel.evaluation_policy.v2", "evaluator_sha256": evaluator_sha, "runtime_dependencies": dependencies or {},
            "seed": 20260907, "downstream_seeds": [20260907, 20260908, 20260909],
            "candidate_timeout_seconds": 30, "worker_memory_gib": 8,
            "downstream_evaluations": 200, "downstream_seconds": 180,
            "environment": runtime.child_env(), "task": "regression",
            "driver_sha256": digest(Path(__file__)), "runtime_sha256": digest(Path(runtime.__file__)),
            "metric": "normalized_rmse_times_frozen_target_scale", "normalized_rmse_floor": 1e-12,
            "primary_failure": "noncompetitive; all-failed required prefix makes cohort incomplete",
            "secondary": "all candidates must succeed", "baseline": "pure_first64",
            "meeting_gate": {"best8_ratio_at_most": .98, "minimum_family_wins": 5,
                             "baselines": ["pure_first64", "generic"], "downstream_ratio_at_most": 1.0}}


def executable_dependencies(executable):
    executable = Path(executable).resolve()
    with executable.open("rb") as stream:
        prefix = stream.read(256)
    if prefix.startswith(b"#!"):
        words = prefix.splitlines()[0][2:].decode().split()
        interpreter = words[-1] if Path(words[0]).name == "env" else words[0]
        executable = Path(shutil.which(interpreter, path=runtime.child_env()["PATH"]) or interpreter).resolve()
    result = subprocess.run(["ldd", str(executable)], env=runtime.child_env(), capture_output=True, text=True)
    if result.returncode and "statically linked" not in result.stdout + result.stderr:
        raise ValueError(f"cannot bind executable runtime dependencies: {executable}: {result.stderr.strip()}")
    if "not found" in result.stdout:
        raise ValueError(f"missing executable runtime dependency: {result.stdout.strip()}")
    paths = {Path(path).resolve() for path in re.findall(r"(/[^\s()]+)", result.stdout)}
    return {str(path): digest(path) for path in sorted(paths | {executable})}


def finite_nonnegative(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def validate_ledgers(launches, ledger, identities, manifest_sha, strict=False):
    serials = set()
    for launch in launches:
        serial = launch.get("attempt_serial")
        if type(serial) is not int or serial <= 0 or serial in serials or launch.get("key") not in identities:
            raise ValueError("invalid or duplicate launch identity")
        if launch.get("manifest_sha256") != manifest_sha or launch.get("status") != "launch_admitted":
            raise ValueError("launch manifest or status mismatch")
        if strict and launch.get("attempt_id") != f"{manifest_sha}:{serial}":
            raise ValueError("launch stable attempt ID mismatch")
        serials.add(serial)
    admitted = {r["attempt_serial"]: r for r in launches}
    finished = set()
    for result in ledger:
        if (result.get("key") not in identities or result.get("status") not in ("complete", "failed", "timeout", "cancelled")
                or not finite_nonnegative(result.get("seconds"))):
            raise ValueError("invalid attempt journal record")
        serial = result.get("attempt_serial")
        if strict and (serial not in admitted or result.get("attempt_id") != admitted[serial]["attempt_id"]
                       or result["key"] != admitted[serial]["key"]):
            raise ValueError("terminal attempt does not match its admitted launch")
        if serial is not None:
            if type(serial) is not int or serial <= 0 or serial in finished:
                raise ValueError("duplicate or invalid completed attempt identity")
            finished.add(serial)


def validate_result(path, view, program, manifest):
    result = read(path / "result.json")
    expected = {"schema": "parallel.seed_candidate_result.v1",
                "key": evaluation_key(view, program, manifest["evaluator_sha256"]),
                "dataset": view["dataset"], "family": view["family"], "cut": view["cut"],
                "program_sha256": sha(program.encode()), "prepared_sha256": view["prepared_sha256"],
                "evaluator_sha256": manifest["evaluator_sha256"], "timeout_seconds": 30, "memory_gib": 8}
    if view.get("evaluation_policy_sha256"):
        expected["evaluation_policy_sha256"] = view["evaluation_policy_sha256"]
    if path.name != expected["key"] or any(result.get(k) != v for k, v in expected.items()):
        raise ValueError("result cache identity mismatch")
    if (path / "program.dsl").read_bytes() != program.encode():
        raise ValueError("cached program changed")
    if result.get("status") not in ("complete", "failed", "timeout", "cancelled"):
        raise ValueError("unsupported cached result status")
    if not finite_nonnegative(result.get("seconds")):
        raise ValueError("invalid cached elapsed time")
    if result["status"] == "complete":
        metrics = read(path / "metrics.json")
        if (digest(path / "metrics.json") != result.get("metric_sha256") or
                not finite_nonnegative(metrics.get("rmse")) or
                not finite_nonnegative(result.get("rmse")) or
                result["rmse"] != metrics["rmse"] * view["target_scale"] or result.get("failure") is not None):
            raise ValueError("cached score does not match bound evaluator metrics")
    elif result.get("rmse") is not None or not isinstance(result.get("failure"), str):
        raise ValueError("failed cached result carries a score or lacks a failure")
    return result


def validate_search(search, observations, view, evaluator_sha):
    seen, semantic, successful = set(), set(), []
    for index, row in enumerate(observations):
        program = row.get("genome")
        if (not isinstance(program, str) or not program or program in seen or
                row.get("evaluation_index") != index or row.get("program_sha256") != sha(program.encode()) or
                row.get("evaluator_sha256") != evaluator_sha or row.get("prepared_sha256") != view["prepared_sha256"] or
                row.get("cut") != view["cut"] or not finite_nonnegative(row.get("runtime_seconds"))):
            raise ValueError("cached downstream observation identity mismatch")
        seen.add(program)
        for field in ("training_sha256", "validation_sha256"):
            if field in view and row.get(field) != view[field]:
                raise ValueError("cached downstream membership identity mismatch")
        metrics = row.get("metrics")
        if row.get("failure") is None:
            if not isinstance(metrics, dict) or not finite_nonnegative(metrics.get("rmse")):
                raise ValueError("invalid downstream evaluator metric")
            scale = row.get("target_scale")
            if not finite_nonnegative(scale) or scale == 0 or scale != view.get("target_scale", scale):
                raise ValueError("invalid downstream target scale")
            signature = metrics.get("semantic_sha256")
            if not isinstance(signature, str) or not signature:
                raise ValueError("missing downstream semantic identity")
            if signature not in semantic:
                successful.append((metrics["rmse"] * scale, program))
                semantic.add(signature)
        elif metrics is not None or not isinstance(row["failure"], str):
            raise ValueError("failed downstream observation carries metrics")
    successful.sort(key=lambda row: row[0])
    best = successful[0] if successful else (None, None)
    if (search.get("best_validation_rmse"), search.get("best_program")) != best:
        raise ValueError("cached downstream winner differs from committed observations")
    if not finite_nonnegative(search.get("wall_seconds")) or len(observations) > 200:
        raise ValueError("invalid downstream cost or evaluation count")


def prefix_outcome(programs, view, results, evaluator_sha, n):
    keys = [evaluation_key(view, program, evaluator_sha) for program in programs[:n]]
    completed = [results[k] for k in keys if k in results and results[k]["status"] != "cancelled"]
    successful = [r["rmse"] for r in completed if r["status"] == "complete"]
    attempts_complete = len(keys) == n and len(completed) == n
    complete = attempts_complete and (n <= 8 or len(successful) == n)
    return {"requested": n, "completed": len(completed), "successful": len(successful),
            "failures": sum(r["status"] != "complete" for r in completed), "complete": complete,
            "attempts_complete": attempts_complete,
            "best_rmse": min(successful) if complete and successful else None,
            "observed_best_rmse": min(successful) if successful else None}


def family_comparison(views, arm, baseline, endpoint, family_total):
    families = collections.defaultdict(list)
    expected = collections.Counter(v["family"] for v in views)
    for view in views:
        a, b = view["arms"][arm][endpoint], view["arms"][baseline][endpoint]
        if a["complete"] and b["complete"] and a["best_rmse"] is not None and b["best_rmse"] is not None:
            scale = view.get("target_scale", 1.0)
            families[view["family"]].append(math.log(max(a["best_rmse"] / scale, 1e-12)) - math.log(max(b["best_rmse"] / scale, 1e-12)))
    ratios = {f: math.exp(sum(values) / len(values)) for f, values in families.items() if len(values) == expected[f]}
    aggregate = math.exp(sum(math.log(r) for r in ratios.values()) / len(ratios)) if ratios else None
    return {"family_balanced_ratio": aggregate, "families_completed": len(ratios), "families_required": family_total,
            "complete": len(ratios) == family_total, "family_ratios": ratios,
            "wins": sum(r < 1 - 1e-9 for r in ratios.values()), "ties": sum(abs(r - 1) <= 1e-9 for r in ratios.values()),
            "losses": sum(r > 1 + 1e-9 for r in ratios.values())}


def report(manifest, rows, results, elapsed=0, physical_attempts=0, downstream=None):
    summaries = []
    logical = 0
    requested_keys = set()
    primary_keys = set()
    logical_by_prefix = {str(n): 0 for n in (1, 8, 16, 64)}
    for view, rankings in rows:
        arms = {}
        for arm, programs in rankings.items():
            arms[arm] = {f"best{n}": prefix_outcome(programs, view, results, manifest["evaluator_sha256"], n) for n in (1, 8, 16, 64)}
            logical += arms[arm]["best64"]["completed"]
            for n in (1, 8, 16, 64):
                logical_by_prefix[str(n)] += arms[arm][f"best{n}"]["completed"]
            requested_keys.update(evaluation_key(view, p, manifest["evaluator_sha256"]) for p in programs[:64])
            primary_keys.update(evaluation_key(view, p, manifest["evaluator_sha256"]) for p in programs[:8])
        summaries.append({"dataset": view["dataset"], "family": view["family"], "cut": view["cut"], "target_scale": view.get("target_scale", 1.0), "arms": arms})
    family_total = len({v["family"] for v, _ in rows})
    comparisons = {f"{arm}_vs_{baseline}": {f"best{n}": family_comparison(summaries, arm, baseline, f"best{n}", family_total) for n in (1, 8, 16, 64)}
                   for arm in ARMS if arm != "generic" for baseline in ("current", "generic") if arm != baseline}
    primary = [comparisons[f"larger_pool_vs_{baseline}"]["best8"] for baseline in ("current", "generic")]
    gate = manifest.get("meeting_gate", {"best8_ratio_at_most": .98, "minimum_family_wins": 5})
    primary_pass = all(c["complete"] and c["family_balanced_ratio"] <= gate["best8_ratio_at_most"] and c["wins"] >= gate["minimum_family_wins"] for c in primary)
    downstream_pass = bool(downstream and downstream.get("complete") and downstream.get("no_aggregate_regression"))
    summary = {"schema": "parallel.seed_comparison_report.v2", "qualification": True, "independent_evaluation": False,
            "final_tests_opened": False, "production_seeder": "unchanged", "production_champion_writes": False,
            "selected_checkpoint": manifest.get("selected_checkpoint_source", manifest.get("checkpoint")),
            "corpus_records": manifest.get("corpus_records"), "views": summaries, "comparisons": comparisons,
            "control_labels": {"current": "pure_first64", "larger_pool": "neural_selected_generated_plus_bank_pool"},
            "normalized_rmse_floor": 1e-12, "endpoint_policy": "Primary top1/8 requires every attempt terminal; failed/timed-out candidates are noncompetitive (infinite RMSE). All-failed prefixes have no finite score. Secondary16/64 requires every candidate successful. Families contribute only after both decision cuts complete.",
            "logical_evaluations": logical_by_prefix["8"], "logical_evaluations_by_prefix": logical_by_prefix,
            "logical_evaluations_available_top64": logical, "physical_fits": sum(r["status"] == "complete" for r in results.values()),
            "top8_physical_fits": sum(k in results and results[k]["status"] == "complete" for k in primary_keys),
            "top8_failures": sum(k in results and results[k]["status"] not in ("complete", "cancelled") for k in primary_keys),
            "top8_unique_attempts": sum(k in results and results[k]["status"] != "cancelled" for k in primary_keys),
            "requested_logical_evaluations_by_prefix": {str(n): len(rows) * len(ARMS) * n for n in (1, 8, 16, 64)},
            "physical_completed_attempts": sum(r["status"] != "cancelled" for r in results.values()),
            "physical_attempts": physical_attempts, "candidate_coverage": {"completed_unique": sum(k in results and results[k]["status"] != "cancelled" for k in requested_keys),
                                                                         "required_unique_top64": len(requested_keys),
                                                                         "completed_unique_top8": sum(k in results and results[k]["status"] != "cancelled" for k in primary_keys),
                                                                         "required_unique_top8": len(primary_keys)},
            "failures": sum(r["status"] not in ("complete", "cancelled") for r in results.values()),
            "candidate_seconds": sum(r.get("seconds", 0) for r in results.values()), "elapsed_seconds": elapsed,
            "primary_gate_passed": primary_pass, "downstream": downstream,
            "positive_meeting_result": primary_pass and downstream_pass,
            "conclusion": "positive qualification; deployment requires separate decision" if primary_pass and downstream_pass else
                          "qualification incomplete" if not all(c["complete"] for c in primary) or (primary_pass and not downstream) else
                          "neutral or negative qualification; retain existing production seeder"}

    for view in summary["views"]:
        view["arms"]["pure_first64"] = view["arms"].pop("current")
    summary["comparisons"] = {key.replace("_vs_current", "_vs_pure_first64").replace("current_vs_", "pure_first64_vs_"): value for key, value in summary["comparisons"].items()}
    return summary


def kill_tree(process):
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=10)


def worker_limits():
    resource.setrlimit(resource.RLIMIT_AS, (8 * 1024 ** 3, 8 * 1024 ** 3))
    # A controller crash must not leave unowned fits running into a GPU handoff.
    ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGKILL, 0, 0, 0)
    if os.getppid() == 1:
        os.kill(os.getpid(), signal.SIGKILL)


def valid_metrics(path):
    try:
        value = read(path)
        rmse = value.get("rmse")
        return finite_nonnegative(rmse)
    except (OSError, ValueError, AttributeError, TypeError):
        return False


def cancellation_trial(destination):
    """Identify interrupted work without changing the native GP journal."""
    journal = destination / "observations.jsonl"
    committed = 0
    if journal.exists():
        with journal.open() as stream:
            for line in stream:
                try:
                    json.loads(line)
                except ValueError:
                    break
                if not line.endswith("\n"):
                    break
                committed += 1
    program = destination / f"candidate-{committed}.dsl"
    return {"completed_observation_rows": committed, "interrupted_evaluation_index": committed,
            "candidate_started": program.exists(), "program_sha256": digest(program) if program.exists() else None,
            "atomic_candidate_metrics_present": (destination / f"candidate-{committed}.json").exists()}


def worker_env(output):
    return runtime.child_env(output)

def compare(args):
    root = args.root.resolve()
    manifest_path = inside(root, args.manifest)
    manifest = read(manifest_path)
    output = runtime.output_root(root, manifest["output"])
    if (manifest.get("schema") not in ("parallel.seed_experiment.v1", "parallel.seed_experiment.v2") or
            manifest.get("production_champion_writes") is not False or manifest_path != output / "manifest.json"):
        raise ValueError("comparison controller requires an isolated experiment manifest")
    # Serialize controllers across cache loading and scheduler waits. The spare
    # CPU lease alone cannot prevent a second controller using a stale queue.
    with runtime.controller(root, output, "comparison"):
        if manifest.get("schema") == "parallel.seed_experiment.v1" and "artifacts" in manifest:
            from verify_experiment import audit
            return audit(root, manifest_path)
        return _compare(args)


def _compare(args):
    from experiment_scheduler import admission_reason, spare_lease
    root = args.root.resolve()
    manifest_path = inside(root, args.manifest)
    manifest = verify(root, manifest_path)
    output = runtime.output_root(root, manifest["output"])
    driver_sha = digest(Path(__file__))
    driver_snapshot = output / "frozen/drivers" / f"seed-experiment-{driver_sha}.py"
    driver_snapshot.parent.mkdir(parents=True, exist_ok=True)
    if not driver_snapshot.exists():
        shutil.copyfile(Path(__file__), driver_snapshot)
    if digest(driver_snapshot) != driver_sha:
        raise ValueError("comparison driver snapshot changed")
    saved_downstream = read(output / "downstream-report.json") if (output / "downstream-report.json").exists() else None
    rows = seal_rankings(root, manifest, manifest_path)
    if not 1 <= args.workers <= 16 or args.prefix not in (8, 16, 64):
        raise ValueError("comparison requires 1..16 workers and prefix 8, 16, or 64")
    results = {}
    identities = {evaluation_key(view, program, manifest["evaluator_sha256"]): (view, program)
                  for view, rankings in rows for programs in rankings.values() for program in programs[:64]}
    launches = runtime.journal(output / "launches.jsonl", repair=True)
    ledger = runtime.journal(output / "attempts.jsonl", repair=True)
    runtime.journal(output / "worker-starts.jsonl", repair=True)
    validate_ledgers(launches, ledger, identities, digest(manifest_path), strict=manifest["schema"].endswith(".v2"))
    for path in (output / "evaluations").iterdir() if (output / "evaluations").exists() else []:
        if not path.is_dir() or path.name not in identities:
            raise ValueError(f"unexpected evaluation cache entry: {path}")
    # Atomic evaluator output is already a completed fit even if the controller
    # died before publishing its result receipt. Recover it without fitting again.
    for metrics_path in (output / "evaluations").glob("*/metrics.json"):
        path = metrics_path.parent
        if (path / "result.json").exists() or path.name not in identities:
            continue
        view, program = identities[path.name]
        if (path / "program.dsl").read_bytes() != program.encode() or not valid_metrics(metrics_path):
            raise ValueError("invalid or incorrectly bound durable evaluator metric")
        metric = read(metrics_path)
        result = {"schema": "parallel.seed_candidate_result.v1", "key": path.name, "status": "complete",
                  "dataset": view["dataset"], "family": view["family"], "cut": view["cut"], "program_sha256": sha(program.encode()),
                  "prepared_sha256": view["prepared_sha256"], "evaluator_sha256": manifest["evaluator_sha256"],
                  "seconds": metric.get("seconds", 0), "elapsed_measurement": "evaluator time only; recovered atomic result after controller interruption",
                  "failure": None, "rmse": metric["rmse"] * view["target_scale"], "metric_sha256": digest(metrics_path),
                  "timeout_seconds": 30, "memory_gib": 8, "timestamp": int(time.time()), "recovered_completed_artifact": True}
        launch = next((r for r in reversed(launches) if r["key"] == path.name), None)
        if launch is None:
            raise ValueError("durable metrics have no admitted attempt identity")
        result["attempt_serial"] = launch["attempt_serial"]
        result["attempt_id"] = launch.get("attempt_id", f"legacy-{launch['attempt_serial']}")
        if view.get("evaluation_policy_sha256"):
            result["evaluation_policy_sha256"] = view["evaluation_policy_sha256"]
        write(path / "result.json", result)
    for path in (output / "evaluations").glob("*/result.json"):
        if path.parent.name not in identities:
            raise ValueError("unexpected evaluation cache entry")
        view, program = identities[path.parent.name]
        result = validate_result(path.parent, view, program, manifest)
        validate_ledgers(launches, [result], identities, digest(manifest_path), strict=manifest["schema"].endswith(".v2"))
        serial = result.get("attempt_serial")
        matching = [r for r in ledger if r.get("attempt_serial") == serial and serial is not None]
        if matching and any(r != result for r in matching):
            raise ValueError("result receipt and attempt journal disagree")
        if not matching and not any(r == result for r in ledger):
            runtime.append(output / "attempts.jsonl", result)
            ledger.append(result)
        results[result["key"]] = result
    committed_serials = {r.get("attempt_serial") for r in ledger}
    for launch in launches:
        if launch["attempt_serial"] in committed_serials:
            continue
        view, program = identities[launch["key"]]
        interrupted = {"schema": "parallel.seed_candidate_result.v1", "key": launch["key"], "status": "cancelled",
            "dataset": view["dataset"], "family": view["family"], "cut": view["cut"], "program_sha256": sha(program.encode()),
            "prepared_sha256": view["prepared_sha256"], "evaluator_sha256": manifest["evaluator_sha256"],
            "seconds": 0, "elapsed_measurement": "unknown: controller disappeared before durable result",
            "elapsed_cost_unknown": True, "failure": "controller_interrupted_before_durable_result", "rmse": None,
            "metric_sha256": None, "timeout_seconds": 30, "memory_gib": 8, "timestamp": launch["timestamp"],
            "attempt_serial": launch["attempt_serial"], "attempt_id": launch.get("attempt_id", f"legacy-{launch['attempt_serial']}")}
        runtime.append(output / "attempts.jsonl", interrupted)
        ledger.append(interrupted)
    tasks = collections.OrderedDict()
    # Fair progress: every view/arm receives its top one before deeper prefixes.
    for rank in range(args.prefix):
        for view, rankings in rows:
            for arm in ARMS:
                program = rankings[arm][rank]
                key = evaluation_key(view, program, manifest["evaluator_sha256"])
                if key not in results or results[key]["status"] == "cancelled":
                    tasks.setdefault(key, (view, program))
    if tasks and manifest["schema"] == "parallel.seed_experiment.v1":
        raise ValueError("v1 experiments are audit-only; freeze a new v2 experiment for new evaluations")
    previous_elapsed = max((read(p).get("elapsed_seconds", 0) for p in
                            (output / "report.json", output / "report-all-success-v1.json") if p.exists()), default=0)
    # The first long-lived controller predates per-invocation receipts. Its
    # cumulative report still preserves wall time, including scheduler waits.
    recorded_invocations = [read(p) for p in (output / "invocations").glob("*.json")]
    primary_sensitivity = output / "report-all-success-v1.json"
    if saved_downstream is not None and primary_sensitivity.exists() and not any(r.get("phase") == "downstream" for r in recorded_invocations):
        primary_elapsed = read(primary_sensitivity)["elapsed_seconds"]
        if previous_elapsed > primary_elapsed:
            write(output / "invocations/legacy-downstream-controller.json", {
                "schema": "parallel.seed_comparison_invocation.v1", "phase": "downstream", "prefix": 8,
                "elapsed_seconds": previous_elapsed - primary_elapsed, "timestamp": int(time.time()),
                "measurement": "Recovered from the original controller cumulative clock minus its preserved primary clock; includes GPU-priority waits",
                "manifest_sha256": digest(manifest_path), "downstream_requested": True})
    begun, deadline = time.monotonic(), time.monotonic() + args.seconds
    attempts_path = output / "attempts.jsonl"
    launches_path = output / "launches.jsonl"
    attempts = max([len(ledger)] + [r.get("attempt_serial", 0) or 0 for r in launches + ledger])
    planned_unique = len({evaluation_key(view, program, manifest["evaluator_sha256"])
                          for view, rankings in rows for programs in rankings.values() for program in programs[:args.prefix]})
    cancellations = 0
    stop = False
    def interrupted(signum, frame):
        nonlocal stop
        stop = True
    old_signals = {s: signal.signal(s, interrupted) for s in (signal.SIGINT, signal.SIGTERM)}
    active = {}
    owned = []
    launch_ids = {}
    comparison_phase = "secondary prefix comparison" if args.prefix > 8 else "primary comparison"
    def publish(reason, phase=None):
        current = report(manifest, rows, results, previous_elapsed + time.monotonic() - begun, attempts, saved_downstream)
        write(output / "report.json", current)
        write(output / "progress.json", {"phase": phase or comparison_phase,
              "reason": reason or ("CPU candidate fits hold the shared spare allocation; binary encoders wait for these CPUs" if active or tasks else "Candidate evaluation complete; shared spare CPUs released"),
              "timestamp": int(time.time()), "gpu_world": 0, "cpu_budget": args.workers,
              "active_workers": len(active), "pending_unique": len(tasks), "cancellations": cancellations,
              "selected_checkpoint": current["selected_checkpoint"], "corpus_records": manifest.get("corpus_records"),
              "physical_fits": current["physical_fits"], "logical_evaluations": current["logical_evaluations"],
              "completed_unique": current["candidate_coverage"]["completed_unique"], "prefix": args.prefix,
              "completed": planned_unique - len(tasks) - len(active), "total": planned_unique})
    def finish(key, status, reason=None):
        nonlocal cancellations
        work = active.pop(key)
        process, started, view, program, log = work
        runtime.retire(owned, process)
        log.close()
        path = output / "evaluations" / key
        try:
            metric = read(path / "metrics.json") if status == "complete" else None
        except (OSError, ValueError, TypeError) as error:
            metric, status, reason = None, "failed", f"invalid evaluator output: {error}"
        rmse = metric.get("rmse") if isinstance(metric, dict) else None
        if status == "complete" and not finite_nonnegative(rmse):
            status, reason = "failed", "invalid RMSE"
        result = {"schema": "parallel.seed_candidate_result.v1", "key": key, "status": status,
                  "dataset": view["dataset"], "family": view["family"], "cut": view["cut"],
                  "program_sha256": sha(program.encode()), "prepared_sha256": view["prepared_sha256"],
                  "evaluator_sha256": manifest["evaluator_sha256"], "seconds": time.monotonic() - started,
                  "failure": reason, "rmse": rmse * view["target_scale"] if status == "complete" else None,
                  "metric_sha256": digest(path / "metrics.json") if metric else None,
                  "timeout_seconds": 30, "memory_gib": 8, "timestamp": int(time.time())}
        result["attempt_serial"] = launch_ids.pop(key, None)
        result["attempt_id"] = f"{digest(manifest_path)}:{result['attempt_serial']}"
        if view.get("evaluation_policy_sha256"):
            result["evaluation_policy_sha256"] = view["evaluation_policy_sha256"]
        write(path / "result.json", result)
        runtime.append(attempts_path, result)
        results[key] = result
        if status == "cancelled":
            cancellations += 1
            tasks[key] = (view, program)
    last_publish = 0
    def cancel_or_harvest(key, reason):
        work = active[key]
        process = work[0]
        metric_path = output / "evaluations" / key / "metrics.json"
        completed = valid_metrics(metric_path)
        if process.poll() is None:
            kill_tree(process)
        # Output may become durable between the first check and killing/reaping.
        if completed or valid_metrics(metric_path):
            finish(key, "complete")
        elif process.returncode == 0:
            finish(key, "failed", "successful worker exit without valid metrics")
        else:
            finish(key, "cancelled", reason)
    try:
        while (tasks or active) and not stop and time.monotonic() < deadline:
            reason = admission_reason(root)
            if reason:
                publish(reason, f"{comparison_phase} paused for mandatory production GPU block")
                time.sleep(min(1, max(0, deadline - time.monotonic())))
                continue
            try:
                with spare_lease(root, "seed-comparison", cpus=args.workers, gpu=False), runtime.owned_ranks(owned):
                    while (tasks or active) and not stop and time.monotonic() < deadline:
                        reason = admission_reason(root)
                        if reason:
                            break
                        while tasks and len(active) < args.workers:
                            key, (view, program) = tasks.popitem(last=False)
                            path = output / "evaluations" / key
                            path.mkdir(parents=True, exist_ok=True)
                            runtime.atomic_bytes(path / "program.dsl", program.encode())
                            # A retried cancellation must not leave its old
                            # result masking a newly committed orphan metric.
                            for filename in ("metrics.json", "result.json"):
                                with runtime.parent_fd(path / filename) as (directory, name):
                                    try:
                                        os.unlink(name, dir_fd=directory)
                                    except FileNotFoundError:
                                        pass
                                    os.fsync(directory)
                            log = (path / "worker.log").open("a")
                            command = [str(inside(root, manifest["evaluator"])), "evaluate-view", "--prepared", str(inside(root, view["prepared"])),
                                       "--cut", str(view["cut"]), "--genome", str(path / "program.dsl"), "--output", str(path / "metrics.json")]
                            attempts += 1
                            launch_ids[key] = attempts
                            runtime.append(launches_path, {"attempt_serial": attempts,
                                "attempt_id": f"{digest(manifest_path)}:{attempts}", "key": key,
                                "program_sha256": sha(program.encode()), "manifest_sha256": digest(manifest_path),
                                "timestamp": int(time.time()), "status": "launch_admitted"})
                            process = runtime.popen(command, stdout=log, stderr=subprocess.STDOUT, env=worker_env(output),
                                                       start_new_session=True, preexec_fn=worker_limits)
                            owned.append(process)
                            active[key] = (process, time.monotonic(), view, program, log)
                            runtime.append(output / "worker-starts.jsonl", {"attempt_serial": attempts, "key": key,
                                "pid": process.pid, "timestamp": int(time.time())})
                        for key, (process, started, view, program, log) in list(active.items()):
                            code = process.poll()
                            if code is not None:
                                finish(key, "complete" if code == 0 and (output / "evaluations" / key / "metrics.json").exists() else "failed", None if code == 0 else f"worker_exit:{code}")
                            elif time.monotonic() - started >= 30:
                                kill_tree(process)
                                if valid_metrics(output / "evaluations" / key / "metrics.json"):
                                    finish(key, "complete")
                                else:
                                    finish(key, "timeout", "candidate_timeout_30_seconds")
                        if time.monotonic() - last_publish >= 5:
                            publish(None)
                            last_publish = time.monotonic()
                        time.sleep(.05)
                    # Handoff terminates only our owned candidates; completed
                    # artifacts survive and interrupted trials reenter the queue.
                    for key, work in list(active.items()):
                        cancel_or_harvest(key, reason or "comparison stopped or deadline")
            except BlockingIOError:
                publish("shared spare CPU allocation owned by binary encoding", f"{comparison_phase} waiting for shared spare CPUs")
                time.sleep(1)
        publish("deadline/interruption" if tasks else None, f"{comparison_phase} paused" if tasks else f"{comparison_phase} complete")
    finally:
        for key, work in list(active.items()):
            cancel_or_harvest(key, "controller interruption")
        for sig, handler in old_signals.items():
            signal.signal(sig, handler)
    downstream = saved_downstream
    if args.downstream and not tasks and not stop and time.monotonic() < deadline:
        downstream = run_downstream(root, manifest, rows, deadline, args.workers)
    final = report(manifest, rows, results, previous_elapsed + time.monotonic() - begun, attempts, downstream)
    final["comparison_driver_sha256"] = driver_sha
    final["comparison_driver_snapshot"] = relative(root, driver_snapshot)
    generation = [read(output / f"proposals/{view['dataset']}/cut-{view['cut']}/rankings.json").get("costs", {}) for view, _ in rows]
    final["proposal_costs"] = {key: sum(c.get(key, 0) for c in generation) for key in ("generation_seconds", "scoring_seconds", "total_seconds")}
    if attempts_path.exists():
        ledger = runtime.journal(attempts_path)
        final["physical_candidate_seconds"] = sum(r.get("seconds", 0) for r in ledger)
        final["attempts_with_unknown_elapsed_cost"] = sum(bool(r.get("elapsed_cost_unknown")) for r in ledger)
        final["cancelled_attempts"] = sum(r["status"] == "cancelled" for r in ledger)
        final["failure_codes"] = dict(collections.Counter(r["failure"] for r in ledger if r["status"] not in ("complete", "cancelled")))
    original_primary = output / "report-all-success-v1.json"
    if original_primary.exists():
        original = read(original_primary)
        final["primary_elapsed_seconds"] = original["elapsed_seconds"]
        final["primary_candidate_seconds"] = original["candidate_seconds"]
    phase = "secondary" if args.prefix > 8 else "downstream" if args.downstream else "primary_or_report_refresh"
    run_receipt = {"schema": "parallel.seed_comparison_invocation.v1", "phase": phase, "prefix": args.prefix,
                   "timestamp": int(time.time()), "elapsed_seconds": time.monotonic() - begun,
                   "manifest_sha256": digest(manifest_path), "physical_attempts_total": attempts,
                   "comparison_driver_sha256": driver_sha,
                   "downstream_requested": args.downstream}
    write(output / "invocations" / f"{time.time_ns()}.json", run_receipt)
    invocations = [read(p) for p in (output / "invocations").glob("*.json")]
    final["secondary_elapsed_seconds"] = sum(r["elapsed_seconds"] for r in invocations if r["phase"] == "secondary")
    final["downstream_invocation_seconds"] = sum(r["elapsed_seconds"] for r in invocations if r["phase"] == "downstream")
    write(output / "report.json", final)
    if args.prefix > 8:
        write(output / "progress.json", {
            "phase": f"{comparison_phase} paused" if tasks else f"{comparison_phase} complete",
            "reason": "Deadline or interruption; completed artifacts preserved for resume" if tasks else
                      "All requested candidate attempts are terminal; failed secondary prefixes remain explicitly incomplete",
            "timestamp": int(time.time()), "gpu_world": 0, "cpu_budget": args.workers,
            "active_workers": 0, "pending_unique": len(tasks), "cancellations": cancellations,
            "selected_checkpoint": final["selected_checkpoint"], "corpus_records": manifest.get("corpus_records"),
            "prefix": args.prefix, "completed": planned_unique - len(tasks), "total": planned_unique})
    return final


def run_downstream(root, manifest, rows, deadline, workers):
    """Run the existing GP with frozen matching populations and an isolated guard."""
    if manifest.get("schema") == "parallel.seed_experiment.v2":
        from qualify_matched_v2 import freeze as freeze_matched, run as run_matched
        output = runtime.output_root(root, manifest["output"])
        destination = output / "matched-v2"
        with runtime.controller(root, destination, "matched"):
            matched = freeze_matched(root, output / "manifest.json", destination, inside(root, manifest["evaluator"]))
            return run_matched(root, matched, max(0, deadline - time.monotonic()), max(1, min(5, workers // 3)))
    stop = False
    def interrupt(signum, frame):
        nonlocal stop
        stop = True
    old = {sig: signal.signal(sig, interrupt) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        return _run_downstream(root, manifest, rows, deadline, workers, lambda: stop)
    finally:
        for sig, handler in old.items():
            signal.signal(sig, handler)


def _run_downstream(root, manifest, rows, deadline, workers, should_stop):
    from experiment_scheduler import admission_reason, spare_lease
    output = runtime.output_root(root, manifest["output"])
    manifest_sha = digest(output / "manifest.json")
    assemblies = read(output / "assemblies-complete.json")
    if assemblies.get("manifest_sha256") != manifest_sha:
        raise ValueError("GP assemblies do not bind frozen experiment")
    bindings = {(entry["dataset"], entry["cut"], entry["arm"]): entry for entry in assemblies["assemblies"]}
    if len(bindings) != len(rows) * 3 or len(bindings) != len(assemblies["assemblies"]):
        raise ValueError("downstream assembly coverage changed")
    vocabulary_sha = digest(inside(root, manifest["corpus"]) / "vocabulary.json")
    tasks, results = collections.OrderedDict(), []
    completed_keys = set()
    last_progress = 0
    def publish(active, reason=None):
        write(output / "progress.json", {"phase": "downstream matched GP comparison", "timestamp": int(time.time()),
              "selected_checkpoint": manifest.get("selected_checkpoint_source", manifest.get("checkpoint")),
              "corpus_records": manifest.get("corpus_records"), "completed": len(results), "total": len(rows) * 3,
              "active_workers": active, "cpu_budget": workers, "gpu_world": 0,
              "reason": reason or "Matched GP fits hold the shared spare CPU allocation; frozen 200-evaluation / 180-second limits apply"})
    def harvest(view, arm, seeds, dest):
        if not (dest / "search.json").exists():
            return False
        search = read(dest / "search.json")
        observations = read(dest / "observations.json")
        if (search.get("seeds_sha256") != digest(seeds) or search.get("vocabulary_sha256") != vocabulary_sha or
                search.get("final_tests_opened") is not False or search.get("completed_evaluations") != len(observations) or
                any(row.get("evaluator_sha256") != manifest["evaluator_sha256"] or row.get("prepared_sha256") != view["prepared_sha256"]
                    or row.get("cut") != view["cut"] for row in observations)):
            raise ValueError("cached downstream search identity mismatch")
        validate_search(search, observations, view, manifest["evaluator_sha256"])
        key = (view["dataset"], view["cut"], arm)
        if key not in completed_keys:
            winner = next((row.get("origin", "unrecorded") for row in observations if row.get("genome") == search.get("best_program")), "unrecorded")
            results.append({"view": view, "arm": arm, "result": search, "winner_origin": winner,
                            "physical_fits": sum(isinstance(row.get("metrics"), dict) for row in observations),
                            "failure_codes": collections.Counter(str(row["failure"]) for row in observations if row.get("failure") is not None),
                            "candidate_seconds": sum(row.get("runtime_seconds", 0) for row in observations)})
            completed_keys.add(key)
        return True
    for view, _ in rows:
        tails = []
        for arm in ("current", "larger_pool", "generic"):
            seeds = output / f"downstream/{view['dataset']}/cut-{view['cut']}/{arm}/seeds.json"
            population = read(seeds)
            seed_sha = digest(seeds)
            binding = bindings.get((view["dataset"], view["cut"], arm), {})
            receipt = read(seeds.parent / "assembly.json")
            ranking_sha = digest(output / f"proposals/{view['dataset']}/cut-{view['cut']}/rankings.json")
            if (binding.get("seeds_sha256") != seed_sha or receipt.get("seeds_sha256") != seed_sha or
                    receipt.get("manifest_sha256") != manifest_sha or receipt.get("rankings_sha256") != ranking_sha):
                raise ValueError("downstream seeds changed after frozen ranking assembly")
            tails.append(population["candidates"][64:])
            if len(population["candidates"]) != 80 or len(tails[-1]) != 16:
                raise ValueError("downstream population must contain 64+10+6 candidates")
            dest = output / f"downstream-results/{view['dataset']}/cut-{view['cut']}/{arm}"
            if not harvest(view, arm, seeds, dest):
                if manifest.get("schema") == "parallel.seed_experiment.v1":
                    raise ValueError("v1 downstream results are audit-only; use qualify_matched_v2 for new blocks")
                tasks[(view["dataset"], view["cut"], arm)] = (view, arm, seeds, dest)
        if any(tail != tails[0] for tail in tails):
            raise ValueError("downstream arms do not share exact retained/exploration tails")
    owned = []
    while tasks and time.monotonic() < deadline and not should_stop():
        if admission_reason(root):
            publish(0, admission_reason(root))
            time.sleep(1)
            continue
        try:
            with spare_lease(root, "seed-comparison-downstream", cpus=workers, gpu=False), runtime.owned_ranks(owned):
                active = {}
                try:
                    while (tasks or active) and time.monotonic() < deadline and not should_stop() and not admission_reason(root):
                        while tasks and len(active) < workers:
                            key, (view, arm, seeds, dest) = tasks.popitem(last=False)
                            if harvest(view, arm, seeds, dest):
                                continue
                            dest.mkdir(parents=True, exist_ok=True)
                            log = (dest / "controller.log").open("a")
                            command = [str(inside(root, manifest["evaluator"])), "search", "--prepared", str(inside(root, view["prepared"])),
                                       "--cut", str(view["cut"]), "--output", str(dest), "--evaluations", "200", "--seconds", "180",
                                       "--qualification", "--seeds", str(seeds), "--vocabulary", str(inside(root, manifest["corpus"]) / "vocabulary.json")]
                            process = runtime.popen(command, stdout=log, stderr=subprocess.STDOUT, env=worker_env(output), start_new_session=True)
                            owned.append(process)
                            active[key] = (process, view, arm, seeds, dest, log)
                        for key, (process, view, arm, seeds, dest, log) in list(active.items()):
                            if process.poll() is not None:
                                runtime.retire(owned, process)
                                log.close()
                                active.pop(key)
                                if not harvest(view, arm, seeds, dest):
                                    write(dest / "failed.json", {"exit_code": process.returncode, "status": "failed"})
                        if time.monotonic() - last_progress >= 5:
                            publish(len(active))
                            last_progress = time.monotonic()
                        time.sleep(.1)
                finally:
                    for key, (process, view, arm, seeds, dest, log) in active.items():
                        kill_tree(process)
                        log.close()
                        if harvest(view, arm, seeds, dest):
                            continue
                        cancelled = read(dest / "cancellation.json").get("cancellations", 0) if (dest / "cancellation.json").exists() else 0
                        write(dest / "cancellation.json", {"status": "cancelled", "reason": admission_reason(root) or "deadline/interruption",
                                                           "resumable": True, "cancellations": cancelled + 1, "timestamp": int(time.time()),
                                                           "interrupted_trial": cancellation_trial(dest)})
                        tasks[key] = (view, arm, seeds, dest)
        except BlockingIOError:
            publish(0, "Shared spare CPU allocation is owned by binary encoding")
            time.sleep(1)
    by_view = collections.defaultdict(dict)
    winner_origins = collections.defaultdict(dict)
    for item in results:
        key = (item["view"]["dataset"], item["view"]["cut"])
        by_view[key][item["arm"]] = item["result"]
        winner_origins[key][item["arm"]] = item["winner_origin"]
    family_logs = {arm: collections.defaultdict(list) for arm in ("current", "generic")}
    expected = collections.Counter(view["family"] for view, _ in rows)
    for view, _ in rows:
        arms = by_view[(view["dataset"], view["cut"])]
        for baseline in family_logs:
            a, b = arms.get("larger_pool", {}).get("best_validation_rmse"), arms.get(baseline, {}).get("best_validation_rmse")
            if a is not None and b is not None and a > 0 and b > 0:
                family_logs[baseline][view["family"]].append(math.log(a / b))
    comparisons = {}
    for baseline, families in family_logs.items():
        family_ratios = {f: math.exp(sum(v) / len(v)) for f, v in families.items() if len(v) == expected[f]}
        comparisons[baseline] = {"family_ratios": family_ratios, "complete": len(family_ratios) == len(expected),
                                 "family_balanced_ratio": math.exp(sum(math.log(v) for v in family_ratios.values()) / len(family_ratios)) if family_ratios else None,
                                 "families_completed": len(family_ratios), "families_required": len(expected),
                                 "wins": sum(value < 1 - 1e-9 for value in family_ratios.values()),
                                 "ties": sum(abs(value - 1) <= 1e-9 for value in family_ratios.values()),
                                 "losses": sum(value > 1 + 1e-9 for value in family_ratios.values())}
    complete = len(results) == len(rows) * 3 and all(c["complete"] for c in comparisons.values())
    direct_improvements = {}
    for baseline in ("current", "generic"):
        direct_improvements[baseline] = sum(
            winner_origins[key].get("larger_pool") == "neural_policy"
            and arms.get("larger_pool", {}).get("best_validation_rmse") is not None
            and arms.get(baseline, {}).get("best_validation_rmse") is not None
            and arms["larger_pool"]["best_validation_rmse"] < arms[baseline]["best_validation_rmse"]
            for key, arms in by_view.items())
    receipt = {"complete": complete, "completed_views": len(results), "required_views": len(rows) * 3,
               "comparisons": comparisons, "no_aggregate_regression": complete and all(c["family_balanced_ratio"] <= 1 for c in comparisons.values()),
               "physical_evaluations": sum(i["result"]["completed_evaluations"] for i in results),
               "physical_fits": sum(i["physical_fits"] for i in results),
               "failed_evaluations": sum(sum(i["failure_codes"].values()) for i in results),
               "failure_codes": dict(sum((i["failure_codes"] for i in results), collections.Counter())),
               "candidate_seconds_observed": sum(i["candidate_seconds"] for i in results),
               "elapsed_search_seconds": sum(i["result"]["wall_seconds"] for i in results),
               "resumed_searches": sum(bool(i["result"].get("resumed")) for i in results),
               "search_clock_accuracy": dict(collections.Counter(i["result"].get("search_clock", {}).get("accuracy", "unknown") for i in results)),
               "resume_unobserved_allowance_seconds": sum(i["result"].get("search_clock", {}).get("resume_unobserved_allowance_seconds", 0) for i in results),
               "elapsed_search_seconds_policy": "Native conservative active search clock; resumed searches include the evaluator's recorded allowance for interrupted work",
               "cancellations": sum(read(p).get("cancellations", 1) for p in (output / "downstream-results").glob("*/cut-*/*/cancellation.json")),
               "evaluation_limit": 200, "seconds_limit": 180, "candidate_timeout_seconds": 30,
               "rng_seed": 20260907, "shared_retained_tails_verified": True, "production_champion_writes": False}
    receipt["winner_origins_by_arm"] = {arm: dict(collections.Counter(item["winner_origin"] for item in results if item["arm"] == arm))
                                        for arm in ("current", "larger_pool", "generic")}
    receipt["direct_neural_selected_winner_improvements"] = direct_improvements
    receipt["attribution_policy"] = "Direct improvement requires a winning program from the frozen neural-selected first64 and lower view RMSE than the named matched control; retained teachers and evolved descendants are not attributed."
    write(output / "downstream-report.json", receipt)
    publish(0, "Downstream comparison complete; shared spare CPUs released" if complete else "Downstream comparison incomplete; completed searches preserved for resume")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("freeze")
    command.add_argument("--campaign", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--evaluator", type=Path, required=True)
    command.add_argument("--selection", type=Path)
    command = commands.add_parser("compare")
    command.add_argument("--manifest", type=Path, required=True)
    command.add_argument("--workers", type=int, default=16)
    command.add_argument("--prefix", type=int, default=8)
    command.add_argument("--seconds", type=int, default=3600)
    command.add_argument("--downstream", action="store_true")
    args = parser.parse_args()
    result = freeze(args) if args.command == "freeze" else compare(args)
    print(json.dumps({k: result[k] for k in ("schema", "output", "selected_checkpoint_source", "selection_panels", "conclusion", "positive_meeting_result", "physical_fits", "logical_evaluations") if k in result}, indent=2))


if __name__ == "__main__":
    runtime.entrypoint(main, "--output" if "freeze" in __import__("sys").argv else "--manifest")
