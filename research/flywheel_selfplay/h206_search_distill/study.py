"""Fixed H206 study, separate GPU/CPU phases and durable incremental proofs."""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import io
import json
import os
import signal
from pathlib import Path

import numpy as np
import torch

from .contracts import Context, ObjectRef, Scope, canonical, digest, load_context, sha
from .executor import evaluate_fixed, restore_verified_state
from .model import ModelCache, PinnedR882Factory
from .native import OutcomeCache, RustS1Evaluator, S3Store, collect_teacher
from .student import tensor_hash
from .supervisor import ReservedPhaseBudget, atomic_record, verify_merged_source

ARMS = {"CTRL": 1, "ACC4": 16, "ACC16": 64, "OC15": 256}
SEEDS = (206101, 206102, 206103)
FOLD_SEEDS = (206141, 206142, 206143)
SELECTOR_SHA = "707c4903a77d48ffc8ae14c820a626df5c2858d3d24df35db3f9718d120bafc4"
CALIBRATION_SHA = "3e6545d371334c505023d4d0593dbf0abf10ef220785c8f30e49f5a5aec5e91d"


def registered_cells(rows):
    """Pure metadata plan: 3 seeds x 5 folds x 4 arms; no input payload reads."""
    ids = [r["context_id"] for r in rows]
    families = sorted({r["family_id"] for r in rows})
    if len(rows) != 1219 or len(set(ids)) != 1219 or len(families) != 205:
        raise ValueError("original H2061219-context/205-family population required")
    cells = []
    for seed, fold_seed in zip(SEEDS, FOLD_SEEDS, strict=True):
        groups = [
            set(g.tolist()) for g in np.array_split(np.random.default_rng(fold_seed).permutation(families), 5)
        ]
        seen = {"A": [], "B": []}
        for fold in range(5):
            role_families = {"A": groups[fold], "B": groups[(fold + 1) % 5]}
            role_families["FIT"] = set(families) - role_families["A"] - role_families["B"]
            role_ids = {
                role: [r["context_id"] for r in rows if r["family_id"] in fs]
                for role, fs in role_families.items()
            }
            if set(role_ids["FIT"]) & (set(role_ids["B"]) | set(role_ids["A"])):
                raise ValueError("FIT/B/A leakage")
            if set(role_ids["A"]) & set(role_ids["B"]):
                raise ValueError("B/A leakage")
            for role in seen:
                seen[role].extend(role_ids[role])
            for arm, k in ARMS.items():
                cells.append(
                    {
                        "cell_id": f"seed-{seed}/fold-{fold}/{arm}",
                        "seed": seed,
                        "fold_seed": fold_seed,
                        "fold": fold,
                        "arm": arm,
                        "k": k,
                        "context_ids": role_ids,
                        "family_ids": {k: sorted(v) for k, v in role_families.items()},
                    }
                )
        if any(sorted(v) != sorted(ids) for v in seen.values()):
            raise ValueError("pooled B/A must cover each context exactly once per seed")
    jaccards = {}
    for seed in SEEDS:
        by_fold = [next(c for c in cells if c["seed"] == seed and c["fold"] == f) for f in range(5)]
        jaccards[str(seed)] = {
            role: [
                [
                    len(set(a["family_ids"][role]) & set(b["family_ids"][role]))
                    / len(set(a["family_ids"][role]) | set(b["family_ids"][role]))
                    for b in by_fold
                ]
                for a in by_fold
            ]
            for role in ("FIT", "B", "A")
        }
    return {
        "schema": "h206-fixed-folds-v1",
        "cells": cells,
        "jaccard": jaccards,
        "rows_sha256": digest(rows),
        "seeds": list(SEEDS),
        "rounds": 3,
        "updates_per_round": 600,
    }


def load_selector(repo):
    """Private instance of the unchanged H177 function, calibrated by H178."""
    path = Path(repo) / "research/flywheel_selfplay/h177_pooled_selector/analysis.py"
    receipt = Path(repo) / "research/flywheel_selfplay/h178_pooled_calibration/h178_receipt.json"
    if sha(path.read_bytes()) != SELECTOR_SHA or sha(receipt.read_bytes()) != CALIBRATION_SHA:
        raise ValueError("H177/H178 selector/calibration source identity mismatch")
    spec = importlib.util.spec_from_file_location("_h206_private_h177_selector", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.PRESPECIFIED_CANDIDATE_STEP = {arm: 1800 for arm in ARMS}
    return module


def panel_statistics(repo, rows, baseline, panels):
    selector = load_selector(repo)
    decisions = selector.pooled_b_select(rows, baseline, panels, seed=206150)
    families = [r["family_id"] for r in rows]
    base = np.asarray(baseline, dtype=np.float64)
    if base.ndim == 1:
        base = np.broadcast_to(base, (3, len(rows)))
    for arm in ARMS:
        gains = np.stack([selector.family_matrix(v, families)[1] for v in panels[arm]]) - (
            np.stack([selector.family_matrix(v, families)[1] for v in base])
        )
        harm = int((gains.mean(axis=0) < 0).sum())
        decisions[arm]["harm_families"] = harm
        decisions[arm]["harm_fraction"] = harm / 205
        decisions[arm]["admitted_with_harm"] = bool(decisions[arm]["admitted"] and harm / 205 <= 0.1)
    return decisions


def pooled_select(repo, rows, baseline, panels):
    decisions = panel_statistics(repo, rows, baseline, panels)
    admitted = [a for a in ARMS if decisions[a]["admitted_with_harm"]]
    selected = (
        max(admitted, key=lambda a: (decisions[a]["mean_gain"], -list(ARMS).index(a))) if admitted else "r882"
    )
    return {
        "selected_arm": selected,
        "decisions": decisions,
        "bootstrap_seed": 206150,
        "selector_source_sha256": SELECTOR_SHA,
        "calibration_sha256": CALIBRATION_SHA,
    }


def checkpoint(learner, store, name, *, local=None):
    out = io.BytesIO()
    torch.save(
        {
            "model": learner.model.state_dict(),
            "optimizer": learner.optimizer.state_dict(),
            "updates": learner.updates,
            "examples": learner.examples,
            "scope_sha256": learner.scope_sha256,
            "gradient_squares": learner.gradient_squares,
        },
        out,
    )
    data = out.getvalue()
    if local is not None:
        local = Path(local)
        local.parent.mkdir(parents=True, exist_ok=True)
        with local.open("xb") as saved:
            saved.write(data)
            saved.flush()
            os.fsync(saved.fileno())
    ref = store.put(name, data)
    return {
        "checkpoint": dataclasses.asdict(ref),
        "state_sha256": learner.state_hash(),
        "updates": learner.updates,
        "examples": learner.examples,
    }


class StudyInputs:
    """Consume per-role, per-cell qualified templates; bind the measured actor late.

    A template is an independently generated, version-bound 256-program slate,
    not an old pool resample. Actor ranking is performed anew in each phase.
    Its role, generator recipe, original-TRAIN derivation and tensor closure must
    already be qualified. No dataset expansion or producer fallback exists here.
    """

    def __init__(self, task, store):
        self.task, self.store = task, store
        roles, exclusion = (ObjectRef(**task["scope"][name]) for name in ("roles", "exclusion"))
        self.scope = Scope(
            roles.read(store),
            exclusion.read(store),
            roles_sha256=roles.sha256,
            exclusion_sha256=exclusion.sha256,
        )

    def acquire(self, role, actor):
        task = self.task
        if role not in ("FIT", "B", "A"):
            raise ValueError("registered role required")
        if role in ("B", "A") and (task["round"] != 3 or task.get("terminal_updates") != 1800):
            raise ValueError("evaluation before fixed terminal endpoint")
        if role == "A":
            lock = json.loads(ObjectRef(**task["b_lock"]).read(self.store))
            if lock.get("status") != "B_LOCKED" or actor not in lock["checkpoint_sha256"]:
                raise ValueError("A acquisition before checkpoint-bound B lock")
        for cid in task["context_ids"][role]:
            self.scope.admit_metadata(cid, role)
        doc = json.loads(ObjectRef(**task["templates"][str(task["round"])][role]).read(self.store))
        if doc["schema"] != "h206-actor-independent-slate-templates-v1" or doc["role"] != role:
            raise ValueError("qualified per-role slate templates required")
        if doc["study_sha256"] != task["admission"]["study_sha256"] or doc["cell_id"] != task["cell_id"]:
            raise ValueError("template cell/study identity mismatch")
        if {r["context_id"] for r in doc["contexts"]} != set(task["context_ids"][role]):
            raise ValueError("template population differs from fixed fold")
        loaded = []
        by_id = {r["context_id"]: r for r in doc["contexts"]}
        for cid in task["context_ids"][role]:
            row = by_id[cid]
            value = dict(row)
            generation_ref = ObjectRef(**value.pop("generation_receipt"))
            value["actor_sha256"] = actor
            context = Context.from_dict(value)
            self.scope.verify(context, role)  # BEFORE generator/tensor/TRAIN bodies.
            generation = json.loads(generation_ref.read(self.store))
            if (
                generation.get("fixture") is not False
                or generation["status"] != "PASS"
                or generation["proposal_source_sha256"] != context.proposal_source_sha256
                or generation["programs_sha256"] != digest(list(context.programs))
                or generation["seed"] != context.seed
                or not 256 <= generation["attempts"] <= 4096
                or len(context.programs) != 256
                or generation["original_train"] != dataclasses.asdict(context.train)
                or generation["cut_sha256"] != context.cut_sha256
            ):
                raise ValueError("unqualified or disconnected actual slate generation")
            loaded.append(load_context(context, self.scope, self.store, role=role))
        if len(loaded) != len(task["context_ids"][role]):
            raise ValueError("duplicate/missing role context")
        return loaded


def preserve_native_scratch(teacher, loaded, native, cache, store, prefix):
    """Deduplicate/archive exact owned context files; restore-check before unlink.

    No source, models, other contexts or foreign cache trees are eligible.
    """
    import hashlib
    import tarfile

    doc = json.loads(teacher.read(store))
    if doc["context_identity"] != loaded.context.identity:
        raise ValueError("scratch custody context mismatch")
    paths = []
    roots = []
    for record in doc["records"]:
        job = Path(record["scratch_job"]).resolve()
        expected = (
            digest({"context": loaded.context.identity, "program": loaded.context.programs[record["index"]]})
            + "-"
        )
        if (
            not job.is_relative_to(native.scratch.resolve())
            or job == native.scratch.resolve()
            or not job.name.startswith(expected)
            or job.is_symlink()
        ):
            raise ValueError("unowned native scratch path")
        if job.exists():
            roots.append(job)
            for file in job.iterdir():
                if file.is_symlink() or not file.is_file():
                    raise ValueError("unknown native scratch member retained")
                paths.append(file)
    cache_path = cache.root / (digest(doc["identity"]) + ".json")
    if cache_path.exists():
        if cache_path.is_symlink():
            raise ValueError("cache custody cannot follow symlinks")
        cached = json.loads(cache_path.read_text())
        if cached["identity"] != doc["identity"] or cached["records"] != doc["records"]:
            raise ValueError("cache/teacher custody mismatch")
        paths.append(cache_path)
    members, unique, identity = {}, {}, {}
    for path in paths:
        body_hash = sha(path.read_bytes())
        st = path.stat()
        identity[str(path)] = (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns)
        members[str(path)] = {"sha256": body_hash, "bytes": st.st_size}
        unique.setdefault(body_hash, path)
    if sum(path.stat().st_size for path in unique.values()) > 2**30:
        raise RuntimeError("bounded native custody pack exceeded; original files retained")
    packed = io.BytesIO()
    with tarfile.open(fileobj=packed, mode="w:gz") as archive:
        manifest = canonical(
            {
                "schema": "h206-owned-native-pack-v1",
                "teacher": dataclasses.asdict(teacher),
                "members": members,
            }
        )
        info = tarfile.TarInfo("MANIFEST.json")
        info.size, info.mtime = len(manifest), 0
        archive.addfile(info, io.BytesIO(manifest))
        for body_hash, path in unique.items():
            info = tarfile.TarInfo("sha256/" + body_hash)
            info.size, info.mtime = path.stat().st_size, 0
            with path.open("rb") as source:
                archive.addfile(info, source)
    ref = store.put(prefix + "/owned-native-pack.tar.gz", packed.getvalue())
    restored = ref.read(store)  # Independent actual exact-VersionId restoration.
    verified = {}
    with tarfile.open(fileobj=io.BytesIO(restored), mode="r:gz") as archive:
        manifest = json.load(archive.extractfile("MANIFEST.json"))
        if manifest["members"] != members:
            raise ValueError("native pack manifest restoration mismatch")
        for body_hash in unique:
            h = hashlib.sha256()
            file = archive.extractfile("sha256/" + body_hash)
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                h.update(chunk)
            if h.hexdigest() != body_hash:
                raise ValueError("native pack actual member restoration mismatch")
            verified[body_hash] = True
    proof = {
        "pack": dataclasses.asdict(ref),
        "teacher": dataclasses.asdict(teacher),
        "members": members,
        "unique_bodies": len(unique),
        "all_restored_sha256_match": True,
    }
    # Publish recovery route BEFORE deleting only the unchanged listed duplicates.
    proof_ref = store.put(prefix + "/owned-native-pack-restoration.json", canonical(proof))
    for path in paths:
        st = path.stat()
        if (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns) != identity[str(path)] or (
            sha(path.read_bytes()) != members[str(path)]["sha256"]
        ):
            raise RuntimeError("scratch changed after custody; retained")
    for path in paths:
        path.unlink()
    for root in set(roots):
        root.rmdir()  # Fails safely if an unlisted member appeared.
    return proof_ref


class PhaseRuntime:
    """Real model/native/learner operations shared by CPU controls and GPU workers."""

    def __init__(self, store, budget, *, scientific=True, journal=None):
        self.store, self.budget, self.scientific = store, budget, scientific
        self.journal = Path(journal).resolve() if journal is not None else None
        if scientific and self.journal is None:
            raise ValueError("scientific phases require an owned durable local journal")
        if self.journal is not None and self.journal.is_relative_to("/tmp"):
            raise ValueError("incremental journal outside /tmp required")
        self.last_checkpoint = None
        self.progress = []

    def propose(self, learner, rows, prefix):
        if not rows:
            raise ValueError("nonempty actor acquisition required")
        actor = learner.state_hash()
        records = []
        for row in rows:
            if (
                row.context.role != "FIT"
                or row.context.actor_sha256 != actor
                or len(row.context.programs) != 256
            ):
                raise ValueError("current-actor FIT256 context required")
            with self.budget.measure("actor_scoring"):
                scores = learner.score(row)
            ref = self.store.put(
                prefix + "/" + row.context.context_id + ".json",
                canonical(
                    {
                        "context": dataclasses.asdict(row.context),
                        "scores": scores.tolist(),
                        "scope_sha256": row.scope_sha256,
                    }
                ),
            )
            records.append(dataclasses.asdict(ref))
        state = checkpoint(learner, self.store, prefix + "/actor-checkpoint.pt")
        return self.store.put(
            prefix + "/proposals.json",
            canonical(
                {"actor": actor, "contexts": records, "state": state, "round_updates": learner.updates}
            ),
        )

    def teachers(self, proposal, scope, native, cache, k, prefix):
        doc = json.loads(proposal.read(self.store))
        records = []
        family_components, reference_families, context_ids = set(), set(), set()
        for i, item in enumerate(doc["contexts"]):
            row = json.loads(ObjectRef(**item).read(self.store))
            context = Context.from_dict(row["context"])
            if context.actor_sha256 != doc["actor"]:
                raise ValueError("persisted proposal actor mismatch")
            loaded = load_context(context, scope, self.store, role="FIT")
            if loaded.scope_sha256 != row["scope_sha256"]:
                raise ValueError("teacher scope differs from proposal scope")
            family_components.add(loaded.family_component_sha256)
            reference_families.update(loaded.reference_families)
            context_ids.add(context.context_id)
            teacher = collect_teacher(
                loaded,
                np.asarray(row["scores"]),
                k,
                native,
                cache,
                self.store,
                f"{prefix}/context-{i}",
                self.budget,
            )
            record = {"context": row["context"], "teacher": dataclasses.asdict(teacher)}
            if self.scientific:
                with self.budget.measure("native_scratch_custody_and_owned_reclamation"):
                    recovery = preserve_native_scratch(
                        teacher, loaded, native, cache, self.store, f"{prefix}/context-{i}"
                    )
                record["native_recovery"] = dataclasses.asdict(recovery)
            records.append(record)
        return self.store.put(
            prefix + "/teachers.json",
            canonical(
                {
                    "actor": doc["actor"],
                    "state": doc["state"],
                    "rows": records,
                    "k": k,
                    "native_calls": native.calls,
                    "native_failures": native.failures,
                    "cache_hits": cache.hits,
                    "context_ids": sorted(context_ids),
                    "family_components": sorted(family_components),
                    "new_independent_families_vs_reference": sorted(family_components - reference_families),
                }
            ),
        )

    def train(self, learner, rows, teachers, prefix, *, round_index, updates=600):
        if self.scientific and updates != 600:
            raise ValueError("exactly600 updates per scientific cell/round")
        if not self.scientific and next(learner.model.parameters()).device.type != "cpu":
            raise ValueError("controlled source fixture cannot use CUDA")
        if learner.updates != (round_index - 1) * updates or len(rows) != len(teachers) or not rows:
            raise ValueError("fixed round ancestry/nonempty teachers required")
        for row, ref in zip(rows, teachers, strict=True):
            learner.teacher_target(row, ref, self.store)  # Verify ALL labels before first update.
        self.progress = []
        self.last_checkpoint = None
        round_before = learner.round_snapshot(rows[0])
        groups = {}
        for i, row in enumerate(rows):
            groups.setdefault(row.family_component_sha256, []).append(i)
        names = sorted(groups)
        rng = np.random.default_rng(rows[0].context.seed)
        batches = [
            [int(rng.choice(groups[str(rng.choice(names))])) for _ in range(32)] for _ in range(updates)
        ]
        plan_ref = self.store.put(prefix + "/batch-plan.json", canonical(batches))
        try:
            for step, indices in enumerate(batches):
                batch = [rows[i] for i in indices]
                refs = [teachers[i] for i in indices]
                before = {k: tensor_hash(p) for k, p in learner.model.named_parameters() if p.requires_grad}
                state_before, gradients_before = learner.state_hash(), dict(learner.gradient_squares)
                forward_before = learner.score(batch[0])
                with self.budget.measure("student_update"):
                    loss = learner.update_batch(batch, refs, self.store)
                after = {k: tensor_hash(p) for k, p in learner.model.named_parameters() if p.requires_grad}
                loss_after = 0.0
                with torch.no_grad():
                    for row, ref in zip(batch, refs, strict=True):
                        target = learner.teacher_target(row, ref, self.store)
                        scores = learner.model(row)
                        loss_after += float(
                            torch.nn.functional.cross_entropy(
                                -scores[None], torch.tensor([target], device=scores.device)
                            )
                        ) / len(batch)
                if not np.isfinite(loss_after):
                    raise ValueError("nonfinite post-update loss")
                record = {
                    "update": learner.updates,
                    "examples": learner.examples,
                    "loss_before": loss,
                    "loss_after": loss_after,
                    "state_before": state_before,
                    "state_after": learner.state_hash(),
                    "parameter_before": before,
                    "parameter_after": after,
                    "changed_parameters": [k for k in before if before[k] != after[k]],
                    "gradient_squared_sums": {
                        k: learner.gradient_squares[k] - gradients_before[k] for k in gradients_before
                    },
                    "forward_before_sha256": sha(np.asarray(forward_before).tobytes()),
                    "forward_after_sha256": sha(learner.score(batch[0]).tobytes()),
                    "batch_plan": dataclasses.asdict(plan_ref),
                    "batch_index": step,
                    "frozen_invariants": "assert_frozen checked before/after update",
                }
                # Remote durability before the NEXT update; nothing depends on a
                # trajectory list surviving in RAM until the last cell completes.
                if self.journal is not None:
                    atomic_record(self.journal / f"update-{learner.updates:06d}.json", record)
                ref = self.store.put(f"{prefix}/steps/{learner.updates:06d}.json", canonical(record))
                self.progress.append(dataclasses.asdict(ref))
            proof = learner.proof_since(round_before, rows[0])
            self.last_checkpoint = checkpoint(
                learner,
                self.store,
                prefix + "/checkpoint.pt",
                local=self.journal / "checkpoint.pt" if self.journal else None,
            )
            return self.store.put(
                prefix + "/round.json",
                canonical(
                    {
                        "round": round_index,
                        **self.last_checkpoint,
                        "proof": proof,
                        "steps": self.progress,
                        "batch_plan": dataclasses.asdict(plan_ref),
                        "terminal": round_index == 3 and learner.updates == 1800,
                    }
                ),
            )
        except BaseException as exc:
            partial = {
                "error": repr(exc),
                "updates": learner.updates,
                "examples": learner.examples,
                "state_sha256": learner.state_hash(),
                "durable_steps": self.progress,
                "round": round_index,
                "status": "INCOMPLETE_NO_RETRY",
            }
            if self.journal is not None:
                atomic_record(self.journal / "INCOMPLETE.json", partial)
            try:
                self.last_checkpoint = checkpoint(
                    learner,
                    self.store,
                    prefix + "/partial-checkpoint.pt",
                    local=self.journal / "partial-checkpoint.pt" if self.journal else None,
                )
                self.store.put(prefix + "/INCOMPLETE.json", canonical({**partial, **self.last_checkpoint}))
            except BaseException as persistence_error:
                if self.journal is not None:
                    atomic_record(
                        self.journal / "custody-failure.json",
                        {
                            "error": repr(persistence_error),
                            "actual_updates": learner.updates,
                            "local_evidence": str(self.journal),
                        },
                    )
            raise

    def score(self, learner, rows, prefix, *, role, b_lock=None):
        if learner.updates != 1800 or any(r.context.role != role for r in rows):
            raise ValueError("only fixed terminal role-local student evaluation")
        panel = evaluate_fixed(
            learner,
            rows,
            checkpoint_sha256=learner.state_hash(),
            store=self.store,
            prefix=prefix,
            b_lock=b_lock,
        )
        # Programs/context refs and scores are locked BEFORE native utility reads.
        return self.store.put(
            prefix + "/score-lock.json",
            canonical(
                {
                    "role": role,
                    "state_sha256": learner.state_hash(),
                    "panel": dataclasses.asdict(panel),
                    "contexts": [dataclasses.asdict(r.context) for r in rows],
                }
            ),
        )

    def utilities(self, locked, scope, native, prefix):
        doc = json.loads(locked.read(self.store))
        panel = json.loads(ObjectRef(**doc["panel"]).read(self.store))["panel"]
        results = {}
        for context_doc in doc["contexts"]:
            context = Context.from_dict(context_doc)
            loaded = load_context(context, scope, self.store, role=doc["role"])
            scores = np.asarray(panel[context.context_id])
            order = sorted(
                range(len(scores)), key=lambda i: (float(scores[i]), sha(context.programs[i].encode()))
            )
            records = []
            for i in order[:256]:
                result = native.evaluate(loaded, i)
                records.append(result)
                self.store.put(f"{prefix}/{context.context_id}/native-{i}.json", canonical(result))
            if any(r["metrics"]["status"] != "completed" for r in records):
                raise RuntimeError("required evaluation fit failed; whole study INCOMPLETE")
            comparators = {}
            for k in (1, 16, 64, 256):
                best = min(records[:k], key=lambda r: (r["metrics"]["fitness"], order.index(r["index"])))
                comparators[str(k)] = {
                    "program_sha256": best["program_sha256"],
                    "utility": -float(best["metrics"]["fitness"])
                    if best["metrics"]["cv_gate_passed"]
                    else -2.0,
                }
            results[context.context_id] = {
                "student": comparators["1"],
                "descriptive_search": comparators,
                "state_sha256": doc["state_sha256"],
                "role": doc["role"],
            }
            self.store.put(
                f"{prefix}/{context.context_id}/utility.json", canonical(results[context.context_id])
            )
        return self.store.put(prefix + "/panel.json", canonical(results))


class StudyController:
    def __init__(self, supervisor, rows, inputs, baseline, *, repo):
        self.supervisor, self.store, self.repo = supervisor, supervisor.store, Path(repo)
        self.rows, self.inputs, self.baseline = rows, inputs, baseline
        self.fold_manifest = registered_cells(rows)
        payload = {"rows": rows, "inputs": inputs, "baseline": dataclasses.asdict(baseline)}
        if digest(payload) != supervisor.admission["study_input_payload_sha256"]:
            raise ValueError("study inputs differ from the actually qualified payload")
        if digest(self.fold_manifest) != supervisor.admission["fold_manifest_sha256"]:
            raise ValueError("study folds differ from the actually qualified partition")

    def run(self):
        state, terminal = {}, {}
        try:
            with self.supervisor.controller():
                self.store.put(
                    "launch.json",
                    canonical(
                        {
                            "admission": self.supervisor.admission,
                            "input_templates": self.inputs,
                            "row_manifest_sha256": digest(self.rows),
                            "original_caps": {"cpu_hours": 400, "gpu_hours": 18},
                            "actual_dope_policy": "each GPU phase persists actual model-setup.json before any update",
                        }
                    ),
                )
                fold_ref = self.store.put("folds.json", canonical(self.fold_manifest))
                baseline_doc = json.loads(self.baseline.read(self.store))
                if baseline_doc["context_ids"] != [r["context_id"] for r in self.rows]:
                    raise ValueError("original native reference population/order mismatch")
                self.baseline_values = np.asarray(baseline_doc["values"], dtype=np.float64)
                if self.baseline_values.shape not in ((len(self.rows),), (3, len(self.rows))) or (
                    not np.isfinite(self.baseline_values).all()
                ):
                    raise ValueError("finite matched original native reference panel required")
                for round_index in (1, 2, 3):
                    for phase in ("propose", "teacher", "train"):
                        for cell in self.fold_manifest["cells"]:
                            key = cell["cell_id"]
                            task = {
                                **cell,
                                **self.inputs[key],
                                "round": round_index,
                                "phase": phase,
                                "fold_manifest": dataclasses.asdict(fold_ref),
                                "previous": state.get((key, phase)),
                                "proposal": state.get((key, "propose")),
                                "teachers": state.get((key, "teacher")),
                                "prior_round": terminal.get(key),
                                "terminal_updates": 1800,
                            }
                            ref = self.supervisor.phase(
                                task,
                                cpu_reserve=self.inputs[key]["cpu_reserve_seconds"][phase],
                                seconds_cap=300
                                if phase == "train"
                                else self.inputs[key]["phase_caps"][phase],
                            )
                            state[(key, phase)] = dataclasses.asdict(ref)
                            if phase == "train":
                                doc = json.loads(ref.read(self.store))
                                if doc["updates"] != 600 * round_index:
                                    raise RuntimeError("cell did not reach its fixed round endpoint")
                                terminal[key] = dataclasses.asdict(ref)
                            self.store.put(
                                f"progress/round-{round_index}/{key}/{phase}.json",
                                canonical({"result": dataclasses.asdict(ref), "round": round_index}),
                            )
                b_panels = self._evaluate("B", terminal, fold_ref)
                selected = pooled_select(self.repo, self.rows, self.baseline_values, b_panels)
                checkpoints = [
                    json.loads(ObjectRef(**r).read(self.store))["state_sha256"] for r in terminal.values()
                ]
                lock = self.store.put(
                    "B-selection-lock.json",
                    canonical(
                        {
                            "status": "B_LOCKED",
                            "checkpoint_sha256": checkpoints,
                            "terminals": terminal,
                            "fold_manifest": dataclasses.asdict(fold_ref),
                            "source": self.supervisor.admission["source"],
                            "input_sha256": digest(self.inputs),
                            "B_panels_sha256": digest({k: v.tolist() for k, v in b_panels.items()}),
                            "selection": selected,
                        }
                    ),
                )
                a_panels = self._evaluate("A", terminal, fold_ref, lock)
                a_stats = panel_statistics(self.repo, self.rows, self.baseline_values, a_panels)
                chosen = selected["selected_arm"]
                cross_fit = (
                    chosen != "r882"
                    and a_stats[chosen]["ci95"][0] > 0
                    and (a_stats[chosen]["harm_fraction"] <= 0.1)
                )
                result = {
                    "status": "CROSS_FIT_CANDIDATE" if cross_fit else "NO_ADVANCE",
                    "selected_arm": chosen,
                    "B": selected,
                    "A_statistics": a_stats,
                    "A": {k: v.tolist() for k, v in a_panels.items()},
                    "B_lock": dataclasses.asdict(lock),
                    "sealed_confirmation": "NOT_RUN",
                    "dependence": "rotated B/A reuse families and related models, not independent confirmation",
                    "confirmed_lift": False,
                    "serving_eligible": False,
                }
                return self.store.put("study-result.json", canonical(result))
        except BaseException as exc:
            self.store.put(
                "study-INCOMPLETE.json",
                canonical(
                    {
                        "status": "INCOMPLETE_NO_RETRY",
                        "error": repr(exc),
                        "terminals": terminal,
                        "durable_phases": {str(k): v for k, v in state.items()},
                        "optimizer_counts": "per-update immutable records and terminal checkpoints",
                        "no_partial_arm_efficacy": True,
                    }
                ),
            )
            raise

    def _evaluate(self, role, terminal, fold_ref, lock=None):
        panels = {a: np.full((3, len(self.rows)), np.nan) for a in ARMS}
        index = {r["context_id"]: i for i, r in enumerate(self.rows)}
        seen = set()
        for cell in self.fold_manifest["cells"]:
            key = cell["cell_id"]
            task = {
                **cell,
                **self.inputs[key],
                "round": 3,
                "phase": "score",
                "role": role,
                "terminal_updates": 1800,
                "prior_round": terminal[key],
                "fold_manifest": dataclasses.asdict(fold_ref),
            }
            if lock is not None:
                task["b_lock"] = dataclasses.asdict(lock)
            score = self.supervisor.phase(
                task,
                cpu_reserve=self.inputs[key]["cpu_reserve_seconds"]["score"],
                seconds_cap=self.inputs[key]["phase_caps"]["score"],
            )
            task.update(phase="utility", locked_score=dataclasses.asdict(score))
            utility = self.supervisor.phase(
                task,
                cpu_reserve=self.inputs[key]["cpu_reserve_seconds"]["utility"],
                seconds_cap=self.inputs[key]["phase_caps"]["utility"],
            )
            data = json.loads(utility.read(self.store))
            if set(data) != set(cell["context_ids"][role]):
                raise ValueError("evaluation role-local population mismatch")
            for cid, record in data.items():
                item = (cell["arm"], cell["seed"], cid)
                if item in seen or record["role"] != role:
                    raise ValueError("duplicate/cross-role pooled observation")
                seen.add(item)
                panels[cell["arm"]][SEEDS.index(cell["seed"]), index[cid]] = record["student"]["utility"]
        if any(not np.isfinite(p).all() for p in panels.values()):
            raise ValueError("missing pooled B/A observations")
        return panels


def worker(task_path, expected_sha):
    """Actual subprocess entrypoint; parent reservation, source and CUDA permit required."""
    from research.flywheel_selfplay.hwave_numeric_threads import (
        admit_numeric_threads,
        capture_numeric_threads,
    )

    from .supervisor import PHASES, GPUWorkerPermit

    body = Path(task_path).read_bytes()
    if sha(body) != expected_sha:
        raise ValueError("phase task identity mismatch")
    task = json.loads(body)
    import ctypes

    if ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGTERM, 0, 0, 0) != 0:
        raise OSError("could not arm own worker parent-death signal")
    if os.getppid() != task["supervisor_pid"]:
        raise RuntimeError("phase supervisor disappeared before worker admission")
    if task["phase"] not in PHASES:
        raise ValueError("unknown phase")
    repo = Path(task["admission"]["source"]["repo"])
    verify_merged_source(repo, task["admission"]["source"])
    import boto3
    from botocore.config import Config

    client = boto3.Session(profile_name=task["admission"]["aws_profile"], region_name="us-east-2").client(
        "s3", config=Config(connect_timeout=5, read_timeout=20, retries={"max_attempts": 1})
    )
    store = S3Store(client, task["admission"]["bucket"], task["admission"]["output_prefix"])
    budget = ReservedPhaseBudget(task["ledger"], task["reservation"], task["stage"])
    numeric = admit_numeric_threads(
        [capture_numeric_threads(torch)], os_threads=len(os.listdir("/proc/self/task"))
    )
    atomic_record(
        Path(task_path).parent / "runtime.json",
        {"pid": os.getpid(), "numeric": numeric, "cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES")},
    )

    def stop(signum, frame):
        raise RuntimeError("supervisor terminal signal " + str(signum))

    signal.signal(signal.SIGTERM, stop)
    inputs = StudyInputs(task, store)
    runtime = PhaseRuntime(store, budget, journal=Path(task_path).parent / "incremental")
    prefix = task["stage"]
    phase = task["phase"]
    if phase in ("teacher", "utility"):
        binary = Path(task["admission"]["native_binary_path"])
        native = RustS1Evaluator(
            binary,
            binary_sha256=task["admission"]["native_binary_sha256"],
            evaluator_sha256=task["admission"]["native_metric_source_sha256"],
            scratch=Path(task["admission"]["native_scratch"]),
        )
        if phase == "teacher":
            cache = OutcomeCache(Path(task["admission"]["native_cache"]))
            ref = runtime.teachers(
                ObjectRef(**task["proposal"]), inputs.scope, native, cache, task["k"], prefix
            )
        else:
            ref = runtime.utilities(ObjectRef(**task["locked_score"]), inputs.scope, native, prefix)
    else:
        permit = GPUWorkerPermit(task, repo, store)
        bindings = (repo / "research/flywheel_selfplay/h206_search_distill/bindings.json").read_bytes()
        factory = PinnedR882Factory(
            repo,
            bindings,
            bindings_sha256=task["admission"]["bindings_sha256"],
            source_pins=task["admission"]["model_source_pins"],
            store=store,
            cache=ModelCache(task["admission"]["model_cache"], repo),
        )
        with factory.runtime():
            with budget.measure("actual_model_setup_and_device_copy"):
                learner = factory.learner_for_phase(
                    permit, seed=task["seed"], run_id=task["admission"]["run_id"]
                )
            store.put(prefix + "/model-setup.json", canonical(factory.receipt))
            previous = None
            if phase == "train":
                previous = json.loads(ObjectRef(**task["teachers"]).read(store))["state"]
            elif task["round"] > 1 or phase == "score":
                previous = json.loads(ObjectRef(**task["prior_round"]).read(store))
            if previous is not None:
                with budget.measure("checkpoint_resume"):
                    restore_verified_state(
                        learner,
                        ObjectRef(**previous["checkpoint"]),
                        store,
                        expected_state_sha256=previous["state_sha256"],
                    )
            if phase == "propose":
                ref = runtime.propose(learner, inputs.acquire("FIT", learner.state_hash()), prefix)
            elif phase == "train":
                teacher_doc = json.loads(ObjectRef(**task["teachers"]).read(store))
                if teacher_doc["actor"] != learner.state_hash() or teacher_doc["k"] != task["k"]:
                    raise ValueError("training teacher/actor/arm mismatch")
                rows = [
                    load_context(Context.from_dict(r["context"]), inputs.scope, store, role="FIT")
                    for r in teacher_doc["rows"]
                ]
                ref = runtime.train(
                    learner,
                    rows,
                    [ObjectRef(**r["teacher"]) for r in teacher_doc["rows"]],
                    prefix,
                    round_index=task["round"],
                )
            else:
                b_lock = ObjectRef(**task["b_lock"]) if task.get("b_lock") else None
                ref = runtime.score(
                    learner,
                    inputs.acquire(task["role"], learner.state_hash()),
                    prefix,
                    role=task["role"],
                    b_lock=b_lock,
                )
    atomic_record(Path(task_path).parent / "result.json", {"result": dataclasses.asdict(ref)})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    modes = ap.add_mutually_exclusive_group(required=True)
    modes.add_argument("--worker", type=Path)
    modes.add_argument(
        "--study", type=Path, help="hash-bound admitted study configuration; never inferred pins"
    )
    ap.add_argument("--task-sha256", required=True)
    args = ap.parse_args()
    if args.worker:
        worker(args.worker, args.task_sha256)
        return
    from .supervisor import PhaseSupervisor

    raw = args.study.read_bytes()
    if sha(raw) != args.task_sha256:
        raise ValueError("admitted study configuration SHA mismatch")
    config = json.loads(raw)
    admission = config["admission"]
    import boto3
    from botocore.config import Config

    client = boto3.Session(profile_name=admission["aws_profile"], region_name="us-east-2").client(
        "s3", config=Config(connect_timeout=5, read_timeout=20, retries={"max_attempts": 1})
    )
    store = S3Store(client, admission["bucket"], admission["output_prefix"])
    supervisor = PhaseSupervisor(
        config["repo"], config["output"], config["cache"], config["ledger"], store, admission
    )
    StudyController(
        supervisor, config["rows"], config["inputs"], ObjectRef(**config["baseline"]), repo=config["repo"]
    ).run()


if __name__ == "__main__":
    main()
