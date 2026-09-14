"""Grown-family partition + Unresolved decisions. Does not apply-contexts or admit synthetics."""
from __future__ import annotations

import hashlib
import hmac
import json
from pathlib import Path

# Same seed and bucket rule as parallel/src/families.rs::stable_assignment.
REGISTRY_SEED = 20260907
LIVE_CONTEXTS = "runs/dataset-inclusion-20260909/correction-final/current-contexts.json"
DEFECTIVE_BINARY = "3c288e95bcdaf2fa"


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def stable_assignment(family, seed=REGISTRY_SEED):
    """families.rs 80/10/10. Not the grown-family rule."""
    hashed = hashlib.sha256(f"{seed}:family:{family}".encode()).hexdigest()
    bucket = int(hashed[:16], 16) % 1_000_000
    if bucket < 800_000:
        partition = "meta_train"
    elif bucket < 900_000:
        partition = "meta_validation"
    else:
        partition = "meta_test"
    return hashed, bucket, partition


def stable_assignment_v2(component_id, seed=REGISTRY_SEED):
    """Codex two-cycle 80/20 on the canonical connected-component id. No sealed test."""
    hashed = hashlib.sha256(f"{seed}:component:{component_id}".encode()).hexdigest()
    bucket = int(hashed[:16], 16) % 1_000_000
    partition = "meta_train" if bucket < 800_000 else "development_validation"
    return hashed, bucket, partition


def is_synthetic(meta, directory=""):
    """Port of audit.rs::synthetic. Name filters alone are not safe."""
    meta = meta or {}
    if meta.get("synthetic_origin") is True:
        return True
    if meta.get("source_real_data") is False:
        return True
    if meta.get("origin") in ("synthetic", "generated", "simulation"):
        return True
    for part in Path(directory).parts:
        if part.startswith("synth_") or part.startswith("synthetic"):
            return True
    return False


def refuse_live_index(output):
    path = Path(output).as_posix()
    if path.endswith(LIVE_CONTEXTS) or LIVE_CONTEXTS in path:
        raise ValueError("refusing to write the live current-contexts.json")
    return True


def artifact(root, ref):
    if not isinstance(ref, dict) or set(ref) != {"path", "sha256"}:
        raise ValueError("durable evidence path and hash required")
    path = Path(ref["path"])
    if not path.is_absolute():
        path = root / path
    path = path.resolve(strict=True)
    if not path.is_relative_to(root):
        raise ValueError("growth evidence escapes workspace")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != ref["sha256"]:
        raise ValueError("growth evidence content changed")
    return data


def assign_new_components(families, live_families=(), sealed_test_families=(),
                          task="regression", provenance_resolved=None,
                          decision_receipts=None, component_ids=None, *,
                          root=None, graph_ref=None, parent_registry_ref=None):
    """Authenticate decisions before assigning genuinely new canonical components.

    Policy and reviewer keys are installed separately under trusted workspace
    custody. No caller Boolean, receipt summary, or alias map grants admission.
    The returned new lineage does not apply contexts or activate execution.
    """
    if task != "regression":
        raise ValueError("live family growth is regression-only")
    if provenance_resolved is not None or component_ids is not None:
        raise ValueError("provenance Booleans and caller component aliases cannot authorize growth")
    if root is None or graph_ref is None or parent_registry_ref is None or not decision_receipts:
        raise ValueError("durable provenance graph, parent registry and decision receipts required")
    families = list(families)
    root = Path(root).resolve(strict=True)
    policy_path = root / "manifests/family-growth-policy.json"
    if policy_path.is_symlink() or not policy_path.resolve().is_relative_to(root):
        raise ValueError("pinned provenance policy escaped workspace custody")
    policy_data = policy_path.read_bytes()
    policy_sha = hashlib.sha256(policy_data).hexdigest()
    policy = json.loads(policy_data)
    if policy.get("schema") != "parallel.family_growth_policy.v3" or policy.get("seed") != REGISTRY_SEED:
        raise ValueError("unsupported pinned provenance policy")
    graph = json.loads(artifact(root, graph_ref))
    parent = json.loads(artifact(root, parent_registry_ref))
    if graph.get("schema") != "parallel.component_graph.v1" or parent.get("schema") != "parallel.family_registry.v1":
        raise ValueError("unsupported graph or parent registry")
    aliases, components = {}, {}
    for component in graph["components"]:
        cid = component["component_id"]
        if not cid or cid in components or not component.get("aliases"):
            raise ValueError("duplicate or empty canonical component")
        components[cid] = component
        names = set(component["aliases"]) | {cid}
        if len(names) != len(component["aliases"]) or cid not in component["aliases"]:
            raise ValueError("component aliases must be unique and include canonical id")
        for name in names:
            if not isinstance(name, str) or not name or name in aliases:
                raise ValueError("ambiguous component alias")
            aliases[name] = cid
    def canonical(name):
        if name not in aliases:
            raise ValueError("unresolved component alias: " + name)
        return aliases[name]
    memberships = {}
    for name, assignment in parent["families"].items():
        # Every old family and source alias must occur in the frozen graph.
        ids = {canonical(alias) for alias in [name, *assignment["source_keys"]]}
        if len(ids) != 1:
            raise ValueError("parent family split across canonical components")
        memberships.setdefault(ids.pop(), set()).add(assignment["partition"])
    blocked_live = {canonical(name) for name in live_families}
    blocked_sealed = {canonical(name) for name in sealed_test_families}
    decisions = {}
    for ref in decision_receipts:
        if set(ref) != {"path", "sha256", "signature"}:
            raise ValueError("authenticated decision receipt required")
        raw = artifact(root, {key: ref[key] for key in ("path", "sha256")})
        decision = json.loads(raw)
        cid = decision.get("component_id")
        if cid not in components or cid in decisions:
            raise ValueError("duplicate or unresolved component decision")
        reviewer = decision.get("reviewer")
        if reviewer not in policy.get("reviewers", []) or not isinstance(reviewer, str) or not reviewer.replace("-", "").replace("_", "").isalnum():
            raise ValueError("unauthorized provenance reviewer")
        key_path = root / "credentials/provenance-review" / (reviewer + ".key")
        if key_path.is_symlink() or not key_path.resolve().is_relative_to(root):
            raise ValueError("reviewer key escaped local custody")
        key = key_path.read_bytes()
        expected = hmac.new(key, ("parallel.family_growth.v3:" + ref["sha256"]).encode(), hashlib.sha256).hexdigest()
        if len(key) < 32 or not hmac.compare_digest(expected, ref["signature"]):
            raise ValueError("unauthenticated provenance decision")
        if decision.get("schema") != "parallel.provenance_decision.v3" or decision.get("admission") != "approved":
            raise ValueError("unsupported or unapproved provenance decision")
        for field, expected in (("graph_sha256", graph_ref["sha256"]),
                                ("parent_registry_sha256", parent_registry_ref["sha256"]),
                                ("policy_sha256", policy_sha), ("task", "regression")):
            if decision.get(field) != expected:
                raise ValueError("stale or conflicting provenance decision: " + field)
        if not all(isinstance(decision.get(field), str) and decision[field].strip() for field in ("target", "units")):
            raise ValueError("missing target or units contract")
        if not decision.get("predictors") or not all(isinstance(p, str) and p for p in decision["predictors"]):
            raise ValueError("missing predictor contract")
        contract = decision.get("entity_time_contract", {})
        if contract.get("strategy") not in ("iid", "grouped", "temporal"):
            raise ValueError("missing entity/time contract")
        if contract["strategy"] == "grouped" and not contract.get("entity"):
            raise ValueError("grouped source lacks entity contract")
        if contract["strategy"] == "temporal" and not all(contract.get(k) is not None for k in ("entity", "order", "horizon", "gap")):
            raise ValueError("temporal source lacks order/horizon/gap contract")
        sources = components[cid].get("source_objects")
        if not sources or decision.get("source_objects") != sources:
            raise ValueError("provenance source identity mismatch")
        for source in sources:
            artifact(root, source)
        prior = decision.get("prior_use_findings", {})
        if prior.get("status") != "unused" or prior.get("parent_registry_sha256") != parent_registry_ref["sha256"] or not prior.get("evidence"):
            raise ValueError("original prior-use evidence required")
        for evidence in prior["evidence"]:
            artifact(root, evidence)
        decisions[cid] = ref
    rows, live, sealed, quarantined, missing = [], [], [], [], []
    for cid in sorted({canonical(name) for name in families}):
        partitions = memberships.get(cid, set())
        if len(partitions) > 1:
            quarantined.append(cid)
        elif "meta_test" in partitions or cid in blocked_sealed:
            sealed.append(cid)
        elif partitions or cid in blocked_live:
            live.append(cid)
        elif cid not in decisions:
            missing.append(cid)
        else:
            hashed, bucket, partition = stable_assignment_v2(cid)
            rows.append(dict(family=cid, component_id=cid, aliases=sorted(components[cid]["aliases"]),
                partition=partition, assignment_hash=hashed, bucket=bucket,
                assignment_method="stable_sha256_component_v2", decision=decisions[cid]))
    return dict(schema="parallel.family_growth_assignment.v3", root=str(root), seed=REGISTRY_SEED,
        requested_families=sorted(set(families)), additional_live_families=sorted(set(live_families)),
        additional_sealed_families=sorted(set(sealed_test_families)),
        graph=graph_ref, parent_registry=parent_registry_ref,
        policy=dict(path=str(policy_path), sha256=policy_sha), decisions=decision_receipts,
        rows=rows, n_train=sum(r["partition"] == "meta_train" for r in rows),
        n_development_validation=sum(r["partition"] == "development_validation" for r in rows),
        n_test=0, excluded_live_components=live, sealed_test_components=sealed,
        quarantined_components=quarantined, missing_decision_components=missing,
        runtime_closure_qualified=False, execution_eligible=False,
        apply_contexts_ran=False, final_tests_opened=False)


def verify_assignment(path):
    supplied = json.loads(Path(path).read_bytes())
    if supplied.get("schema") != "parallel.family_growth_assignment.v3":
        raise ValueError("unsupported growth assignment")
    expected = assign_new_components(supplied["requested_families"],
        live_families=supplied["additional_live_families"],
        sealed_test_families=supplied["additional_sealed_families"], root=supplied["root"],
        graph_ref=supplied["graph"], parent_registry_ref=supplied["parent_registry"],
        decision_receipts=supplied["decisions"])
    if expected != supplied:
        raise ValueError("growth assignment differs from original evidence")
    return expected


def decide_unresolved(rows, source_root=None):
    """Draft decisions. Evidenced Unresolved stays Unresolved; never auto-admit."""
    decisions = []
    for row in rows or []:
        dataset = row.get("dataset") or row.get("id") or ""
        directory = row.get("directory") or ""
        meta = row.get("meta") or {}
        source_path = row.get("source_path")
        if DEFECTIVE_BINARY in dataset or DEFECTIVE_BINARY in directory:
            decisions.append(dict(
                key=dataset or DEFECTIVE_BINARY,
                admission="excluded",
                family=row.get("family") or "",
                reason="defective binary: no train.csv/test.csv",
                evidence=[],
            ))
            continue
        if is_synthetic(meta, directory):
            decisions.append(dict(
                key=dataset,
                admission="excluded",
                family=row.get("family") or "",
                reason="synthetic; audit.rs::synthetic() is the only safe filter",
                evidence=[],
            ))
            continue
        exists = bool(source_path)
        if source_root and source_path:
            exists = (Path(source_root) / source_path).exists() if not Path(source_path).is_absolute() else Path(source_path).exists()
        if exists and meta.get("source_real_data") is not False:
            decisions.append(dict(
                key=dataset,
                admission="unresolved",
                family=row.get("family") or "",
                reason="evidenced source_path; human adjudication required; not auto-admitted",
                evidence=[],
            ))
            continue
        decisions.append(dict(
            key=dataset,
            admission="unresolved",
            family=row.get("family") or "",
            reason="observational origin needs source evidence; legacy real flags are insufficient",
            evidence=[],
        ))
    approved = sum(1 for d in decisions if d["admission"] == "approved")
    if approved:
        raise ValueError("decisions draft must not auto-admit")
    return dict(
        schema="parallel.unresolved_decisions_draft.v1",
        writes_initial_decisions=False,
        auto_admitted=0,
        n=len(decisions),
        excluded=sum(1 for d in decisions if d["admission"] == "excluded"),
        unresolved=sum(1 for d in decisions if d["admission"] == "unresolved"),
        decisions=decisions,
    )


def v5_canary_contract():
    """Canary is a new index only. Not live apply. Not the grown-family rule."""
    return dict(
        schema="parallel.v5_canary_new_index_only.v1",
        partition_rule="not_selected",
        selected_rule="frozen_80_20_hash",
        writes_live_index=False,
        apply_contexts_ran=False,
        apply_contexts_outer_flock=False,
        awaiting_claude_ack=True,
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(verify_assignment(args.verify), sort_keys=True, separators=(",", ":"), allow_nan=False))
