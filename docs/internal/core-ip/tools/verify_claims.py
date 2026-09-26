#!/usr/bin/env python3
"""Validate scoped publication claims against an evidence dependency graph (W12-T01/T05).

Evidence nodes bind a subject, scope, source commit, kind and parents. A claim must match the
subject, scope and source of every node it cites, and its stage must be backed by a node of an
admissible kind; a synthetic fixture never backs live, measurement or promotion claims. Revoking
a node revokes exactly its descendants; any claim citing a revoked node must itself be recorded
as revoked. History is append-only. Digests authenticate bytes, not the truth of a producer.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import re

CLAIMS = Path(__file__).resolve().parents[1] / "claims" / "claims.json"
SCHEMA = "jope.core-ip-claims/v1"
KINDS = {"source-trace", "synthetic-fixture", "real-corpus-fixture", "hosted-ci", "live-host",
         "measurement", "decision", "historical-record"}
STAGE_KINDS = {
    "implementation": {"source-trace"},
    "fixture": {"synthetic-fixture", "real-corpus-fixture"},
    "hosted-ci": {"hosted-ci"},
    "live": {"live-host"},
    "measurement": {"measurement"},
    "promotion": {"decision"},
    "historical": {"historical-record"},
}
CLAIM_FIELDS = ("id", "statement", "subject", "metric", "units", "denominator", "comparison",
                "baseline", "scope", "source_commit", "observed_utc", "uncertainty", "limitations",
                "stage", "evidence", "status")
SHA = re.compile(r"[0-9a-f]{40}")
DIGEST = re.compile(r"[0-9a-f]{64}")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_bytes(root, node_id, node, require_custody):
    """Repository evidence is always verified; custody evidence when present or required."""
    reference = node.get("path")
    require(isinstance(reference, str) and DIGEST.fullmatch(node.get("sha256") or ""),
            f"{node_id}: evidence path and digest")
    if node.get("location") == "repository":
        relative = PurePosixPath(reference)
        require(not relative.is_absolute() and ".." not in relative.parts,
                f"{node_id}: repository path traversal")
        path = root / relative
    else:
        require(node.get("location") == "custody" and PurePosixPath(reference).is_absolute(),
                f"{node_id}: custody evidence needs an absolute path")
        path = Path(reference)
        try:
            present = path.exists() or path.is_symlink()
        except OSError:  # an inaccessible custody root, e.g. another user's home on a CI runner
            present = False
        if not present:
            require(not require_custody, f"{node_id}: custody evidence unavailable")
            return False
    require(path.is_file() and not path.is_symlink(), f"{node_id}: evidence file missing")
    require(hashlib.sha256(path.read_bytes()).hexdigest() == node["sha256"],
            f"{node_id}: evidence digest mismatch")
    return True


def revoked_closure(nodes, revocations):
    revoked = {r["node"] for r in revocations}
    changed = True
    while changed:
        changed = False
        for node_id, node in nodes.items():
            if node_id not in revoked and revoked & set(node["depends_on"]):
                revoked.add(node_id)
                changed = True
    return revoked


def verify(path=CLAIMS, root=None, require_custody=False):
    path = Path(path)
    root = root or path.parents[4]
    document = json.loads(path.read_text())
    require(document.get("schema") == SCHEMA, "claims schema")
    nodes = document.get("evidence", {})
    require(isinstance(nodes, dict), "evidence map")
    verified = 0
    for node_id, node in nodes.items():
        require(node.get("kind") in KINDS, f"{node_id}: kind")
        require(SHA.fullmatch(node.get("source_commit") or ""), f"{node_id}: source commit")
        require(node.get("subject") and node.get("scope"), f"{node_id}: subject and scope")
        parents = node.get("depends_on")
        require(isinstance(parents, list) and all(p in nodes for p in parents),
                f"{node_id}: unknown parent")
        verified += check_bytes(root, node_id, node, require_custody)
    state, visiting, done = {}, set(), set()

    def visit(node_id):
        require(node_id not in visiting, f"{node_id}: dependency cycle")
        if node_id in done:
            return
        visiting.add(node_id)
        for parent in nodes[node_id]["depends_on"]:
            visit(parent)
        visiting.discard(node_id)
        done.add(node_id)

    for node_id in nodes:
        visit(node_id)
    revocations = document.get("revocations", [])
    last = 0
    for revocation in revocations:
        require(revocation.get("node") in nodes, "revocation of unknown evidence")
        require(revocation.get("issuer") and revocation.get("cause") and revocation.get("effective_utc"),
                "revocation issuer, cause and time")
        require(isinstance(revocation.get("sequence"), int) and revocation["sequence"] > last,
                "revocation sequence must increase (append-only)")
        last = revocation["sequence"]
    revoked = revoked_closure(nodes, revocations)
    ids = Counter()
    for claim in document.get("claims", []):
        require(set(claim) == set(CLAIM_FIELDS), f"{claim.get('id')}: claim fields")
        cid = claim["id"]
        ids[cid] += 1
        require(ids[cid] == 1, f"{cid}: duplicate claim")
        require(claim["stage"] in STAGE_KINDS, f"{cid}: stage")
        require(claim["status"] in {"active", "revoked", "superseded"}, f"{cid}: status")
        require(all(claim[f] for f in ("statement", "subject", "metric", "units", "denominator",
                                       "scope", "uncertainty")), f"{cid}: incomplete binding")
        require(isinstance(claim["limitations"], list) and claim["limitations"], f"{cid}: limitations")
        require(SHA.fullmatch(claim["source_commit"] or ""), f"{cid}: source commit")
        cited = claim["evidence"]
        require(isinstance(cited, list) and cited and all(e in nodes for e in cited),
                f"{cid}: evidence references")
        for node_id in cited:
            node = nodes[node_id]
            require(node["subject"] == claim["subject"] and node["scope"] == claim["scope"]
                    and node["source_commit"] == claim["source_commit"],
                    f"{cid}: subject, scope or source differs from evidence {node_id}")
        require(any(nodes[e]["kind"] in STAGE_KINDS[claim["stage"]] for e in cited),
                f"{cid}: no {claim['stage']} evidence")
        if claim["stage"] in {"live", "measurement", "promotion"}:
            require(not any(nodes[e]["kind"] == "synthetic-fixture" for e in cited),
                    f"{cid}: synthetic evidence cannot support a {claim['stage']} claim")
        if revoked & set(cited):
            require(claim["status"] == "revoked", f"{cid}: cites revoked evidence but is {claim['status']}")
    return {
        "evidence": len(nodes), "verified_bytes": verified, "custody_unverified": len(nodes) - verified,
        "revoked_evidence": sorted(revoked), "claims": dict(Counter(c["status"] for c in document.get("claims", []))),
        "scope": "claim-to-evidence binding and revocation closure; evidence truth needs independent review",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claims", type=Path, default=CLAIMS)
    parser.add_argument("--require-custody", action="store_true",
                        help="fail when custody evidence is not present on this host")
    args = parser.parse_args()
    print(json.dumps(verify(args.claims, require_custody=args.require_custody), sort_keys=True))
