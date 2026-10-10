"""Synthetic seven-body closure; never a production reward verifier."""

import hashlib
import json
import math

from rlfw.onpolicy.groups import (
    Draw,
    GroupSpec,
    PolicyIdentity,
    ResolvedGroup,
    verify_q,
)
from rlfw.research_hwm import ARTIFACT_ROLES, HWMKey, Publication, canonical_bytes
from rlfw.trie_logprob import TrieLogProb

SHA = hashlib.sha256(b"onpolicy-fixture").hexdigest()


def key_for(context="c"):
    return HWMKey(
        context,
        "dataset-v1",
        "cut1",
        ("f0", "f1"),
        SHA,
        SHA,
        SHA,
        "rmse",
        "lower",
        "opposite-fold-eps",
        "fixture-v1",
        SHA,
        SHA,
    )


def fixture_publication(q, *, context="c", program=SHA):
    key = key_for(context)
    grid = round(q * 1_000_000)
    bodies = {
        role: canonical_bytes(
            {
                "role": role,
                "key": key.digest,
                "program": program,
                "q_grid": grid,
                "parity": True,
                "execution": f"exec-{program}",
            }
        )
        for role in ARTIFACT_ROLES
    }
    refs = {
        role: {
            "sha256": hashlib.sha256(body).hexdigest(),
            "bytes": len(body),
            "uri": f"fixture://{role}",
            "version_id": "synthetic-immutable-v1",
        }
        for role, body in bodies.items()
    }
    publication = Publication(
        key.digest, program, f"exec-{program}", "qualified_complete", grid, refs
    )

    def verify(k, p):
        for role in ARTIFACT_ROLES:
            body, ref = bodies[role], p.artifacts[role]
            parsed = json.loads(body)
            if not (
                hashlib.sha256(body).hexdigest() == ref["sha256"]
                and len(body) == ref["bytes"]
                and ref["version_id"] == "synthetic-immutable-v1"
                and parsed["key"] == k.digest == p.hwm_key
                and parsed["parity"] is True
                and parsed["q_grid"] == p.q_grid
                and parsed["program"] == p.program_sha256
                and parsed["execution"] == p.execution_key
            ):
                return False
        return True

    return key, publication, verify


def q_for(q=0.2, *, context="c", parent="p", program=SHA):
    key, publication, verify = fixture_publication(q, context=context, program=program)
    return verify_q(
        key,
        publication,
        parent=parent,
        role_check=lambda k, p: True,
        verify_closure=verify,
    )


def logs(probabilities):
    return tuple(TrieLogProb(math.log(p), (math.log(p),)) for p in probabilities)


def group_for(qs=(0.3, 1.2), probabilities=(0.4, 0.6), *, h=0.2, best_of=2):
    q = tuple(
        q_for(x, program=hashlib.sha256(str(i).encode()).hexdigest())
        for i, x in enumerate(qs)
    )
    policy = PolicyIdentity(SHA, SHA, SHA, SHA, SHA)
    draws = tuple(
        Draw(str(i), outcome.program, lp)
        for i, (outcome, lp) in enumerate(zip(q, logs(probabilities)))
    )
    spec = GroupSpec(
        "g",
        "p",
        "c",
        q[0].comparison,
        policy,
        10,
        100.0,
        draws,
        best_of,
        None if h is None else q_for(h),
    )
    return ResolvedGroup(spec, q)
