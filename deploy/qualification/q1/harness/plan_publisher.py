#!/usr/bin/env python3
"""Reviewed source of the sole fixed-state API01 publisher; never evaluate rows.

Derived from api01-t0710/freeze_plan.py SHA256 cc770052b8acbb21ab65d1051cd505aca62bfd65bad61881ad2b47d019524d10.
Install exact reviewed bytes at that original path; no alternate state or ledger.
"""

import argparse
import datetime
import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
import importlib.machinery
import importlib.util
from pathlib import Path
from plan_publisher_shape import (
    _authorize_module_source as _shape__authorize_module_source,
    _module_filename as _shape__module_filename,
    pin_materials as _shape_pin_materials,
    validate_runtime_receipt as _shape_validate_runtime_receipt,
)

sys.dont_write_bytecode = True
E = Path("/home/ubuntu/jope_plan/evidence/QLT-04/api01-t0710")
REPO = Path("/home/ubuntu/JopeDime")
BASE = "deploy/qualification/q1/"
LEDGER = Path("/home/ubuntu/jope_plan/ns/qualification/q1_ledger")
EVIDENCE = Path("/home/ubuntu/jope_plan/evidence")
BUNDLE = "94fd9587675c7806bd0a0582b6e97540cd2fdc7ebbc51958411b905185cad85c"
DOPE = "c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814"
GH = "/home/ubuntu/.local/bin/gh-role"
ENV = dict(os.environ, GIT_OPTIONAL_LOCKS="0", PYTHONDONTWRITEBYTECODE="1")
PRS = (1530, 1528, 1534, 1556, 1591, 1598, 1604, 1620, 1638, 1642, 1676)
R19_DOC_SHA = "ec6fdfae636d1a08d5d0cf8f80fbb9e32631a671ff2c2fcfddf2f327c3be8103"
R19_AUDIT_SHA = "f0fd94ed3848dc7af06ee0e25e08f41ea73828b2cf3387484c53a34c8dc2f242"
R19_VERIFICATION_SHA = (
    "234a47d271a3728ac3f92904098ef97d9e7d13ac281aac587e61584e38075f0d"
)


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(p):
    return sha(Path(p).read_bytes())


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def git(*args):
    return subprocess.check_output(["git", "-C", str(REPO), *args], env=ENV)


def raw(source, path):
    return git("show", source + ":" + path)


def markers():
    return any(
        (LEDGER / (BUNDLE + "." + s + ".json")).exists()
        or (LEDGER / (BUNDLE + "." + s + ".json")).is_symlink()
        for s in ("started", "finished")
    )


def public(*args):
    return subprocess.check_output(
        [GH, "writer", *args], env=ENV, text=True, timeout=20
    )


def regular(path):
    p = Path(path)
    assert p.is_file() and not any(x.is_symlink() for x in (p, *p.parents)), str(p)
    return p


FROZEN_SOURCE = "ccaf14b210b3b55a350218e60f6b49ef76524cf3"
# Exact reviewed dependency bytes. Changing a dependency requires a reviewed pin update.
MODULE_PINS = {
    "common": (
        "deploy/qualification/v1.0/providers/common.py",
        "0df9807cf0438df01b39e105d4870008da0876002583dadc94fcc0d76e7fcfda",
        True,
    ),
    "m3_full": (
        "deploy/qualification/v1.0/providers/m3_full.py",
        "454cab4fb5ded27167a6437ce44e66a41d665c4749de574a9868e59b01b4957f",
        True,
    ),
    "_q1_waiver_gates": (
        "deploy/qualification/v1.0/gates_receipt.py",
        "d78c6fc3f8e9c30cdd2164ca4d4a6c3268db61498f08a75d3572b6fa20ce70a9",
        False,
    ),
    "q1_bindings": (
        BASE + "harness/q1_bindings.py",
        "005ba5824db317ca3c8a2dcda5569e7c7479353fabb10c4abe567414fd4d235e",
        False,
    ),
    "q1": (
        BASE + "harness/q1.py",
        "3b1b23739de2014f10bc9741baf7db98ba960fedb932507ad6087ae0ac880011",
        False,
    ),
    "q1_cut": (
        BASE + "harness/q1_cut.py",
        "1ab459f42b945b685fe28c064559a5fc5ac2c1991a67e69809678c27d9505e9d",
        False,
    ),
    "q1_dedup": (
        BASE + "harness/q1_dedup.py",
        "c19e253ccfe7b0c02cb00ba9218451723d76b9f15fbca96a2f869446232e7653",
        True,
    ),
}


def _authorize_module_source(source, frozen):
    return _shape__authorize_module_source(source, frozen, globals())



def _module_filename(name, source, relative, filename):
    return _shape__module_filename(name, source, relative, filename, globals())



class _PinnedGitLoader(importlib.machinery.SourceFileLoader):
    """Standard import protocol over verified Git bytes, with no disk/cache input."""

    def __init__(self, name, filename, data, expected_sha256):
        super().__init__(name, str(filename))
        self._data = data
        self._expected_sha256 = expected_sha256

    def path_stats(self, path):
        # No filesystem timestamp means importlib never accepts or writes bytecode.
        raise OSError("filesystem source and bytecode caches are not inputs")

    def get_data(self, path):
        if path != self.path:
            raise OSError("only the fixed source path is admitted")
        require(sha(self._data) == self._expected_sha256, "pinned module bytes changed")
        return self._data


def module(name, source, filename):
    """Import one allowlisted, source-authorized and content-pinned dependency."""
    require(name in MODULE_PINS, "unreviewed module name")
    relative, expected_sha256, frozen = MODULE_PINS[name]
    _authorize_module_source(source, frozen)
    filename = _module_filename(name, source, relative, filename)
    data = raw(source, relative)
    require(sha(data) == expected_sha256, "reviewed module Git bytes differ")
    loader = _PinnedGitLoader(name, filename, data, expected_sha256)
    spec = importlib.util.spec_from_loader(name, loader)
    require(spec is not None, "module import specification unavailable")
    value = importlib.util.module_from_spec(spec)
    existed, previous = name in sys.modules, sys.modules.get(name)
    sys.modules[name] = value
    try:
        loader.exec_module(value)
    except BaseException:
        if existed:
            sys.modules[name] = previous
        else:
            sys.modules.pop(name, None)
        raise
    return value


def require(value, message):
    if not value:
        raise ValueError(message)


def pin_materials(source, metadata_source, main_source):
    return _shape_pin_materials(source, metadata_source, main_source, globals())



def validate_runtime_receipt(receipt, plan, pin, metadata_source):
    return _shape_validate_runtime_receipt(receipt, plan, pin, metadata_source, globals())



def regenerate(preview=False, source_override=None, metadata_source=None):
    require(sys.flags.optimize == 0, "Publisher requires assertion-enabled Python")
    identity = json.loads(
        (E.parent / "api01-t0640/ORIGINAL-LEDGER-IDENTITY.json").read_text()
    )
    observed = {
        "driver_host": os.uname().nodename.split(".")[0],
        "ledger": str(LEDGER),
        "resolved": str(LEDGER.resolve()),
        "device": LEDGER.stat().st_dev,
        "inode": LEDGER.stat().st_ino,
    }
    assert observed == identity, (
        "Original driver/physical ledger differs; never sign a fresh-host ledger"
    )
    assert not markers() and not (E.parent / "api01-t0250/dispatch.json").exists(), (
        "Candidate already consumed/dispatched; do not regenerate/retry"
    )
    if not preview:
        signing = Path("/home/ubuntu/jope_plan/signing")
        name = "sol_v1_api01-QLT-04.json"
        existing = [
            signing / b / name
            for b in ("requests", "done", "rejected")
            if (signing / b / name).exists()
        ]
        if existing:
            assert (E / "READINESS.json").exists(), (
                "Existing request needs original stopped-head audit"
            )
            prior = json.loads((E / "READINESS.json").read_text())
            request = json.loads(existing[0].read_text())
            assert (
                prior["plan_sha256"] == request["sha256"]
                and prior["plan"] == request["artifact"]
            )
            assert file_sha(regular(request["artifact"])) == request["sha256"]
            return dict(
                prior, request_already_filed=True, automatic_source_change_refused=True
            )
    prs = {
        str(n): json.loads(
            public(
                "pr",
                "view",
                str(n),
                "--repo",
                "neverhuman/JopeDime",
                "--json",
                "state,headRefOid,mergedAt,mergeCommit,url",
            )
        )
        for n in PRS
    }
    blocked = [
        int(n) for n, r in prs.items() if r["state"] != "MERGED" or not r["mergeCommit"]
    ]
    if blocked and not preview:
        result = {
            "utc": now(),
            "status": "WAITING_FOR_CONSUMER_MERGE",
            "unmerged_prs": blocked,
            "prs": prs,
            "source_frozen": False,
            "plan_regenerated": False,
            "signable": False,
            "Q1": "NOT_RUN",
            "authority": "T0710 frozenF=3; worker fix GOV03/RES03; no ruling question",
            "signing_request_filed": False,
        }
        save(E / "STATE.json", result)
        return result
    main_source = public(
        "api", "repos/neverhuman/JopeDime/commits/main", "--jq", ".sha"
    ).strip()
    qualified = E / "QUALIFIED-INPUTS.json"
    handoff = json.loads(qualified.read_text()) if qualified.exists() else None
    source = source_override or (handoff["source_commit"] if handoff else main_source)
    assert re.fullmatch("[0-9a-f]{40}", source)
    if source_override and handoff:
        assert source_override == handoff["source_commit"], (
            "Requested source differs from actual qualified inputs"
        )
    exists = subprocess.run(
        ["git", "-C", str(REPO), "cat-file", "-e", source + "^{commit}"],
        env=ENV,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if exists.returncode:
        git("fetch", "--no-tags", "origin", source)
    if source != main_source:
        if subprocess.run(
            ["git", "-C", str(REPO), "cat-file", "-e", main_source + "^{commit}"],
            env=ENV,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode:
            git("fetch", "--no-tags", "origin", main_source)
        if not preview:
            git("merge-base", "--is-ancestor", source, main_source)
    for n, r in prs.items():
        if r["state"] == "MERGED":
            git("merge-base", "--is-ancestor", r["mergeCommit"]["oid"], source)
    pin, pin_source, pin_files = pin_materials(source, metadata_source, main_source)
    code_source = metadata_source or source
    folder = E / ("preview" if preview else "frozen") / source
    if metadata_source:
        folder = folder / "metadata" / metadata_source
    folder.mkdir(parents=True, exist_ok=True)
    amendment = json.loads(raw(source, BASE + "amendments/t0610.json"))
    r19_bytes = raw(source, BASE + "PREREG-AMENDMENT-R19.md")
    assert sha(r19_bytes) == R19_DOC_SHA, "R19 exact reviewed amendment differs"
    roster_path = amendment["roster_cut_packet_path"]
    roster_bytes = raw(source, roster_path)
    assert sha(roster_bytes) == amendment["roster_cut_packet_sha256"]
    roster = json.loads(roster_bytes)
    aggregate_bytes = raw(source, amendment["fold_manifest_path"])
    assert sha(aggregate_bytes) == amendment["fold_manifest_sha256"]
    aggregate = json.loads(aggregate_bytes)
    byid = {r["dataset_id"]: r for r in aggregate["records"]}
    # Only four metadata artifacts are materialized, not another checkout. Run the
    # exact frozen validator/codec source bytes with those exact input artifacts.
    validation = folder / "validation-inputs"
    validation.mkdir(exist_ok=True)
    artifacts = {
        "PROTOCOL": BASE + "harness/protocol-v1.json",
        "VOCABULARY": "crates/jope-engine/assets/grammar/featmath.json",
        "ENGINE_VOCABULARY": "crates/jope-engine/assets/grammar/featmath-jain-v1.json",
        "EXECUTABLE_OPS": "crates/jope-engine/assets/grammar/jain-math-executable-9fa594e2.json",
    }
    inputs = []
    bindings = module(
        "q1_bindings",
        code_source,
        validation / "q1_bindings.py",
    )
    for name, path in artifacts.items():
        data = raw(source, path)
        dest = validation / Path(path).name
        dest.write_bytes(data)
        setattr(bindings, name, dest)
        inputs.append(
            {
                "path": path,
                "sha256": sha(data),
                "git_blob": git("rev-parse", source + ":" + path).decode().strip(),
            }
        )
    # Reviewed pin metadata is separate from the actual compiled evaluator source.
    for relative, data in pin_files:
        dest = validation / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        inputs.append(
            {
                "path": BASE + relative,
                "sha256": sha(data),
                "git_blob": git("rev-parse", pin_source + ":" + BASE + relative)
                .decode()
                .strip(),
                "source_commit": pin_source,
            }
        )
    bindings.EVALUATOR_PIN = validation / "evaluator-pin.json"
    prereg = validation / "PREREGISTRATION.md"
    prereg.write_bytes(raw(source, BASE + "PREREGISTRATION.md"))
    q1 = module("q1", code_source, validation / "q1.py")
    cut = module("q1_cut", code_source, validation / "q1_cut.py")
    # Current T1115 source uses the frozen owner membership and no R19 override.
    selected = {}
    unknown = []
    cut_checks = []
    duplicates = []
    original_roster = json.loads(
        (E.parent.parent / "QLT-02-F02/roster_t0341.json").read_text()
    )
    original_ids = set(original_roster["members"]["catalog_dataset_hash_ids"])
    for row in roster["official_roster"]:
        definition = json.loads(raw(source, row["cut_definition_path"]))
        assert cut.cut_sha256(definition) == row["cut_sha256"]
        assert definition == byid[row["dataset_id"]]["cut_definition"]
        assert definition["dataset_sha256"] == row["dataset_sha256"]
        folds = definition["folds"]
        assert len(folds) == 3 and all(f["repeat"] == 0 for f in folds)
        all_rows = set(range(definition["train_rows"]))
        validation_seen = set()
        for f in folds:
            valid = f["validation"]
            assert (
                valid == sorted(set(valid))
                and valid
                and not (validation_seen & set(valid))
            )
            assert f["train"] == sorted(all_rows - set(valid))
            validation_seen.update(valid)
        assert validation_seen == all_rows
        key = row.get("family_key")
        authoritative = (
            row.get("family_binding_status") == "authoritative"
            and isinstance(key, str)
            and bool(re.fullmatch("[0-9a-f]{40}|[0-9a-f]{64}", key))
        )
        holdout = row.get("holdout_group") or (
            "sealed_meta_test" if row["dataset_id"] in original_ids else None
        )
        d = {
            "id": row["dataset_id"],
            "task": "regression",
            "dataset_sha256": row["dataset_sha256"],
            "family": key,
            "family_authoritative": authoritative,
            "rows": row["rows"],
            "features": row["features"],
            "cut_sha256": row["cut_sha256"],
            "holdout_group": holdout,
        }
        if authoritative:
            if key in selected:
                previous = selected[key]
                keep, drop = sorted([previous, d], key=lambda x: x["dataset_sha256"])
                selected[key] = keep
                duplicates.append(
                    {
                        "family": key,
                        "selected": keep["id"],
                        "excluded": drop["id"],
                        "rule": "smallest eligible dataset SHA before outcomes",
                    }
                )
            else:
                selected[key] = d
        else:
            unknown.append(d)
        cut_checks.append(
            {
                "id": d["id"],
                "cut_sha256": d["cut_sha256"],
                "three_disjoint_validation_partitions_and_ascending_complements": True,
            }
        )
    datasets = sorted([*selected.values(), *unknown], key=lambda x: x["dataset_sha256"])
    current = json.loads(raw(source, BASE + "amendments/t1115.json"))
    assert (
        current["required_distinct_families_per_subgroup"] == 5
        and current["supersedes_R19_four_wide_exception_for_new_plan"] is True
    )
    assert current["owner_direction_sha256"] == file_sha(
        "/home/ubuntu/nudges/lanes_t1115.md"
    )
    assert current["source_roster"]["sha256"] == sha(roster_bytes)

    def source_json(reference):
        data = raw(source, reference["path"])
        assert sha(data) == reference["sha256"]
        return json.loads(data)

    source_json(current["identity_table"])
    membership = source_json(current["membership"])
    declared = source_json(current["selection"])
    source_json(current["wide_census"])
    dedup = module(
        "q1_dedup",
        source,
        validation / "q1_dedup.py",
    )
    datasets, excluded = dedup.select(roster, membership)
    assert (
        datasets == declared["datasets"]
        and excluded == declared["excluded_family_variants"]
    )
    assert len(datasets) == 97 and len({d["family"] for d in datasets}) == 97
    unknown = []
    duplicates = excluded
    subgroup_counts = dedup.subgroup_counts(datasets)
    r20 = json.loads(raw(source, BASE + "amendments/r20.json"))
    assert r20["pre_outcome"] is True and r20["required_families"] == {
        "small": 5,
        "large": 5,
        "narrow": 5,
        "wide": 3,
    }
    assert r20["owner_instruction_sha256"] == file_sha(
        "/home/ubuntu/nudges/lanes_t1230.md"
    )
    assert sha(raw(source, r20["wide_gap"]["path"])) == r20["wide_gap"]["sha256"]
    source_json(r20["wide_census"])
    assert r20["wide_census"]["sha256"] == q1.R20_CENSUS_SHA256
    census_path = folder / "rosters" / (q1.R20_CENSUS_SHA256 + ".json")
    census_path.parent.mkdir(exist_ok=True)
    census_bytes = raw(source, r20["wide_census"]["path"])
    census_path.write_bytes(census_bytes)
    inputs.append(
        {
            "path": r20["wide_census"]["path"],
            "sha256": sha(census_bytes),
            "git_blob": git("rev-parse", source + ":" + r20["wide_census"]["path"])
            .decode()
            .strip(),
        }
    )

    assert not any(
        d["id"] == "03bff9080ce043dc"
        or d["family"] == "b86887a811d3f27fd4c9a96416433442b2652259"
        for d in datasets
    ), "R9: gina_agnostic was in r782 training; never admit or move it to sealed"
    closure_path = E.parent / "res03-t0218/exclusion-aware-closure.json"
    closure_bytes = closure_path.read_bytes()
    assert (
        sha(closure_bytes)
        == "8e786b2dcb153a5c1c4f43dbcb773b53df6f0dd7d325c94f8d7ec2f73352d3e8"
    ), (
        "R9: previously frozen training-closure bytes changed; do not reclassify training families after r782 training"
    )
    closure = json.loads(closure_bytes)
    evaluator = (
        handoff.get("evaluator_sha256")
        if handoff
        else (pin["sha256"] if metadata_source else None)
    )
    plan = {
        "schema": "jope.q1-plan/v1",
        "mode": "qualification",
        "source_commit": source,
        "preregistration_sha256": file_sha(prereg),
        "backend_sha256": sha(raw(source, BASE + "harness/production_backend.py")),
        "budget": 24,
        "seeds": [0, 1, 2],
        "datasets": datasets,
        "pins": dict(
            bundle_sha256=BUNDLE,
            dope_sha256=DOPE,
            evaluator_sha256=evaluator,
            **bindings.expected_pins(),
            training_family_manifest_sha256=file_sha(closure_path),
        ),
        "training_families": closure["authoritative_bound_family_keys"],
        "timeout_seconds": 600,
        "ledger_root": str(LEDGER),
        "wide_census_sha256": q1.R20_CENSUS_SHA256,
    }
    if metadata_source:
        plan["metadata_source_commit"] = metadata_source
    subgroup_minima = {g: q1.subgroup_minimum(plan, g) for g in q1.GROUPS}
    cohort_ok = all(n >= subgroup_minima[g] for g, n in subgroup_counts.items())
    # T1115 changes selection/family namespace and prereg/source provenance only.
    baseline_path = (
        E.parent
        / "api01-t0840/worker/0dd48aebc2e9fd19b888ad14ab9a5752b0d59f36/plan.regenerated-merged-worker.json"
    )
    baseline = json.loads(baseline_path.read_text())
    allowed = {
        "metadata_source_commit",
        "source_commit",
        "datasets",
        "preregistration_sha256",
        "minimum_wide_families",
        "wide_census_sha256",
    }
    compare = {k: v for k, v in plan.items() if k not in allowed}
    expected = {k: v for k, v in baseline.items() if k not in allowed}
    if evaluator is not None:
        assert expected["pins"]["evaluator_sha256"] is None
        expected["pins"] = dict(expected["pins"], evaluator_sha256=evaluator)
    assert compare == expected, "R20 changed a frozen execution pin/budget/closure"
    assert "minimum_wide_families" not in plan
    r19_admission_ok = False
    save(
        folder / "R20-PLAN-DELTA.json",
        {
            "source_commit": source,
            "baseline": str(baseline_path),
            "allowed_changes": sorted(allowed),
            "all_other_fields_identical": True,
            "datasets": 97,
            "distinct_families": 97,
            "excluded_variants": len(excluded),
            "subgroups": subgroup_counts,
            "required_minima": subgroup_minima,
            "cohort_minima_met": cohort_ok,
            "roster_sha256": sha(roster_bytes),
            "membership_sha256": current["membership"]["sha256"],
            "selection_sha256": current["selection"]["sha256"],
            "payload_reads": 0,
            "budget_reset": False,
        },
    )
    plan_path = folder / "plan.json"
    save(plan_path, plan)
    try:
        q1.validate_plan(plan, prereg)
        validator = "PASS"
    except (ValueError, TypeError) as error:
        validator = str(error)
    # Actual qualification must already exist. A plan/JSON source implementation is
    # never promoted to runtime qualification by this tool.
    qualification_ok = False
    qualification_reason = "Actual source-bound runtime QUALIFIED-INPUTS.json absent"
    if handoff:
        try:
            assert (
                handoff["source_commit"] == source
                and handoff["status"] == "QUALIFIED_INPUTS_READY"
            )
            assert handoff["evaluator_host"] in ("xbabe1", "xbabe4")
            artifact = regular(handoff["evaluator_artifact"])
            assert artifact.is_relative_to(E / "evaluator") and artifact.name in (
                "jopedime",
                "jopedime-r782",
                "jopedime-evaluator",
            )
            with artifact.open("rb") as stream:
                assert stream.read(4) in (
                    b"\x7fELF",
                    b"\xcf\xfa\xed\xfe",
                    b"\xfe\xed\xfa\xcf",
                )
            assert file_sha(artifact) == evaluator, (
                "Actual evaluator artifact bytes differ"
            )
            receipt_path = regular(handoff["runtime_receipt"]["path"])
            assert file_sha(receipt_path) == handoff["runtime_receipt"]["sha256"]
            receipt = json.loads(receipt_path.read_text())
            assert (
                receipt["source_commit"] == source
                and receipt["evaluator_sha256"] == evaluator
                and receipt["backend_sha256"] == plan["backend_sha256"]
            )
            assert receipt["pins"] == plan["pins"]
            assert (
                receipt["status"] in ("PASS", "PASS_WITH_WAIVER")
                and receipt["fold_count"] == 3
                and receipt["bundle_sha256"] == BUNDLE
                and receipt["dope_sha256"] == DOPE
            )
            assert (
                type(receipt["learned_forward_count"]) is int
                and receipt["learned_forward_count"] > 0
            )
            assert (
                type(receipt["executed_qualification_checks"]) is int
                and receipt["executed_qualification_checks"] > 0
            )
            if metadata_source:
                validate_runtime_receipt(receipt, plan, pin, metadata_source)
            else:
                assert (
                    receipt["status"] == "PASS"
                    and type(receipt["failed_qualification_checks"]) is int
                    and receipt["failed_qualification_checks"] == 0
                )
            assert receipt["charged_admissions"] == {
                "heuristic": 24,
                "learned": 24,
            } and all(type(v) is int for v in receipt["charged_admissions"].values())
            closure_ref = handoff["identity_closure_receipt"]
            closure_receipt = regular(closure_ref["path"])
            assert (
                closure_receipt.is_relative_to(EVIDENCE)
                and file_sha(closure_receipt) == closure_ref["sha256"]
            )
            closure_proof = json.loads(closure_receipt.read_text())
            assert (
                closure_proof["status"]
                == ("PASS_PINNED_NAMESPACE" if metadata_source else "PASS")
                and closure_proof["source_commit"] == source
            )
            assert (
                closure_proof["training_family_manifest_sha256"]
                == plan["pins"]["training_family_manifest_sha256"]
            )
            if metadata_source:
                without_metadata = {
                    k: v for k, v in plan.items() if k != "metadata_source_commit"
                }
                assert closure_proof["qualification_plan_digest"] == q1.digest(
                    without_metadata
                )
            else:
                assert closure_proof["qualification_plan_digest"] == q1.digest(plan)
            overlap = closure_proof["overlap_counts"]
            assert set(overlap) >= {
                "family",
                "dataset_id",
                "dataset_sha256",
                "train_blob_sha256",
                "aliases",
                "split_member",
            }
            assert all(type(v) is int and v == 0 for v in overlap.values())
            assert receipt["qualified_dataset_cut_identities"] == [
                {k: d[k] for k in ("id", "dataset_sha256", "cut_sha256")}
                for d in datasets
            ]
            for log in receipt["execution_logs"]:
                log_path = regular(log["path"])
                assert log_path.is_relative_to(EVIDENCE)
                assert file_sha(log_path) == log["sha256"] and type(log["exit"]) is int
                assert log["exit"] == 0 or (
                    metadata_source
                    and log.get("id") == "M3"
                    and log["exit"] == 1
                    and log.get("disposition") == "WAIVED_V1_0_0"
                    and log["sha256"] == pin["admission_inputs"]["m3_driver"]
                )
            assert receipt["execution_logs"]
            qualification_ok = True
            qualification_reason = (
                "Actual source/identity/count/log-byte binding verified"
            )
        except (AssertionError, KeyError, ValueError, TypeError, OSError) as error:
            qualification_reason = "Runtime qualification mismatch: " + str(error)
    admission_path = E.parent / "api01-t0840/R18-ADMISSION.json"
    admission = (
        json.loads(admission_path.read_text()) if admission_path.exists() else {}
    )
    r18_admission_ok = False
    if admission.get("status") == "ADMITTED_NEW_VERIFIED_FAMILY":
        admitted = admission["candidate"]
        baseline = json.loads(
            (
                E.parent / "api01-t0830/plan.regenerated-r14-and-worker1556.json"
            ).read_text()
        )
        assert admitted["family_key"] not in {
            d["family"] for d in baseline["datasets"] if d["features"] > 20
        }
        assert admitted["dataset_id"] not in {d["id"] for d in baseline["datasets"]}
        assert any(
            d["id"] == admitted["dataset_id"]
            and d["family"] == admitted["family_key"]
            and d["features"] > 20
            for d in datasets
        )
        verified = regular(admission["qlt02_verification"]["path"])
        assert file_sha(verified) == admission["qlt02_verification"]["sha256"]
        verified_record = json.loads(verified.read_text())
        assert verified_record["all_source_pins_match"] is True
        assert (
            not str(verified_record.get("result", "")).lower().startswith("no eligible")
        )
        verified_candidate = verified_record["candidate"]
        assert (
            verified_candidate["dataset_id"] == admitted["dataset_id"]
            and verified_candidate["family_key"] == admitted["family_key"]
        )
        r18_admission_ok = (
            admission.get("qlt02_positive_new_candidate_verified") is True
        )
    signable = (
        not preview
        and not blocked
        and validator == "PASS"
        and qualification_ok
        and cohort_ok
    )
    paths = (
        git(
            "ls-tree",
            "-r",
            "--name-only",
            code_source,
            BASE,
            "crates/jope-service/src/http/runs.rs",
            "crates/jope-service/src/http/runs",
            "crates/jope-service/src/run/surrogate.rs",
            "crates/jope-service/src/run/learned_tests.rs",
            "crates/jope-engine/assets/grammar/featmath.json",
            "crates/jope-engine/assets/grammar/featmath-jain-v1.json",
            "crates/jope-engine/assets/grammar/jain-math-executable-9fa594e2.json",
            "Cargo.toml",
            "Cargo.lock",
            "research/flywheel_selfplay/qualification_lock_fifo.py",
        )
        .decode()
        .splitlines()
    )
    if metadata_source:
        for row in inputs:
            if row.get("source_commit", source) == source:
                require(
                    raw(code_source, row["path"]) == raw(source, row["path"]),
                    "metadata checkout changes a frozen scientific input",
                )
        require(
            raw(code_source, BASE + "PREREGISTRATION.md")
            == raw(source, BASE + "PREREGISTRATION.md"),
            "metadata checkout changes preregistration",
        )
        for row in roster["official_roster"]:
            path = row["cut_definition_path"]
            require(
                raw(code_source, path) == raw(source, path),
                "metadata checkout changes frozen cut bytes",
            )
    source_files = []
    for path in paths:
        data = raw(code_source, path)
        source_files.append(
            {
                "path": path,
                "source_commit": code_source,
                "git_blob": git("rev-parse", code_source + ":" + path).decode().strip(),
                "sha256": sha(data),
                "bytes": len(data),
            }
        )
    frozen = {
        "utc": now(),
        "source_commit": source,
        "tree": git("rev-parse", source + "^{tree}").decode().strip(),
        "preview_only": preview,
        "contains_merged_prs": prs,
        "source_files": source_files,
        "metadata_source_commit": code_source,
        "metadata_source_tree": git("rev-parse", code_source + "^{tree}")
        .decode()
        .strip(),
        "evaluator_build_source_commit": source,
        "validator_source": "exact reviewed metadata Git blobs; frozen compiled-source scientific assets relocated with byte pins",
        "validation_inputs": inputs,
        "private_ref": None,
    }
    if not preview:
        ref = "refs/custody/api01-q1-t0710/" + source
        existing = subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "--verify", ref],
            env=ENV,
            text=True,
            capture_output=True,
        )
        if existing.returncode == 0:
            assert existing.stdout.strip() == source
        else:
            git("update-ref", ref, source, "")
        frozen["private_ref"] = ref
    save(folder / "SOURCE-FREEZE.json", frozen)
    save(folder / "CUT-CHECKS.json", cut_checks)
    save(
        folder / "SELECTION.json",
        {
            "duplicates_resolved_before_outcomes": duplicates,
            "unknown_families_retained_without_synthesis": len(unknown),
        },
    )
    result = {
        "utc": now(),
        "status": "PRE_MERGE_PREVIEW" if preview else "SOURCE_FROZEN_PLAN_REGENERATED",
        "source_commit": source,
        "metadata_source_commit": code_source,
        "source_tree": frozen["tree"],
        "source_frozen": not preview,
        "plan_regenerated": True,
        "plan": str(plan_path),
        "plan_sha256": file_sha(plan_path),
        "plan_digest": q1.digest(plan),
        "source_freeze": str(folder / "SOURCE-FREEZE.json"),
        "datasets": len(datasets),
        "unknown_authoritative_families": len(unknown),
        "wide": sum(d["features"] > 20 for d in datasets),
        "minimum_wide_families": 4,
        "verified_three_way_cuts": len(cut_checks),
        "validator": validator,
        "runtime_qualification": qualification_reason,
        "r18_admission": admission.get("status", "WAITING_FOR_R18_VERIFICATION"),
        "r18_verified_new_family_admitted": r18_admission_ok,
        "r19_admission": "OWNER_APPROVED_EXISTING_FOUR_WIDE_FAMILIES",
        "r19_admission_ok": r19_admission_ok,
        "r19_plan_delta": str(folder / "R19-PLAN-DELTA.json"),
        "signable": signable,
        "Q1": "NOT_RUN",
        "signing_request_filed": False,
        "unmerged_prs": blocked,
        "budget_reset": False,
        "authority": "T0915 R19: outcome-blind run-only wide minimum four; identical153-cut roster; all remaining gates retained",
    }
    result.update(
        minimum_wide_families=3,
        required_families_per_subgroup=subgroup_minima,
        wide_census_sha256=q1.R20_CENSUS_SHA256,
        subgroups=subgroup_counts,
        family_deficits={
            g: subgroup_minima[g] - n
            for g, n in subgroup_counts.items()
            if n < subgroup_minima[g]
        },
        r19_admission="SUPERSEDED_BY_R20",
        r19_plan_delta=None,
        r20_plan_delta=str(folder / "R20-PLAN-DELTA.json"),
        cohort_minima_met=cohort_ok,
        selected_verified_three_way_cuts=97,
        authority="R20: all three exact eligible wide families under pinned census;97 owner-family representatives; other subgroup minima five; all four verdicts and other gates retained",
    )
    save(folder / "READINESS.json", result)
    save(E / ("PREVIEW.json" if preview else "READINESS.json"), result)
    if not preview:
        save(E / "STATE.json", result)
        original = E.parent / "api01-t0250"
        prestage = original / "PRESTAGE.json"
        prior_bytes = prestage.read_bytes()
        prior_copy = folder / "PRIOR-PRESTAGE.json"
        if not prior_copy.exists():
            prior_copy.write_bytes(prior_bytes)
        new_prestage = {
            "schema": "jope.q1-api01-t0710-prestage/v1",
            "utc": now(),
            "status": "SOURCE_FROZEN_AWAITING_QUALIFIED_SIGNED_HANDOFF",
            "source_commit": source,
            "metadata_source_commit": code_source,
            "source_tree": frozen["tree"],
            "source_files": source_files,
            "frozen_F_fold_count": 3,
            "bundle_sha256": BUNDLE,
            "dope_sha256": DOPE,
            "allowed_heavy_hosts": ["xbabe1", "xbabe4"],
            "canonical_ledger": str(LEDGER),
            "source_freeze_receipt": str(folder / "SOURCE-FREEZE.json"),
            "prior_prestage_record": str(prior_copy),
            "prior_prestage_sha256": file_sha(prior_copy),
            "runtime_qualified": qualification_ok,
            "plan_signable": signable,
            "Q1": "NOT_RUN",
        }
        pending = original / "PRESTAGE.t0710.pending.json"
        save(pending, new_prestage)
        os.replace(pending, prestage)
    if signable:
        directory = Path("/home/ubuntu/jope_plan/signing")
        name = "sol_v1_api01-QLT-04.json"
        request = {
            "kind": "q1",
            "artifact": str(plan_path),
            "sha256": result["plan_sha256"],
            "signature_out": "/home/ubuntu/jope_plan/ns/qualification/q1_approval/"
            + result["plan_digest"]
            + ".sig",
            "lane": "sol_v1_api01",
            "purpose": "QLT-04 Q1 fixed r782; R20 exact three-wide census, other minima five, threefold F; actual qualified plan. ENG signs; lane never signs.",
        }
        for bucket in ("requests", "done", "rejected"):
            p = directory / bucket / name
            if p.exists():
                assert json.loads(p.read_text()) == request, (
                    "Different request already exists; never duplicate/overwrite"
                )
                result["signing_request_filed"] = True
                break
        else:
            assert not Path(request["signature_out"]).exists()
            pending = directory / "requests" / ("." + name + ".pending")
            if pending.exists():
                assert json.loads(pending.read_text()) == request, (
                    "Unexpected incomplete request retained for audit"
                )
            else:
                with pending.open("x") as stream:
                    stream.write(json.dumps(request, indent=2) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
            os.link(pending, directory / "requests" / name)
            pending.unlink()
            result["signing_request_filed"] = True
        save(folder / "READINESS.json", result)
        save(E / "READINESS.json", result)
        save(E / "STATE.json", result)
    assert not markers(), (
        "Concurrent candidate start: retain evidence, never automatically retry"
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["freeze", "preview"])
    parser.add_argument("--source")
    parser.add_argument("--metadata-source")
    args = parser.parse_args()
    os.setpriority(os.PRIO_PROCESS, 0, 19)
    with (E / "FREEZE.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        result = regenerate(
            preview=args.mode == "preview",
            source_override=args.source,
            metadata_source=args.metadata_source,
        )
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result.get("plan_regenerated") else 3)
