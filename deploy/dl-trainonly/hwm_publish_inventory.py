"""Payload validation and inventory projection for the train-only HWM publisher."""

import sys

_parent = sys.modules.get("hwm_publish", sys.modules["__main__"])
BINARIES = _parent.BINARIES
C2_HEADER_SCHEMA = _parent.C2_HEADER_SCHEMA
G3_DECISIONS_SCHEMA = _parent.G3_DECISIONS_SCHEMA
G3_DECISIONS_SHA256 = _parent.G3_DECISIONS_SHA256
HANDOFF_SHA = _parent.HANDOFF_SHA
HEADER_SCHEMA = _parent.HEADER_SCHEMA
LANES = _parent.LANES
OBJECT_SCHEMA = _parent.OBJECT_SCHEMA
PLAN_SCHEMA = _parent.PLAN_SCHEMA
ROLELESS = _parent.ROLELESS
CUT = _parent.CUT
DSL = _parent.DSL
Path = _parent.Path
Sources = _parent.Sources
catalog_roles = _parent.catalog_roles
clean_output = _parent.clean_output
collections = _parent.collections
common = _parent.common
decode = _parent.decode
digest = _parent.digest
finite = _parent.finite
read_regular = _parent.read_regular
re = _parent.re
require = _parent.require
s3io = _parent.s3io
sha_field = _parent.sha_field
one_holdout = _parent.one_holdout
__file__ = _parent.__file__


def selected_payload(final, row, event, header, header_sha, final_sha, family_source=None, role_source=None):
    fields = (
        "context_sha256",
        "cut_id",
        "dataset_hash",
        "fold",
        "ledger",
        "use",
        "view_id",
        "set",
        "p3",
    )
    for name in fields:
        require(final.get(name) == row[name], "input/final identity mismatch: " + name)
    for name in ("context_sha256", "cut_id", "use", "view_id", "set", "outcome"):
        require(
            event.get(name) == final.get(name),
            "ledger/final identity mismatch: " + name,
        )
    require(
        final.get("schema") == "jope.dl-s1-prod-context-final/v1",
        "unsupported final schema",
    )
    require(final.get("header_sha256") == header_sha, "wrong frozen header")
    require(
        final.get("s1_input_sha256") == row["s1_input"]["sha256"],
        "wrong stripped S1 input",
    )
    require(row.get("shape", {}).get("k") == 2, "S1 requires exactly two folds")
    require(
        final.get("hwm", {}).get("cut_id") == row["cut_id"]
        and final["hwm"].get("comparable_only_with_same_cut_id") is True,
        "wrong HWM comparison",
    )
    require(CUT.fullmatch(row["cut_id"]), "unsupported cut identity")
    require(
        re.fullmatch(r"[0-9a-f]{16}", row["dataset_hash"])
        and type(row["fold"]) is int
        and 0 <= row["fold"] < 7,
        "bad dataset/fold identity",
    )
    require(
        final.get("binaries")
        == {k: header["binaries"][v] for k, v in BINARIES.items()},
        "unadmitted binary",
    )
    attempts = final["attempts"]
    require(
        type(attempts["charged"]) is int
        and 0
        <= attempts["charged"]
        <= header["plan_from_pinned_s1_budget"]["per_context"],
        "over-budget attempts",
    )
    require(
        attempts["finished"]
        == attempts["charged"]
        == sum(attempts["by_status"].values())
        and all(type(n) is int and n >= 0 for n in attempts["by_status"].values()),
        "incomplete attempt accounting",
    )
    require(
        event["attempts"]
        == {"by_status": attempts["by_status"], "charged": attempts["charged"]},
        "ledger attempts mismatch",
    )
    g0 = final["g0"]["manifest"]
    require(
        g0["context_sha256"] == row["context_sha256"]
        and g0["selection_protocol"] == "jope.s1-two-fold/v1",
        "wrong G0 provenance",
    )
    require(
        type(g0["attempt_accounting_verified"]) is bool
        and type(g0["sampling_resume_qualified"]) is bool,
        "missing G0 qualification",
    )
    obj = {
        "schema": OBJECT_SCHEMA,
        "context_sha256": row["context_sha256"],
        "cut_id": row["cut_id"],
        "dataset_hash": row["dataset_hash"],
        "fold": row["fold"],
        "scope": row["use"],
        "rights": row["ledger"],
        "family": row["p3"],
        "outcome": final["outcome"],
        "provenance": {
            "header_sha256": header_sha,
            "final_sha256": final_sha,
            "input_sha256": sha_field(row["s1_input"]["sha256"]),
            "context_json_sha256": sha_field(row["context_json_sha256"]),
            "summary_sha256": sha_field(final["summary_tar_gz_sha256"]),
            "binaries": final["binaries"],
        },
        "attempts": {
            "charged": attempts["charged"],
            "finished": attempts["finished"],
            "by_status": attempts["by_status"],
        },
        "g0_qualification": {
            k: g0[k]
            for k in ("attempt_accounting_verified", "sampling_resume_qualified")
        },
        "authority": "progress_evidence_only",
        "selection_protocol": "jope.s1-two-fold/v1",
    }
    if family_source is not None:
        # A family/role decided outside the header-pinned registry/holdout names its decision file. Registry-sourced records stay
        # byte-identical to the merged publisher (their source is already bound through provenance.header_sha256).
        obj["provenance"]["family_source"] = family_source
    if role_source is not None:
        # ONE-HOLDOUT: a P3 heldout context published as training names the pinned role map that made it training data.
        obj["provenance"]["role_source"] = role_source
    # Status names are the only free keys admitted to the sanitized payload.
    require(
        all(re.fullmatch(r"[a-z_]{1,64}", k) for k in attempts["by_status"]),
        "invalid attempt status",
    )
    if final["outcome"] == "selection_reached":
        sel = final["selection"]
        require(
            sel["leader_test_rmse_field"] is None, "selection carries a forbidden score"
        )
        require(
            sel["engine_status_schema"] == "jope.island-engine-status/s1-v1"
            and g0["ready"] is True,
            "wrong selection protocol",
        )
        ratio, fitness = finite(sel["cv_geomean_rmse_ratio"]), finite(sel["fitness"])
        require(0 <= ratio <= 0.90, "CV gate not met")
        program = sel["program"]
        text, ph = program["dsl"], sha_field(sel["program_sha256"])
        require(
            isinstance(text, str)
            and DSL.fullmatch(text)
            and not re.search(r"\b(test|target|label)\b", text, re.IGNORECASE),
            "non-program data in DSL",
        )
        require(
            digest(text.encode())
            == ph
            == program["program_sha256"]
            == event["selected_program_sha256"],
            "wrong selected program",
        )
        require(
            event["selected_cv_ratio"] == ratio
            and event["selected_fitness"] == fitness,
            "ledger score mismatch",
        )
        require(
            isinstance(program["parents"], list) and len(program["parents"]) <= 2,
            "invalid parent list",
        )
        parents = [sha_field(p) for p in program["parents"]]
        require(
            type(program["birth_seq"]) is int
            and program["birth_seq"] >= 0
            and re.fullmatch(r"[a-z_]{1,64}", program["operator"]),
            "invalid variation metadata",
        )
        obj["selection"] = {
            "cv_geomean_rmse_ratio": ratio,
            "fitness": fitness,
            "program": {
                "dsl": text,
                "program_sha256": ph,
                "parents": parents,
                "operator": program["operator"],
                "birth_seq": program["birth_seq"],
            },
        }
    else:
        require(
            final["outcome"] == "g0_not_ready"
            and "selection" not in final
            and all(
                event.get(k) is None
                for k in (
                    "selected_program_sha256",
                    "selected_cv_ratio",
                    "selected_fitness",
                )
            ),
            "unknown terminal outcome",
        )
    clean_output(obj)
    return obj


def inventory(source, header_sha, ledger_pins=None):
    """Authenticate independently pinned inputs and append-ledger finals, then project metadata.

    No traversal of contexts, caches, summary members or input tables. Only finals
    committed by the two collector ledgers enter the frozen publication set.
    """
    source = Path(source).absolute()
    src = Sources()
    header = src.doc(source / "ledger.header.json", header_sha)
    require(
        header.get("schema") in (HEADER_SCHEMA, C2_HEADER_SCHEMA)
        and header.get("selection", "").startswith("inner-CV only on inner-train"),
        "not an admitted train-only S1 ledger",
    )
    require(
        header["plan_from_pinned_s1_budget"]["protocol"] == "jope.s1-fixed-equal/v1",
        "unadmitted budget protocol",
    )
    # Read exactly the lanes the header declares (a header without a ledgers map declares both, as before).
    declared = header.get("ledgers", dict.fromkeys(LANES))
    require(
        isinstance(declared, dict) and declared and set(declared) <= set(LANES),
        "unknown ledger lanes",
    )
    ledger_rows, pins = {}, {}
    for lane in [name for name in LANES if name in declared]:
        p = source / f"ledger-{lane}.jsonl"
        pin = (ledger_pins or {}).get(lane)
        data = src.read(
            p, pin["sha256"] if pin else None, pin["bytes"] if pin else None
        )
        src.refs[str(p.absolute())]["prefix"] = True
        require(data and data.endswith(b"\n"), "torn ledger snapshot")
        rows = [decode(line) for line in data.splitlines()]
        require(
            rows[0].get("event") == "open"
            and rows[0].get("header_sha256") == header_sha
            and rows[0].get("ledger") == lane,
            "wrong collector ledger",
        )
        ledger_rows[lane] = rows
        pins[lane] = {"sha256": digest(data), "bytes": len(data)}
    input_ref = header["input_list"]
    lists = [src.doc(input_ref["path"], input_ref["sha256"])]
    joined = set()
    for rows in ledger_rows.values():
        for event in rows:
            if event["event"] != "join" or event["join_record_sha256"] in joined:
                continue
            joined.add(event["join_record_sha256"])
            join = src.doc(event["join_record"], event["join_record_sha256"])
            require(
                join["schema"] == "jope.dl-s1-prod-join/v1"
                and join["header_sha256"] == header_sha,
                "wrong input join",
            )
            lists.append(src.doc(join["input_list"], event["input_list_sha256"]))
    deny = src.doc(header["denylist"]["path"], header["denylist"]["sha256"])
    p3 = header["p3"]
    holdout = src.doc(p3["holdout"], p3["holdout_sha256"])
    registry = src.doc(p3["registry"], p3["registry_sha256"])
    refused = {r["view_id"] for r in deny["contexts"]}
    ds_family = {}
    for fid, family in registry["families"].items():
        for member in family.get("members") or []:
            ds = member.get("dataset_hash")
            if ds:
                require(
                    ds not in ds_family or ds_family[ds] == fid,
                    "ambiguous family registry",
                )
                ds_family[ds] = fid
    assignment = dict(holdout["assignment"])
    registry_ds = set(ds_family)
    g3 = header.get("g3_decisions")
    g3_source = None
    if g3 is not None:
        # Appended identities only: the pinned registry/holdout keep precedence (a dataset or family they already know is never
        # re-assigned), exactly as the campaign-2 input list was built; a ROLELESS_BRIDGE is never an admitted role.
        require(header.get("schema") == C2_HEADER_SCHEMA, "G3 decisions bind only a campaign-2 header")
        require(isinstance(g3, dict) and set(g3) == {"path", "sha256"}, "invalid G3 decisions pin")
        g3_pin = _parent.G3_DECISIONS_SHA256
        require(sha_field(g3["sha256"]) == g3_pin, "not the owner-named G3 decision file")
        decisions = src.doc(g3["path"], sha_field(g3["sha256"]))
        g3_source = {"path": str(Path(g3["path"]).absolute()), "sha256": g3_pin}
        require(decisions.get("schema") == G3_DECISIONS_SCHEMA, "unsupported G3 decisions schema")
        identities = decisions.get("per_identity")
        require(isinstance(identities, list) and identities, "G3 decisions carry no identities")
        for ident in identities:
            ds, fid, role = ident.get("dataset_hash"), ident.get("family_id"), ident.get("role")
            require(
                isinstance(ds, str) and re.fullmatch(r"[0-9a-f]{16}", ds)
                and isinstance(fid, str) and re.fullmatch(r"fam-[0-9a-f]{64}", fid)
                and isinstance(role, str) and re.fullmatch(r"[A-Za-z_]{1,32}", role),
                "invalid G3 decision identity",
            )
            ds_family.setdefault(ds, fid)
            assignment.setdefault(fid, ROLELESS.get(role, role))
    admitted, catalog_of, training_of = {}, {}, {}
    for doc in lists:
        catalog = catalog_roles(src, doc["catalog_p3"]) if "catalog_p3" in doc else None
        training, role_source = one_holdout(src, doc["one_holdout"]) if "one_holdout" in doc else (None, None)
        for row in doc["admitted"]:
            cid = sha_field(row["context_sha256"])
            require(
                cid not in admitted or (admitted[cid] == row and catalog_of.get(cid) == catalog),
                "conflicting admitted input",
            )
            admitted[cid] = row
            if catalog is not None:
                catalog_of[cid] = catalog
            if training is not None and cid in training:
                training_of[cid] = (training[cid], role_source)
    objects, counts, seen = [], collections.Counter(), set()
    for lane, rows in ledger_rows.items():
        assigned = {}
        for event in rows:
            if event["event"] == "assign":
                assigned[event["context_sha256"]] = event
            if event["event"] != "final":
                continue
            cid = sha_field(event["context_sha256"])
            require(
                cid not in seen and cid in admitted and cid in assigned,
                "duplicate/unadmitted/unassigned final",
            )
            seen.add(cid)
            row = admitted[cid]
            require(
                row["ledger"] == lane
                and assigned[cid]["cut_id"] == row["cut_id"]
                and assigned[cid]["use"] == row["use"],
                "wrong participant/scope",
            )
            ds = row["dataset_hash"]
            family, role, family_source = ds_family.get(ds), None, None
            if family is not None:
                role = assignment.get(family)
                if ds not in registry_ds:
                    family_source = g3_source  # an appended G3 identity (campaign-2 header)
            elif cid in catalog_of:
                # A catalog dataset unknown to the registry: family from the list's approved decision file(s). A family that the
                # pinned holdout (or the G3 decisions) already assigns keeps that role: registry/holdout precedence, exactly as the
                # S1 input prep resolved it. The record names the decision file the row's p3_source cites (else the first).
                decided = catalog_of[cid].get(ds)
                require(decided is not None, "dataset absent from the catalog decisions")
                family, role = decided[0], assignment.get(decided[0], decided[1])
                cited = str(row.get("p3_source", "")).rpartition(" ")[2]
                family_source = next((x for x in decided[2] if cited and x["sha256"].startswith(cited)), decided[2][0])
            require(
                isinstance(family, str) and re.fullmatch(r"fam-[0-9a-f]{64}", family),
                "invalid family identity",
            )
            # ONE-HOLDOUT: heldout is a training role only for a context its list's pinned role map names, with this P3.
            heldout_training = role == "heldout" and cid in training_of and training_of[cid][0] == row["p3"]
            require(
                row["p3"] == {"family_id": family, "assignment": role}
                and (role in ("train_fit", "train_nonpool", "dev") or heldout_training)
                and row["view_id"] not in refused,
                "unadmitted family/input",
            )
            require(
                row["use"] == ("selection_only" if role == "dev" else "train_teacher"),
                "scope escalation",
            )
            data = src.read(
                source / "contexts" / cid / "final.json", event["final_sha256"]
            )
            final = decode(data)
            require(
                final["summary_tar_gz_sha256"] == event["summary_sha256"],
                "wrong summary reference",
            )
            # Hash the sealed summary as an opaque object, never decompress or extract it.
            summary = src.read(
                source / "contexts" / cid / "summary.tar.gz", event["summary_sha256"]
            )
            require(len(summary) == final["summary_bytes"], "wrong summary size")
            obj = selected_payload(final, row, event, header, header_sha, digest(data), family_source,
                                   training_of[cid][1] if heldout_training else None)
            body = common.canonical(obj)
            outcome = obj["outcome"]
            counts[outcome] += 1
            prefix = (
                s3io.V2_RESEARCH_ONLY_KEY if lane == "research" else s3io.V2_PREFIX_KEY
            )
            kind = "hwm" if outcome == "selection_reached" else "terminal"
            key = f"{prefix}gp-{kind}/{row['dataset_hash']}/f{row['fold']}/{cid}/{digest(body)}.json"
            objects.append({"key": key, "body": obj})
    objects.sort(key=lambda o: o["key"])
    plan = {
        "schema": PLAN_SCHEMA,
        "adapter_sha256": digest(read_regular(__file__)),
        "custody_sha256": HANDOFF_SHA,
        "source": str(source),
        "header_sha256": header_sha,
        "ledgers": pins,
        "sources": sorted(src.refs.values(), key=lambda r: r["path"]),
        "counts": dict(sorted(counts.items())),
        "objects_sha256": digest(common.canonical(objects)),
        "artifact_count": len(objects),
    }
    return plan, objects
