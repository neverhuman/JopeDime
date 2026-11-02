"""Namespace-injected implementation pieces for the reviewed Q1 publisher."""

def _authorize_module_source( source, frozen, _namespace):
    FROZEN_SOURCE = _namespace['FROZEN_SOURCE']
    git = _namespace['git']
    re = _namespace['re']
    require = _namespace['require']
    subprocess = _namespace['subprocess']
    require(
        isinstance(source, str) and re.fullmatch(r"[0-9a-f]{40}", source),
        "full module source SHA required",
    )
    if frozen:
        require(source == FROZEN_SOURCE, "module requires the frozen compiled source")
    else:
        try:
            git("merge-base", "--is-ancestor", FROZEN_SOURCE, source)
            git("merge-base", "--is-ancestor", source, "origin/main")
        except subprocess.CalledProcessError as error:
            raise ValueError(
                "module metadata source is not reviewed main ancestry"
            ) from error
def _module_filename( name, source, relative, filename, _namespace):
    """Allow only the existing canonical or relocated scientific-asset context."""
    E = _namespace['E']
    FROZEN_SOURCE = _namespace['FROZEN_SOURCE']
    Path = _namespace['Path']
    REPO = _namespace['REPO']
    _authorize_module_source = _namespace['_authorize_module_source']
    require = _namespace['require']
    filename = Path(filename)
    require(
        filename.is_absolute()
        and ".." not in filename.parts
        and not any(p.is_symlink() for p in (filename, *filename.parents)),
        "unsafe module filename",
    )
    if name in {"common", "m3_full", "_q1_waiver_gates"}:
        require(
            filename == REPO / relative, "module filename differs from fixed Git path"
        )
    else:
        require(
            filename.is_relative_to(E),
            "module filename outside original publisher state",
        )
        parts = filename.relative_to(E).parts
        require(
            len(parts) in (4, 6)
            and parts[0] in {"preview", "frozen"}
            and parts[1] == FROZEN_SOURCE
            and parts[-2:] == ("validation-inputs", Path(relative).name),
            "unreviewed module asset context",
        )
        if len(parts) == 6:
            require(parts[2] == "metadata", "module metadata context required")
            _authorize_module_source(parts[3], False)
            require(
                name == "q1_dedup" or parts[3] == source,
                "module metadata context/source differs",
            )
        else:
            require(source == FROZEN_SOURCE, "relocated module source differs")
    return filename
def pin_materials( source, metadata_source, main_source, _namespace):
    """Read the reviewed descriptor separately from its actual compiled source."""
    BASE = _namespace['BASE']
    Path = _namespace['Path']
    git = _namespace['git']
    json = _namespace['json']
    raw = _namespace['raw']
    re = _namespace['re']
    require = _namespace['require']
    sha = _namespace['sha']
    pin_source = metadata_source or source
    require(
        re.fullmatch(r"[0-9a-f]{40}", pin_source) is not None,
        "metadata source must be a full SHA",
    )
    git("merge-base", "--is-ancestor", pin_source, main_source)
    if metadata_source:
        git("merge-base", "--is-ancestor", source, pin_source)
    pin_bytes = raw(pin_source, BASE + "evaluator-pin.json")
    pin = json.loads(pin_bytes)
    require(
        pin["schema"] == "jope.q1-evaluator-pin/v1"
        and pin["status"] == "BUILD_IDENTITY_ONLY_NOT_RUNTIME_QUALIFICATION",
        "unsupported evaluator build pin",
    )
    if metadata_source:
        require(
            pin["source_commit"] == source
            and pin["source_tree"]
            == git("rev-parse", source + "^{tree}").decode().strip(),
            "pin compiled source/tree differs",
        )
        require(
            pin["sha256"]
            == "4e9239a1126589a42c3d80b1742d463d1b2331def59e5a08796169b4f3cec177"
            and pin["target"] == "x86_64-unknown-linux-gnu",
            "unreviewed Linux replacement identity",
        )
    else:
        require(
            pin["sha256"]
            == "76721923b663dd2ad2f9fe82c787791e2b97ec32426efa5eab8ec81f146ab0a6",
            "replacement pin requires explicit metadata source",
        )
    materials = [("evaluator-pin.json", pin_bytes)]
    require(
        len(pin["reproductions"]) == 2,
        "two recorded compilations required; independence scope stays explicit",
    )
    for ref in (pin["recipe"], *pin["reproductions"]):
        relative = Path(ref["path"])
        require(
            not relative.is_absolute() and ".." not in relative.parts,
            "pin metadata path escapes qualification root",
        )
        data = raw(pin_source, BASE + relative.as_posix())
        require(
            sha(data) == ref["sha256"], "evaluator reproduction/recipe bytes differ"
        )
        materials.append((relative.as_posix(), data))
    return pin, pin_source, materials
def validate_runtime_receipt( receipt, plan, pin, metadata_source, _namespace):
    """Validate the pinned measured components; retain one raw M3 failure."""
    BUNDLE = _namespace['BUNDLE']
    DOPE = _namespace['DOPE']
    EVIDENCE = _namespace['EVIDENCE']
    REPO = _namespace['REPO']
    file_sha = _namespace['file_sha']
    json = _namespace['json']
    module = _namespace['module']
    regular = _namespace['regular']
    require = _namespace['require']
    sys = _namespace['sys']
    require(
        receipt.get("status") == "PASS_WITH_WAIVER"
        and receipt.get("release_version") == "v1.0.0",
        "Linux admission requires explicit v1.0.0 waiver disposition",
    )
    source = plan["source_commit"]
    evaluator = plan["pins"]["evaluator_sha256"]
    require(
        receipt["source_commit"] == source
        and receipt.get("metadata_source_commit") == metadata_source
        and receipt["evaluator_sha256"] == evaluator
        and receipt["backend_sha256"] == plan["backend_sha256"]
        and receipt["pins"] == plan["pins"],
        "runtime receipt source/evaluator/backend/pins differ",
    )
    require(
        receipt["fold_count"] == 3
        and receipt["bundle_sha256"] == BUNDLE
        and receipt["dope_sha256"] == DOPE
        and receipt["charged_admissions"] == {"heuristic": 24, "learned": 24},
        "runtime fold/model/budget identity differs",
    )
    require(
        receipt["qualified_dataset_cut_identities"]
        == [
            {k: d[k] for k in ("id", "dataset_sha256", "cut_sha256")}
            for d in plan["datasets"]
        ],
        "runtime selected dataset/cut identities differ",
    )
    require(
        plan.get("metadata_source_commit") == metadata_source
        and pin["source_commit"] == source
        and pin["sha256"] == evaluator,
        "compiled and metadata sources are not explicitly bound",
    )
    expected = {
        "native",
        "native_driver",
        "m2",
        "m2_report",
        "m3_raw",
        "m3_driver",
        "link_inputs",
        "identity_closure",
        "reproduction",
    }
    require(
        set(pin["admission_inputs"]) == expected,
        "reviewed admission pin inventory differs",
    )
    require(
        set(receipt["components"]) == expected,
        "complete reviewed component inventory required",
    )
    data = {}
    for name, ref in receipt["components"].items():
        path = regular(ref["path"])
        require(
            path.is_relative_to(EVIDENCE) and path.stat().st_size < 8 * 1024 * 1024,
            "component must be bounded qualification evidence",
        )
        require(
            ref["sha256"] == pin["admission_inputs"][name] == file_sha(path),
            "reviewed component bytes differ: " + name,
        )
        data[name] = json.loads(path.read_text())
    for name in [
        "native",
        "native_driver",
        "m2",
        "m3_driver",
        "link_inputs",
        "identity_closure",
        "reproduction",
    ]:
        item = data[name]
        require(
            item.get("source_commit") == source
            and item.get("evaluator_sha256", item.get("artifact_sha256")) == evaluator,
            "component source/evaluator differs: " + name,
        )
    native = data["native"]
    require(
        native["status"] == "PASS_PRODUCTION_NATIVE_COMPONENT"
        and native["backend_sha256"] == plan["backend_sha256"]
        and native["version"]["source_sha40"] == source,
        "actual native runtime/source binding required",
    )
    cases = native["qualification_cases"]
    require(
        len(cases) == 5
        and len({x["id"] for x in cases}) == 5
        and all(x["passed"] is True for x in cases),
        "all five native cases must pass",
    )
    observations = {x["policy"]: x for x in native["observations"]}
    require(
        set(observations) == {"heuristic", "learned"}
        and all(
            x["charged"] == x["completed"] == 24
            and x["folds_per_completed"] == 3
            and x["exact_terminal_replay"] is True
            and x["adapter_exit"] == 0
            for x in observations.values()
        ),
        "native budget/fold/terminal replay differs",
    )
    require(
        receipt["learned_forward_count"] == native["activity"]["learned_forward_count"],
        "actual learned-forward count differs",
    )
    require(
        native["stop"]["engine_exit"] == 0
        and native["stop"]["owned_group_processes_remaining"] == []
        and native["activity"]["learned_forward_count"] > 0,
        "native activity/teardown required",
    )
    require(
        data["native_driver"]["exit"] == 0
        and data["m3_driver"]["exit"] == 1
        and data["m3_driver"]["status"] == "EXCLUSIVE_RELEASED",
        "raw native/M3 driver exits must be retained",
    )
    m2 = data["m2"]
    report = data["m2_report"]
    require(
        m2["status"] == "PASS_EXACT_ORIGINAL_LINK_INPUTS"
        and m2["compile_exit"] == m2["replay_exit"] == 0,
        "M2 original-library replay must pass",
    )
    require(
        m2["original_link_inputs_sha256"] == pin["admission_inputs"]["link_inputs"]
        and m2["report_sha256"] == pin["admission_inputs"]["m2_report"],
        "M2 original link/report binding differs",
    )
    require(
        report["source"] == source
        and report["passed"] is True
        and report["contexts"] == 50
        and report["slates"] == 100
        and report["failed_values"] == 0
        and report["optimizer_updates"] == 0,
        "complete unchanged M2 comparisons required",
    )
    repro = data["reproduction"]
    require(
        repro["verdict"] == "MATCH"
        and repro["reused_cargo_units"] == 0
        and repro["compiled_units"] == 288
        and repro["source_clean_after"] is True,
        "clean compilation reproduction required; no second-owner claim inferred",
    )
    raw_m3 = data["m3_raw"]
    require(
        raw_m3["status"] == "NUMERICAL_FAIL" and raw_m3["source_sha"] == source,
        "raw M3 failure/source must be retained",
    )
    result = dict(raw_m3, status="FAIL", artifact_sha256=evaluator)
    names = ["common", "m3_full", "_q1_waiver_gates"]
    previous = {name: sys.modules.get(name) for name in names}
    try:
        for name in names[:2]:
            path = "deploy/qualification/v1.0/providers/" + name + ".py"
            module(name, source, REPO / path)
        path = "deploy/qualification/v1.0/gates_receipt.py"
        adapter = module(names[2], metadata_source, REPO / path)
        waiver = adapter._m3_waiver(
            result, regular(receipt["m3_waiver_ruling"]), "v1.0.0"
        )
    finally:
        for name, old in previous.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old
    require(
        receipt.get("M3_raw_status") == "NUMERICAL_FAIL"
        and receipt.get("M3_disposition") == waiver["disposition"],
        "raw M3 and separate waiver disposition required",
    )
    counts = {
        "executed_qualification_checks": 10,
        "passed_qualification_checks": 9,
        "failed_qualification_checks": 1,
        "waived_qualification_checks": 1,
        "unwaived_failed_qualification_checks": 0,
    }
    require(
        all(type(receipt.get(k)) is int and receipt[k] == v for k, v in counts.items()),
        "raw failed=1/waived=1/unwaived=0 counts required",
    )
    closure = data["identity_closure"]
    require(
        closure["status"] == "PASS_PINNED_NAMESPACE"
        and closure["training_family_manifest_sha256"]
        == plan["pins"]["training_family_manifest_sha256"],
        "reviewed namespace closure required",
    )
    require(
        all(type(v) is int and v == 0 for v in closure["overlap_counts"].values()),
        "training/qualification namespace overlap",
    )
    selected = {x["id"]: x for x in closure["selected_bindings"]}
    require(
        set(selected) == {x["id"] for x in plan["datasets"]},
        "complete selected dataset/cut bindings required",
    )
    for row in plan["datasets"]:
        require(
            all(
                selected[row["id"]][key] == row[key]
                for key in ["dataset_sha256", "cut_sha256"]
            )
            and selected[row["id"]]["family"] == "fam-" + row["family"],
            "namespace dataset/family/cut binding differs",
        )
    return waiver
