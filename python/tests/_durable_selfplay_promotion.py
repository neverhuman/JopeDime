"""Test group re-exported by test_durable_selfplay for stable test IDs."""

from test_durable_selfplay import *  # noqa: F403

def test_dual_tier_promotion_keeps_internal_and_milestone_thresholds_separate():
    internal = evaluate_gate(gate_evidence(0.002))
    assert internal["pass"]
    assert internal["promotion_tier"] == "internal"
    assert not internal["milestone_pass"]
    milestone = evaluate_gate(gate_evidence(0.01))
    assert milestone["pass"]
    assert milestone["promotion_tier"] == "milestone"

def test_a_non_conditional_checkpoint_can_never_be_promoted():
    """val_conditional_advantage <= 0 means the head ignores the dataset."""

    dead = gate_evidence(0.05)
    dead["candidate"]["val_conditional_advantage"] = -0.060
    result = evaluate_gate(dead)
    assert not result["pass"]
    assert not result["milestone_pass"]
    assert result["promotion_tier"] == "rejected"
    assert "candidate_policy_is_not_conditional" in result["reasons"]
    assert "candidate_policy_is_not_conditional" in result["milestone_reasons"]
    assert result["candidate_conditional_advantage"] == -0.060

    zero = gate_evidence(0.05)
    zero["candidate"]["val_conditional_advantage"] = 0.0
    assert "candidate_policy_is_not_conditional" in evaluate_gate(zero)["reasons"]

def test_conditionality_margins_separate_internal_from_milestone():
    thin = gate_evidence(0.05)
    thin["candidate"]["val_conditional_advantage"] = 0.004
    thin["baseline"]["val_conditional_advantage"] = -0.02
    result = evaluate_gate(thin)
    assert not result["pass"]
    assert "conditional_advantage_below_half_point" in result["reasons"]

    internal = gate_evidence(0.05)
    internal["candidate"]["val_conditional_advantage"] = 0.03
    internal["baseline"]["val_conditional_advantage"] = 0.02
    result = evaluate_gate(internal)
    assert result["promotion_tier"] == "internal"
    assert result["conditional_advantage_lift"] == pytest.approx(0.01)
    assert "conditional_advantage_lift_below_two_points" in result["milestone_reasons"]

def test_unmeasured_conditionality_is_fail_closed_not_zero():
    for side in ("candidate", "baseline"):
        evidence = gate_evidence(0.05)
        del evidence[side]["val_conditional_advantage"]
        result = evaluate_gate(evidence)
        assert not result["pass"]
        assert f"{side}_conditional_advantage_evidence_missing" in result["reasons"]
        assert result[f"{side}_conditional_advantage"] is None
        assert result["conditional_advantage_lift"] is None
        # Missing evidence is not silently reported as a non-conditional policy.
        assert "candidate_policy_is_not_conditional" not in result["reasons"]

    null_valued = gate_evidence(0.05)
    null_valued["candidate"]["val_conditional_advantage"] = None
    result = evaluate_gate(null_valued)
    assert "candidate_conditional_advantage_evidence_missing" in result["reasons"]

def test_new_task_verdicts_are_advisory_until_anchored():
    """U3: an unanchored branch is recorded (`"advisory": true`) but can never
    alter `pass` / `promotion_tier` / the global reasons; an accepted anchor in
    accepted-by-task.json flips it to gating, exactly like a legacy branch."""

    evidence = gate_evidence(0.05)
    result = evaluate_gate(evidence)
    assert result["pass"] is True and result["promotion_tier"] == "milestone"
    assert result["task_verdicts"]["regression"]["advisory"] is False
    for task in ("timeseries", "multiclass"):
        verdict = result["task_verdicts"][task]
        assert verdict["advisory"] is True
        assert verdict["pass"] is False
        assert verdict["pairwise_lift"] is None
        assert result["task_pairwise_deltas"][task] is None
    assert result["tasks_passing"] == ["regression", "binary"]

    # Advisory evidence, even a hard regression, never blocks the global gate.
    evidence["baseline"]["timeseries_pairwise_accuracy"] = 0.6
    evidence["candidate"]["timeseries_pairwise_accuracy"] = 0.5
    result = evaluate_gate(evidence)
    assert result["pass"] is True and result["promotion_tier"] == "milestone"
    verdict = result["task_verdicts"]["timeseries"]
    assert verdict["advisory"] is True
    assert "timeseries_regressed_more_than_half_point" in verdict["reasons"]
    assert not any(r.startswith("timeseries_") for r in result["milestone_reasons"])

    # Once anchored, the same evidence gates.
    anchor = {"tasks": {"timeseries": {"checkpoint_sha256": "c" * 64}}}
    gated = evaluate_gate(evidence, anchor)
    assert gated["pass"] is False and gated["promotion_tier"] == "rejected"
    assert "timeseries_regressed_more_than_half_point" in gated["reasons"]
    assert gated["task_verdicts"]["timeseries"]["advisory"] is False
    assert gated["task_verdicts"]["multiclass"]["advisory"] is True

def test_conditionality_gate_parity_fixture_matches_python():
    """The Rust port in src/bin/jope_selfplay_promotion.rs asserts the same file.

    Regenerate after an intentional gate change with
    JOPE_REGENERATE_PARITY_FIXTURE=1, then re-run `cargo test` for the port.
    """

    fixture = json.loads(PARITY_FIXTURE.read_text())
    regenerate = os.environ.get("JOPE_REGENERATE_PARITY_FIXTURE") == "1"
    for case in fixture["cases"]:
        produced = json.dumps(evaluate_gate(case["evidence"]), sort_keys=True, separators=(",", ":"))
        if regenerate:
            case["expected_gate_compact_json"] = produced
            continue
        assert produced == case["expected_gate_compact_json"], case["name"]
    if regenerate:
        PARITY_FIXTURE.write_text(json.dumps(fixture, indent=2, sort_keys=True) + "\n")

def test_every_wave_seed_has_the_exact_256_slot_composition():
    seed = wave_seed(
        task="regression",
        dataset="dataset",
        wave_id="strict-wave-000001",
        local=[candidate_row(index, "local") for index in range(32)],
        dl=[candidate_row(index + 100, "dl") for index in range(64)],
        common=[candidate_row(index + 200, "common") for index in range(96)],
    )
    assert len(seed) == 256
    assert {
        group: sum(row["slot_group"] == group for row in seed)
        for group in ("local_elite", "dl_proposal", "common_winner", "offspring")
    } == {
        "local_elite": 32,
        "dl_proposal": 64,
        "common_winner": 96,
        "offspring": 64,
    }

def test_controller_has_no_file_waiting_production_phases():
    handlers = default_handlers()
    assert handlers["generate_gate_proposals"].__name__ == "_generate_gate_proposals"
    assert handlers["evaluate_frozen_gate"].__name__ == "_evaluate_frozen_gate"
    assert handlers["run_sotarena_qualification"].__name__ == "_run_sotarena_qualification"

def test_sealed_audit_is_one_shot_and_never_training_eligible(tmp_path):
    registry = SealedAuditRegistry(tmp_path / "sealed.sqlite")
    entry = {
        "task": "regression",
        "dataset_hash": "unseen",
        "source_family": "unseen-family",
        "train_sha256": "1" * 64,
        "test_sha256": "2" * 64,
        "incumbent_score": 0.5,
    }
    audit_id = registry.register(
        entry,
        checkpoint_sha256="3" * 64,
        forbidden_datasets=set(),
        forbidden_families=set(),
    )
    registry.emit(audit_id, "(pipe X (ridge))")
    with pytest.raises(RuntimeError, match="exactly one"):
        registry.emit(audit_id, "(pipe X (standardize) (ridge))")
    result = registry.score(
        audit_id,
        score=0.6,
        receipt_sha256="4" * 64,
        compiled_interpreted_parity=True,
        prediction_parity=True,
    )
    assert result["super_human"]
    assert not result["training_ingestion_eligible"]
    registry.close()

def test_systemd_services_are_independently_restartable():
    # ADR-0003 (PR #13/#32) turned every unit into a `.service.in` template rendered by
    # deploy/systemd/render.sh, so the units are named `*.service.in` and spell the bundle
    # directory as @JOPE_BUNDLE@ instead of a literal runtime-bundles path. The guarantee
    # this test exists to defend is unchanged: each service restarts on its own and executes
    # out of the versioned runtime bundle, never an ad-hoc path.
    #
    # This is the template-level half. The rendered half — that substitution leaves no
    # @JOPE_* token, no /home/ literal, and every ExecStart under the rendered roots — is
    # covered in Rust by crates/jope-campaigns/tests/systemd_unit_paths.rs
    # (rendered_units_resolve_only_under_the_rendered_roots), which drives the same
    # jope_paths::systemd::render_tree that render.sh shells out to. Duplicating it here
    # would drag a cargo build into the Python lane for no extra coverage.
    root = Path(__file__).parents[2] / "deploy" / "systemd" / "selfplay"
    required = {
        "jope-selfplay-corpus-freezer.service.in",
        "jope-selfplay-ingestion.service.in",
        "jope-selfplay-promotion.service.in",
        "jope-selfplay-queue.service.in",
        "jope-selfplay-supply.service.in",
        "jope-selfplay-trainer-watchdog.service.in",
    }
    units = {unit.name: unit for unit in root.glob("jope-selfplay-*.service.in")}
    assert required <= units.keys()
    for name in required:
        unit = units[name]
        text = unit.read_text()
        assert "Restart=always" in text
        assert "ExecStart=@JOPE_BUNDLE@/" in text
