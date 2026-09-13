"""Expert-iteration strategy diagnostics; no scientific certification or launch."""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

SCHEMA = "parallel.expert_iteration_setup.v2"
RECOVERY_PARENT = "c2f229c9b8c582e11e0e62ec011a9ccb066820fbcee5e2da9d0ad5d63780160e"
QUALITY_SHA = "447bc00e19ad27ae90442fb02bc7544d2c6557876abc92df7782ab6c32d1bdb2"

# The current implementation has no qualified prefix-value/search contract.
# This is an implementation boundary, not a mathematical impossibility.
AGZ = dict(
    template="alphago_zero",
    outcome="discrete_z",
    improvement="mcts_visit_counts",
    value_domain="any_prefix_board",
    tabula_rasa=True,
    applicable=False,
    mcts_now=False,
    mcts_intrinsically_invalid=False,
)

# Anthony, Tian, Barber 2017. Expert = budgeted search; policy imitates
# complete expert programs; ranking compares full-program RMSE.
EI = dict(
    template="expert_iteration",
    citation="Anthony, Tian, Barber 2017",
    policy="grammar_masked_sequence_model",
    expert="fixed_budget_gp",
    legacy_expert_budget=dict(evaluations=1000, seconds=600),
    value_domain="complete_program_rmse_only",
    ranking="dense_pairwise_v2_bradley_terry",
    listwise="percentile_softmax_kl",
    league_arms=("neural", "generic", "random", "teacher"),
    applicable=True,
    optimality_proof=False,
)

LOSS = dict(
    sequence_nll=1.0,
    ranking_bt=0.25,
    ranking_listwise=0.1,
    runtime_huber=0.1,
    ablation_bt=0.05,
    stability=0.0,
    ranking_backward="dense_pairwise_v2",
    runtime_objective="huber_log1p_seconds_v2",
    nll_row_weight="0.05 + percentile**4",
    teacher_in_each_comparison=True,
)

ARCH = dict(layers=6, width=512, heads=4, ffn=2048, max_seq=128, dropout=0.1,
            dataset_dim=1536, target_keep_dim=512, batch="256x8", new_steps=1000,
            task_cap_seconds=300, optimizer="carry")

MEASURED = dict(
    quality_sha256=QUALITY_SHA,
    neural_seed_rate=0.0,
    neural_seed_beats=0,
    neural_seed_n=140,
    final_gp_rate=0.014285714285714285,
    final_gp_beats=2,
    final_gp_n=140,
    final_gp_margin_rate=0.0,
    vs_random="unavailable",
    independent_evaluation=False,
    source_report_verified_here=False,
)

FAKE_UPGRADES = (
    "unqualified_mcts_gain_claim",
    "unqualified_width_gain_claim",
    "unqualified_loss_gain_claim",
    "f32_teachers",
    "cuda_gpr",
    "occupancy_as_intelligence",
    "2_of_140_as_world_class",
    "receipts_as_p1",
    "env_bypass_grant_hold",
)

REQUIRED_FOR_FLYWHEEL = (
    "real_native_grant",
    "new_corpus_identity",
    "new_gp_identity",
    "original_admitted_programs",
    "one_train_on_successor_teaching",
    "independent_matched_gain_evidence",
    "p1_accepted",
    "p2_successor_discoveries",
)


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def value_from_rmse(rmse, median=1.0):
    return 1.0 / (1.0 + rmse / median)


def claude_example_pair_weight_ratio():
    """Formula-only check of Claude 23:24 example. Not a teaching-group measurement."""
    close = abs(value_from_rmse(1.0) - value_from_rmse(1.004596))
    far = abs(value_from_rmse(1.0) - value_from_rmse(3.0))
    return dict(
        close_weight=close,
        far_weight=far,
        ratio=far / close if close else None,
        science_agreed_scan=False,
        scope="formula_only",
    )


def refuse_alphago_zero():
    return not AGZ["applicable"] and EI["applicable"] and EI["value_domain"] == "complete_program_rmse_only"


def refuse_fake_upgrade(name):
    return name in FAKE_UPGRADES


def pairwise_is_the_value_target():
    """Legacy API: a relative preference target is not an outcome-value target."""
    return False


def pairwise_is_the_value_head():
    return False


def current_rates_are_not_flywheel(rates=None):
    """Legacy API: rates alone cannot establish qualified flywheel evidence.

    This does not establish that learning has or has not occurred. Caller flags
    cannot replace original independent evaluation and discovery joins.
    """
    return True


def rate_series_diagnostic(history):
    """Describe rate trends with complete identity labels; verify no evidence."""
    base = dict(qualified_flywheel=False, independent_evidence_verified=False)
    if not isinstance(history, (list, tuple)) or len(history) < 2:
        return dict(base, status="unavailable", reason="at_least_two_rows_required")
    seeds, repairs = [], []
    for row in history:
        if not isinstance(row, dict):
            return dict(base, status="unavailable", reason="invalid_row")
        rates = [row.get(key) for key in ("neural_seed_rate", "gp_improves_on_seed")]
        if any(type(rate) not in (int, float) or not 0 <= rate <= 1
               or not math.isfinite(rate) for rate in rates):
            return dict(base, status="unavailable", reason="invalid_rate")
        if any(not isinstance(row.get(key), str)
               or re.fullmatch(r"[0-9a-f]{64}", row[key]) is None
               for key in ("parent", "corpus", "gp")):
            return dict(base, status="unavailable", reason="missing_or_invalid_lineage")
        seeds.append(rates[0])
        repairs.append(rates[1])
    return dict(base, status="diagnostic_only", rows=len(history),
                seed_rate_strictly_increasing=all(b > a for a, b in zip(seeds, seeds[1:])),
                repair_rate_strictly_decreasing=all(b < a for a, b in zip(repairs, repairs[1:])),
                distinct_parents=len({row["parent"] for row in history}),
                distinct_corpora=len({row["corpus"] for row in history}),
                distinct_gp_runs=len({row["gp"] for row in history}),
                identity_labels_verified=False)


def flywheel_getting_smarter(history):
    """Legacy Boolean API fails closed until independent original joins exist."""
    return False


def strategy():
    return dict(
        schema=SCHEMA,
        template=EI,
        rejected_template=AGZ,
        loss=LOSS,
        arch=ARCH,
        measured=MEASURED,
        refuse_alphago_zero=refuse_alphago_zero(),
        pairwise_is_the_value_target=pairwise_is_the_value_target(),
        pairwise_is_the_value_head=False,
        critic_kind="relative_program_ranking_score",
        outcome_calibration_verified=False,
        critic_shares_decoder_trunk=False,
        fake_upgrades=list(FAKE_UPGRADES),
        required_for_flywheel=list(REQUIRED_FOR_FLYWHEEL),
        flywheel_effect=False,
        flywheel_getting_smarter=False,
        launch_train_allowed=False,
        serving_written=False,
        science_agreed_218x_scan=False,
        pair_weight_218x="unverified_teaching_groups",
        one_hundred_eight_blocks_one_parent=False,
        parent_count_this_lane="unrecounted",
        peer_constraints=dict(
            ei_is_framework_not_optimality_proof=True,
            mcts_intrinsically_invalid=False,
            mcts_now=False,
            pairwise_is_ordinal_not_agz_calibration=True,
            claude_seed_series="not_copied_unverified_by_this_lane",
        ),
        note=("Expert Iteration is a relevant framework, not proof of best design. "
              "Pairwise BT trains relative program preferences on a disjoint critic. "
              "Prefix-value/search alternatives need separate qualification. Rate trends and "
              "identity labels are diagnostics; attributable gain requires original admissions, "
              "independent matched evaluation and accepted P1/P2 discovery joins. "
              "The supplied gen-66 rates and source digest are historical diagnostics, not "
              "independent evidence verified by this helper. The legacy expert budget is "
              "historical configuration, not current authorization. A formula-only pair-weight "
              "ratio is not a measured training-gradient contribution."),
    )


def emit(dest):
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    body = strategy()
    body["identity"] = digest({k: body[k] for k in body if k != "identity"})
    path = dest / "STRATEGY.json"
    summary_path = dest / "SUMMARY.json"
    if path.exists() or summary_path.exists():
        raise FileExistsError("strategy histories are immutable; use a new destination")
    with path.open("x") as stream:
        stream.write(json.dumps(body, sort_keys=True, indent=2, allow_nan=False) + "\n")
    summary = dict(
        schema="parallel.expert_iteration_setup_summary.v2",
        strategy_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        refuse_alphago_zero=True,
        pairwise_is_the_value_target=False,
        pairwise_is_the_value_head=False,
        flywheel_effect=False,
        flywheel_getting_smarter=False,
        launch_train_allowed=False,
        measured_seed_rate=MEASURED["neural_seed_rate"],
        measured_final_rate=MEASURED["final_gp_rate"],
    )
    with summary_path.open("x") as stream:
        stream.write(json.dumps(summary, sort_keys=True, indent=2, allow_nan=False) + "\n")
    return summary


if __name__ == "__main__":
    print(json.dumps(emit(Path("parallel/runs/flywheel-worldclass-20260911/expert-iteration-setup-v2")), indent=2))
