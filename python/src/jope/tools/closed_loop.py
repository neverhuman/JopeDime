"""Closed-loop gates. Never assigns serving. Never launches a live train."""
from __future__ import annotations

from four_task_campaign import RECOVERY_SHA, isolate
from flywheel_plan import may_serve, search_beating_training_allowed
from selfplay_milestones import existence_proof, frozen_diagnostic_pass, specialist_loop, win_rates, world_class

RECOVERY_NLL = 1.394498497
RECOVERY_BT = 0.533676382
LIVE_GP_PID = 774452


def two_panels(old_nll, old_bt, new_family_ok):
    old_ok, reasons = frozen_diagnostic_pass(old_nll, old_bt)
    if not old_ok:
        return False, reasons
    if not new_family_ok:
        return False, ["new-family panel failed"]
    return True, []


def launch_train_allowed(hold, automatic_training, teachers_growth, new_teacher_beats,
                         corpus_is_recovery=True, new_gp_identity=False, admission_v2=False,
                         reuse_excluded_from_growth=False, two_panels_predeclared=False,
                         historical_import_complete=False, namespace_enabled=False):
    # Retained diagnostics have no execution capability. The governed native
    # task/reservation and corpus verifiers are the only permitted training gate.
    return False, "diagnostic-only helper; native funded task required"


def grown_family_train(old_nll, old_bt, new_family_ok, diagnostic_ok, probe_ok, paper_ok,
                       challenger_qr, neural_seed_rate, **train_kwargs):
    """One grown-family train gate. Never launches. may_serve stays four-conjunct."""
    panels_ok, panel_reasons = two_panels(old_nll, old_bt, new_family_ok)
    train_kwargs = dict(train_kwargs)
    train_kwargs.pop("two_panels_predeclared", None)
    allowed, reason = launch_train_allowed(
        two_panels_predeclared=panels_ok, **train_kwargs
    )
    serve_ok, serve_reason = serve_decision(
        diagnostic_ok, probe_ok, paper_ok, challenger_qr, neural_seed_rate
    )
    return dict(
        schema="parallel.grown_family_train_gate.v2",
        train_launched=False,
        launch_allowed=allowed,
        launch_reason=reason,
        two_panels=panels_ok,
        panel_reasons=panel_reasons,
        may_serve=serve_ok,
        serve_reason=serve_reason,
        overlapping_3000_step_jobs=False,
        final_tests_opened=False,
    )


def serve_decision(diagnostic_ok, probe_ok, paper_ok, challenger_qr, neural_seed_rate):
    return False, "diagnostic-only helper; complete original promotion evidence required"


def loop_status(registry_teacher_refit_rate, mixed_corpus, served_from_own_panel, task="regression"):
    return specialist_loop(task, served_from_own_panel, registry_teacher_refit_rate, mixed_corpus)


def binary_campaign_bind(inventory_sha, checkpoint_sha):
    camp = isolate("binary", inventory_sha)
    if checkpoint_sha == RECOVERY_SHA:
        raise ValueError("binary specialist cannot bind recovery")
    return dict(campaign=camp, checkpoint_sha256=checkpoint_sha, live_gp_pid=LIVE_GP_PID,
                wired_to_live_gp=False)


def snapshot(rows, frozen_bt=0.56, hold="held"):
    rates = win_rates(rows)
    return dict(
        schema="parallel.closed_loop_status.v1",
        hold=hold,
        live_gp_unswapped=True,
        search_beating_allowed=search_beating_training_allowed(frozen_bt),
        existence_proof=existence_proof(rates),
        win_rates=rates,
        may_serve=may_serve(False, False, False, False),
    )


def from_stage_d(stage_d, hold="held"):
    """Bind the real Stage D receipt. Does not invent a 5% win."""
    first = (stage_d or {}).get("predicted_first_neural_seed_vs_teacher") or {}
    n = int(first.get("n") or 0)
    beats = int(first.get("beats_teacher") or 0)
    rate = 0.0 if n == 0 else beats / n
    return dict(
        schema="parallel.closed_loop_from_stage_d.v1",
        hold=hold,
        live_gp_unswapped=True,
        neural_seed_n=n,
        neural_seed_beats=beats,
        neural_seed_rate=rate,
        existence_proof=existence_proof({"neural_seed_rate": rate}),
        search_beating_allowed=False,
        may_serve=may_serve(False, False, False, False),
        enrichment_off=True,
        source=stage_d.get("source") if isinstance(stage_d, dict) else None,
        awaiting_agent_acks=["Codex registry namespace after coverage", "Claude new-index-only canary"],
    )


def az2017_cycle(old_nll, old_bt, new_family_ok, diagnostic_ok, probe_ok, paper_ok,
                 challenger_qr, neural_seed_rate, rates=None, **train_kwargs):
    """One AZ-2017 evaluate-before-replace cycle. Never launches or serves."""
    gate = grown_family_train(
        old_nll, old_bt, new_family_ok, diagnostic_ok, probe_ok, paper_ok,
        challenger_qr, neural_seed_rate, **train_kwargs
    )
    rates = rates or {"neural_seed_rate": neural_seed_rate, "final_gp_rate": 0.0,
                      "final_gp_margin_rate": 0.0}
    missing = []
    if not train_kwargs.get("new_gp_identity"):
        missing.append("new_gp_identity")
    if not train_kwargs.get("admission_v2"):
        missing.append("admission_v2")
    if not train_kwargs.get("reuse_excluded_from_growth"):
        missing.append("reuse_excluded_from_growth")
    if not train_kwargs.get("historical_import_complete"):
        missing.append("historical_import_complete")
    if not train_kwargs.get("two_panels_predeclared") and not gate["two_panels"]:
        missing.append("two_panels_predeclared")
    if train_kwargs.get("corpus_is_recovery", True):
        missing.append("new_corpus_identity")
    if not existence_proof(rates):
        missing.append("existence_5pct")
    if not paper_ok:
        missing.append("original_paper_producers")
    if not challenger_qr:
        missing.append("challenger_quality_report")
    return dict(
        schema="parallel.az2017_cycle.v1",
        evaluate_before_replace=True,
        az2018_latest_net=False,
        train_launched=False,
        launch_allowed=gate["launch_allowed"],
        launch_reason=gate["launch_reason"],
        two_panels=gate["two_panels"],
        may_serve=False if missing else gate["may_serve"],
        claimed_savings_745=False,
        world_class=world_class(rates),
        existence_proof=existence_proof(rates),
        neural_seed_rate=rates["neural_seed_rate"],
        final_gp_rate=rates["final_gp_rate"],
        final_gp_margin_rate=rates["final_gp_margin_rate"],
        missing_conjuncts=missing,
        overlapping_3000_step_jobs=False,
        loss_unchanged=True,
        width_512=True,
        live_index_written=False,
    )
