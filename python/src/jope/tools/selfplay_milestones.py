"""AZ-2017 specialist loop gates and numeric bars. Serving is never assigned here."""
from __future__ import annotations
from scientific_metadata import canonical, finite

RECOVERY_NLL = 1.394498497
RECOVERY_BT = 0.533676382
NLL_GUARD = 0.005
BT_GUARD = 0.01

BARS = dict(
    neural_seed_beats_teacher=0.10,
    final_gp_beats_teacher=0.20,
    final_gp_beats_teacher_by_margin=0.05,
    regression_existence_neural_seed=0.05,
    paper_relative_margin=0.01,
)


def ranking_bt_stable(bt, recovery_bt=RECOVERY_BT):
    return finite(bt) and finite(recovery_bt) and recovery_bt > 0 and bt <= recovery_bt * (1 + BT_GUARD)


def search_beating_allowed(frozen_bt, recovery_bt=RECOVERY_BT):
    return ranking_bt_stable(frozen_bt, recovery_bt)


def frozen_diagnostic_pass(nll, bt, recovery_nll=RECOVERY_NLL, recovery_bt=RECOVERY_BT):
    if not all(finite(x) for x in (nll, bt, recovery_nll, recovery_bt)) or min(recovery_nll, recovery_bt) <= 0:
        return False, ["diagnostic requires finite nonnegative nll/bt and positive references"]
    reasons = []
    if nll > recovery_nll * (1 + NLL_GUARD):
        reasons.append("diagnostic regression: sequence_nll")
    if bt > recovery_bt * (1 + BT_GUARD):
        reasons.append("diagnostic regression: ranking_bt")
    return not reasons, reasons


def may_serve(diagnostic_ok, probe_ok, paper_ok, quality_report_for_challenger=False):
    """Legacy flags are diagnostics; only original promotion evidence may serve."""
    if not quality_report_for_challenger:
        return False, ["serve blocked: challenger quality-report missing is not independent evidence"]
    if not diagnostic_ok:
        return False, ["serve blocked: frozen diagnostic"]
    if not probe_ok:
        return False, ["serve blocked: 16-cut probe"]
    if not paper_ok:
        return False, ["serve blocked: paper promotion gate"]
    return False, ["serve blocked: diagnostic flags cannot replace original promotion evidence"]


def win_rates(rows, teacher_key="best_retained_teacher", seed_key="best_initial_neural_seed",
              final_key="best_final_gp", margin=BARS["paper_relative_margin"]):
    seed_n = seed_beats = final_n = final_beats = margin_beats = 0
    seen = set()
    duplicates = invalid = 0
    for row in rows or []:
        if not isinstance(row, dict):
            invalid += 1
            continue
        try:
            identity = canonical(row)
        except (ValueError, TypeError, OverflowError):
            invalid += 1
            continue
        if identity in seen:
            duplicates += 1
            continue
        seen.add(identity)
        teacher, seed, final = row.get(teacher_key), row.get(seed_key), row.get(final_key)
        if finite(teacher) and finite(seed):
            seed_n += 1
            if seed < teacher:
                seed_beats += 1
        if finite(teacher) and finite(final):
            final_n += 1
            if final < teacher:
                final_beats += 1
                if teacher > 0 and (teacher - final) / teacher >= margin:
                    margin_beats += 1
    return dict(
        neural_seed_n=seed_n,
        neural_seed_beats=seed_beats,
        neural_seed_rate=seed_beats / seed_n if seed_n else 0.0,
        final_gp_n=final_n,
        final_gp_beats=final_beats,
        final_gp_rate=final_beats / final_n if final_n else 0.0,
        final_gp_margin_rate=margin_beats / final_n if final_n else 0.0,
        duplicate_rows=duplicates, invalid_rows=invalid,
        scope="available unique diagnostic rows; original execution and family independence unverified",
        independent_evaluation=False,
    )


def existence_proof(rates, require=BARS["regression_existence_neural_seed"]):
    """Aggregate rates cannot prove original, attributable discoveries."""
    return False


def world_class(rates):
    """No diagnostic Boolean grants the missing scientific qualification."""
    return False


def descriptive_bars(rates):
    if not isinstance(rates, dict) or not all(
            finite(rates.get(k)) and rates[k] <= 1
            for k in ("neural_seed_rate", "final_gp_rate", "final_gp_margin_rate")):
        return False
    return (rates["neural_seed_rate"] >= BARS["neural_seed_beats_teacher"]
            and rates["final_gp_rate"] >= BARS["final_gp_beats_teacher"]
            and rates["final_gp_margin_rate"] >= BARS["final_gp_beats_teacher_by_margin"])


def specialist_loop(task, served_from_own_panel, registry_teacher_refit_rate, mixed_corpus=False):
    if mixed_corpus:
        return False, ["specialist labels mixed across tasks"]
    if registry_teacher_refit_rate != 0:
        return False, ["teacher/seed re-fit rate must be 0 on repeated generations"]
    if not served_from_own_panel:
        return False, ["serve only a net that beat this task frozen panel"]
    return True, []
