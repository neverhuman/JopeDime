"""Frozen first-completed-cell admission rule; no partial outer scoring."""
import math


def forecast(first_seconds, workers, now, stop):
    assert first_seconds > 0 and isinstance(workers, int) and 4 <= workers <= 7
    # Whole-cell batching, 15% reserve for varying folds, 0.9 efficiency and 5min audit/score reserve.
    remaining_batches = math.ceil(35 / workers)
    remaining_seconds = remaining_batches * first_seconds / .9 * 1.15 + 300
    finish = now + remaining_seconds
    return dict(first_cell_seconds=first_seconds, workers=workers, remaining_cells=35,
                remaining_batches=remaining_batches, safety_multiplier=1.15, efficiency=.9, audit_score_reserve_seconds=300,
                forecast_epoch=now, projected_finish_epoch=finish, compute_stop_epoch=stop,
                admit=finish < stop, rule='ceil(35/workers)*first_completed_cell_seconds/.9*1.15+300 < remaining window')
