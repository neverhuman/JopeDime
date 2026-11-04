"""Frozen first-completed-cell admission rule; no partial outer scoring."""
import math


def forecast(first_seconds, workers, now, stop):
    assert first_seconds > 0 and isinstance(workers, int) and 1 <= workers <= 5
    # Whole-cell batching, 15% reserve for varying folds, 10min audit/score reserve.
    remaining_batches = math.ceil(35 / workers)
    remaining_seconds = remaining_batches * first_seconds * 1.15 + 600
    finish = now + remaining_seconds
    return dict(first_cell_seconds=first_seconds, workers=workers, remaining_cells=35,
                remaining_batches=remaining_batches, safety_multiplier=1.15, audit_score_reserve_seconds=600,
                forecast_epoch=now, projected_finish_epoch=finish, compute_stop_epoch=stop,
                admit=finish < stop, rule='ceil(35/workers)*first_completed_cell_seconds*1.15+600 < remaining window')
