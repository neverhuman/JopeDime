"""Only the H186 kernel coefficient varies; all other constants stay native."""

from parents import h186_treatment

GRID = (1.0, 1.5, 2.0, 2.5, 3.0, 4.0)
GRID_ARMS = ("R882", "KX1.5", "KX2.0", "KX2.5", "KX3.0", "KX4.0")
adapter_components = h186_treatment.adapter_components
scores = h186_treatment.scores


def constants(arm):
    return dict(
        kernel_multiplier=GRID[GRID_ARMS.index(arm)], topk=8, null_multiplier=1.0
    )
