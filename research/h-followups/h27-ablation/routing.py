"""Pure CPU feature-routing helpers; no data or model files are read."""

FINGERPRINT_COORDINATES = 8


def route_scaled_features(scaled, *, include_fingerprint):
    """Return a routed copy, zeroing the trailing eight scaled coordinates in control.

    Scaling is deliberately upstream and must be fitted on DL-Train only. This
    function accepts already-scaled synthetic or admitted rows and never fits a
    scaler or opens an input file.
    """
    row = list(scaled)
    if len(row) < FINGERPRINT_COORDINATES:
        raise ValueError("feature row is shorter than the fingerprint block")
    if not include_fingerprint:
        row[-FINGERPRINT_COORDINATES:] = [0.0] * FINGERPRINT_COORDINATES
    return row
