"""Transverse coordinates for the fixed causal planes of the JAIN display."""

import math


SPATIAL_LAYOUTS = ("layered", "iris", "twist", "hourglass", "prism", "manta")
ROWS_PER_COLUMN = 32


def transverse_position(layout, t, index, count):
    """Place a canonical node index in its stage's y/z plane.

    The caller owns the causal x coordinate and the mapping from node and
    edge identities to indices. Only this stage's transverse shape changes.
    """
    if layout not in SPATIAL_LAYOUTS:
        raise ValueError(f"Unknown JAIN spatial layout: {layout}")
    if not 0 <= t <= 1 or not 0 <= index < count:
        raise ValueError("Expected normalized stage depth and a valid node index")

    column, row = divmod(index, ROWS_PER_COLUMN)
    columns = (count + ROWS_PER_COLUMN - 1) // ROWS_PER_COLUMN
    rows = min(ROWS_PER_COLUMN, count - column * ROWS_PER_COLUMN)
    if layout == "layered":
        y = 0. if columns == 1 else -1.1 + 2.2 * column / (columns - 1)
        z = 0. if rows == 1 else -4.35 + 8.7 * row / (rows - 1)
        return y, z
    if count == 1:
        return 0., 0.

    u = 0. if rows == 1 else 2. * row / (rows - 1) - 1.
    v = 0. if columns == 1 else 2. * column / (columns - 1) - 1.
    wave = math.sin(math.tau * t)
    arch = math.sin(math.pi * t)

    if layout == "iris":
        # Curved spokes, with a clear center and a slow turn along causal depth.
        angle = math.tau * (column + .5) / columns + 1.35 * t + .28 * u / columns
        radius = (.26 + .74 * (u + 1.) / 2.) * (.86 + .14 * arch)
        return (2.40 * radius * math.cos(angle),
                3.90 * radius * math.sin(angle) + .18 * wave)

    if layout == "twist":
        # Rotate a flat sheet through an elliptical cross section.
        angle = .15 + .82 * wave
        cosine, sine = math.cos(angle), math.sin(angle)
        return (2.42 * (.66 * v * cosine - .93 * u * sine),
                3.85 * (.92 * u * cosine + .38 * v * sine))

    if layout == "hourglass":
        # Keep depth separation through the narrow waist of the silhouette.
        offset = .60 * arch
        scale = .40 + .60 * abs(2. * t - 1.) ** .85
        return (1.58 * v + .55 * math.sin(math.pi * u) * (2. * t - 1.),
                offset + (3.85 - offset) * scale * u)

    if layout == "prism":
        # Draw concentric, staggered hexagonal shells without a duplicate seam.
        side, blend = divmod(6. * (row + .45 * (column % 2)) / rows, 1.)
        angle = math.pi / 2. + math.pi * side / 3. + .48 * (t - .5)
        next_angle = angle + math.pi / 3.
        y = (1. - blend) * math.cos(angle) + blend * math.cos(next_angle)
        z = (1. - blend) * math.sin(angle) + blend * math.sin(next_angle)
        radius = (.55 + .45 * (v + 1.) / 2.) * (.90 + .10 * arch)
        return 2.65 * radius * y, 3.97 * radius * z

    # Manta: bow the depth columns and sweep the wings with a traveling bend.
    return (1.65 * v + .90 * u ** 2,
            (2.75 + .55 * arch) * u + .46 * wave * (1. - .65 * v ** 2) + .28 * v)
