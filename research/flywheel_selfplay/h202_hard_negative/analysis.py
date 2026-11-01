"""H202 paired-family median inference and strict primary/control/placebo gates."""

import numpy as np
from design import ALL_ARMS, BOOT_SEED, DRAWS


def bt(w):
    n = w.shape[-1]
    games = w + np.swapaxes(w, -1, -2) + 0.02 * (1 - np.eye(n))
    wins = w.sum(axis=-1) + 0.01 * (n - 1)
    strength = np.ones(w.shape[:-1])
    for _ in range(400):
        denominator = (
            games / np.maximum(strength[..., :, None] + strength[..., None, :], 1e-12)
        ).sum(axis=-1)
        strength = np.clip(wins / denominator, 1e-6, 1e6)
        strength /= np.maximum(strength[..., :1], 1e-12)
    return 1500 + 400 * np.log10(np.maximum(strength, 1e-9))


def summarize(rows, panel):
    panel = np.asarray(panel, dtype=np.float64)
    if panel.shape != (3, len(rows), 4) or not np.isfinite(panel).all():
        raise ValueError("complete 3-seed x context x 4-policy panel required")
    labels = np.asarray([r["family_id"] for r in rows])
    families = sorted(set(labels))
    family = np.stack([panel[:, labels == f].mean(axis=1) for f in families], axis=1)
    n = len(families)
    ix = np.random.default_rng(BOOT_SEED).integers(0, n, (DRAWS, n))
    weights = np.stack([(ix == j).sum(axis=1) for j in range(n)], axis=1)

    def estimate(value):
        seeds = value.mean(axis=1)
        boot = np.median(value[:, ix].mean(axis=2), axis=0)
        return dict(
            per_seed=seeds.tolist(),
            median=float(np.median(seeds)),
            ci95=np.quantile(boot, [0.025, 0.975]).tolist(),
            positive_seeds=int((seeds > 0).sum()),
            p_one_sided=float((1 + (boot <= 0).sum()) / (DRAWS + 1)),
        )

    contrasts = {
        a + "_minus_R882": estimate(family[:, :, j] - family[:, :, 0])
        for j, a in enumerate(ALL_ARMS[1:], 1)
    }
    for j, a in [(2, "CTRL40"), (3, "HNEG_SHUF")]:
        contrasts["HNEG40_minus_" + a] = estimate(family[:, :, 1] - family[:, :, j])
    harm = {
        a: dict(
            seed_average_family_fraction=float(
                np.mean((family[:, :, j] - family[:, :, 0]).mean(axis=0) < 0)
            ),
            per_seed_family_fraction=np.mean(
                family[:, :, j] < family[:, :, 0], axis=1
            ).tolist(),
            per_seed_context_fraction=np.mean(
                panel[:, :, j] < panel[:, :, 0], axis=1
            ).tolist(),
        )
        for j, a in enumerate(ALL_ARMS)
    }
    for a in ALL_ARMS[1:]:
        contrasts[a + "_minus_R882"]["holm_m"] = 1 if a == "HNEG40" else None
        contrasts[a + "_minus_R882"]["holm_p"] = (
            contrasts[a + "_minus_R882"]["p_one_sided"] if a == "HNEG40" else None
        )

    def efficacy(arm):
        c = contrasts[arm + "_minus_R882"]
        return bool(
            c["median"] > 0.001 + 1e-12
            and c["ci95"][0] > 0
            and c["positive_seeds"] == 3
            and harm[arm]["seed_average_family_fraction"] <= 0.10
            and c["p_one_sided"] < 0.05
        )

    primary = efficacy("HNEG40")
    beats_ctrl = contrasts["HNEG40_minus_CTRL40"]["median"] > 1e-12
    beats_shuf = contrasts["HNEG40_minus_HNEG_SHUF"]["median"] > 1e-12
    placebo = efficacy("HNEG_SHUF")
    candidate = bool(primary and beats_ctrl and beats_shuf and not placebo)
    diff = panel[:, :, :, None] - panel[:, :, None, :]
    wins = np.where(diff > 1e-4, 1.0, np.where(diff < -1e-4, 0.0, 0.5)) * (
        1 - np.eye(4)
    )
    fw = np.stack([wins[:, labels == f].mean(axis=1) for f in families], axis=1)
    points = bt(fw.sum(axis=1))
    bootstrap = []
    for start in range(0, DRAWS, 250):
        bootstrap.append(
            np.median(
                bt(np.einsum("qf,sfij->sqij", weights[start : start + 250], fw)), axis=0
            )
        )
    eboot = np.concatenate(bootstrap)
    elo = {
        a: dict(
            per_seed=points[:, j].tolist(),
            median=float(np.median(points[:, j])),
            ci95=np.quantile(eboot[:, j], [0.025, 0.975]).tolist(),
        )
        for j, a in enumerate(ALL_ARMS)
    }
    return dict(
        contexts=len(rows),
        families=n,
        utility={a: estimate(family[:, :, j]) for j, a in enumerate(ALL_ARMS)},
        contrasts=contrasts,
        harm=harm,
        Elo=elo,
        minus2_count_per_seed={
            a: (panel[:, :, j] == -2).sum(axis=1).tolist()
            for j, a in enumerate(ALL_ARMS)
        },
        HNEG_efficacy_gate=primary,
        HNEG_beats_CTRL=bool(beats_ctrl),
        HNEG_beats_SHUF=bool(beats_shuf),
        SHUF_efficacy_gate=placebo,
        verdict="CANDIDATE" if candidate else "NO_ADVANCE",
        serving_promotion=False,
        bootstrap_seed=BOOT_SEED,
        bootstrap_draws=DRAWS,
    )
