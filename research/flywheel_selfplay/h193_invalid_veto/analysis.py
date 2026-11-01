"""FIT-only thresholds, B-only choices; imported H191/H177 pooled-A inference."""

import hashlib
import json
from pathlib import Path
import numpy as np
from parents import h191, h177
from treatment import theta_grid, veto_pick, median_theta

SEEDS = (193101, 193102, 193103)
FOLD_SEEDS = dict(zip(SEEDS, (193141, 193142, 193143)))
BOOT_SEED = 193150
READOUT_ARMS = ("R882", "VETO", "Q90", "Q75", "Q50")
h191.SEEDS, h191.FOLD_SEEDS, h191.BOOT_SEED = SEEDS, FOLD_SEEDS, BOOT_SEED
h191.READOUT_ARMS = ("R882", "KXSEL", "Q90", "Q75", "Q50")
original_bindings = h191.bindings


def bindings(seed=SEEDS[0]):
    return original_bindings(seed)


h191.bindings = bindings
partition, fold_manifest = h191.partition, h191.fold_manifest
H177_DECISION_RULE = h191.H177_DECISION_RULE


def select_a(rows, panel):
    result = h191.select_a(rows, panel)
    result.update(primary="VETO", imported_internal_policy_label="KXSEL")
    return result


def summarize(rows, panel):
    result = h191.summarize(rows, panel)
    result["contrasts"]["VETO"] = result["contrasts"].pop("KXSEL")
    return result


def verdict(admission, summary, receipt, digest):
    internal = dict(summary, contrasts={"KXSEL": summary["contrasts"]["VETO"]})
    return h191.verdict(admission, internal, receipt, digest)


def digest_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def grid_panel(rows, costs, probabilities, utilities, grid):
    panel = []
    counts = [dict(vetoed=0, candidates=0, fallback=0, contexts=0) for _ in grid]
    choices = []
    for row in rows:
        cid = row["context_sha256"]
        picks = []
        for i, theta in enumerate(grid):
            pick, count = veto_pick(costs[cid], probabilities[cid], theta)
            picks.append(pick)
            for key in count:
                counts[i][key] += count[key]
        choices.append(picks)
        panel.append(np.asarray(utilities[cid])[picks])
    for count in counts:
        count["veto_rate"] = (
            count["vetoed"] / count["candidates"] if count["candidates"] else 0.0
        )
        count["fallback_rate"] = (
            count["fallback"] / count["contexts"] if count["contexts"] else 0.0
        )
    return np.asarray(panel, np.float64), np.asarray(choices), counts


def select_cell(fit, b, probabilities, costs, utilities):
    if {r["family_id"] for r in fit} & {r["family_id"] for r in b}:
        raise ValueError("FIT/B family overlap")
    grid = theta_grid([probabilities[r["context_sha256"]] for r in fit])
    panel, picks, counts = grid_panel(b, costs, probabilities, utilities, grid)
    means = h177.family_matrix(panel, [r["family_id"] for r in b])[1].mean(axis=0)
    index = int(np.argmax(means))
    return (
        dict(
            grid=grid,
            selected_index=index,
            theta=grid[index],
            b_family_means=means.tolist(),
            b_panel_sha256=hashlib.sha256(panel.tobytes()).hexdigest(),
            b_counts=counts,
        ),
        panel,
        picks,
    )


def stage1(train, probabilities, costs, utilities):
    cells = {}
    panels = np.full((3, len(train), 4), np.nan)
    positions = {r["context_sha256"]: i for i, r in enumerate(train)}
    for si, seed in enumerate(SEEDS):
        seen = set()
        for outer in range(5):
            fit, b, a = partition(train, seed, outer)
            families = [{r["family_id"] for r in rows} for rows in (fit, b, a)]
            if any(families[i] & families[j] for i, j in ((0, 1), (0, 2), (1, 2))):
                raise RuntimeError("FIT/B/A overlap")
            cell, panel, picks = select_cell(fit, b, probabilities, costs, utilities)
            cell.update(
                seed=seed,
                outer=outer,
                fit_families=sorted(families[0]),
                b_families=sorted(families[1]),
                a_families=sorted(families[2]),
            )
            cells[f"s{seed}-f{outer}"] = cell
            for j, row in enumerate(b):
                cid = row["context_sha256"]
                if cid in seen:
                    raise RuntimeError("duplicate B context")
                seen.add(cid)
                panels[si, positions[cid]] = panel[j]
        if seen != set(positions):
            raise RuntimeError("incomplete B coverage")
    val_grids = {
        str(seed): [
            median_theta([cells[f"s{seed}-f{k}"]["grid"][i] for k in range(5)])
            for i in range(4)
        ]
        for seed in SEEDS
    }
    val_theta = {
        str(seed): median_theta([cells[f"s{seed}-f{k}"]["theta"] for k in range(5)])
        for seed in SEEDS
    }
    return dict(
        wave="H193",
        cells=cells,
        val_grids=val_grids,
        val_theta=val_theta,
        locked_before_a_and_treatment_val=True,
    ), panels


class SelectionGuard:
    def __init__(self):
        self.path = self.digest = None

    def lock(self, path):
        if self.path is not None:
            raise RuntimeError("already locked")
        value = json.loads(Path(path).read_text())
        if (
            value["wave"] != "H193"
            or not value["locked_before_a_and_treatment_val"]
            or set(value["cells"]) != {f"s{s}-f{k}" for s in SEEDS for k in range(5)}
        ):
            raise RuntimeError("incomplete selection lock")
        if value["b_panel_sha256"] != digest_file(Path(path).parent / "B-panel.npz"):
            raise RuntimeError("B panel mutation")
        for cell in value["cells"].values():
            means = np.asarray(cell["b_family_means"])
            if (
                means.shape != (4,)
                or not np.isfinite(means).all()
                or cell["selected_index"] != int(np.argmax(means))
                or cell["theta"] != cell["grid"][cell["selected_index"]]
            ):
                raise RuntimeError("B argmax/tie inconsistency")
        for seed in SEEDS:
            if value["val_theta"][str(seed)] != median_theta(
                [value["cells"][f"s{seed}-f{k}"]["theta"] for k in range(5)]
            ):
                raise RuntimeError("Val policy not B-locked")
        for seed in SEEDS:
            expected = [
                median_theta(
                    [value["cells"][f"s{seed}-f{k}"]["grid"][i] for k in range(5)]
                )
                for i in range(4)
            ]
            if value["val_grids"][str(seed)] != expected:
                raise RuntimeError("Val descriptive grid not FIT-locked")
        self.path, self.digest = Path(path), digest_file(path)

    def require(self):
        if self.path is None or digest_file(self.path) != self.digest:
            raise RuntimeError("A/Val requires immutable full B lock")
        return json.loads(self.path.read_text())


def stage2(rows, costs, probabilities, utilities, guard, stage):
    lock = guard.require()
    if stage not in ("A", "VAL"):
        raise ValueError("stage")
    panel = np.full((3, len(rows), 5), np.nan)
    counts = {}
    positions = {r["context_sha256"]: i for i, r in enumerate(rows)}
    for si, seed in enumerate(SEEDS):
        cells = (
            [(f"s{seed}-f{k}", lock["cells"][f"s{seed}-f{k}"]) for k in range(5)]
            if stage == "A"
            else [
                (
                    f"VAL-s{seed}",
                    dict(
                        grid=lock["val_grids"][str(seed)],
                        theta=lock["val_theta"][str(seed)],
                    ),
                )
            ]
        )
        seen = set()
        for key, cell in cells:
            selected = (
                [r for r in rows if r["family_id"] in cell["a_families"]]
                if stage == "A"
                else rows
            )
            grid = [None, cell["theta"], *cell["grid"][1:]]
            values, picks, tally = grid_panel(
                selected, costs, probabilities, utilities, grid
            )
            counts[key] = tally
            for j, row in enumerate(selected):
                cid = row["context_sha256"]
                if cid in seen:
                    raise RuntimeError("duplicate A/Val")
                seen.add(cid)
                panel[si, positions[cid]] = values[j]
        if seen != set(positions):
            raise RuntimeError("incomplete A/Val")
    return panel, counts
