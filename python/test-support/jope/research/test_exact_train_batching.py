from __future__ import annotations

from collections import defaultdict
import random

from test_exact_train import (
    EQUAL_QUOTA, equal_quota_draw, recipe_e_batch_lane_counts,
    recipe_e_evidence_lane,
)


def make_grouped_batches(
    *, seed, epochs_drawn, recipe, task, accepted_parent_sha256,
    regression_fraction,
):
    def equal_quota_epoch_rows(rows, epoch):
        pools = defaultdict(list)
        for row in rows:
            metadata = row.metadata or {}
            pools[
                (str(metadata.get("task_type")), str(metadata.get("dataset_hash")))
            ].append(row)
        rng = random.Random((int(seed) & 0xFFFF_FFFF) * 1_000_003 + epoch * 2_654_435_761)
        drawn_by_family = defaultdict(lambda: defaultdict(list))
        for key in sorted(pools):
            pool = pools[key]
            quota = int((pool[0].metadata or {}).get("epoch_quota") or EQUAL_QUOTA)
            selected = equal_quota_draw(pool, quota, rng)
            family = str((pool[0].metadata or {}).get("source_family") or "")
            if not family:
                raise ValueError("training row lacks source_family for balanced sampling")
            drawn_by_family[family][key].extend(selected)
        if not drawn_by_family:
            return []
        # Every family contributes the same total weight. Inside a family,
        # datasets are visited round-robin in a seeded order, so a large source
        # family cannot dominate and a deep dataset cannot dominate its family.
        family_quota = min(
            sum(len(dataset_rows) for dataset_rows in datasets.values())
            for datasets in drawn_by_family.values()
        )
        balanced = []
        for family in sorted(drawn_by_family):
            datasets = drawn_by_family[family]
            keys = sorted(datasets)
            rng.shuffle(keys)
            positions = {key: 0 for key in keys}
            family_rows = []
            while len(family_rows) < family_quota:
                progressed = False
                for key in keys:
                    position = positions[key]
                    if position < len(datasets[key]):
                        family_rows.append(datasets[key][position])
                        positions[key] += 1
                        progressed = True
                        if len(family_rows) == family_quota:
                            break
                if not progressed:
                    break
            if len(family_rows) != family_quota:
                raise ValueError("family-balanced sampler could not satisfy its quota")
            balanced.extend(family_rows)
        return balanced

    def grouped_batches(rows, batch_size, rng):
        from jope.train import _dataset_grouped_batches

        rows = list(rows)
        # train_v3 routes BOTH its epoch batching and its validation batching
        # through this hook. Resample the train split only: the validation corpus
        # already holds exactly VALIDATION_QUOTA rows per dataset and has to stay
        # byte-identical across epochs, or the checkpoint-selection metric moves
        # for sampling reasons instead of learning reasons.
        # ("development" is what jope.split_integrity renames "validation" to; it
        # does not write the rename back onto the row, but accept it anyway.)
        splits = {str((row.metadata or {}).get("split")) for row in rows}
        is_validation = bool(rows) and splits <= {"validation", "development"}
        if recipe == "e" and rows and not is_validation:
            epochs_drawn["count"] += 1
            if any(
                str((row.metadata or {}).get("task_type")) != task
                for row in rows
            ):
                raise ValueError("recipe E batch crossed task identity")
            pair_members = defaultdict(dict)
            lane_rows = defaultdict(list)
            for row in rows:
                metadata = row.metadata or {}
                lane = recipe_e_evidence_lane(
                    metadata, accepted_parent_sha256 or ""
                )
                if lane == "censored_descendant":
                    continue
                if lane == "exact_current_parent_pairs":
                    pair_id = metadata["causal_pair_id"]
                    pair_lane = metadata["causal_lane"]
                    if pair_lane in pair_members[pair_id]:
                        raise ValueError(
                            "recipe E has an ambiguous current-parent pair lane"
                        )
                    pair_members[pair_id][pair_lane] = row
                else:
                    lane_rows[lane].append(row)
            pairs = [
                (
                    members["native_dl_direct"],
                    members["matched_gp_comparator"],
                )
                for _, members in sorted(pair_members.items())
                if set(members)
                == {"native_dl_direct", "matched_gp_comparator"}
            ]
            required_lanes = {
                "uniform_broad_replay",
                "completed_descendant_trajectories",
                "validity_instability_hard_negatives",
            }
            missing = sorted(
                lane for lane in required_lanes if not lane_rows[lane]
            )
            if missing or not pairs:
                raise ValueError(
                    f"recipe E evidence lanes are incomplete: {missing}"
                )
            epoch = epochs_drawn["count"]
            for lane in required_lanes:
                lane_rows[lane] = equal_quota_epoch_rows(
                    lane_rows[lane], epoch
                )
                rng.shuffle(lane_rows[lane])
            rng.shuffle(pairs)
            nominal = recipe_e_batch_lane_counts(batch_size, 0)
            batch_count = max(
                1,
                (len(lane_rows["uniform_broad_replay"])
                 + nominal["uniform_broad_replay"] - 1)
                // nominal["uniform_broad_replay"],
                (len(pairs) * 2
                 + nominal["exact_current_parent_pairs"] - 1)
                // nominal["exact_current_parent_pairs"],
                (len(lane_rows["completed_descendant_trajectories"])
                 + nominal["completed_descendant_trajectories"] - 1)
                // nominal["completed_descendant_trajectories"],
                (len(lane_rows["validity_instability_hard_negatives"])
                 + nominal["validity_instability_hard_negatives"] - 1)
                // nominal["validity_instability_hard_negatives"],
            )

            def cycle(pool, start, size):
                return [
                    pool[(start + index) % len(pool)]
                    for index in range(size)
                ]

            output = []
            lane_offsets = defaultdict(int)
            pair_offset = 0
            for batch_index in range(batch_count):
                counts = recipe_e_batch_lane_counts(
                    batch_size, batch_index
                )
                batch = []
                broad_lane = "uniform_broad_replay"
                batch.extend(
                    cycle(
                        lane_rows[broad_lane],
                        lane_offsets[broad_lane],
                        counts[broad_lane],
                    )
                )
                lane_offsets[broad_lane] += counts[broad_lane]
                pairs_per_batch = (
                    counts["exact_current_parent_pairs"] // 2
                )
                selected_pairs = cycle(
                    pairs, pair_offset, pairs_per_batch
                )
                pair_offset += pairs_per_batch
                batch.extend(
                    row
                    for pair in selected_pairs
                    for row in pair
                )
                for lane in (
                    "completed_descendant_trajectories",
                    "validity_instability_hard_negatives",
                ):
                    batch.extend(
                        cycle(
                            lane_rows[lane],
                            lane_offsets[lane],
                            counts[lane],
                        )
                    )
                    lane_offsets[lane] += counts[lane]
                if len(batch) != batch_size:
                    raise ValueError("recipe E batch composition drift")
                output.append(batch)
            return output
        if recipe == "d" and rows and not is_validation:
            epochs_drawn["count"] += 1
            broad = equal_quota_epoch_rows(rows, epochs_drawn["count"])
            pair_members = defaultdict(dict)
            pair_flags = {}
            for row in rows:
                metadata = row.metadata or {}
                pair_id = metadata.get("causal_pair_id")
                lane = metadata.get("causal_lane")
                if (
                    pair_id
                    and lane in {"native_dl_direct", "matched_gp_comparator"}
                    and metadata.get(
                        "causal_accepted_parent_checkpoint_sha256"
                    )
                    == accepted_parent_sha256
                ):
                    if lane in pair_members[pair_id]:
                        raise ValueError("causal recipe has an ambiguous pair lane")
                    pair_members[pair_id][lane] = row
                    pair_flags[pair_id] = bool(
                        metadata.get("causal_pair_hard")
                        or metadata.get("causal_pair_novel")
                    )
            pairs = [
                (
                    members["native_dl_direct"],
                    members["matched_gp_comparator"],
                )
                for pair_id, members in sorted(pair_members.items())
                if set(members)
                == {"native_dl_direct", "matched_gp_comparator"}
            ]
            if not pairs:
                raise ValueError(
                    "causal recipe has no complete pair for its accepted parent"
                )
            weighted_pairs = []
            for pair_id, members in sorted(pair_members.items()):
                if set(members) != {
                    "native_dl_direct",
                    "matched_gp_comparator",
                }:
                    continue
                pair = (
                    members["native_dl_direct"],
                    members["matched_gp_comparator"],
                )
                weighted_pairs.append(pair)
                if pair_flags[pair_id]:
                    weighted_pairs.append(pair)
            rng.shuffle(weighted_pairs)
            broad_quota = (batch_size * 3) // 4
            paired_rows = batch_size - broad_quota
            if paired_rows < 2 or paired_rows % 2:
                raise ValueError("causal recipe batch cannot preserve exact pairs")
            broad_rows = [
                row
                for batch in _dataset_grouped_batches(
                    broad, broad_quota, rng
                )
                for row in batch
            ]
            batch_count = max(
                1, (len(broad_rows) + broad_quota - 1) // broad_quota
            )

            def cycle(pool, start, size):
                return [
                    pool[(start + index) % len(pool)] for index in range(size)
                ]

            output = []
            pairs_per_batch = paired_rows // 2
            for index in range(batch_count):
                batch = cycle(
                    broad_rows, index * broad_quota, broad_quota
                )
                selected_pairs = cycle(
                    weighted_pairs,
                    index * pairs_per_batch,
                    pairs_per_batch,
                )
                batch.extend(
                    row for direct_comparator in selected_pairs for row in direct_comparator
                )
                output.append(batch)
            return output
        if rows and not is_validation:
            epochs_drawn["count"] += 1
            rows = equal_quota_epoch_rows(rows, epochs_drawn["count"])

        regression = [
            row
            for row in rows
            if str((row.metadata or {}).get("task_type")) == "regression"
        ]
        binary = [
            row
            for row in rows
            if str((row.metadata or {}).get("task_type")) == "binary"
        ]
        if not regression or not binary:
            return _dataset_grouped_batches(rows, batch_size, rng)
        regression_quota = round(batch_size * regression_fraction)
        binary_quota = batch_size - regression_quota
        if min(regression_quota, binary_quota) < 1:
            raise ValueError("task mixture must retain both task types")

        def ordered(pool, quota):
            return [
                row
                for batch in _dataset_grouped_batches(pool, quota, rng)
                for row in batch
            ]

        regression_rows = ordered(regression, regression_quota)
        binary_rows = ordered(binary, binary_quota)
        count = max(
            (len(regression_rows) + regression_quota - 1) // regression_quota,
            (len(binary_rows) + binary_quota - 1) // binary_quota,
        )

        def cycle(pool, start, size):
            return [pool[(start + index) % len(pool)] for index in range(size)]

        return [
            cycle(regression_rows, index * regression_quota, regression_quota)
            + cycle(binary_rows, index * binary_quota, binary_quota)
            for index in range(count)
        ]
    return grouped_batches
