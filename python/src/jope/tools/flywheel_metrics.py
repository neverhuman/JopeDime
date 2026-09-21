#!/usr/bin/env python3
"""Flywheel yardsticks: matched-seed contrasts, panel null, signatures. Never writes.

Every function is a pure reduction of receipts already on disk. Missing or
non-finite inputs are reported as None with an explicit reason; nothing is
imputed. Log-ratios are ln(neural / control): negative means neural is lower
(better) RMSE. Clusters are (dataset, cut); resampling is over clusters.
"""

import math
import random
import re
import statistics

from continuous_quality_report import OWN_LINEAGE_ORIGINS, metric
from resume_expired_cycle import require

SCHEMA = "parallel.flywheel_metrics.v1"
ARMS = ("neural", "generic", "shuffled")
OWN_ORIGINS = OWN_LINEAGE_ORIGINS
TAIL_ORIGINS = ("incumbent_cut_teacher", "retained_cut_teacher")
PANEL_METRICS = ("sequence_nll", "ranking_bt", "ranking_listwise", "runtime_huber")
BINDING_KEYS = ("panel_sha256", "corpus_sha256", "executable_sha256")
GUARD_FLOORS = dict(g_nll=0.005, g_bt=0.01)  # selfplay_milestones.NLL_GUARD / BT_GUARD
TIE = 1e-12
TAIL_CAVEAT = "shared-tail descendants not excludable (no parent edges, gp.rs:1892)"
TOKEN_RULE = "dsl_surface_tokens_v1: parentheses and whitespace-delimited atoms of the genome string"
FRESH_RULE = "distinct record ids in replay policy fresh_records (one record = one (context, program) label)"


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def shared_tail(origin):
    """Mirror continuous_quality_report.search_row's retained-teacher rule."""
    return origin in TAIL_ORIGINS or "champion" in origin


def percentile(values, fraction):
    """Linear interpolation between order statistics (position = fraction * (n - 1))."""
    require(values, "percentile of an empty sample")
    require(0 <= fraction <= 1, "percentile fraction outside [0, 1]")
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    low = int(math.floor(position))
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def cluster_bootstrap_ci(values_by_cluster, resamples=2000, seed=20260912, alpha=0.10):
    """Percentile CI of the mean of per-cluster means; clusters resampled with replacement."""
    require(
        isinstance(values_by_cluster, dict) and values_by_cluster,
        "cluster bootstrap requires at least one cluster",
    )
    require(type(resamples) is int and resamples >= 1 and 0 < alpha < 1, "invalid bootstrap policy")
    means, n_values = [], 0
    for key in sorted(values_by_cluster, key=repr):
        values = list(values_by_cluster[key])
        require(values and all(finite(v) for v in values), f"cluster {key!r} lacks finite values")
        means.append(statistics.fmean(values))
        n_values += len(values)
    rng = random.Random(seed)
    stats = sorted(statistics.fmean(rng.choices(means, k=len(means))) for _ in range(resamples))
    last = len(stats) - 1
    return dict(
        mean=statistics.fmean(means),
        ci_low=stats[min(last, int(len(stats) * alpha / 2))],
        ci_high=stats[min(last, int(len(stats) * (1 - alpha / 2)))],
        n_clusters=len(means),
        n_values=n_values,
        resamples=resamples,
        seed=seed,
        alpha=alpha,
        statistic="mean_of_cluster_means",
        method="percentile",
    )


def _arm_statistic(row, level):
    if level == "final":
        return row.get("best_final_gp"), False
    if "best_initial_arm" in row:
        return row.get("best_initial_arm"), False
    # Reports written before the best_initial_arm producer: the neural and shuffled
    # arms' initial slots are all origin neural_policy, so the legacy key is exact.
    if row.get("arm") in ("neural", "shuffled"):
        return row.get("best_initial_neural_seed"), True
    return None, True


def _contrast(groups, control, level):
    clusters, per_cut, counts = (
        {},
        {},
        dict(wins=0, ties=0, losses=0, incomplete=0, missing_values=0, fallback_rows=0),
    )
    for key in sorted(groups, key=repr):
        arms = groups[key]
        if "neural" not in arms or control not in arms:
            counts["incomplete"] += 1
            continue
        a, fallback_a = _arm_statistic(arms["neural"], level)
        b, fallback_b = _arm_statistic(arms[control], level)
        counts["fallback_rows"] += fallback_a + fallback_b
        if not (finite(a) and finite(b) and a > 0 and b > 0):
            counts["missing_values"] += 1
            continue
        r = math.log(a / b)
        outcome = "ties" if abs(r) < TIE else "wins" if r < 0 else "losses"
        counts[outcome] += 1
        cluster = (arms["neural"]["dataset"], arms["neural"]["cut"])
        clusters.setdefault(cluster, []).append(r)
        cut = per_cut.setdefault(
            cluster,
            dict(
                dataset=cluster[0],
                cut=cluster[1],
                family=arms["neural"].get("family"),
                n=0,
                wins=0,
                ties=0,
                losses=0,
                log_ratios=[],
            ),
        )
        cut["n"] += 1
        cut[outcome] += 1
        cut["log_ratios"].append(r)
    for cut in per_cut.values():
        cut["mean_log_ratio"] = statistics.fmean(cut["log_ratios"])
    result = dict(
        control=control,
        level=level,
        n_cuts=len(clusters),
        n_pairs=sum(c["n"] for c in per_cut.values()),
        mean_log_ratio=None,
        ci90=None,
        per_cut=[per_cut[k] for k in sorted(per_cut, key=repr)],
        **counts,
    )
    if clusters:
        ci = cluster_bootstrap_ci(clusters)
        result.update(mean_log_ratio=ci["mean"], ci90=[ci["ci_low"], ci["ci_high"]], bootstrap=ci)
    else:
        result["reason"] = f"no matched (neural, {control}) pairs with finite positive statistics"
    return result


def matched_seed_contrast(rows, partition, level="seed_only"):
    """Per-cut ln(neural/control) with a cluster-bootstrap 90% CI; ties when |r| < 1e-12."""
    require(level in ("seed_only", "final"), "unknown contrast level: " + repr(level))
    groups = {}
    for row in rows:
        if row.get("partition") != partition:
            continue
        require(row.get("arm") in ARMS, "unknown comparison arm: " + repr(row.get("arm")))
        key = (row["dataset"], row["cut"], row["rng_seed"], row.get("generation"))
        arms = groups.setdefault(key, {})
        require(row["arm"] not in arms, "duplicate comparison arm for " + repr(key))
        arms[row["arm"]] = row
    result = _contrast(groups, "generic", level)
    result.update(
        schema=SCHEMA,
        partition=partition,
        statistic="ln(neural / control); negative favours neural",
        win_rule="r < -1e-12 (neural lower)",
        tie_rule="|r| < 1e-12",
        cluster="(dataset, cut)",
        tail_caveat=TAIL_CAVEAT if level == "final" else None,
        vs_shuffled=_contrast(groups, "shuffled", level),
        final_tests_opened=False,
    )
    return result


def gp_improves_on_seed(rows, arm="neural"):
    """1[best_final_gp < arm seed best] over rows of one arm with both values finite."""
    seed_key = "best_initial_neural_seed" if arm == "neural" else "best_initial_arm"
    n = improved = 0
    gains = []
    for row in rows:
        if row.get("arm") != arm:
            continue
        seed, final = row.get(seed_key), row.get("best_final_gp")
        if not (finite(seed) and finite(final)):
            continue
        n += 1
        improved += final < seed
        gains.append((seed - final) / seed if seed > 0 else 0.0)
    return dict(
        n=n,
        improved=improved,
        rate=improved / n if n else None,
        mean_relative_gain=statistics.fmean(gains) if gains else None,
        arm=arm,
        seed_key=seed_key,
        rule="improved iff best_final_gp < seed best; relative gain (seed - final) / seed",
    )


def best_final_excluding_shared_tail(observations):
    """Best evaluated outcome among own-lineage origins; tail descendants stay contaminated."""
    best, histogram = None, {}
    for observation in observations:
        if observation.get("failure") is not None:
            continue
        score = metric(observation)
        if score is None:
            continue
        origin = observation.get("origin", "")
        histogram[origin] = histogram.get(origin, 0) + 1
        if origin in OWN_ORIGINS and (best is None or score < best[0]):
            best = (score, origin, observation.get("program_sha256"))
    return dict(
        best_rmse=best[0] if best else None,
        origin=best[1] if best else None,
        program_sha256=best[2] if best else None,
        tail_contaminated=True,
        units="original_target",
        included_origins=list(OWN_ORIGINS),
        excluded_origins=sorted(o for o in histogram if o not in OWN_ORIGINS),
        origin_histogram=dict(sorted(histogram.items())),
        caveat=TAIL_CAVEAT,
        reason=None if best else "no finite own-lineage outcome",
    )


def quality_partitions(report):
    """Per-partition triplet counts of one quality report under continuous_status.summarize_quality's rules.

    The report's rows were bound to their jobs when the reporter wrote them; the jobs manifest
    synthesized here only satisfies the summarizer's row/rng binding shape, it asserts nothing new.
    """
    from continuous_status import summarize_quality

    rows = report["rows"]
    return summarize_quality(
        report, dict(jobs=[dict(output=r["output"], rng_seed=r["rng_seed"]) for r in rows])
    )


def regression_signature(quality_by_generation, partition="meta_train"):
    """Per-generation neural_exclusive_wins and direct_seed_beats_teacher.

    Accepts a list of continuous_status quality rows ({generation, partitions}), a dict
    generation -> such a row, a dict generation -> partitions, or a dict generation -> full
    parallel.continuous_quality.v1 report (summarized from its rows).
    """
    if isinstance(quality_by_generation, list):
        table = {}
        for row in quality_by_generation:
            require(
                type(row.get("generation")) is int and row["generation"] not in table,
                "duplicate or unnumbered quality row",
            )
            table[row["generation"]] = row
    else:
        table = dict(quality_by_generation)
    generations = sorted(table)
    require(all(type(g) is int for g in generations), "quality generations must be integers")
    by_partition, checkpoints = {}, []
    for generation in generations:
        row = table[generation]
        require(isinstance(row, dict), f"quality entry for generation {generation} is not an object")
        partitions = (
            quality_partitions(row) if isinstance(row.get("rows"), list) else row.get("partitions", row)
        )
        checkpoints.append(row.get("checkpoint"))
        for name, counts in partitions.items():
            if not isinstance(counts, dict):
                continue
            series = by_partition.setdefault(
                name,
                dict(
                    generations=[],
                    neural_exclusive_wins=[],
                    direct_seed_beats_teacher=[],
                    neural_losses=[],
                    neural_best_ties=[],
                    neural_final_beats_teacher=[],
                ),
            )
            series["generations"].append(generation)
            for key in (
                "neural_exclusive_wins",
                "direct_seed_beats_teacher",
                "neural_losses",
                "neural_best_ties",
                "neural_final_beats_teacher",
            ):
                series[key].append(counts.get(key))
    chosen = by_partition.get(partition)
    require(chosen is not None or not generations, f"partition {partition!r} absent from every quality row")
    return dict(
        schema=SCHEMA,
        partition=partition,
        generations=generations,
        checkpoints=checkpoints,
        neural_exclusive_wins=[
            dict(zip(chosen["generations"], chosen["neural_exclusive_wins"])).get(g) for g in generations
        ]
        if chosen
        else [],
        direct_seed_beats_teacher=[
            dict(zip(chosen["generations"], chosen["direct_seed_beats_teacher"])).get(g) for g in generations
        ]
        if chosen
        else [],
        by_partition=by_partition,
        final_tests_opened=False,
    )


def monitoring_rows(monitoring):
    """Flatten a continuous_status monitoring block into binding-tagged rows (current + archived segments)."""
    rows = []
    for block in [monitoring, *monitoring.get("archived_policies", [])]:
        binding = {key: block["binding"][key] for key in BINDING_KEYS}
        rows.extend(dict(row, binding=binding) for row in block["history"])
    return rows


def _receipt_order(row):
    step = row.get("evaluated_step")
    return (step if type(step) is int else -1, row["timestamp"], row.get("checkpoint", ""))


def panel_null(receipts, metrics=PANEL_METRICS, min_pairs=10):
    """Block-to-block relative |delta| per metric between consecutive receipts of one binding."""
    segments = {}
    for row in receipts:
        require(
            all(finite(row.get(m)) for m in metrics),
            "nonfinite panel metric in " + str(row.get("checkpoint")),
        )
        require(finite(row.get("timestamp")), "monitor receipt lacks a timestamp")
        binding = row.get("binding") or {}
        segments.setdefault(tuple(binding.get(k) for k in BINDING_KEYS), []).append(row)
    deltas = {m: [] for m in metrics}
    extremes = {m: None for m in metrics}
    zero_reference = {m: 0 for m in metrics}
    summaries = []
    for key in sorted(segments, key=repr):
        rows = sorted(segments[key], key=_receipt_order)
        for previous, current in zip(rows, rows[1:]):
            for m in metrics:
                if previous[m] == 0:
                    zero_reference[m] += 1
                    continue
                rel = abs(current[m] - previous[m]) / abs(previous[m])
                deltas[m].append(rel)
                if extremes[m] is None or rel > extremes[m]["relative_delta"]:
                    extremes[m] = dict(
                        relative_delta=rel,
                        previous=previous[m],
                        current=current[m],
                        previous_checkpoint=previous.get("checkpoint"),
                        current_checkpoint=current.get("checkpoint"),
                    )
        summaries.append(
            dict(
                binding=dict(zip(BINDING_KEYS, key)),
                n_rows=len(rows),
                n_pairs=max(0, len(rows) - 1),
                first_checkpoint=rows[0].get("checkpoint"),
                last_checkpoint=rows[-1].get("checkpoint"),
            )
        )
    per_metric = {}
    for m in metrics:
        values = deltas[m]
        per_metric[m] = dict(
            p50=percentile(values, 0.5) if values else None,
            p90=percentile(values, 0.9) if values else None,
            max=max(values) if values else None,
            n_pairs=len(values),
            zero_reference_pairs=zero_reference[m],
            extreme=extremes[m],
        )

    def guard(name, metric_name):
        p90 = per_metric.get(metric_name, {}).get("p90")
        return max(GUARD_FLOORS[name], 2 * p90) if p90 is not None else None

    n_pairs = min((per_metric[m]["n_pairs"] for m in metrics), default=0)
    return dict(
        schema=SCHEMA,
        metrics=per_metric,
        guards=dict(
            g_nll=guard("g_nll", "sequence_nll"),
            g_bt=guard("g_bt", "ranking_bt"),
            floors=dict(GUARD_FLOORS),
            rule="g = max(floor, 2 * p90 of block-to-block relative |delta|)",
            pairs_sufficient=n_pairs >= min_pairs,
            min_pairs=min_pairs,
            reason=None if n_pairs else "no consecutive receipt pairs",
        ),
        n_receipts=sum(s["n_rows"] for s in summaries),
        n_segments=len(summaries),
        segments=summaries,
        relative_delta="abs(next - previous) / abs(previous)",
        percentile_method="linear_interpolation",
        order="(evaluated_step, timestamp, checkpoint) within one (panel, corpus, executable) binding",
        final_tests_opened=False,
    )


def tokens(genome):
    return re.findall(r"\(|\)|[^\s()]+", genome)


def diversity(seed_populations, snapshots):
    """Unique neural emissions per context, DSL token length, fresh programs per snapshot."""
    unique_per_context, lengths, fresh = [], [], []
    for population in seed_populations:
        candidates = population.get("candidates")
        require(isinstance(candidates, list), "seed population lacks candidates")
        neural = [c["genome"] for c in candidates if c.get("origin") == "neural_policy"]
        require(all(isinstance(g, str) for g in neural), "seed genome is not a string")
        unique_per_context.append(len(set(neural)))
        lengths.extend(len(tokens(g)) for g in neural)
    for snapshot in snapshots:
        if type(snapshot.get("unique_fresh_programs")) is int:
            fresh.append(snapshot["unique_fresh_programs"])
            continue
        records = snapshot.get("fresh_records")
        require(isinstance(records, dict), "snapshot lacks fresh_records or unique_fresh_programs")
        fresh.append(len({record for group in records.values() for record in group}))
    return dict(
        schema=SCHEMA,
        unique_neural_per_context_median=statistics.median(unique_per_context)
        if unique_per_context
        else None,
        unique_neural_per_context=unique_per_context,
        n_populations=len(unique_per_context),
        mean_token_length=statistics.fmean(lengths) if lengths else None,
        n_neural_emissions=len(lengths),
        unique_fresh_programs_median=statistics.median(fresh) if fresh else None,
        unique_fresh_programs=fresh,
        n_snapshots=len(fresh),
        token_rule=TOKEN_RULE,
        fresh_rule=FRESH_RULE,
        final_tests_opened=False,
    )
