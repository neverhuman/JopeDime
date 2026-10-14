"""Strict baseline observations from owner exports; no execution or qualification.

All windows are half-open UTC seconds. Coverage is an explicit producer claim,
never inferred from available rows. Missing proofs produce NULL, not zero.
"""

import json
import math
import statistics
from collections import Counter

from .operational import FEEDBACK_STAGES, UNMEASURED, row

DOMAINS = ("throughput", "arrivals", "feedback", "phases", "cache", "cost")
COMPARISON = (
    "task",
    "dataset_sha256",
    "cut_id",
    "evaluator_sha256",
    "vocabulary_sha256",
    "canonicalizer_sha256",
    "protocol",
    "role_map_sha256",
)


def number(value, name, *, nonnegative=True):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name}: finite number required")
    if not math.isfinite(value) or nonnegative and value < 0:
        raise ValueError(f"{name}: invalid number")
    return value


def interval(value):
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError("explicit interval required")
    start, end = (number(x, "clock", nonnegative=False) for x in value)
    if end <= start:
        raise ValueError("positive interval required")
    return start, end


def unique(records, key):
    """Identical aliases count once; conflicting physical identities refuse."""
    seen = {}
    for record in records:
        identity = record.get(key)
        if not isinstance(identity, str) or not identity:
            raise ValueError(f"missing {key}")
        if identity in seen and seen[identity] != record:
            raise ValueError(f"conflicting {key}")
        seen[identity] = record
    return list(seen.values())


def duration(start, end):
    """Clock records retain domain and combined synchronization uncertainty."""
    if start is None or end is None:
        return None, None
    a = number(start["at"], "clock", nonnegative=False)
    b = number(end["at"], "clock", nonnegative=False)
    if not start.get("domain") or not end.get("domain"):
        return None, None
    uncertainties = [start.get("uncertainty_seconds"), end.get("uncertainty_seconds")]
    if start["domain"] != end["domain"] and any(x is None for x in uncertainties):
        return None, None
    uncertainty = sum(
        number(x, "clock uncertainty") for x in uncertainties if x is not None
    )
    if b < a:
        if start["domain"] == end["domain"] or a - b > uncertainty:
            raise ValueError("clock_order")
        return None, uncertainty
    return b - a, uncertainty


class Baseline:
    def __init__(self, exports, *, window, date, commit, sources):
        self.start, self.end = interval(window)
        self.hours = (self.end - self.start) / 3600
        self.exports, self.date, self.commit, self.sources = (
            exports,
            date,
            commit,
            sources,
        )
        self.rows = []
        if set(exports) - set(DOMAINS):
            raise ValueError("unknown export domain")
        for domain, export in exports.items():
            if (
                export.get("schema") != "jope.rlfw-baseline-export/v1"
                or export.get("domain") != domain
            ):
                raise ValueError("baseline export schema/domain")
            if interval(export["window"]) != (self.start, self.end):
                raise ValueError("export window mismatch; no prorating")
            if type(export.get("complete")) is not bool or not export.get("scope"):
                raise ValueError("explicit scope and coverage required")

    def data(self, domain):
        return self.exports.get(domain, {}).get("data", {})

    def complete(self, domain):
        return self.exports.get(domain, {}).get("complete") is True

    def add(self, domain, metric, value, unit, *, n=None, reason=None, **details):
        export = self.exports.get(domain, {})
        if value is not None:
            number(value, metric, nonnegative=False)
        metadata = {
            "schema": "jope.rlfw-baseline-observation/v1",
            "window": [self.start, self.end],
            "producer_commit": export.get("producer_commit"),
            "producer_evidence_status": export.get("evidence_status"),
            "coverage_complete": export.get("complete"),
            "reason": reason if value is None else None,
            **details,
        }
        if value is None and metadata["reason"] is None:
            metadata["reason"] = (
                "missing_export" if not export else "missing_proof_or_coverage"
            )
        self.rows.append(
            row(
                metric,
                value,
                date=self.date,
                commit=self.commit,
                source=self.sources.get(
                    domain, self.sources.get("manifest", "unmeasured")
                ),
                unit=unit,
                scope=export.get("scope", "baseline/" + domain),
                n=n,
                note="Owner-export readback only; NON-QUALIFYING. "
                + (metadata["reason"] or ""),
                details=metadata,
                context_set=export.get("cohort_sha256", "unmeasured"),
            )
        )

    def throughput(self):
        data = self.data("throughput")
        allocations = unique(data.get("allocations", []), "span_id")
        grouped, allocated, quota = {}, 0.0, 0.0
        for span in allocations:
            a, b = interval(span["interval"])
            cores = number(span["cores"], "allocation")
            if cores <= 0 or not span.get("allocation_id"):
                raise ValueError("invalid_allocation")
            for c, d in grouped.setdefault(span["allocation_id"], []):
                if max(a, c) < min(b, d):
                    raise ValueError("overlapping_allocation")
            grouped[span["allocation_id"]].append((a, b))
            amount = cores * max(0, min(b, self.end) - max(a, self.start)) / 3600
            if span.get("source_kind") == "reservation" and span.get(
                "owner_receipt_sha256"
            ):
                allocated += amount
            elif span.get("source_kind") == "quota_capacity":
                quota += amount
            else:
                raise ValueError("allocation provenance required")
        reserved_complete = (
            self.complete("throughput") and data.get("allocation_complete") is True
        )
        # A quota is explicitly a different denominator, even when numerically equal.
        if any(s.get("source_kind") != "reservation" for s in allocations):
            reserved_complete = False
        allocated = allocated if reserved_complete else None
        counters = unique(data.get("cpu_counters", []), "counter_id")
        cpu = 0.0
        for counter in counters:
            a, b = interval(counter["interval"])
            if (a, b) != (self.start, self.end):
                raise ValueError("CPU counter window mismatch; no prorating")
            if (
                not counter.get("invocation_id")
                or not counter.get("scope_id")
                or not counter.get("receipt_sha256")
            ):
                raise ValueError("CPU counter identity required")
            if not isinstance(counter.get("ancestors"), list):
                raise ValueError("CPU scope relationships required")
            first = number(counter["start_seconds"], "CPU counter")
            last = number(counter["end_seconds"], "CPU counter")
            if last < first:
                raise ValueError("counter_reset")
            cpu += last - first
        kinds = {c.get("counter_kind") for c in counters}
        if kinds - {"evaluator_cgroup", "process_rusage"} or len(kinds) > 1:
            raise ValueError("CPU counter kinds cannot be mixed")
        for i, counter in enumerate(counters):
            for other in counters[i + 1 :]:
                if (
                    counter["scope_id"] == other["scope_id"]
                    or counter["scope_id"] in other["ancestors"]
                    or other["scope_id"] in counter["ancestors"]
                ):
                    raise ValueError("cpu_double_count")
        active = (
            cpu / 3600
            if counters
            and self.complete("throughput")
            and data.get("cpu_complete") is True
            else None
        )
        counts = data.get("counts")
        counts_valid = (
            self.complete("throughput")
            and data.get("counts_complete") is True
            and counts is not None
        )
        if counts is not None:
            if interval(counts["interval"]) != (self.start, self.end):
                counts_valid = False
            for name, value in counts.items():
                if name != "interval" and value is not None:
                    if type(value) is not int or value < 0:
                        raise ValueError("nonnegative integer counters required")
        programs = counts.get("programs") if counts_valid else None
        self.add(
            "throughput",
            "allocated_core_hours",
            allocated,
            "allocated core-hours",
            denominator="owner reservation spans",
        )
        self.add(
            "throughput",
            "quota_capacity_core_hours",
            quota if allocations else None,
            "quota core-hours",
            denominator="quota ceiling; not reserved allocation",
        )
        self.add(
            "throughput",
            "active_cpu_core_hours",
            active,
            "active CPU-hours",
            counter_kind=next(iter(kinds), None),
        )
        self.add(
            "throughput", "report_window_seconds", self.end - self.start, "seconds"
        )
        for name in (
            "programs",
            "folds",
            "predictions",
            "compiles",
            "cached_disclosures",
            "physical_attempts",
            "missing_outcomes",
        ):
            self.add(
                "throughput", name, counts.get(name) if counts_valid else None, name
            )
        for metric, denominator in [
            ("programs_per_allocated_core_hour", allocated),
            ("programs_per_active_cpu_hour", active),
            ("programs_per_wall_hour", self.hours),
        ]:
            value = (
                programs / denominator
                if programs is not None and denominator is not None and denominator > 0
                else None
            )
            self.add(
                "throughput",
                metric,
                value,
                "programs/hour",
                denominator_value=denominator,
            )
        attempts = counts.get("physical_attempts") if counts_valid else None
        missing = counts.get("missing_outcomes") if counts_valid else None
        if missing is not None and attempts is not None and missing > attempts:
            raise ValueError("missing outcomes exceed physical attempts")
        self.add(
            "throughput",
            "missing_outcome_rate",
            missing / attempts if missing is not None and attempts else None,
            "fraction",
        )
        demand = data.get("thread_configuration")
        product = ratio = None
        if demand:
            workers, inner, cores = (
                number(demand[k], k)
                for k in ("workers", "inner_threads", "declared_cores")
            )
            if min(workers, inner, cores) <= 0:
                raise ValueError("positive thread configuration required")
            product, ratio = workers * inner, workers * inner / cores
        self.add(
            "throughput", "configured_thread_product", product, "configured threads"
        )
        self.add(
            "throughput",
            "configured_oversubscription_ratio",
            ratio,
            "ratio",
            measured_peak_threads=None,
        )

    def arrivals(self):
        data = self.data("arrivals")
        records = unique(data.get("records", []), "outcome_id")
        complete = (
            self.complete("arrivals") and data.get("lifetime_novelty_complete") is True
        )
        pairs, programs = set(), set()
        catalog = data.get("frozen_catalog_sha256")
        catalog_complete = complete and bool(catalog)
        catalog_membership = {}
        identities, first_qualification, first_program = {}, {}, {}
        roles = {event.get("role") for event in records if event.get("role")}
        if len(roles) > 1:
            raise ValueError("mixed role cohorts must be reported separately")
        for event in records:
            # Frozen catalog membership does not depend on context or report window.
            program = event.get("canonical_program_sha256")
            membership = event.get("in_frozen_catalog")
            if (
                catalog
                and program
                and event.get("catalog_membership_proof_sha256")
                and type(membership) is bool
            ):
                catalog_key = (catalog, program)
                if (
                    catalog_key in catalog_membership
                    and catalog_membership[catalog_key] != membership
                ):
                    raise ValueError("conflicting frozen catalog membership")
                catalog_membership[catalog_key] = membership
            identity = event.get("comparison_identity", {})
            if not all(identity.get(k) for k in COMPARISON) or not all(
                event.get(k)
                for k in ("context_id", "parent_id", "canonical_program_sha256", "role")
            ):
                complete = catalog_complete = False
                continue
            key = (
                event["context_id"],
                event["canonical_program_sha256"],
                json.dumps(identity, sort_keys=True),
            )
            bound = (event["parent_id"], event["role"])
            if key in identities and identities[key] != bound:
                raise ValueError("comparison_identity_mismatch")
            identities[key] = bound
            proofs = (
                "qualification_receipt_sha256",
                "remote_version_readback_sha256",
                "first_qualified_proof_sha256",
            )
            if event.get("qualified") is not True or not all(
                event.get(k) for k in proofs
            ):
                complete = catalog_complete = False
                continue
            at = number(
                event["first_qualified_at"], "qualification clock", nonnegative=False
            )
            if key in first_qualification and first_qualification[key] != at:
                raise ValueError("conflicting first_qualification clock")
            first_qualification[key] = at
            if not self.start <= at < self.end:
                continue
            pairs.add(key)
            if (
                not event.get("catalog_membership_proof_sha256")
                or type(event.get("in_frozen_catalog")) is not bool
                or not event.get("first_program_seen_proof_sha256")
                or event.get("first_program_seen_at") is None
            ):
                catalog_complete = False
            elif not event["in_frozen_catalog"]:
                first = number(
                    event["first_program_seen_at"],
                    "program novelty clock",
                    nonnegative=False,
                )
                program = event["canonical_program_sha256"]
                if program in first_program and first_program[program] != first:
                    raise ValueError("conflicting global program novelty clock")
                first_program[program] = first
                if first > at:
                    raise ValueError("program novelty after first qualification")
                if self.start <= first < self.end:
                    programs.add(program)
        outcomes = len(pairs) if complete else None
        noncatalog = len(programs) if complete and catalog_complete else None
        parent_counts = dict(Counter(identities[key][0] for key in pairs))
        self.qualified_outcomes = outcomes
        self.add(
            "arrivals",
            "new_qualified_context_program_outcomes",
            outcomes,
            "context-program outcomes",
            parent_counts=parent_counts if complete else None,
            role=next(iter(roles), None),
        )
        self.add(
            "arrivals",
            "qualified_context_program_outcomes_per_hour",
            outcomes / self.hours if outcomes is not None else None,
            "outcomes/hour",
        )
        self.add(
            "arrivals", "new_qualified_non_catalog_programs", noncatalog, "programs"
        )
        self.add(
            "arrivals",
            "qualified_non_catalog_programs_per_hour",
            noncatalog / self.hours if noncatalog is not None else None,
            "programs/hour",
        )

    def feedback(self):
        data = self.data("feedback")
        observed_until = data.get("observed_until")
        if observed_until is not None and observed_until.get("at") != self.end:
            raise ValueError("observation clock must equal fixed window end")
        episodes = unique(data.get("episodes", []), "episode_id")
        if not episodes:
            for stage in FEEDBACK_STAGES:
                self.add("feedback", "feedback_timestamp:" + stage, None, "UTC seconds")
        complete_values, qualified_values, exposure_values, censored = [], [], [], []
        censored_episodes, unknown_latency = 0, 0
        for episode in episodes:
            if not all(
                episode.get(k)
                for k in (
                    "context_id",
                    "parent_id",
                    "outcome_id",
                    "target_packet_sha256",
                    "support_sha256",
                )
            ):
                raise ValueError("feedback identity required")
            raw = episode.get("clocks", {})
            clocks = {}
            for stage in (*FEEDBACK_STAGES, "first_exposure"):
                clock = raw.get(stage)
                if (
                    clock is not None
                    and number(clock["at"], "clock", nonnegative=False) <= self.end
                ):
                    clocks[stage] = clock
            ordered = [clocks[s] for s in FEEDBACK_STAGES if s in clocks]
            for first, last in zip(ordered, ordered[1:]):
                duration(first, last)
            update_fields = (
                "consumption_receipt_sha256",
                "parameter_change_receipt_sha256",
                "input_snapshot_sha256",
                "learner_invocation",
            )
            update = episode.get("optimizer_update")
            update_verified = (
                type(update) is int
                and update > 0
                and all(episode.get(k) for k in update_fields)
                and all(
                    episode.get("comparison_identity", {}).get(k) for k in COMPARISON
                )
            )
            if not update_verified:
                clocks.pop("first_trained", None)
            if not episode.get("exposure_receipt_sha256"):
                clocks.pop("first_exposure", None)
            proposed = clocks.get("proposed")
            trained = clocks.get("first_trained")
            elapsed, uncertainty = duration(proposed, trained)
            qualified, _ = duration(clocks.get("remotely_qualified"), trained)
            exposure, _ = duration(proposed, clocks.get("first_exposure"))
            if elapsed is not None:
                complete_values.append(elapsed)
            elif trained is not None:
                unknown_latency += 1
            if trained is None:
                censored_episodes += 1
            if qualified is not None:
                qualified_values.append(qualified)
            if exposure is not None:
                exposure_values.append(exposure)
            age = None
            if proposed is not None and trained is None:
                # The observation clock must have the same time domain or a bound.
                age, _ = duration(proposed, self.data("feedback").get("observed_until"))
                if age is not None:
                    censored.append(age)
            common = {
                k: episode[k]
                for k in ("episode_id", "context_id", "parent_id", "outcome_id")
            }
            for stage in FEEDBACK_STAGES:
                self.add(
                    "feedback",
                    "feedback_timestamp:" + stage,
                    clocks.get(stage, {}).get("at"),
                    "UTC seconds",
                    **common,
                    verified_first_update=update_verified,
                )
            for first, last in zip(FEEDBACK_STAGES, FEEDBACK_STAGES[1:]):
                value, bound = duration(clocks.get(first), clocks.get(last))
                self.add(
                    "feedback",
                    f"feedback_stage_seconds:{first}->{last}",
                    value,
                    "seconds",
                    **common,
                    uncertainty_seconds=bound,
                )
            for name, value in [
                ("feedback_to_first_update_seconds", elapsed),
                ("qualified_feedback_to_first_update_seconds", qualified),
                ("T_feedback", exposure),
                ("feedback_censored_age_seconds", age),
            ]:
                self.add(
                    "feedback",
                    name,
                    value,
                    "seconds",
                    **common,
                    right_censored=trained is None,
                    uncertainty_seconds=uncertainty,
                )
        for name, values in [
            ("proposal_to_update", complete_values),
            ("qualified_to_update", qualified_values),
            ("proposal_to_exposure", exposure_values),
        ]:
            for suffix, value in [
                ("mean_seconds", statistics.mean(values) if values else None),
                (
                    "p95_seconds",
                    sorted(values)[math.ceil(0.95 * len(values)) - 1]
                    if values
                    else None,
                ),
            ]:
                self.add(
                    "feedback",
                    name + ":completed_" + suffix,
                    value,
                    "seconds",
                    n=len(values),
                    completed_only=True,
                    censored_n=censored_episodes,
                    episodes=len(episodes),
                )
        self.add(
            "feedback",
            "feedback_completed_n",
            len(complete_values) if self.complete("feedback") else None,
            "episodes",
        )
        self.add(
            "feedback",
            "feedback_censored_n",
            censored_episodes if self.complete("feedback") else None,
            "episodes",
        )
        self.add(
            "feedback",
            "feedback_unknown_latency_n",
            unknown_latency if self.complete("feedback") else None,
            "episodes",
        )
        self.add(
            "feedback",
            "feedback_censored_age_mean_seconds",
            statistics.mean(censored) if censored else None,
            "seconds",
            n=len(censored),
        )

    def phases(self):
        data = self.data("phases")
        if (
            data.get("observed_until") is not None
            and data["observed_until"].get("at") != self.end
        ):
            raise ValueError("observation clock must equal fixed window end")
        records = unique(data.get("records", []), "phase_id")
        for name in ("queue", "compile", "execution", "custody"):
            values, ages = [], []
            for phase in records:
                if phase.get("name") != name:
                    continue
                start, end = phase.get("start"), phase.get("end")
                if name == "queue":
                    start, end = phase.get("enqueued"), phase.get("dequeued")
                elif name == "custody" and not all(
                    phase.get(k) for k in ("remote_version_id", "readback_sha256")
                ):
                    end = None
                if (
                    end is not None
                    and number(end["at"], "phase clock", nonnegative=False) > self.end
                ):
                    end = None
                value, _ = duration(start, end)
                if value is not None:
                    values.append(value)
                elif end is None and start is not None:
                    age, _ = duration(start, data.get("observed_until"))
                    if age is not None:
                        ages.append(age)
            self.add(
                "phases",
                name + ":completed_mean_seconds",
                statistics.mean(values) if values else None,
                "seconds",
                n=len(values),
                censored_n=len(ages),
                phases_not_additive=True,
            )
            self.add(
                "phases",
                name + ":censored_age_mean_seconds",
                statistics.mean(ages) if ages else None,
                "seconds",
                n=len(ages),
            )
        gauges = data.get("queue_events", [])
        area, depth, cursor = 0.0, None, self.start
        previous = None
        for sample in gauges:
            at = number(sample["at"], "queue clock", nonnegative=False)
            current = sample["depth"]
            if (
                type(current) is not int
                or current < 0
                or previous is not None
                and at <= previous
            ):
                raise ValueError("invalid queue event stream")
            previous = at
            if at <= self.start:
                depth = current
            elif at < self.end:
                if depth is not None:
                    area += depth * (at - cursor)
                cursor, depth = at, current
        valid = (
            self.complete("phases")
            and data.get("queue_event_coverage_complete") is True
            and gauges
            and gauges[0]["at"] <= self.start
        )
        if depth is not None:
            area += depth * (self.end - cursor)
        self.add(
            "phases",
            "queue_time_weighted_depth",
            area / (self.end - self.start) if valid else None,
            "items",
            reason="queue_coverage_incomplete",
        )
        self.add(
            "phases", "queue_pending_at_window_end", depth if valid else None, "items"
        )
        # No phase sum is exposed: compile/custody/execution may nest or overlap.

    def cache(self):
        data = self.data("cache")
        records = unique(data.get("lookups", []), "lookup_id")
        identity, kind = data.get("identity_sha256"), data.get("cache_kind")
        for lookup in records:
            if (
                not identity
                or not kind
                or lookup.get("identity_sha256") != identity
                or lookup.get("cache_kind") != kind
            ):
                raise ValueError("cache_identity_mismatch")
            if type(lookup.get("hit")) is not bool:
                raise ValueError("explicit cache hit/miss required")
        total = len(records) if self.complete("cache") and identity and kind else None
        hits = sum(r["hit"] for r in records) if total is not None else None
        self.add(
            "cache",
            "cache_lookups",
            total,
            "lookups",
            cache_kind=kind,
            identity_sha256=identity,
        )
        self.add(
            "cache",
            "cache_hits",
            hits,
            "hits",
            cache_kind=kind,
            identity_sha256=identity,
        )
        self.add(
            "cache",
            "cache_hit_rate",
            hits / total if total else None,
            "fraction",
            cache_kind=kind,
            identity_sha256=identity,
        )

    def cost(self):
        data = self.data("cost")
        records = unique(data.get("records", []), "ledger_event_id")
        identities = {
            (r.get("currency"), r.get("cost_kind"), r.get("pricing_identity"))
            for r in records
        }
        if len(identities) > 1 or any(
            currency != "USD" or kind not in ("estimate", "billed", "unmeasured")
            for currency, kind, _ in identities
        ):
            raise ValueError("cost_identity_mismatch")
        eligible, stock = [], []
        for record in records:
            if record.get("usd") is not None:
                number(record["usd"], "invalid_cost")
            if record.get("bytes") is not None:
                number(record["bytes"], "bytes")
            if record.get("request_operations") is not None and (
                type(record["request_operations"]) is not int
                or record["request_operations"] < 0
            ):
                raise ValueError("nonnegative request counter required")
            a, b = interval(record["interval"])
            if record.get("bytes_measure_kind") == "stock":
                stock.append(record)
            if (a, b) == (self.start, self.end) and record.get(
                "window_kind"
            ) == "report_window":
                eligible.append(record)
        for i, record in enumerate(records):
            for other in records[i + 1 :]:
                if (
                    record["bucket"] != other["bucket"]
                    or record["cost_category"] != other["cost_category"]
                ):
                    continue
                a, b = interval(record["interval"])
                c, d = interval(other["interval"])
                p, q = record["prefix"], other["prefix"]
                # Object-store prefixes are literal, including the empty root.
                overlaps = p.startswith(q) or q.startswith(p)
                if overlaps and max(a, c) < min(b, d):
                    raise ValueError("scope_double_count")
        known = [r["usd"] for r in eligible if r.get("usd") is not None]
        expected = set(data.get("expected_categories", []))
        complete = (
            self.complete("cost")
            and bool(expected)
            and expected == {r["cost_category"] for r in eligible}
            and all(r.get("usd") is not None for r in eligible)
            and all(r.get("pricing_identity") for r in eligible)
        )
        total = sum(known) if complete else None
        kind = next(iter(identities), (None, None, None))[1]
        self.add("cost", "total_usd", total, "USD", cost_kind=kind)
        self.add(
            "cost",
            "known_partial_usd",
            sum(known) if known else None,
            "USD",
            cost_kind=kind,
            not_a_total=True,
        )
        added = [r for r in eligible if r.get("bytes_measure_kind") == "added"]
        self.add(
            "cost",
            "bytes_added",
            sum(r["bytes"] for r in added)
            if self.complete("cost")
            and data.get("bytes_added_complete") is True
            and added
            and all(r.get("bytes") is not None for r in added)
            else None,
            "bytes",
        )
        self.add(
            "cost",
            "stock_bytes",
            sum(r["bytes"] for r in stock)
            if stock
            and all(
                r.get("bytes") is not None and r.get("stock_as_of") == self.end
                for r in stock
            )
            else None,
            "stock bytes",
            not_daily_additions=True,
        )
        self.add(
            "cost",
            "request_operations",
            sum(r["request_operations"] for r in eligible)
            if self.complete("cost")
            and data.get("requests_complete") is True
            and eligible
            and all(r.get("request_operations") is not None for r in eligible)
            else None,
            "requests",
        )
        outcomes = getattr(self, "qualified_outcomes", None)
        arrivals = self.exports.get("arrivals", {})
        cost = self.exports.get("cost", {})
        same = all(
            arrivals.get(k) and arrivals.get(k) == cost.get(k)
            for k in ("scope", "cohort_sha256")
        )
        self.add(
            "cost",
            "usd_per_qualified_outcome",
            total / outcomes if total is not None and outcomes and same else None,
            "USD/outcome",
            reason="missing_matching_cost_or_positive_qualified_denominator",
        )


def baseline_rows(exports, *, window, date, commit, sources):
    result = Baseline(exports, window=window, date=date, commit=commit, sources=sources)
    for domain in DOMAINS:
        getattr(result, domain)()
    available = {record["metric"] for record in result.rows}
    for metric, unit in UNMEASURED.items():
        if metric not in available:
            result.add(
                "availability",
                metric,
                None,
                unit,
                reason="no_current_identity_bound_owner_export",
            )
    return result.rows
