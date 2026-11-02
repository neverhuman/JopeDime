"""Test methods split from test_no_winner_bootstrap for maintainability."""

from .test_no_winner_bootstrap import *  # noqa: F401,F403

def test_coverage_probabilities_drive_five_exploit_two_explore_one_audit(self):
    packet = self.arms().prepare_slate(self.contexts[0], self.arm, 20261009)
    self.assertEqual(len(packet.selected), 8)
    self.assertEqual(
        [c.program.dsl for c in packet.selected[:5]],
        ["leaf31", "leaf30", "leaf29", "leaf28", "leaf27"],
    )
    self.assertTrue(all(c.predicted["q"] == -1.0 for c in packet.slate))
    self.assertTrue(all(c.propensity is None for c in packet.selected[:5]))
    self.assertEqual([c.propensity for c in packet.selected[5:7]], [1 / 27, 1 / 26])
    self.assertEqual(packet.selected[-1].propensity, 1 / 40)
    self.assertEqual(
        packet.selected,
        coverage_selection(
            packet.slate,
            packet.predicted_probabilities,
            packet.raw_draws[-1],
            packet.selection_seed,
        ),
    )
    self.assertEqual(CoverageSlate.from_dict(packet.to_dict()), packet)
    self.assertEqual(len(packet.raw_draws), 33)


def test_unknown_probability_and_packet_changes_refused(self):
    for bad in (float("nan"), -0.1, 1.1, None):
        with self.subTest(bad=bad), self.assertRaises(ValueError):
            self.arms(
                probabilities=lambda cid, arm, rows, pin, bad=bad: [
                    bad for _ in rows
                ]
            ).prepare_slate(self.contexts[0], self.arm, 20261009)
    with self.assertRaises(ValueError):
        self.arms(probabilities=lambda *a: []).prepare_slate(
            self.contexts[0], self.arm, 20261009
        )
    packet = (
        self.arms().prepare_slate(self.contexts[0], self.arm, 20261009).to_dict()
    )
    packet["selected"] = list(reversed(packet["selected"]))
    with self.assertRaises(ValueError):
        CoverageSlate.from_dict(packet)


@mock.patch("rlfw.arena.oracle.time.time", return_value=1791540000.0)
def test_busy_arena_blocks_all_proposals_and_evaluations(self, clock):
    value = self.scheduler(granted=False)
    with self.assertRaises(RuntimeError):
        value.run()
    self.assertEqual(self.draws, 0)
    self.assertEqual(self.calls, [])
    self.assertEqual(value.coverage_report()["contexts_attempted"], 0)


@mock.patch("rlfw.arena.oracle.time.time", return_value=1791540000.0)
def test_first_labels_measured_only_and_exact_partial_yield_costs(self, clock):
    value = self.scheduler()
    rows = value.run(max_contexts=1)
    self.assertEqual(len(rows), 8)
    self.assertTrue(all(r.q_grid is not None and r.q_grid >= 0 for r in rows))
    self.assertTrue(all(r.charged for r in rows))
    report = value.coverage_report()
    self.assertEqual(report["contexts_attempted"], 1)
    self.assertEqual(report["first_label_yield"], 1)
    self.assertEqual(report["exit_usable_yield"], 1)
    self.assertEqual(len(report["unattempted_contexts_unknown"]), 2)
    self.assertEqual(report["evaluator_wall_seconds"], 2 * len(self.calls))
    self.assertEqual(report["evaluator_cpu_seconds"], 3 * len(self.calls))
    c = next(c for c in self.contexts if c.context_id == rows[0].context_id)
    events = value.ledger.events(c.hwm_key)
    self.assertEqual(events[0]["event"], "first_label")
    self.assertTrue(all(e["publication"]["source"] == "measured" for e in events))
    self.assertEqual(
        [e["publication"]["q_grid"] for e in events], [r.q_grid for r in rows]
    )


@mock.patch("rlfw.arena.oracle.time.time", return_value=1791540000.0)
def test_g0_and_program_failure_never_become_zero_q(self, clock):
    value = self.scheduler(evaluate=lambda r: self.result(r, "program_failed"))
    rows = value.run(max_contexts=1)
    self.assertTrue(all(r.q_grid is None and r.q is None for r in rows))
    report = value.coverage_report()
    self.assertEqual(report["first_label_yield"], 0)
    self.assertEqual(report["exit_usable_yield"], 0)
    c = next(c for c in self.contexts if c.context_id == rows[0].context_id)
    self.assertTrue(
        all(
            e["publication"]["q_grid"] is None
            for e in value.ledger.events(c.hwm_key)
        )
    )


@mock.patch("rlfw.arena.oracle.time.time", return_value=1791540000.0)
def test_exhausted_infra_unknown_attempts_publish_refusals_and_censor(self, clock):
    def evaluate(request):
        return {**self.result(request, "infra_unknown"), "cost_unknown": True}

    value = self.scheduler(evaluate=evaluate)
    rows = value.run(max_contexts=1)
    self.assertEqual(len(rows), 8)
    self.assertEqual(len(self.calls), 24)
    self.assertEqual(sum(row.attempts for row in rows), 24)
    self.assertTrue(all(row.outcome == "censored" for row in rows))
    self.assertTrue(all(row.censor_reason == "infra_unknown" for row in rows))
    self.assertTrue(all(row.q_grid is None and row.q is None for row in rows))
    events = [
        event
        for context in self.contexts
        for event in value.ledger.events(context.hwm_key)
    ]
    self.assertEqual(len(events), 8)
    self.assertTrue(
        all(
            event["event"] == "refused"
            and event["reason"] == "infra_unknown"
            and event["publication"]["q_grid"] is None
            for event in events
        )
    )
    report = value.coverage_report()
    self.assertEqual(report["contexts_attempted"], 1)
    self.assertEqual(report["contexts_with_valid_program"], 0)
    self.assertEqual(report["contexts_exit_usable"], 0)
    self.assertEqual(report["censored_logical_slots"], 8)
    self.assertEqual(report["parity_unqualified_logical_slots"], 0)
    self.assertEqual(len(report["closed_contexts"]), 1)
    self.assertEqual(report["unknown_cost_attempts"], 24)
    self.assertIsNone(report["evaluator_wall_seconds"])
    self.assertIsNone(report["evaluator_cpu_seconds"])


@mock.patch("rlfw.arena.oracle.time.time", return_value=1791540000.0)
def test_r5_parity_unqualified_is_charged_once_without_retry_or_label(self, clock):
    value = self.scheduler(evaluate=self.parity_result)
    rows = value.run(max_contexts=1)
    self.assertEqual(len(rows), 8)
    distinct = {r.program_sha256 for r in rows}
    # One native request per distinct program; repeats reuse its terminal fate.
    self.assertEqual(len(self.calls), len(distinct))
    self.assertTrue(all(call["attempt"] == 1 for call in self.calls))
    self.assertTrue(
        all(
            r.outcome == "parity_unqualified"
            and r.charged
            and r.attempts == 1
            and not r.qualified
            and r.q is None
            and r.q_grid is None
            and r.receipt_sha256 is None
            and r.censor_reason is None
            and r.unknown_cost_attempts == 0
            for r in rows
        )
    )
    self.assertTrue(all(value.committed(r) for r in rows))
    self.assertEqual(sum(not r.cache_hit for r in rows), len(distinct))
    self.assertEqual(sum(r.wall_seconds for r in rows), 2.0 * len(distinct))
    events = [e for c in self.contexts for e in value.ledger.events(c.hwm_key)]
    self.assertEqual(len(events), 8)
    self.assertTrue(
        all(
            e["event"] == "refused"
            and e["reason"] is not None
            and e["publication"]["outcome"] == "parity_unqualified"
            and e["publication"]["q_grid"] is None
            and e["best_q_grid"] is None
            for e in events
        )
    )
    report = value.coverage_report()
    self.assertEqual(report["contexts_attempted"], 1)
    self.assertEqual(report["contexts_with_valid_program"], 0)
    self.assertEqual(report["censored_logical_slots"], 0)
    self.assertEqual(report["parity_unqualified_logical_slots"], 8)
    self.assertEqual(len(report["closed_contexts"]), 1)
    self.assertEqual(report["unknown_cost_attempts"], 0)
    self.assertEqual(report["evaluator_wall_seconds"], 2.0 * len(distinct))
    self.assertEqual(report["evaluator_cpu_seconds"], 3.0 * len(distinct))
    self.assertEqual(report["optimizer_updates"], 0)


@mock.patch("rlfw.arena.oracle.time.time", return_value=1791540000.0)
def test_r5_parity_failure_keeps_the_context_and_measured_siblings(self, clock):
    bad = hashlib.sha256(b"leaf31").hexdigest()

    def evaluate(request):
        if request["program"]["program_sha256"] == bad:
            return self.parity_result(request, error=None)
        return self.result(request)

    value = self.scheduler(evaluate=evaluate)
    rows = value.run(max_contexts=1)
    parity = [r for r in rows if r.program_sha256 == bad]
    measured = [r for r in rows if r.program_sha256 != bad]
    self.assertTrue(parity and measured)
    self.assertEqual(
        sum(c["program"]["program_sha256"] == bad for c in self.calls), 1
    )
    self.assertTrue(
        all(r.outcome == "parity_unqualified" and r.q is None for r in parity)
    )
    self.assertTrue(all(r.charged and r.attempts == 1 for r in rows))
    self.assertTrue(all(r.qualified and r.q_grid >= 0 for r in measured))
    report = value.coverage_report()
    self.assertEqual(report["contexts_with_valid_program"], 1)
    self.assertEqual(report["parity_unqualified_logical_slots"], len(parity))
    self.assertEqual(len(report["closed_contexts"]), 1)
    c = next(c for c in self.contexts if c.context_id == rows[0].context_id)
    events = value.ledger.events(c.hwm_key)
    self.assertIn("first_label", {e["event"] for e in events})
    self.assertTrue(
        all(
            e["publication"]["q_grid"] is None
            for e in events
            if e["publication"]["program_sha256"] == bad
        )
    )


@mock.patch("rlfw.arena.oracle.time.time", return_value=1791540000.0)
def test_r5_refuses_legacy_parity_custody_retries(self, clock):
    value = self.scheduler(evaluate=lambda r: self.result(r, "parity_custody"))
    with self.assertRaisesRegex(ValueError, "R5 refuses legacy"):
        value.run(max_contexts=1)
    attempts = value.db.execute("SELECT number,result FROM attempts").fetchall()
    self.assertTrue(attempts)
    self.assertTrue(all(n == 1 and r is None for n, r in attempts))
    self.assertEqual(len(self.calls), len(attempts))
    self.assertEqual(value.rows(), [])
    self.assertTrue(all(not value.ledger.events(c.hwm_key) for c in self.contexts))
    self.assertIsNone(value.coverage_report()["evaluator_wall_seconds"])


@mock.patch("rlfw.arena.oracle.time.time", return_value=1791540000.0)
def test_role_change_after_execution_and_missing_artifacts_cannot_publish(
    self, clock
):
    changed = False

    def role(cid):
        return (
            {**self.role(cid), "role": "development"} if changed else self.role(cid)
        )

    def evaluate(request):
        nonlocal changed
        result = self.result(request)
        changed = True
        return result

    value = self.scheduler(evaluate=evaluate, role=role)
    with self.assertRaises(ValueError):
        value.run(max_contexts=1)
    self.assertEqual(value.rows(), [])
    self.assertTrue(all(not value.ledger.events(c.hwm_key) for c in self.contexts))
    self.assertGreater(value.coverage_report()["unknown_cost_attempts"], 0)
    self.assertIsNone(value.coverage_report()["evaluator_wall_seconds"])


@mock.patch("rlfw.arena.oracle.time.time", return_value=1791540000.0)
def test_numeric_closure_refusal_stays_null_for_all_slots(self, clock):
    self.artifacts = {}
    value = self.scheduler()
    with self.assertRaises(ValueError):
        value.run(max_contexts=1)
    self.assertEqual(value.rows(), [])
    events = [e for c in self.contexts for e in value.ledger.events(c.hwm_key)]
    self.assertEqual(len(events), 8)
    self.assertTrue(all(e["reason"] == "missing_closure" for e in events))
    self.assertTrue(all(e["publication"]["q_grid"] is None for e in events))


@mock.patch("rlfw.arena.oracle.time.time", return_value=1791540000.0)
def test_mixed_packet_restart_reuses_every_raw_draw_without_inference(self, clock):
    # Model dispatches are window-guarded, so the fixture sets its wall clock.
    first = self.scheduler()
    selected = first.slate(self.contexts[0], self.arm, 20261009)
    first.close()
    resumed = self.scheduler()
    resumed.arms.sample_many = lambda *args: self.fail("restart must not decode")
    resumed.arms.score_heads = lambda *args: self.fail(
        "restart must not score heads"
    )
    resumed.arms.probabilities = lambda *args: self.fail(
        "restart must not recalibrate"
    )
    self.assertEqual(resumed.slate(self.contexts[0], self.arm, 20261009), selected)


@mock.patch("rlfw.arena.oracle.time.time", return_value=1791540000.0)
def test_usable_yield_strict_gap_boundary(self, clock):
    values = [0, 10000]

    def evaluate(request):
        return self.result(request, q_grid=values[len(self.calls) % 2])

    value = self.scheduler(evaluate=evaluate)
    value.run(max_contexts=1)
    report = value.coverage_report()
    self.assertEqual(report["contexts_with_valid_program"], 1)
    self.assertEqual(report["contexts_exit_usable"], 0)
    self.assertEqual(
        report["new_parent_families"],
        sorted(
            {value.context_by_id[c].lineage_id for c in value.closed_contexts()}
            - {"parent0"}
        ),
    )


TEST_METHODS = (
    "test_coverage_probabilities_drive_five_exploit_two_explore_one_audit",
    "test_unknown_probability_and_packet_changes_refused",
    "test_busy_arena_blocks_all_proposals_and_evaluations",
    "test_first_labels_measured_only_and_exact_partial_yield_costs",
    "test_g0_and_program_failure_never_become_zero_q",
    "test_exhausted_infra_unknown_attempts_publish_refusals_and_censor",
    "test_r5_parity_unqualified_is_charged_once_without_retry_or_label",
    "test_r5_parity_failure_keeps_the_context_and_measured_siblings",
    "test_r5_refuses_legacy_parity_custody_retries",
    "test_role_change_after_execution_and_missing_artifacts_cannot_publish",
    "test_numeric_closure_refusal_stays_null_for_all_slots",
    "test_mixed_packet_restart_reuses_every_raw_draw_without_inference",
    "test_usable_yield_strict_gap_boundary",
)
