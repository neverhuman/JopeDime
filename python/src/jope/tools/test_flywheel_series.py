import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import flywheel_brake as brake
import flywheel_series as series
from resume_expired_cycle import atomic, sha

EXE_A, EXE_B, EVAL, CORPUS = "a" * 64, "b" * 64, "e" * 64, "c" * 64
HELDOUT, TRAIN = 16, 4
TOOLS = Path(__file__).resolve().parent
B1_IDENTITY = "dense_pairwise_v2+seed_value_scaled_bt_v1"


def per_cut(offset=0.0):
    return [
        dict(
            dataset="ds%d" % i,
            cut=0,
            family="fam",
            n=10,
            wins=10,
            ties=0,
            losses=0,
            mean_log_ratio=-0.02 - 0.001 * i + offset,
        )
        for i in range(HELDOUT)
    ]


def baseline_fixture(exe=EXE_A, panel=None, last_generation=0):
    panel = panel or dict(sequence_nll=1.4, ranking_bt=0.5, ranking_listwise=2.2, runtime_huber=0.07)
    contrast = dict(
        mean_log_ratio=-0.0275,
        ci90=[-0.03, -0.025],
        wins=160,
        ties=0,
        losses=0,
        n_pairs=160,
        n_cuts=HELDOUT,
        per_cut=per_cut(),
    )
    # Shape of flywheel_baseline.build(): executables as digests at the top level, paths under binding.executable_paths.
    return dict(
        schema="parallel.flywheel_baseline.v1",
        final_tests_opened=False,
        frozen_at=1789238332,
        manifest_sha256="m" * 64,
        executables=dict(cpu_evaluator=EVAL, monitor_legacy=exe, training=exe, packing="f" * 64),
        checkpoints=dict(
            learning=dict(
                path="ckpt/serving",
                sha256="s0",
                step=6000,
                ranking_backward="dense_pairwise_v2",
                nll_row_weight="unavailable:nll_row_weight not recorded by this checkpoint format",
            ),
            serving="unavailable:serving checkpoint state missing",
        ),
        metric_definitions_sha256=None,
        generation_range=dict(requested=10, count=1, first=last_generation, last=last_generation),
        binding=dict(
            manifest_sha256="m" * 64,
            output="run",
            executable_paths=dict(
                cpu_evaluator=dict(path="bin/gp", sha256=EVAL),
                monitor_legacy=dict(path="bin/monitor", sha256=exe),
            ),
        ),
        metrics=dict(
            panel_legacy=dict(
                learning=dict(
                    checkpoint="ckpt/serving", checkpoint_sha256="s0", evaluated_step=6000, **panel
                ),
                metric_definitions_sha256=None,
            ),
            panel_null=dict(
                guards=dict(g_nll=0.005, g_bt=0.01),
                guards_borrowed=True,
                guards_derived=None,
                receipts_on_binding=3,
            ),
            matched_heldout=dict(
                n_generations=10,
                seed_only=contrast,
                final=dict(contrast, per_cut=[]),
                tail_caveat="shared tail",
            ),
            matched_train="unavailable:no reported generations",
            gp_improves_on_seed=dict(n=100, improved=90, rate=0.9, mean_relative_gain=0.2),
            diversity=dict(
                unique_neural_per_context_median=40,
                mean_token_length=12.0,
                unique_fresh_programs_median=500,
                unique_fresh_programs=[480, 500, 520],
            ),
        ),
    )


def monitor_receipt(step, nll, bt, exe=EXE_A, checkpoint=None, sha_prefix="c"):
    checkpoint = checkpoint or "ckpt/%s%d" % (sha_prefix, step)
    return dict(
        schema="parallel.frozen_monitoring.v1",
        checkpoint=checkpoint,
        evaluated_step=step,
        timestamp=1789238332 + step,
        sampled_examples=2048,
        family_balanced=True,
        independent_evaluation=False,
        final_tests_opened=False,
        binding=dict(
            checkpoint_sha256=checkpoint.rsplit("/", 1)[1],
            corpus_sha256=CORPUS,
            panel_sha256="p" * 64,
            executable_sha256=exe,
        ),
        sequence_nll=nll,
        ranking_bt=bt,
        ranking_listwise=2.2,
        runtime_huber=0.07,
    )


def seed_value_receipt(checkpoint, accuracy, by_context=None):
    """parallel.seed_value_monitoring.v1: the critic against realized S on the 16 reporting contexts (never packed)."""
    return dict(
        schema="parallel.seed_value_monitoring.v1",
        checkpoint=checkpoint,
        generation=3,
        final_tests_opened=False,
        binding=dict(checkpoint_sha256=checkpoint.rsplit("/", 1)[1]),
        nontie_pairwise_accuracy_vs_S=accuracy,
        ndcg10_vs_S=0.8,
        n_groups=48,
        contexts=16,
        nontie_pairwise_accuracy_by_context=by_context,
    )


def quality_rows(generation, output_prefix):
    rows, jobs = [], []
    for i in range(HELDOUT + TRAIN):
        partition = "meta_validation" if i < HELDOUT else "meta_train"
        for arm, initial in (("neural", 0.9 - 0.001 * i), ("generic", 1.0), ("shuffled", 1.05)):
            output = "%s/%s/ds%d/cut-0/search" % (output_prefix, arm, i)
            seeds = "%s/%s/ds%d/cut-0/seeds.json" % (output_prefix, arm, i)
            jobs.append(
                dict(
                    output=output,
                    seeds=seeds,
                    prepared="ds%d" % i,
                    prepared_sha256="pre%d" % i,
                    cut=0,
                    rng_seed=7 + i,
                    generation=generation,
                )
            )
            rows.append(
                dict(
                    output=output,
                    arm=arm,
                    dataset="ds%d" % i,
                    family="fam",
                    partition=partition,
                    cut=0,
                    rng_seed=7 + i,
                    prepared_sha256="pre%d" % i,
                    evaluator_sha256=EVAL,
                    completed_evaluations=100,
                    best_initial_neural_seed=initial if arm != "generic" else None,
                    best_initial_arm=initial,
                    best_final_gp=initial * 0.8,
                    best_retained_teacher=0.95,
                    best_final_origin="gp_evolution",
                )
            )
    return rows, jobs


def export_records(generation=0):
    """Rows in the shape successor_ledger.export_rows writes (row0_label_verified, verification_reasons; no a_ctx)."""
    records = []
    for index, (arm, role) in enumerate(
        (
            ("neural", "candidate"),
            ("neural", "candidate"),
            ("generic", "control_generic"),
            ("shuffled", "control_shuffled"),
        )
    ):
        records.append(
            dict(
                generation=generation,
                episode_id="ep%d" % index,
                arm=arm,
                role=role,
                r0=1.0,
                child=dict(rmse=0.9 if arm == "neural" else 1.1),
                flip=arm == "neural",
                status="measured",
                completed_at=1789241000,
                evaluator_sha256=EVAL,
                driver_sha256="d" * 64,
                verified=True,
                row0_label_verified=True,
                verification_reasons=[],
            )
        )
    records.append(
        dict(
            generation=generation,
            episode_id="ep4",
            arm="neural",
            role="replicate_panel",
            r0=1.0,
            child=dict(rmse=None, failure="timeout"),
            flip=False,
            status="deadline",
            completed_at=1789241001,
            evaluator_sha256=EVAL,
            driver_sha256="d" * 64,
            verified=False,
            row0_label_verified=None,
            verification_reasons=["row0_label_unverifiable: seed row unmeasured"],
        )
    )
    return records


def policy_block(generation=0, corpus=CORPUS):
    """policy['seed_values'] as successor_targets.build writes it: a_own_ci is a {low, high, resamples} dict, a_ctx per arm."""
    return dict(
        schema="parallel.seed_values.v1",
        generation=generation,
        corpus_sha256=corpus,
        info_test=dict(
            p_flip=0.2,
            a_own_ci=dict(low=0.05, high=0.3, resamples=2000),
            passed=True,
            a_own=dict(
                coefficient=0.4,
                ci90=dict(low=0.05, high=0.3, resamples=2000),
                n_rows=24,
                n_contexts=2,
                reason=None,
            ),
        ),
        counts=dict(groups=4, refused_contexts=1, tie_fraction_median=0.3),
        floors={"0": 0.002, "1": 0.004},
        a_ctx=dict(
            control_role="control_generic",
            by_arm=dict(neural=0.05, generic=0.0, shuffled=0.0),
            contexts_with_control=1,
            contexts_without_control=[],
        ),
    )


def build_run(
    root,
    monitors=(),
    generations=(0,),
    episodes=True,
    blocks=True,
    identities=None,
    seed_value_monitors=(),
    replay_policy=True,
):
    root = Path(root)
    manifest = dict(
        schema="parallel.continuous_learning.v1",
        output="run",
        monitor_executable="bin/monitor",
        cpu_executable="bin/gp",
        bindings={"bin/monitor": EXE_A, "bin/gp": EVAL},
        contexts_sha256="x",
        vocabulary_sha256="v",
    )
    atomic(root / "manifest.json", manifest)
    atomic(root / "baseline.json", baseline_fixture())
    run = root / "run"
    (run / "monitoring").mkdir(parents=True)
    candidates = []
    for receipt in monitors:
        name = "policy-" + receipt["binding"]["checkpoint_sha256"]
        atomic(run / "monitoring" / (name + ".json"), receipt)
        candidates.append(
            dict(checkpoint=receipt["checkpoint"], checkpoint_sha256=receipt["binding"]["checkpoint_sha256"])
        )
        (root / receipt["checkpoint"]).mkdir(parents=True, exist_ok=True)
        identity = (identities or {}).get(receipt["checkpoint"], "dense_pairwise_v2")
        atomic(
            root / receipt["checkpoint"] / "state.json",
            dict(ranking_backward=identity, step=receipt["evaluated_step"]),
        )
    for receipt in seed_value_monitors:
        atomic(run / "monitoring" / ("seed-value-%s.json" % receipt["binding"]["checkpoint_sha256"]), receipt)
    (root / "ckpt/serving").mkdir(parents=True, exist_ok=True)
    atomic(root / "ckpt/serving/state.json", dict(ranking_backward="dense_pairwise_v2", step=6000))
    candidates.sort(key=lambda c: c["checkpoint"])
    atomic(
        run / "state.json",
        dict(
            manifest_sha256=sha(root / "manifest.json"),
            serving_checkpoint="ckpt/serving",
            serving_checkpoint_sha256="s0",
            candidates=candidates,
            learning_checkpoint=candidates[-1]["checkpoint"] if candidates else "ckpt/serving",
        ),
    )
    for number in generations:
        directory = run / ("generation-%06d" % number)
        directory.mkdir(parents=True)
        rows, jobs = quality_rows(number, "run/generation-%06d" % number)
        for job in jobs:
            path = root / job["seeds"]
            path.parent.mkdir(parents=True, exist_ok=True)
            atomic(
                path,
                dict(
                    schema="parallel.seed_population.v1",
                    candidates=[
                        dict(genome="(pipe X (ridge a%d))" % k, origin="neural_policy") for k in range(40)
                    ],
                ),
            )
        atomic(directory / "jobs.json", dict(schema="parallel.gp_jobs.v1", jobs=jobs))
        atomic(
            directory / "generation.json",
            dict(number=number, checkpoint="ckpt/serving", checkpoint_sha256="s0", corpus="corpus"),
        )
        atomic(
            directory / "quality-report.json",
            dict(
                schema="parallel.continuous_quality.v1",
                timestamp=1789240000 + number,
                jobs_sha256=sha(directory / "jobs.json"),
                rows=rows,
                final_tests_opened=False,
            ),
        )
    if blocks:
        block = run / "cycle-000001/train-2"
        block.mkdir(parents=True)
        atomic(
            block / "complete.json",
            dict(
                schema="parallel.training_complete.v1",
                final_tests_opened=False,
                corpus_sha256=CORPUS,
                latest=dict(checkpoint="ckpt/c9000", state_sha256="c9000"),
                steps=3000,
                seed_values_absent="seed_values_absent: no export for this corpus",
            ),
        )
    if replay_policy:
        # The deployed controller stores replay policies under replay-membership/<identity>/policy.json only.
        for name, fresh, stamp in (
            ("older", {"g0": [7, 8]}, 1789238000),
            ("abc", {"g1": [1, 2, 3], "g2": [4, 5]}, 1789239000),
        ):
            path = run / "replay-membership" / name / "policy.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            atomic(path, dict(schema="parallel.replay_policy.v2", fresh_records=fresh))
            os.utime(path, (stamp, stamp))
    if episodes:
        successor = run / "successor"
        (successor / "economics").mkdir(parents=True)
        (successor / "seed-values").mkdir()
        atomic(successor / "seed-values-0.json", dict(schema="parallel.seed_value.v1", rows=export_records()))
        atomic(
            successor / "economics/generation-0.json",
            dict(
                tier="A",
                reserved_core_seconds=3.6,
                attempted_fits=10,
                successful_fits=9,
                unique_fits_per_core_second=2.5,
                timeouts=1,
            ),
        )
        atomic(successor / "seed-values" / (CORPUS + ".json"), policy_block())
    return root


def monitor_row(step, nll, bt=0.5, comparable=True, lineage="learning", identifier=None):
    delta = (
        dict(sequence_nll=nll - 1.4, ranking_bt=bt - 0.5, ranking_listwise=0.0, runtime_huber=0.0)
        if comparable
        else "incomparable"
    )
    return dict(
        schema=series.SCHEMA,
        kind="monitor",
        id=identifier or "m%d" % step,
        timestamp=1789238332 + step,
        evaluated_step=step,
        checkpoint="ckpt/c%d" % step,
        checkpoint_sha256="c%d" % step,
        lineage=lineage,
        segment={},
        panel=dict(
            sequence_nll=nll, ranking_bt=bt, ranking_listwise=2.2, runtime_huber=0.07, baseline_delta=delta
        ),
        final_tests_opened=False,
    )


def baseline_digest():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "baseline.json"
        atomic(path, baseline_fixture())
        return series.baseline_receipt(path)


class SeriesProducer(unittest.TestCase):
    def test_idempotent_append_and_latest_rewrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = build_run(directory, monitors=[monitor_receipt(9000, 1.45, 0.5)])
            series_path, latest = root / "run/flywheel-series.jsonl", root / "runs/flywheel-latest.json"
            fresh = series.produce(root, "manifest.json", "baseline.json", series_path, latest)
            kinds = sorted(r["kind"] for r in fresh)
            self.assertEqual(kinds, ["block", "episodes", "generation", "monitor"])
            first = series_path.read_bytes()
            self.assertEqual(series.produce(root, "manifest.json", "baseline.json", series_path, latest), [])
            self.assertEqual(series_path.read_bytes(), first)
            for row in series.series_rows(series_path):
                self.assertEqual(row["schema"], series.SCHEMA)
                self.assertIs(row["final_tests_opened"], False)
            published = json.loads(latest.read_text())
            self.assertEqual(published["schema"], series.LATEST_SCHEMA)
            self.assertEqual(published["manifest_sha256"], sha(root / "manifest.json"))
            self.assertEqual(published["series_rows"], 4)
            self.assertEqual(published["baseline"]["segment"]["monitor_executable_sha256"], EXE_A)
            self.assertIsNone(published["baseline"]["segment"]["nll_row_weight"])
            self.assertEqual(
                published["baseline"]["guards"],
                dict(g_nll=0.005, g_bt=0.01, borrowed=True, receipts_on_binding=3),
            )
            self.assertNotIn("per_cut", json.dumps(published["baseline"]["matched_heldout"]))
            self.assertIsNone(published["brake"])
            self.assertIsNone(published["verdict"])
            self.assertIs(published["final_tests_opened"], False)

    def test_segment_break_makes_baseline_delta_incomparable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = build_run(
                directory,
                monitors=[monitor_receipt(9000, 1.45, 0.5), monitor_receipt(12000, 1.45, 0.5, exe=EXE_B)],
                generations=(),
                episodes=False,
                blocks=False,
            )
            rows = series.monitor_rows(
                root,
                root / "run",
                series.read(root / "manifest.json"),
                series.baseline_receipt(root / "baseline.json"),
            )
            by_step = {r["evaluated_step"]: r for r in rows}
            self.assertAlmostEqual(by_step[9000]["panel"]["baseline_delta"]["sequence_nll"], 0.05)
            self.assertEqual(by_step[9000]["lineage"], "learning")
            self.assertEqual(by_step[12000]["panel"]["baseline_delta"], "incomparable")
            self.assertEqual(by_step[12000]["segment"]["monitor_executable_sha256"], EXE_B)
            self.assertNotEqual(by_step[9000]["segment"], by_step[12000]["segment"])

    def test_objective_identity_change_stays_comparable_and_can_trip_f2(self):
        # PREREGISTRATION F2: rejects are counted from the first monitored block after a new objective identity, against
        # the same monitor definitions. The identity is displayed in the segment but never makes a checkpoint incomparable.
        monitors = [monitor_receipt(9000, 1.4, 0.5)] + [
            monitor_receipt(step, 1.5, 0.5) for step in (12000, 15000, 18000)
        ]
        identities = {"ckpt/c%d" % step: B1_IDENTITY for step in (12000, 15000, 18000)}
        with tempfile.TemporaryDirectory() as directory:
            root = build_run(
                directory,
                monitors=monitors,
                generations=(),
                episodes=False,
                blocks=False,
                identities=identities,
            )
            baseline = series.baseline_receipt(root / "baseline.json")
            rows = series.monitor_rows(root, root / "run", series.read(root / "manifest.json"), baseline)
            by_step = {r["evaluated_step"]: r for r in rows}
            self.assertEqual(by_step[12000]["segment"]["ranking_backward"], B1_IDENTITY)
            self.assertNotEqual(
                by_step[12000]["segment"]["ranking_backward"], baseline["segment"]["ranking_backward"]
            )
            self.assertAlmostEqual(by_step[12000]["panel"]["baseline_delta"]["sequence_nll"], 0.1)
            self.assertEqual(
                series.panel_segment(by_step[12000]["segment"]), series.panel_segment(baseline["segment"])
            )
            self.assertNotIn("ranking_backward", series.PANEL_SEGMENT_KEYS)
            decision = brake.decide(rows, baseline, None, "m", 1)
            self.assertTrue(decision["held"])
            self.assertEqual(decision["tripped_by"], ["F2"])
            self.assertEqual(decision["windows"]["F2"], [["policy-c12000", "policy-c15000", "policy-c18000"]])

    def test_monitor_rows_follow_evaluation_order_not_file_names(self):
        # monitoring/<policy>-<checkpoint sha>.json: lexical order of the shas (z9000 > a12000) is not step order.
        monitors = [
            monitor_receipt(9000, 1.4, 0.5, sha_prefix="z"),
            monitor_receipt(30000, 1.4, 0.5, sha_prefix="b"),
            monitor_receipt(12000, 1.4, 0.5, sha_prefix="a"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = build_run(directory, monitors=monitors, generations=(), episodes=False, blocks=False)
            rows = series.monitor_rows(
                root,
                root / "run",
                series.read(root / "manifest.json"),
                series.baseline_receipt(root / "baseline.json"),
            )
            self.assertEqual([r["evaluated_step"] for r in rows], [9000, 12000, 30000])
            self.assertEqual([r["id"] for r in rows], ["policy-z9000", "policy-a12000", "policy-b30000"])
            self.assertEqual(
                sorted(r["id"] for r in rows), ["policy-a12000", "policy-b30000", "policy-z9000"]
            )  # the file order

    def test_seed_value_monitoring_receipt_joins_its_checkpoint_row(self):
        monitors = [monitor_receipt(9000, 1.4, 0.5), monitor_receipt(12000, 1.4, 0.5)]
        receipts = [
            seed_value_receipt("ckpt/c12000", 0.58, by_context=[0.55 + 0.005 * i for i in range(16)]),
            seed_value_receipt("ckpt/orphan", 0.9),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = build_run(
                directory,
                monitors=monitors,
                generations=(),
                episodes=False,
                blocks=False,
                seed_value_monitors=receipts,
            )
            rows = series.monitor_rows(
                root,
                root / "run",
                series.read(root / "manifest.json"),
                series.baseline_receipt(root / "baseline.json"),
            )
            self.assertEqual(
                [r["kind"] for r in rows], ["monitor", "monitor"]
            )  # a seed-value receipt is never its own row
            by_step = {r["evaluated_step"]: r for r in rows}
            self.assertNotIn("seed_value_monitoring", by_step[9000]["panel"])
            joined = by_step[12000]["panel"]["seed_value_monitoring"]
            self.assertEqual(
                (joined["nontie_pairwise_accuracy_vs_S"], joined["contexts"], len(joined["by_context"])),
                (0.58, 16, 16),
            )
            self.assertEqual(joined["receipt_sha256"], sha(root / "run/monitoring/seed-value-c12000.json"))

    def test_matched_contrast_ci_on_sixteen_heldout_cuts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = build_run(directory, episodes=False, blocks=False)
            rows = series.generation_rows(
                root,
                root / "run",
                series.read(root / "manifest.json"),
                series.baseline_receipt(root / "baseline.json"),
            )
            self.assertEqual(len(rows), 1)
            row = rows[0]
            seed_only = row["matched"]["val"]["seed_only"]
            self.assertEqual(seed_only["n_cuts"], HELDOUT)
            self.assertEqual((seed_only["wins"], seed_only["ties"], seed_only["losses"]), (HELDOUT, 0, 0))
            self.assertEqual(len(seed_only["ci90"]), 2)
            self.assertLess(seed_only["ci90"][1], 0)
            self.assertNotIn("per_cut", seed_only)
            self.assertEqual(len(row["matched"]["val"]["trailing10"]["per_cut"]), HELDOUT)
            self.assertEqual(row["matched"]["val"]["trailing10"]["generations"], [0])
            self.assertEqual(
                row["matched"]["val"]["final"]["tail_caveat"], series.flywheel_metrics.TAIL_CAVEAT
            )
            self.assertEqual(row["matched"]["train"]["seed_only"]["n_cuts"], TRAIN)
            self.assertEqual(row["gp_improves_on_seed"]["neural"]["rate"], 1.0)
            self.assertEqual(row["best_final_origin"], {"gp_evolution": 3 * (HELDOUT + TRAIN)})
            self.assertEqual(row["diversity"]["unique_neural_per_context_median"], 40)
            # Replay snapshots come from replay-membership/*/policy.json in mtime order: 2 then 5 unique fresh programs.
            self.assertEqual(row["snapshots"], ["older", "abc"])
            self.assertEqual(row["diversity"]["unique_fresh_programs"], [2, 5])
            self.assertTrue(row["baseline_comparable"])
            self.assertEqual(row["codegen"], "pending")
            self.assertIn("discoveries_per_hour", row["unavailable"])

    def test_best_initial_arm_is_derived_from_each_arms_ledger_and_cached(self):
        with tempfile.TemporaryDirectory() as directory:
            root = build_run(directory)
            report_path = root / "run/generation-000000/quality-report.json"
            report = series.read(report_path)
            initial = {}
            for row in report["rows"]:
                initial[row["output"]] = row.pop(
                    "best_initial_arm"
                )  # live reports predate the producer of this key
                ledger = root / row["output"] / "observations.json"
                ledger.parent.mkdir(parents=True, exist_ok=True)
                # slots 0..63 carry the initial population (one failed slot, one worse slot); slot 70 is a later GP fit and must not count
                atomic(
                    ledger,
                    [
                        dict(
                            initial_seed_slot=0,
                            failure=None,
                            metrics=dict(rmse=initial[row["output"]] * 1.5, units="original_target"),
                        ),
                        dict(
                            initial_seed_slot=1,
                            failure=None,
                            metrics=dict(rmse=initial[row["output"]], units="original_target"),
                        ),
                        dict(initial_seed_slot=2, failure="timeout", metrics=None),
                        dict(
                            initial_seed_slot=70,
                            failure=None,
                            metrics=dict(rmse=initial[row["output"]] * 0.1, units="original_target"),
                        ),
                    ],
                )
            atomic(report_path, report)
            manifest, baseline = (
                series.read(root / "manifest.json"),
                series.baseline_receipt(root / "baseline.json"),
            )
            cache = root / "run/successor/scoreboard/derived"
            plain = series.generation_rows(root, root / "run", manifest, baseline)[0]
            self.assertIn("missing_values", json.dumps(plain["matched"]["train"]["seed_only"]))
            self.assertEqual(
                plain["best_initial_arm_derived"]["counts"], {"observations_ledger": 3 * (HELDOUT + TRAIN)}
            )  # derived even without a cache dir
            row = series.generation_rows(root, root / "run", manifest, baseline, cache_dir=cache)[0]
            train = row["matched"]["train"]["seed_only"]
            self.assertEqual(
                (train["wins"] + train["ties"] + train["losses"], train["n_pairs"]), (TRAIN, TRAIN)
            )
            self.assertEqual(
                row["best_initial_arm_derived"]["counts"], {"observations_ledger": 3 * (HELDOUT + TRAIN)}
            )
            self.assertIn("never imputed", row["best_initial_arm_derived"]["rule"])
            self.assertEqual(row["matched"]["val"]["seed_only"]["n_pairs"], HELDOUT)
            cached = series.read(cache / "initial-arm-generation-000000.json")
            self.assertEqual(
                (cached["schema"], cached["report_sha256"], len(cached["values"])),
                (series.DERIVED_SCHEMA, sha(report_path), 3 * (HELDOUT + TRAIN)),
            )
            self.assertEqual(
                cached["values"][report["rows"][0]["output"]]["value"], initial[report["rows"][0]["output"]]
            )
            # the cache is reused as long as the report sha matches, and ignored when it does not
            for ledger in root.glob("run/generation-000000/*/*/cut-0/search/observations.json"):
                ledger.unlink()
            again = series.generation_rows(root, root / "run", manifest, baseline, cache_dir=cache)[0]
            self.assertEqual(again["matched"]["train"]["seed_only"]["n_pairs"], TRAIN)
            stale = dict(cached, report_sha256="0" * 64)
            atomic(cache / "initial-arm-generation-000000.json", stale)
            without = series.generation_rows(root, root / "run", manifest, baseline, cache_dir=cache)[0]
            self.assertEqual(
                without["best_initial_arm_derived"]["counts"], {"ledger_missing": 3 * (HELDOUT + TRAIN)}
            )

    def test_generation_without_replay_snapshots_says_so(self):
        with tempfile.TemporaryDirectory() as directory:
            root = build_run(directory, episodes=False, blocks=False, replay_policy=False)
            (root / "run/cycle-000001/train-2").mkdir(parents=True)
            atomic(
                root / "run/cycle-000001/train-2/policy.json",
                dict(schema="parallel.replay_policy.v2", fresh_records={"g": [1]}),
            )  # never read
            row = series.generation_rows(
                root,
                root / "run",
                series.read(root / "manifest.json"),
                series.baseline_receipt(root / "baseline.json"),
            )[0]
            self.assertEqual(row["snapshots"], [])
            self.assertEqual(row["diversity"]["unique_fresh_programs"], [])
            f3 = brake.evaluate_f3([row], baseline_digest())
            self.assertEqual(f3["fresh_unavailable"], 1)
            self.assertEqual(f3["decisions"][0]["values"]["unique_fresh_programs"], brake.FRESH_UNAVAILABLE)
            self.assertEqual(f3["decisions"][0]["decision"], "accepted")

    def test_episode_and_block_rows_carry_counts_not_zeros(self):
        with tempfile.TemporaryDirectory() as directory:
            root = build_run(directory)
            manifest = series.read(root / "manifest.json")
            episodes = series.episode_rows(root, manifest, root / "run/successor")
            self.assertEqual(len(episodes), 1)
            successor = episodes[0]["successor"]
            self.assertEqual(
                successor["roles"],
                dict(candidate=2, control_generic=1, control_shuffled=1, replicate_panel=1),
            )
            self.assertEqual((successor["completed"], successor["deadline"]), (4, 1))
            self.assertEqual(
                successor["arms"]["neural"],
                dict(
                    n=3, flips=2, better=2, tied=0, worse=0, failed=1, superseded=0, child_beats_incumbent=0
                ),
            )
            self.assertEqual(successor["p_flip"], 0.2)
            self.assertEqual(
                successor["a_own_ci"], [0.05, 0.3]
            )  # normalized from the producer's {low, high, resamples}
            self.assertEqual(successor["a_own_ci_resamples"], 2000)
            self.assertEqual(
                successor["a_ctx"], dict(neural=0.05, generic=0.0, shuffled=0.0)
            )  # from the policy block, not export rows
            self.assertEqual(successor["a_ctx_contexts_with_control"], 1)
            self.assertEqual(successor["eps_median"], 0.003)
            self.assertEqual((successor["row0_mismatches"], successor["row0_verified_rows"]), (0, 4))
            self.assertEqual(successor["fits_per_core_second"], 2.5)
            self.assertEqual(episodes[0]["segment"]["driver_sha256"], "d" * 64)
            self.assertNotIn("a_ctx", episodes[0]["unavailable"])
            blocks = series.block_rows(root / "run")
            self.assertEqual(len(blocks), 1)
            self.assertEqual(blocks[0]["block_index"], 1)
            self.assertEqual(blocks[0]["seed_values"]["consumed"], "no")
            self.assertIn("seed_values_absent", blocks[0]["seed_values"]["reason"])
            self.assertIsNone(blocks[0]["seed_value_bt"])
            (root / "run/successor/seed-values" / (CORPUS + ".json")).unlink()
            without_block = series.episode_rows(root, manifest, root / "run/successor")[0]
            self.assertEqual(
                without_block["successor"]["a_ctx"], dict(neural=None, generic=None, shuffled=None)
            )
            self.assertEqual(without_block["unavailable"]["a_ctx"], "seed_values policy block absent")
            self.assertIsNone(without_block["successor"]["a_own_ci"])

    def test_row0_mismatches_read_the_ledger_shape_and_trip_f5(self):
        with tempfile.TemporaryDirectory() as directory:
            root = build_run(directory)
            successor = root / "run/successor"
            records = export_records()
            records[1].update(
                verified=False,
                row0_label_verified=False,
                verification_reasons=["row0_label_mismatch: 1.0000000001 != admitted 1.0"],
            )
            atomic(successor / "seed-values-0.json", dict(schema="parallel.seed_value.v1", rows=records))
            # admitted-sources.json as successor_ledger.admitted_sources writes it: reasons nested under excluded[].unverified[].
            atomic(
                successor / "admitted-sources.json",
                dict(
                    schema="parallel.successor_sources.v1",
                    sources=[],
                    row0_mismatches=2,
                    excluded=[
                        dict(
                            generation=0,
                            context_key="p:0",
                            verified=3,
                            reasons=["row0_label_mismatch"],
                            reason="2 unverified episode(s): row0_label_mismatch",
                            unverified=[
                                dict(
                                    episode_id="ep1",
                                    reasons=["row0_label_mismatch: 1.0000000001 != admitted 1.0"],
                                    row0_label_verified=False,
                                ),
                                dict(
                                    episode_id="ep9",
                                    reasons=["row0_label_mismatch: 0.5 != admitted 0.4"],
                                    row0_label_verified=False,
                                ),
                                dict(
                                    episode_id="ep7",
                                    reasons=[
                                        "row0_label_unverified: no admitted seed RMSE supplied for parent x"
                                    ],
                                    row0_label_verified=None,
                                ),
                            ],
                        )
                    ],
                ),
            )
            self.assertEqual(
                series.row0_mismatches(records, successor), 2
            )  # ep1 counted once, ep9 once, ep7 is not a mismatch
            rows = series.episode_rows(root, series.read(root / "manifest.json"), successor)
            self.assertEqual(rows[0]["successor"]["row0_mismatches"], 2)
            decision = brake.decide(rows, baseline_digest(), None, "m", 1)
            self.assertEqual(decision["tripped_by"], ["F5"])
            self.assertTrue(decision["held"])
            self.assertEqual(decision["rejects"][0]["row0_mismatches"], 2)
            (successor / "admitted-sources.json").unlink()
            self.assertEqual(series.row0_mismatches(records, successor), 1)
            self.assertEqual(series.row0_mismatches(export_records(), successor), 0)

    def test_cli_runs_and_reports_appended_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = build_run(directory, monitors=[monitor_receipt(9000, 1.45, 0.5)])
            result = subprocess.run(
                [
                    sys.executable,
                    str(TOOLS / "flywheel_series.py"),
                    "--root",
                    str(root),
                    "--manifest",
                    "manifest.json",
                    "--baseline",
                    "baseline.json",
                ],
                text=True,
                capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            summary = json.loads(result.stdout)
            self.assertEqual(summary["appended"], 4)
            self.assertTrue((root / "run/flywheel-series.jsonl").is_file())
            self.assertTrue((root / "runs/flywheel-latest.json").is_file())

    def test_foreign_rows_and_missing_baseline_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "series.jsonl"
            path.write_text(json.dumps(dict(schema="other", kind="monitor", id="x")) + "\n")
            with self.assertRaises(RuntimeError):
                series.series_rows(path)
            bad = Path(directory) / "baseline.json"
            atomic(bad, dict(schema="parallel.flywheel_baseline.v1", final_tests_opened=True))
            with self.assertRaises(RuntimeError):
                series.baseline_receipt(bad)

    def test_ci_bounds_accepts_both_interval_shapes(self):
        self.assertEqual(series.ci_bounds(dict(low=0.05, high=0.3, resamples=2000)), [0.05, 0.3])
        self.assertEqual(series.ci_bounds([0.05, 0.3]), [0.05, 0.3])
        self.assertEqual(series.ci_bounds((0.05, 0.3)), [0.05, 0.3])
        for value in (
            None,
            dict(low=None, high=0.3),
            [0.05],
            [float("nan"), 0.3],
            "unavailable",
            dict(resamples=2000),
        ):
            self.assertIsNone(series.ci_bounds(value), value)


class Brake(unittest.TestCase):
    def setUp(self):
        self.baseline = baseline_digest()

    def test_f2_trips_only_after_three_consecutive_rejects(self):
        reject = lambda step: monitor_row(step, 1.5)  # 1.5 > 1.4 * 1.005
        accept = lambda step: monitor_row(step, 1.4)
        two = [reject(9000), reject(12000)]
        decision = brake.decide(two, self.baseline, None, "m", 1)
        self.assertFalse(decision["held"])
        self.assertEqual(decision["consecutive"]["F2"], 2)
        three = two + [reject(15000)]
        decision = brake.decide(three, self.baseline, None, "m", 1)
        self.assertTrue(decision["held"])
        self.assertEqual(decision["tripped_by"], ["F2"])
        self.assertTrue(decision["reason"].startswith("F2 frozen panel: 3 consecutive"))
        self.assertIn("m9000,m12000,m15000", decision["reason"])
        self.assertEqual(decision["fires"]["F2"], 1)
        self.assertEqual(decision["windows"]["F2"], [["m9000", "m12000", "m15000"]])
        broken = [reject(9000), reject(12000), accept(15000), reject(18000), reject(21000)]
        self.assertFalse(brake.decide(broken, self.baseline, None, "m", 1)["held"])
        exact = brake.evaluate_f2(
            [reject(9000), accept(12000), monitor_row(15000, 1.4 * 1.005)], self.baseline
        )
        self.assertEqual([d["decision"] for d in exact["decisions"]], ["rejected", "accepted", "accepted"])
        self.assertEqual(
            brake.evaluate_f2([monitor_row(9000, 1.4, bt=0.5 * 1.01 + 1e-9)], self.baseline)["rejects"][0][
                "exceeded"
            ],
            ["ranking_bt"],
        )

    def test_f2_latches_on_evidence_not_on_invocation_timing(self):
        reject = lambda step: monitor_row(step, 1.5)
        accept = lambda step: monitor_row(step, 1.4)
        # [R, R, R, A] evaluated once, after the accept: the window still exists and must hold.
        late = brake.decide(
            [reject(9000), reject(12000), reject(15000), accept(18000)], self.baseline, None, "m", 1
        )
        self.assertTrue(late["held"])
        self.assertEqual((late["tripped_by"], late["fires"]["F2"], late["consecutive"]["F2"]), (["F2"], 1, 0))
        self.assertEqual(late["windows"]["F2"], [["m9000", "m12000", "m15000"]])
        # Rejects separated by an incomparable row (another monitor definition) are still consecutive decisions.
        split = [reject(9000), monitor_row(12000, 1.5, comparable=False), reject(15000), reject(18000)]
        decision = brake.decide(split, self.baseline, None, "m", 1)
        self.assertTrue(decision["held"])
        self.assertEqual(decision["falsifiers"]["F2"]["skipped_incomparable"], 1)
        self.assertEqual(decision["windows"]["F2"], [["m9000", "m15000", "m18000"]])
        # Release acknowledges exactly the window's rejects; the accept that followed never hides them again.
        released = brake.release(late, "operator: reviewed the three rejects", 2)
        self.assertEqual(released["acknowledged_ids"], ["m12000", "m15000", "m9000"])
        after = brake.decide(
            [reject(9000), reject(12000), reject(15000), accept(18000)], self.baseline, released, "m", 3
        )
        self.assertFalse(after["held"])
        self.assertEqual(after["fires"]["F2"], 0)

    def test_serving_duplicate_and_incomparable_rows_do_not_count(self):
        rows = [
            monitor_row(9000, 1.5, lineage="serving"),
            monitor_row(12000, 1.5, comparable=False),
            monitor_row(15000, 1.5, identifier="old"),
            monitor_row(15000, 1.4, identifier="new"),
        ]
        rows[-1]["timestamp"] += 1
        result = brake.evaluate_f2(rows, self.baseline)
        self.assertEqual(result["skipped_incomparable"], 1)
        self.assertEqual([d["id"] for d in result["decisions"]], ["new"])
        self.assertFalse(result["tripped"])
        self.assertFalse(brake.evaluate_f2([], self.baseline)["evaluable"])
        no_guards = dict(self.baseline, guards=dict(g_nll=None, g_bt=None))
        self.assertIn("guards", brake.evaluate_f2(rows, no_guards)["reason"])

    def test_f5_trips_immediately_and_pauses_once(self):
        row = dict(
            schema=series.SCHEMA,
            kind="episodes",
            id="seed-values-3",
            generation=3,
            successor=dict(row0_mismatches=1),
            final_tests_opened=False,
        )
        decision = brake.decide([row], self.baseline, None, "m", 1)
        self.assertTrue(decision["held"])
        self.assertEqual(decision["tripped_by"], ["F5"])
        self.assertIn("row-0 RMSE mismatch", decision["reason"])
        dry = brake.pause(None, decision, None)
        self.assertEqual((dry["due"], dry["invoked"], dry["dry_run"]), (True, False, True))
        invoked = brake.pause('%s -c "print(42)"' % sys.executable, decision, None)
        self.assertEqual(
            (invoked["invoked"], invoked["returncode"], invoked["stdout"].strip()), (True, 0, "42")
        )
        again = brake.pause('%s -c "print(43)"' % sys.executable, decision, dict(decision, pause=invoked))
        self.assertEqual((again["invoked"], again["repeated"], again["stdout"].strip()), (True, False, "42"))
        clean_row = dict(row, successor=dict(row0_mismatches=0))
        self.assertFalse(brake.decide([clean_row], self.baseline, None, "m", 1)["held"])

    def test_f3_three_consecutive_generations_hold_without_pausing(self):
        def generation(number, unique):
            return dict(
                schema=series.SCHEMA,
                kind="generation",
                id="generation-%06d" % number,
                generation=number,
                diversity=dict(
                    unique_neural_per_context_median=unique,
                    mean_token_length=12.0,
                    unique_fresh_programs=[500],
                    unique_fresh_programs_median=500,
                ),
                final_tests_opened=False,
            )

        rows = [generation(n, 10) for n in range(3)]
        decision = brake.decide(rows, self.baseline, None, "m", 1)
        self.assertTrue(decision["held"])
        self.assertEqual(decision["tripped_by"], ["F3"])
        self.assertFalse(brake.pause("true", decision, None)["due"])
        self.assertFalse(brake.decide(rows[:2], self.baseline, None, "m", 1)["held"])
        self.assertTrue(
            brake.decide(rows + [generation(3, 40)], self.baseline, None, "m", 1)["held"]
        )  # evidence latch
        tokens = [
            dict(
                generation(n, 40),
                diversity=dict(
                    unique_neural_per_context_median=40,
                    mean_token_length=20.0,
                    unique_fresh_programs=[500],
                    unique_fresh_programs_median=500,
                ),
            )
            for n in range(3)
        ]
        self.assertTrue(brake.evaluate_f3(tokens, self.baseline)["tripped"])
        fresh = [
            dict(
                generation(n, 40),
                diversity=dict(
                    unique_neural_per_context_median=40,
                    mean_token_length=12.0,
                    unique_fresh_programs=[100],
                    unique_fresh_programs_median=500,
                ),
            )
            for n in range(3)
        ]
        self.assertTrue(brake.evaluate_f3(fresh, self.baseline)["tripped"])
        blind = [
            dict(
                generation(n, 40),
                diversity=dict(
                    unique_neural_per_context_median=40,
                    mean_token_length=12.0,
                    unique_fresh_programs=[],
                    unique_fresh_programs_median=None,
                ),
            )
            for n in range(3)
        ]
        result = brake.evaluate_f3(blind, self.baseline)
        self.assertFalse(result["tripped"])
        self.assertEqual(result["fresh_unavailable"], 3)
        self.assertTrue(
            all(d["values"]["unique_fresh_programs"] == brake.FRESH_UNAVAILABLE for d in result["decisions"])
        )

    def test_hold_latches_until_operator_release(self):
        rows = [monitor_row(s, 1.5) for s in (9000, 12000, 15000)]
        held = brake.decide(rows, self.baseline, None, "m", 1)
        held["pause"] = brake.pause(None, held, None)
        latched = brake.decide([monitor_row(9000, 1.4)], self.baseline, held, "m", 2)
        self.assertTrue(latched["held"])
        self.assertTrue(latched["latched_from_previous"])
        self.assertEqual(latched["hold_id"], held["hold_id"])
        self.assertFalse(brake.pause(None, latched, held)["due"])
        with self.assertRaises(RuntimeError):
            brake.release(held, "", 3)
        released = brake.release(held, "operator: rebuilt monitor verified", 3)
        self.assertFalse(released["held"])
        self.assertEqual(sorted(released["acknowledged_ids"]), ["m12000", "m15000", "m9000"])
        after = brake.decide(rows, self.baseline, released, "m", 4)
        self.assertFalse(after["held"])
        self.assertEqual(after["released"]["note"], "operator: rebuilt monitor verified")
        again = brake.decide(
            rows + [monitor_row(s, 1.5) for s in (18000, 21000, 24000)], self.baseline, released, "m", 5
        )
        self.assertTrue(again["held"])

    def test_cli_writes_brake_json_in_the_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = build_run(
                directory,
                monitors=[monitor_receipt(s, 1.5, 0.5) for s in (9000, 12000, 15000)],
                generations=(),
                episodes=False,
                blocks=False,
            )
            series.produce(
                root,
                "manifest.json",
                "baseline.json",
                root / "run/flywheel-series.jsonl",
                root / "runs/flywheel-latest.json",
            )
            command = [
                sys.executable,
                str(TOOLS / "flywheel_brake.py"),
                "--root",
                str(root),
                "--manifest",
                "manifest.json",
                "--baseline",
                "baseline.json",
            ]
            result = subprocess.run(command, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            summary = json.loads(result.stdout)
            self.assertTrue(summary["held"])
            written = json.loads((root / "run/brake.json").read_text())
            self.assertEqual(written["schema"], brake.SCHEMA)
            self.assertEqual(written["tripped_by"], ["F2"])
            self.assertEqual(written["manifest_sha256"], sha(root / "manifest.json"))
            self.assertTrue(written["pause"]["dry_run"])
            self.assertIs(written["final_tests_opened"], False)
            appended = series.produce(
                root,
                "manifest.json",
                "baseline.json",
                root / "run/flywheel-series.jsonl",
                root / "runs/flywheel-latest.json",
            )
            self.assertEqual([r["kind"] for r in appended], ["brake"])
            self.assertTrue(json.loads((root / "runs/flywheel-latest.json").read_text())["brake"]["held"])
            result = subprocess.run(command + ["--release", "operator note"], text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(json.loads((root / "run/brake.json").read_text())["held"])


if __name__ == "__main__":
    unittest.main()
