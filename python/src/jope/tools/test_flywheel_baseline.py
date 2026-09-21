import copy
import hashlib
import json
import os
import re
import sqlite3
import tempfile
import unittest
from pathlib import Path

import flywheel_baseline as baseline
import promotion_gate
import scientific_metadata
import selfplay_milestones
from resume_expired_cycle import atomic, sha

PREREGISTRATION = Path(__file__).resolve().parents[2] / baseline.PREREGISTRATION


class FakeRun:
    """A two-generation, three-arm run with the receipts the baseline binds; every digest is real."""

    def __init__(self, root):
        self.root = root
        self.run = "runs/run/learning"
        self.exe = {}
        for name, payload in (
            ("cpu", b"cpu evaluator bytes"),
            ("monitor", b"monitor bytes"),
            ("packing", b"packing bytes"),
        ):
            path = root / "bin" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
            self.exe[name] = sha(path)
        self.put(
            "runs/monitor-corpus/index.json",
            dict(schema="parallel.teaching_corpus.v1", label_policy="reference_v1"),
        )
        self.monitor_corpus_sha = sha(root / "runs/monitor-corpus/index.json")
        self.put(
            "runs/panel.json", dict(panel_sha256="p" * 64, corpus_sha256=self.monitor_corpus_sha, contexts=16)
        )
        self.put(
            "runs/parent-manifest.json", dict(schema="parallel.continuous_learning.v1", release_revision="v3")
        )
        self.store()
        self.manifest = dict(
            schema="parallel.continuous_learning.v1",
            release_revision="test-v4",
            run_identity="test-run",
            bindings={
                "bin/cpu": self.exe["cpu"],
                "bin/monitor": self.exe["monitor"],
                "bin/packing": self.exe["packing"],
            },
            cpu_executable="bin/cpu",
            monitor_executable="bin/monitor",
            training_executable="bin/monitor",
            packing_executable="bin/packing",
            output=self.run,
            contexts_sha256="c" * 64,
            vocabulary_sha256="v" * 64,
            monitor_corpus="runs/monitor-corpus",
            monitor_panel="runs/panel.json",
            teaching_store="runs/store",
            parent_manifest=dict(
                path="runs/parent-manifest.json", sha256=sha(root / "runs/parent-manifest.json")
            ),
        )
        self.manifest_path = root / "manifest.json"
        self.put("manifest.json", self.manifest)
        self.learning_corpus = self.run + "/snapshot-000001/corpus"
        self.put(
            self.learning_corpus + "/index.json",
            dict(schema="parallel.teaching_corpus.v1", label_policy="reference_v1", records=100),
        )
        self.learning_ckpt = self.run + "/cycle-000001/train-2/checkpoints/step-9000"
        self.serving_ckpt = "runs/recovery/checkpoint"
        self.put(
            self.learning_ckpt + "/state.json",
            dict(
                step=9000,
                ranking_backward="dense_pairwise_v2",
                runtime_objective="huber_log1p_seconds_v2",
                corpus_sha256=sha(root / self.learning_corpus / "index.json"),
                vocabulary_sha256="v" * 64,
            ),
        )
        self.put(
            self.serving_ckpt + "/state.json",
            dict(
                step=3000,
                ranking_backward="dense_pairwise_v2",
                runtime_objective="huber_log1p_seconds_v2",
                corpus_sha256="k" * 64,
                vocabulary_sha256="v" * 64,
            ),
        )
        self.learning_sha = sha(root / self.learning_ckpt / "state.json")
        self.serving_sha = sha(root / self.serving_ckpt / "state.json")
        self.put(
            self.run + "/state.json",
            dict(
                manifest_sha256=sha(self.manifest_path),
                checkpoint=self.learning_ckpt,
                learning_checkpoint=self.learning_ckpt,
                learning_corpus=self.learning_corpus,
                corpus=self.learning_corpus,
                serving_checkpoint=self.serving_ckpt,
                serving_checkpoint_sha256=self.serving_sha,
                completed_blocks=2,
                generations=[],
                snapshots=[],
                trained_snapshots=[],
                tasks={},
            ),
        )
        self.monitoring()
        self.put(self.run + "/cycle-000000/train-2/complete.json", dict(seconds=240.0, steps=6000))
        self.put(self.run + "/cycle-000001/train-2/complete.json", dict(seconds=250.0, steps=9000))
        for name, fresh, stamp in (("abc", {"1": [10, 11], "2": [12]}, 0), ("def", {"3": [13]}, 3600)):
            path = self.put(
                self.run + f"/replay-membership/{name}/policy.json",
                dict(schema="parallel.replay_policy.v2", fresh_records=fresh),
            )
            os.utime(path, (stamp, stamp))
        self.generation(
            1, 6000, dict(A=(1.0, 0.9, "gp_evolution", 1.1), B=(2.0, 1.8, "retained_cut_teacher", 1.7))
        )
        self.generation(
            2,
            9600,
            dict(A=(1.05, 0.85, "neural_policy", 1.1), B=(1.6, 1.5, "gp_evolution", 1.7)),
            annotate=True,
        )
        self.gp_service()
        self.regression = "runs/registry-integration-20260910/health-v1/qualification-status/latest.json"
        # A chained run whose ranking_bt (a loss) rises from cycle 0 to cycle 2 and is rolled back at cycle 3: the pair
        # is first cycle -> last cycle before the largest single-step drop. Cycle 1 sits in an archived monitor segment.
        cycle = lambda i: "runs/chain/cycle-%06d/train-2/checkpoints/step-3000" % i
        history = [
            dict(
                checkpoint=cycle(0),
                timestamp=1,
                ranking_bt=0.5708,
                sequence_nll=1.40,
                ranking_listwise=2.2,
                runtime_huber=0.07,
            ),
            dict(
                checkpoint=cycle(2),
                timestamp=3,
                ranking_bt=0.6796,
                sequence_nll=1.43,
                ranking_listwise=2.4,
                runtime_huber=0.07,
            ),
            dict(
                checkpoint=cycle(3),
                timestamp=4,
                ranking_bt=0.54,
                sequence_nll=1.42,
                ranking_listwise=2.3,
                runtime_huber=0.07,
            ),
        ]
        archived = [
            dict(
                checkpoint=cycle(1),
                timestamp=2,
                ranking_bt=0.66,
                sequence_nll=1.41,
                ranking_listwise=2.3,
                runtime_huber=0.07,
            ),
            dict(
                checkpoint=cycle(2),
                timestamp=2.5,
                ranking_bt=0.6796,
                sequence_nll=1.43,
                ranking_listwise=2.4,
                runtime_huber=0.07,
            ),
        ]
        binding = dict(
            panel_sha256="p" * 64,
            corpus_sha256=self.monitor_corpus_sha,
            executable_sha256=self.exe["monitor"],
        )
        self.put(
            self.regression,
            dict(
                schema="parallel.continuous_status.v1",
                monitoring=dict(
                    binding=binding,
                    baseline=dict(history[0], ranking_bt=0.5337),
                    latest=dict(history[-1], ranking_bt=0.5421),
                    history=history,
                    archived_policies=[
                        dict(binding=dict(binding, executable_sha256="8" * 64), history=archived)
                    ],
                ),
            ),
        )

    def put(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic(path, value)
        return path

    def store(self):
        path = self.root / "runs/store/store.sqlite"
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(path)
        connection.execute("CREATE TABLE binding (identity TEXT NOT NULL)")
        connection.execute("INSERT INTO binding VALUES (?)", ("store-identity",))
        connection.commit()
        connection.close()

    def monitor(self, checkpoint, checkpoint_sha, step, timestamp, nll, bt, **changes):
        value = dict(
            schema="parallel.frozen_monitoring.v1",
            binding=dict(
                panel_sha256="p" * 64,
                corpus_sha256=self.monitor_corpus_sha,
                executable_sha256=self.exe["monitor"],
                checkpoint_sha256=checkpoint_sha,
            ),
            checkpoint=checkpoint,
            evaluated_step=step,
            timestamp=timestamp,
            sampled_examples=2048,
            family_balanced=True,
            independent_evaluation=False,
            training_selection_used=False,
            final_tests_opened=False,
            sequence_nll=nll,
            ranking_bt=bt,
            ranking_listwise=2.2,
            runtime_huber=0.07,
        )
        value["binding"].update({k: v for k, v in changes.items() if k in value["binding"]})
        return value

    def monitoring(self):
        policy = hashlib.sha256(
            (
                self.exe["monitor"] + ":" + sha(self.root / "runs/panel.json") + ":" + self.monitor_corpus_sha
            ).encode()
        ).hexdigest()
        self.put(
            self.run + f"/monitoring/{policy}-{self.serving_sha}.json",
            self.monitor(self.serving_ckpt, self.serving_sha, 3000, 1000, 1.40, 0.50),
        )
        self.put(
            self.run + f"/monitoring/{policy}-{'m' * 64}.json",
            self.monitor(
                self.run + "/cycle-000000/train-2/checkpoints/step-6000", "m" * 64, 6000, 2000, 1.42, 0.51
            ),
        )
        self.put(
            self.run + f"/monitoring/{policy}-{self.learning_sha}.json",
            self.monitor(self.learning_ckpt, self.learning_sha, 9000, 3000, 1.41, 0.52),
        )
        self.put(
            self.run + f"/monitoring/{'x' * 64}-{self.serving_sha}.json",
            self.monitor(
                self.serving_ckpt, self.serving_sha, 3000, 500, 1.39, 0.49, executable_sha256="x" * 64
            ),
        )
        self.put(self.run + "/monitoring/notes.json", dict(schema="something.else"))

    def generation(self, number, timestamp, contexts, annotate=False):
        directory = f"{self.run}/generation-{number:06d}"
        jobs, rows = [], []
        for key, (seed, final, origin, teacher) in sorted(contexts.items()):
            partition = "meta_validation" if key == "A" else "meta_train"
            for arm, (arm_seed, arm_final, arm_origin) in (
                ("neural", (seed, final, origin)),
                ("generic", (seed + 0.2, final + 0.1, "generic_comparison")),
                ("shuffled", (seed + 0.3, final + 0.2, "neural_policy")),
            ):
                output = f"{directory}/{arm}/ds{key}-cut0"
                seeds = output + "/seeds.json"
                self.put(
                    seeds,
                    dict(
                        candidates=[
                            dict(genome="(add x y)", origin="neural_policy"),
                            dict(genome="(mul x y)", origin="neural_policy"),
                            dict(genome="(add x y)", origin="neural_policy"),
                            dict(genome="(t x)", origin="incumbent_cut_teacher"),
                        ]
                    ),
                )
                jobs.append(
                    dict(
                        output=output,
                        seeds=seeds,
                        seeds_sha256=sha(self.root / seeds),
                        prepared=f"runs/prepared/ds{key}",
                        prepared_sha256="q" * 64,
                        cut=0,
                        rng_seed=123,
                        generation=number,
                        parent_checkpoint=self.learning_ckpt,
                    )
                )
                row = dict(
                    output=output,
                    dataset="ds" + key,
                    family="fam" + key,
                    partition=partition,
                    cut=0,
                    rng_seed=123,
                    arm=arm,
                    metric="rmse",
                    units="original_target",
                    best_final_gp=arm_final,
                    best_retained_teacher=teacher,
                    best_initial_neural_seed=seed if arm != "generic" else None,
                    best_initial_arm=arm_seed,
                    best_final_origin=arm_origin,
                )
                if annotate:
                    row["best_final_excluding_shared_tail"] = dict(
                        best_rmse=arm_final, origin=arm_origin, tail_contaminated=True
                    )
                rows.append(row)
        self.put(
            f"{directory}/{'neural'}/dsA-cut0/observations.json",
            [
                dict(
                    origin="neural_policy",
                    initial_seed_slot=0,
                    metrics=dict(rmse=contexts["A"][0], units="original_target"),
                ),
                dict(origin="gp_evolution", metrics=dict(rmse=contexts["A"][1], units="original_target")),
                dict(origin="retained_cut_teacher", metrics=dict(rmse=0.95, units="original_target")),
            ],
        )
        self.put(
            directory + "/generation.json",
            dict(number=number, checkpoint=self.learning_ckpt, checkpoint_sha256=self.learning_sha),
        )
        jobs_path = self.put(
            directory + "/jobs.json",
            dict(schema="parallel.gp_jobs.v1", qualification=True, vocabulary="vocab.json", jobs=jobs),
        )
        self.put(
            directory + "/quality-report.json",
            dict(
                schema="parallel.continuous_quality.v1",
                timestamp=timestamp,
                jobs_sha256=sha(jobs_path),
                rows=rows,
                independent_evaluation=False,
                final_tests_opened=False,
            ),
        )

    def gp_service(self):
        (self.root / "runs/dead/a").mkdir(parents=True)
        (self.root / "runs/dead/a/observations.jsonl").write_text(
            json.dumps(dict(evaluator_sha256="f" * 64)) + "\n"
        )
        self.put("runs/dead/b/observations.json", [dict(evaluator_sha256=self.exe["cpu"])])
        self.put(
            "runs/gp-service.json",
            dict(
                schema="parallel.gp_service.v1",
                timestamp=9000,
                phase="running",
                workers=104,
                active=104,
                queued=250,
                retry_pending=3,
                completed_views=5000,
                completed_evaluations=4000000,
                evaluations_per_second=205.0,
                elapsed_seconds=20000,
                executable_sha256=self.exe["cpu"],
                failed_views=4,
                failed_rows=[
                    dict(output="runs/dead/a", attempt=1),
                    dict(output="runs/dead/b", attempt=2),
                    dict(output="runs/dead/c", attempt=1),
                    dict(output="runs/dead/a", attempt=2),
                ],
            ),
        )

    def build(self, generations=10):
        return baseline.build(self.root, self.manifest_path, generations)

    def argv(self, output, *extra):
        return [
            "--root",
            str(self.root),
            "--manifest",
            str(self.manifest_path),
            "--generations",
            "10",
            "--output",
            str(output),
            *extra,
        ]


class FlywheelBaselineTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name).resolve()
        self.fixture = FakeRun(self.root)

    def test_receipt_binds_the_live_release(self):
        receipt = self.fixture.build()
        f = self.fixture
        self.assertEqual(receipt["schema"], "parallel.flywheel_baseline.v1")
        self.assertIs(receipt["final_tests_opened"], False)
        self.assertEqual(receipt["manifest_sha256"], sha(f.manifest_path))
        self.assertEqual(receipt["binding"]["manifest_sha256"], sha(f.manifest_path))
        self.assertEqual(receipt["parent_manifest_sha256"], sha(self.root / "runs/parent-manifest.json"))
        self.assertEqual(receipt["release_revision"], "test-v4")
        self.assertEqual(
            receipt["executables"],
            dict(
                cpu_evaluator=f.exe["cpu"],
                monitor_legacy=f.exe["monitor"],
                training=f.exe["monitor"],
                packing=f.exe["packing"],
            ),
        )
        self.assertEqual(
            receipt["binding"]["executable_paths"]["cpu_evaluator"], dict(path="bin/cpu", sha256=f.exe["cpu"])
        )
        self.assertEqual(receipt["teaching_store_binding"]["identities"], ["store-identity"])
        self.assertEqual(receipt["checkpoints"]["learning"]["sha256"], f.learning_sha)
        self.assertEqual(receipt["checkpoints"]["learning"]["step"], 9000)
        self.assertIsNone(receipt["checkpoints"]["learning"]["nll_row_weight"])
        self.assertIn("nll_row_weight", receipt["checkpoints"]["learning"]["absent_fields"])
        self.assertEqual(receipt["checkpoints"]["serving"]["sha256"], f.serving_sha)
        self.assertEqual(receipt["learning_corpus_sha256"], sha(self.root / f.learning_corpus / "index.json"))
        self.assertEqual(receipt["monitor_corpus_sha256"], f.monitor_corpus_sha)
        self.assertEqual(receipt["panel_sha256"], "p" * 64)
        self.assertTrue(receipt["binding"]["corpus_binding_matches_index"])
        self.assertEqual(receipt["contexts_sha256"], "c" * 64)
        self.assertEqual(receipt["vocabulary_sha256"], "v" * 64)
        self.assertEqual(receipt["label_policy"], "reference_v1")
        self.assertEqual(receipt["percentile_formula"], "1 - i/(n-1)")
        self.assertIsNone(receipt["metric_definitions_sha256"])
        self.assertEqual(
            {k: receipt["generation_range"][k] for k in ("count", "first", "last")},
            dict(count=2, first=1, last=2),
        )
        self.assertEqual(receipt["frozen_at"], 9600)
        for name in (
            "flywheel_baseline",
            "flywheel_metrics",
            "promotion_gate",
            "selfplay_milestones",
            "continuous_status",
        ):
            self.assertEqual(len(receipt["producers"][name]["sha256"]), 64)

    def test_frozen_panel_numbers_are_taken_from_the_bound_monitor_policy(self):
        legacy = self.fixture.build()["metrics"]["panel_legacy"]
        self.assertEqual(legacy["receipts_on_binding"], 3)
        self.assertEqual(sum(legacy["receipts_other_bindings"].values()), 1)
        self.assertEqual(legacy["ignored_receipts"], [self.fixture.run + "/monitoring/notes.json"])
        self.assertEqual((legacy["learning"]["sequence_nll"], legacy["learning"]["ranking_bt"]), (1.41, 0.52))
        self.assertEqual((legacy["serving"]["sequence_nll"], legacy["serving"]["ranking_bt"]), (1.40, 0.50))
        self.assertEqual([r["evaluated_step"] for r in legacy["series"]], [3000, 6000, 9000])

    def test_panel_null_borrows_guards_below_ten_receipts(self):
        null = self.fixture.build()["metrics"]["panel_null"]
        self.assertEqual(null["n_receipts"], 4)
        self.assertEqual((null["n_rows"], null["receipts_on_binding"]), (3, 3))
        self.assertEqual(
            null["binding"],
            dict(
                panel_sha256="p" * 64,
                corpus_sha256=self.fixture.monitor_corpus_sha,
                executable_sha256=self.fixture.exe["monitor"],
            ),
        )
        self.assertEqual(null["n_segments"], 1)  # only the bound monitor policy feeds the null
        self.assertEqual(null["sequence_nll"]["n_pairs"], 2)
        self.assertTrue(null["guards_borrowed"])
        self.assertIsNone(null["guards_derived"])
        self.assertEqual(null["guards"], dict(g_nll=0.005, g_bt=0.01))
        self.assertAlmostEqual(
            null["guards_from_insufficient_null"]["g_nll"], max(0.005, 2 * null["sequence_nll"]["p90"])
        )
        self.assertEqual(null["pooled_all_bindings"]["n_segments"], 2)
        self.assertIn("never a guard source", null["pooled_all_bindings"]["label"])
        reference = self.fixture.build()["metrics"]["recovery_reference"]
        self.assertEqual(
            (reference["nll"], reference["bt"]),
            (selfplay_milestones.RECOVERY_NLL, selfplay_milestones.RECOVERY_BT),
        )

    def test_panel_null_guards_derive_from_the_bound_segment_only(self):
        f = self.fixture
        directory = self.root / f.run / "monitoring"
        policy = hashlib.sha256(
            (
                f.exe["monitor"] + ":" + sha(self.root / "runs/panel.json") + ":" + f.monitor_corpus_sha
            ).encode()
        ).hexdigest()
        # Eight foreign-binding receipts with ~20% swings and seven more live ones with ~0.1% swings.
        for i in range(8):
            f.put(
                f.run + f"/monitoring/{'x' * 64}-{'f%02d' % i}{'0' * 61}.json",
                f.monitor(
                    "ckpt/foreign%d" % i,
                    ("f%02d" % i) + "0" * 61,
                    12000 + 3000 * i,
                    5000 + i,
                    1.4 * (1.2 if i % 2 else 1.0),
                    0.5 * (1.2 if i % 2 else 1.0),
                    executable_sha256="x" * 64,
                ),
            )
        for i in range(7):
            f.put(
                f.run + f"/monitoring/{policy}-{'l%02d' % i}{'0' * 61}.json",
                f.monitor(
                    "ckpt/live%d" % i,
                    ("l%02d" % i) + "0" * 61,
                    12000 + 3000 * i,
                    4000 + i,
                    1.41 * (1.001 if i % 2 else 1.0),
                    0.52 * (1.001 if i % 2 else 1.0),
                ),
            )
        null = f.build()["metrics"]["panel_null"]
        self.assertEqual((null["n_receipts"], null["receipts_on_binding"]), (19, 10))
        self.assertFalse(null["guards_borrowed"])
        self.assertEqual(null["guards"], null["guards_derived"])
        self.assertLess(
            null["guards"]["g_bt"], 0.05
        )  # ~2 x p90 of the live 0.1%-4% swings, never the foreign 20% ones
        self.assertGreater(null["pooled_all_bindings"]["guards_if_pooled"]["g_bt"], 0.3)
        self.assertEqual(null["pooled_all_bindings"]["n_segments"], 2)
        # Four live receipts alone (below the minimum) keep the guards borrowed no matter how many foreign ones exist.
        for path in sorted(directory.glob(policy + "-l0[3-6]*.json")):
            path.unlink()
        null = f.build()["metrics"]["panel_null"]
        self.assertEqual(null["receipts_on_binding"], 6)
        self.assertTrue(null["guards_borrowed"])
        self.assertEqual(null["guards"], dict(g_nll=0.005, g_bt=0.01))

    def test_regression_pair_is_first_cycle_to_the_block_before_the_rollback(self):
        pair = self.fixture.build()["metrics"]["regression_pair"]
        self.assertEqual(pair["pair"], [0.5708, 0.6796])
        self.assertEqual(pair["receipt_sha256"], sha(self.root / self.fixture.regression))
        self.assertEqual((pair["baseline"]["ranking_bt"], pair["latest"]["ranking_bt"]), (0.5337, 0.5421))
        self.assertEqual(pair["run"], "runs/chain")
        self.assertEqual((pair["first"]["cycle"], pair["end"]["cycle"], pair["peak"]["cycle"]), (0, 2, 2))
        self.assertEqual(
            pair["end"]["executable_sha256"], "8" * 64
        )  # the earliest evaluation of cycle 2 sits in the archived segment
        self.assertEqual(pair["first"]["executable_sha256"], self.fixture.exe["monitor"])
        self.assertEqual(
            pair["rollback"], dict(from_cycle=2, to_cycle=3, drop=0.54 - 0.6796, to_ranking_bt=0.54)
        )
        self.assertEqual(
            (pair["history_rows"], pair["cycles"], pair["cycle_range"], pair["shared_cycles"]),
            (5, 4, [0, 3], 1),
        )
        self.assertEqual(pair["cross_binding_disagreements"], [])
        self.assertEqual(
            [
                (s["executable_sha256"][:1], s["first_cycle"], s["last_cycle"], s["rows"])
                for s in pair["segments"]
            ],
            [(self.fixture.exe["monitor"][:1], 0, 3, 3), ("8", 1, 2, 2)],
        )
        self.assertAlmostEqual(pair["end_over_first"], 0.6796 / 0.5708)

    @unittest.skipUnless(
        (Path(__file__).resolve().parents[1] / baseline.DEFAULT_REGRESSION_RECEIPT).is_file(),
        "local registry-integration receipt absent",
    )
    def test_regression_pair_reproduces_the_preregistered_pair_on_the_real_receipt(self):
        root = Path(__file__).resolve().parents[1]
        pair = baseline.regression_pair(root, root / baseline.DEFAULT_REGRESSION_RECEIPT)
        self.assertEqual(pair["run"], "runs/continuous-learning-v2-20260909")
        self.assertEqual(
            pair["pair"], [0.570828303694725, 0.6795630231499672]
        )  # 0.5708 -> 0.6796, the 74-block chain
        self.assertEqual((pair["first"]["cycle"], pair["end"]["cycle"]), (0, 73))
        self.assertEqual((pair["rollback"]["from_cycle"], pair["rollback"]["to_cycle"]), (73, 74))
        self.assertAlmostEqual(pair["end_over_first"], 1.19, places=2)
        self.assertEqual(
            (pair["baseline"]["ranking_bt"], pair["latest"]["ranking_bt"]),
            (0.5336763821542263, 0.5420861840248108),
        )
        self.assertEqual((pair["history_rows"], pair["cycles"], pair["cycle_range"]), (133, 108, [0, 107]))
        self.assertEqual(pair["cross_binding_disagreements"], [])
        self.assertEqual([s["executable_sha256"][:8] for s in pair["segments"]], ["8ca21024", "3382cdcb"])
        self.assertEqual(
            (pair["first"]["executable_sha256"][:8], pair["end"]["executable_sha256"][:8]),
            ("8ca21024", "8ca21024"),
        )  # both in the archived segment
        self.assertEqual(
            pair["segments"][1]["first_cycle"], 60
        )  # the current segment re-monitored cycles 60..81 with identical numbers

    def test_matched_contrasts_and_signatures_over_the_window(self):
        metrics = self.fixture.build()["metrics"]
        heldout = metrics["matched_heldout"]
        self.assertEqual(heldout["n_cuts"], 1)
        self.assertEqual(
            (heldout["seed_only"]["wins"], heldout["seed_only"]["ties"], heldout["seed_only"]["losses"]),
            (2, 0, 0),
        )
        self.assertEqual(heldout["seed_only"]["vs_shuffled"]["wins"], 2)
        self.assertEqual(heldout["final"]["wins"], 2)
        self.assertEqual(heldout["final"]["tail_caveat"], baseline.TAIL_CAVEAT)
        self.assertLess(heldout["seed_only"]["mean_log_ratio"], 0)
        self.assertEqual(len(heldout["seed_only"]["ci90"]), 2)
        self.assertEqual(metrics["matched_train"]["n_cuts"], 1)
        self.assertEqual(
            (
                metrics["gp_improves_on_seed"]["neural"]["n"],
                metrics["gp_improves_on_seed"]["neural"]["improved"],
            ),
            (4, 4),
        )
        self.assertEqual(metrics["gp_improves_on_seed"]["generic"]["n"], 4)
        self.assertEqual(
            metrics["best_final_origin"]["all"],
            {"gp_evolution": 2, "retained_cut_teacher": 1, "neural_policy": 5, "generic_comparison": 4},
        )
        tail = metrics["best_final_excluding_shared_tail"]
        self.assertEqual(
            (tail["n_rows"], tail["ledger_missing"], tail["source"]),
            (7, 5, ["observations_ledger", "quality_row"]),
        )
        self.assertTrue(tail["tail_contaminated"])
        signature = metrics["regression_signature"]
        self.assertEqual(signature["generations"], [1, 2])
        self.assertEqual(signature["neural_exclusive_wins"], [1, 1])
        self.assertEqual(signature["direct_seed_beats_teacher"], [0, 1])
        self.assertEqual(signature["summary_failures"], {})
        self.assertEqual(metrics["quality_partitions"]["2"]["meta_validation"]["triplets"], 1)
        self.assertEqual(metrics["win_rates"]["neural_seed_n"], 4)

    def test_discoveries_diversity_throughput_and_queue(self):
        metrics = self.fixture.build()["metrics"]
        replay = metrics["discoveries_per_hour"]["replay_policies"]
        self.assertEqual((replay["fresh_records_total"], replay["per_hour"]), (4, 4.0))
        beats = metrics["discoveries_per_hour"]["new_teacher_beats"]
        self.assertEqual((beats["total"], beats["per_generation"]), (3, {"1": 1, "2": 2}))
        self.assertAlmostEqual(beats["per_hour"], 3 / (3600 / 3600))
        diversity = metrics["diversity"]
        self.assertEqual(diversity["unique_neural_per_context_median"], 2)
        self.assertEqual(diversity["mean_token_length"], 5.0)
        self.assertEqual(diversity["unique_fresh_programs_median"], 2)
        self.assertEqual(metrics["emission_census"]["unique_neural_per_context_median"], 2)
        self.assertEqual(metrics["emission_census"]["mean_token_length"], 5.0)
        throughput = metrics["throughput"]
        self.assertEqual(throughput["gp"]["evaluations_per_second"], 205.0)
        self.assertEqual(throughput["gp"]["views_per_hour"], 900.0)
        self.assertEqual(throughput["train_block_seconds"], dict(blocks=2, median=245.0, last=250.0))
        self.assertEqual(throughput["cadence_seconds"]["intervals"], 1)
        queue = metrics["queue"]
        self.assertEqual((queue["queued"], queue["retry_pending"], queue["ready"]), (250, 3, 247))
        self.assertEqual(
            (queue["dead_letter_candidates"], queue["ledger_missing"], queue["failed_outputs"]), (1, 1, 3)
        )
        self.assertEqual(queue["foreign_evaluators"], {"f" * 64: 1})

    def test_window_takes_the_last_generations(self):
        receipt = self.fixture.build(generations=1)
        self.assertEqual(
            {k: receipt["generation_range"][k] for k in ("count", "first", "last")},
            dict(count=1, first=2, last=2),
        )
        self.assertEqual(receipt["metrics"]["matched_heldout"]["seed_only"]["n_pairs"], 1)
        self.assertTrue(
            receipt["metrics"]["discoveries_per_hour"]["new_teacher_beats"]["per_hour"].startswith(
                "unavailable:"
            )
        )

    def test_legacy_rows_derive_best_initial_arm_from_their_own_ledger(self):
        report_path = self.root / self.fixture.run / "generation-000002/quality-report.json"
        report = json.loads(report_path.read_text())
        generic = next(r for r in report["rows"] if r["arm"] == "generic" and r["dataset"] == "dsA")
        neural = next(r for r in report["rows"] if r["arm"] == "neural" and r["dataset"] == "dsB")
        expected = generic.pop("best_initial_arm")
        neural.pop("best_initial_arm")  # no ledger on disk for this one: it must stay missing, never imputed
        atomic(report_path, report)
        self.fixture.put(
            generic["output"] + "/observations.json",
            [
                dict(
                    origin="generic_comparison",
                    initial_seed_slot=3,
                    metrics=dict(rmse=expected, units="original_target"),
                ),
                dict(
                    origin="generic_comparison",
                    initial_seed_slot=7,
                    metrics=dict(rmse=expected + 0.4, units="original_target"),
                ),
                dict(
                    origin="generic_comparison",
                    initial_seed_slot=9,
                    failure="timeout",
                    metrics=dict(rmse=0.01, units="original_target"),
                ),
                dict(
                    origin="incumbent_cut_teacher",
                    initial_seed_slot=70,
                    metrics=dict(rmse=0.05, units="original_target"),
                ),
                dict(origin="gp_evolution", metrics=dict(rmse=0.02, units="original_target")),
            ],
        )
        receipt = self.fixture.build()
        derived = receipt["generation_range"]["best_initial_arm_derived"]
        self.assertEqual((derived["observations_ledger"], derived["ledger_missing"]), (1, 1))
        window = baseline.generation_window(self.root, self.root / self.fixture.run, 10)
        rows = baseline.window_rows(window, self.root, {})
        derived_row = next(
            r for r in rows if r["arm"] == "generic" and r["dataset"] == "dsA" and r["generation"] == 2
        )
        self.assertEqual(derived_row["best_initial_arm"], expected)
        missing_row = next(
            r for r in rows if r["arm"] == "neural" and r["dataset"] == "dsB" and r["generation"] == 2
        )
        self.assertNotIn("best_initial_arm", missing_row)

    def test_receipt_is_immutable_deterministic_and_verifiable(self):
        output = self.root / "baseline/flywheel-baseline-v1.json"
        self.assertEqual(baseline.main(self.fixture.argv(output)), 0)
        first = output.read_bytes()
        # The immutable bytes exclude the sections read from moving service files; they live in the sidecar.
        expected = self.fixture.build()
        live = baseline.split_live(expected)
        self.assertEqual(first, promotion_gate.canonical(expected) + b"\n")
        self.assertEqual(set(live), set(baseline.LIVE_SECTIONS))
        self.assertNotIn("evaluations_per_second", first.decode())
        snapshot = json.loads(output.with_name("flywheel-baseline-v1.live-snapshot.json").read_text())
        self.assertEqual(snapshot["schema"], "parallel.flywheel_baseline_live_snapshot.v1")
        self.assertEqual(snapshot["receipt_sha256"], hashlib.sha256(first).hexdigest())
        self.assertEqual(snapshot["sections"]["throughput"]["gp"]["evaluations_per_second"], 205.0)
        self.assertEqual(baseline.main(self.fixture.argv(output)), 0)
        self.assertEqual(output.read_bytes(), first)
        self.assertEqual(baseline.main(self.fixture.argv(output, "--verify")), 0)
        report = self.root / self.fixture.run / "generation-000002/quality-report.json"
        value = json.loads(report.read_text())
        value["rows"][0]["best_final_gp"] = 0.5
        atomic(report, value)
        with self.assertRaisesRegex(RuntimeError, "immutable receipt changed"):
            baseline.main(self.fixture.argv(output))
        self.assertEqual(output.read_bytes(), first)
        self.assertEqual(baseline.main(self.fixture.argv(output, "--verify")), 1)

    def test_required_bindings_fail_closed(self):
        (self.root / "bin/cpu").write_bytes(b"rebuilt evaluator")
        with self.assertRaisesRegex(RuntimeError, "executable binding changed"):
            self.fixture.build()
        (self.root / "bin/cpu").write_bytes(b"cpu evaluator bytes")
        self.fixture.build()
        for key, message in (
            ("monitor_executable", "manifest lacks executable key"),
            ("output", "manifest lacks output"),
            ("monitor_panel", "manifest lacks monitor_panel"),
        ):
            manifest = copy.deepcopy(self.fixture.manifest)
            del manifest[key]
            atomic(self.fixture.manifest_path, manifest)
            with self.assertRaisesRegex(RuntimeError, message):
                self.fixture.build()
        atomic(self.fixture.manifest_path, self.fixture.manifest)
        jobs = self.root / self.fixture.run / "generation-000001/jobs.json"
        value = json.loads(jobs.read_text())
        value["jobs"][0]["rng_seed"] = 124
        atomic(jobs, value)
        with self.assertRaisesRegex(RuntimeError, "quality jobs binding changed"):
            self.fixture.build()

    def test_optional_inputs_missing_are_explicit_not_fabricated(self):
        for name in (
            "runs/gp-service.json",
            self.fixture.regression,
            "runs/store/store.sqlite",
            self.fixture.run + "/replay-membership/abc/policy.json",
            self.fixture.run + "/replay-membership/def/policy.json",
            self.fixture.run + "/cycle-000000/train-2/complete.json",
            self.fixture.run + "/cycle-000001/train-2/complete.json",
        ):
            (self.root / name).unlink()
        for path in sorted((self.root / self.fixture.run / "monitoring").glob("*.json")):
            path.unlink()
        receipt = self.fixture.build()
        metrics = receipt["metrics"]
        for value in (
            metrics["queue"],
            metrics["throughput"]["gp"],
            metrics["regression_pair"],
            receipt["teaching_store_binding"],
            metrics["discoveries_per_hour"]["replay_policies"],
            metrics["throughput"]["train_block_seconds"],
            metrics["panel_legacy"]["learning"],
            metrics["panel_legacy"]["serving"],
        ):
            self.assertTrue(isinstance(value, str) and value.startswith("unavailable:"), value)
        self.assertTrue(metrics["panel_null"]["guards_borrowed"])
        self.assertEqual(metrics["panel_null"]["guards"], dict(g_nll=0.005, g_bt=0.01))
        self.assertIsNone(metrics["diversity"]["unique_fresh_programs_median"])
        self.assertEqual(receipt["frozen_at"], 9600)
        (self.root / self.fixture.run / "state.json").unlink()
        receipt = self.fixture.build()
        for value in (
            receipt["checkpoints"]["learning"],
            receipt["checkpoints"]["serving"],
            receipt["learning_corpus_sha256"],
            receipt["label_policy"],
            receipt["percentile_formula"],
        ):
            self.assertTrue(value.startswith("unavailable:"), value)
        self.assertEqual(receipt["manifest_sha256"], sha(self.fixture.manifest_path))

    def test_no_generations_yields_explicit_gaps_and_still_binds(self):
        for directory in sorted((self.root / self.fixture.run).glob("generation-*")):
            (directory / "quality-report.json").unlink()
        receipt = self.fixture.build()
        self.assertEqual(receipt["generation_range"]["count"], 0)
        for key in (
            "matched_heldout",
            "matched_train",
            "gp_improves_on_seed",
            "regression_signature",
            "diversity",
        ):
            self.assertTrue(receipt["metrics"][key].startswith("unavailable:"), key)
        self.assertEqual(receipt["frozen_at"], 3000)


class PreregistrationTests(unittest.TestCase):
    def test_block_parses_and_its_digest_is_pinned_in_the_document(self):
        result = baseline.preregistration(PREREGISTRATION)
        block = result["block"]
        self.assertEqual(block["schema"], "parallel.flywheel_preregistration.v1")
        text = PREREGISTRATION.read_text()
        pinned = re.search(r"preregistration_sha256: `([0-9a-f]{64})`", text)
        self.assertIsNotNone(pinned, "document must pin its own block digest")
        self.assertEqual(result["sha256"], pinned.group(1))
        self.assertEqual(result["sha256"], scientific_metadata.digest(block))
        self.assertEqual(baseline.preregistration(PREREGISTRATION)["sha256"], result["sha256"])

    def test_block_fixes_the_falsifiers_exactly(self):
        block = baseline.preregistration(PREREGISTRATION)["block"]
        stats = block["statistics"]
        self.assertEqual(
            (
                stats["n_clusters"],
                stats["window_generations"],
                stats["alpha"],
                stats["resamples"],
                stats["seed"],
            ),
            (16, 10, 0.10, 2000, 20260912),
        )
        self.assertEqual(stats["secondary"]["win_rate_delta"], 0.13)
        self.assertEqual(stats["secondary"]["flag"], "cluster_overstated_2x_to_3x")
        f = block["falsifiers"]
        self.assertEqual(sorted(f), ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9"])
        # F9 (replay concentration, held-66000 diagnosis) is pre-registered with its parent, both read-out panels
        # and the three verdicts; it can select the next block's sampler but never release the hold by itself.
        self.assertEqual(f["F9"]["parent"]["step"], 63000)
        self.assertEqual(f["F9"]["protocol_sha256_prefix"], "d8d57069")
        self.assertEqual([p["name"] for p in f["F9"]["readout_panels"]], ["reserved", "old_frozen"])
        self.assertEqual(
            f["F9"]["thresholds"],
            dict(reserved_delta_max=-0.005, old_panel_tolerance=0.005, interval=0.95, resamples=10000),
        )
        self.assertEqual(sorted(f["F9"]["verdicts"]), ["inconclusive", "repair_signal"])
        self.assertIn("cannot release", f["F9"]["hold_rule"])
        self.assertEqual(f["F1"]["secondary"]["win_rate_delta"], 0.13)
        self.assertEqual(f["F1"]["honest_expectation"], "inconclusive_underpowered")
        self.assertEqual(f["F2"]["guards"]["g_nll"], "max(0.005, 2 * p90 rel|delta sequence_nll|)")
        self.assertEqual(f["F2"]["consecutive_rejects"], 3)
        self.assertEqual(
            (
                f["F3"]["unique_fresh_programs_fraction_of_baseline_median"],
                f["F3"]["unique_emissions_per_context_min"],
                f["F3"]["mean_tokens_over_baseline"],
                f["F3"]["consecutive_generations"],
            ),
            (0.5, 32, 1.3, 3),
        )
        self.assertEqual(
            (f["F4"]["p_flip_min"], f["F4"]["a_own_ci_alpha"], f["F4"]["minimum_generations"]),
            (0.10, 0.10, 5),
        )
        self.assertEqual(f["F5"]["rule"], "row-0 metrics.rmse bit-equal to the seed admitted RMSE")
        self.assertEqual(f["F6"]["tolerance"], 1e-9)
        self.assertEqual(f["F8"]["accuracy_gain_min"], 0.02)
        self.assertEqual(
            (f["F8"]["hold_below_baseline"], f["F8"]["hold_consecutive_blocks"], f["F8"]["by_block"]),
            (0.01, 3, 10),
        )
        self.assertEqual(
            block["verdict"]["outcomes"],
            ["smarter_confirmed", "no_learning_detected", "harmful_stopped", "inconclusive_underpowered"],
        )
        self.assertEqual(block["verdict"]["generations"], [5, 10, 20])
        self.assertEqual(block["owner_decisions"]["held_out_surface"]["n_clusters"], 16)
        self.assertEqual(block["owner_decisions"]["consumption_order"], "gated_critic_first")
        self.assertAlmostEqual(block["statistics"]["mde_n16"]["multiplier_of_sigma_cut"], 0.879, places=3)
        self.assertIs(block["final_tests_opened"], False)


if __name__ == "__main__":
    unittest.main()
