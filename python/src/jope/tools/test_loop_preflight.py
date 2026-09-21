import json
import tempfile
import unittest
from pathlib import Path

import loop_preflight as lp
from resume_expired_cycle import sha

RUN = "runs/x"


def job(generation, dataset, arm, cut=0):
    output = "%s/learning/generation-%06d/%s/%s/cut-%d/search" % (RUN, generation, arm, dataset, cut)
    return dict(output=output, generation=generation, cut=cut)


def tree():
    files = {}
    gen_a = [
        job(39, "d1", "neural"),
        job(39, "d2", "neural"),
        job(39, "d3", "generic"),
        job(39, "d4", "shuffled"),
        job(39, "d5", "neural"),
    ]
    gen_b = [job(40, "e1", "neural"), job(40, "e2", "generic"), job(40, "e3", "shuffled")]
    gen_c = [job(41, "f1", "neural")]
    for number, jobs in ((39, gen_a), (40, gen_b), (41, gen_c)):
        files["%s/learning/generation-%06d/jobs.json" % (RUN, number)] = dict(jobs=jobs)
    # gen 39: one completed, three missing outputs, one partial (directory without search.json)
    files[gen_a[0]["output"] + "/search.json"] = dict(schema="parallel.search.v1")
    files[gen_a[4]["output"] + "/observations.json"] = "[]"
    # gen 40: two completed, one quarantined
    files[gen_b[0]["output"] + "/search.json"] = "{}"
    files[gen_b[1]["output"] + "/search.json"] = "{}"
    files[gen_b[2]["output"] + "/service-quarantine.json"] = dict(reason="prior evaluator conflict")
    # gen 41: complete with a report (skipped from the incomplete list)
    files[gen_c[0]["output"] + "/search.json"] = "{}"
    files[RUN + "/learning/generation-000041/quality-report.json"] = dict(schema="parallel.quality_report.v1")
    files["runs/gp-priority-jobs.json"] = dict(
        schema="parallel.gp_jobs.v1", jobs=gen_c + gen_b + gen_a
    )  # newest generation first
    files["runs/gp-service.json"] = dict(
        phase="running",
        active=104,
        queued=500,
        retry_pending=0,
        ordinary_ready=187,
        continuation_ready=0,
        dispatch_policy=dict(schema="parallel.gp_dispatch_policy.v1"),
        executable_sha256="e" * 64,
        active_rows=[1, 2],
    )
    files[RUN + "/health/latest.json"] = dict(
        timestamp=1.0,
        gp=dict(active=104, queued=500, active_rows=[dict(slot=0)]),
        gpu=[dict(index=0, utilization=0.0)],
    )
    files[RUN + "/learning/cycle-000021/train-2/checkpoints/step-69000/state.json"] = dict(step=69000)
    files[RUN + "/learning/state.json"] = dict(
        manifest_sha256="m" * 64,
        learning_checkpoint=RUN + "/learning/cycle-000021/train-2/checkpoints/step-69000",
        learning_corpus=RUN + "/learning/snapshot-1/corpus",
        completed_blocks=22,
        gp_replenishing=False,
        training_diagnostic=dict(consecutive_rejections=1, held=False),
        training_growth=dict(eligible=False),
        generations=[
            dict(path=RUN + "/learning/generation-%06d" % n, number=n, stage=4, checkpoint_sha256="c" * 64)
            for n in (39, 40, 41)
        ],
    )
    files[RUN + "/learning/control.json"] = dict(
        schema="parallel.continuous_control.v1", paused=False, manifest_sha256="m" * 64, f11_pause={}
    )
    files[RUN + "/successor/decisions/f11-rollback-status.json"] = dict(
        schema="parallel.rollback_verification.v1", status="training_disabled", deviations=[]
    )
    return lp.MemoryTree(files)


class LoopPreflightTests(unittest.TestCase):
    def test_classification_counts_positions_and_summary(self):
        body = lp.collect(tree(), RUN)
        by_number = {g["number"]: g for g in body["generations"]}
        self.assertEqual(
            by_number[39]["counts"], dict(completed=1, pending_no_output=3, partial=1, quarantined=0)
        )
        self.assertEqual(
            by_number[40]["counts"], dict(completed=2, pending_no_output=0, partial=0, quarantined=1)
        )
        self.assertTrue(by_number[41]["report"])
        self.assertIsNone(by_number[41]["jobs"])  # generations with a report are not re-classified
        self.assertEqual([g["number"] for g in body["incomplete"]], [39, 40])
        self.assertEqual(body["totals"], dict(completed=3, pending_no_output=3, partial=1, quarantined=1))
        pending = {p["output"]: p for p in by_number[39]["pending"]}
        self.assertEqual(
            sorted(pending),
            [
                "generic/d3/cut-0/search",
                "neural/d2/cut-0/search",
                "neural/d5/cut-0/search",
                "shuffled/d4/cut-0/search",
            ],
        )
        self.assertEqual(
            pending["neural/d2/cut-0/search"]["queue_index"], 5
        )  # newest-first file: gen 41 (1) + gen 40 (3) + position in gen 39
        self.assertEqual(pending["neural/d5/cut-0/search"]["kind"], "partial")
        self.assertEqual(by_number[40]["pending"][0]["kind"], "quarantined")
        self.assertEqual(body["queue"]["entries"], 9)
        self.assertEqual(body["gp_service"]["ordinary_ready"], 187)
        self.assertEqual((body["health"]["active_rows"], body["health"]["gpu"]), (1, [0.0]))
        self.assertEqual(
            body["state"]["learning_checkpoint_state_sha256"],
            tree().sha256(RUN + "/learning/cycle-000021/train-2/checkpoints/step-69000/state.json"),
        )
        self.assertEqual(
            (body["control"]["paused"], body["rollback_status"]["status"]), (False, "training_disabled")
        )
        self.assertEqual(body["errors"], [])
        digest = lp.summary(body)
        self.assertEqual(digest["totals"]["pending_no_output"], 3)

    def test_missing_service_file_is_recorded_not_fatal(self):
        t = tree()
        del t.files["runs/gp-service.json"]
        body = lp.collect(t, RUN)
        self.assertIsNone(body["gp_service"])
        self.assertTrue(body["errors"] and "gp-service.json" in body["errors"][0])

    def test_remote_script_compiles_and_never_runs_main(self):
        script = lp.remote_script("/root", RUN)
        compile(script, "remote", "exec")
        self.assertIn("LOOP_PREFLIGHT_REMOTE", script)
        self.assertTrue(script.rstrip().endswith("sort_keys=True))"))

    def test_local_tree_cli_writes_an_immutable_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "root"
            for path, content in tree().files.items():
                target = root / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content)
            output = Path(tmp) / "review" / "preflight.json"
            receipt = lp.main(["--local-root", str(root), "--run", RUN, "--output", str(output)])
            self.assertEqual(receipt["schema"], lp.SCHEMA)
            self.assertEqual(receipt["totals"]["quarantined"], 1)
            digest = sha(output)
            self.assertEqual(receipt["receipt_sha256"], digest)
            self.assertEqual(json.loads(output.read_text())["totals"]["pending_no_output"], 3)
            with self.assertRaisesRegex(RuntimeError, "immutable receipt changed"):
                lp.main(
                    ["--local-root", str(root), "--run", RUN, "--output", str(output)]
                )  # a later timestamp differs


if __name__ == "__main__":
    unittest.main()
