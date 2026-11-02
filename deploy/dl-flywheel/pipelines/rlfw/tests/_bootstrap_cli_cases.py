"""Test methods split from test_no_winner_bootstrap for maintainability."""

from .test_no_winner_bootstrap import *  # noqa: F401,F403

def test_cli_defaults_off_cpu_and_refuses_before_runtime_import(self):
    args = [
        "--spec",
        "missing.json",
        "--runtime-binding",
        "missing.py",
        "--policy-code-root",
        "missing",
        "--policy-qualification",
        "missing.json",
    ]
    self.assertEqual(parser().parse_args(args).device, "cpu")
    with self.assertRaises(ValueError):
        main(args)
    with self.assertRaises(ValueError):
        Program.from_dict(
            {
                "dsl": "x",
                "tokens": [1, 2],
                "program_sha256": hashlib.sha256(b"x").hexdigest(),
                "q_grid": 0,
            }
        )


@mock.patch("rlfw.arena.oracle.time.time", return_value=1791572400.0)
@mock.patch("rlfw.policy_binding.verify_manifest")
@mock.patch("rlfw.bench_cli.verify_runtime_scope")
def test_enabled_cli_executes_synthetic_authorized_wave_and_reports(
    self, scope, manifest, clock
):
    binding = self.root / "binding.py"
    binding.write_text(
        "from rlfw.tests.test_no_winner_bootstrap import RUNTIME_FIXTURES\n"
        "def build_bootstrap(spec, *, device='cpu'):\n"
        "    return RUNTIME_FIXTURES[spec.game_id]\n"
    )
    calibration = self.root / "calibration.json"
    calibration.write_text(
        json.dumps(
            {
                "schema": "jope.rlfw-q-positive-calibration/v1",
                "passed": True,
                "partition": "parent-held-train",
                "checkpoint_parameter_sha256": SHA,
                "trie_sha256": SHA,
                "normalizer_sha256": SHA,
                "tokenizer_sha256": SHA,
            }
        )
    )
    qualification = self.root / "qualification.json"
    qualification.write_text(
        json.dumps(
            {
                "passed": True,
                "draws": 1000,
                "determinism_passed": True,
                "maximum_logprob_error": 0.0,
                "checkpoint_parameter_sha256s": [SHA],
            }
        )
    )
    pin = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    spec = replace(
        self.spec,
        runtime_binding_sha256=pin(binding),
        calibration_sha256=pin(calibration),
        policy_qualification_sha256=pin(qualification),
        runtime_files=tuple(
            (p.name, pin(p)) for p in (binding, calibration, qualification)
        ),
    )
    spec_path = self.root / "bootstrap.json"
    spec_path.write_text(json.dumps(spec.to_dict()))
    window_path = self.root / "campaign-window.json"
    window_path.write_text(json.dumps(self.window.to_manifest()))
    code_root = self.root / "code_pin"
    code_root.mkdir()
    catalog = [
        {
            "dsl": f"leaf{i}",
            "program_sha256": hashlib.sha256(f"leaf{i}".encode()).hexdigest(),
            "tokens": [1, i + 3, 2],
        }
        for i in range(6640)
    ]

    @contextlib.contextmanager
    def admission():
        yield True

    RUNTIME_FIXTURES[spec.game_id] = {
        "catalog": catalog,
        "sample": lambda *a: None,
        "sample_many": self.sample_many,
        "score_heads": self.heads,
        "coverage_probabilities": lambda cid, arm, rows, pin: [0.7 for _ in rows],
        "evaluate": self.result,
        "role_check": self.role,
        "reserve_empty_arena": admission,
        "arena_queue_empty": lambda: True,
    }
    self.addCleanup(RUNTIME_FIXTURES.pop, spec.game_id, None)
    args = [
        "--enable-no-winner-bootstrap",
        "--spec",
        str(spec_path),
        "--campaign-window",
        str(window_path),
        "--runtime-binding",
        str(binding),
        "--policy-code-root",
        str(code_root),
        "--policy-qualification",
        str(qualification),
        "--max-contexts",
        "1",
        "--pool-workers",
        "1",
    ]
    with mock.patch("builtins.print"):
        self.assertEqual(main(args), 0)
    report = json.loads((Path(spec.out_dir) / "coverage-report.json").read_text())
    self.assertEqual(report["contexts_attempted"], 1)
    self.assertEqual(report["contexts_with_valid_program"], 1)
    self.assertEqual(report["optimizer_updates"], 0)
    self.assertEqual(report["teacher_admissions"], 0)
    self.assertTrue((Path(spec.out_dir) / "decisions.jsonl").exists())
    scope.assert_called_once()
    manifest.assert_called_once()


@mock.patch("rlfw.bench_cli.verify_runtime_scope")
def test_enabled_cli_refuses_absent_unbound_or_closed_window(self, scope):
    spec_path = self.root / "bootstrap.json"
    spec_path.write_text(json.dumps(self.spec.to_dict()))
    window_path = self.root / "campaign-window.json"
    args = [
        "--enable-no-winner-bootstrap",
        "--spec",
        str(spec_path),
        "--runtime-binding",
        "missing.py",
        "--policy-code-root",
        "missing",
        "--policy-qualification",
        "missing.json",
    ]
    with self.assertRaisesRegex(ValueError, "campaign window"):
        main(args)
    open_now = self.window.created_unix + 3600
    for name, manifest, now, error in (
        ("absent", None, open_now, FileNotFoundError),
        (
            "other window",
            replace(self.window, boot_id="b").to_manifest(),
            open_now,
            ValueError,
        ),
        (
            "no schema",
            {**self.window.to_manifest(), "schema": None},
            open_now,
            ValueError,
        ),
        (
            "closed",
            self.window.to_manifest(),
            self.window.deadline_unix,
            RuntimeError,
        ),
    ):
        if manifest is not None:
            window_path.write_text(json.dumps(manifest))
        with (
            self.subTest(name),
            mock.patch("rlfw.arena.oracle.time.time", return_value=now),
            self.assertRaises(error),
        ):
            main(args + ["--campaign-window", str(window_path)])
    with self.assertRaises(ValueError):
        main(args + ["--campaign-window", "/proc/sys/kernel/random/boot_id"])
    scope.assert_not_called()


TEST_METHODS = (
    "test_cli_defaults_off_cpu_and_refuses_before_runtime_import",
    "test_enabled_cli_executes_synthetic_authorized_wave_and_reports",
    "test_enabled_cli_refuses_absent_unbound_or_closed_window",
)
