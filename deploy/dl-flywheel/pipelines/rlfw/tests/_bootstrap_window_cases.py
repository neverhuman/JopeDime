"""Test methods split from test_no_winner_bootstrap for maintainability."""

from .test_no_winner_bootstrap import *  # noqa: F401,F403

def test_protected_roles_status_and_missing_closure_hard_refused(self):
    c = self.contexts[0]
    for change in (
        {"role": "development"},
        {"role": "sealed"},
        {"rights_pending": True},
        {"excluded": True},
        {"sealed": True},
        {"vector_sha256": ""},
        {"prepared_input_sha256": ""},
        {"winner_status": "usable_hwm"},
        {"fold": False},
    ):
        with self.subTest(change=change), self.assertRaises(ValueError):
            replace(c, **change)
    with self.assertRaises(ValueError):
        replace(c, identity_sha256="b" * 64)
    for key, value in (
        ("role", "development"),
        ("inputs_verified", False),
        ("vector_verified", False),
        ("excluded", True),
        ("rights_pending", True),
        ("context_id", "other"),
        ("view_id", "other"),
        ("fold", 1),
        ("fold", False),
        ("inputs_verified", 1),
        ("excluded", 0),
    ):
        with self.subTest(key=key), self.assertRaises(ValueError):
            check_training_context(
                c,
                lambda cid, key=key, value=value: {**self.role(cid), key: value},
                SHA,
            )


def test_seeded_wave_one_original_parent_and_exact_native_identity(self):
    contexts = self.contexts + (
        replace(self.contexts[0], context_id="sibling", view_id="sibling-view"),
    )
    role = lambda cid: next(
        c for c in contexts if c.context_id == cid
    ).authority_value(SHA)
    wave = select_wave(contexts, role, SHA)
    self.assertEqual(len(wave), 3)
    self.assertEqual(wave, select_wave(tuple(reversed(contexts)), role, SHA))
    self.assertTrue(all(c in contexts for c in wave))
    self.assertTrue(all(c.fold == 0 for c in wave))
    self.assertEqual(BootstrapSpec.from_dict(self.spec.to_dict()), self.spec)


def test_injected_campaign_window_is_required_and_exactly_72_hours(self):
    window = self.window
    self.assertEqual(self.spec.deadline_unix, window.deadline_unix)
    with mock.patch("rlfw.arena.oracle.time.time", return_value=WINDOW_CREATED):
        # The approved launcher's capture reads all three clocks together.
        self.assertEqual(CampaignWindow.opened_now(), window)
    for change in (
        {"window_hours": 71},
        {"window_hours": 72.0},
        {"window_hours": True},
        {"deadline_unix": window.deadline_unix + 1},
        {"deadline_unix": window.deadline_unix - 1},
        {"created_unix": float("nan")},
        {"created_monotonic": -1.0},
        {"created_monotonic": float("inf")},
        {"boot_id": ""},
        {"boot_id": " fixture-boot"},
        {"boot_id": None},
    ):
        with self.subTest(change=change), self.assertRaises(ValueError):
            replace(window, **change)
    for absent in (None, window.to_manifest()):
        with self.subTest(absent=absent), self.assertRaises(ValueError):
            replace(self.spec, campaign_window=absent)
    with self.assertRaises(ValueError):
        replace(self.spec, kpi_version="rlfw-kpi/v1")
    manifest = window.to_manifest()
    self.assertEqual(CampaignWindow.from_manifest(manifest), window)
    for bad in (
        None,
        {k: v for k, v in manifest.items() if k != "schema"},
        {k: v for k, v in manifest.items() if k != "boot_id"},
        {**manifest, "renewed": True},
        {**manifest, "schema": "jope.rlfw-bootstrap-campaign-window/v0"},
    ):
        with self.subTest(bad=bad), self.assertRaises(ValueError):
            CampaignWindow.from_manifest(bad)
    value = self.spec.to_dict()
    value.pop("campaign_window")
    for bad in (
        value,
        {**self.spec.to_dict(), "schema": "jope.rlfw-bootstrap-spec/v1"},
    ):
        with self.assertRaises(ValueError):
            BootstrapSpec.from_dict(bad)


def test_closed_campaign_window_refuses_proposals_and_execution(self):
    window = self.window
    for name, now, skew, boot in (
        ("at deadline", window.deadline_unix, 0.0, "fixture-boot"),
        ("after deadline", window.deadline_unix + 1, 0.0, "fixture-boot"),
        ("before opening", window.created_unix - 1, 0.0, "fixture-boot"),
        (
            "monotonic expiry",
            window.created_unix + 3600,
            72 * 3600.0,
            "fixture-boot",
        ),
        ("other boot", window.created_unix + 3600, 0.0, "other-boot"),
    ):
        with (
            self.subTest(name),
            mock.patch("rlfw.arena.oracle.time.time", return_value=now),
        ):
            self.skew, self.boot = skew, boot
            value = self.scheduler()
            with self.assertRaises(RuntimeError):
                value.run(max_contexts=1)
            self.assertEqual((self.draws, self.calls, value.rows()), (0, [], []))
            self.assertEqual(value.coverage_report()["contexts_attempted"], 0)
            value.close()
    # The oracle repeats the check before every execution of a prepared slate.
    self.skew, self.boot = 0.0, "fixture-boot"
    with mock.patch("rlfw.arena.oracle.time.time") as clock:
        clock.return_value = window.created_unix + 3600
        value = self.scheduler()
        context = self.contexts[0]
        candidate = value.slate(context, self.arm, 20261009)[0]
        clock.return_value = window.deadline_unix
        with self.assertRaisesRegex(RuntimeError, "deadline"):
            value.oracle.resolve(
                context,
                self.arm,
                20261009,
                1,
                candidate,
                duplicate=False,
                commit=value.commit,
                attempt_store=value,
            )
    self.assertEqual((self.calls, value.rows()), ([], []))
    self.assertEqual(
        value.attempt_history((context.context_id, self.arm.name, 20261009, 1)), []
    )


def test_reboot_between_attempts_blocks_next_request(self):
    self.attempt_boundary(self, "other boot")


def test_monotonic_expiry_between_attempts_blocks_next_request(self):
    self.attempt_boundary(self, "monotonic expiry")


def test_wall_expiry_between_attempts_blocks_next_request(self):
    self.attempt_boundary(self, "wall deadline")


def test_wall_expiry_during_reservation_blocks_request(self):
    self.reservation_boundary(self, "wall deadline")


def test_monotonic_expiry_during_reservation_blocks_request(self):
    self.reservation_boundary(self, "monotonic expiry")


def test_reboot_during_reservation_blocks_request(self):
    self.reservation_boundary(self, "other boot")


def test_expiry_during_admission_blocks_proposals(self):
    for change in CLOCK_CHANGES:
        with self.subTest(change):
            self.admission_boundary(self.fresh(), change)


def test_window_closing_between_model_dispatches_stops_the_next(self):
    for change in CLOCK_CHANGES:
        with self.subTest(change):
            self.dispatch_boundary(self.fresh(), change)


def test_no_literal_campaign_deadline_in_rlfw_sources(self):
    package = Path(__file__).resolve().parents[1]
    sources = sorted(
        p
        for p in package.rglob("*.py")
        if "tests" not in p.relative_to(package).parts
    )
    self.assertIn(package / "no_winner_bootstrap.py", sources)
    self.assertIn(package / "bench_cli.py", sources)
    # Unix-second values (2001..2286), as numbers or digit strings, and calendar
    # dates are refused anywhere in the package source outside its fixtures.
    text = re.compile(r"\b20\d\d-[01]\d-[0-3]\d\b|\b[1-9]\d{9}(?:\.\d+)?\b")
    found = []
    for path in sources:
        for node in ast.walk(ast.parse(path.read_bytes(), filename=str(path))):
            value = node.value if isinstance(node, ast.Constant) else None
            if (type(value) in (int, float) and 1e9 <= abs(value) < 1e10) or (
                isinstance(value, str) and text.search(value)
            ):
                found.append((str(path.relative_to(package)), node.lineno, value))
    self.assertEqual(found, [])


TEST_METHODS = (
    "test_protected_roles_status_and_missing_closure_hard_refused",
    "test_seeded_wave_one_original_parent_and_exact_native_identity",
    "test_injected_campaign_window_is_required_and_exactly_72_hours",
    "test_closed_campaign_window_refuses_proposals_and_execution",
    "test_reboot_between_attempts_blocks_next_request",
    "test_monotonic_expiry_between_attempts_blocks_next_request",
    "test_wall_expiry_between_attempts_blocks_next_request",
    "test_wall_expiry_during_reservation_blocks_request",
    "test_monotonic_expiry_during_reservation_blocks_request",
    "test_reboot_during_reservation_blocks_request",
    "test_expiry_during_admission_blocks_proposals",
    "test_window_closing_between_model_dispatches_stops_the_next",
    "test_no_literal_campaign_deadline_in_rlfw_sources",
)
