from dataclasses import replace
from fractions import Fraction
import unittest

from guarded_job import CpuPoint, Guard, Health, Leaf, RootScopeCandidate, ScopeFacts, cpu_interval


class GuardTests(unittest.TestCase):
    def setUp(self):
        self.leaf = Leaf("a"*64, "/sys/fs/cgroup/owned.slice/jope-research-fixture.scope", 31, 101, ((123, 456),))
        self.scope = ScopeFacts(self.leaf, 0, True, False, True, True, True, True, True, self.leaf.lease_sha256)
        self.health = Health(True, True, True, True)
        self.guard = Guard(self.leaf, deadline_ns=100_000_000_000)
        self.last_point = None

    def points(self, percent=20, own_cores=0, cpus=128, t=1):
        before = CpuPoint(t*1_000_000_000, tuple(range(cpus)), (1000,)*8, 1_000_000)
        # Total 1000 ticks per interval; idle alone distinguishes host percent.
        after = CpuPoint((t+1)*1_000_000_000, tuple(range(cpus)),
                         (1000+percent*10, 1000, 1000, 1000+(100-percent)*10, 1000, 1000, 1000, 1000),
                         1_000_000+own_cores*1_000_000)
        return before, after

    def tick(self, percent=20, own_cores=0, scope=None, health=None, cpus=128):
        before, after = self.points(percent, own_cores, cpus)
        if self.last_point is not None:
            before = self.last_point
            delta = (percent*10, 0, 0, (100-percent)*10, 0, 0, 0, 0)
            after = CpuPoint(before.monotonic_ns+1_000_000_000, tuple(range(cpus)),
                             tuple(a+b for a, b in zip(before.ticks, delta)),
                             before.owned_usage_usec+own_cores*1_000_000)
        self.last_point = after
        return self.guard.observe(before, after, after.monotonic_ns, scope or self.scope, health or self.health)

    def ready(self, **kw):
        self.tick(**kw)
        self.tick(**kw)
        return self.tick(**kw)

    def test_three_safe_samples_before_quota_and_release_plan(self):
        self.assertEqual(self.tick()["mode"], "HOLD")
        self.assertEqual(self.tick()["mode"], "HOLD")
        plan = self.tick()
        self.assertEqual(plan["mode"], "RUN_CANDIDATE")
        self.assertEqual(plan["actions"][0], {"op": "set_cpu_max", "quota_usec": 800000, "period_usec": 100000})
        self.assertEqual([a["op"] for a in plan["actions"]], ["set_cpu_max", "readback_cpu_max", "release_own_freeze_after_readback"])
        self.assertFalse(plan["live_execution_qualified"])
        self.assertFalse(plan["actions_executed"])

    def test_65_percent_holds_before_70_and_resets_safe_streak(self):
        self.ready()
        p = self.tick(65)
        self.assertEqual(p["mode"], "HOLD")
        self.assertEqual(p["actions"][0]["op"], "freeze_own_leaf")
        self.assertEqual(self.tick()["mode"], "HOLD")
        self.assertEqual(self.tick()["mode"], "HOLD")
        self.assertEqual(self.tick()["mode"], "RUN_CANDIDATE")

    def test_70_percent_records_breach_and_latches_stop(self):
        p = self.tick(70)
        self.assertEqual(p["mode"], "STOP")
        self.assertEqual(p["breach_samples"], [1])
        self.assertEqual(p["peak_host_pct"], 70)
        self.assertEqual(self.tick()["mode"], "STOP")

    def test_foreign_surge_freezes_only_named_leaf(self):
        self.ready()
        p = self.tick(88, own_cores=3)
        self.assertEqual(p["leaf_path"], self.leaf.path)
        self.assertEqual(p["actions"], [{"op": "freeze_own_leaf"}, {"op": "readback_frozen"}])
        self.assertEqual(p["breach_samples"], [4])
        self.assertEqual(p["host_pct"], 88)

    def test_actual_online_cpus_bound_quota_not_assumed_128(self):
        p = self.ready(percent=58, cpus=32)
        self.assertEqual(p["actual_online_cpus"], 32)
        self.assertEqual(p["actions"][0]["quota_usec"], 64000)

    def test_owned_foreign_accounting_retains_one_core_alignment_margin(self):
        before, after = self.points(55, 4)
        view = cpu_interval(before, after, after.monotonic_ns)
        self.assertEqual(view["owned_cores"], 4)
        self.assertEqual(view["foreign_upper_cores"], Fraction(55, 100)*128-3)
        p = self.ready(percent=59, own_cores=4)
        self.assertEqual(p["mode"], "RUN_CANDIDATE")
        self.assertEqual(p["actions"][0]["quota_usec"], 428000)

    def test_small_positive_quota_below_minimum_is_held(self):
        self.assertEqual(self.ready(percent=60)["mode"], "HOLD")

    def test_sample_age_and_sampling_interval_fail_closed(self):
        before, after = self.points()
        for b, a, now in [(before, after, after.monotonic_ns+250000001),
                          (before, after, after.monotonic_ns-1),
                          (before, replace(after, monotonic_ns=before.monotonic_ns+1250000001), before.monotonic_ns+1250000001),
                          (before, replace(after, monotonic_ns=before.monotonic_ns+899999999), before.monotonic_ns+899999999)]:
            with self.subTest(now=now):
                g = Guard(self.leaf, deadline_ns=100_000_000_000)
                p = g.observe(b, a, now, self.scope, self.health)
                self.assertEqual(p["mode"], "STOP")
                self.assertEqual(p["actions"][0]["op"], "freeze_own_leaf")

    def test_hotplug_counter_reset_and_missing_sample_refused(self):
        before, after = self.points()
        for a in [None, replace(after, online_ids=tuple(range(127))), replace(after, ticks=(0,)*8),
                  replace(after, owned_usage_usec=0), replace(after, ticks=before.ticks)]:
            with self.subTest(point=a):
                g = Guard(self.leaf, deadline_ns=100_000_000_000)
                p = g.observe(before, a, after.monotonic_ns, self.scope, self.health)
                self.assertEqual(p["mode"], "STOP")

    def test_inconsistent_owned_cpu_not_subtracted_to_create_headroom(self):
        p = self.tick(percent=1, own_cores=8)
        self.assertEqual(p["mode"], "STOP")
        self.assertIn("inconsistent", p["reason"])

    def test_replaying_one_good_interval_cannot_count_as_three(self):
        before, after = self.points()
        self.guard.observe(before, after, after.monotonic_ns, self.scope, self.health)
        plan = self.guard.observe(before, after, after.monotonic_ns, self.scope, self.health)
        self.assertEqual(plan["mode"], "STOP")
        self.assertIn("repeated/gapped", plan["reason"])

    def test_replaced_inode_pid_reuse_foreign_member_no_write_plan(self):
        for leaf in [replace(self.leaf, inode=999), replace(self.leaf, device=99),
                     replace(self.leaf, member_starts=((123, 457),)),
                     replace(self.leaf, member_starts=((123, 456), (124, 100))),
                     replace(self.leaf, lease_sha256="b"*64),
                     replace(self.leaf, path="/sys/fs/cgroup/other.slice/jope-research-fixture.scope")]:
            with self.subTest(leaf=leaf):
                g = Guard(self.leaf, deadline_ns=100_000_000_000)
                p = self.tick(scope=replace(self.scope, leaf=leaf))
                self.assertEqual(p["mode"], "STOP")
                self.assertEqual(p["actions"], [])

    def test_child_or_escaped_membership_never_targets_unknown_process(self):
        for scope in [replace(self.scope, child_cgroups=1), replace(self.scope, descendants_confined=False),
                      replace(self.scope, delegated_leaf_qualified=False)]:
            p = self.tick(scope=scope)
            self.assertEqual(p["actions"], [])

    def test_ancestor_hold_cannot_be_thawed_by_leaf_plan(self):
        p = self.tick(scope=replace(self.scope, ancestor_frozen=True))
        self.assertEqual(p["mode"], "STOP")
        self.assertNotIn("release_own_freeze_after_readback", [x["op"] for x in p["actions"]])
        self.assertEqual(p["leaf_path"], self.leaf.path)

    def test_foreign_or_unknown_leaf_freeze_never_released(self):
        for frozen, owner in [(True, None), (True, "b"*64), (None, None), (False, "a"*64)]:
            with self.subTest(frozen=frozen, owner=owner):
                p = self.tick(scope=replace(self.scope, leaf_frozen=frozen, freeze_owner_sha256=owner))
                self.assertEqual(p["mode"], "STOP")
                self.assertEqual(p["actions"], [])

    def test_already_running_owned_leaf_needs_quota_readback_without_thaw(self):
        scope = replace(self.scope, leaf_frozen=False, freeze_owner_sha256=None)
        p = self.ready(scope=scope)
        self.assertEqual(p["mode"], "RUN_CANDIDATE")
        self.assertEqual([x["op"] for x in p["actions"]], ["set_cpu_max", "readback_cpu_max"])

    def test_watchdog_and_readback_qualification_failure(self):
        for field in ["parent_watchdog_alive", "parent_watchdog_qualified", "controller_readback_qualified"]:
            with self.subTest(field=field):
                g = Guard(self.leaf, deadline_ns=100_000_000_000)
                before, after = self.points()
                p = g.observe(before, after, after.monotonic_ns, replace(self.scope, **{field: False}), self.health)
                self.assertEqual(p["mode"], "STOP")
                self.assertEqual(p["actions"][0]["op"], "freeze_own_leaf")

    def test_all_resource_guards_explicitly_true(self):
        for field in ["ram_ok", "swap_ok", "temperature_ok", "storage_ok"]:
            for value in [False, None, 1]:
                with self.subTest(field=field, value=value):
                    p = self.tick(health=replace(self.health, **{field: value}))
                    self.assertEqual(p["mode"], "STOP")

    def test_expired_bound_deadline_stops_without_budget_reset(self):
        self.guard = Guard(self.leaf, deadline_ns=2_000_000_000)
        self.assertEqual(self.tick()["mode"], "STOP")

    def test_unsafe_shared_ancestor_paths_and_duplicate_pid_refused(self):
        for leaf in [replace(self.leaf, path="/sys/fs/cgroup"),
                     replace(self.leaf, path="/sys/fs/cgroup/jarambit-evo.slice"),
                     replace(self.leaf, path="/sys/fs/cgroup/../jope-research-x.scope"),
                     replace(self.leaf, member_starts=((123, 456), (123, 457))),
                     replace(self.leaf, member_starts=()), replace(self.leaf, inode=True)]:
            with self.subTest(leaf=leaf), self.assertRaises(ValueError):
                Guard(leaf, deadline_ns=100_000_000_000)

    def test_maximum_owned_budget_is_bounded_and_rejects_bool(self):
        for value in [9, 0, True, 2.5]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                Guard(self.leaf, deadline_ns=100_000_000_000, max_owned_cores=value)


class RootScopeTests(unittest.TestCase):
    def setUp(self):
        self.spec = RootScopeCandidate("a"*64, "b"*64, "c"*64, "d"*32, tuple(range(128)), (22, 23),
            1000, 1000, "/home/ubuntu/jope-evidence/native-feedback-upgrade-20261020/future-fixture",
            ("/usr/bin/python3", "/qualified-fixture/start_barrier.py", "--lease", "c"*64),
            6*1024**3, 8*1024**3, 3600)

    def test_root_owned_exact_scope_does_not_assume_user_delegation(self):
        p = self.spec.proposal()
        self.assertEqual(p["unit"], "jope-research-aaaaaaaaaaaa-"+"d"*32+".scope")
        self.assertFalse(p["live_execution_qualified"])
        self.assertFalse(p["commands_executed"])
        self.assertFalse(p["existing_wrapper_modified"])
        self.assertFalse(p["shared_service_or_ancestor_write"])
        self.assertEqual(p["root_exec_environment"], {"PATH": "/usr/sbin:/usr/bin:/sbin:/bin", "LANG": "C.UTF-8"})
        self.assertIn("--slice=jarambit-evo.slice", p["launch_argv_candidate"])
        self.assertIn("--property=CPUQuota=25%", p["launch_argv_candidate"])

    def test_literal_argv_cleans_environment_and_drops_privilege_before_barrier(self):
        spec = replace(self.spec, barrier_argv=("/usr/bin/python3", "/fixture/a b.py", "$(literal)$HOME"))
        args = spec.proposal()["launch_argv_candidate"]
        self.assertEqual(args[-3:], list(spec.barrier_argv))
        self.assertIn("--expand-environment=no", args)
        self.assertLess(args.index("/usr/bin/setpriv"), args.index("/usr/bin/env"))
        self.assertLess(args.index("/usr/bin/env"), args.index("/usr/bin/python3"))
        self.assertIn("-i", args)
        self.assertIn("CUDA_VISIBLE_DEVICES=", args)
        self.assertNotIn("--preserve-env", args)

    def test_controls_only_exact_generated_unit_and_runtime_quota(self):
        q = self.spec.control_argv("quota", quota_usec=142801)
        self.assertIn("--runtime", q)
        self.assertIn("CPUQuota=142.801%", q)
        self.assertIn(self.spec.unit, q)
        self.assertEqual(self.spec.control_argv("freeze")[-2:], ["freeze", self.spec.unit])
        self.assertEqual(self.spec.control_argv("thaw")[-2:], ["thaw", self.spec.unit])
        for op in ["kill", "stop", "restart", "revert", "set-property", "*"]:
            with self.subTest(op=op), self.assertRaises(ValueError):
                self.spec.control_argv(op)

    def test_quota_cannot_exceed_exact_allocated_cpu_count(self):
        for q in [200001, 24999, 0, True, -1, None]:
            with self.subTest(quota=q), self.assertRaises(ValueError):
                self.spec.control_argv("quota", quota_usec=q)
        with self.assertRaises(ValueError):
            self.spec.control_argv("freeze", quota_usec=25000)

    def test_invalid_scope_nonce_or_identity_cannot_target_foreign_unit(self):
        for field, value in [("nonce", "*.slice"), ("nonce", "x"*32), ("task_sha256", "--system"),
                             ("lease_sha256", ""), ("source_sha256", "b"*63)]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                replace(self.spec, **{field: value}).proposal()

    def test_cpu_roster_and_nonroot_identity_must_be_explicit(self):
        for field, value in [("allowed_ids", (128,)), ("allowed_ids", tuple(range(9))),
                             ("allowed_ids", (22, 22)), ("online_ids", (1, 0)), ("uid", 0), ("gid", True)]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                replace(self.spec, **{field: value}).proposal()

    def test_resource_bounds_cannot_expand_during_proposal(self):
        for field, value in [("runtime_seconds", 3601), ("runtime_seconds", 0),
                             ("memory_max_bytes", 9*1024**3), ("memory_high_bytes", 9*1024**3),
                             ("device_visibility", "all"), ("device_visibility", 0)]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                replace(self.spec, **{field: value}).proposal()

    def test_unsafe_cwd_or_option_shaped_entry_refused(self):
        for field, value in [("cwd", "/tmp/job"), ("cwd", "/home/ubuntu/jope-evidence/../JopeDime"),
                             ("barrier_argv", ("--shell",)), ("barrier_argv", ("/usr/bin/python3", "bad\narg"))]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                replace(self.spec, **{field: value}).proposal()


if __name__ == "__main__":
    unittest.main()
