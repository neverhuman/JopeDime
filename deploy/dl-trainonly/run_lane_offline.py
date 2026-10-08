#!/usr/bin/env python3
"""Run only audited DL source fixtures, offline and with GPUs hidden."""
import argparse
import contextlib
import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import socket
import sys
import time
import unittest
from unittest import mock

from verify_lane import verify

MODULES = (
    "tests.test_rowsel_raw_split_v2", "tests.test_rowsel_split_v2_groups", "tests.test_v2_rowsel_raw",
    "tests.test_v2_rowsel_raw_helper", "tests.test_v2_rowsel_build", "tests.test_v2_rowsel_build_split_v2",
    "tests.test_v2_rowsel_build_split_v2g", "tests.test_v2_nonraw_split_v2", "tests.test_v2_nonraw_split_v2g",
    "tests.test_v2_publish", "tests.test_rowsel", "tests.test_nonnumeric", "tests.test_t1raw",
    "tests.test_t1prep", "tests.test_trainonly", "tests.test_s3io", "tests.test_split_defaults", "tests.test_v2_provenance",
    "tests.test_jtt1", "tests.test_payload_strip", "tests.test_quarantine", "tests.test_admission_fail_closed", "test_hwm_publish", "test_gpu_manifest_cut",
    "test_manifest_binding",
    "tests.test_v2_rowsel_raw_v2g", "tests.test_target_linear", "tests.test_v2_rowsel_raw_fold_isolation",
    "tests.test_v2_nonraw_target_linear", "tests.test_v2_rowsel_build_fallback", "tests.test_v2_rowsel_build_fold_isolation", "tests.test_v2_build_local_fold_isolation",
    "tests.test_v2_partial_refusal_manifest",
)
PRIVATE_INPUT_ROOTS = (
    Path("/home/ubuntu/scratch/jopedime-dl-training-v1"),
    Path("/home/ubuntu/scratch/jopedime-dltrain-audit-v1"),
)
HOST_CASES = {
    "tests.test_v2_rowsel_build.Scope.test_real_scope_is_the_admitted_class": "requires pinned private incoming-scope metadata and qualified host custody",
    "tests.test_v2_rowsel_build.FrozenScope.test_real_frozen_scope": "requires pinned private frozen-scope/rights metadata and qualified host custody",
    "tests.test_v2_rowsel_build_split_v2.CodeCustody.test_summary_completes_when_recorded_bytes_are_in_custody": "summary reads the pinned private NONRAW-SPLIT-A owner decision; requires qualified host custody",
    "tests.test_v2_rowsel_build_split_v2.CodeCustody.test_summary_fails_when_a_recorded_hash_cannot_be_produced": "summary reads the pinned private NONRAW-SPLIT-A owner decision; requires qualified host custody",
    "tests.test_v2_rowsel_build_split_v2.Registry.test_the_default_route_builds_the_previous_builders_bytes": "loads the private-lane builder copy ed00684a (same bytes retained as lane/historical/v2_rowsel_build.py.ed00684a...); requires qualified host custody",
    "tests.test_v2_rowsel_build_split_v2.Scope.test_a_fallback_dataset_rebuilt_on_raw_is_refused": "class setup loads pinned private scale-check sources and tree-scope receipts; requires qualified host custody",
    "tests.test_v2_rowsel_build_split_v2.Scope.test_the_three_trees": "class setup loads pinned private scale-check sources and tree-scope receipts; requires qualified host custody",
    "tests.test_v2_rowsel_build_split_v2g.Evidence.test_a_changed_later_repeat_is_not_reproduced": "class setup loads the pinned private registry port gsk.py 83183db4; requires qualified host custody",
    "tests.test_v2_rowsel_build_split_v2g.Evidence.test_a_field_and_value_grouping_mismatch_is_refused": "class setup loads the pinned private registry port gsk.py 83183db4; requires qualified host custody",
    "tests.test_v2_rowsel_build_split_v2g.Evidence.test_ambiguous_columns_need_identical_component_structure": "class setup loads the pinned private registry port gsk.py 83183db4; requires qualified host custody",
    "tests.test_v2_rowsel_build_split_v2g.Evidence.test_fewer_components_than_outer_folds_plan_trailing_empty_folds": "class setup loads the pinned private registry port gsk.py 83183db4; requires qualified host custody",
    "tests.test_v2_rowsel_build_split_v2g.Evidence.test_group_values_are_canonical_float64_values": "class setup loads the pinned private registry port gsk.py 83183db4; requires qualified host custody",
    "tests.test_v2_rowsel_build_split_v2g.Evidence.test_missing_cells_are_one_group": "class setup loads the pinned private registry port gsk.py 83183db4; requires qualified host custody",
    "tests.test_v2_rowsel_build_split_v2g.Evidence.test_the_reproducing_column_is_found": "class setup loads the pinned private registry port gsk.py 83183db4; requires qualified host custody",
    "tests.test_v2_rowsel_build_split_v2g.Summary.test_summary_of_a_v2gsplit_tree": "reads the pinned private NONRAW-SPLIT-A/GROUP-UNION-B owner decisions and relay; requires qualified host custody",
    "tests.test_v2_rowsel_build_split_v2g.Supplement.test_a_context_count_short_of_the_effective_folds_is_a_problem": "reads the pinned private NONRAW-SPLIT-A/GROUP-UNION-B owner decisions and relay; requires qualified host custody",
    "tests.test_v2_rowsel_build_split_v2g.Supplement.test_supplement_plan_build_and_summary": "reads the pinned private NONRAW-SPLIT-A/GROUP-UNION-B owner decisions and relay; requires qualified host custody",
    "tests.test_v2_rowsel_build_split_v2g.V2.test_the_v2_route_builds_the_bytes_c0b989b6_built": "loads the private-lane code-custody builder c0b989b6 (same bytes retained as lane/historical/v2_rowsel_build.py.c0b989b6...); requires qualified host custody",
    "tests.test_v2_rowsel_build_split_v2g.Evidence.test_a_reproducing_column_tied_to_the_target_is_refused_per_context_at_build": "class setup loads the pinned private registry port gsk.py 83183db4; requires qualified host custody",
    "tests.test_v2_rowsel_build_split_v2g.Evidence.test_the_census_reads_no_target_with_the_registry_port": "class setup loads the pinned private registry port gsk.py 83183db4; requires qualified host custody",
    "tests.test_v2_rowsel_build_split_v2g.Evidence.test_planning_never_decides_on_a_tie_with_the_registry_port_fields": "class setup loads the pinned private registry port gsk.py 83183db4; requires qualified host custody",
    "tests.test_split_defaults.BeforeAfter.test_v2": "compares against the private code-custody builder bytes fcfe3828 (same bytes retained in lane/historical/); requires qualified host custody",
    "tests.test_split_defaults.BeforeAfter.test_v2g": "compares against the private code-custody builder bytes fcfe3828 (same bytes retained in lane/historical/); requires qualified host custody",
    "tests.test_split_defaults.PortCheck.test_the_registry_replay_lineage_equals_fcfe3828s": "compares against the private code-custody builder bytes fcfe3828 (same bytes retained in lane/historical/); requires qualified host custody",
    "tests.test_split_defaults.RawDefaults.test_the_explicit_v1_replay_builds_28be7f98s_bytes": "compares against the private code-custody builder bytes 28be7f98 (same bytes retained in lane/historical/); requires qualified host custody",
    "tests.test_split_defaults.RawDefaults.test_the_v2_path_builds_28be7f98s_bytes": "compares against the private code-custody builder bytes 28be7f98 (same bytes retained in lane/historical/); requires qualified host custody",
    "tests.test_split_defaults.V2Path.test_v2_build_never_fits_on_every_row": "the v2 build reads the pinned private NONRAW-SPLIT-A owner decision; requires qualified host custody",
    "tests.test_split_defaults.V2Path.test_v2g_build_never_fits_on_every_row": "the v2 build reads the pinned private NONRAW-SPLIT-A owner decision; requires qualified host custody",
    "tests.test_v2_provenance.Resolver.test_custody_file_must_hash_to_its_name": "runs a scratch copy of tools/v2_provenance.py as a child process; the offline guard refuses every subprocess launch",
    "tests.test_v2_provenance.Resolver.test_f1_unresolved_gpu_runner_fails": "runs a scratch copy of tools/v2_provenance.py as a child process; the offline guard refuses every subprocess launch",
    "tests.test_v2_provenance.Resolver.test_f2_every_row_of_a_published_doc_is_scanned": "runs a scratch copy of tools/v2_provenance.py as a child process; the offline guard refuses every subprocess launch",
    "tests.test_v2_provenance.Resolver.test_f3_missing_hashes_are_unresolved": "runs a scratch copy of tools/v2_provenance.py as a child process; the offline guard refuses every subprocess launch",
    "tests.test_v2_provenance.Resolver.test_full_sha_resolves_and_passes": "runs a scratch copy of tools/v2_provenance.py as a child process; the offline guard refuses every subprocess launch",
    "tests.test_v2_provenance.Resolver.test_non_hex_64_chars_is_not_a_sha": "runs a scratch copy of tools/v2_provenance.py as a child process; the offline guard refuses every subprocess launch",
    "tests.test_v2_provenance.Resolver.test_publish_refused_before_any_client_when_unresolved": "runs a scratch copy of tools/v2_provenance.py as a child process; the offline guard refuses every subprocess launch",
    "tests.test_v2_provenance.Resolver.test_short_prefix_never_binds": "runs a scratch copy of tools/v2_provenance.py as a child process; the offline guard refuses every subprocess launch",
    "tests.test_v2_provenance.Resolver.test_summary_classification": "runs a scratch copy of tools/v2_provenance.py as a child process; the offline guard refuses every subprocess launch",
    "tests.test_v2_provenance.Resolver.test_unknown_full_sha_fails_closed": "runs a scratch copy of tools/v2_provenance.py as a child process; the offline guard refuses every subprocess launch",
    "tests.test_v2_rowsel_build_fallback.Pilot.test_the_real_record_and_the_pilot_plan": "reads the pinned private drifted-fallback lane record (rb.DRIFTED_RECORD) and plans from it; requires qualified host custody",
    "tests.test_v2_rowsel_build_fallback.SummaryCounts.test_group_annotation_dropped_counts_only_built_datasets": "summarize_v2 hashes the pinned private owner-decision/plan inputs (sha_file under the private roots); requires qualified host custody",
    "tests.test_target_linear.Control.test_2a5429bf_is_refused": "reads the real LINEAR-DETERMINED control context 2a5429bf from the private PUBLISH-rowsel-raw-nonnumeric-v2 tree; requires qualified host custody",
}
# Counted from the real offline collection of MODULES on the pinned handoff, not estimated.
EXPECTED_COLLECTED = 753
# The strict lane runs as disjoint module shards, each under its own `timeout 900s` in deploy/ci/fast-lane.sh (the budget
# is unchanged), then --combine proves the shards cover MODULES exactly and add up to the pinned totals above. Balanced by
# measured per-module time; each shard's collected count is pinned from the same real offline collection.
SHARDS = (
    (
        "tests.test_rowsel_raw_split_v2", "tests.test_v2_rowsel_raw", "tests.test_v2_rowsel_build_split_v2",
        "tests.test_v2_nonraw_split_v2g", "tests.test_t1raw", "tests.test_trainonly", "tests.test_payload_strip", "test_hwm_publish",
        "tests.test_v2_nonraw_target_linear",
    ),
    (
        "tests.test_v2_rowsel_build", "tests.test_v2_rowsel_build_split_v2g", "tests.test_v2_nonraw_split_v2", "tests.test_rowsel",
        "tests.test_s3io", "tests.test_v2_provenance", "tests.test_jtt1", "tests.test_admission_fail_closed", "test_gpu_manifest_cut",
        "tests.test_target_linear", "tests.test_v2_rowsel_raw_fold_isolation", "tests.test_v2_partial_refusal_manifest",
    ),
    (
        "tests.test_rowsel_split_v2_groups", "tests.test_v2_rowsel_raw_helper", "tests.test_v2_publish", "tests.test_nonnumeric",
        "tests.test_t1prep", "tests.test_split_defaults", "tests.test_quarantine", "test_manifest_binding", "tests.test_v2_rowsel_raw_v2g",
        "tests.test_v2_rowsel_build_fallback", "tests.test_v2_rowsel_build_fold_isolation", "tests.test_v2_build_local_fold_isolation",
    ),
)
EXPECTED_COLLECTED_BY_SHARD = (255, 262, 236)


def module_of(case):
    """The MODULES entry a host case belongs to (exactly one, matched on the dotted prefix)."""
    found = [m for m in MODULES if case == m or case.startswith(m + ".")]
    if len(found) != 1:
        raise ValueError(f"host case {case} maps to {len(found)} modules")
    return found[0]


def shard_host_cases(modules):
    return {case: why for case, why in HOST_CASES.items() if module_of(case) in modules}


def check_shards():
    flat = [m for shard in SHARDS for m in shard]
    if len(flat) != len(set(flat)) or set(flat) != set(MODULES) or len(flat) != len(MODULES):
        raise ValueError("SHARDS must partition MODULES exactly")
    if len(EXPECTED_COLLECTED_BY_SHARD) != len(SHARDS) or sum(EXPECTED_COLLECTED_BY_SHARD) != EXPECTED_COLLECTED:
        raise ValueError("per-shard expected counts must add up to EXPECTED_COLLECTED")
    if sum(len(shard_host_cases(shard)) for shard in SHARDS) != len(HOST_CASES):
        raise ValueError("every host case must belong to exactly one shard")


def case_ids(suite):
    for test in suite:
        if isinstance(test, unittest.TestSuite):
            yield from case_ids(test)
        else:
            yield test.id()


def make_audit(forbidden, probes, home):
    secret_roots = tuple(Path(home) / name for name in (".aws", ".ssh", ".config/gh", ".config/jopedime/secrets", ".claude"))

    def audit(event, values):
        if event == "open" and isinstance(values[0], (str, bytes)):
            lexical = Path(os.path.abspath(os.fsdecode(values[0])))

            def check(path):
                if any(path == prefix or path.is_relative_to(prefix) for prefix in secret_roots):
                    forbidden.append("credential-file read")
                    raise RuntimeError("offline fixtures forbid credential-file reads")
                if any(path == prefix or path.is_relative_to(prefix) for prefix in PRIVATE_INPUT_ROOTS):
                    forbidden.append("private-lane input read")
                    raise RuntimeError("offline fixtures forbid private-lane input reads")

            check(lexical)
            check(lexical.resolve())
        if event in ("subprocess.Popen", "os.posix_spawn") and values[0] == "/sbin/ldconfig" and values[1] == ["/sbin/ldconfig", "-p"]:
            probes.append({"event": event, "executable": values[0], "argv": values[1]})
            return
        if (event.startswith("socket.") or event.startswith("os.spawn")
                or event in ("subprocess.Popen", "os.system", "os.posix_spawn", "os.fork", "os.forkpty", "os.exec")):
            forbidden.append(event)
            raise RuntimeError("offline fixtures forbid " + event)

    return audit


@contextlib.contextmanager
def private_inputs_unavailable(unavailable):
    # Host decorators and optional ancestor pointers both query Path.exists.
    # Synthetic fixtures must see the same unavailable inputs on every host.
    exists = Path.exists

    def offline_exists(path):
        absolute = Path(os.path.abspath(os.fspath(path)))
        if any(absolute == prefix or absolute.is_relative_to(prefix) for prefix in PRIVATE_INPUT_ROOTS):
            unavailable.append(str(absolute))
            return False
        return exists(path)

    with mock.patch.object(Path, "exists", offline_exists):
        yield


def offline_suite(loader, modules, unavailable, host_cases=None):
    host_cases = HOST_CASES if host_cases is None else host_cases
    with private_inputs_unavailable(unavailable):
        suite = loader.loadTestsFromNames(modules)

    def cases(nested):
        for test in nested:
            if isinstance(test, unittest.TestSuite):
                yield from cases(test)
            else:
                yield test

    tests = list(cases(suite))
    names = [test.id() for test in tests]
    if len(names) != len(set(names)) or not set(host_cases) <= set(names):
        raise ValueError("offline collection lacks unique declared host cases")
    selected = unittest.TestSuite(test for test in tests if test.id() not in host_cases)
    return selected, names


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard", help="I/N: run only module shard I of the pinned N (CI runs each under its own timeout)")
    parser.add_argument("--combine", action="store_true", help="verify OUTPUT/shard-1..N receipts and write OUTPUT/receipt.json")
    args = parser.parse_args()
    check_shards()
    if args.combine:
        return combine(args.output.resolve())
    modules, host_cases, expected, shard = MODULES, HOST_CASES, EXPECTED_COLLECTED, None
    if args.shard:
        index, _, total = args.shard.partition("/")
        if not (index.isdigit() and total.isdigit() and int(total) == len(SHARDS) and 1 <= int(index) <= len(SHARDS)):
            raise SystemExit(f"--shard must be I/{len(SHARDS)}")
        shard = int(index)
        modules = SHARDS[shard - 1]
        host_cases, expected = shard_host_cases(modules), EXPECTED_COLLECTED_BY_SHARD[shard - 1]
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    scratch = out / "scratch"
    scratch.mkdir()
    os.environ["TMPDIR"] = str(scratch)
    for name in list(os.environ):
        if name.startswith(("AWS_", "JOPE_AWS_", "GH_", "GITHUB_")):
            os.environ.pop(name)
    os.environ.update({"AWS_EC2_METADATA_DISABLED": "true", "AWS_CONFIG_FILE": "/dev/null",
                       "AWS_SHARED_CREDENTIALS_FILE": "/dev/null", "AWS_ACCESS_KEY_ID": "offline-fixture",
                       "AWS_SECRET_ACCESS_KEY": "offline-fixture", "AWS_DEFAULT_REGION": "us-east-2",
                       "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                       "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": ""})
    sys.dont_write_bytecode = True
    cpus = sorted(os.sched_getaffinity(0))[:4]
    os.sched_setaffinity(0, cpus)
    limit = 16 * 1024**3
    old_soft, old_hard = resource.getrlimit(resource.RLIMIT_AS)
    memory = min(limit, old_hard) if old_hard != resource.RLIM_INFINITY else limit
    resource.setrlimit(resource.RLIMIT_AS, (memory, old_hard))
    forbidden, probes = [], []
    sys.addaudithook(make_audit(forbidden, probes, Path.home()))
    # urllib3 otherwise probes IPv6 by creating a socket during boto3 import.
    # Offline fixtures have no network capability; keep every socket event denied.
    socket.has_ipv6 = False
    custody = Path(__file__).with_name("lane")
    before = verify(custody)
    def adapter_pins():
        return {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                for name in ("hwm_publish.py", "test_hwm_publish.py", "gpu_manifest_cut.py", "test_gpu_manifest_cut.py",
                             "manifest_binding.py", "test_manifest_binding.py", "run_lane_offline.py")}
    adapter_before = adapter_pins()
    root = custody / "src/dl_training_v1"
    os.chdir(root)
    sys.path.insert(0, str(root))
    versions = {name: importlib.metadata.version(name) for name in ("numpy", "boto3", "botocore")}
    unavailable = []
    suite, collected_names = offline_suite(unittest.defaultTestLoader, modules, unavailable, host_cases)
    names = list(case_ids(suite))
    start = datetime.datetime.now(datetime.timezone.utc).isoformat()
    tick = time.monotonic()
    with private_inputs_unavailable(unavailable), (out / "unittest.log").open("x") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    after = verify(custody)
    adapter_after = adapter_pins()
    report = {"schema": "jope.dl-offline-fixtures/v2", "start_utc": start,
              "end_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "seconds": time.monotonic() - tick,
              "argv": sys.argv, "python": sys.executable, "cwd": str(root), "dependencies": versions,
              "handoff_sha256": before["handoff_sha256"], "source_unchanged": before == after and adapter_before == adapter_after,
              "adapter_source_sha256": adapter_before,
              "shard": f"{shard}/{len(SHARDS)}" if shard else None, "expected_collected": expected,
              "modules": list(modules), "case_ids": names, "collection_case_ids": collected_names,
              "collected": len(collected_names), "executed": result.testsRun,
              "excluded": len(host_cases), "excluded_cases": host_cases,
              "unavailable_host_input_probes": sorted(set(unavailable)),
              "failed": len(result.failures) + len(result.errors), "skipped": len(result.skipped),
              "expected_failures": len(result.expectedFailures), "unexpected_successes": len(result.unexpectedSuccesses),
              "forbidden_attempts": forbidden, "library_cache_probes": probes,
              "offline_network": {"ipv6_capability_probe": "disabled; every socket audit event refused"},
              "offline_private_inputs": "unavailable without stat during collection and execution; private reads refused",
              "resource_limits": {"affinity_cpus": cpus, "address_space_bytes": memory, "threads": 1},
              "log_sha256": hashlib.sha256((out / "unittest.log").read_bytes()).hexdigest(),
              "scope": "audited synthetic offline CPU fixtures; no live corpus, pilot, training, GPU, Lambda, S3 or runtime adoption"}
    with (out / "receipt.json").open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({key: report[key] for key in ("collected", "executed", "excluded", "failed", "skipped", "seconds", "handoff_sha256", "source_unchanged")}))
    return not (result.wasSuccessful() and not forbidden and before == after and adapter_before == adapter_after and len(collected_names) == expected
                and len(names) == result.testsRun == expected - len(host_cases)
                and set(collected_names) - set(names) == set(host_cases)
                and not result.skipped and not result.expectedFailures and not result.unexpectedSuccesses)


def combine(out):
    """Fail unless OUTPUT/shard-1..N are the pinned shards, each clean, and together the exact full strict lane."""
    receipts, errors = [], []
    for index in range(1, len(SHARDS) + 1):
        shard_dir = out / f"shard-{index}"
        try:
            receipt = json.loads((shard_dir / "receipt.json").read_bytes())
            log_ok = hashlib.sha256((shard_dir / "unittest.log").read_bytes()).hexdigest() == receipt["log_sha256"]
        except (OSError, ValueError, KeyError) as err:
            errors.append(f"shard {index}: unreadable receipt or log ({type(err).__name__})")
            continue
        receipts.append(receipt)
        if not log_ok:
            errors.append(f"shard {index}: unittest.log does not match its receipt")
        if receipt.get("shard") != f"{index}/{len(SHARDS)}" or receipt.get("modules") != list(SHARDS[index - 1]):
            errors.append(f"shard {index}: not the pinned shard")
        if (receipt["collected"] != EXPECTED_COLLECTED_BY_SHARD[index - 1] or receipt["failed"] or receipt["skipped"]
                or receipt["expected_failures"] or receipt["unexpected_successes"] or receipt["forbidden_attempts"]
                or not receipt["source_unchanged"] or receipt["executed"] != receipt["collected"] - receipt["excluded"]):
            errors.append(f"shard {index}: not clean or not the pinned count")
    if len(receipts) == len(SHARDS):
        collected = [case for r in receipts for case in r["collection_case_ids"]]
        executed = [case for r in receipts for case in r["case_ids"]]
        excluded = {case for r in receipts for case in r["excluded_cases"]}
        if len(collected) != len(set(collected)) or len(collected) != EXPECTED_COLLECTED:
            errors.append(f"union collected {len(collected)} != {EXPECTED_COLLECTED} unique identities")
        if len(executed) != EXPECTED_COLLECTED - len(HOST_CASES) or sum(r["executed"] for r in receipts) != len(executed):
            errors.append(f"union executed {len(executed)} != {EXPECTED_COLLECTED - len(HOST_CASES)}")
        if excluded != set(HOST_CASES) or set(collected) - set(executed) != set(HOST_CASES):
            errors.append("union exclusions are not exactly HOST_CASES")
        for key in ("handoff_sha256", "adapter_source_sha256", "dependencies", "python"):
            if len({json.dumps(r[key], sort_keys=True) for r in receipts}) != 1:
                errors.append(f"shards disagree on {key}")
    report = {"schema": "jope.dl-offline-fixtures-combined/v1", "shards": len(SHARDS), "verdict": "FAIL" if errors else "PASS",
              "errors": errors, "collected": sum(r["collected"] for r in receipts), "executed": sum(r["executed"] for r in receipts),
              "excluded": sum(r["excluded"] for r in receipts), "failed": sum(r["failed"] for r in receipts),
              "skipped": sum(r["skipped"] for r in receipts),
              "seconds_by_shard": [round(r["seconds"], 1) for r in receipts],
              "handoff_sha256": receipts[0]["handoff_sha256"] if receipts else None,
              "source_unchanged": bool(receipts) and all(r["source_unchanged"] for r in receipts),
              "shard_receipt_sha256": [hashlib.sha256((out / f"shard-{i}" / "receipt.json").read_bytes()).hexdigest()
                                       for i in range(1, len(receipts) + 1)]}
    with (out / "receipt.json").open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({key: report[key] for key in ("verdict", "collected", "executed", "excluded", "failed", "skipped", "seconds_by_shard", "errors")}))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
