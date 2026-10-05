"""Compile real S1 source mutants and require assertion failures, never compile errors.

Only generated single-file mutants live in the checkout's reused target/. The
tracked source, Cargo manifests, locks and historical campaign copies stay intact.
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent

MUTANTS = {
    "island-engine": {
        "fitness_test_ratio": (
            "0.80 * mean +",
            "0.50 * test_ratio.unwrap_or(f64::INFINITY).max(1e-300).ln() + 0.30 * mean +",
            "selection_is_blind_to_the_official_test",
        ),
        "cvgated_test_rmse": (
            ".cmp(&a.cv_gate_passed)\n",
            ".cmp(&a.cv_gate_passed)\n                .then_with(|| cmp_f64(a.test_rmse.unwrap_or(f64::INFINITY), b.test_rmse.unwrap_or(f64::INFINITY)))\n",
            "selection_is_blind_to_the_official_test",
        ),
        "cvstable_test_tiebreak": (
            "Policy::CvStable => cmp_f64(a.cv_geomean_rmse_ratio, b.cv_geomean_rmse_ratio)\n                .then_with(|| cmp_f64(a.cv_log_ratio_sd, b.cv_log_ratio_sd))\n",
            "Policy::CvStable => cmp_f64(a.cv_geomean_rmse_ratio, b.cv_geomean_rmse_ratio)\n                .then_with(|| cmp_f64(a.cv_log_ratio_sd, b.cv_log_ratio_sd))\n                .then_with(|| cmp_f64(a.test_rmse.unwrap_or(f64::INFINITY), b.test_rmse.unwrap_or(f64::INFINITY)))\n",
            "selection_is_blind_to_the_official_test",
        ),
        "semantic_test_predictions": (
            "let prediction_values = oof_predictions;",
            "let mut prediction_values = oof_predictions;\n        prediction_values.extend(_test_predictions);",
            "selection_is_blind_to_the_official_test",
        ),
        "balanced_test_ratio": (
            "a.fitness\n            .total_cmp(&b.fitness)",
            "(a.fitness + a.test_rmse_ratio.unwrap_or(f64::INFINITY))\n            .total_cmp(&(b.fitness + b.test_rmse_ratio.unwrap_or(f64::INFINITY)))",
            "selection_is_blind_to_the_official_test",
        ),
        "historical_semantic_cache": (
            "if row.selection_protocol != SELECTION_PROTOCOL {",
            "if false {",
            "stale_resume_and_semantic_cache_are_refused",
        ),
        "test_case_indices": (
            "|| !sample.test_case_indices.is_empty()",
            "|| false",
            "test_cases_and_test_wide_projections_are_refused",
        ),
        "diagnostic_availability_veto": (
            "let gate = geo <= 0.90 && wins >= 5;",
            'if test_rmse.is_none() { bail!("test diagnostic required") }\n        let gate = geo <= 0.90 && wins >= 5;',
            "missing_failed_or_malformed_test_diagnostics_cannot_censor_cv",
        ),
        "duplicate_live_delivery": (
            "if pending.is_none() {",
            "if false {",
            "legacy_failed_live_admission_cannot_mint_s1_history",
        ),
        "legacy_failed_live_admission": (
            'if pack["schema"] != "jope.s1-invocation/v1"',
            'if pack["schema"] != "jope.s1-invocation/v1"',
            "legacy_failed_live_admission_cannot_mint_s1_history",
        ),
        "legacy_failed_resume": (
            "validate_history(&value, &self.context)?;",
            "// legacy failed rows admitted",
            "legacy_failed_resume_cannot_change_seen_programs",
        ),
    },
    "lambda-worker": {
        "scaler_scored_rows": (
            "let s = fit_scaler(x, y)?;",
            "let mut xa = x.clone(); xa.extend(vx.iter().cloned());\n    let mut ya = y.to_vec(); ya.extend_from_slice(vy);\n    let s = fit_scaler(&xa, &ya)?;",
            "fold_fit_ignores_its_validation_rows",
        ),
        "prescale_all_rows": (
            "let mut cv = Vec::with_capacity(d.k);",
            "let mut d = d;\n    let mut xa = d.train_x.clone(); xa.extend(d.test_x.iter().cloned());\n    let mut ya = d.train_y.clone(); ya.extend_from_slice(&d.test_y);\n    let s = fit_scaler(&xa, &ya)?;\n    d.train_x = scale_rows(&d.train_x, &s); d.test_x = scale_rows(&d.test_x, &s);\n    d.train_y = scale_y(&d.train_y, &s); d.test_y = scale_y(&d.test_y, &s);\n    let mut cv = Vec::with_capacity(d.k);",
            "evaluation_fits_ignore_outer_test_and_validation_rows",
        ),
        "constant_span": (
            "if h - l > 0.0 { h - l } else { 1.0 }",
            "if h - l > 0.0 { h - l } else { 1e-12 }",
            "scaler_is_train_rows_only_and_constant_columns_divide_by_one",
        ),
        "timeout_exceeds_reservation": (
            "!(100..=90000).contains(&r.candidate_timeout_ms)",
            "!(100..=120000).contains(&r.candidate_timeout_ms)",
            "request_timeout_matches_the_ninety_second_reservation",
        ),
    },
}


def intended_rust_failure(output, assertion, exit_code):
    """A named failure alone does not prove the intended assertion was reached."""
    section = re.search(
        rf"---- tests::{re.escape(assertion)} stdout ----\n(.*?)\n\nfailures:",
        output,
        re.DOTALL,
    )
    if not section or exit_code == 0:
        return False
    panic = section.group(1)
    markers = {
        "missing_failed_or_malformed_test_diagnostics_cannot_censor_cv": [
            "S1_ASSERT_DIAGNOSTIC_ADMISSION"
        ],
        "legacy_failed_live_admission_cannot_mint_s1_history": [
            "S1_ASSERT_LEGACY_LIVE_ADMISSION",
            "S1_ASSERT_REPLAY_REFUSED",
        ],
    }.get(assertion, [])
    return (
        "running 1 test\n" in output
        and f"test tests::{assertion} ... FAILED" in output
        and "test result: FAILED. 0 passed; 1 failed; 0 ignored;" in output
        and (
            re.search(r"assertion(?: .*?)? failed", panic) is not None
            or any(marker in panic for marker in markers)
        )
    )


def panic_is_in_intended_test(output, mutant, source, assertion):
    declaration = re.search(rf"(?m)^    fn {re.escape(assertion)}\(", source)
    if declaration is None:
        return False
    next_function = re.search(
        r"(?m)^    (?:#\[test\]|fn )", source[declaration.end() :]
    )
    end = declaration.end() + next_function.start() if next_function else len(source)
    first_line = source[: declaration.start()].count("\n") + 1
    last_line = source[:end].count("\n") + 1
    sites = re.findall(r"panicked at ([^\n]+):([0-9]+):[0-9]+:", output)
    return (
        len(sites) == 1
        and Path(sites[0][0]).resolve() == mutant.resolve()
        and first_line <= int(sites[0][1]) < last_line
    )


def intended_python_failure(output, exit_code):
    return (
        exit_code != 0
        and "FAIL: test_fixed_equal_ignores_outer_test_for_every_stage" in output
        and "AssertionError" in output
        and "Ran 1 test" in output
        and "FAILED (failures=1)" in output
        and "ERROR:" not in output
    )


def check_failure_recognition():
    # Reproduce the previous false kill: a FAILED label with an unrelated fixture panic.
    name = "legacy_failed_resume_cannot_change_seen_programs"
    template = f"running 1 test\ntest tests::{name} ... FAILED\n\nfailures:\n\n---- tests::{name} stdout ----\n\nthread 'x' panicked at fixture.rs:1187:34:\n{{panic}}\n\nfailures:\n    tests::{name}\n\ntest result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 10 filtered out\n"
    assert not intended_rust_failure(
        template.format(panic="index out of bounds: len is 0 but index is 0"), name, 101
    )
    assert intended_rust_failure(
        template.format(panic="assertion failed: e.load_resume(&path).is_err()"),
        name,
        101,
    )
    assert not intended_python_failure(
        "ERROR: test_fixed_equal_ignores_outer_test_for_every_stage\nAssertionError\nRan 1 test\nFAILED (errors=1)",
        1,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    check_failure_recognition()
    target = Path(os.environ.get("CARGO_TARGET_DIR", ROOT / "target")).resolve()
    generated = target / "s1-mutants"
    generated.mkdir(parents=True, exist_ok=True)
    wrapper = generated / "rustc-wrapper.py"
    wrapper.write_text("""#!/usr/bin/env python3
import os,sys
from pathlib import Path
args=sys.argv[1:]
original=Path(os.environ['S1_ORIGINAL']).resolve()
args=[os.environ['S1_MUTANT'] if Path(a).resolve()==original else a for a in args]
os.execv(args[0],args)
""")
    wrapper.chmod(0o700)
    env = os.environ | {
        "CARGO_INCREMENTAL": "0",
        "CARGO_PROFILE_DEV_DEBUG": "line-tables-only",
        "CARGO_BUILD_JOBS": "4",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    }
    env.pop("RUSTC_WRAPPER", None)
    results = {}
    for crate, mutations in MUTANTS.items():
        manifest = HERE / "rust" / crate / "Cargo.toml"
        source = manifest.parent / "src" / "main.rs"
        text = source.read_text()
        baseline = subprocess.run(
            [
                "cargo",
                "test",
                "--offline",
                "--locked",
                "--manifest-path",
                str(manifest),
                "--target-dir",
                str(target),
                "--",
                "--test-threads=2",
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=900,
        )
        (args.out / f"{crate}-baseline.log").write_text(
            baseline.stdout + baseline.stderr
        )
        if (
            baseline.returncode != 0
            or "test result: ok." not in baseline.stdout
            or re.search(r"running [1-9][0-9]* tests", baseline.stdout) is None
        ):
            raise RuntimeError(f"{crate} baseline did not pass")
        for name, (old, new, assertion) in mutations.items():
            if text.count(old) != 1:
                raise RuntimeError(f"mutant anchor changed: {name}")
            mutant = generated / f"{name}.rs"
            mutated = text.replace(old, new)
            if name == "legacy_failed_live_admission":
                start = mutated.index(
                    '        if pack["schema"] != "jope.s1-invocation/v1"'
                )
                end = mutated.index('        if record["context_sha256"]', start)
                mutated = (
                    mutated[:start]
                    + "        // mutant: legacy live records admitted\n"
                    + mutated[end:]
                )
            if name == "test_case_indices":
                guard = 'bail!("S1: test case indices are not admitted")'
                if mutated.count(guard) != 1:
                    raise RuntimeError("second test-case guard changed")
                mutated = mutated.replace(guard, "()")
            mutant.write_text(mutated)
            mutation_env = env | {
                "RUSTC_WRAPPER": str(wrapper),
                "S1_ORIGINAL": str(source),
                "S1_MUTANT": str(mutant),
            }
            command = [
                "cargo",
                "rustc",
                "--offline",
                "--locked",
                "--manifest-path",
                str(manifest),
                "--target-dir",
                str(target),
                "--tests",
                "--message-format=json",
                "--",
                "--cfg",
                f"s1_mutant_{name}",
            ]
            build = subprocess.run(
                command, env=mutation_env, capture_output=True, text=True, timeout=900
            )
            (args.out / f"{name}-build.log").write_text(build.stdout + build.stderr)
            if build.returncode != 0:
                raise RuntimeError(f"{name}: compile errors cannot kill a mutant")
            artifacts = [
                json.loads(line)
                for line in build.stdout.splitlines()
                if line.startswith("{")
            ]
            test_artifacts = [
                a
                for a in artifacts
                if a.get("reason") == "compiler-artifact"
                and a.get("executable")
                and a.get("profile", {}).get("test")
                and Path(a["target"]["src_path"]) == source
            ]
            if len(test_artifacts) != 1 or test_artifacts[0].get("fresh") is not False:
                raise RuntimeError(
                    f"{name}: expected one newly compiled test executable"
                )
            run = subprocess.run(
                [
                    test_artifacts[0]["executable"],
                    "--exact",
                    f"tests::{assertion}",
                    "--test-threads=1",
                ],
                env=env,
                capture_output=True,
                text=True,
                timeout=60,
            )
            (args.out / f"{name}-test.log").write_text(run.stdout + run.stderr)
            killed = intended_rust_failure(
                run.stdout + run.stderr, assertion, run.returncode
            ) and panic_is_in_intended_test(
                run.stdout + run.stderr, mutant, mutated, assertion
            )
            results[name] = {
                "compiled": True,
                "test_exit": run.returncode,
                "killed": killed,
                "assertion": assertion,
                "executed_tests": 1,
                "panic_site_in_intended_test": killed,
                "filter": "exact intended test; other tests excluded from this mutation experiment",
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "mutant_sha256": hashlib.sha256(mutant.read_bytes()).hexdigest(),
            }
            if not killed:
                raise RuntimeError(
                    f"{name}: mutant survived or failed outside the intended assertion"
                )
    # Channel 4 mutants run the real shipped continuation function with its tests.
    source = HERE / "s1_budget.py"
    text = source.read_text()
    for name, replacement in {
        "budget_test_margin": "return [view for view, hwm in rows_by_view if not hwm.get('beats_with_margin', False)]",
        "budget_test_ratio": "return [view for view, hwm in rows_by_view if hwm.get('ratio', 0) < 0.95]",
        "budget_unbounded": None,
    }.items():
        mutated = (
            text.replace("not 1 <= stage <= GENERATIONS", "not 1 <= stage")
            if replacement is None
            else text.replace("return [view for view, _ in rows_by_view]", replacement)
        )
        if mutated == text:
            raise RuntimeError("channel-4 mutant anchor changed")
        runner = "import types,sys,unittest; m=types.ModuleType('s1_budget'); exec(compile(sys.stdin.read(), 's1_budget.py', 'exec'),m.__dict__); sys.modules['s1_budget']=m; import test_s1_budget; r=unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite([test_s1_budget.BudgetTests('test_fixed_equal_ignores_outer_test_for_every_stage')])); sys.exit(not r.wasSuccessful())"
        run = subprocess.run(
            [sys.executable, "-c", runner],
            input=mutated,
            cwd=HERE,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )
        (args.out / f"{name}-test.log").write_text(run.stdout + run.stderr)
        killed = intended_python_failure(run.stdout + run.stderr, run.returncode)
        results[name] = {
            "compiled": True,
            "test_exit": run.returncode,
            "killed": killed,
        }
        if not killed:
            raise RuntimeError(f"{name}: continuation mutant survived")
    proof = {
        "schema": "jope.s1-mutant-proof/v2",
        "failure_recognition_controls": 3,
        "unrelated_panics_or_python_errors_counted_as_kills": 0,
        "results": results,
        "killed": len(results),
        "source_scope": "production scorer, comparator, semantic cache, evaluator and continuation; synthetic fixtures",
        "execution_admitted": False,
    }
    (args.out / "mutants.json").write_text(
        json.dumps(proof, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"killed": len(results), "failed": 0}))


if __name__ == "__main__":
    main()
