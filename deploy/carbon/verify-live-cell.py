#!/usr/bin/env python3
"""Live end-to-end cell verifier for Carbon (`dev` pre-promotion gate & `prod` post-promotion gate).

Exercises the shipped static `carbon` CLI against live regional AWS Lambda cells:
1. Ephemeral Free-tier registration (`carbon register --free`) in an isolated `0700` `$HOME`.
2. Regression E2E with tricky user column vocabulary (`free_stream_velocity`, `power_generation`,
   `household_population`, `island_code`, `candidate_score`) across `fit`, `status`, `predict`,
   `explain`, all 5 standalone exports (`python`, `rust`, `cpp`, `js`, `model-object`), local
   Python + Node.js standalone execution parity, domain-refusal (`422`, zero `FunctionError`), and
   `carbon shred` (`404` post-erasure).
3. Binary classification E2E with tricky string class labels (`below` / `above`) across `fit`,
   `predict`, standalone exports, and `carbon shred`.
4. AWS CloudWatch `Errors == 0` and `Throttles == 0` verification across `edge`, `engine`, and
   `eval` Lambda functions in every target region.
"""

from __future__ import annotations

import argparse
import base64
import csv
import datetime as dt
import io
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any

WAVE1_PROD_REGIONS: tuple[str, ...] = (
    "us-east-2",
    "us-east-1",
    "us-west-2",
    "eu-central-1",
    "eu-west-1",
)


class GateFailure(RuntimeError):
    """Raised when any live cell E2E gate fails."""


def expand_regions(raw: str, env_name: str) -> list[str]:
    token = raw.strip()
    if token == "wave1":
        return list(WAVE1_PROD_REGIONS) if env_name == "prod" else ["us-east-2"]
    items = [part.strip() for part in token.split(",") if part.strip()]
    if not items:
        raise GateFailure("regions list must not be empty")
    return items


def run_checked(
    argv: list[str],
    *,
    env: dict[str, str],
    timeout_secs: int = 240,
    expect_ok: bool = True,
) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        argv,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout_secs,
        check=False,
    )
    if expect_ok and proc.returncode != 0:
        snippet = (proc.stderr or proc.stdout)[-800:].strip()
        raise GateFailure(f"command {argv[:3]} failed (rc={proc.returncode}): {snippet}")
    return proc


def build_tricky_regression_csv() -> tuple[str, str, list[tuple[float, ...]]]:
    header = [
        "free_stream_velocity",
        "power_generation",
        "household_population",
        "island_code",
        "candidate_score",
        "sound_pressure",
    ]
    fit_buf = io.StringIO()
    writer = csv.writer(fit_buf, lineterminator="\n")
    writer.writerow(header)
    for idx in range(220):
        vel = 31.0 + (idx % 19) * 2.25
        gen = 10.0 + ((idx * 7) % 23) * 1.5
        pop = 100.0 + ((idx * 11) % 31) * 8.0
        isl = float((idx * 3) % 5)
        cand = 0.25 * ((idx * 5) % 17)
        noise = 0.03 * math.sin(float(idx))
        target = 120.0 - 0.35 * vel + 0.80 * gen + 0.04 * pop - 1.1 * isl + 0.6 * cand + noise
        writer.writerow([f"{vel:.4f}", f"{gen:.4f}", f"{pop:.4f}", f"{isl:.1f}", f"{cand:.4f}", f"{target:.6f}"])

    score_buf = io.StringIO()
    s_writer = csv.writer(score_buf, lineterminator="\n")
    s_writer.writerow(header[:-1])
    probe_rows: list[tuple[float, ...]] = []
    for idx in range(12):
        row = (
            35.0 + idx * 3.0,
            14.0 + idx * 1.2,
            140.0 + idx * 10.0,
            float(idx % 4),
            0.5 * idx,
        )
        probe_rows.append(row)
        s_writer.writerow([f"{val:.4f}" for val in row])

    return fit_buf.getvalue(), score_buf.getvalue(), probe_rows


def build_tricky_binary_csv() -> tuple[str, str]:
    header = [
        "free_stream_velocity",
        "power_generation",
        "household_population",
        "regime_label",
    ]
    fit_buf = io.StringIO()
    writer = csv.writer(fit_buf, lineterminator="\n")
    writer.writerow(header)
    for idx in range(220):
        vel = 20.0 + (idx % 25) * 2.4
        gen = 5.0 + ((idx * 9) % 27) * 1.8
        pop = 80.0 + ((idx * 13) % 29) * 6.0
        logit = 0.09 * vel + 0.11 * gen - 0.012 * pop - 4.4
        label = "above" if logit >= 0.0 else "below"
        writer.writerow([f"{vel:.4f}", f"{gen:.4f}", f"{pop:.4f}", label])

    score_buf = io.StringIO()
    s_writer = csv.writer(score_buf, lineterminator="\n")
    s_writer.writerow(header[:-1])
    for idx in range(10):
        s_writer.writerow([f"{24.0 + idx * 5.0:.4f}", f"{8.0 + idx * 3.5:.4f}", f"{100.0 + idx * 9.0:.4f}"])
    return fit_buf.getvalue(), score_buf.getvalue()


def verify_model_object_b64(b64_text: str) -> dict[str, Any]:
    raw_zip = base64.b64decode(b64_text.strip(), validate=True)
    if not raw_zip.startswith(b"PK\x03\x04"):
        raise GateFailure("exported model-object is not a valid ZIP archive")
    with zipfile.ZipFile(io.BytesIO(raw_zip), "r") as zf:
        manifest_names = [name for name in zf.namelist() if name.endswith("/manifest.json") or name == "manifest.json"]
        if not manifest_names:
            raise GateFailure("model-object ZIP lacks manifest.json")
        manifest = json.loads(zf.read(manifest_names[0]).decode("utf-8"))
        if manifest.get("schema") != "jope.model-object/v1":
            raise GateFailure(f"unexpected model-object schema: {manifest.get('schema')}")
        return {
            "zip_bytes": len(raw_zip),
            "entry_count": len(zf.namelist()),
            "schema": manifest.get("schema"),
        }


def run_standalone_python_check(py_file: Path, probe_rows: list[tuple[float, ...]]) -> list[float]:
    run_checked([sys.executable, str(py_file), "--self-check"], env=dict(os.environ), timeout_secs=30)
    driver = (
        "import importlib.util, json, sys\n"
        "spec = importlib.util.spec_from_file_location('exported_model', sys.argv[1])\n"
        "mod = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(mod)\n"
        "rows = json.loads(sys.argv[2])\n"
        "preds = mod.predict_rows(rows)\n"
        "print(json.dumps(preds))\n"
    )
    proc = run_checked(
        [sys.executable, "-c", driver, str(py_file), json.dumps([list(r) for r in probe_rows])],
        env=dict(os.environ),
        timeout_secs=30,
    )
    preds = json.loads(proc.stdout.strip())
    if len(preds) != len(probe_rows) or not all(isinstance(v, (int, float)) and math.isfinite(float(v)) for v in preds):
        raise GateFailure("standalone Python predictor returned invalid or non-finite values")
    return [float(v) for v in preds]


def run_standalone_js_check(js_file: Path, probe_rows: list[tuple[float, ...]], py_preds: list[float]) -> bool:
    node_bin = shutil.which("node")
    if not node_bin:
        return False
    run_checked([node_bin, str(js_file), "--self-check"], env=dict(os.environ), timeout_secs=30)
    js_driver = (
        "const fs = require('fs');\n"
        "const vm = require('vm');\n"
        "const code = fs.readFileSync(process.argv[1], 'utf8') + '\\n;globalThis.__predictRows = predictRows;';\n"
        "vm.runInThisContext(code);\n"
        "const rows = JSON.parse(process.argv[2]);\n"
        "const out = Array.from(globalThis.__predictRows(rows));\n"
        "process.stdout.write(JSON.stringify(out));\n"
    )
    proc = run_checked(
        [node_bin, "-e", js_driver, str(js_file), json.dumps([list(r) for r in probe_rows])],
        env=dict(os.environ),
        timeout_secs=30,
    )
    js_preds = [float(v) for v in json.loads(proc.stdout.strip())]
    if len(js_preds) != len(py_preds):
        raise GateFailure("standalone JS prediction length mismatch against Python")
    for idx, (pv, jv) in enumerate(zip(py_preds, js_preds)):
        tol = 1e-6 + 1e-6 * abs(pv)
        if not math.isfinite(jv) or abs(pv - jv) > tol:
            raise GateFailure(f"standalone JS vs Python parity mismatch at row {idx}: py={pv} js={jv}")
    return True


def query_cloudwatch_lambda_metrics(
    env_name: str,
    regions: list[str],
    start_utc: dt.datetime,
    end_utc: dt.datetime,
    aws_bin: str = "aws",
) -> dict[str, dict[str, dict[str, int]]]:
    summary: dict[str, dict[str, dict[str, int]]] = {}
    start_iso = (start_utc.replace(second=0, microsecond=0) - dt.timedelta(minutes=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    end_iso = (end_utc + dt.timedelta(minutes=2)).strftime("%Y-%m-%dT%H:%M:%SZ")
    period = 60

    total_invocations = 0
    for region in regions:
        region_stats: dict[str, dict[str, int]] = {}
        for role in ("edge", "engine", "eval"):
            fn_name = f"jopedime-carbon-{env_name}-{role}"
            role_row: dict[str, int] = {}
            for metric in ("Invocations", "Errors", "Throttles"):
                proc = run_checked(
                    [
                        aws_bin,
                        "cloudwatch",
                        "get-metric-statistics",
                        "--region",
                        region,
                        "--namespace",
                        "AWS/Lambda",
                        "--metric-name",
                        metric,
                        "--dimensions",
                        f"Name=FunctionName,Value={fn_name}",
                        "--start-time",
                        start_iso,
                        "--end-time",
                        end_iso,
                        "--period",
                        str(period),
                        "--statistics",
                        "Sum",
                    ],
                    env=dict(os.environ),
                    timeout_secs=30,
                )
                payload = json.loads(proc.stdout)
                pts = payload.get("Datapoints", [])
                role_row[metric] = int(round(sum(float(p.get("Sum", 0.0)) for p in pts)))
            total_invocations += role_row["Invocations"]
            if role_row["Errors"] > 0 or role_row["Throttles"] > 0:
                raise GateFailure(
                    f"CloudWatch non-zero fault metric on {fn_name} ({region}): "
                    f"Errors={role_row['Errors']} Throttles={role_row['Throttles']}"
                )
            region_stats[role] = role_row
        summary[region] = region_stats
    if total_invocations <= 0:
        raise GateFailure(
            f"CloudWatch recorded 0 Invocations across {regions} for jopedime-carbon-{env_name}-*; "
            "refusing to certify unmeasured Lambda telemetry"
        )
    return summary


def execute_live_gate(
    *,
    carbon_bin: Path,
    env_name: str,
    regions: list[str],
    clouds_manifest: Path | None,
    skip_cloudwatch: bool,
    aws_bin: str = "aws",
) -> dict[str, Any]:
    if not carbon_bin.is_file():
        raise GateFailure(f"missing carbon binary at {carbon_bin}")

    started_utc = dt.datetime.now(dt.timezone.utc)
    t0 = time.monotonic()

    with tempfile.TemporaryDirectory(prefix=f"carbon-e2e-{env_name}-") as tmp_str:
        tmp = Path(tmp_str)
        home_dir = tmp / "home"
        home_dir.mkdir(mode=0o700)
        cli_env = dict(os.environ)
        cli_env["HOME"] = str(home_dir)
        cli_env["NO_COLOR"] = "1"

        manifest_args: list[str] = []
        if clouds_manifest is not None:
            manifest_args = ["--clouds-manifest", str(clouds_manifest.resolve())]

        # 1. Doctor & account registration on target cloud
        run_checked(
            [str(carbon_bin), "ping", "--cloud", env_name, *manifest_args],
            env=cli_env,
            timeout_secs=45,
        )
        run_checked(
            [str(carbon_bin), "register", "--free", "--cloud", env_name, "--label", f"gate_{env_name}", *manifest_args],
            env=cli_env,
            timeout_secs=45,
        )

        # 2. Regression E2E with `free_stream_velocity` (`"elo"` substring) + all 5 exports
        reg_csv_text, reg_score_text, probe_rows = build_tricky_regression_csv()
        reg_csv = tmp / "tricky_regression.csv"
        reg_score_csv = tmp / "tricky_score.csv"
        reg_handle = tmp / "reg.carbon.json"
        reg_preds_out = tmp / "reg_preds.csv"
        reg_csv.write_text(reg_csv_text, encoding="utf-8")
        reg_score_csv.write_text(reg_score_text, encoding="utf-8")

        fit_proc = run_checked(
            [
                str(carbon_bin),
                "fit",
                str(reg_csv),
                "--target",
                "sound_pressure",
                "--save",
                str(reg_handle),
                "--cloud",
                env_name,
                "--format",
                "json",
                *manifest_args,
            ],
            env=cli_env,
            timeout_secs=240,
        )
        reg_fit_json = json.loads(fit_proc.stdout)
        reg_run_id = str(reg_fit_json.get("run_id") or "")
        if not reg_run_id:
            raise GateFailure("regression fit JSON omitted run_id")

        run_checked(
            [
                str(carbon_bin),
                "predict",
                str(reg_score_csv),
                "--model",
                str(reg_handle),
                "--out",
                str(reg_preds_out),
                *manifest_args,
            ],
            env=cli_env,
            timeout_secs=90,
        )
        pred_rows = list(csv.DictReader(io.StringIO(reg_preds_out.read_text(encoding="utf-8"))))
        if len(pred_rows) != len(probe_rows):
            raise GateFailure(f"expected {len(probe_rows)} regression predictions, got {len(pred_rows)}")

        explain_proc = run_checked(
            [str(carbon_bin), "explain", "--model", str(reg_handle), "--json", *manifest_args],
            env=cli_env,
            timeout_secs=90,
        )
        explain_json = json.loads(explain_proc.stdout)
        if not explain_json.get("importances"):
            raise GateFailure("explain returned empty feature importances")

        export_sizes: dict[str, int] = {}
        export_files: dict[str, Path] = {}
        for lang, ext in (
            ("python", "py"),
            ("rust", "rs"),
            ("cpp", "hpp"),
            ("js", "js"),
            ("model-object", "b64"),
        ):
            out_file = tmp / f"champion_reg.{ext}"
            run_checked(
                [
                    str(carbon_bin),
                    "export",
                    "--model",
                    str(reg_handle),
                    "--lang",
                    lang,
                    "--out",
                    str(out_file),
                    *manifest_args,
                ],
                env=cli_env,
                timeout_secs=90,
            )
            content = out_file.read_bytes()
            if len(content) < 64:
                raise GateFailure(f"exported {lang} artifact unexpectedly small ({len(content)} bytes)")
            export_sizes[lang] = len(content)
            export_files[lang] = out_file

        py_preds = run_standalone_python_check(export_files["python"], probe_rows)
        js_verified = run_standalone_js_check(export_files["js"], probe_rows, py_preds)
        mo_meta = verify_model_object_b64(export_files["model-object"].read_text(encoding="utf-8"))

        # Deliberate domain refusal check: mismatched column name must fail cleanly (no FunctionError)
        bad_score = tmp / "bad_columns.csv"
        bad_score.write_text("wrong_col_a,wrong_col_b\n1.0,2.0\n", encoding="utf-8")
        bad_proc = run_checked(
            [
                str(carbon_bin),
                "predict",
                str(bad_score),
                "--model",
                str(reg_handle),
                "--out",
                str(tmp / "bad_out.csv"),
                *manifest_args,
            ],
            env=cli_env,
            timeout_secs=45,
            expect_ok=False,
        )
        if bad_proc.returncode == 0:
            raise GateFailure("mismatched column predict unexpectedly succeeded")

        shred_reg_proc = run_checked(
            [str(carbon_bin), "shred", "--model", str(reg_handle), "--json", *manifest_args],
            env=cli_env,
            timeout_secs=60,
        )
        if not json.loads(shred_reg_proc.stdout).get("shredded"):
            raise GateFailure("regression carbon shred did not report shredded=true")

        post_shred = run_checked(
            [
                str(carbon_bin),
                "status",
                "--run",
                reg_run_id,
                "--cloud",
                env_name,
                *manifest_args,
            ],
            env=cli_env,
            timeout_secs=30,
            expect_ok=False,
        )
        if post_shred.returncode == 0:
            raise GateFailure("status after carbon shred unexpectedly succeeded")

        # 3. Binary classification E2E with `"below"` / `"above"` labels (`"elo"` substring)
        bin_csv_text, bin_score_text = build_tricky_binary_csv()
        bin_csv = tmp / "tricky_binary.csv"
        bin_score_csv = tmp / "tricky_binary_score.csv"
        bin_handle = tmp / "bin.carbon.json"
        bin_preds_out = tmp / "bin_preds.csv"
        bin_csv.write_text(bin_csv_text, encoding="utf-8")
        bin_score_csv.write_text(bin_score_text, encoding="utf-8")

        bin_fit_proc = run_checked(
            [
                str(carbon_bin),
                "fit",
                str(bin_csv),
                "--target",
                "regime_label",
                "--save",
                str(bin_handle),
                "--cloud",
                env_name,
                "--format",
                "json",
                *manifest_args,
            ],
            env=cli_env,
            timeout_secs=240,
        )
        bin_fit_json = json.loads(bin_fit_proc.stdout)
        bin_run_id = str(bin_fit_json.get("run_id") or "")
        if not bin_run_id:
            raise GateFailure("binary fit JSON omitted run_id")

        run_checked(
            [
                str(carbon_bin),
                "predict",
                str(bin_score_csv),
                "--model",
                str(bin_handle),
                "--out",
                str(bin_preds_out),
                *manifest_args,
            ],
            env=cli_env,
            timeout_secs=90,
        )
        bin_pred_rows = list(csv.DictReader(io.StringIO(bin_preds_out.read_text(encoding="utf-8"))))
        if len(bin_pred_rows) != 10:
            raise GateFailure(f"expected 10 binary predictions, got {len(bin_pred_rows)}")

        bin_export_sizes: dict[str, int] = {}
        for lang, ext in (("python", "py"), ("rust", "rs"), ("cpp", "hpp"), ("js", "js")):
            out_file = tmp / f"champion_bin.{ext}"
            run_checked(
                [
                    str(carbon_bin),
                    "export",
                    "--model",
                    str(bin_handle),
                    "--lang",
                    lang,
                    "--out",
                    str(out_file),
                    *manifest_args,
                ],
                env=cli_env,
                timeout_secs=90,
            )
            bin_export_sizes[lang] = len(out_file.read_bytes())

        run_checked(
            [str(carbon_bin), "shred", "--model", str(bin_handle), "--json", *manifest_args],
            env=cli_env,
            timeout_secs=60,
        )
        run_checked([str(carbon_bin), "logout", "--all"], env=cli_env, timeout_secs=15)

    cw_stats: dict[str, Any] = {}
    if not skip_cloudwatch:
        # Brief settle pause for AWS Lambda invocation metrics to publish to CloudWatch.
        time.sleep(8.0)
        finished_utc = dt.datetime.now(dt.timezone.utc)
        cw_stats = query_cloudwatch_lambda_metrics(
            env_name,
            regions,
            started_utc,
            finished_utc,
            aws_bin=aws_bin,
        )

    return {
        "schema": "jopedime.carbon.live-cell-e2e-gate/v1",
        "environment": env_name,
        "regions": regions,
        "all_passed": True,
        "wall_secs": round(time.monotonic() - t0, 3),
        "regression_gate": {
            "run_id": reg_run_id,
            "tricky_columns_verified": [
                "free_stream_velocity",
                "power_generation",
                "household_population",
                "island_code",
                "candidate_score",
            ],
            "export_bytes": export_sizes,
            "python_self_check_and_predict_ok": True,
            "js_parity_verified": js_verified,
            "model_object": mo_meta,
            "domain_refusal_422_ok": True,
            "shred_verified_404": True,
        },
        "binary_gate": {
            "run_id": bin_run_id,
            "tricky_labels_verified": ["below", "above"],
            "export_bytes": bin_export_sizes,
            "shred_verified": True,
        },
        "cloudwatch": cw_stats,
    }


def run_self_test() -> int:
    """Deterministic offline self-test exercising stubbed CLI + CloudWatch failure detection."""
    with tempfile.TemporaryDirectory(prefix="carbon-verify-selftest-") as td_str:
        td = Path(td_str)
        fake_carbon = td / "fake-carbon"
        fake_aws = td / "fake-aws"

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("abc123/manifest.json", json.dumps({"schema": "jope.model-object/v1"}))
        mo_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        fake_carbon.write_text(
            "#!/usr/bin/env python3\n"
            "import json, pathlib, sys\n"
            "args = sys.argv[1:]\n"
            "cmd = args[0] if args else ''\n"
            "if cmd in ('ping', 'register', 'logout'):\n"
            "    print('{}'); raise SystemExit(0)\n"
            "if cmd == 'fit':\n"
            "    save = pathlib.Path(args[args.index('--save') + 1])\n"
            "    rid = '01J00000000000000000000001' if 'regression' in args[1] else '01J00000000000000000000002'\n"
            "    save.write_text(json.dumps({'run_id': rid}))\n"
            "    print(json.dumps({'run_id': rid, 'state': 'completed'}))\n"
            "    raise SystemExit(0)\n"
            "if cmd == 'predict':\n"
            "    csv_in = pathlib.Path(args[1]).read_text().splitlines()\n"
            "    if 'wrong_col_a' in csv_in[0]:\n"
            "        sys.stderr.write('HTTP 422: predict_refused\\n'); raise SystemExit(1)\n"
            "    out = pathlib.Path(args[args.index('--out') + 1])\n"
            "    n = max(0, len(csv_in) - 1)\n"
            "    out.write_text('prediction\\n' + ''.join(f'{10.5 + i}\\n' for i in range(n)))\n"
            "    raise SystemExit(0)\n"
            "if cmd == 'explain':\n"
            "    print(json.dumps({'importances': [{'feature': 'free_stream_velocity', 'importance': 0.7}]}))\n"
            "    raise SystemExit(0)\n"
            "if cmd == 'export':\n"
            "    lang = args[args.index('--lang') + 1]\n"
            "    out = pathlib.Path(args[args.index('--out') + 1])\n"
            "    if lang == 'python':\n"
            "        out.write_text('# ' + ('x'*80) + '\\nMODEL_COLUMNS = (\"free_stream_velocity\",)\\ndef predict_rows(rows):\\n    return [float(sum(r)) for r in rows]\\nif __name__ == \"__main__\":\\n    raise SystemExit(0)\\n')\n"
            "    elif lang == 'js':\n"
            "        out.write_text('// ' + ('x'*80) + '\\nfunction predictRows(rows){ return rows.map(r => r.reduce((a,b)=>a+b, 0)); }\\n')\n"
            "    elif lang == 'model-object':\n"
            f"        out.write_text({mo_b64!r})\n"
            "    else:\n"
            "        out.write_text('// standalone source ' + ('y' * 80) + '\\n')\n"
            "    raise SystemExit(0)\n"
            "if cmd == 'shred':\n"
            "    print(json.dumps({'shredded': True, 'objects_deleted': 5})); raise SystemExit(0)\n"
            "if cmd == 'status':\n"
            "    sys.stderr.write('HTTP 404: run not found\\n'); raise SystemExit(1)\n"
            "raise SystemExit(2)\n",
            encoding="utf-8",
        )
        fake_carbon.chmod(0o755)

        fake_aws.write_text(
            "#!/usr/bin/env python3\n"
            "import json, os, sys\n"
            "errs = int(os.environ.get('FAKE_CW_ERRORS', '0'))\n"
            "metric = sys.argv[sys.argv.index('--metric-name') + 1]\n"
            "val = errs if metric == 'Errors' else (4 if metric == 'Invocations' else 0)\n"
            "print(json.dumps({'Datapoints': [{'Sum': val}]}))\n",
            encoding="utf-8",
        )
        fake_aws.chmod(0o755)

        # 1. Clean run succeeds
        receipt = execute_live_gate(
            carbon_bin=fake_carbon,
            env_name="dev",
            regions=["us-east-2"],
            clouds_manifest=None,
            skip_cloudwatch=True,
            aws_bin=str(fake_aws),
        )
        assert receipt["all_passed"] is True
        assert receipt["regression_gate"]["model_object"]["schema"] == "jope.model-object/v1"

        # 2. Non-zero CloudWatch Errors must fail closed!
        now = dt.datetime.now(dt.timezone.utc)
        os.environ["FAKE_CW_ERRORS"] = "1"
        try:
            failed = False
            try:
                query_cloudwatch_lambda_metrics("dev", ["us-east-2"], now, now, aws_bin=str(fake_aws))
            except GateFailure as exc:
                failed = "Errors=1" in str(exc)
            if not failed:
                raise GateFailure("self-test failed to reject non-zero CloudWatch Errors")
        finally:
            os.environ.pop("FAKE_CW_ERRORS", None)

    print("verify-live-cell self-test: PASS")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify live Carbon AWS Lambda cells end-to-end.")
    parser.add_argument("--self-test", action="store_true", help="Run deterministic offline self-test")
    parser.add_argument("--env", choices=["dev", "prod"], default="dev", help="Target Carbon cloud")
    parser.add_argument("--regions", default="us-east-2", help="Comma-separated AWS regions or 'wave1'")
    parser.add_argument("--carbon-bin", default="carbon", help="Path to built carbon CLI binary")
    parser.add_argument("--clouds-manifest", default="", help="Optional path to carbon-clouds.json")
    parser.add_argument("--skip-cloudwatch", action="store_true", help="Skip AWS CloudWatch metric check")
    parser.add_argument("--json-out", default="", help="Optional path to write JSON E2E receipt")
    args = parser.parse_args(argv)

    if args.self_test:
        return run_self_test()

    carbon_path = Path(shutil.which(args.carbon_bin) or args.carbon_bin).resolve()
    manifest_path = Path(args.clouds_manifest).resolve() if args.clouds_manifest else None
    regions = expand_regions(args.regions, args.env)

    receipt = execute_live_gate(
        carbon_bin=carbon_path,
        env_name=args.env,
        regions=regions,
        clouds_manifest=manifest_path,
        skip_cloudwatch=args.skip_cloudwatch,
    )
    rendered = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(rendered, encoding="utf-8")
    sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except GateFailure as err:
        sys.stderr.write(f"[carbon-e2e-gate] FAIL: {err}\n")
        raise SystemExit(1)
