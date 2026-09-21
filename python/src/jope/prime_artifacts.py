"""Certification-only portable artifacts for Prime operators.

An emitter existing here does not promote an operator. Quarantined operators
require an explicit ``certification=True`` call, and the manifest preserves the
registry lifecycle/capability flags separately from the sources under test.
"""

from __future__ import annotations

import ast
import hashlib
import json
import numbers
import textwrap
from dataclasses import dataclass
from typing import Any

import numpy as np

from .grammars.prime import (
    GRAMMAR,
    entry_version,
    plan_version,
    registry_hash,
    registry_snapshot,
    registry_snapshot_hash,
)
from .prime_ops import MAX_WIDTH, PRIME_VERSION, fit_transform

ARTIFACT_SCHEMA = "jope-prime-portable-artifact-v2"
TEST_VECTOR_SCHEMA = "jope-prime-test-vectors-v1"

_BANNED_SOURCE_TERMS = (
    "/home/",
    "/tmp/",
    "battle_",
    "candidate",
    "chromosome",
    "crossover",
    "evolut",
    "featmath",
    "fitness",
    "genetic",
    "genome",
    "gp_",
    "jmath",
    "jope",
    "kernel",
    "newmath",
    "sys.path",
)


@dataclass(frozen=True)
class CompileArtifact:
    portable_python: str
    portable_rust: str
    optimized_cuda_cpp: str | None
    manifest: dict[str, Any]
    test_vectors: dict[str, Any]

    @property
    def artifact_hash(self) -> str:
        return str(self.manifest["artifact_hash"])


def _sha256(data: bytes | str) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True)


def _json_hash(value: Any) -> str:
    return _sha256(_canonical_json(value))


def _registry_value(value: Any) -> Any:
    if value is None:
        return None
    if not isinstance(value, dict) or len(value) != 1:
        raise ValueError(f"malformed registry value: {value!r}")
    return next(iter(value.values()))


def _operator_entry(operator: str) -> dict[str, Any]:
    matches = [entry for entry in registry_snapshot() if entry.get("name") == operator]
    if len(matches) != 1:
        raise ValueError(f"unknown Prime operator {operator!r}")
    entry = matches[0]
    if entry.get("role") != "transform" or "prime" not in entry.get("tags", []):
        raise ValueError(f"{operator!r} is not a Prime transform")
    return entry


def _bounded_int(value: Any, parameter: dict[str, Any]) -> int:
    name = str(parameter["name"])
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(f"CountSketch parameter {name!r} must be an integer")
    result = int(value)
    minimum = parameter.get("min")
    maximum = parameter.get("max")
    if minimum is not None and result < int(minimum):
        raise ValueError(f"CountSketch parameter {name!r} is below {int(minimum)}")
    if maximum is not None and result > int(maximum):
        raise ValueError(f"CountSketch parameter {name!r} exceeds {int(maximum)}")
    return result


def _count_sketch_params(entry: dict[str, Any], params: dict[str, Any] | None) -> dict[str, Any]:
    supplied = dict(params or {})
    append = supplied.pop("append", False)
    if not isinstance(append, bool):
        raise ValueError("CountSketch parameter 'append' must be boolean")
    registry_params = {str(item["name"]): item for item in entry.get("params", [])}
    unknown = sorted(set(supplied) - set(registry_params))
    if unknown:
        raise ValueError(f"unknown CountSketch parameters: {', '.join(unknown)}")
    resolved: dict[str, Any] = {}
    for name, parameter in registry_params.items():
        value = supplied.get(name, _registry_value(parameter.get("default")))
        resolved[name] = _bounded_int(value, parameter)
    if not 0 <= resolved["seed"] <= (1 << 64) - 1:
        raise ValueError("CountSketch seed must fit in an unsigned 64-bit integer")
    if not 1 <= resolved["n"] <= MAX_WIDTH:
        raise ValueError(f"CountSketch width must be in 1..{MAX_WIDTH}")
    resolved["append"] = append
    return resolved


def _input_matrix(test_rows: Any | None, input_columns: int) -> np.ndarray:
    if isinstance(input_columns, bool) or not isinstance(input_columns, numbers.Integral):
        raise ValueError("input_columns must be an integer")
    input_columns = int(input_columns)
    if not 1 <= input_columns <= MAX_WIDTH:
        raise ValueError(f"input_columns must be in 1..{MAX_WIDTH}")
    if test_rows is None:
        test_rows = [
            [
                ((row + 1) * (column + 2) * (-1 if (row + column) % 2 else 1)) / 7.0
                for column in range(input_columns)
            ]
            for row in range(4)
        ]
    matrix = np.asarray(test_rows, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("test_rows must be a two-dimensional matrix")
    if matrix.shape[0] == 0:
        raise ValueError("test_rows must contain at least one row")
    if matrix.shape[1] != input_columns:
        raise ValueError(f"test_rows has {matrix.shape[1]} columns; input contract requires {input_columns}")
    return np.ascontiguousarray(np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0))


def lint_portable_source(source: str, language: str) -> list[str]:
    """Return deterministic source-independence and vocabulary violations."""
    violations = [term for term in _BANNED_SOURCE_TERMS if term in source.lower()]
    if language == "python":
        try:
            tree = ast.parse(source)
        except SyntaxError as error:
            return sorted(set(violations + [f"python syntax: {error.msg}"]))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".", 1)[0] not in {
                        "csv",
                        "json",
                        "math",
                        "numpy",
                        "sys",
                    }:
                        violations.append(f"python import:{alias.name}")
            elif isinstance(node, ast.ImportFrom):
                violations.append(f"python import-from:{node.module or '.'}")
    elif language == "rust":
        for marker in ("extern crate", "include!(", "include_bytes!(", "include_str!("):
            if marker in source:
                violations.append(f"rust dependency:{marker}")
    else:
        raise ValueError(f"unknown source language {language!r}")
    return sorted(set(violations))


def _python_source(*, width: int, seed: int, append: bool, input_columns: int) -> str:
    return textwrap.dedent(
        f'''\
        """Deterministic CountSketch feature transform."""

        import sys

        import numpy as np

        WIDTH = {width}
        SEED = {seed}
        APPEND = {append!r}
        INPUT_COLUMNS = {input_columns}
        MAX_OUTPUT_COLUMNS = {MAX_WIDTH}
        _MASK64 = (1 << 64) - 1


        def _hash_pair(index):
            value = ((index + 1) * 0x9E3779B97F4A7C15 + SEED * 0xD1B54A32D192ED03) & _MASK64
            value ^= value >> 30
            value = (value * 0xBF58476D1CE4E5B9) & _MASK64
            value ^= value >> 27
            value = (value * 0x94D049BB133111EB) & _MASK64
            value ^= value >> 31
            return value & 0x7FFFFFFFFFFFFFFF, (1.0 if value & 1 else -1.0)


        def transform(rows):
            matrix = np.asarray(rows, dtype=np.float64)
            if matrix.ndim != 2:
                raise ValueError("expected a two-dimensional matrix")
            if matrix.shape[1] != INPUT_COLUMNS:
                raise ValueError("input column count does not match the artifact contract")
            matrix = np.ascontiguousarray(
                np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
            )
            sketch = np.zeros((matrix.shape[0], WIDTH), dtype=np.float64)
            for column in range(INPUT_COLUMNS):
                bucket, sign = _hash_pair(column)
                sketch[:, bucket % WIDTH] += sign * matrix[:, column]
            sketch = np.nan_to_num(sketch, nan=0.0, posinf=0.0, neginf=0.0)
            if not APPEND:
                return sketch[:, :MAX_OUTPUT_COLUMNS]
            cap = min(MAX_OUTPUT_COLUMNS, INPUT_COLUMNS + WIDTH)
            left = min(INPUT_COLUMNS, cap)
            right = cap - left
            return np.concatenate((matrix[:, :left], sketch[:, :right]), axis=1)


        def main(argv=None):
            args = list(sys.argv[1:] if argv is None else argv)
            if len(args) != 2:
                raise SystemExit("usage: count_sketch INPUT.csv OUTPUT.csv")
            rows = np.loadtxt(args[0], delimiter=",", ndmin=2)
            np.savetxt(args[1], transform(rows), delimiter=",", fmt="%.17g")


        if __name__ == "__main__":
            main()
        '''
    )


def _rust_source(*, width: int, seed: int, append: bool, input_columns: int) -> str:
    append_text = "true" if append else "false"
    return textwrap.dedent(
        f"""\
        #![forbid(unsafe_code)]

        use std::env;
        use std::fmt::Write as _;
        use std::fs;

        const WIDTH: usize = {width};
        const SEED: u64 = {seed};
        const APPEND: bool = {append_text};
        const INPUT_COLUMNS: usize = {input_columns};
        const MAX_OUTPUT_COLUMNS: usize = {MAX_WIDTH};

        fn hash_pair(index: usize) -> (u64, f64) {{
            let mut value = (index as u64 + 1)
                .wrapping_mul(0x9E3779B97F4A7C15)
                .wrapping_add(SEED.wrapping_mul(0xD1B54A32D192ED03));
            value ^= value >> 30;
            value = value.wrapping_mul(0xBF58476D1CE4E5B9);
            value ^= value >> 27;
            value = value.wrapping_mul(0x94D049BB133111EB);
            value ^= value >> 31;
            (value & 0x7FFFFFFFFFFFFFFF, if value & 1 == 1 {{ 1.0 }} else {{ -1.0 }})
        }}

        pub fn transform(input: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, String> {{
            let mut output = Vec::with_capacity(input.len());
            for row in input {{
                if row.len() != INPUT_COLUMNS {{
                    return Err("input column count does not match the artifact contract".into());
                }}
                let clean: Vec<f64> = row
                    .iter()
                    .map(|value| if value.is_finite() {{ *value }} else {{ 0.0 }})
                    .collect();
                let mut sketch = vec![0.0; WIDTH];
                for (column, value) in clean.iter().enumerate() {{
                    let (bucket, sign) = hash_pair(column);
                    sketch[bucket as usize % WIDTH] += sign * value;
                }}
                for value in &mut sketch {{
                    if !value.is_finite() {{
                        *value = 0.0;
                    }}
                }}
                if APPEND {{
                    let cap = (INPUT_COLUMNS + WIDTH).min(MAX_OUTPUT_COLUMNS);
                    let left = INPUT_COLUMNS.min(cap);
                    let right = cap - left;
                    let mut combined = Vec::with_capacity(cap);
                    combined.extend_from_slice(&clean[..left]);
                    combined.extend_from_slice(&sketch[..right]);
                    output.push(combined);
                }} else {{
                    output.push(sketch[..WIDTH.min(MAX_OUTPUT_COLUMNS)].to_vec());
                }}
            }}
            Ok(output)
        }}

        fn read_rows(path: &str) -> Result<Vec<Vec<f64>>, String> {{
            let contents = fs::read_to_string(path).map_err(|error| error.to_string())?;
            let mut rows = Vec::new();
            for (line_number, line) in contents.lines().enumerate() {{
                if line.trim().is_empty() {{
                    continue;
                }}
                let row = line
                    .split(',')
                    .map(|value| value.trim().parse::<f64>().map_err(|error| error.to_string()))
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|error| format!("line {{}}: {{}}", line_number + 1, error))?;
                rows.push(row);
            }}
            Ok(rows)
        }}

        fn write_rows(path: &str, rows: &[Vec<f64>]) -> Result<(), String> {{
            let mut contents = String::new();
            for row in rows {{
                for (column, value) in row.iter().enumerate() {{
                    if column > 0 {{
                        contents.push(',');
                    }}
                    write!(&mut contents, "{{:.17e}}", value).map_err(|error| error.to_string())?;
                }}
                contents.push('\\n');
            }}
            fs::write(path, contents).map_err(|error| error.to_string())
        }}

        fn run() -> Result<(), String> {{
            let args: Vec<String> = env::args().skip(1).collect();
            if args.len() != 2 {{
                return Err("usage: count_sketch INPUT.csv OUTPUT.csv".into());
            }}
            let input = read_rows(&args[0])?;
            let output = transform(&input)?;
            write_rows(&args[1], &output)
        }}

        fn main() {{
            if let Err(error) = run() {{
                eprintln!("{{}}", error);
                std::process::exit(2);
            }}
        }}
        """
    )


def compile_artifact(
    operator: str,
    params: dict[str, Any] | None = None,
    *,
    input_columns: int = 1,
    test_rows: Any | None = None,
    certification: bool = False,
) -> CompileArtifact:
    """Compile a portable artifact without changing registry promotion state."""
    spec = GRAMMAR.transforms.get(operator)
    if spec is None:
        raise ValueError(f"unknown Prime operator {operator!r}")
    entry = _operator_entry(operator)
    lifecycle = dict(entry.get("lifecycle") or {})
    if not spec.generate and not certification:
        raise RuntimeError(
            f"Prime operator {operator!r} is quarantined; pass certification=True only for conformance"
        )
    if operator != "count_sketch":
        raise NotImplementedError(f"portable certification emitter not implemented for {operator!r}")

    resolved = _count_sketch_params(entry, params)
    matrix = _input_matrix(test_rows, input_columns)
    input_columns = int(input_columns)
    expected, _ = fit_transform(operator, matrix, resolved)
    vectors = {
        "schema": TEST_VECTOR_SCHEMA,
        "input": matrix.tolist(),
        "expected_predictions": expected.tolist(),
        "rows": int(matrix.shape[0]),
        "input_columns": input_columns,
        "output_columns": int(expected.shape[1]),
        "numerical_tier": "exact_f64",
    }
    vectors_hash = _json_hash(vectors)
    python = _python_source(
        width=int(resolved["n"]),
        seed=int(resolved["seed"]),
        append=bool(resolved["append"]),
        input_columns=input_columns,
    )
    rust = _rust_source(
        width=int(resolved["n"]),
        seed=int(resolved["seed"]),
        append=bool(resolved["append"]),
        input_columns=input_columns,
    )
    violations = {
        "python": lint_portable_source(python, "python"),
        "rust": lint_portable_source(rust, "rust"),
    }
    if any(violations.values()):
        raise RuntimeError(f"portable source lint failed: {violations}")

    source_hashes = {
        "portable_python": _sha256(python),
        "portable_rust": _sha256(rust),
    }
    manifest_base = {
        "schema": ARTIFACT_SCHEMA,
        "plan_version": plan_version(),
        "operator": {
            "id": int(entry["id"]),
            "name": operator,
            "semantic_version": int(entry["semantic_version"]),
            "implementation_version": PRIME_VERSION,
        },
        "operator_registry": {
            "version": entry_version(),
            "semantic_sha256": registry_hash(),
            "snapshot_sha256": registry_snapshot_hash(),
        },
        "parameters": resolved,
        "seeds": {"count_sketch": int(resolved["seed"])},
        "numerical_tier": "exact_f64",
        "input_contract": {
            "rank": 2,
            "columns": input_columns,
            "dtype": "float64",
            "non_finite": "replace_with_zero",
            "maximum_output_columns": MAX_WIDTH,
        },
        "output_contract": {"columns": int(expected.shape[1])},
        "registry_lifecycle": lifecycle,
        "registry_capabilities": dict(entry.get("capabilities") or {}),
        "certification": {
            "requested": bool(certification),
            "promotion_eligible": bool(spec.generate),
            "portable_python_source": True,
            "portable_rust_source": True,
            "optimized_cuda_cpp_source": False,
        },
        "source_sha256": source_hashes,
        "test_vectors": {
            "schema": TEST_VECTOR_SCHEMA,
            "sha256": vectors_hash,
            "rows": int(matrix.shape[0]),
        },
        "source_lint": {"version": 1, "violations": violations},
    }
    manifest_payload_sha256 = _json_hash(manifest_base)
    artifact_hash = _json_hash(
        {
            "manifest_payload_sha256": manifest_payload_sha256,
            "source_sha256": source_hashes,
            "test_vectors_sha256": vectors_hash,
        }
    )
    manifest = {
        **manifest_base,
        "manifest_payload_sha256": manifest_payload_sha256,
        "artifact_hash": artifact_hash,
    }
    return CompileArtifact(
        portable_python=python,
        portable_rust=rust,
        optimized_cuda_cpp=None,
        manifest=manifest,
        test_vectors=vectors,
    )


def certify_count_sketch(
    params: dict[str, Any] | None = None,
    *,
    input_columns: int = 1,
    test_rows: Any | None = None,
) -> CompileArtifact:
    """Build CountSketch sources for conformance while it remains quarantined."""
    return compile_artifact(
        "count_sketch",
        params,
        input_columns=input_columns,
        test_rows=test_rows,
        certification=True,
    )


def artifact_manifest(
    operator: str,
    params: dict[str, Any] | None = None,
    *,
    input_columns: int = 1,
    test_rows: Any | None = None,
    certification: bool = False,
) -> dict[str, Any]:
    """Return the same truthful manifest attached to a compiled artifact."""
    return compile_artifact(
        operator,
        params,
        input_columns=input_columns,
        test_rows=test_rows,
        certification=certification,
    ).manifest
