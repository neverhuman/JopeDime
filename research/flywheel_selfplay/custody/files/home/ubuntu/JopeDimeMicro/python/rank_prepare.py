"""DSL-only candidate AST featurization (412D input dictionary)."""
import re
import numpy as np


def features(dsl: str) -> dict:
    ops = re.findall(r"\(([^()]+)\)", dsl)
    result = {"steps": float(len(ops)), "dsl_characters": float(len(dsl))}
    for pos, content in enumerate(ops):
        tokens = content.split()
        op = tokens[0]
        result["count." + op] = result.get("count." + op, 0) + 1
        result[f"position{pos}.{op}"] = 1.0
        if pos == len(ops) - 1:
            result["estimator." + op] = 1.0
        for token in tokens[1:]:
            if "=" not in token:
                continue
            key, value = token.split("=", 1)
            if key == "seed":
                continue
            try:
                value = float(value)
            except ValueError:
                if value not in ["true", "false"]:
                    continue
                value = float(value == "true")
            value = float(np.sign(value) * np.log1p(abs(value)))
            result[f"parameter.{op}.{key}.sum"] = result.get(f"parameter.{op}.{key}.sum", 0) + value
            result[f"parameter.{op}.{key}.count"] = result.get(f"parameter.{op}.{key}.count", 0) + 1
            result[f"position{pos}.{key}"] = value
    return result
