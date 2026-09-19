#!/usr/bin/env python3
"""validate-receipt.py <schema.json> <document.json> — fail closed if the document does not
match the schema. Uses the `jsonschema` package when present; otherwise a built-in validator
for the JSON Schema subset the receipts in schemas/ use (type, required, properties,
additionalProperties, enum, const, pattern, items, minimum, maximum, minItems, oneOf/anyOf via
`type` lists). Unknown keywords are an error, so the fallback can never silently accept."""
import json
import re
import sys

SUPPORTED = {
    "$schema", "$id", "title", "description", "type", "required", "properties",
    "additionalProperties", "enum", "const", "pattern", "items", "minimum", "maximum",
    "minItems", "maxItems", "examples", "default", "$comment",
}
TYPES = {
    "object": dict, "array": list, "string": str, "boolean": bool, "null": type(None),
}


def check(schema, doc, path, errors):
    unknown = set(schema) - SUPPORTED
    if unknown:
        errors.append(f"{path}: schema uses unsupported keywords {sorted(unknown)}")
        return
    t = schema.get("type")
    if t is not None:
        types = t if isinstance(t, list) else [t]
        ok = False
        for name in types:
            if name == "integer":
                ok = ok or (isinstance(doc, int) and not isinstance(doc, bool))
            elif name == "number":
                ok = ok or (isinstance(doc, (int, float)) and not isinstance(doc, bool))
            else:
                ok = ok or isinstance(doc, TYPES[name])
        if not ok:
            errors.append(f"{path}: expected type {types}, got {type(doc).__name__}")
            return
    if "enum" in schema and doc not in schema["enum"]:
        errors.append(f"{path}: {doc!r} not in {schema['enum']}")
    if "const" in schema and doc != schema["const"]:
        errors.append(f"{path}: {doc!r} != const {schema['const']!r}")
    if "pattern" in schema and isinstance(doc, str) and not re.search(schema["pattern"], doc):
        errors.append(f"{path}: {doc!r} does not match /{schema['pattern']}/")
    if isinstance(doc, (int, float)) and not isinstance(doc, bool):
        if "minimum" in schema and doc < schema["minimum"]:
            errors.append(f"{path}: {doc} < minimum {schema['minimum']}")
        if "maximum" in schema and doc > schema["maximum"]:
            errors.append(f"{path}: {doc} > maximum {schema['maximum']}")
    if isinstance(doc, dict):
        for key in schema.get("required", []):
            if key not in doc:
                errors.append(f"{path}: missing required property {key!r}")
        props = schema.get("properties", {})
        for key, value in doc.items():
            if key in props:
                check(props[key], value, f"{path}.{key}", errors)
            elif schema.get("additionalProperties", True) is False:
                errors.append(f"{path}: additional property {key!r} is not allowed")
            elif isinstance(schema.get("additionalProperties"), dict):
                check(schema["additionalProperties"], value, f"{path}.{key}", errors)
    if isinstance(doc, list):
        if "minItems" in schema and len(doc) < schema["minItems"]:
            errors.append(f"{path}: {len(doc)} items < minItems {schema['minItems']}")
        if "maxItems" in schema and len(doc) > schema["maxItems"]:
            errors.append(f"{path}: {len(doc)} items > maxItems {schema['maxItems']}")
        if "items" in schema:
            for i, item in enumerate(doc):
                check(schema["items"], item, f"{path}[{i}]", errors)


def main():
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    with open(sys.argv[1]) as f:
        schema = json.load(f)
    with open(sys.argv[2]) as f:
        doc = json.load(f)
    try:
        import jsonschema  # type: ignore

        validator = jsonschema.Draft202012Validator(schema)
        errors = [f"{'/'.join(str(p) for p in e.absolute_path) or '$'}: {e.message}"
                  for e in validator.iter_errors(doc)]
        from importlib.metadata import version

        engine = f"jsonschema {version('jsonschema')}"
    except ImportError:
        errors = []
        check(schema, doc, "$", errors)
        engine = "built-in subset validator"
    if errors:
        print(f"validate-receipt: {sys.argv[2]} FAILS {schema.get('$id', sys.argv[1])} ({engine}):",
              file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 1
    print(f"validate-receipt: ok {sys.argv[2]} matches {schema.get('$id', sys.argv[1])} ({engine})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
