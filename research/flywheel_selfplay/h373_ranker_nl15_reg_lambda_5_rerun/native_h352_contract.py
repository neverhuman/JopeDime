"""H352 native defaults, indexed by unchanged constructor identity.

Learning-rate/tree-count selection remains adaptive within the unchanged pinned
H352 grids. Normalize those two constructor fields when comparing their native
representations; every other native value, including derived seeds, is exact.
No historical picks, labels, gains or chosen configurations enter this contract.
"""

from pathlib import Path
import hashlib, json
from native_audit import compare_native

D = Path("/home/ubuntu/jope-data/h373-jrb01")
ADAPTIVE = {"learning_rate", "n_estimators"}
CONTRACT = None


def key(kind, params):
    return (
        kind
        + ":"
        + json.dumps(
            {
                k: v
                for k, v in params.items()
                if k
                not in ADAPTIVE | ({"reg_lambda"} if kind == "LGBMRanker" else set())
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )


def normalized(native):
    return {
        k: v for k, v in native.items() if k not in {"learning_rate", "num_iterations"}
    }


def build(rows):
    records = {}
    choices = {}
    for row in rows:
        kind = row["kind"]
        p = row["constructor_params"]
        v = row["native_params"]
        k = key(kind, p)
        choice = [p["learning_rate"], p["n_estimators"]]
        choices.setdefault(kind, [])
        if choice not in choices[kind]:
            choices[kind].append(choice)
        if k in records:
            assert normalized(records[k]["native"]) == normalized(v)
        else:
            records[k] = dict(constructor=p, native=v)
    return dict(
        records=records,
        unchanged_grid_choices=choices,
        adaptive_fields=sorted(ADAPTIVE),
        sole_treatment_delta={"lambda_l2": [1, 5]},
        historical_outcomes_included=False,
    )


def validate(params, native, kind, contract=None):
    global CONTRACT
    if contract is None:
        if CONTRACT is None:
            CONTRACT = json.loads((D / "h352-native-contract.json").read_text())
        contract = CONTRACT
    compare_native(params, native, kind)
    assert [params["learning_rate"], params["n_estimators"]] in contract[
        "unchanged_grid_choices"
    ][kind]
    base = contract["records"][key(kind, params)]
    expected = dict(base["native"])
    assert base["constructor"]["min_child_samples"] == (
        20 if kind == "LGBMRanker" else 25
    )
    assert base["constructor"]["reg_lambda"] == (1.0 if kind == "LGBMRanker" else 2.0)
    # These are values selected by unchanged H352 logic, not interventions.
    expected["learning_rate"] = str(params["learning_rate"])
    expected["num_iterations"] = str(params["n_estimators"])
    if kind == "LGBMRanker":
        expected["lambda_l2"] = "5"
    assert set(expected) == set(native)
    for k, want in expected.items():
        if k in {"learning_rate", "num_iterations"}:
            assert float(native[k]) == float(want), (k, want, native[k])
        else:
            assert native[k] == want, (k, want, native[k])
    return True


def compile_reference():
    from nl15_native_reference import compare_native as check_h352

    rows = []
    pins = {}
    for p in sorted(
        (D / "nl15_reference").glob("P*/fold_*/root_*/native_fit_audit.jsonl")
    ):
        raw = p.read_bytes()
        pins[str(p.relative_to(D))] = hashlib.sha256(raw).hexdigest()
        for line in raw.splitlines():
            row = json.loads(line)
            check_h352(row["constructor_params"], row["native_params"], row["kind"])
            rows.append(row)
    assert len(pins) == 36 and len(rows) == 36504
    contract = build(rows)
    for row in rows:
        kind = row["kind"]
        p = dict(row["constructor_params"])
        v = dict(row["native_params"])
        if kind == "LGBMRanker":
            p["reg_lambda"] = 5.0
            v["lambda_l2"] = "5"
        validate(p, v, kind, contract)

    raw = (json.dumps(contract, separators=(",", ":"), sort_keys=True) + "\n").encode()
    (D / "h352-native-contract.json").write_bytes(raw)
    receipt = dict(
        status="PASS",
        H352_native_fit_records=len(rows),
        H352_task_audits=36,
        source_sha256=pins,
        contract_sha256=hashlib.sha256(raw).hexdigest(),
        native_identities=len(contract["records"]),
        historical_outcomes_in_contract=False,
    )
    (D / "h352-native-contract-receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    return receipt
