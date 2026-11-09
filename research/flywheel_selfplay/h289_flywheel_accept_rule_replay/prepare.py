"""Validate recorded candidate utilities and construct an identity-aligned replay matrix."""

from pathlib import Path
import json
import hashlib
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DOPE = "c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814"
EVAL = "e79a7715c1531ccb8c4055ab96971101cd4034ec3a71d83bca6d25b0e604ca83"
BASE = 1.1389572403748425


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, v):
    Path(p).write_text(json.dumps(v, indent=2, allow_nan=False) + "\n")


def main():
    inp = ROOT / "inputs"
    selected = json.loads((inp / "selected.json").read_text())
    assert (
        sha(inp / "selected.json")
        == "981960ad628f5029f8c54552c7b1cb1d440072e7e9dcffecdea52c6a3493a0f1"
    )
    rows = selected["dev"]
    assert len(rows) == 91 and len({r["family_id"] for r in rows}) == 19
    assert all(
        r["dl_split"] == "DL-Val"
        and r["source_split"] in ("train", "inner")
        and r["task"] == "regression"
        for r in rows
    )
    best = json.loads((inp / "best_dev_checkpoint.json").read_text())
    assert (
        best["best_round"] == 3175
        and best["evaluation_identity"] == EVAL
        and best["best_utility"] == BASE
    )
    contract = best["evaluation_contract"]
    assert contract["dope_sha256"] == DOPE and contract[
        "selected_packet_sha256"
    ] == sha(inp / "selected.json")
    assert [r["context_sha256"] for r in rows] == [
        x["cid"] for x in contract["contexts"]
    ]
    for r, c in zip(rows, contract["contexts"]):
        assert (
            r["pool_sha256"] == c["pool_sha256"]
            and r["train_only_jtt1_sha256"] == c["training_table_sha256"]
        )
    baseline = np.asarray(best["utility"]["per_context"], float)
    assert baseline.shape == (91,) and np.isfinite(baseline).all()
    assert abs(float(np.mean(baseline)) - BASE) < 1e-15
    families = np.asarray([r["family_id"] for r in rows])
    fids = sorted(set(families))
    groups = [np.flatnonzero(families == f) for f in fids]

    def validate_stats(d):
        u = np.asarray(d["per_context"], float)
        assert u.shape == (91,) and np.isfinite(u).all()
        assert len(d["selected_programs"]) == 91
        assert (
            d["n_reg"] == 91 and d["n_bin"] == 0 and abs(u.mean() - d["mean"]) < 1e-14
        )
        assert set(d["per_family"]) == set(fids)
        for f, g in zip(fids, groups):
            assert abs(u[g].mean() - d["per_family"][f]) < 1e-14
        return u

    validate_stats(best["utility"])
    inv = json.loads((inp / "capture-inventory.json").read_text())
    pin = {r["path"]: r for r in inv["files"]}
    journal = {}
    for line in (inp / "adapter_admissions.jsonl").read_text().splitlines():
        d = json.loads(line)
        n = int(d["round"])
        if n in journal:
            assert journal[n] == d, "conflicting admission rows"
        journal[n] = d
    rounds = []
    utilities = []
    programs = []
    excluded = []
    metadata = []
    for n in range(3175, inv["latest_complete_round"] + 1):
        path = inp / "rounds" / f"round-{n}.json"
        if not path.exists():
            excluded.append(dict(round=n, reason="missing_round_file"))
            continue
        assert sha(path) == pin["rounds/" + path.name]["sha256"]
        d = json.loads(path.read_text())
        if (
            d.get("completed_round") is False
            or d.get("candidate_evaluated") is False
            or d.get("evaluation_identity") != EVAL
            or not isinstance(d.get("candidate_dev_utility"), dict)
        ):
            excluded.append(
                dict(round=n, reason="incomplete_or_identity_or_candidate_missing")
            )
            continue
        assert d["round"] == n and d["dope_sha256"] == DOPE and d["dope_checksum_match"]
        u = validate_stats(d["candidate_dev_utility"])
        a = d["adapter_admission"]
        j = journal[n]
        assert (
            j["evaluation_identity"] == EVAL
            and a["accepted"] == j["accepted"]
            and abs(u.mean() - j["candidate_utility"]) < 1e-14
        )
        if n == 3175:
            assert np.array_equal(u, baseline)
        rounds.append(n)
        utilities.append(u)
        programs.append(d["candidate_dev_utility"]["selected_programs"])
        metadata.append(
            dict(
                round=n,
                candidate_mean=float(u.mean()),
                journal_candidate_mean=j["candidate_utility"],
                recorded_accepted=j["accepted"],
                recorded_best_round=j["best_round"],
                candidate_parameter_sha256=d.get("candidate_parameter_sha256"),
                source_sha256=pin["rounds/" + path.name]["sha256"],
            )
        )
    assert rounds and rounds[0] == 3175
    values = np.asarray(utilities)
    programs = np.asarray(programs)
    invalid = np.zeros(values.shape, bool)
    evidence = {}
    for i, r in enumerate(rows):
        pool = Path("/home/ubuntu/jope-data/h50/input/pools") / (
            r["context_sha256"] + ".npz"
        )
        assert sha(pool) == r["pool_sha256"]
        with np.load(pool, allow_pickle=False) as z:
            hashes = [x.decode() for x in z["program_sha256"]]
            lookup = {h: j for j, h in enumerate(hashes)}
            fit = z["fitness"]
            gate = z["gate"]
            assert len(lookup) == len(hashes)
        used = {}
        for h in sorted(set(programs[:, i])):
            j = lookup[h]
            used[h] = dict(
                fitness=float(fit[j]),
                gate=bool(gate[j]),
                utility=float(-fit[j] if gate[j] else -2.0),
            )
        for k, h in enumerate(programs[:, i]):
            assert values[k, i] == used[h]["utility"], (rounds[k], i, h)
            invalid[k, i] = not used[h]["gate"]
        evidence[r["context_sha256"]] = dict(
            pool_sha256=r["pool_sha256"], programs=used
        )
    out = ROOT / "prepared"
    out.mkdir()
    np.savez_compressed(
        out / "history.npz",
        rounds=np.asarray(rounds),
        utilities=values,
        programs=programs,
        invalid=invalid,
        families=families,
        cids=np.array([r["context_sha256"] for r in rows]),
    )
    write(out / "selected-program-evidence.json", evidence)
    write(out / "round-metadata.json", metadata)
    regression = dict(
        status="PASS",
        r3175_mean=float(baseline.mean()),
        expected_mean=BASE,
        mean_bit_exact=float(baseline.mean()) == BASE,
        contexts=91,
        families=19,
        r3175_candidate_equals_best=True,
        all_candidate_program_utilities_exact=True,
        recorded_candidates=len(rounds) - 1,
        first_round=rounds[0],
        last_round=rounds[-1],
        excluded=excluded,
        evaluation_identity=EVAL,
        dope_identity=DOPE,
        no_forwards=True,
        history_sha256=sha(out / "history.npz"),
    )
    write(out / "regression.json", regression)
    print(json.dumps(regression))


if __name__ == "__main__":
    main()
