"""Check TRAIN-only source bytes against H305's independent replay pins."""

from pathlib import Path
import hashlib
import json
import trainonly_loader as tl

EXPECTED = {
    "prior": "9f376cdbb4ef0ca94938da486b8f4677563986b03e2a979389bd6b2c1847e14e",
    "projection": "6178ec557257927c753fcfbc152d7c4d45f02f550825e8d795385d092b6e0002",
    "lm48": "f01ee60f738f08e94fc80953dd40a2b14fe2eb991cc5a104886a154f44dde00c",
    "dope629": "d3f91651a5241e7e98263df536dfee8152432a4757e50d96f13828bdbd2e2c50",
}


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def main():
    root = Path("/home/ubuntu/jope-data")
    sel = root / "h50/input/selected.json"
    assert (
        sha(sel) == "981960ad628f5029f8c54552c7b1cb1d440072e7e9dcffecdea52c6a3493a0f1"
    )
    rows = json.loads(sel.read_text())["train"]
    cids = [r["context_sha256"] for r in rows]
    assert len(cids) == 1219
    identities = [[c, sha(root / "h50/input/pools" / f"{c}.npz")] for c in cids]
    poolsha = hashlib.sha256(json.dumps(identities).encode()).hexdigest()
    assert poolsha == "436fb6220accc33ec8c49991f9b6a5d8a2dfa7da508287938d703e9a019de8a1"
    costsha = sha(root / "h276/inputs/r882-train-costs.npz")
    assert costsha == "57004a91a315b372cde62b126be20edd4eb0f337942e08b19bd85438b520f69f"
    observed = {}
    for file, keys in [
        ("pool_priors.npz", [("base_prior_cost", "prior"), ("proj16", "projection")]),
        ("context_features.npz", [("lm48_z", "lm48"), ("dope629", "dope629")]),
    ]:
        with tl.TrainOnlyNumpy(set(cids)).load(root / "h58/cache" / file) as z:
            for key, label in keys:
                a = z[key]
                observed[label] = hashlib.sha256(a.tobytes()).hexdigest()
                assert observed[label] == EXPECTED[label]
    out = dict(
        status="PASS",
        reference="H305 independent TRAIN-only replay pins",
        selected_sha256=sha(sel),
        costs_sha256=costsha,
        pool_identities_sha256=poolsha,
        train_array_sha256=observed,
        loader_counters=tl.COUNTERS,
        train_contexts=1219,
        dlval_features_read=0,
    )
    (Path(__file__).parent / "train_input_receipt.json").write_text(
        json.dumps(out, indent=2) + "\n"
    )
    print(json.dumps(out))


if __name__ == "__main__":
    main()
