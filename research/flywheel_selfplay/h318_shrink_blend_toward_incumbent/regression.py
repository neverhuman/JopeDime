"""Exact frozen-H308 replay before any H318 alpha selection or new model fit."""

from runtime import *


def main():
    G = initialize()
    rows = G["train_rows"]
    cids = [r["context_sha256"] for r in rows]
    families = np.array([r["family_id"] for r in rows])
    names = sorted(set(families))
    scope = Scope(cids, cids)
    sys.addaudithook(scope.hook)
    P8 = packets(rows, scope)
    out = {}
    panels = {}
    for part in PARTS:
        wave = "H303" if part == "P0" else "H308"
        prefix = D / "references" / wave / ("scratch" if part == "P0" else part)
        with np.load(D / "references" / (wave + "_picks.npz"), allow_pickle=False) as z:
            want = z["ENS_VOTE2_NESTED" + ("" if part == "P0" else "_" + part)].astype(
                np.int32
            )
        got = np.full_like(want, -1)
        folds = {}
        for k in range(4):
            ix = [i for i, r in enumerate(rows) if fold(part, r["family_id"]) == k]
            tc = [cids[i] for i in ix]
            for si, seed in enumerate(SEEDS):
                lp = prefix / f"fold_{k}" / f"root_{seed}" / "fit_lock.json"
                lock = json.loads(lp.read_text())
                pick = score_models(
                    lp.parent / "refit", lock["members"], P8, tc, lock["chosen"]
                )
                got[si, ix] = pick
                wh = hashlib.sha256(want[si, ix].tobytes()).hexdigest()
                gh = hashlib.sha256(pick.tobytes()).hexdigest()
                folds[f"{k}:{seed}"] = dict(
                    got=gh, want=wh, equal=gh == wh, lock_sha256=sha(lp)
                )
        u = np.array(
            [
                [
                    [P8[c]["utility_k"][0], P8[c]["utility_k"][int(got[s, i])]]
                    for i, c in enumerate(cids)
                ]
                for s in range(3)
            ],
            dtype=np.float64,
        )
        gains = np.array(
            [
                [
                    u[s, families == name, 1].mean() - u[s, families == name, 0].mean()
                    for name in names
                ]
                for s in range(3)
            ]
        ).mean(axis=1)
        expected = {
            "P0": 0.0643163334285782,
            "P1": 0.04821480030677507,
            "P2": 0.04360142642936269,
        }[part]
        out[part] = dict(
            per_fold=folds,
            per_seed_pick_sha256=[hashlib.sha256(a.tobytes()).hexdigest() for a in got],
            per_seed_equal=[bool(np.array_equal(a, b)) for a, b in zip(got, want)],
            gains=gains.tolist(),
            median=float(np.median(gains)),
            expected_median=expected,
            median_exact=float(np.median(gains)) == expected,
        )
        panels[part] = got
    status = (
        "PASS"
        if all(
            r["median_exact"]
            and all(r["per_seed_equal"])
            and all(f["equal"] for f in r["per_fold"].values())
            for r in out.values()
        )
        else "REGRESSION_FAIL"
    )
    durable(
        P / "source/regression_receipt.json",
        dict(
            status=status,
            partitions=out,
            dlval_feature_rows_read=0,
            contexts=len(cids),
            families=len(names),
            physical_feature_bytes=scope.bytes,
            mode="frozen model replay, no refit",
            finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        ),
    )
    durable(
        P / "source/regression_gate.json",
        dict(status=status, receipt_sha256=sha(P / "source/regression_receipt.json")),
    )
    durable(
        D / "references/regression_input_reads.json",
        dict(ranges=scope.ranges, packet_hashes=scope.packet_hashes),
    )
    np.savez_compressed(D / "references/regression_picks.npz", **panels)
    print(
        json.dumps(
            dict(status=status, medians={p: r["median"] for p, r in out.items()})
        ),
        flush=True,
    )
    assert status == "PASS", "REGRESSION_FAIL"


if __name__ == "__main__":
    main()
