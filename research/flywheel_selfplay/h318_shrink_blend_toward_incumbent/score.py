"""One locked outer-test evaluation, both paired policies from identical model outputs."""

from runtime import *


def main():
    done = json.loads((D / "fit_done.json").read_text())
    assert done["status"] == "PASS" and len(done["tasks"]) == 36
    G = initialize()
    rows = G["train_rows"]
    cids = [r["context_sha256"] for r in rows]
    allcids = set(cids)
    f = np.array([r["family_id"] for r in rows])
    idx = {c: i for i, c in enumerate(cids)}
    locks = {}
    for part in PARTS:
        for k in range(4):
            for seed in SEEDS:
                tdir = D / part / f"fold_{k}" / f"root_{seed}"
                lp = tdir / "fit_lock_h318.json"
                lock = json.loads(lp.read_text())
                assert (
                    lock["outer_test_rows_read_before_lock"]
                    == lock["outer_test_packets_built_before_lock"]
                    == lock["denied_reads"]
                    == 0
                )
                assert (
                    sha(tdir / "fit_lock.json") == lock["base_fit_lock_sha256"]
                    and sha(tdir / "input_identity.json")
                    == lock["input_identity_sha256"]
                )
                for name, digest in lock["source_hashes"].items():
                    assert sha(P / "source" / name) == digest, name
                for m in lock["members"]:
                    for model in [m["ranker"], m["veto"], *m["bags"]]:
                        assert sha(tdir / "refit" / model["file"]) == model["sha256"]
                rec = json.loads((tdir / "model_archive.json").read_text())
                assert rec["exact_version_get_verified"] and rec[
                    "members_verified"
                ] == len(rec["members"])
                locks[f"{part}/{k}/{seed}"] = dict(
                    path=str(lp.relative_to(D)), sha256=sha(lp), alpha=lock["alpha"]
                )
    durable(
        P / "source/lock_receipt.json",
        dict(
            status="PASS",
            locks=locks,
            count=36,
            model_files_verified=1080,
            outer_rows_read_before_lock=0,
            finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        ),
    )
    U = np.full((9, len(cids), 3), np.nan, dtype=np.float64)
    invalid = np.ones_like(U, dtype=bool)
    picks = np.full((9, len(cids), 2), -1, dtype=np.int32)
    test_receipts = []
    for pi, part in enumerate(PARTS):
        for k in range(4):
            testrows = [r for r in rows if fold(part, r["family_id"]) == k]
            tc = [r["context_sha256"] for r in testrows]
            ix = [idx[c] for c in tc]
            for seed in SEEDS:
                marker = D / part / f"fold_{k}" / f"root_{seed}" / "outer_once.json"
                assert not marker.exists()
                durable(
                    marker,
                    dict(
                        state="STARTED",
                        fit_lock_sha256=locks[f"{part}/{k}/{seed}"]["sha256"],
                    ),
                )
            scope = Scope(tc, allcids)
            sys.addaudithook(scope.hook)
            try:
                P8 = packets(testrows, scope)
                for si, seed in enumerate(SEEDS):
                    tdir = D / part / f"fold_{k}" / f"root_{seed}"
                    lock = json.loads((tdir / "fit_lock_h318.json").read_text())
                    base = score_models(
                        tdir / "refit", lock["members"], P8, tc, lock["base_chosen"]
                    )
                    blend = blend_picks(base, lock["alpha"])
                    cell = pi * 3 + si
                    for n, c in enumerate(tc):
                        j = ix[n]
                        choices = [0, int(base[n]), int(blend[n])]
                        U[cell, j] = [P8[c]["utility_k"][q] for q in choices]
                        invalid[cell, j] = [
                            not bool(P8[c]["gate_k"][q]) for q in choices
                        ]
                    picks[cell, ix, 0] = base
                    picks[cell, ix, 1] = blend
                    durable(
                        tdir / "outer_once.json",
                        dict(
                            state="COMPLETE",
                            fit_lock_sha256=locks[f"{part}/{k}/{seed}"]["sha256"],
                            contexts=len(tc),
                            model_scores=1,
                            base_pick_sha256=hashlib.sha256(base.tobytes()).hexdigest(),
                            blend_pick_sha256=hashlib.sha256(
                                blend.tobytes()
                            ).hexdigest(),
                        ),
                    )
                test_receipts.append(
                    dict(
                        partition=part,
                        fold=k,
                        contexts=tc,
                        physical_feature_bytes=scope.bytes,
                        packet_hashes=scope.packet_hashes,
                        ranges=scope.ranges,
                    )
                )
            finally:
                scope.active = False
    assert (picks >= 0).all() and np.isfinite(U).all()
    # Fresh BASE must match the independently replayed historical baseline exactly.
    with np.load(D / "references/regression_picks.npz", allow_pickle=False) as z:
        matches = {
            part: bool(np.array_equal(picks[i * 3 : i * 3 + 3, :, 0], z[part]))
            for i, part in enumerate(PARTS)
        }
    (D / "out").mkdir(exist_ok=True)
    np.savez_compressed(
        D / "out/panel.npz",
        u=U,
        invalid=invalid,
        picks=picks,
        cids=np.array(cids),
        families=f,
    )
    durable(D / "out/outer_read_receipts.json", test_receipts)
    status = "PASS" if all(matches.values()) else "REGRESSION_FAIL"
    durable(
        P / "source/score_receipt.json",
        dict(
            status=status,
            fresh_base_pick_matches=matches,
            outer_model_score_calls=36,
            fit_locks=36,
            panel_sha256=sha(D / "out/panel.npz"),
            dev91_rows_read=0,
            r3175_train="NOT_MEASURED",
        ),
    )
    print(json.dumps(dict(status=status, matches=matches)), flush=True)
    assert status == "PASS", "REGRESSION_FAIL: fresh BASE mismatch"


if __name__ == "__main__":
    main()
