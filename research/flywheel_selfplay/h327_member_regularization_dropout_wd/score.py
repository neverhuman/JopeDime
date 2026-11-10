"""Once-only locked REG outer evaluation paired with exact-regressed frozen BASE."""

from runtime import *


def main():
    assert json.loads((D / "fit_done.json").read_text())["status"] == "PASS"
    assert json.loads((P / "source/lock_review.json").read_text())["status"] == "PASS"
    G = initialize()
    rows = G["train_rows"]
    cids = [r["context_sha256"] for r in rows]
    f = np.array([r["family_id"] for r in rows])
    allcids = set(cids)
    idx = {c: i for i, c in enumerate(cids)}
    locks = {}
    for part in PARTS:
        for k in range(4):
            for seed in SEEDS:
                td = D / part / f"fold_{k}" / f"root_{seed}"
                lp = td / "fit_lock_h327.json"
                lock = json.loads(lp.read_text())
                assert (
                    lock["outer_test_rows_read_before_lock"]
                    == lock["outer_test_packets_built_before_lock"]
                    == lock["denied_reads"]
                    == 0
                )
                assert (
                    sha(td / "fit_lock.json") == lock["member_fit_lock_sha256"]
                    and sha(td / "input_identity.json") == lock["input_identity_sha256"]
                )
                assert (
                    lock["parameter_receipt"]["status"] == "PASS"
                    and sha(td / "parameter_audit.jsonl")
                    == lock["parameter_receipt"]["audit_sha256"]
                )
                for name, h in lock["source_hashes"].items():
                    assert sha(P / "source" / name) == h, name
                for member in lock["members"]:
                    for model in [member["ranker"], member["veto"], *member["bags"]]:
                        assert sha(td / "refit" / model["file"]) == model["sha256"]
                arc = json.loads((td / "model_archive.json").read_text())
                assert arc["exact_version_get_verified"] and arc[
                    "members_verified"
                ] == len(arc["members"])
                locks[f"{part}/{k}/{seed}"] = dict(
                    sha256=sha(lp), chosen=lock["reg_chosen"]
                )
    durable(
        P / "source/lock_receipt.json",
        dict(
            status="PASS",
            locks=locks,
            count=36,
            model_files_verified=1080,
            outer_rows_read_before_lock=0,
        ),
    )
    with np.load(D / "references/out/panel.npz", allow_pickle=False) as z:
        assert z["cids"].tolist() == cids and z["families"].tolist() == f.tolist()
        U = z["u"].copy()
        invalid = z["invalid"].copy()
        picks = z["picks"].copy()
        U[:, :, 2] = np.nan
        picks[:, :, 1] = -1
    receipts = []
    for pi, part in enumerate(PARTS):
        for k in range(4):
            tr = [r for r in rows if fold(part, r["family_id"]) == k]
            tc = [r["context_sha256"] for r in tr]
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
                packet = packets(tr, scope)
                for si, seed in enumerate(SEEDS):
                    td = D / part / f"fold_{k}" / f"root_{seed}"
                    lock = json.loads((td / "fit_lock_h327.json").read_text())
                    pp = score_models(
                        td / "refit", lock["members"], packet, tc, lock["reg_chosen"]
                    )
                    cell = pi * 3 + si
                    for n, c in enumerate(tc):
                        j = ix[n]
                        assert U[cell, j, 0] == packet[c]["utility_k"][0]
                        assert bool(invalid[cell, j, 0]) == not_gate(
                            packet[c]["gate_k"][0]
                        )
                        U[cell, j, 2] = packet[c]["utility_k"][int(pp[n])]
                        invalid[cell, j, 2] = not_gate(packet[c]["gate_k"][int(pp[n])])
                    picks[cell, ix, 1] = pp
                    durable(
                        td / "outer_once.json",
                        dict(
                            state="COMPLETE",
                            contexts=len(tc),
                            model_scores=1,
                            fit_lock_sha256=locks[f"{part}/{k}/{seed}"]["sha256"],
                            reg_pick_sha256=hashlib.sha256(pp.tobytes()).hexdigest(),
                        ),
                    )
                receipts.append(
                    dict(
                        partition=part,
                        fold=k,
                        contexts=tc,
                        ranges=scope.ranges,
                        packet_hashes=scope.packet_hashes,
                        bytes=scope.bytes,
                    )
                )
            finally:
                scope.active = False
    assert np.isfinite(U).all() and (picks >= 0).all()
    with np.load(D / "references/regression_picks.npz", allow_pickle=False) as z:
        assert all(
            np.array_equal(picks[i * 3 : i * 3 + 3, :, 0], z[part])
            for i, part in enumerate(PARTS)
        )
    (D / "out").mkdir()
    np.savez_compressed(
        D / "out/panel.npz",
        u=U,
        invalid=invalid,
        picks=picks,
        cids=np.array(cids),
        families=f,
    )
    durable(D / "out/outer_read_receipts.json", receipts)
    durable(
        P / "source/score_receipt.json",
        dict(
            status="PASS",
            outer_model_score_calls=36,
            fit_locks=36,
            baseline="Read-only H318 BASE panel, exact-regressed before REG fitting",
            panel_sha256=sha(D / "out/panel.npz"),
            dev91_rows_read=0,
            r3175_train="NOT_MEASURED",
        ),
    )
    print("PASS: 36 once-only REG scores; BASE paired; incumbent utilities exact")


def not_gate(x):
    return not bool(x)


if __name__ == "__main__":
    main()
