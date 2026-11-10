"""Paired frozen-source nested fits; only sample weights change in member fitting."""

from runtime import *
from concurrent.futures import ProcessPoolExecutor
import time, gc
from archive_models import preserve_cell
import weights


def task(args):
    part, k, seed = args
    t0 = time.time()
    G = r303.G
    allrows = G["train_rows"]
    allcids = {r["context_sha256"] for r in allrows}
    rows = [r for r in allrows if fold(part, r["family_id"]) != k]
    allowed = {r["context_sha256"] for r in rows}
    scope = Scope(allowed, allcids, fit=True)
    sys.addaudithook(scope.hook)
    tdir = D / part / f"fold_{k}" / f"root_{seed}"
    assert not (tdir / "fit_lock.json").exists(), (
        "No hidden resume over partial or completed fit"
    )
    tdir.mkdir(parents=True, exist_ok=True)
    weights.AUDIT = tdir / "weight_audit.jsonl"
    weights.ALLOWED = allowed
    captured = []
    original = r303.family_gain

    def capture(picks, cids, packets, families):
        assert set(cids) == allowed
        captured.append((np.asarray(picks, dtype=np.int32).copy(), list(cids)))
        return original(picks, cids, packets, families)

    try:
        G["P8"] = packets(rows, scope)
        assert set(G["P8"]) == allowed
        G["fam_to_fold"] = {r["family_id"]: fold(part, r["family_id"]) for r in allrows}
        r303.SCRATCH = D / part
        r303.family_gain = capture
        famw = r303.task((k, seed))
        assert len(captured) == 18
        idx = next(
            i
            for i, x in enumerate(famw["inner_table"])
            if all(x[key] == famw["chosen"][key] for key in ["tau", "k", "a"])
        )
        oof, cids = captured[idx]
        weight_receipt = weights.audit_summary(weights.AUDIT)
        expected = {(m["seed"], m["member_index"]) for m in famw["members"]}
        assert expected == {(a302.EXPECTED_MEMBER_SEEDS[seed][i], i) for i in range(3)}
        assert scope.denied == 0 and scope.contexts == allowed
        input_ids = dict(
            packet_hashes=scope.packet_hashes,
            ranges=scope.ranges,
            allowed_contexts=sorted(allowed),
            forbidden_contexts=sorted(allcids - allowed),
        )
        durable(tdir / "input_identity.json", input_ids)
        np.savez_compressed(
            tdir / "inner_selected_oof.npz", picks=oof, cids=np.array(cids)
        )
        lock = dict(
            schema="jope.h322.fit-lock/v1",
            partition=part,
            outer_fold=k,
            root_seed=seed,
            member_fit_lock_sha256=sha(tdir / "fit_lock.json"),
            famw_chosen=famw["chosen"],
            arm="FAMW",
            weight_receipt=weight_receipt,
            weight_transform_sha256=sha(D / "setup/weight_transform.json"),
            inner_oof_sha256=sha(tdir / "inner_selected_oof.npz"),
            input_identity_sha256=sha(tdir / "input_identity.json"),
            members=famw["members"],
            outer_test_rows_read_before_lock=0,
            outer_test_packets_built_before_lock=0,
            allowed_contexts=len(allowed),
            forbidden_contexts=len(allcids - allowed),
            feature_bytes_read=scope.bytes,
            denied_reads=scope.denied,
            source_hashes=json.loads(os.environ["H322_CODE_PINS"]),
            prereg_commit=json.loads((P / "prereg-published.json").read_text())["head"],
            finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        )
        durable(tdir / "fit_lock_h322.json", lock)
        archive = preserve_cell(tdir)
        result = dict(
            partition=part,
            fold=k,
            seed=seed,
            arm="FAMW",
            weight_fit_calls=weight_receipt["fit_calls"],
            famw=famw["chosen"],
            seconds=time.time() - t0,
            lock_sha256=sha(tdir / "fit_lock_h322.json"),
            archive=archive,
        )
        print(json.dumps(result), flush=True)
        return result
    finally:
        weights.AUDIT = None
        weights.ALLOWED = None
        r303.family_gain = original
        G.pop("P8", None)
        scope.active = False
        gc.collect()


def main():
    assert (
        json.loads((P / "source/regression_gate.json").read_text())["status"] == "PASS"
    )
    G = initialize()
    durable(D / "setup/weight_transform.json", weights.install(G))
    os.environ["H322_CODE_PINS"] = json.dumps(
        {f.name: sha(f) for f in (P / "source").glob("*.py")}, sort_keys=True
    )
    tasks = [(p, k, s) for p in PARTS for k in range(4) for s in SEEDS]
    with ProcessPoolExecutor(max_workers=6) as ex:
        results = list(ex.map(task, tasks))
    durable(
        D / "fit_done.json",
        dict(
            status="PASS", tasks=results, locks=36, outer_test_rows_read_before_lock=0
        ),
    )
    print("H322 FIT COMPLETE 36 locks", flush=True)


if __name__ == "__main__":
    main()
