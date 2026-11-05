"""One inference pass of frozen H352 policies; outcomes are read after pick lock."""

import hashlib
import os
from pathlib import Path
import numpy as np
from common import (
    D,
    E,
    S,
    PARTS,
    ROOTS,
    DOPE,
    durable,
    load,
    prereg,
    rows,
    sha,
    utc,
    fold,
    require_cid,
    install_read_guard,
)


class DevCache:
    def __init__(self, dev_rows):
        import trainonly_loader as tl

        self.tl = tl
        self.allowed = {r["context_sha256"] for r in dev_rows}
        self.expected = {}
        self.ranges = []
        selected = load(D / "input/selected.json")
        train = {r["context_sha256"] for r in selected["train"]}
        for fn, keys in [
            ("pool_priors.npz", ("base_prior_cost", "proj16")),
            ("context_features.npz", ("lm48_z", "dope629")),
        ]:
            path = D / "input" / fn
            cids = [str(c) for c in tl.read_all(path, "cids.npy")]
            assert (
                len(cids) == 1310
                and set(cids[:1219]) == train
                and set(cids[1219:]) == self.allowed
            )
            indices = {c: i for i, c in enumerate(cids)}
            offsets = (
                tl.read_all(path, "offsets.npy") if fn == "pool_priors.npz" else None
            )
            for c in self.allowed:
                i = indices[c]
                for key in keys:
                    self.expected[(fn, key, c)] = (
                        (int(offsets[i]), int(offsets[i + 1]))
                        if offsets is not None
                        else (i, i + 1)
                    )

    def read(self, cid, fn, member):
        require_cid(cid)
        lo, hi = self.expected[(fn, member, cid)]
        path = D / "input" / fn
        off, shape, dtype = self.tl._member_layout(path, member + ".npy")
        assert 0 <= lo < hi <= shape[0]
        stride = (
            int(np.prod(shape[1:])) * dtype.itemsize
            if len(shape) > 1
            else dtype.itemsize
        )
        fd = os.open(path, os.O_RDONLY)
        try:
            raw = os.pread(fd, (hi - lo) * stride, off + lo * stride)
        finally:
            os.close(fd)
        assert len(raw) == (hi - lo) * stride
        self.ranges.append(
            dict(
                context=cid,
                file=fn,
                member=member,
                lo=lo,
                hi=hi,
                bytes=len(raw),
                sha256=hashlib.sha256(raw).hexdigest(),
            )
        )
        return np.frombuffer(raw, dtype=dtype).reshape((hi - lo,) + shape[1:]).copy()


def pool(cid, member, outcomes_locked=False):
    require_cid(cid, member, outcomes_locked)
    assert not outcomes_locked or (D / "PICKS-LOCK.json").exists()
    with np.load(D / "input/pools" / f"{cid}.npz", allow_pickle=False) as z:
        return z[member].copy()


def forbid_fitting():
    import lightgbm as lgb

    counter = {"attempts": 0}

    def refuse(*args, **kwargs):
        counter["attempts"] += 1
        raise RuntimeError("H356 reuses frozen models; fitting/updating is forbidden")

    lgb.train = refuse
    lgb.LGBMModel.fit = refuse
    lgb.Booster.update = refuse
    lgb.Booster.refit = refuse
    return counter


def verify_models():
    from native_audit import parse, observed_leaves

    pins = load(S / "input_pins.json")
    expected = {
        str(Path(k).relative_to("h352-jrb01")): v
        for p in pins["h352_model_parts"]
        for k, v in p["members"].items()
    }
    native = {"rankers": 0, "vetos": 0}
    locks = {}
    for part in PARTS:
        for k in range(4):
            for root in ROOTS:
                td = D / "models" / part / f"fold_{k}" / f"root_{root}"
                lp = td / "fit_lock.json"
                rel = str(lp.relative_to(D / "models"))
                assert sha(lp) == expected[rel]["sha256"]
                lk = load(lp)
                assert len(lk["inner_table"]) == 18 and len(lk["members"]) == 3
                for m in lk["members"]:
                    for kind, model in [
                        ("ranker", m["ranker"]),
                        ("veto", m["veto"]),
                        *[("ranker", b) for b in m["bags"]],
                    ]:
                        p = td / "refit" / model["file"]
                        rel = str(p.relative_to(D / "models"))
                        assert sha(p) == model["sha256"] == expected[rel]["sha256"]
                        text = p.read_text()
                        v = parse(text)
                        assert v["boosting"] == "gbdt" and int(v["num_threads"]) == 1
                        if kind == "ranker":
                            assert (
                                v["objective"] == "lambdarank"
                                and int(v["num_leaves"]) == 15
                                and int(v["max_depth"]) == 3
                                and int(v["min_data_in_leaf"]) == 20
                            )
                            assert (
                                float(v["lambda_l1"]) == 0
                                and float(v["lambda_l2"]) == 1
                                and float(v["sigmoid"]) == 1
                                and float(v["min_gain_to_split"]) == 0
                                and float(v["min_sum_hessian_in_leaf"]) == 0.001
                            )
                            assert observed_leaves(text, "LGBMRanker")["maximum"] <= 8
                            native["rankers"] += 1
                        else:
                            assert (
                                v["objective"] == "binary"
                                and int(v["min_data_in_leaf"]) == 25
                                and float(v["lambda_l1"]) == 1
                                and float(v["lambda_l2"]) == 2
                            )
                            native["vetos"] += 1
                locks[(part, k, root)] = lk
    assert native == dict(rankers=972, vetos=108)
    durable(
        D / "FROZEN-MODELS-VERIFIED.json",
        dict(
            status="PASS",
            utc=utc(),
            locks=36,
            models=1080,
            native=native,
            all_model_hashes_match_H352=True,
        ),
    )
    return locks


def verify_inputs():
    p = load(S / "input_pins.json")
    r = load(D / "FETCH-PASS.json")
    assert r["status"] == "PASS"
    for row in rows():
        assert (
            sha(D / "input/pools" / f"{row['context_sha256']}.npz")
            == row["pool_sha256"]
        )
    for name, digest in [
        ("context_features.npz", p["context_features_sha256"]),
        ("selected.json", p["selected_sha256"]),
        ("best_dev_checkpoint.json", p["r882_reference_sha256"]),
        ("r3175-costs.npz", p["r3175_costs"]["sha256"]),
    ] + [(s["name"], s["sha256"]) for s in p["retained_input_snapshots"]]:
        assert sha(D / "input" / name) == digest, name


def main():
    publication = prereg()
    install_read_guard()
    verify_inputs()
    locks = verify_models()
    counter = forbid_fitting()
    os.environ["H265_PAUSE_FILE"] = str(D / "h356.pause.json")
    import h265_core as core
    import topk
    import veto
    import bagged
    import ensemble as ens
    import run_h303 as r303

    # Vendor functions are byte-exact; only scoring functions are called.
    dev = rows()
    cids = [r["context_sha256"] for r in dev]
    families = [r["family_id"] for r in dev]
    ids = set(cids)
    cache = DevCache(dev)
    durable(
        D / "INFERENCE-STARTED.json",
        dict(utc=utc(), prereg=publication["head"], frozen_policies=9, model_fits=0),
        exclusive=True,
    )
    costs = {}
    with np.load(D / "input/r882-secondary-costs.npz", allow_pickle=False) as z:
        assert set(z.files) == ids
        for c in cids:
            costs[c] = z[c].astype(np.float32)
    X = []
    indices = []
    for row in dev:
        c = row["context_sha256"]
        prog = pool(c, "program").astype(np.float32)
        assert len(prog) == row["n_candidates"] == len(costs[c])
        prior = cache.read(c, "pool_priors.npz", "base_prior_cost").astype(np.float32)
        proj = cache.read(c, "pool_priors.npz", "proj16").astype(np.float32)
        lm = cache.read(c, "context_features.npz", "lm48_z")[0].astype(np.float32)
        dope = cache.read(c, "context_features.npz", "dope629")[0].astype(np.float32)
        order = np.argsort(costs[c].astype(np.float64), kind="mergesort")
        chosen, x = topk.build_topk(
            costs[c],
            prior,
            prog[order[:8]],
            core.compute_sem24_numpy(prog),
            proj,
            lm,
            dope,
            8,
        )
        assert x.shape == (8, 568)
        X.append(x)
        indices.append(chosen)
    X = np.stack(X)
    indices = np.asarray(indices)
    picks = np.full((9, 91), -1, dtype=np.int32)
    raw = {}
    policies = []
    groups = []
    for pi, part in enumerate(PARTS):
        for si, root in enumerate(ROOTS):
            row_index = pi * 3 + si
            policies.append(dict(partition=part, root_seed=root))
            for k in range(4):
                ix = [i for i, f in enumerate(families) if fold(part, f) == k]
                td = D / "models" / part / f"fold_{k}" / f"root_{root}"
                lk = locks[(part, k, root)]
                ch = lk["chosen"]
                tag = f"{part}-F{k}-R{root}"
                mark = D / "score_markers" / (tag + ".json")
                durable(
                    mark,
                    dict(
                        state="STARTED",
                        utc=utc(),
                        contexts=[cids[i] for i in ix],
                        fit_lock_sha256=sha(td / "fit_lock.json"),
                    ),
                    exclusive=True,
                )
                if ix:
                    outs = r303.score_members(
                        td / "refit",
                        lk["members"],
                        X[ix].reshape(len(ix) * 8, 568),
                        len(ix),
                        ens,
                    )
                    ml = [
                        r303.member_picks(o, ch["tau"], ch["k"], veto, core, bagged)
                        for o in outs
                    ]
                    picks[row_index, ix] = r303.ens_picks(ml, ch["a"], ens)
                    for mi, o in enumerate(outs):
                        for key in ["raw_scores", "p_inv", "bag_scores"]:
                            raw[f"{tag}-M{mi}-{key}"] = np.asarray(o[key])
                record = dict(
                    state="COMPLETE",
                    utc=utc(),
                    contexts=[cids[i] for i in ix],
                    model_prediction_batches=1 if ix else 0,
                    fit_lock_sha256=sha(td / "fit_lock.json"),
                    chosen=ch,
                    picks_sha256=hashlib.sha256(
                        picks[row_index, ix].tobytes()
                    ).hexdigest(),
                )
                durable(mark, record)
                groups.append(record)
    assert np.all((picks >= 0) & (picks < 8)) and counter["attempts"] == 0
    np.savez_compressed(D / "predictions.npz", **raw)
    np.savez_compressed(
        D / "picks.npz",
        picks=picks,
        top8_indices=indices,
        cids=np.asarray(cids),
        families=np.asarray(families),
    )
    durable(
        D / "PICKS-LOCK.json",
        dict(
            utc=utc(),
            policies=policies,
            groups=36,
            picks_sha256=sha(D / "picks.npz"),
            predictions_sha256=sha(D / "predictions.npz"),
            feature_matrix_sha256=hashlib.sha256(X.tobytes()).hexdigest(),
            model_fits=0,
            dev_fitness_gate_members_decoded_before_pick_lock=0,
        ),
        exclusive=True,
    )
    # Only now decode the fixed outcomes and reference utility records.
    U = np.empty((9, 91), dtype=np.float64)
    invalid_flags = np.empty((9, 91), dtype=bool)
    u882 = []
    u3175 = []
    i882 = []
    i3175 = []
    with np.load(D / "input/r3175-costs.npz", allow_pickle=False) as z:
        assert set(z.files) == ids
        for i, c in enumerate(cids):
            fitness = pool(c, "fitness", True).astype(np.float64)
            gate = pool(c, "gate", True).astype(bool)
            utility = np.where(gate, -fitness, -2.0)
            j882 = int(indices[i, 0])
            j3175 = int(np.argmin(z[c].astype(np.float32)))
            selected = indices[i, picks[:, i]]
            U[:, i] = utility[selected]
            invalid_flags[:, i] = (~gate[selected]) | (U[:, i] <= -2.0 + 1e-9)
            u882.append(float(utility[j882]))
            u3175.append(float(utility[j3175]))
            i882.append(bool(not gate[j882] or utility[j882] <= -2 + 1e-9))
            i3175.append(bool(not gate[j3175] or utility[j3175] <= -2 + 1e-9))
    stored = load(D / "input/round-3175.json")["dev_utility"]
    reference = load(D / "input/best_dev_checkpoint.json")
    reference_map = dict(
        zip(
            [r["cid"] for r in reference["evaluation_contract"]["contexts"]],
            reference["utility"]["per_context"],
        )
    )
    assert (
        set(reference_map) == ids
        and list(u3175) == stored["per_context"]
        and all(u882[i] == reference_map[c] for i, c in enumerate(cids))
    )
    assert (
        abs(np.mean(u3175) - 1.1389572403748425) <= 1e-12
        and abs(np.mean(u882) - 1.1377941898164525) <= 1e-12
    )
    assert (
        np.all(np.isfinite(U))
        and np.all(np.isfinite(u3175))
        and np.all(np.isfinite(u882))
    )
    from statistics_h356 import summarize, decide

    primary = summarize(U, np.asarray(u3175), families)
    secondary = summarize(U, np.asarray(u882), families)
    criteria, verdict = decide(primary)
    for s, i in [(primary, i3175), (secondary, i882)]:
        s["invalid_per_policy"] = invalid_flags.sum(axis=1).tolist()
        s["reference_invalid"] = sum(i)
        for n, cell in enumerate(s["per_policy"]):
            cell.update(
                policies[n],
                switches_vs_r882=int(np.sum(picks[n] != 0)),
                invalid=int(invalid_flags[n].sum()),
            )
    np.savez_compressed(
        D / "utilities.npz",
        NL15=U,
        invalid=invalid_flags,
        r882=np.asarray(u882),
        r3175=np.asarray(u3175),
        families=np.asarray(families),
        cids=np.asarray(cids),
    )
    durable(
        D / "CACHE-RANGES.json",
        dict(
            status="PASS",
            contexts=91,
            ranges=cache.ranges,
            range_records=len(cache.ranges),
            TRAIN_feature_rows_decoded=0,
            other_DLVal_rows_decoded=0,
        ),
    )
    result = dict(
        wave="H356",
        status="COMPLETE",
        verdict=verdict,
        criteria=criteria,
        prereg_commit=publication["head"],
        h352_source_commit="02dfee2f7e0b8543324494dd4f99d074cd6b93b6",
        primary_vs_r3175=primary,
        secondary_vs_r882=secondary,
        elo_labels=dict(
            primary="reference = r3175 slot0 (dev91)",
            secondary="reference = r882 slot0 (dev91)",
        ),
        policy="Unchanged H352 nine partition/root policies; original family-hash fold routing; original r882 top8 input view; same frozen picks compared with both references; no fold averaging or dev selection",
        dev_contexts=91,
        dev_families=19,
        regression=dict(
            r3175_equal=91,
            r3175_mean=float(np.mean(u3175)),
            r882_equal=91,
            r882_mean=float(np.mean(u882)),
        ),
        model_fits=0,
        TRAIN_refits=0,
        model_updates=0,
        frozen_models=1080,
        fit_locks=36,
        inference_group_markers=36,
        inference_retries=0,
        dev_outcome_members_decoded_before_pick_lock=0,
        protected_or_other_DLVal_payload_rows=0,
        dope=DOPE,
        picks_sha256=sha(D / "picks.npz"),
        utilities_sha256=sha(D / "utilities.npz"),
        cache_ranges_sha256=sha(D / "CACHE-RANGES.json"),
        limitations="19 reused development families and nine dependent policies; no fresh-cohort or serving qualification; no all-TRAIN refit model was introduced",
        live_flywheel_mutated=False,
    )
    durable(D / "metrics.json", result)
    durable(E / "metrics.json", result)
    durable(
        D / "SCORE-COMPLETE.json",
        dict(utc=utc(), verdict=verdict, model_fits=0, contexts=91, policies=9),
    )
    print(
        __import__("json").dumps(
            dict(
                verdict=verdict,
                criteria=criteria,
                primary_median_elo=primary["median_elo"],
                paired_lift=primary["median_family_equal_lift"],
                lcb95=primary["paired_lift_lcb95"],
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
