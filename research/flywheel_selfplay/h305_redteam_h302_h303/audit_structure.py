"""Independently reconstruct family admission sets and fit-lock chronology."""

from pathlib import Path
import datetime
import hashlib
import json
import subprocess
import numpy as np
from independent_replay import outer_fold


def stream_seed(*parts):
    return (
        int.from_bytes(
            hashlib.sha256(":".join(map(str, parts)).encode()).digest()[:8], "little"
        )
        & 0x7FFFFFFF
    )


def inner_h303(outer, family):
    return (
        hashlib.sha256(
            b"H303-inner-v1\0" + str(outer).encode() + b"\0" + family.encode()
        ).digest()[-1]
        % 4
    )


def member_inner_sets(families, seed):
    ordered = sorted(families)
    shuffled = np.random.default_rng(
        stream_seed(seed, "h278-full-train-folds")
    ).permutation(ordered)
    return [set(shuffled[i::4]) for i in range(4)]


def partition_audit(families):
    all_families = set(map(str, families))
    out = []
    for k in range(4):
        test = {f for f in all_families if outer_fold(f) == k}
        train = all_families - test
        details = []
        for root in (30001, 30002, 30003):
            for member in range(3):
                seed = stream_seed(root, "H300_ENS_VOTE2_MEMBER", member)
                folds = member_inner_sets(train, seed)
                assert set.union(*folds) == train
                assert all(
                    not (a & b) for i, a in enumerate(folds) for b in folds[i + 1 :]
                )
                details.append(
                    dict(
                        root=root,
                        member=member,
                        seed=seed,
                        inner_sizes=list(map(len, folds)),
                        selection_test_overlap=len(set.union(*folds) & test),
                        final_fit_test_overlap=len(train & test),
                    )
                )
        nested = []
        for i in range(4):
            assessment = {f for f in train if inner_h303(k, f) == i}
            fit = train - assessment
            assert not (fit & test) and not (assessment & test)
            nested.append(
                dict(inner=i, fit=len(fit), selection=len(assessment), outer_overlap=0)
            )
        out.append(
            dict(
                outer=k,
                test_count=len(test),
                train_count=len(train),
                test_families_sha256=hashlib.sha256(
                    json.dumps(sorted(test)).encode()
                ).hexdigest(),
                inherited_H287_training_selection_overlap=len(test),
                members=details,
                h303_inner=nested,
            )
        )
    return out


def git_time(commit):
    raw = subprocess.check_output(
        [
            "git",
            "-C",
            "/home/ubuntu/JopeDime",
            "show",
            "-s",
            "--format=%H %at %ct",
            commit,
        ],
        text=True,
    ).split()
    return dict(commit=raw[0], author_epoch=int(raw[1]), committer_epoch=int(raw[2]))


def chronology(inventory, prereg):
    stamp = git_time(prereg)
    locks = [x for x in inventory if x["name"].endswith("fit_lock.json")]
    assert locks
    return dict(
        prereg=stamp,
        lock_times=[
            dict(
                name=x["name"],
                epoch=x["mtime"],
                utc=datetime.datetime.fromtimestamp(
                    x["mtime"], datetime.timezone.utc
                ).isoformat(),
                after_prereg=x["mtime"]
                > max(stamp["author_epoch"], stamp["committer_epoch"]),
            )
            for x in locks
        ],
        all_locks_after_prereg=all(
            x["mtime"] > max(stamp["author_epoch"], stamp["committer_epoch"])
            for x in locks
        ),
        evidence_limit="Tar mtimes are mutable timestamps, and locks contain no internal absolute score timestamp. Source orders frozen-model outer scoring after lock. This supports, but does not prove, no undocumented earlier score or remote push time.",
    )


def check_nested_lock(lock):
    table = lock["inner_table"]
    assert len(table) == 18
    grid = {(float(x["tau"]), int(x["k"]), int(x["a"])) for x in table}
    assert grid == {
        (t, k, a) for t in (0.05, 0.1, 0.15) for k in (5, 6, 7) for a in (2, 3)
    }
    top = max(x["inner_gain"] for x in table)
    tied = [x for x in table if x["inner_gain"] >= top - 1e-4]
    chosen = min(tied, key=lambda x: (x["inner_switches"], -x["k"], -x["a"], x["tau"]))
    assert chosen == lock["chosen"] and len(tied) == lock["n_tied"]
    assert len(lock["members"]) == 3
    assert all(x["selected_tau"] == chosen["tau"] for x in lock["members"])
    return dict(
        outer=lock["outer_fold"],
        root=lock["root_seed"],
        chosen=chosen,
        selection_rule_matches=True,
    )


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--custody", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    a = ap.parse_args()
    with np.load(a.data, allow_pickle=False) as z:
        partitions = partition_audit(z["families"])
    out = dict(
        partitions=partitions,
        h302_chronology=chronology(
            json.loads((a.custody / "archive-inventory.json").read_text()), "2a77787c3d"
        ),
        h303_prereg=git_time("43e00dd31f"),
        direct_admission="PASS source-reconstructed member fit and lr/n/margin selection; no runtime fit-row trace was archived",
        h302_inherited_selection="FAIL (a): inherited tau=.10, q=6 and agreement=2 were evaluated/retained using all205 TRAIN families before H302",
        h303_design_scope="Within-wave tau/q/agreement choice is nested in source. Prior choice of dataset, r882 proposer, K=8, feature schema, algorithm and grid still used earlier TRAIN outcomes; not an untouched family population.",
    )
    (a.custody / "source/structure_audit.json").write_text(
        json.dumps(out, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "folds": [(x["test_count"], x["train_count"]) for x in partitions],
                "chronology": out["h302_chronology"],
            }
        )
    )
