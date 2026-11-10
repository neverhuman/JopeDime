"""Preregistered metadata-only neighbor predicate, preserving unknown comparisons."""

import re, unicodedata, math, hashlib, json


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def normalize(name):
    text = unicodedata.normalize("NFKC", str(name)).casefold().strip()
    return re.sub(r"[^\w]+|_+", "_", text, flags=re.UNICODE).strip("_")


def names(values):
    return sorted({normalize(v) for v in values if normalize(v)})


def valid(value):
    return value is not None and str(value).strip().casefold() not in {
        "",
        "none",
        "null",
        "unknown",
        "nan",
        "n/a",
        "-",
    }


def same(a, b):
    return bool(set(a) & set(b)) if a and b else None


def either(a, b):
    return (
        True if a is True or b is True else False if a is False and b is False else None
    )


def both(a, b):
    return (
        False if a is False or b is False else True if a is True and b is True else None
    )


def compare(d, t):
    source = same(d["source_ids"], t["source_ids"])
    root = same(d["lineage_roots"], t["lineage_roots"])
    a = either(source, root)
    b = same(d["dedupe_fingerprints"], t["dedupe_fingerprints"])
    dc, tc = set(d["columns"]), set(t["columns"])
    j = len(dc & tc) / len(dc | tc) if dc and tc else None
    rows = (
        abs(math.log(d["n_rows"] / t["n_rows"]))
        if d.get("n_rows", 0)
        and t.get("n_rows", 0)
        and d["n_rows"] > 0
        and t["n_rows"] > 0
        else None
    )
    c = both(
        j >= 0.8 if j is not None else None, rows <= 0.10 if rows is not None else None
    )
    clauses = {"a": a, "b": b, "c": c}
    status = (
        "NEIGHBORED"
        if any(v is True for v in clauses.values())
        else "CLEAN"
        if all(v is False for v in clauses.values())
        else "UNKNOWN"
    )
    similarity = 1.0 if a is True or b is True else j
    return dict(
        status=status,
        clauses=clauses,
        source_id_match=source,
        lineage_root_match=root,
        jaccard=j,
        log_row_ratio_abs=rows,
        similarity=similarity,
        shared_source_ids=sorted(set(d["source_ids"]) & set(t["source_ids"])),
        shared_roots=sorted(set(d["lineage_roots"]) & set(t["lineage_roots"])),
        shared_fingerprints=sorted(
            set(d["dedupe_fingerprints"]) & set(t["dedupe_fingerprints"])
        ),
    )


def classify(dev, train):
    pairs = []
    for d in dev:
        for t in train:
            pairs.append(
                dict(
                    dev_dataset=d["dataset_id"],
                    train_dataset=t["dataset_id"],
                    train_family=t["family_id"],
                    **compare(d, t),
                )
            )
    matched = [p for p in pairs if p["status"] == "NEIGHBORED"]
    unknown = [p for p in pairs if p["status"] == "UNKNOWN"]
    observed = [p["similarity"] for p in pairs if p["similarity"] is not None]
    status = "NEIGHBORED" if matched else "UNKNOWN" if unknown else "CLEAN"
    sim = max(observed) if observed else None
    return dict(
        status=status,
        triggered_clauses=sorted(
            {k for p in matched for k, v in p["clauses"].items() if v is True}
        ),
        witnesses=matched,
        comparison_count=len(pairs),
        unknown_comparisons=len(unknown),
        missing_similarity_comparisons=sum(p["similarity"] is None for p in pairs),
        max_train_similarity=sim,
        similarity_is_lower_bound=bool(
            sim != 1.0 and (unknown or any(p["similarity"] is None for p in pairs))
        ),
        all_pair_receipts=pairs,
    )
