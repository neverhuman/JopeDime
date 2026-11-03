from __future__ import annotations

from test_exact_train import *
from test_exact_train_sampling import FREEZE_GUARD, FREEZE_HISTORY

def split_median_divergence(epoch_draw_stats: dict) -> dict:
    """train-vs-validation per-dataset epoch-draw median ratio, per task and overall.

    Returns {scope: {"train": m, "validation": m, "ratio": r}}. `ratio` is the
    larger median over the smaller, so it is >= 1 and direction-free; a scope
    with no rows on one side is reported with ratio None and is not asserted on
    (a task can legitimately be absent from a split only if the freeze already
    failed the both-splits-populated check above).
    """
    medians = {
        key: float(value["median"])
        for key, value in epoch_draw_stats.items()
        if value.get("datasets")
    }
    scopes: dict[str, dict] = {}
    tasks = sorted({key.split("/", 1)[0] for key in medians})
    for task in tasks:
        train = medians.get(f"{task}/train")
        validation = medians.get(f"{task}/validation")
        scopes[task] = {
            "train": train,
            "validation": validation,
            "ratio": (
                max(train, validation) / min(train, validation)
                if train and validation
                else None
            ),
        }
    return scopes


def create_catalog_binding(
    *,
    pointer_path: Path = CATALOG_POINTER,
    root: Path = ROOT,
) -> Path:
    pointer = json.loads(pointer_path.read_text())
    catalog_path = Path(pointer["catalog"]).resolve()
    data_path = Path(pointer["data"]).resolve()
    if sha256_file(catalog_path) != pointer["catalog_sha256"]:
        raise RuntimeError("catalog pointer content hash drift")
    if sha256_file(data_path) != pointer["data_sha256"]:
        raise RuntimeError("catalog data content hash drift")
    bindings = root / "training" / "catalog-bindings"
    snapshots = bindings / "pointer-snapshots"
    snapshots.mkdir(parents=True, exist_ok=True)
    canonical_pointer = json.dumps(
        pointer, sort_keys=True, separators=(",", ":")
    ).encode()
    pointer_identity = hashlib.sha256(canonical_pointer).hexdigest()
    pointer_snapshot = snapshots / f"{pointer_identity}.json"
    if pointer_snapshot.is_file():
        if json.loads(pointer_snapshot.read_text()) != pointer:
            raise RuntimeError("immutable catalog pointer snapshot drift")
    else:
        atomic_json(pointer_snapshot, pointer)
    binding = {
        "schema": "CatalogBinding.v1",
        "catalog_path": str(catalog_path),
        "catalog_sha256": pointer["catalog_sha256"],
        "data_path": str(data_path),
        "data_sha256": pointer["data_sha256"],
        "pointer_snapshot_path": str(pointer_snapshot),
        "pointer_snapshot_sha256": sha256_file(pointer_snapshot),
        "created_at_ms": int(pointer_path.stat().st_mtime_ns // 1_000_000),
    }
    binding_identity = hashlib.sha256(
        json.dumps(binding, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    binding_path = bindings / f"{binding_identity}.json"
    if binding_path.is_file():
        if json.loads(binding_path.read_text()) != binding:
            raise RuntimeError("immutable catalog binding drift")
    else:
        atomic_json(binding_path, binding)
    return binding_path


def resolve_catalog_binding(
    manifest: dict,
    explicit: Path | None = None,
) -> Path:
    if explicit is not None:
        candidates = [explicit]
    elif manifest.get("catalog_binding"):
        candidates = [Path(manifest["catalog_binding"])]
    else:
        candidates = sorted(
            (ROOT / "training" / "catalog-bindings").glob("*.json")
        )
    for path in candidates:
        if not path.is_file():
            continue
        binding = json.loads(path.read_text())
        if (
            binding.get("schema") == "CatalogBinding.v1"
            and binding.get("catalog_sha256") == manifest["catalog_sha256"]
            and sha256_file(Path(binding["catalog_path"]))
            == binding["catalog_sha256"]
            and sha256_file(Path(binding["data_path"])) == binding["data_sha256"]
        ):
            return path
    live = create_catalog_binding()
    binding = json.loads(live.read_text())
    if binding["catalog_sha256"] != manifest["catalog_sha256"]:
        raise RuntimeError(
            "no historical CatalogBinding.v1 matches the frozen corpus"
        )
    return live


def ensure_packed(
    manifest_path: Path,
    *,
    catalog_binding_path: Path | None = None,
) -> dict:
    from test_exact_packed_corpus import materialize

    manifest = json.loads(manifest_path.read_text())
    binding = resolve_catalog_binding(manifest, catalog_binding_path)
    return materialize(
        manifest_path,
        ROOT / "training" / "packed",
        catalog_binding_path=binding,
    )


def publish_corpus_pointer(
    manifest_path: Path,
    packed_manifest: Path,
    watermark: int,
    *,
    root: Path = ROOT,
) -> dict:
    """Write training/current-corpus.json, but NEVER backwards.

    `jope_selfplay_trainer_watchdog.rs:594` hard-fails its cycle when this
    pointer regresses ("training corpus pointer regressed from X to Y"), so a
    single backwards write stops the DL trainer from ever being resubmitted.
    Regression is now structurally possible because two independent producers
    advance this file: the promotion tournament (at ITS pinned watermark) and the
    standalone freezer (at the live COUNT(*)). Monotonicity is enforced here, in
    the one function both of them go through, rather than in either of them.
    """
    path = root / "training" / "current-corpus.json"
    pointer = {
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "packed_manifest": str(packed_manifest),
        "packed_manifest_sha256": sha256_file(packed_manifest),
        "watermark": int(watermark),
    }
    if path.is_file():
        try:
            live = int(json.loads(path.read_text()).get("watermark") or 0)
        except (json.JSONDecodeError, TypeError, ValueError):
            live = 0
        if live > int(watermark):
            return {
                **pointer,
                "published": False,
                "live_watermark": live,
                "reason": "pointer is monotone; live watermark is ahead",
            }
    atomic_json(path, pointer)
    return {**pointer, "published": True}


def publish_corpus_freeze_state(
    *,
    watermark: int,
    stage: str,
    manifest_path: Path,
    catalog_binding_path: Path,
    packed_manifest_path: Path | None = None,
) -> dict:
    stages = {"rows_frozen": 0, "packed": 1, "published": 2}
    if stage not in stages:
        raise ValueError("invalid corpus freeze stage")
    path = (
        ROOT
        / "ops"
        / "corpus-freeze-state-v2"
        / f"watermark-{watermark:09d}.json"
    )
    prior = json.loads(path.read_text()) if path.is_file() else None
    if prior is not None and stages[prior["stage"]] > stages[stage]:
        return prior
    value = {
        "schema": "CorpusFreezeState.v2",
        "watermark": int(watermark),
        "catalog_binding_path": str(catalog_binding_path),
        "catalog_binding_sha256": sha256_file(catalog_binding_path),
        "selected_rows_path": str(manifest_path),
        "selected_rows_sha256": sha256_file(manifest_path),
        "packed_corpus_path": (
            str(packed_manifest_path) if packed_manifest_path is not None else None
        ),
        "packed_corpus_sha256": (
            sha256_file(packed_manifest_path)
            if packed_manifest_path is not None
            else None
        ),
        "stage": stage,
        "revision": int((prior or {}).get("revision", -1)) + 1,
    }
    atomic_json(path, value)
    return value
