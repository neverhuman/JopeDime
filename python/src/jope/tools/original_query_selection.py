"""Commit query selections from metadata; never authorize or read query outcomes.

The artifact is usable input to a future native selection/access authority.
Local byte consistency does not prove independent publication before outcomes,
original execution, family independence, or permission to open a query payload.
"""

import argparse
import hashlib
import json
import math
import os
import stat
import tempfile
import time
from pathlib import Path, PurePosixPath

SCHEMA = "parallel.original_query_selection.v1"
REQUEST_SCHEMA = "parallel.original_query_selection_request.v1"
PRODUCER_PATH = "parallel/tools/original_query_selection.py"
MAX_FILE = 8 * 1024 * 1024
MAX_TOTAL = 64 * 1024 * 1024
MAX_POOLS = 4096
QUERY_CONTRACT = dict(
    schema="parallel.query_contract.v1",
    task="regression",
    metric="rmse",
    units="original_target",
    outcome_policy="strict_finite_v1",
    feature_space="raw_features_v1",
)
ROLES = {"training_diagnostic": "meta_train", "promotion_validation": "meta_validation"}


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def exact(row, keys, label):
    require(isinstance(row, dict) and set(row) == set(keys), label + ": unknown or missing fields")


def unsigned(value):
    return type(value) is int and 0 <= value < 2**64


def sha256(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def text(value):
    return isinstance(value, str) and 0 < len(value) <= 512 and value.strip() == value


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def parts(name):
    require(
        isinstance(name, str) and 0 < len(name) <= 4096 and "\0" not in name,
        "bounded relative metadata path required",
    )
    path = PurePosixPath(name)
    require(
        not path.is_absolute()
        and path.as_posix() == name
        and not {"", ".", ".."} & set(path.parts)
        and path.suffix == ".json",
        "canonical workspace-relative JSON metadata path required",
    )
    require(
        not any(
            p.lower() in {"data", "raw", "meta_test", "development_validation"} or "sealed" in p.lower()
            for p in path.parts
        ),
        "raw or sealed/development payload path rejected before open",
    )
    return path.parts


def reference(row):
    exact(row, ("path", "sha256"), "metadata reference")
    parts(row["path"])
    require(sha256(row["sha256"]), "exact metadata digest required")
    return dict(row)


def signature(value):
    return tuple(
        getattr(value, key)
        for key in ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns", "st_ctime_ns")
    )


def decode(data):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate metadata JSON key")
            result[key] = value
        return result

    value = json.loads(data, object_pairs_hook=unique)
    canonical(value)  # Reject NaN, infinities and overflowing JSON exponents.
    return value


class Metadata:
    """Bounded regular files, reached through non-symlink directory descriptors."""

    def __init__(self, root):
        self.root = Path(root).absolute()
        require(
            not any(p.is_symlink() for p in (self.root, *self.root.parents)), "workspace symlink rejected"
        )
        self.fd = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        self.total = 0
        self.loaded = {}
        self.stats = {}

    def close(self):
        os.close(self.fd)

    def directory(self, name, create=False):
        components = parts(name)
        fd = os.dup(self.fd)
        try:
            for component in components[:-1]:
                if create:
                    try:
                        os.mkdir(component, 0o755, dir_fd=fd)
                        os.fsync(fd)
                    except FileExistsError:
                        pass
                child = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
                os.close(fd)
                fd = child
            return fd, components[-1]
        except BaseException:
            os.close(fd)
            raise

    def read(self, ref):
        ref = reference(ref)
        if ref["path"] in self.loaded:
            prior, value = self.loaded[ref["path"]]
            require(prior == ref, "one source path cannot name conflicting bytes")
            return value
        require(len(self.loaded) < MAX_POOLS + 8, "metadata reference count exceeded")
        parent, name = self.directory(ref["path"])
        try:
            fd = os.open(name, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW, dir_fd=parent)
            with os.fdopen(fd, "rb") as stream:
                before = os.fstat(stream.fileno())
                require(
                    stat.S_ISREG(before.st_mode) and before.st_size <= MAX_FILE,
                    "bounded regular metadata required",
                )
                data = stream.read(MAX_FILE + 1)
                require(
                    len(data) <= MAX_FILE
                    and signature(before)
                    == signature(os.fstat(stream.fileno()))
                    == signature(os.stat(name, dir_fd=parent, follow_symlinks=False)),
                    "metadata changed during capture",
                )
        finally:
            os.close(parent)
        self.total += len(data)
        require(self.total <= MAX_TOTAL, "aggregate metadata bound exceeded")
        require(hashlib.sha256(data).hexdigest() == ref["sha256"], "original metadata bytes changed")
        value = decode(data)
        self.loaded[ref["path"]] = (ref, value)
        self.stats[ref["path"]] = signature(before)
        return value

    def publish(self, name, value):
        require(name not in self.loaded, "selection cannot overwrite an input")
        data = canonical(value) + b"\n"
        require(len(data) <= MAX_FILE, "selection artifact exceeds metadata bound")
        for path, observed in self.stats.items():
            directory, leaf = self.directory(path)
            try:
                require(
                    signature(os.stat(leaf, dir_fd=directory, follow_symlinks=False)) == observed,
                    "metadata changed before selection publication",
                )
            finally:
                os.close(directory)
        parent, leaf = self.directory(name, create=True)
        staging = None
        try:
            # TemporaryFile creates a fresh inode; link through the pinned parent
            # directory publishes complete bytes without replacing any old entry.
            with tempfile.NamedTemporaryFile(dir="/proc/self/fd/" + str(parent), delete=False) as f:
                staging = Path(f.name).name
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
                os.fchmod(f.fileno(), 0o444)
            os.link(staging, leaf, src_dir_fd=parent, dst_dir_fd=parent, follow_symlinks=False)
            os.fsync(parent)
        finally:
            if staging is not None:
                os.unlink(staging, dir_fd=parent)
            os.close(parent)
        return hashlib.sha256(data).hexdigest()


def source_binding(store, ref):
    release = store.read(ref)
    require(
        isinstance(release, dict) and release.get("schema") == "parallel.registry_source_freeze.v1",
        "original source-freeze metadata required",
    )
    files = release.get("files")
    require(
        isinstance(files, dict)
        and 0 < len(files) <= 20000
        and release.get("source_closure_sha256") == digest(files),
        "source closure identity changed",
    )
    for name, row in files.items():
        require(
            isinstance(name, str)
            and not PurePosixPath(name).is_absolute()
            and ".." not in PurePosixPath(name).parts,
            "invalid source-closure member",
        )
        exact(row, ("sha256", "bytes", "executable"), "source member")
        require(
            sha256(row["sha256"]) and unsigned(row["bytes"]) and type(row["executable"]) is bool,
            "invalid source-closure metadata",
        )
    with Path(__file__).open("rb") as stream:
        code = stream.read(256 * 1024 + 1)
    require(len(code) <= 256 * 1024, "selection producer source bound exceeded")
    own = hashlib.sha256(code).hexdigest()
    require(
        files.get(PRODUCER_PATH, {}).get("sha256") == own and files[PRODUCER_PATH]["bytes"] == len(code),
        "selection producer differs from frozen source release",
    )
    return dict(source_closure_sha256=release["source_closure_sha256"], producer_sha256=own)


def contexts(store, request):
    index = store.read(request["context_index"])
    inventory = store.read(request["family_inventory"])
    require(
        isinstance(index, dict)
        and index.get("schema") == "parallel.context_index.v1"
        and type(index.get("dimensions")) is int
        and index["dimensions"] == 2048
        and type(index.get("qualification")) is bool
        and sha256(index.get("assembly_sha256")),
        "native context index required",
    )
    require(
        isinstance(inventory, dict)
        and inventory.get("schema") == "parallel.family_contamination_inventory.v1"
        and inventory.get("context_index_sha256") == request["context_index"]["sha256"]
        and inventory.get("final_tests_opened") is False
        and inventory.get("csv_or_target_values_opened") is False
        and inventory.get("known_conflicts") == []
        and unsigned(inventory.get("unresolved_memberships")),
        "family inventory binding or conflict",
    )
    rows, prepared = index.get("encodings"), inventory.get("prepared")
    require(
        isinstance(rows, list)
        and len(rows) <= MAX_POOLS
        and isinstance(prepared, list)
        and len(prepared) <= MAX_POOLS,
        "bounded context metadata required",
    )
    members, sources, families = {}, {}, {}
    for row in prepared:
        require(
            isinstance(row, dict)
            and text(row.get("family"))
            and text(row.get("dataset"))
            and text(row.get("prepared"))
            and sha256(row.get("prepared_sha256"))
            and sha256(row.get("source_sha256")),
            "original family/source identity absent",
        )
        family, partition, source = row["family"], row.get("partition"), row["source_sha256"]
        require(
            partition in ("meta_train", "meta_validation", "meta_test", "development_validation"),
            "unknown family partition",
        )
        require(
            families.setdefault(family, partition) == partition
            and sources.setdefault(source, (family, partition)) == (family, partition),
            "family/source crosses frozen partitions",
        )
        require(row["prepared_sha256"] not in members, "duplicate or ambiguous preparation")
        members[row["prepared_sha256"]] = row
    available = {}
    for row in rows:
        require(
            isinstance(row, dict) and unsigned(row.get("cut")) and sha256(row.get("prepared_sha256")),
            "invalid native context identity",
        )
        key = (row["prepared_sha256"], row["cut"])
        require(key not in available, "duplicate native context")
        member = members.get(key[0])
        require(
            member is not None
            and all(row.get(k) == member.get(k) for k in ("dataset", "family", "partition", "prepared")),
            "context/family inventory conflict",
        )
        cuts = member.get("cuts")
        require(isinstance(cuts, list), "original cut metadata missing")
        by_cut = {}
        for cut in cuts:
            require(
                isinstance(cut, dict) and unsigned(cut.get("index")) and cut["index"] not in by_cut,
                "ambiguous original fold identity",
            )
            by_cut[cut["index"]] = cut
        cut = by_cut.get(key[1])
        require(
            cut is not None
            and all(
                sha256(cut.get(k)) for k in ("training_sha256", "validation_sha256", "preprocessing_sha256")
            )
            and sha256(row.get("sha256"))
            and sha256(row.get("receipt_sha256")),
            "original context/fold byte binding absent",
        )
        available[key] = dict(
            dataset=row["dataset"],
            family=row["family"],
            partition=row["partition"],
            prepared_sha256=key[0],
            cut=key[1],
            source_sha256=member["source_sha256"],
            training_sha256=cut["training_sha256"],
            validation_sha256=cut["validation_sha256"],
            preprocessing_sha256=cut["preprocessing_sha256"],
            context_sha256=row["sha256"],
            encoding_receipt_sha256=row["receipt_sha256"],
            assembly_sha256=index["assembly_sha256"],
        )
    chosen = []
    require(
        isinstance(request["contexts"], list) and 0 < len(request["contexts"]) <= 512,
        "bounded selected context identities required",
    )
    seen = set()
    for row in request["contexts"]:
        exact(row, ("prepared_sha256", "cut"), "selected context")
        require(sha256(row["prepared_sha256"]) and unsigned(row["cut"]), "invalid selected context")
        key = (row["prepared_sha256"], row["cut"])
        require(key not in seen and key in available, "duplicate or unindexed selected context")
        seen.add(key)
        value = available[key]
        require(
            value["partition"] == ROLES[request["role"]],
            "query role cannot select sealed or wrong-partition context",
        )
        chosen.append(value)
    return sorted(chosen, key=lambda row: (row["family"], row["dataset"], row["cut"], row["prepared_sha256"]))


def build(store, request_ref):
    request_ref = reference(request_ref)
    request = store.read(request_ref)
    exact(
        request,
        (
            "schema",
            "experiment_id",
            "scope",
            "role",
            "source_release",
            "context_index",
            "family_inventory",
            "evaluation",
            "arms",
            "rng_seeds",
            "contexts",
            "programs_per_pool",
            "pools",
        ),
        "selection request",
    )
    require(
        request["schema"] == REQUEST_SCHEMA
        and request["role"] in ROLES
        and text(request["experiment_id"])
        and text(request["scope"]),
        "selection request role or identity invalid",
    )
    for name in ("source_release", "context_index", "family_inventory"):
        reference(request[name])
    evaluation = request["evaluation"]
    exact(
        evaluation,
        (
            "evaluator_sha256",
            "runtime_sha256",
            "numerical_contract_sha256",
            "vocabulary_sha256",
            "query_contract",
        ),
        "evaluation metadata",
    )
    require(
        evaluation["query_contract"] == QUERY_CONTRACT
        and all(sha256(value) for key, value in evaluation.items() if key.endswith("_sha256")),
        "unsupported or incomplete evaluation contract",
    )
    release = source_binding(store, request["source_release"])
    selected_contexts = contexts(store, request)  # Must reject role/family errors before pools open.
    seeds = request["rng_seeds"]
    require(
        isinstance(seeds, list)
        and 0 < len(seeds) <= 64
        and all(unsigned(seed) for seed in seeds)
        and len(set(seeds)) == len(seeds),
        "distinct native unsigned evaluation seeds required",
    )
    arms = {}
    require(
        isinstance(request["arms"], list) and 0 < len(request["arms"]) <= 16, "bounded original arms required"
    )
    for arm in request["arms"]:
        exact(arm, ("name", "parent_checkpoint_sha256", "original_parent_receipt"), "arm")
        require(
            text(arm["name"])
            and arm["name"] != "shuffled"
            and arm["name"] not in arms
            and sha256(arm["parent_checkpoint_sha256"]),
            "invalid or duplicate arm",
        )
        reference(arm["original_parent_receipt"])  # Pointer only; never opens outcome metadata.
        arms[arm["name"]] = arm
    limit = request["programs_per_pool"]
    require(unsigned(limit) and 0 < limit <= 256, "bounded candidate count required")
    expected = {
        (c["prepared_sha256"], c["cut"], arm, seed): c
        for c in selected_contexts
        for arm in arms
        for seed in seeds
    }
    require(
        len(expected) <= MAX_POOLS
        and isinstance(request["pools"], list)
        and len(request["pools"]) == len(expected),
        "incomplete or oversized context/arm/seed grid",
    )
    pools = {}
    for pool in request["pools"]:
        exact(
            pool,
            ("prepared_sha256", "cut", "arm", "rng_seed", "population", "original_proposal_receipt"),
            "pool",
        )
        require(
            sha256(pool["prepared_sha256"])
            and unsigned(pool["cut"])
            and unsigned(pool["rng_seed"])
            and text(pool["arm"]),
            "invalid pool coordinates",
        )
        key = (pool["prepared_sha256"], pool["cut"], pool["arm"], pool["rng_seed"])
        require(key in expected and key not in pools, "duplicate or unselected pool")
        reference(pool["population"])
        reference(pool["original_proposal_receipt"])  # Not an access capability.
        pools[key] = pool
    selected = []
    for key in sorted(pools):
        pool, context = pools[key], expected[key]
        population = store.read(pool["population"])
        exact(
            population,
            (
                "schema",
                "prepared_sha256",
                "cut",
                "training_sha256",
                "context_sha256",
                "checkpoint_sha256",
                "candidates",
            ),
            "native seed population",
        )
        require(
            population["schema"] == "parallel.seed_population.v1"
            and unsigned(population["cut"])
            and population["prepared_sha256"] == context["prepared_sha256"]
            and population["cut"] == context["cut"]
            and population["training_sha256"] == context["training_sha256"]
            and population["context_sha256"] == context["context_sha256"]
            and population["checkpoint_sha256"] == arms[key[2]]["parent_checkpoint_sha256"],
            "candidate pool context or parent differs from frozen grid",
        )
        require(
            isinstance(population["candidates"], list) and limit <= len(population["candidates"]) <= 256,
            "insufficient or oversized candidate pool",
        )
        candidates = {}
        for slot, row in enumerate(population["candidates"]):
            require(
                isinstance(row, dict)
                and {"genome", "origin"} <= set(row)
                and not set(row) - {"genome", "origin", "rank_log_weight", "policy_logprob"}
                and not {"rank_log_weight", "policy_logprob"} <= set(row),
                "candidate contains outcomes, conflicting aliases or unknown fields",
            )
            require(
                isinstance(row["genome"], str)
                and 0 < len(row["genome"].encode()) <= 32768
                and text(row["origin"]),
                "invalid program metadata",
            )
            weight = row.get("rank_log_weight", row.get("policy_logprob"))
            require(
                weight is None
                or (type(weight) in (int, float) and abs(weight) <= 1e300 and math.isfinite(weight)),
                "invalid diagnostic rank weight",
            )
            program_sha = hashlib.sha256(row["genome"].encode()).hexdigest()
            require(program_sha not in candidates, "duplicate program or conflicting origin in original pool")
            candidates[program_sha] = dict(
                program_sha256=program_sha, genome=row["genome"], origin=row["origin"], original_slot=slot
            )
        for program_sha in sorted(candidates)[:limit]:
            row = dict(
                context=context,
                arm=key[2],
                rng_seed=key[3],
                parent_checkpoint_sha256=population["checkpoint_sha256"],
                original_parent_receipt=arms[key[2]]["original_parent_receipt"],
                population=pool["population"],
                original_proposal_receipt=pool["original_proposal_receipt"],
                **candidates[program_sha],
            )
            row["selection_key"] = digest(
                dict(
                    experiment_id=request["experiment_id"],
                    scope=request["scope"],
                    role=request["role"],
                    evaluation=evaluation,
                    candidate=row,
                )
            )
            selected.append(row)
    return dict(
        schema=SCHEMA,
        request=request_ref,
        experiment_id=request["experiment_id"],
        scope=request["scope"],
        role=request["role"],
        evaluation=evaluation,
        source=release,
        input_artifacts=sorted((ref for ref, _ in store.loaded.values()), key=lambda ref: ref["path"]),
        selection_rule="all_requested_context_arm_seed_cells; canonical_program_sha256_first_v1",
        contexts=selected_contexts,
        arms=[arms[name] for name in sorted(arms)],
        rng_seeds=sorted(seeds),
        programs_per_pool=limit,
        candidates=selected,
        original_receipt_execution_verified=False,
        independent_publication_order_verified=False,
        independent_family_membership_verified=False,
        query_access_allowed=False,
        unavailable_requirements=[
            "native original proposal/parent receipt and family membership joins",
            "independent witness commit of this selection before outcome access",
            "native query payload descriptor, original completion, role and access authorization joins",
        ],
    )


def freeze(root, request_ref, output):
    store = Metadata(root)
    try:
        value = build(store, request_ref)
        value["local_publication"] = dict(
            observed_unix_ns=time.time_ns(),
            observed_monotonic_ns=time.monotonic_ns(),
            independent_clock_verified=False,
        )
        observed = store.publish(output, value)
        return dict(
            selection=dict(path=output, sha256=observed),
            candidates=len(value["candidates"]),
            status="metadata_selection_committed",
            query_access_allowed=False,
        )
    finally:
        store.close()


def verify(root, selection_ref):
    store = Metadata(root)
    try:
        value = store.read(selection_ref)
        require(isinstance(value, dict), "selection artifact required")
        publication = value.get("local_publication")
        exact(
            publication,
            ("observed_unix_ns", "observed_monotonic_ns", "independent_clock_verified"),
            "local publication observation",
        )
        require(
            unsigned(publication["observed_unix_ns"])
            and unsigned(publication["observed_monotonic_ns"])
            and 0 < publication["observed_unix_ns"] <= time.time_ns()
            and publication["observed_monotonic_ns"] > 0
            and publication["independent_clock_verified"] is False,
            "invalid local clock observation",
        )
        # Output is not an input to its own derivation; never accept a circular hash join.
        del store.loaded[selection_ref["path"]]
        expected = build(store, value.get("request"))
        require(
            value == dict(expected, local_publication=publication),
            "selection differs from original metadata derivation",
        )
        return dict(
            status="metadata_selection_verified",
            selection=reference(selection_ref),
            candidates=len(expected["candidates"]),
            query_access_allowed=False,
            independent_publication_order_verified=False,
            original_receipt_execution_verified=False,
        )
    finally:
        store.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("freeze", "verify"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    ref = dict(path=args.input, sha256=args.sha256)
    if args.operation == "freeze":
        require(args.output is not None, "new selection output required")
        result = freeze(args.root, ref, args.output)
    else:
        require(args.output is None, "verification never writes a selection")
        result = verify(args.root, ref)
    print(canonical(result).decode())
