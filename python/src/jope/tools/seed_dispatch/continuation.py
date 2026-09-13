"""Offline continuation qualification; never dispatches work or edits training labels."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


ARMS = ("neural", "generic", "shuffled")
ROLES = ("candidate", "control_generic", "control_shuffled", "gp_final", "replicate_panel", "reporting")
SEED_ORIGINS = ("neural_policy", "generic_comparison")
PROTOCOL_V2 = "parallel.seed_continuation.qualification.v2"
PROTOCOL_V3 = "parallel.seed_continuation.qualification.v3"


def plan(population, prepared, generic, *, prepared_sha, population_sha, generic_sha,
         evaluator_sha, vocabulary_sha, excluded_families, evaluations=2, seconds=6, rng_seed=1, candidate_seconds=2,
         arm=None, role=None, seed_origin="neural_policy", replicate=0,
         protocol=PROTOCOL_V2, generation=None, select=None, row0_source_evaluator_sha=None):
    # Successor-ledger extension: with the defaults the identity is byte-identical to v2.
    # arm/role/replicate/seed_origin/generation bind the episode role (v3 identity); select limits
    # episodes to the listed genomes or program digests; row0_source_evaluator_sha (Tier-B) records the
    # evaluator whose admitted row-0 label the seed reproduces when evaluator_sha names a different
    # executing binary. Never rewrites fitness or labels.
    if row0_source_evaluator_sha is not None and arm is None:
        raise ValueError("A row-0 source evaluator binding requires an arm")
    if seed_origin not in SEED_ORIGINS:
        raise ValueError("Seed origin must be neural_policy or generic_comparison")
    if arm is not None and (arm not in ARMS or role not in ROLES or replicate not in (0, 1)):
        raise ValueError("Episode role binding requires arm, role and replicate in {0,1}")
    if arm is None and (role is not None or replicate != 0 or generation is not None
                        or protocol != PROTOCOL_V2 or seed_origin != "neural_policy"):
        raise ValueError("Role, replicate, generation, protocol and seed-origin bindings require an arm")
    if generation is not None and (type(generation) is not int or generation < 0):
        raise ValueError("Generation binding must be a nonnegative integer")
    selected = None if select is None else set(select)
    if prepared["task"] != "regression" or prepared["meta_partition"] != "meta_train":
        raise ValueError("Only training regression cuts are admitted; qualification never opens sealed partitions")
    if any(prepared["family"].removeprefix("rdatasets-") == family.removeprefix("rdatasets-") for family in excluded_families):
        raise ValueError("Excluded family")
    if not prepared["clean_evaluation"]:
        raise ValueError("Requires clean preparation")
    if not 2 <= evaluations <= 200 or not 1 <= seconds <= 180 or not 1 <= candidate_seconds <= 30 or seconds < 2*candidate_seconds+2:
        raise ValueError("Qualification budget requires2..200 fits and sufficient time for two bounded attempts")
    if candidate_seconds != 2:
        raise ValueError("Shared-pool continuation requires explicit2s resource protocol")
    if not isinstance(rng_seed, int) or not 0 <= rng_seed < 2**64:
        raise ValueError("RNG must be u64")
    cut = next(c for c in prepared["cuts"] if c["index"] == population["cut"])
    if population["schema"] != "parallel.seed_population.v1" or population["prepared_sha256"] != prepared_sha or population["training_sha256"] != cut["training_sha256"]:
        raise ValueError("Population preparation/cut identity mismatch")
    if len(population["candidates"]) != 80:
        raise ValueError("Native interface requires 80 emitted candidates")
    generic_genomes = list(dict.fromkeys(generic))
    if len(generic_genomes) < 80 or any(not isinstance(g, str) or not g for g in generic_genomes):
        raise ValueError("Requires at least 80 distinct nonempty generic genomes")
    episodes, emissions, identities = [], [], {}
    for slot, candidate in enumerate(population["candidates"]):
        if candidate["origin"] != seed_origin:
            emissions.append(dict(slot=slot, status="not_neural_emission" if seed_origin == "neural_policy" else "not_seed_origin",
                                  origin=candidate["origin"]))
            continue
        genome = candidate.get("genome")
        if not isinstance(genome, str) or not genome:
            emissions.append(dict(slot=slot, status="invalid_empty_genome"))
            continue
        parent_sha = digest(genome.encode())
        if selected is not None and genome not in selected and parent_sha not in selected:
            emissions.append(dict(slot=slot, status="not_selected"))
            continue
        if genome not in identities:
            seed_population = {k: population[k] for k in ("schema", "prepared_sha256", "cut", "training_sha256", "context_sha256", "checkpoint_sha256")}
            seed_population["candidates"] = [dict(genome=genome, origin=seed_origin)] + [dict(genome=g, origin="generic_bootstrap") for g in generic_genomes if g != genome][:79]
            identity = dict(protocol=protocol, parent_program_sha256=parent_sha,
                producing_checkpoint_sha256=population["checkpoint_sha256"], source_population_sha256=population_sha,
                generic_population_sha256=generic_sha, prepared_sha256=prepared_sha, dataset=prepared["dataset"],
                family=prepared["family"], partition="meta_train", family_partition_frozen=prepared["family_partition_frozen"], cut=population["cut"],
                training_sha256=cut["training_sha256"], validation_sha256=cut["validation_sha256"],
                preprocessing_sha256=cut["preprocessing_sha256"], evaluator_sha256=evaluator_sha,
                vocabulary_sha256=vocabulary_sha, rng_seed=rng_seed, evaluations=evaluations, seconds=seconds,
                candidate_seconds=candidate_seconds, resource_protocol="single_candidate_2s_8gib_v1", seeds_sha256=digest(encoded(seed_population)), excluded_families=sorted(excluded_families),
                credit_semantics="population_value_only; offspring retain own execution targets",
                final_tests_opened=False)
            if arm is not None:
                identity.update(seed_arm=arm, role=role, replicate=replicate, seed_origin=seed_origin)
                if generation is not None:
                    identity["generation"] = generation
                if row0_source_evaluator_sha is not None:
                    identity["row0_source_evaluator_sha256"] = row0_source_evaluator_sha
            episode_id = digest(encoded(identity))
            identities[genome] = episode_id
            episodes.append(dict(episode_id=episode_id, identity=identity, seeds=seed_population,
                status="pending_native_grammar_validation", requires_protocol="forced_first_mutation_v2"))
        emissions.append(dict(slot=slot, status="mapped", episode_id=identities[genome]))
    return dict(schema="parallel.seed_continuation_plan.v1", canonical_deduplication="pending_native_parse",
                dispatch_authorized=False, emissions=emissions, episodes=episodes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    blobs = {k: Path(spec.pop(k)).read_bytes() for k in ("population", "prepared", "generic")}
    result = plan(*(json.loads(blobs[k]) for k in ("population", "prepared", "generic")),
                  prepared_sha=digest(blobs["prepared"]), population_sha=digest(blobs["population"]),
                  generic_sha=digest(blobs["generic"]), **spec)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "plan.json").write_bytes(encoded(result))
    for episode in result["episodes"]:
        directory = args.output / episode["episode_id"]
        directory.mkdir()
        (directory / "seeds.json").write_bytes(encoded(episode["seeds"]))
        (directory / "identity.json").write_bytes(encoded(episode["identity"]))


if __name__ == "__main__":
    main()
