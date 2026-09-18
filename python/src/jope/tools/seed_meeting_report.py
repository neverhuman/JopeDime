#!/usr/bin/env python3
"""Render the frozen qualification receipts into a concise meeting readout."""
import argparse
import collections
import json
from pathlib import Path

from seed_experiment import digest, read, write


def percent(value):
    return "incomplete" if value is None else f"{100*(value-1):+.2f}%"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--experiment", type=Path, default=Path("runs/seed-quality-experiment-v1"))
    parser.add_argument("--binary-experiment", type=Path, default=Path("runs/binary-qualification-v1"))
    args = parser.parse_args()
    root = args.root.resolve()
    output, binary_output = root / args.experiment, root / args.binary_experiment
    manifest, report, binary = [read(p) for p in (output/"manifest.json", output/"report.json", binary_output/"report.json")]
    deterministic_path=binary_output/"deterministic-v2/report.json"
    deterministic=read(deterministic_path) if deterministic_path.exists() else None
    rankings = [read(output/f"proposals/{v['dataset']}/cut-{v['cut']}/rankings.json") for v in manifest["views"]]
    origins = collections.Counter(origin["kind"] for r in rankings for c in r["candidate_pool"]
        if c["program"] in r["rankings"]["larger_pool"][:8] for origin in c["origins"])
    bank = read(output/"frozen-bank.json")
    source_paths = [output/"manifest.json", output/"report.json", output/"rankings-frozen.json",
                    output/"frozen-bank.json", binary_output/"report.json", binary_output/"preparation-verification.json"]
    optional = [output/"methodology.json", output/"downstream-report.json", output/"cancellation-evidence.json",
                output/"qualification-invariants.json", binary_output/"restart-verification.json", binary_output/"costs.json",
                binary_output/"artifact-verification.json", binary_output/"reproducibility/verification.json",
                binary_output/"reproducibility/numerical-review.json", deterministic_path,
                binary_output/"deterministic-v2/pair-qualification.json", binary_output/"deterministic-v2/launch-policy.json"]
    source_paths.extend(p for p in optional if p.exists())
    audit = dict(schema="parallel.seed_meeting_readout.v1", sources=[dict(path=str(p.relative_to(root)),sha256=digest(p)) for p in source_paths],
        production_seeder_changed=False, positive_meeting_result=report["positive_meeting_result"],
        generated_views=len(rankings), bank_programs=bank["count"], larger_top8_origin_slots=dict(origins),
        minimum_pool=min(len(r["candidate_pool"]) for r in rankings), maximum_pool=max(len(r["candidate_pool"]) for r in rankings),
        reward_max_abs_error=max(r["reward_runtime_parity"]["reward_max_abs_error"] for r in rankings),
        runtime_max_abs_error=max(r["reward_runtime_parity"]["runtime_max_abs_error"] for r in rankings),
        selected_order_identical=all(r["reward_runtime_parity"]["selected_order_identical"] for r in rankings),
        binary_completed_contexts=binary["completed_contexts"], binary_source_families=binary["admitted_datasets"],
        binary_deterministic_publication=deterministic,
        binary_preferred_index=deterministic["context_index"] if deterministic else binary["context_index"])
    write(output/"meeting.json", audit)
    lines = ["# Seed quality and binary encoding qualification", "",
        "The larger-pool selector did not meet the deployment threshold; the production seeder remains unchanged."
        if not report["positive_meeting_result"] else "The larger-pool selector met the qualification threshold. The production seeder remains unchanged.", "",
        f"Primary best-of-eight RMSE changed by {percent(report['comparisons']['larger_pool_vs_current']['best8']['family_balanced_ratio'])} "
        f"versus current selection and {percent(report['comparisons']['larger_pool_vs_generic']['best8']['family_balanced_ratio'])} versus generic seeds. "
        "The predeclared gate required at least 2% lower RMSE against both, wins on at least five families, and no downstream aggregate regression.", "",
        f"Frozen checkpoint: `{manifest['selected_checkpoint_source']}`. Its corpus contains {manifest['corpus_records']:,} records. "
        "Two unused decision cuts from each of eight previously used validation families were selected after reconstructing the checkpoint-selection panels. These are qualification results, not an independent final test.", "",
        "| Ranking | Top-one RMSE vs generic | Best-of-eight RMSE vs generic | Best-eight family wins / ties / losses |",
        "|---|---:|---:|---:|"]
    for arm in ("current","larger_pool","earlier_checkpoint","shuffled_context","random"):
        c=report["comparisons"][f"{arm}_vs_generic"]
        b=c["best8"]
        lines.append(f"| {arm.replace('_',' ')} | {percent(c['best1']['family_balanced_ratio'])} ({c['best1']['families_completed']}/8 families) | {percent(b['family_balanced_ratio'])} ({b['families_completed']}/8 families) | {b['wins']} / {b['ties']} / {b['losses']} |")
    lines.extend(["| generic | reference | reference | reference |", "",
        "Ratios are geometric means of per-cut ratios, with equal family weight. Lower is better. " + report["endpoint_policy"], "",
        "Failure aggregation was clarified after outcomes had been written. The original all-success sensitivity report is preserved as `report-all-success-v1.json`; rankings and the deployment threshold were frozen before outcomes.", "",
        "| Family | Larger / current best-eight ratio | Larger / generic best-eight ratio |", "|---|---:|---:|"])
    current=report["comparisons"]["larger_pool_vs_current"]["best8"]
    generic=report["comparisons"]["larger_pool_vs_generic"]["best8"]
    for family in sorted(current["family_ratios"]):
        lines.append(f"| {family} | {current['family_ratios'][family]:.4f} | {generic['family_ratios'].get(family, float('nan')):.4f} |")
    coverage=report["candidate_coverage"]
    lines.extend(["",f"Primary coverage: {coverage['completed_unique_top8']}/{coverage['required_unique_top8']} distinct cut/program attempts; "
        f"{report.get('top8_physical_fits',report['physical_fits'])} successful fits and {report.get('top8_failures',report['failures'])} failures. "
        "Candidate timeout: 30 seconds; worker address-space limit: 8 GiB. Exact results are reused across controls.", "",
        f"Primary elapsed time: {report.get('primary_elapsed_seconds',report['elapsed_seconds']):.2f} seconds; "
        f"primary candidate time: {report.get('primary_candidate_seconds',report['candidate_seconds']):.2f} aggregate seconds. "
        f"All-prefix physical attempts: {report['physical_completed_attempts']}; all-prefix successful fits: {report['physical_fits']}.", "",
        f"All-prefix logical coverage: {report.get('logical_evaluations_by_prefix', {})}; requested: {report.get('requested_logical_evaluations_by_prefix', {str(n): len(rankings)*6*n for n in (1,8,16,64)})}. "
        f"Terminal unique coverage: {coverage['completed_unique']}/{coverage['required_unique_top64']}; failures: {report['failures']}. "
        f"Secondary invocation elapsed time, including scheduler waits: {report.get('secondary_elapsed_seconds', 0):.2f} seconds. "
        f"All-prefix aggregate candidate time: {report.get('physical_candidate_seconds', report['candidate_seconds']):.2f} seconds; "
        f"cancelled candidate attempts: {report.get('cancelled_attempts', 0)}.", "",
        "Best-of-16/64 are secondary endpoints; only complete successful prefixes contribute, so ratios on subsets cannot replace the primary result.", "",
        "| Secondary endpoint | Larger vs current | Complete families | Larger vs generic | Complete families |",
        "|---|---:|---:|---:|---:|"])
    for n in (16,64):
        a=report['comparisons']['larger_pool_vs_current'][f'best{n}']
        b=report['comparisons']['larger_pool_vs_generic'][f'best{n}']
        lines.append(f"| Best-of-{n} | {percent(a['family_balanced_ratio'])} | {a['families_completed']}/8 | {percent(b['family_balanced_ratio'])} | {b['families_completed']}/8 |")
    lines.extend(["",
        f"Each larger pool contains {audit['minimum_pool']}–{audit['maximum_pool']} unique candidates, including a frozen bank of {bank['count']} exact, representable training champions. "
        f"Top-eight origin slots: {dict(origins)}. These describe program origin; they do not establish a causal neural improvement. "
        f"All {len(rankings)} checkpoint parity checks have maximum reward/runtime error {audit['reward_max_abs_error']}/{audit['runtime_max_abs_error']} and unchanged ordering.", ""])
    proposal_costs=report.get("proposal_costs", {})
    if proposal_costs:
        lines.extend([f"Summed per-view generation time: {proposal_costs.get('generation_seconds', 0):.3f} seconds; "
            f"scoring time: {proposal_costs.get('scoring_seconds', 0):.3f} seconds. These measurements exclude checkpoint/corpus freezing and process startup.", ""])
    downstream=report.get("downstream")
    if downstream:
        lines.append("Downstream GP used identical 10 retained / 6 exploration tails, RNG seeds, and 200-evaluation / 180-second limits per search.")
        lines.append(f"Completed searches: {downstream['completed_views']}/{downstream['required_views']}; "
            f"physical evaluations: {downstream['physical_evaluations']:,}; cancellations: {downstream['cancellations']}; "
            f"resumed searches: {downstream.get('resumed_searches', 0)}. "
            f"Aggregate native search clock: {downstream['elapsed_search_seconds']:.2f} seconds, including "
            f"{downstream.get('resume_unobserved_allowance_seconds', 0):.2f} seconds of conservative interrupted-work allowances.")
        if "physical_fits" in downstream:
            lines.append(f"Successful downstream fits: {downstream['physical_fits']:,}; failed evaluations: {downstream['failed_evaluations']} "
                f"({downstream['failure_codes']}); committed candidate runtime: {downstream['candidate_seconds_observed']:.2f} seconds.")
        for baseline, comparison in downstream.get("comparisons",{}).items():
            lines.append(f"Larger versus {baseline}: {percent(comparison.get('family_balanced_ratio'))}; complete={comparison.get('complete')}; "
                f"family wins/ties/losses={comparison.get('wins', 'pending')}/{comparison.get('ties', 'pending')}/{comparison.get('losses', 'pending')}.")
        lines.append(f"Winning-program origins by arm: {downstream.get('winner_origins_by_arm', {})}. "
            f"Direct selected-seed improvements: {downstream.get('direct_neural_selected_winner_improvements', {})}. "
            "These counts exclude retained teachers and evolved descendants and are operational attribution, not proof of a causal neural gain.")
    else:
        lines.append("Downstream GP completion is pending; see `downstream-report.json` when available.")
    lines.extend(["", f"Binary encoding completed {binary['completed_contexts']}/{binary['admitted_contexts']} contexts from Adult, German credit, Ionosphere, and Sonar. "
        f"Components: {binary.get('components',{})}. Cuts retain source references, row indices, and training-only preprocessing; no cut tables were copied. "
        "The frozen regression projection was reused in a separate binary qualification index. Sonar acquisition/entity grouping remains unresolved beyond preserved duplicate groups. Original final tests remain unopened.", ""])
    if deterministic:
        lines.extend([f"The preferred binary qualification index is `{deterministic['context_index']}` with {deterministic['completed_contexts']}/80 contexts. "
            f"Serial dispatch of the same frozen Hyperion encoder passed independent byte-exact replay on {deterministic['pair_views']} views "
            f"(maximum errors {deterministic['pair_max_abs_error']}). Target and foundation were already reproduced exactly and are reused. "
            "The original batched outputs and failed exact-check receipt remain preserved. This verifies 20 independent replay pairs, not a fresh repeat of all 80.", ""])
    lines.extend([
        "Next priorities remain strict exact-result caching and overlapping predictor export with corpus preparation. The measured seed result does not justify deploying the larger-pool selector.", "",
        "Reproducibility: `manifest.json`, `rankings-frozen.json`, `frozen-bank.json`, `report.json`, `methodology.json`, and `meeting.json` bind the inputs, rankings, measurement policy, and outputs.", ""])
    if (binary_output/"costs.json").exists():
        costs=read(binary_output/"costs.json")
        lines.extend([f"Binary pipeline time: {costs['initial_pipeline_seconds']:.2f} seconds for the initial 20; "
            f"{costs['expanded_pipeline_seconds']:.2f} seconds for expansion to 80. Failed dependency setup attempts consumed "
            f"{costs['failed_setup_attempt_seconds']:.2f} observed seconds; package installation time was not measured. "
            "Restart verification reused completed artifacts without GPU admission.", ""])
    if (binary_output/"reproducibility/verification.json").exists():
        replay=read(binary_output/"reproducibility/verification.json")
        lines.extend([f"Original batched-mode recomputation: {replay['independently_recomputed_views']} views and "
            f"{replay['independently_recomputed_raw_components']} raw components; byte-identical={replay['all_outputs_byte_identical']}; "
            f"maximum absolute errors={replay['max_abs_error']}; elapsed stage time={replay['encoding_seconds']:.2f} seconds.", ""])
    if deterministic:
        lines.extend([f"Serial verification and deterministic publication consumed {deterministic['seconds']:.2f} admitted-stage seconds. "
            "The launch policy and unchanged encoder hashes are bound in the deterministic index; no projection was refitted and no cut tables were copied.", ""])
    (output/"meeting.md").write_text("\n".join(lines))
    print(json.dumps(audit))


if __name__ == "__main__":
    main()
