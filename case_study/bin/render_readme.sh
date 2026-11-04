#!/usr/bin/env bash
# render_readme.sh <case_dir> : write <case_dir>/README.md from case.json and objective.json.
# Text below the "<!-- notes -->" marker of an existing README (results, business context added by
# hand) is preserved.
set -euo pipefail
d=$(cd "${1:?usage: render_readme.sh <case_dir>}" && pwd); spec=$d/case.json
notes=""; [ -f "$d/README.md" ] && notes=$(sed -n '/^<!-- notes -->$/,$p' "$d/README.md" | tail -n +2)
objective=$( [ -f "$d/objective.json" ] && jq . "$d/objective.json" || echo '(not yet calibrated: run case_study/bin/calibrate.sh on this case)')
{
jq -r '"# \(.name | sub("^[0-9]+_"; "") | gsub("_"; " ") | (.[0:1] | ascii_upcase) + .[1:])\n",
  "> \(.business_question)\n",
  "| Dataset | |", "| --- | --- |",
  "| Catalog id | `\(.dataset.id)` |",
  "| Name | \(.dataset.display_name // "unnamed in catalog") |",
  "| Source | \(.dataset.source.platform // "?") `\(.dataset.source.id // "?")`\(if .dataset.source.url then " — <" + .dataset.source.url + ">" else "" end) |",
  "| Shape | \(.dataset.rows) rows × \(.dataset.features) features (\(.dataset.categorical_features) categorical); train \(.dataset.train_rows), test \(.dataset.test_rows) |",
  "| Licence | \(.dataset.source.license // "not stated") |",
  "| Time column | \(.dataset.time_column // "none") |",
  "| Sealed check | `case_study/bin/check_sealed.sh \(.dataset.id)` → CLEAR |",
  "",
  "## Why this objective\n", .why, "",
  "Business constraint: **\(.constraint)**. Accuracy metric: `\(.accuracy | tojson)`.\n"' "$spec"
echo '## Objective (`objective.json`)'; echo; echo '```json'; echo "$objective"; echo '```'; echo
jq -r 'if .calibration.status == "calibrated" then
  "Calibrated from arm D on rng seed \(.calibration.rng_seed) (\(.calibration.medians.candidates) measured candidates; not a main seed). Rule: \(.calibration.rules). Train target σ = \(.calibration.target_sigma).\n"
  else "Calibration pending: `case_study/bin/calibrate.sh case_study/\(.name)` sets the limits/prices from a declared calibration run before the main runs.\n" end' "$spec"
cat <<'MD'
`baseline.json` ranks by the same accuracy metric alone, with non-binding limits, so the same
resource readings are recorded for every candidate in every arm.

## Arms and budget

MD
jq -r '"| Arm | Generation 0 | Ranked by |", "| --- | --- | --- |",
  (.arms | to_entries[] | "| \(.key) | \(if .value.seeds then "bootstrap + 11 random + DL seeds (≤ 11)" else "bootstrap + 11 random" end) | `\(if .value.fitness == "objective" then "objective.json" else "baseline.json" end)` |"),
  "", "Budget \(.budget.evaluations) evaluations (≈ generation 0 + 8–10 generations), \(.budget.seconds) s per run; RNG seeds \(.rng_seeds | map(tostring) | join(", ")); `--threads 1`.\n"' "$spec"
cat <<MD
## Run

\`\`\`sh
case_study/bin/run_case.sh case_study/$(basename "$d") --smoke    # one seed, tiny budget
case_study/bin/run_case.sh case_study/$(basename "$d")            # all arms and seeds
\`\`\`

Results land in \`~/jope-data/case_study/$(basename "$d")/<UTC>/\`: \`SUMMARY.md\` (champion per run,
best-so-far objective by generation) and \`generations.csv\`.

## How to read the result

- **A vs C**: the objective's effect. Does the champion become cheaper (faster, smaller,
  lower-latency) at a stated accuracy cost?
- **A vs B**: the generation-0 seeds' effect under the business objective.
- A run marked \`resource_infeasible\` in \`observations.json\` was refused by a limit, never
  ranked.

## Caveats

MD
jq -r '.caveats[] | "- " + .' "$spec"
echo "- Measured readings are host- and load-dependent; \`run.json\` records the host, load and limits of each run."
echo
echo "<!-- notes -->"
if [ -n "$notes" ]; then echo "$notes"; fi
} > "$d/README.md.tmp" && mv "$d/README.md.tmp" "$d/README.md"
echo "render_readme: $d/README.md"
