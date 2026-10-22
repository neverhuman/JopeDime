#!/usr/bin/env bash
# Derive the frontier-comparison plot tables from the archived match matrix.
#
#   scripts/derive-frontier.sh OUTPUT_DIR
#
# Input: artifacts/benchmark/match-matrix.txt, 64 rows (families) by 6 columns (pairs). Column k is
# the first entrant's seed-averaged match score in pair k, in RESULTS.json order:
#   Astra-Sol, Astra-Gemini, Astra-Opus, Sol-Gemini, Sol-Opus, Gemini-Opus.
# Outputs (committed copies live in data/; scripts/check.sh regenerates and compares them):
#   frontier-pairwise.csv      row entrant, column entrant, family-weighted score, majority-family wins
#   frontier-family-strip.csv  one row per family and entrant: Copeland score (0-3) and best-of-four flag
#   frontier-ratios.csv        the six geometric-mean RMSE ratios of RESULTS.json, in its order
# The diagonal of frontier-pairwise.csv is written as score 0.5 with majority -1 (drawn blank).
set -euo pipefail
paper=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
out=${1:?usage: derive-frontier.sh OUTPUT_DIR}
mkdir -p -- "$out"
matrix="$paper/artifacts/benchmark/match-matrix.txt"
LC_ALL=C awk -v pairwise="$out/frontier-pairwise.csv" -v strip="$out/frontier-family-strip.csv" '
  BEGIN { split("Opus Gemini Sol Astra", name, " ")   # plot order (y = 1..4)
          # pair k: first entrant index, second entrant index (plot order)
          split("4 3|4 2|4 1|3 2|3 1|2 1", P, "|")
          for (k = 1; k <= 6; k++) { split(P[k], ab, " "); A[k] = ab[1]; B[k] = ab[2] } }
  NF == 0 { next }
  NF != 6 { printf "match matrix row %d has %d fields\n", NR, NF > "/dev/stderr"; exit 2 }
  { f++
    for (e = 1; e <= 4; e++) s[f, e] = 0
    for (k = 1; k <= 6; k++) {
      v = $k + 0
      if (v < 0 || v > 1) { printf "score out of range at row %d\n", NR > "/dev/stderr"; exit 2 }
      s[f, A[k]] += v; s[f, B[k]] += 1 - v
      score[A[k], B[k]] += v; score[B[k], A[k]] += 1 - v
      if (v > 0.5) maj[A[k], B[k]]++; else if (v < 0.5) maj[B[k], A[k]]++
    } }
  END {
    if (f != 64) { printf "expected 64 families, found %d\n", f > "/dev/stderr"; exit 2 }
    print "row,col,score,majority" > pairwise
    for (r = 1; r <= 4; r++) for (c = 1; c <= 4; c++)
      if (r == c) printf "%d,%d,0.5000,-1\n", r, c > pairwise
      else printf "%d,%d,%.4f,%d\n", r, c, score[r, c] / f, maj[r, c] + 0 > pairwise
    # Best of four per family: highest Copeland score; ties share the family.
    for (i = 1; i <= f; i++) {
      m = -1; for (e = 1; e <= 4; e++) if (s[i, e] > m + 1e-9) m = s[i, e]
      top[i] = m; w[i] = 0; for (e = 1; e <= 4; e++) if (s[i, e] > m - 1e-9) { if (w[i] == 0) w[i] = e; else w[i] = 9 }
      key[i] = sprintf("%d %09.4f %04d", (w[i] == 9 ? 5 : w[i]), 3 - m, i) }
    # Order families by winner (plot order), then by the winner'"'"'s score, then by input row.
    n = 0; for (i = 1; i <= f; i++) ord[++n] = i
    for (i = 2; i <= n; i++) { t = ord[i]; j = i - 1
      while (j >= 1 && key[ord[j]] > key[t]) { ord[j + 1] = ord[j]; j-- } ord[j + 1] = t }
    print "x,y,score,best,family_row" > strip
    for (x = 1; x <= n; x++) { i = ord[x]
      for (e = 1; e <= 4; e++)
        printf "%d,%d,%.4f,%d,%d\n", x, e, s[i, e], (s[i, e] > top[i] - 1e-9 ? 1 : 0), i > strip }
  }' "$matrix"
jq -r '
  def short: {"Opus5p5_Extra":"Opus","Gemini4_argon_high":"Gemini","GPT6p1_Sol_Max":"Sol","GPT6_Astra_Max":"Astra"}[.];
  "y,label,ratio,lo,hi,p",
  (.pairwise | to_entries[] |
    "\(.key + 1),\(.value.a | short) / \(.value.b | short),\(.value.geometric_rmse_ratio * 10000 | round / 10000),\(.value.ratio_ci95[0] * 10000 | round / 10000),\(.value.ratio_ci95[1] * 10000 | round / 10000),\(.value.p * 10000 | round / 10000)")
' "$paper/artifacts/benchmark/RESULTS.json" > "$out/frontier-ratios.csv"
