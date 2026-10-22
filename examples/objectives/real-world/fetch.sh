#!/usr/bin/env bash
# Fetch the public datasets of the real-world objective benchmark into a directory OUTSIDE the
# checkout, verify pinned digests and write header CSVs. Bulk data never enters Git.
#
#   bash examples/objectives/real-world/fetch.sh [DATA_DIR]
#
# DATA_DIR defaults to $JOPE_FITNESS_DATA, else $HOME/jope-data/fitness-real-world.
# Needs curl, gzip, awk and sha256sum. freMTPL2freq is optional: it is fetched from OpenML
# (licence recorded there as CC0; the CASdatasets upstream is GPL>=2, so it is never vendored).
set -euo pipefail
root=$(cd "$(dirname "$0")/../../.." && pwd)
data=${1:-${JOPE_FITNESS_DATA:-$HOME/jope-data/fitness-real-world}}
mkdir -p "$data"
data=$(cd "$data" && pwd)
case "$data/" in
  "$root"/*) echo "fetch.sh: refusing to write datasets inside the checkout ($data)" >&2; exit 2 ;;
esac
mkdir -p "$data/raw" "$data/csv"

fetch() { # file url sha256
  local out="$data/raw/$1"
  if [ ! -f "$out" ]; then
    curl -fsSL --retry 3 --max-time 600 -o "$out.part" "$2"
    mv "$out.part" "$out"
  fi
  if ! echo "$3  $out" | sha256sum -c --quiet - >/dev/null 2>&1; then
    echo "fetch.sh: $1 does not match its pinned sha256 $3; the source changed" >&2
    exit 3
  fi
}

pmlb=https://media.githubusercontent.com/media/EpistasisLab/pmlb/master/datasets
tsv_to_csv() { # gz name
  gzip -dc "$data/raw/$1" | tr '\t' ',' > "$data/csv/$2.csv"
}

# PMLB (MIT; Olson et al. 2017). Original sources: StatLib.
fetch 503_wind.tsv.gz "$pmlb/503_wind/503_wind.tsv.gz" \
  2cc1e040bdc20737674a00039cd80e16758c511e697e524ed17e4335ec93e9a6
fetch 537_houses.tsv.gz "$pmlb/537_houses/537_houses.tsv.gz" \
  d383fd58bb79760ffaf0b615942fd47d926a19e75a8f980b79fbe9ad479d9cbf
fetch 522_pm10.tsv.gz "$pmlb/522_pm10/522_pm10.tsv.gz" \
  ecbabaa5a989a8c11f5a57e874c135f386da5379d6689ae6d5d4264a652fa5e0
fetch 547_no2.tsv.gz "$pmlb/547_no2/547_no2.tsv.gz" \
  e116f24d2cabc7388a0714f1d0bf3a85f7613e3fc2b88642c47154fece000c73
tsv_to_csv 503_wind.tsv.gz wind
tsv_to_csv 537_houses.tsv.gz houses
tsv_to_csv 522_pm10.tsv.gz pm10
tsv_to_csv 547_no2.tsv.gz no2

# Shiller's monthly S&P 500 series (ODC-PDDL-1.0, datasets/s-and-p-500). The file grows every
# month and its latest months lack dividends and earnings, so the digest pins rows through
# 2023-07 and the features use months through 2023-06.
spx="$data/raw/s-and-p-500.csv"
if [ ! -f "$spx" ]; then
  curl -fsSL --retry 3 --max-time 600 -o "$spx.part" \
    https://raw.githubusercontent.com/datasets/s-and-p-500/main/data/data.csv
  mv "$spx.part" "$spx"
fi
pinned=$(awk -F, 'NR == 1 || $1 <= "2023-07-01"' "$spx" | sha256sum | cut -d' ' -f1)
if [ "$pinned" != "0e0eb501cd45d55a6b4cb2aad4431969289542ad57f0c7f5917985b7c3b76e51" ]; then
  echo "fetch.sh: S&P 500 rows through 2023-07 changed upstream (sha256 $pinned)" >&2
  exit 3
fi
# Features known at month t; target is the next month's total return (price + dividend/12).
# Shiller's prices are monthly averages, which makes adjacent returns positively correlated, so
# last month's return is deliberately not a feature.
awk -F, 'NR > 1 && $1 >= "1900-01-01" && $1 <= "2023-07-01" { print }' "$spx" | awk -F, '
  { date[NR] = $1; p[NR] = $2; d[NR] = $3; e[NR] = $4; cpi[NR] = $5; rate[NR] = $6; pe10[NR] = $10 }
  END {
    print "dividend_yield,earnings_yield,cape_yield,long_rate,inflation_12m,momentum_12m,next_return"
    for (t = 13; t < NR; t++) {
      if (p[t] <= 0 || p[t - 12] <= 0 || cpi[t - 12] <= 0 || pe10[t] <= 0) continue
      printf "%.8f,%.8f,%.8f,%.4f,%.8f,%.8f,%.8f\n", d[t] / p[t], e[t] / p[t], 1 / pe10[t], \
        rate[t], cpi[t] / cpi[t - 12] - 1, p[t] / p[t - 12] - 1, \
        (p[t + 1] + d[t] / 12) / p[t] - 1
    }
  }' > "$data/csv/sp500.csv"

# freMTPL2freq (OpenML 41214), optional: claim frequency with exposure.
if [ "${FETCH_FREMTPL:-0}" = 1 ]; then
  fetch freMTPL2freq.arff https://openml.org/data/v1/download/20649148/freMTPL2freq.arff \
    "${FREMTPL_SHA256:?set FREMTPL_SHA256 to the digest you verified for the OpenML file}"
  awk -f "$root/examples/objectives/real-world/arff-to-csv.awk" "$data/raw/freMTPL2freq.arff" \
    | awk -F, 'NR == 1 { for (i = 1; i <= NF; i++) c[$i] = i
                         print "VehPower,VehAge,DrivAge,BonusMalus,VehBrand,VehGas,Area,Density,Region,exposure,frequency"; next }
               (NR - 2) % 20 == 0 {
                 exposure = $c["Exposure"]; claims = $c["ClaimNb"] > 4 ? 4 : $c["ClaimNb"]
                 if (exposure > 1) exposure = 1
                 printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%.8f\n", $c["VehPower"], $c["VehAge"], $c["DrivAge"],
                   $c["BonusMalus"], $c["VehBrand"], $c["VehGas"], $c["Area"], $c["Density"], $c["Region"],
                   exposure, claims / exposure }' > "$data/csv/fremtpl.csv"
fi

for f in "$data"/csv/*.csv; do
  printf '%-12s %7d rows  %s\n' "$(basename "$f" .csv)" $(($(wc -l < "$f") - 1)) "$f"
done
