#!/usr/bin/env bash
# docs/paper/eurekazero/scripts/build.sh — check the sources, then typeset the north-star paper.
#
#   bash scripts/build.sh            check, typeset and check the PDF (make pdf)
#   bash scripts/build.sh --clean    remove only the files this build produces
#
# The product goes to the lane's reused target directory, $repo/target/eurekazero-paper (override
# with EZ_BUILD_DIR): EurekaZero-NorthStar.pdf with its TeX log and build-environment.txt. Needs
# XeLaTeX, latexmk and BibTeX (TeX Live 2023 or later), Poppler, jq, curl, unzip and sha256sum.
# Fonts come from fonts.lock (the same pinned, SHA-256-checked archives as the InnovationZero
# paper), cached outside the working tree (EZ_FONT_CACHE, default ~/.cache/innovationzero-fonts,
# shared with that paper). The font archives are the only network reads.
set -euo pipefail
paper=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
repo=$(git -C "$paper" rev-parse --show-toplevel)
build=${EZ_BUILD_DIR:-$repo/target/eurekazero-paper}
cache=${EZ_FONT_CACHE:-${XDG_CACHE_HOME:-$HOME/.cache}/innovationzero-fonts}
job=EurekaZero-NorthStar

if [ "${1:-}" = --clean ]; then
  # Never blanket-delete target/: only this build's known products.
  for ext in aux bbl blg fdb_latexmk fls log out toc xdv pdf build.txt; do rm -f -- "$build/$job.$ext"; done
  rm -f -- "$build/gap-ledger.tex" "$build/claim-ledger.tex" "$build/build-environment.txt"
  rm -rf -- "$build/synthetic-check"
  exit 0
fi
[ $# -eq 0 ] || { echo "usage: $0 [--clean]" >&2; exit 2; }

for tool in xelatex latexmk bibtex pdfinfo pdftotext pdffonts jq curl unzip sha256sum awk; do
  command -v "$tool" >/dev/null || { echo "build.sh: $tool is required" >&2; exit 2; }
done
mkdir -p -- "$build" "$cache/archives" "$cache/fonts"

while read -r digest url members; do
  case "$digest" in ''|'#'*) continue ;; esac
  archive="$cache/archives/$(basename "$url")"
  if [ ! -f "$archive" ] || ! echo "$digest  $archive" | sha256sum -c --status; then
    echo "fetching $url"
    curl -fsSL --retry 3 -o "$archive.part" "$url"
    mv -- "$archive.part" "$archive"
  fi
  echo "$digest  $archive" | sha256sum -c --status || { echo "build.sh: SHA-256 mismatch for $url" >&2; exit 3; }
  for m in $members; do
    # Re-extract from the verified archive, so a modified cached font cannot be reused.
    unzip -o -q -j "$archive" "$m" -d "$cache/fonts"
  done
done < "$paper/fonts.lock"

bash "$paper/scripts/check.sh" --prepare "$build"

# Reproducible dates: the committer time of the reviewed main snapshot unless overridden.
export SOURCE_DATE_EPOCH=${SOURCE_DATE_EPOCH:-1792763355} FORCE_SOURCE_DATE=1 TZ=UTC
export TEXINPUTS="$paper//:$build:" BIBINPUTS="$paper:"
{
  printf 'SOURCE_DATE_EPOCH=%s\n' "$SOURCE_DATE_EPOCH"
  printf 'source_head=%s\n' "$(git -C "$repo" rev-parse HEAD)"
  printf 'source_dirty=%s\n' "$(git -C "$repo" status --porcelain -- "$paper" | wc -l)"
  printf 'content=north-star draft; every result is SYNTHETIC (see northstar/gaps.json)\n'
  # sed reads to the end, so a version banner longer than one write cannot raise SIGPIPE under pipefail.
  latexmk -version | sed -n '1,2p'
  xelatex --version | sed -n 1p
  bibtex --version | sed -n 1p
  pdfinfo -v 2>&1 | sed -n 1p
  sha256sum "$paper/fonts.lock"
} > "$build/build-environment.txt"

pretex="\\def\\EZFontDir{$cache/fonts/}"
(cd "$paper" && latexmk -xelatex -interaction=nonstopmode -halt-on-error -file-line-error \
   -no-shell-escape -outdir="$build" -jobname="$job" -usepretex="$pretex" main.tex \
   > "$build/$job.build.txt" 2>&1) || {
  tail -n 40 "$build/$job.build.txt" >&2
  echo "build.sh: main.tex failed (full output: $build/$job.build.txt)" >&2; exit 1; }

bash "$paper/scripts/check.sh" --pdf-dir "$build"
printf '%-28s %4s pages\n' "$job.pdf" "$(pdfinfo "$build/$job.pdf" | awk '/^Pages:/{print $2}')"
echo "Built in $build"
