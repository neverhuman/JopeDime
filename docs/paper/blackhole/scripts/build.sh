#!/usr/bin/env bash
# docs/paper/blackhole/scripts/build.sh — check sources, then typeset EurekaZero: The Tabular Black Hole.
#
#   bash scripts/build.sh            check, typeset and verify the PDF (make pdf)
#   bash scripts/build.sh --clean    remove only the files this build produces
#
# Output lands in the lane's reused target directory, $repo/target/blackhole-paper (override with
# BH_BUILD_DIR): EurekaZero-BlackHole.pdf with its TeX log and build-environment.txt.
# Fonts come from fonts.lock (SIL OFL 1.1, shared cache ~/.cache/innovationzero-fonts).
set -euo pipefail
paper=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
repo=$(git -C "$paper" rev-parse --show-toplevel)
build=${BH_BUILD_DIR:-$repo/target/blackhole-paper}
job=EurekaZero-BlackHole

if [ "${1:-}" = --clean ]; then
  for ext in aux bbl blg fdb_latexmk fls log out toc xdv pdf build.txt; do rm -f -- "$build/$job.$ext"; done
  rm -f -- "$build/claim-ledger.tex" "$build/build-environment.txt"
  rm -rf -- "$build/self-test"*
  exit 0
fi
[ $# -eq 0 ] || { echo "usage: $0 [--clean]" >&2; exit 2; }

if [ -n "${EZ_FONT_CACHE:-}" ]; then
  cache=$EZ_FONT_CACHE
else
  case ${XDG_CACHE_HOME:-} in
    /*) cache=$XDG_CACHE_HOME/innovationzero-fonts ;;
    *)
      case ${HOME:-} in
        /*) cache=$HOME/.cache/innovationzero-fonts ;;
        *) printf 'build.sh: HOME must be an absolute path\n' >&2; exit 2 ;;
      esac
      ;;
  esac
fi

if command -v xelatex >/dev/null; then
  tex_engine=xelatex
  latexmk_engine=-xelatex
elif command -v lualatex >/dev/null; then
  tex_engine=lualatex
  latexmk_engine=-lualatex
else
  echo "build.sh: xelatex or lualatex is required" >&2
  exit 2
fi

for tool in latexmk bibtex pdfinfo pdftotext pdffonts jq curl unzip sha256sum awk; do
  command -v "$tool" >/dev/null || { echo "build.sh: $tool is required" >&2; exit 2; }
done
mkdir -p -- "$build" "$cache/archives" "$cache/fonts" "$build/texmf-var"
export TEXMFVAR="$build/texmf-var"

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
    unzip -o -q -j "$archive" "$m" -d "$cache/fonts"
  done
done < "$paper/fonts.lock"

bash "$paper/scripts/check.sh" --prepare "$build"

export SOURCE_DATE_EPOCH=${SOURCE_DATE_EPOCH:-1794808800} FORCE_SOURCE_DATE=1 TZ=UTC
export TEXINPUTS="$paper//:$build:" BIBINPUTS="$paper:"
{
  printf 'SOURCE_DATE_EPOCH=%s\n' "$SOURCE_DATE_EPOCH"
  printf 'source_head=%s\n' "$(git -C "$repo" rev-parse HEAD)"
  printf 'source_dirty=%s\n' "$(git -C "$repo" status --porcelain -- "$paper" | wc -l)"
  printf 'content=measured black-hole white paper (see evidence/claims.json)\n'
  latexmk -version | sed -n '1,2p'
  "$tex_engine" --version | sed -n 1p
  bibtex --version | sed -n 1p
  pdfinfo -v 2>&1 | sed -n 1p
  sha256sum "$paper/fonts.lock"
} > "$build/build-environment.txt"

pretex="\\def\\BHFontDir{$cache/fonts/}"
(cd "$paper" && latexmk "$latexmk_engine" -interaction=nonstopmode -halt-on-error -file-line-error \
   -no-shell-escape -outdir="$build" -jobname="$job" -usepretex="$pretex" main.tex \
   > "$build/$job.build.txt" 2>&1) || {
  tail -n 45 "$build/$job.build.txt" >&2
  echo "build.sh: main.tex failed (full output: $build/$job.build.txt)" >&2; exit 1; }

bash "$paper/scripts/check.sh" --pdf-dir "$build"
printf '%-30s %4s pages\n' "$job.pdf" "$(pdfinfo "$build/$job.pdf" | awk '/^Pages:/{print $2}')"
echo "Built in $build"
