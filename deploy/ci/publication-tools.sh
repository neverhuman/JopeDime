#!/usr/bin/env bash
# Run the publication command with verified, task-local Ubuntu/TeX dependencies.
# No apt install, maintainer scripts, root privileges or image mutation.
set -euo pipefail
repo=$(git rev-parse --show-toplevel)
cache="$repo/target/tool-cache/publication"
evidence=${JOPE_PAPER_NATIVE_EVIDENCE:-$repo/target/innovationzero-paper/native}
mkdir -p "$evidence" "$repo/target/tmp"
receipt="$evidence/dependencies.json"
runtime=$(python3 "$repo/deploy/ci/browser-deps.py" --lock "$repo/deploy/ci/publication-deps.lock.json" --cache "$cache" --receipt "$receipt")
export PATH="$runtime/usr/bin:$PATH"
export LD_LIBRARY_PATH="$runtime/usr/lib/x86_64-linux-gnu:$runtime/lib/x86_64-linux-gnu:$runtime/usr/lib:$runtime/lib"
FONTCONFIG_FILE="$cache/fontconfig-$(sha256sum "$repo/deploy/ci/publication-deps.lock.json" | cut -d' ' -f1).xml"
export FONTCONFIG_FILE
export TMPDIR="$repo/target/tmp"
# Debian normally creates these files in package-maintainer scripts. Generate only
# the English XeLaTeX configuration in our writable state, never in the deb tree.
state="$cache/tex-state"
mkdir -p "$state/config/web2c" "$state/config/tex/generic/config" "$state/var" "$state/home" "$state/local" "$state/bin"
# Debian registers this command through update-alternatives, which we never run.
# Bind it directly to the verified package binary in our task-local PATH instead.
test -x "$runtime/usr/bin/bibtex.original"
ln -sfn "$runtime/usr/bin/bibtex.original" "$state/bin/bibtex"
export PATH="$state/bin:$PATH"
cp "$runtime/var/lib/tex-common/fmtutil-cnf/texlive/texlive-xetex.cnf" "$state/config/web2c/fmtutil.cnf"
printf 'english hyphen.tex\n=usenglish\n=USenglish\n=american\n' > "$state/config/tex/generic/config/language.dat"
printf '\\addlanguage{USenglish}{hyphen}{}{0}{3}{2}\n' > "$state/config/tex/generic/config/language.def"
export TEXMFROOT="$runtime/usr/share/texlive" TEXMFDIST="$runtime/usr/share/texlive/texmf-dist" TEXMFDEBIAN="$runtime/usr/share/texmf"
export TEXMFCNF="$runtime/usr/share/texlive/texmf-dist/web2c"
export TEXMFSYSCONFIG="$state/config" TEXMFCONFIG="$state/config" TEXMFSYSVAR="$state/var" TEXMFVAR="$state/var"
export TEXMFHOME="$state/home" TEXMFLOCAL="$state/local"
export TEXMF="{$state/config,$state/var,$TEXMFDEBIAN,$TEXMFDIST}"
# --sys selects the explicitly task-local system-state paths above, without sudo.
fmtutil --sys --byfmt xelatex > "$evidence/format.log" 2>&1
test $# -gt 0 || { echo 'publication-tools.sh: command required' >&2; exit 2; }
exec "$@"
