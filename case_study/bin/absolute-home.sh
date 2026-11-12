#!/usr/bin/env bash
# Print HOME when it is absolute. Case-study defaults join this to .cache/jope
# or jope-data. A relative or empty HOME must not name either directory.
# JOPE_CACHE, JOPE_CASE_OUT, JOPE_CASE_CATALOG, JOPE_CASE_BASELINES, and
# JOPE_CASE_TEST_REGISTRY stay verbatim: callers use ${OVERRIDE:-$(this)/...}.
set -euo pipefail
case ${HOME:-} in
  /*) printf '%s\n' "$HOME" ;;
  *) printf 'case_study: HOME must be an absolute path\n' >&2; exit 2 ;;
esac
