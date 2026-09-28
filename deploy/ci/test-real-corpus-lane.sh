#!/usr/bin/env bash
# Exercise custody refusals and root binding without compiling or accessing private data.
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p target/tmp
scratch=$(mktemp -d "$PWD/target/tmp/test-host-lane.XXXXXX")
trap 'chmod -R u+w -- "$scratch"; rm -r -- "$scratch"' EXIT
mkdir -p "$scratch/deploy/ci" "$scratch/bin" "$scratch/custody/"{sotarena,state,registry}
cp deploy/ci/{real-corpus-lane.sh,check-test-routing.sh,real-corpus-tests.txt} "$scratch/deploy/ci/"
# check-test-routing.sh also reads the routing reasons from the source attributes: copy the files
# that declare them, and give each mocked suite the crate directory nextest reports as its cwd.
git grep -l -E 'ignore[[:space:]]*=[[:space:]]*"(real-corpus|deferred-classification):' -- crates |
  while IFS= read -r file; do mkdir -p "$scratch/$(dirname "$file")"; cp "$file" "$scratch/$file"; done
dirs=$(cargo metadata --no-deps --locked --format-version 1 |
  jq -c --arg root "$PWD/" '[.packages[] | {key: .name, value: (.manifest_path | ltrimstr($root) | rtrimstr("/Cargo.toml"))}] | from_entries')
printf 'synthetic routing fixture only\n' > "$scratch/custody/PROVENANCE.txt"
manifest() {
  (cd "$scratch/custody" && find . -type f ! -path './MANIFEST.sha256' -print0 | sort -z | xargs -0 sha256sum > MANIFEST.sha256)
  chmod -R a-w "$scratch/custody"
}
manifest
# Only the selected test collection is mocked. The real host wrapper and its
# actual custody validation/root assignments execute unchanged.
sed -e 's/#.*//' -e '/^[[:space:]]*$/d' deploy/ci/real-corpus-tests.txt |
  jq -Rn --arg scratch "$scratch" --argjson dirs "$dirs" '{"rust-suites": (reduce inputs as $line ({}; ($line | [scan("[^[:space:]]+")]) as $r | .[$r[0]].cwd=($scratch + "/" + $dirs[$r[0] | split("::")[0]]) | .[$r[0]].testcases[$r[1]]={"ignored":true,"filter-match":{"status":"matches"}}))}' > "$scratch/collection.json"
cat > "$scratch/bin/cargo" <<'MOCK'
#!/usr/bin/env bash
set -euo pipefail
[[ $JOPE_TEST_SOTARENA_ROOT == "$TEST_CUSTODY/state/arena" ]]
[[ $SOTARENA_REPORTER_ROOT == "$TEST_CUSTODY/state/arena" ]]
[[ $SOTARENA_ROOT == "$TEST_CUSTODY/sotarena" && $JOPE_STATE == "$TEST_CUSTODY/state" && $JOPE_REGISTRY == "$TEST_CUSTODY/registry" ]]
[[ $JOPE_BINARY42_TEST_DATA_ROOT == "$TEST_CUSTODY/binary42/data" ]]
[[ $JOPE_BINARY42_TEST_REPORT_ROOT == "$TEST_CUSTODY/binary42/reports" ]]
case "$*" in
  'nextest list '*) cat "$TEST_COLLECTION" ;;
  'nextest run '*) printf 'mock selected execution\n' >> "$TEST_EXECUTION" ;;
  *) exit 2 ;;
esac
MOCK
chmod +x "$scratch/bin/cargo"
export TEST_CUSTODY=$scratch/custody TEST_COLLECTION=$scratch/collection.json TEST_EXECUTION=$scratch/execution
export PATH=$scratch/bin:$PATH
export JOPE_TEST_SOTARENA_ROOT=/outside SOTARENA_REPORTER_ROOT=/outside
export JOPE_BINARY42_TEST_DATA_ROOT=/outside JOPE_BINARY42_TEST_REPORT_ROOT=/outside
run() { bash "$scratch/deploy/ci/real-corpus-lane.sh" "$scratch/custody" "$@"; }
refuse() { if run "$@"; then echo 'host lane accepted invalid custody' >&2; exit 1; fi; }
run
[[ $(wc -l < "$TEST_EXECUTION") == 1 ]]
refuse --run-ignored all
chmod u+w "$scratch/custody/PROVENANCE.txt"
refuse
printf 'altered\n' >> "$scratch/custody/PROVENANCE.txt"
chmod a-w "$scratch/custody/PROVENANCE.txt"
refuse
chmod -R u+w "$scratch/custody"
manifest
chmod u+w "$scratch/custody"
printf 'unlisted\n' > "$scratch/custody/unlisted"
chmod a-w "$scratch/custody/unlisted" "$scratch/custody"
refuse
chmod u+w "$scratch/custody"
rm "$scratch/custody/unlisted"
ln -s /outside "$scratch/custody/escape"
chmod a-w "$scratch/custody"
refuse
[[ $(wc -l < "$TEST_EXECUTION") == 1 ]]
echo 'host lane: root binding control and 5 custody refusals passed'
