#!/bin/bash -p
# deploy/release/release.sh: fast signed host release of JopeDime (owner decision D-D, 2026-09-27).
#
#   deploy/release/release.sh <vX.Y.Z> <merge-commit-sha>             # dry run (the default)
#   deploy/release/release.sh <vX.Y.Z> <merge-commit-sha> --publish   # the same run, then publish
#
# Run it on xbabe2 from the root of a clean checkout or registered worktree of neverhuman/JopeDime.
# HEAD must be <merge-commit-sha>, and that commit must equal origin/main after a fresh fetch.
# Run it as shown: the #! line starts the system bash, /bin/bash, by absolute path and in
# privileged mode (bash -p), so no PATH lookup chooses the interpreter. `bash -p
# deploy/release/release.sh` is accepted as well when that bash is the root-owned system one
# (preflight checks the running interpreter like a host tool). A plain `bash
# deploy/release/release.sh` is refused (see "bash itself" below).
#
# A dry run does the following, and it never tags, pushes or publishes:
#   - exports the commit with `git archive` and verifies the export against the commit's tree
#     (paths, blob ids and modes, nothing extra). The build, the receipt and the SBOM read only
#     that export; the working tree is never mounted or scanned, so a file hidden from
#     `git status` (assume-unchanged, skip-worktree, ignored) cannot reach the release;
#   - builds `jope` from the export inside the pinned rust:1.97.1-bookworm image (glibc 2.36);
#   - checks the glibc ceiling and that no CUDA/NVIDIA/LibTorch library is linked;
#   - writes a build receipt, SBOMs, release notes (extracted from the committed CHANGELOG.md),
#     a release manifest and SHA256SUMS;
#   - signs SHA256SUMS with the root-held cosign key in /etc/jope-release, then verifies it all.
# --publish runs the same steps, and only when every one has passed does it:
#   - create and push an annotated tag;
#   - create the GitHub release;
#   - download that release again and verify it.
#
# Evidence goes under /home/ubuntu/jope-evidence/release-<tag>/:
#   - <dryrun|publish>-<UTC stamp>/ is new for every run and never overwritten. It holds
#     release.log, dist/, tools.json (the tools that ran), sbom/ (syft's configuration and version)
#     and the checks, and is sealed by a SHA256SUMS written as the run ends. Its src/ is the
#     verified export that was built (reproducible with `git archive <sha>`), and its home/ is the
#     build's HOME; the seal leaves out src/, tmp/ (TMPDIR, 0700: the containers' temporary
#     directory, syft's home and cache, and the empty configuration directories of jq, docker and
#     cosign) and tuf/.
#   - target/ is the Cargo target directory and cargo home. It is shared by the runs for one tag,
#     so a publish run after a dry run reuses the compiled dependencies. The workspace crates are
#     rebuilt every run, because the export's files carry the extraction time. The cargo home may
#     hold only cargo's own caches; a config.toml, credentials or anything else there is refused.
#     The cache is as trustworthy as the ubuntu account, which is also the signing boundary
#     (docs/release/README.md, "Key custody").
#   - custody/ (the key-custody receipt) is never touched.
#   - release-tmp/ (0700, directly under the evidence root) is TMPDIR from the first command of every
#     run until its run directory exists; after that TMPDIR is the run's tmp/ (also 0700). Nothing
#     is ever written under /tmp (see pin_tmpdir), and the rule is carried across every process
#     boundary explicitly: each container gets the run's tmp/container/ as its TMPDIR and its /tmp
#     (builder_container), root's sign shell exports a TMPDIR inside its own /root work directory
#     (SIGN_SCRIPT), and every env -i run names its TMPDIR.
#
# Tools: syft and cosign are pinned by sha256 and by the version they report (SYFT_*, COSIGN_*),
# and syft runs only as a checked copy. The host tools (HOST_TOOLS: coreutils, grep, sed, find, tar,
# cmp/diff, jq, git, docker, objdump, sudo and the rest) are looked up only in TRUSTED_PATH, the
# system directories, which the first lines make PATH (read-only) before any command runs; each
# must be a root-owned file reached only through root-owned directories and symlinks that neither
# group nor others can write (check_host_tool, trusted_chain), and their versions are recorded in
# the run's tools.json. No pinned or host
# tool reads per-user configuration that could change what it reports to a check or writes into a
# release (~/.jq, ~/.gitconfig, ~/.docker, TAR_OPTIONS, LD_* and the like; see "The tools' own
# configuration" below). Two commands are neither pinned nor host tools, and HOST_TOOLS names them
# with the reasons: the host's cargo and gh-role. Like syft and cosign, both are run by absolute
# path (HOST_CARGO, GH_ROLE), never looked up.
# There is no Python in this script. docs/release/README.md explains how anyone can verify a release.
#
# Testing: deploy/release/test-release.sh sources this file; the CI hygiene lane runs it. The
# JOPE_RELEASE_TEST* overrides are honoured only when JOPE_RELEASE_TEST=1 and the working directory
# and every override path are under $HOME/.cache/jope-test-tmp/ (/home/ubuntu/.cache/jope-test-tmp/
# on xbabe2; the CI runner container's root filesystem is read-only outside its $HOME). Test mode
# stops after the guards, and it refuses --publish. Any JOPE_RELEASE_TEST* variable seen outside
# that context stops a real run.

# bash itself. A bash started without -p has, before this line, imported the exported functions of
# the environment (any of which stands in for the command or the builtin of its name: git, unset,
# command, exit), run a BASH_ENV file, and applied SHELLOPTS and BASHOPTS (noexec runs nothing and
# exits 0; nocasematch makes [[ ]] accept an uppercase SHA). bash -p does none of that, and it ignores
# CDPATH and GLOBIGNORE as well. So release.sh refuses any other bash. It also refuses an exported
# SHELLOPTS: bash -p ignores it, but a bash started without -p takes 'privileged' from it, and nothing
# else would tell the two apart. Only keywords and parameter expansions decide this, because an
# imported function can stand in for any builtin, printf and exit included, but not for [[ ]]; the
# last lines guard main() with the same test, so a stood-in exit cannot carry the run past this one.
# What this cannot refuse (SHELLOPTS=noexec with a plain bash, which runs no line; a BASH_ENV file,
# which runs before this line) is docs/release/README.md's "What remains".
if [[ ! -o privileged ]]; then
  printf 'release.sh: REFUSED: %s\n' "bash was not started in privileged mode; run deploy/release/release.sh itself (its #! line runs bash -p) or bash -p deploy/release/release.sh, never a plain bash, which has already imported exported functions and applied BASH_ENV, SHELLOPTS and BASHOPTS" >&2
  exit 1
fi
if [[ ${SHELLOPTS@a} == *x* ]]; then
  printf 'release.sh: REFUSED: %s\n' "SHELLOPTS is exported, and a bash started without -p would take privileged mode from it; run release.sh without it (env -u SHELLOPTS deploy/release/release.sh ...)" >&2
  exit 1
fi
set -euo pipefail
shopt -s inherit_errexit

# PATH. Every command the script runs by name is looked up in TRUSTED_PATH, the system directories
# that the host's packages install into, and never in the operator's PATH, whose first entry on
# xbabe2 is ~/.local/bin, which the operator can write. PATH is set here, before the first command
# runs, and made read-only, so that nothing later, in this shell or in a caller that sources this
# file, can put another directory in front of it (assigning PATH also empties bash's table of
# remembered command paths). /usr/local/sbin and /usr/local/bin are left out: no host tool lives
# there, the host's packages never install there, and on xbabe2 /usr/local/bin holds root-owned
# symlinks into ~/.local/bin (docker-compose). The tools that the operator can write and that the
# script still runs (cosign, syft, the host cargo and gh-role) are named by absolute path below. The
# same value is spelled out in SIGN_SCRIPT, for root's shell; keep the two in step.
readonly TRUSTED_PATH=/usr/sbin:/usr/bin:/sbin:/bin
PATH=$TRUSTED_PATH
export PATH
readonly PATH

readonly REPO_SLUG=neverhuman/JopeDime
readonly REPO_URL=https://github.com/neverhuman/JopeDime.git
# rust:1.97.1-bookworm (Debian glibc 2.36, rustc 1.97.1), pinned by digest. The pin was verified on
# xbabe2 on 2026-09-28; the receipt is ~/jope-evidence/release-v0.2.0/custody/builder-image.txt.
readonly BUILDER_IMAGE=rust@sha256:0e2bcaef56d041a486784e54104a81aebe0da44bd03019bd70bc0401e42e4a97
readonly BUILDER_IMAGE_NAME=rust:1.97.1-bookworm
readonly BUILDER_TOOLCHAIN=1.97.1
readonly MAX_GLIBC=2.36
readonly TARGET_TRIPLE=x86_64-unknown-linux-gnu
readonly REQUIRED_HOST=xbabe2
readonly EVIDENCE_ROOT=/home/ubuntu/jope-evidence
# The sign step spells this path out literally inside root's sh -c; keep the two in step.
readonly KEY_DIR=/etc/jope-release
readonly PUBKEY_PATH=docs/release/signing.pub
readonly PUBKEY_SHA256=de84945ec5326877d202d0738e655492a8b2c3b794873a2eb248939c4d326e4e
# The tag is created by the same identity that creates the release (gh-role writer).
readonly TAGGER_EMAIL=130782313+jepsontaylor@users.noreply.github.com
readonly COSIGN=/home/ubuntu/.local/bin/cosign
# ~/.local/bin/cosign is writable by ubuntu, and the sign step runs cosign as root with the key
# password in its environment. Root therefore copies it into a root-only directory, checks the
# copy against this digest and runs only the copy. The digest is cosign v2.4.3 as installed on
# xbabe2 (checked 2026-09-28); an upgrade is a reviewed change to these two lines.
readonly COSIGN_VERSION=v2.4.3
readonly COSIGN_SHA256=caaad125acef1cb81d58dcdc454a1e429d09a750d1e9e2b3ed1aed8964454708
readonly SYFT=/home/ubuntu/.local/bin/syft
# The SBOMs signed into the release are syft's output, so syft is pinned like cosign. It is also
# writable by ubuntu: preflight refuses it unless its sha256 is this digest and `syft version`
# reports this version, and make_sbom copies it into the run's tmp/, checks the copy the same way
# and runs only the copy. The digest is syft 1.40.0 (linux/amd64, commit 11e871566b35) as installed
# on xbabe2 (checked 2026-09-28); an upgrade is a reviewed change to these two lines.
readonly SYFT_VERSION=1.40.0
readonly SYFT_SHA256=eb9714fb8e4b8f2a647e7bb312f1e0b9f83a7aa30418658bf46583cfa83d27d2
readonly GH_ROLE=/home/ubuntu/.local/bin/gh-role
# The host's rustup shim (a symlink to ~/.cargo/bin/cargo). It is run by this path, never looked up:
# see HOST_TOOLS below for why it may be operator-writable.
readonly HOST_CARGO=/home/ubuntu/.local/bin/cargo
# HOST_TOOLS: every command that the script runs by name on the host, as the operator or as root
# through sudo (test, cat, chown, chmod and sh, and SIGN_SCRIPT's mktemp, mkdir, cp, chmod,
# sha256sum, cat and rm), except the two named below. They are the host's packages (Ubuntu's, and
# Docker's apt repository for docker), root-owned and replaced by routine security updates, so they
# are not pinned by digest: a pin would stop every release after each update, and only root can
# change them, which is inside the trust boundary already (docs/release/README.md, "Key custody").
# What preflight checks instead (require_tools, check_host_tool, trusted_chain):
#   - each name is looked up only in TRUSTED_PATH, which is PATH (read-only) from the first lines on,
#     so a copy in ~/.local/bin, first on the operator's PATH, is never found, let alone run;
#   - the file found there, and everything on the way to it, is walked from / down without following
#     a link unchecked: every directory and the file itself must be root-owned and closed to group
#     and other writes, and every symlink root-owned, with its target walked the same way. A
#     directory that the operator owns anywhere on the way would let the operator swap what lies
#     beneath it, root-owned or not, so it refuses the tool;
#   - the checks run stat and readlink by absolute path (STAT, READLINK), after proving with bash
#     builtins alone that the operator cannot write them or any directory above them (check_helpers),
#     so a stat that the operator put first on a PATH cannot vouch for anything;
#   - a lookup of the name must return that same file, and the commands root runs through sudo are
#     given its absolute path (host_path), because sudo would otherwise search its own secure_path,
#     /usr/local first; SIGN_SCRIPT sets root's PATH to TRUSTED_PATH for the same reason.
# The run records each one's path, sha256 and package version in tools.json.
# bash is the interpreter of the #! line, /bin/bash, run by absolute path; preflight checks the
# running interpreter ($BASH) with trusted_chain as well, which also refuses a `bash -p release.sh`
# whose bash is not the system one.
# The pinned syft and cosign above are run by path. Only two commands are neither pinned nor host
# tools, and this is why that is acceptable:
#   - cargo (cargo_bin: HOST_CARGO, the host's rustup shim, which the operator can write)
#     runs only `cargo metadata --no-deps` in preflight, on the checkout, and it reads the operator's
#     cargo and rustup configuration. It decides nothing on its own: build_jope runs cargo metadata
#     again inside the pinned image, on the verified export, and refuses the run unless that answer
#     is also right. The builder image's own cargo, rustc and ldd are pinned by the image digest.
#   - gh-role (GH_ROLE, publish only) is the host's credential router, which the operator can write
#     and which reads the operator's configuration. It hands git the GitHub token and runs
#     `gh release view`, `create` and `download` as the writer role. It is trusted as the ubuntu
#     account is (README, "Key custody"), and it cannot change what was signed: it uploads dist/,
#     which is signed before it runs, and the readback compares what it downloads with dist/ byte for
#     byte and checks the signature again.
readonly -a HOST_TOOLS=(bash cat chmod chown cmp cp date diff docker dpkg-query env find git grep
  head id install jq ls mkdir mktemp objdump readlink realpath rm rmdir sed sh sha256sum sort stat
  sudo tail tar tee test timeout tr uname wc)
# The helpers that the host-tool checks themselves run, always by absolute path (see check_helpers).
readonly STAT=/usr/bin/stat READLINK=/usr/bin/readlink
# host_path[<name>]: the absolute path that require_tools checked for each host tool. Commands that
# root runs through sudo are given these paths, never a name.
declare -gA host_path=()
helpers_checked=0
# Docker passes a container no variable that is not named with -e, and an image's /tmp is its own,
# so the host's TMPDIR stops at the container boundary. builder_container therefore bind-mounts the
# run's tmp/container/ at CONTAINER_TMPDIR, sets TMPDIR to it, and mounts the same directory over the
# container's /tmp too, for glibc's tmpfile(3) and tools that ignore TMPDIR. CONTAINER_SYSTEM_TMP is a
# path inside the container, never on the host.
readonly CONTAINER_TMPDIR=/release-tmp
readonly CONTAINER_SYSTEM_TMP=/tmp

# The tools' own configuration. No per-user file (~/.jq, ~/.gitconfig, ~/.docker and the like) and
# no variable of the operator's may change what a pinned or host tool reports to a check or writes
# into a release. docs/release/README.md ("Tool configuration") has the audit, tool by tool, and what
# remains (the repository's own git config, ~/.netrc, a bash started without -p) and why it cannot.
# This block is all builtins and runs before main(), so every command the script starts inherits it.
#   - bash runs in privileged mode (see "bash itself" above), so it imported no function and applied
#     no BASH_ENV, SHELLOPTS or BASHOPTS. bash -p still reads a few variables that change what the
#     script does, and they go: BASH_COMPAT (another bash version's semantics, such as a quoted =~
#     operand read as a pattern), TMOUT (a timeout on every read, which ends a loop over a tool's
#     output at the first pause in it: extract_notes over git cat-file, verify_export over find),
#     FUNCNEST (a nesting limit, which stops the run wherever that depth is first reached) and
#     GLOBSORT (bash 5.3's glob order). BASH_ENV goes too, so that no child bash runs it. LC_ALL=C
#     is set here, before parse_args matches anything. Exported functions stay in the environment of
#     the commands the script runs, where only a bash would import them, and no host tool is a bash
#     script (README, "What remains").
#   - The dynamic loader's LD_* variables (LD_PRELOAD, LD_LIBRARY_PATH, LD_AUDIT and the rest) and
#     glibc's GCONV_PATH, LOCPATH and GLIBC_TUNABLES would load or tune code from a path of the
#     operator's choosing inside every host tool.
#   - git reads neither /etc/gitconfig nor ~/.gitconfig (GIT_CONFIG_NOSYSTEM, GIT_CONFIG_GLOBAL). On
#     xbabe2, ~/.gitconfig rewrites the GitHub URLs of other repositories to the local forge; one
#     more such line would send the fetch, the tag checks and the push somewhere else. git follows
#     no replace refs (GIT_NO_REPLACE_OBJECTS), which would make archive, ls-tree and cat-file show
#     another commit's tree as $sha's, and keeps no GIT_* variable but the committer's name and
#     email. The keys set through GIT_CONFIG_COUNT outrank every configuration file: no per-user
#     ignore or attributes file (~/.config/git/ignore hides untracked files from git status even
#     without ~/.gitconfig), no repository hooks, no fsmonitor answering git status, no signed tag,
#     and no push.followTags, which would push every other local annotated tag on the released
#     commit's history that origin lacks along with the release tag (push_release_tag also passes
#     --no-follow-tags).
#     check_git_config refuses URL rewrites, remotes named after $remote or another URL, and TLS
#     settings in the repository's own configuration.
#     The GitHub credential helper is the GH array's -c option, which outranks all of these.
#   - tar takes options from TAR_OPTIONS (--exclude=... drops files from the export, which
#     verify_export would then refuse), objdump its object format from GNUTARGET (GNUTARGET=binary
#     hides every NEEDED entry), and GNU tools and bash behave differently under POSIXLY_CORRECT.
#     grep 3.11 ignores GREP_OPTIONS; it goes anyway.
#   - JQ_*, DOCKER_* and DPKG_* go, and git's TLS trust stays the system's (SSL_CERT_FILE,
#     SSL_CERT_DIR, CURL_CA_BUNDLE). jq, docker and cosign get empty private configuration
#     directories from their wrappers (jq, docker, check_cosign and cosign_verify), and syft runs in
#     syft_env.
unset -v -- BASH_COMPAT TMOUT FUNCNEST GLOBSORT BASH_ENV
for _name in "${!LD_@}" "${!GIT_@}" "${!JQ_@}" "${!DOCKER_@}" "${!DPKG_@}"; do
  case $_name in GIT_COMMITTER_NAME|GIT_COMMITTER_EMAIL) ;; *) unset -v -- "$_name" ;; esac
done
unset -v -- _name GCONV_PATH LOCPATH GLIBC_TUNABLES TAR_OPTIONS GNUTARGET POSIXLY_CORRECT \
  GREP_OPTIONS SSL_CERT_FILE SSL_CERT_DIR CURL_CA_BUNDLE
export LC_ALL=C
export GIT_CONFIG_NOSYSTEM=1 GIT_CONFIG_GLOBAL=/dev/null GIT_NO_REPLACE_OBJECTS=1 GIT_CONFIG_COUNT=6 \
  GIT_CONFIG_KEY_0=core.excludesFile GIT_CONFIG_VALUE_0=/dev/null \
  GIT_CONFIG_KEY_1=core.attributesFile GIT_CONFIG_VALUE_1=/dev/null \
  GIT_CONFIG_KEY_2=core.hooksPath GIT_CONFIG_VALUE_2=/dev/null \
  GIT_CONFIG_KEY_3=core.fsmonitor GIT_CONFIG_VALUE_3=false \
  GIT_CONFIG_KEY_4=tag.gpgSign GIT_CONFIG_VALUE_4=false \
  GIT_CONFIG_KEY_5=push.followTags GIT_CONFIG_VALUE_5=false
# The docker client talks to the system daemon through this socket, whatever DOCKER_HOST or a docker
# context says (see the docker wrapper). It is set after the DOCKER_* variables are gone.
readonly DOCKER_SOCKET=unix:///var/run/docker.sock
# Test mode is confined to the invoking account's test scratch root (see "Testing" above). main()
# resolves it with realpath right after pin_tmpdir: only builtins run before TMPDIR is pinned.
TEST_ROOT=''
readonly MANIFEST_SCHEMA=jope.release-manifest/v1
readonly BUILD_COMMAND="cargo build --release --locked -p jope-cli --bin jope --message-format=json-render-diagnostics"
# The same prefixes as GPU_LIBRARY_PREFIXES in crates/jope-engine/src/build_receipt.rs (lowercase).
readonly -a GPU_LIBRARY_PREFIXES=(libcuda libcudart libcudnn libcublas libcurand libcusparse
  libcufft libcusolver libcupti libcufile libcutensor libnpp libnvjpeg libnccl libnvrtc
  libnvjitlink libnvtoolsext libnvidia libtorch libc10)
# These git -c arguments send GitHub HTTPS credentials through the writer role, and only that role.
readonly -a GH=(-c 'credential.https://github.com.helper='
  -c "credential.https://github.com.helper=!$GH_ROLE writer auth git-credential")

# The sign step, run by root's sh as: sh -c "$SIGN_SCRIPT" sh <sig> <sums> <cosign> <sha256>.
# The password is read by root's shell and reaches only the environment of the pinned cosign copy.
# It never appears in an argument list, in the calling shell's environment, or in the log.
# The literal /etc/jope-release must stay in step with KEY_DIR. The copy goes under /root (0700,
# on /) because /run is mounted noexec on xbabe2. sudo resets the environment, so root's shell
# inherits no TMPDIR: mktemp is given its directory explicitly, and before anything else runs the
# shell exports TMPDIR as tmp/ (0700) inside that root-only work directory, which the EXIT trap
# removes. cosign, cp, chmod and sha256sum all see it. Its first line after set -eu makes root's
# PATH the literal TRUSTED_PATH, so that mktemp and the rest resolve to the files require_tools
# checked rather than through sudo's secure_path, which searches /usr/local first. It is read with
# the read builtin, so that nothing but builtins runs before main() pins TMPDIR; the here-document is
# far smaller than a pipe buffer.
# shellcheck disable=SC2016  # expanded by root's sh, not here
IFS= read -r -d '' SIGN_SCRIPT <<'ROOT' || true
set -eu
PATH=/usr/sbin:/usr/bin:/sbin:/bin
export PATH
work=$(mktemp -d /root/jope-release-sign.XXXXXX)
trap 'rm -rf -- "$work"' EXIT
TMPDIR=$work/tmp
export TMPDIR
mkdir -m 0700 -- "$TMPDIR"
cp -- "$3" "$work/cosign"
chmod 0500 -- "$work/cosign"
got=$(sha256sum < "$work/cosign")
got=${got%% *}
if [ "$got" != "$4" ]; then
  echo "cosign at $3 has sha256 $got, not the pinned $4; refusing to run it as root" >&2
  exit 3
fi
cd /etc/jope-release
COSIGN_PASSWORD="$(cat signing.password)" "$work/cosign" sign-blob --yes --key signing.key \
  --tlog-upload=false --output-signature "$1" "$2"
ROOT
readonly SIGN_SCRIPT

tag='' sha='' version='' publish=0 mode=dryrun test_mode=0 stamp=''
remote=$REPO_URL cargo_bin=$HOST_CARGO key_dir=$KEY_DIR out_root=$EVIDENCE_ROOT
repo_root='' release_root='' run_dir='' target_root='' dist='' stage=init
metadata_json='' notes_text='' bin='' bin_container='' observed_glibc='' tee_pid=''
tag_object='' release_url='' checks_file='' src_dir='' src_tree='' last_check_ok=''
syft_bin=$SYFT syft_report='' cosign_report=''
early_log=()

say() { printf 'release.sh: %s\n' "$*" >&2; }
note() { early_log+=("$*"); say "$*"; }
die() { printf 'release.sh: REFUSED: %s\n' "$*" >&2; exit 1; }
step() { printf '\n== %s  %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >&2; }
sha256_of() { local h; h=$(sha256sum -- "$1"); printf '%s\n' "${h%% *}"; }

usage() {
  cat <<'EOF'
usage: deploy/release/release.sh <vX.Y.Z> <merge-commit-sha> [--publish]

Without --publish this is a dry run: it builds, signs, verifies and writes the evidence, but it
never tags, pushes or publishes. Run it on xbabe2 from the root of a clean checkout whose HEAD is
the merge commit and equals origin/main. See docs/release/README.md.
EOF
}

# --- temporary files -----------------------------------------------------------------------------

# main() calls this before anything else, and only builtins run before the export (test-release.sh
# checks both). With TMPDIR unset, bash writes a here-string or here-document larger than a pipe
# buffer to /tmp/sh-thd.*, and sort, mktemp, git and other tools use /tmp as well; nothing of a
# release may touch /tmp. A real run pins $EVIDENCE_ROOT/release-tmp. When any JOPE_RELEASE_TEST*
# variable is set it pins release-tmp under the test root instead, so that the tests never need the
# real evidence root; init_test_mode then refuses the run unless test mode is valid. open_run_dir
# moves TMPDIR into the run directory once that exists.
pin_tmpdir() {
  local base=$EVIDENCE_ROOT
  if compgen -v JOPE_RELEASE_TEST >/dev/null; then base=${HOME:?HOME must be set}/.cache/jope-test-tmp; fi
  export TMPDIR=$base/release-tmp
  mkdir -p -- "$base" || die "cannot create $base"
  mkdir -m 0700 -- "$TMPDIR" 2>/dev/null || [[ -d $TMPDIR ]] || die "cannot create TMPDIR $TMPDIR"
  [[ -d $TMPDIR && ! -L $TMPDIR && -O $TMPDIR ]] || die "TMPDIR $TMPDIR is not a real directory owned by $(id -un)"
  chmod 0700 -- "$TMPDIR" || die "cannot make TMPDIR $TMPDIR private (0700)"
}

# --- the tools' configuration directories ---------------------------------------------------------

# private_dir <dir>: creates <dir> (0700) when it is missing, and refuses it unless it is a real,
# empty directory owned by the operator. The wrappers below give a tool one of these as its HOME or
# its configuration directory, so that the tool finds no configuration there. Once the directory
# exists, only builtins run.
private_dir() {
  if [[ ! -e $1 && ! -L $1 ]]; then
    mkdir -m 0700 -- "$1" 2>/dev/null || [[ -d $1 ]] || die "cannot create $1"
  fi
  [[ -d $1 && ! -L $1 && -O $1 ]] || die "$1 is not a real directory owned by $(id -un)"
  if compgen -G "$1/*" >/dev/null || compgen -G "$1/.[!.]*" >/dev/null || compgen -G "$1/..?*" >/dev/null; then
    die "$1 is not empty; a tool would read its configuration from there"
  fi
}

# jq 1.7 reads $HOME/.jq (and $HOME/jq/main.jq) into every program it runs, and a definition there
# changes what the program means: one that redefines startswith and any makes check_sboms accept a
# document that is not SPDX, and the same could change write_manifest's signed output, VERSION_JQ's
# verdict or select_executable's choice. So every jq of this script is this function, which runs jq
# with HOME set to jq-home/, an empty private directory in the current TMPDIR. jq reads no other file
# of its own, and its other variables (JQ_*) are unset above.
jq() {
  private_dir "${TMPDIR:?}/jq-home"
  HOME=$TMPDIR/jq-home command jq "$@"
}

# The docker client reads ~/.docker/config.json (a current context, which chooses the daemon; proxy
# variables, which it adds to every container; credential helpers and plugin hooks, which it runs)
# and takes its daemon from DOCKER_HOST or a context. So every docker command of this script is this
# function: DOCKER_CONFIG is docker-config/, an empty private directory in the current TMPDIR, and
# DOCKER_HOST is the system daemon's socket; no other DOCKER_* variable is set (unset above). The
# builder image is pinned by digest as well.
docker() {
  private_dir "${TMPDIR:?}/docker-config"
  DOCKER_CONFIG=$TMPDIR/docker-config DOCKER_HOST=$DOCKER_SOCKET command docker "$@"
}

# --- argument and test-mode handling -----------------------------------------------------------

valid_tag() { [[ $1 =~ ^v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$ ]]; }

parse_args() {
  if [[ ${1-} == -h || ${1-} == --help ]]; then usage; exit 0; fi
  case $# in
    2) ;;
    3) [[ $3 == --publish ]] || { usage >&2; die "unknown option: $3"; }; publish=1 ;;
    *) usage >&2; die "expected <tag> <merge-commit-sha> [--publish]" ;;
  esac
  tag=$1 sha=$2
  valid_tag "$tag" \
    || die "tag must match ^v[0-9]+\\.[0-9]+\\.[0-9]+\$ (semver, no leading zeros, no suffix): '$tag'"
  [[ $sha =~ ^[0-9a-f]{40}$ ]] || die "merge commit must be a full 40-character lowercase SHA-1: '$sha'"
  version=${tag#v}
  if (( publish )); then mode=publish; fi
}

under_test_root() {
  local canonical
  canonical=$(realpath -m -- "$1")
  [[ $canonical/ == "$TEST_ROOT"/* ]]
}

init_test_mode() {
  local -a names
  mapfile -t names < <(compgen -v JOPE_RELEASE_TEST || true)
  (( ${#names[@]} )) || return 0
  if [[ ${JOPE_RELEASE_TEST:-} != 1 ]] || ! under_test_root "$(pwd -P)"; then
    die "test overrides (${names[*]}) are only honoured with JOPE_RELEASE_TEST=1 inside $TEST_ROOT/; unset them for a real release"
  fi
  local var
  for var in JOPE_RELEASE_TEST_REMOTE JOPE_RELEASE_TEST_CARGO JOPE_RELEASE_TEST_OUTROOT \
    JOPE_RELEASE_TEST_KEYDIR; do
    [[ -n ${!var:-} ]] || die "test mode requires $var"
    under_test_root "${!var}" || die "$var must be under $TEST_ROOT/: ${!var}"
  done
  if [[ -n ${JOPE_RELEASE_TEST_STAMP:-} && ! $JOPE_RELEASE_TEST_STAMP =~ ^[0-9]{8}T[0-9]{6}Z$ ]]; then
    die "JOPE_RELEASE_TEST_STAMP must look like 20260928T000000Z"
  fi
  test_mode=1
  remote=$JOPE_RELEASE_TEST_REMOTE
  cargo_bin=$JOPE_RELEASE_TEST_CARGO
  out_root=$JOPE_RELEASE_TEST_OUTROOT
  key_dir=$JOPE_RELEASE_TEST_KEYDIR
  # Optional: a syft stand-in, which preflight still checks against the real pin (see preflight).
  if [[ -n ${JOPE_RELEASE_TEST_SYFT:-} ]]; then
    under_test_root "$JOPE_RELEASE_TEST_SYFT" \
      || die "JOPE_RELEASE_TEST_SYFT must be under $TEST_ROOT/: $JOPE_RELEASE_TEST_SYFT"
    syft_bin=$JOPE_RELEASE_TEST_SYFT
  fi
  if (( publish )); then die "test mode never publishes"; fi
}

host_name() {
  if (( test_mode )) && [[ -n ${JOPE_RELEASE_TEST_HOSTNAME:-} ]]; then
    printf '%s\n' "$JOPE_RELEASE_TEST_HOSTNAME"
  else
    uname -n
  fi
}

run_stamp() {
  if (( test_mode )) && [[ -n ${JOPE_RELEASE_TEST_STAMP:-} ]]; then
    printf '%s\n' "$JOPE_RELEASE_TEST_STAMP"
  else
    date -u +%Y%m%dT%H%M%SZ
  fi
}

# Root-only key custody. In test mode a plain directory under the test root stands in, with no sudo.
# sudo is given the path that require_tools checked (host_path), never a name to look up.
key_exists() {
  if (( test_mode )); then
    [[ -e $key_dir/$1 ]]
  else
    sudo -n "${host_path[test]:?require_tools has not checked test}" -e "$key_dir/$1"
  fi
}
key_cat() {
  if (( test_mode )); then
    cat -- "$key_dir/$1"
  else
    sudo -n "${host_path[cat]:?require_tools has not checked cat}" -- "$key_dir/$1"
  fi
}

# --- pure helpers (unit-tested by test-release.sh) ----------------------------------------------

# Prints the names of the regular files directly in $1, sorted in the C locale. The mode is
# "artifacts" (skip SHA256SUMS-* and *.sig) or "all". It refuses symlinks, subdirectories, other
# file types, and names outside [A-Za-z0-9._+-], which keeps sha256sum lines unambiguous.
dist_names() (
  dir=$1 which=${2:-all}
  shopt -s nullglob dotglob
  [[ -d $dir && ! -L $dir ]] || die "not a real directory: $dir"
  names=()
  for path in "$dir"/*; do
    name=${path##*/}
    [[ ! -L $path ]] || die "refusing symlink in $dir: $name"
    [[ -f $path ]] || die "refusing non-regular file (directory or special) in $dir: $name"
    [[ $name =~ ^[A-Za-z0-9][A-Za-z0-9._+-]*$ ]] || die "refusing file name outside [A-Za-z0-9._+-] in $dir: '$name'"
    if [[ $which == artifacts ]]; then
      case $name in SHA256SUMS-*|*.sig) continue ;; esac
    fi
    names+=("$name")
  done
  if (( ${#names[@]} )); then printf '%s\n' "${names[@]}" | LC_ALL=C sort; fi
)

# Writes $1/SHA256SUMS-$2.txt covering every regular file in $1 except SHA256SUMS-* and *.sig,
# sorted by name, in `sha256sum -c` format.
write_sums() {
  local dir=$1 t=$2 listing sums
  listing=$(dist_names "$dir" artifacts) || exit 1
  [[ -n $listing ]] || die "no release files to checksum in $dir"
  local -a names
  mapfile -t names <<<"$listing"
  sums=$(cd "$dir" && sha256sum -- "${names[@]}") || die "sha256sum failed in $dir"
  printf '%s\n' "$sums" > "$dir/SHA256SUMS-$t.txt"
}

# Prints {"name": "sha256", ...} for dist_names "$1" "$2", in name order.
digests_json() {
  local dir=$1 listing
  listing=$(dist_names "$dir" "$2") || exit 1
  if [[ -z $listing ]]; then printf '{}\n'; return 0; fi
  local -a names
  mapfile -t names <<<"$listing"
  (cd "$dir" && sha256sum -- "${names[@]}") \
    | jq -Rn '[inputs | capture("^(?<h>[0-9a-f]{64})  (?<n>.+)$") | {key: .n, value: .h}] | from_entries'
}

# Reads a CHANGELOG on stdin and prints the section "## [<version>] ..." up to the next "## [".
# Returns 1 when the section is missing or has no "- " entries.
extract_notes() {
  local want="## [$1]" line in=0 entries=0
  local -a lines=()
  while IFS= read -r line || [[ -n $line ]]; do
    if [[ $line == '## ['* ]]; then
      if (( in )); then break; fi
      if [[ $line == "$want" || $line == "$want "* ]]; then in=1; fi
    fi
    [[ $line != '<!-- generated by git-cliff -->' ]] || continue
    if (( in )); then
      lines+=("$line")
      if [[ $line == '- '* ]]; then entries=$((entries + 1)); fi
    fi
  done
  (( entries > 0 )) || return 1
  printf '%s\n' "${lines[@]}"
}

# Reads objdump output on stdin and prints the highest GLIBC_x.y[.z] version, or nothing.
max_glibc() {
  { grep -oE 'GLIBC_[0-9]+(\.[0-9]+)+' || true; } | sed 's/^GLIBC_//' | sort -Vu | tail -n1
}

# True when version $1 <= version $2 (sort -V ordering, so 2.4 < 2.36).
version_le() { [[ $(printf '%s\n%s\n' "$1" "$2" | sort -V | tail -n1) == "$2" ]]; }

# Reads `objdump -p` output on stdin and prints its NEEDED entries.
needed_libs() { sed -n 's/^[[:space:]]*NEEDED[[:space:]]\{1,\}\([^[:space:]]\{1,\}\).*/\1/p'; }

# Reads library names on stdin and prints those that are CUDA, NVIDIA or LibTorch libraries.
gpu_libs() {
  local lib prefix
  while IFS= read -r lib; do
    for prefix in "${GPU_LIBRARY_PREFIXES[@]}"; do
      if [[ ${lib,,} == "$prefix"* ]]; then printf '%s\n' "$lib"; break; fi
    done
  done
}

# Prints the path (inside the container) of the single bin executable $3 of the package whose
# manifest is $2, taken from the cargo JSON messages in $1. It fails unless the build finished
# successfully and exactly one such executable was reported. (It replaces the Python
# cargo-artifact.py select.)
select_executable() {
  jq -Rnr --arg m "$2" --arg n "$3" '
    [inputs | select(startswith("{")) | fromjson] as $msgs
    | ([$msgs[] | select(.reason == "build-finished")] | last | .success?) as $ok
    | [$msgs[] | select(.reason == "compiler-artifact" and .manifest_path == $m
        and .target.name == $n and any(.target.kind[]; . == "bin")
        and ((.profile.test // false) | not) and .executable != null) | .executable] as $exes
    | if $ok != true then error("cargo did not report a successful completed build")
      elif ($exes | length) != 1 then
        error("expected exactly one executable for \($n) in \($m), found \($exes | length)")
      else $exes[0] end' "$1"
}

# Maps a container path under /target to the host path under $target_root.
map_target_path() {
  local p=$1
  [[ $p == /target/?* && $p != *'/../'* && $p != *'/..' ]] || die "executable is not under /target: $p"
  printf '%s/%s\n' "$target_root" "${p#/target/}"
}

# Runs binary $1 without arguments, with an empty environment except PATH and TMPDIR (env -i drops
# the exported TMPDIR, so it is passed on by name) and $3 as the working directory. It prints the
# output and succeeds only when the first line is exactly "jope $2", which is the line the README
# cites for a bare `jope`.
binary_prints_version() {
  local out
  out=$(cd "$3" && env -i PATH=/usr/bin:/bin TMPDIR="${TMPDIR:?}" timeout 60 "$1" </dev/null) || return 1
  printf '%s\n' "$out"
  [[ ${out%%$'\n'*} == "jope $2" ]]
}

# The sha256 of the committed blob $1:$2, or nothing when the path is absent at that commit.
blob_sha256() {
  local h
  if [[ $(git cat-file -t "$1:$2" 2>/dev/null) == blob ]]; then
    h=$(git cat-file blob "$1:$2" | sha256sum)
    printf '%s\n' "${h%% *}"
  fi
}

# models/PINNED at commit $1 as a JSON array, with blank and '#' lines dropped ([] when absent).
pinned_models() {
  if [[ $(git cat-file -t "$1:models/PINNED" 2>/dev/null) == blob ]]; then
    git cat-file blob "$1:models/PINNED" \
      | jq -Rnc '[inputs | gsub("^\\s+|\\s+$"; "") | select(length > 0 and (startswith("#") | not))]'
  else
    printf '[]\n'
  fi
}

# Writes $1/release-manifest-$2.json. Its arguments are the dist directory, the tag, the commit,
# the crate version and the observed max glibc. The artifacts are every release file in dist
# except SHA256SUMS-* and *.sig; the manifest itself does not exist yet when this runs.
write_manifest() {
  local dir=$1 t=$2 commit=$3 ver=$4 observed=$5 out artifacts models datasets pinned
  out=$dir/release-manifest-$t.json
  [[ ! -e $out ]] || die "$out already exists"
  artifacts=$(digests_json "$dir" artifacts) || exit 1
  models=$(blob_sha256 "$commit" models/manifest.json)
  datasets=$(blob_sha256 "$commit" datasets/manifest.json)
  pinned=$(pinned_models "$commit")
  jq -n --arg schema "$MANIFEST_SCHEMA" --arg tag "$t" --arg sha "$commit" --arg ver "$ver" \
    --arg image "$BUILDER_IMAGE" --arg max "$MAX_GLIBC" --arg observed "$observed" \
    --arg models "$models" --arg datasets "$datasets" \
    --arg syft_ver "$SYFT_VERSION" --arg syft_sha "$SYFT_SHA256" \
    --arg cosign_ver "$COSIGN_VERSION" --arg cosign_sha "$COSIGN_SHA256" \
    --argjson pinned "$pinned" --argjson artifacts "$artifacts" '
    {schema: $schema, tag: $tag, git_sha: $sha, crate_version: $ver, builder_image: $image,
     tools: {syft: {version: $syft_ver, sha256: $syft_sha},
             cosign: {version: $cosign_ver, sha256: $cosign_sha}},
     max_glibc: $max, observed_max_glibc: $observed, models_pinned: $pinned,
     models_manifest_sha256: (if $models == "" then null else $models end),
     datasets_manifest_sha256: (if $datasets == "" then null else $datasets end),
     artifacts: $artifacts}' > "$out"
}

# The cargo metadata verdict: "OK" when jope-cli (which carries [workspace.package] version) and
# every other workspace member outside crates/smartcluster/ are at $want. Otherwise it prints one
# MISMATCH or ERROR line per problem.
# shellcheck disable=SC2016  # a jq program; $root, $want etc. are jq variables
readonly VERSION_JQ='
  .workspace_root as $root
  | .workspace_members as $members
  | [.packages[] | select(.id as $id | any($members[]; . == $id))] as $ws
  | ($ws | map(select(.name == "jope-cli"))) as $cli
  | if ($cli | length) != 1 then "ERROR expected one jope-cli workspace member, found \($cli | length)"
    elif $cli[0].version != $want then "MISMATCH jope-cli \($cli[0].version) != tag version \($want)"
    else
      [$ws[] | select(.manifest_path | startswith($root + "/crates/smartcluster/") | not)
             | select(.version != $want)
             | "MISMATCH \(.name) \(.version) != \($want) (\(.manifest_path))"]
      | if length == 0 then "OK" else join("\n") end
    end'

# --- tools ---------------------------------------------------------------------------------------

# pinned_digest <name> <path> <sha256> <version>: refuses <path> unless it is an executable file
# whose sha256 is the pin. It never runs <path>.
pinned_digest() {
  local got
  [[ -f $2 && -x $2 ]] || die "required tool missing: $2"
  got=$(sha256_of "$2")
  [[ $got == "$3" ]] \
    || die "$2 has sha256 $got, not the pinned $1 $4 ($3); re-pin it in release.sh through a reviewed change"
}

# syft_env <dir> <command...>: runs a syft command with nothing inherited from the caller. TMPDIR,
# HOME, the XDG directories and syft's cache all lie under <dir> (0700), so neither an operator's
# syft configuration nor an earlier run's cache can change what syft reports, and syft writes
# nothing outside <dir> except the files its -o options name.
syft_env() {
  local dir=$1
  shift
  mkdir -p -- "$dir/home" "$dir/config" "$dir/cache" "$dir/tmp" || die "cannot create $dir"
  chmod 0700 -- "$dir" || die "cannot make $dir private (0700)"
  env -i PATH=/usr/bin:/bin TMPDIR="$dir/tmp" HOME="$dir/home" XDG_CONFIG_HOME="$dir/config" \
    XDG_CACHE_HOME="$dir/cache" SYFT_CACHE_DIR="$dir/cache/syft" SYFT_CHECK_FOR_APP_UPDATE=false "$@"
}

# check_syft <path> <sha256> <version> <dir>: refuses a syft that is not the pinned build. The digest
# is checked first, so a replaced binary is never run; then `syft version`, run in syft_env <dir>,
# must report <version>. It prints syft's version document (JSON).
check_syft() {
  local out ver
  pinned_digest syft "$1" "$2" "$3"
  out=$(syft_env "$4" "$1" version -o json) || die "$1 version -o json failed"
  ver=$(printf '%s\n' "$out" | jq -r '.version? // empty' 2>/dev/null) || ver=''
  [[ $ver == "$3" ]] \
    || die "$1 reports version '$ver', not the pinned syft $3; re-pin SYFT_VERSION and SYFT_SHA256 in release.sh through a reviewed change"
  printf '%s\n' "$out"
}

# check_cosign <path> <sha256> <version>: the same for cosign, whose version is the gitVersion of
# `cosign version --json`. cosign reads COSIGN_*, SIGSTORE_*, TUF_ROOT and token variables and keeps
# its TUF cache under ~/.sigstore, so on the host it always runs with an empty environment: PATH,
# TMPDIR, and HOME set to cosign-home/, an empty private directory (the same in cosign_verify).
check_cosign() {
  local out ver
  pinned_digest cosign "$1" "$2" "$3"
  private_dir "${TMPDIR:?}/cosign-home"
  out=$(env -i PATH=/usr/bin:/bin HOME="$TMPDIR/cosign-home" TMPDIR="$TMPDIR" "$1" version --json 2>/dev/null) \
    || die "$1 version --json failed"
  ver=$(printf '%s\n' "$out" | jq -r '.gitVersion? // empty' 2>/dev/null) || ver=''
  [[ $ver == "$3" ]] \
    || die "$1 reports version '$ver', not the pinned cosign $3; re-pin COSIGN_VERSION and COSIGN_SHA256 in release.sh through a reviewed change"
  printf '%s\n' "$out"
}

# trusted_chain <absolute path>: succeeds only when the operator cannot have replaced what <path>
# names. It walks the path from / down, one component at a time, with lstat (STAT, never following a
# link): every directory on the way and the final file must be owned by root and closed to group
# and other writes, and every symlink must be owned by root and is then followed, its target walked
# the same way, from the link's own directory or from / when it is absolute. So a directory that the
# operator owns anywhere on the way refuses the path, even when a symlink in it leads to root-owned
# directories (the operator could swap that symlink), and so does any directory on the way to a
# symlink's target. At most 40 symlinks are followed, as the kernel does. The final file must be a
# regular file. It says why it refuses and returns 1; it never exits. It runs only STAT and READLINK
# (see check_helpers) besides builtins.
trusted_chain() {
  local rest=$1 at='' comp p out owner mode kind='' hops=0
  if [[ $rest != /* ]]; then say "not an absolute path: '$1'"; return 1; fi
  out=$("$STAT" -c '%u %a %F' -- /) || { say "cannot stat /"; return 1; }
  if [[ ! $out =~ ^([0-9]+)\ ([0-7]+)\ (.+)$ ]] || [[ ${BASH_REMATCH[1]} != 0 ]] \
    || (( 8#${BASH_REMATCH[2]} & 8#022 )); then
    say "/ is not owned by root and closed to group and other writes ($out)"
    return 1
  fi
  while :; do
    while [[ $rest == /* ]]; do rest=${rest#/}; done
    [[ -n $rest ]] || break
    comp=${rest%%/*}
    if [[ $rest == */* ]]; then rest=${rest#*/}; else rest=''; fi
    case $comp in
      .) continue ;;
      ..) at=${at%/*}; continue ;;
    esac
    p=$at/$comp
    out=$("$STAT" -c '%u %a %F' -- "$p" 2>/dev/null) || { say "$1: cannot stat $p"; return 1; }
    [[ $out =~ ^([0-9]+)\ ([0-7]+)\ (.+)$ ]] || { say "$1: unexpected stat output for $p: $out"; return 1; }
    owner=${BASH_REMATCH[1]} mode=${BASH_REMATCH[2]} kind=${BASH_REMATCH[3]}
    if [[ $kind == 'symbolic link' ]]; then
      hops=$((hops + 1))
      (( hops <= 40 )) || { say "$1: more than 40 symlinks"; return 1; }
      [[ $owner == 0 ]] || { say "$1: $p is a symlink owned by uid $owner, not root"; return 1; }
      out=$("$READLINK" -- "$p") || out=''
      [[ -n $out ]] || { say "$1: cannot read the symlink $p"; return 1; }
      # An absolute target is walked from /, a relative one from the link's directory ($at).
      if [[ $out == /* ]]; then at=''; fi
      rest=$out${rest:+/$rest}
      continue
    fi
    if [[ $owner != 0 ]] || (( 8#$mode & 8#022 )); then
      say "$1: $p is not owned by root and closed to group and other writes (uid $owner, mode $mode)"
      return 1
    fi
    at=$p
  done
  [[ $kind == 'regular file' || $kind == 'regular empty file' ]] \
    || { say "$1: resolves to ${at:-/}, which is a $kind, not a regular file"; return 1; }
}

# check_helpers: STAT and READLINK are what trusted_chain trusts, so they are trusted only once the
# operator provably cannot have replaced them. First with bash builtins alone ([[ -w ]] is access(2)
# for this account): none of /, /usr, /usr/bin or the two files may be a symlink or writable by the
# operator. Then trusted_chain, run by them, must accept both. Root can write every path, so the -w
# test says nothing for it; root is inside the trust boundary, and the ownership checks still apply.
check_helpers() {
  local p
  if (( EUID != 0 )); then
    for p in / /usr /usr/bin "$STAT" "$READLINK"; do
      [[ -e $p && ! -L $p && ! -w $p ]] \
        || die "$p is missing, a symlink, or writable by $(id -un), so $STAT and $READLINK cannot be trusted to check the host tools"
    done
  fi
  for p in "$STAT" "$READLINK"; do
    trusted_chain "$p" || die "$p must be root-owned and closed to group and other writes, and so must every directory above it"
  done
  helpers_checked=1
}

# resolve_host_tool <name>: the file that a lookup of <name> in TRUSTED_PATH finds, in builtins only:
# the first <dir>/<name> that is an executable file, as bash searches PATH.
resolve_host_tool() {
  local rest=$TRUSTED_PATH: dir
  [[ -n $1 && $1 != */* ]] || return 1
  while [[ -n $rest ]]; do
    dir=${rest%%:*} rest=${rest#*:}
    if [[ -n $dir && -f $dir/$1 && -x $dir/$1 ]]; then printf '%s\n' "$dir/$1"; return 0; fi
  done
  return 1
}

# check_host_tool <name>: refuses a host tool that the operator could have replaced, and prints its
# absolute path. The name is looked up in TRUSTED_PATH only, whatever PATH a caller had; the file
# found there must pass trusted_chain; and a lookup of the name in this shell must return that same
# file, so no later lookup can run another one.
check_host_tool() {
  local path
  (( helpers_checked )) || check_helpers
  path=$(resolve_host_tool "$1") || die "required tool missing from $TRUSTED_PATH: $1"
  [[ $PATH == "$TRUSTED_PATH" ]] || die "PATH is not $TRUSTED_PATH"
  [[ $(type -P -- "$1") == "$path" ]] || die "a lookup of $1 would not run $path"
  trusted_chain "$path" \
    || die "host tool $1 resolves to $path, which is not a root-owned file reached only through root-owned directories and symlinks closed to group and other writes; use the root-owned system copy"
  printf '%s\n' "$path"
}

# Prints the tools of this run as JSON: the pinned syft and cosign with the version documents
# preflight read from them, and for every host tool its path (as require_tools checked it), the file
# it resolves to, that file's sha256 and its Debian package and version.
tools_json() {
  local name path real pkg ver
  local -a rows=()
  for name in "${HOST_TOOLS[@]}"; do
    path=${host_path[$name]:?require_tools has not checked $name}
    real=$(realpath -e -- "$path") || die "cannot resolve $path"
    pkg=$(dpkg-query -S -- "$real" 2>/dev/null) || pkg=$(dpkg-query -S -- "$path" 2>/dev/null) || pkg=''
    pkg=${pkg%%$'\n'*}
    pkg=${pkg%%: *}
    pkg=${pkg%%:*}
    ver=''
    # shellcheck disable=SC2016  # a dpkg-query format, not a shell expansion
    if [[ -n $pkg ]]; then ver=$(dpkg-query -W -f '${Version}' -- "$pkg" 2>/dev/null) || ver=''; fi
    rows+=("$(jq -nc --arg name "$name" --arg path "$path" --arg real "$real" \
      --arg sha "$(sha256_of "$real")" --arg pkg "$pkg" --arg ver "$ver" '
      {name: $name, path: $path, resolved: $real, sha256: $sha,
       package: (if $pkg == "" then null else $pkg end),
       package_version: (if $ver == "" then null else $ver end)}')")
  done
  printf '%s\n' "${rows[@]}" | jq -s --argjson syft "$syft_report" --argjson cosign "$cosign_report" \
    --arg syft_path "$SYFT" --arg syft_ver "$SYFT_VERSION" --arg syft_sha "$SYFT_SHA256" \
    --arg cosign_path "$COSIGN" --arg cosign_ver "$COSIGN_VERSION" --arg cosign_sha "$COSIGN_SHA256" '
    {schema: "jope.release-tools/v1",
     pinned: {syft: {path: $syft_path, version: $syft_ver, sha256: $syft_sha, reported: $syft},
              cosign: {path: $cosign_path, version: $cosign_ver, sha256: $cosign_sha, reported: $cosign}},
     host: .}'
}

# --- guards --------------------------------------------------------------------------------------

require_tools() {
  local tool
  check_helpers
  for tool in "${HOST_TOOLS[@]}"; do
    host_path[$tool]=$(check_host_tool "$tool") || exit 1
  done
  # The interpreter: /bin/bash through the #! line, or the bash that ran `bash -p release.sh`.
  trusted_chain "$BASH" \
    || die "this script runs in $BASH, not the root-owned system bash; run deploy/release/release.sh itself (its #! line is /bin/bash -p)"
  [[ -f $HOST_CARGO && -x $HOST_CARGO ]] || die "required tool missing: $HOST_CARGO"
  # Checked here, before anything is built or signed. The sign step checks root's own copy of
  # cosign again, make_sbom its copy of syft, and cosign_verify cosign before each verification.
  cosign_report=$(check_cosign "$COSIGN" "$COSIGN_SHA256" "$COSIGN_VERSION") || exit 1
  syft_report=$(check_syft "$syft_bin" "$SYFT_SHA256" "$SYFT_VERSION" "$TMPDIR/syft-preflight") || exit 1
  if (( publish )); then [[ -x $GH_ROLE ]] || die "required tool missing: $GH_ROLE"; fi
}

# git status does not look at index entries marked assume-unchanged or skip-worktree, so a clean
# status says nothing about them. Every entry must be a plain tracked file ("H" in ls-files -v).
check_index_flags() {
  local flagged
  flagged=$(git ls-files -v | grep -v '^H ' | head -n 20) || true
  [[ -z $flagged ]] \
    || die "index entries are marked assume-unchanged (lowercase), skip-worktree (S) or unmerged, so git status cannot vouch for them (clear them with git update-index --no-assume-unchanged / --no-skip-worktree):"$'\n'"$flagged"
}

# git reads no /etc/gitconfig or ~/.gitconfig here (see "The tools' own configuration"), so the
# repository's own configuration is the only one left that could send the fetch, the tag checks or
# the push to a server other than $remote. Three kinds of key could, and none of them may be set:
#   - a URL rewrite: url.<base>.insteadOf or url.<base>.pushInsteadOf;
#   - a remote named $remote itself, or after any other URL (a name that contains ://). git looks a
#     URL given on its command line up as a remote name first, so remote.<that URL>.url would send
#     the fetch, the tag checks and the push to another repository, its pushurl would send the push
#     alone, and its vcs, proxy, uploadpack and receivepack keys would apply as well. Remotes with
#     other names, such as origin, are left alone: every git command here is given $remote, never a
#     remote name. (git ignores a remote whose name begins with /, so a test remote that is an
#     absolute path cannot be redirected this way.)
#   - a TLS setting, which would let another server answer for $remote: http.sslVerify,
#     http.sslCAInfo or http.sslCAPath, also per URL.
# Key names can hold credentials, so they are counted, not printed.
check_git_config() {
  local rc=0 out key n=0
  out=$(git config --name-only --get-regexp \
    '^url\..*\.(push)?insteadof$|^remote\.|^http\.(.*\.)?(sslverify|sslcainfo|sslcapath)$') || rc=$?
  case $rc in
    0) ;;
    1) return 0 ;;
    *) die "cannot read the repository's git config (git config exited $rc)" ;;
  esac
  # A key is <section>.<name>.<variable>, with only the section and the variable lowercased, so the
  # name is compared exactly, as git compares it.
  while IFS= read -r key; do
    case $key in
      "remote.$remote".* | remote.*://*) n=$((n + 1)) ;;
      remote.*) ;;
      *) n=$((n + 1)) ;;
    esac
  done <<<"$out"
  (( n == 0 )) \
    || die "the repository's git config sets $n key(s) that could send the fetch, the tag checks or the push somewhere other than $remote: a URL rewrite (url.*.insteadOf, url.*.pushInsteadOf), a remote named $remote or after another URL (remote.<URL>.*), or a TLS setting (http.*sslVerify, http.*sslCAInfo, http.*sslCAPath); list them with: git config --show-origin --get-regexp '^(url|remote|http)\\.'"
}

preflight() {
  stage=preflight
  local host here top dirty head origin_main remote_tags channel verdict pub_sha
  host=$(host_name)
  # An exact match on the short host name (up to the first dot): xbabe20 or my-xbabe2 is not xbabe2.
  [[ ${host%%.*} == "$REQUIRED_HOST" ]] || die "must run on $REQUIRED_HOST (hostname is '$host')"
  note "ok: host $host"
  note "ok: TMPDIR $TMPDIR (0700)"
  if (( ! test_mode )); then
    require_tools
    note "ok: PATH is $TRUSTED_PATH; every host tool, bash and every directory and symlink on the way to them are root-owned and closed to group and other writes; cosign $COSIGN_VERSION and syft $SYFT_VERSION match their pins"
  elif [[ -n ${JOPE_RELEASE_TEST_SYFT:-} ]]; then
    # Test mode has no docker, sudo or pinned tools. A syft stand-in is still checked against the
    # real pin, so the tests can show that a replaced syft stops a run here.
    check_syft "$syft_bin" "$SYFT_SHA256" "$SYFT_VERSION" "$TMPDIR/syft-preflight" >/dev/null || exit 1
  fi

  here=$(pwd -P)
  [[ $here/ != /tmp/* ]] || die "refusing to run from under /tmp: $here"
  top=$(git rev-parse --show-toplevel 2>/dev/null) || die "not inside a git working tree: $here"
  [[ $(cd "$top" && pwd -P) == "$here" ]] || die "run from the repository root ($top), not $here"
  repo_root=$here
  dirty=$(git status --porcelain=v1 --untracked-files=all --ignore-submodules=none) \
    || die "git status failed"
  [[ -z $dirty ]] || die "working tree is not clean (tracked changes or untracked files):"$'\n'"$dirty"
  check_index_flags
  note "ok: clean tree at $repo_root (no assume-unchanged or skip-worktree entries)"
  check_git_config
  note "ok: git reads only the repository's config, which rewrites no URL, names no remote after $remote or another URL, and changes no TLS setting"

  head=$(git rev-parse --verify 'HEAD^{commit}')
  [[ $head == "$sha" ]] || die "HEAD is $head, not the given merge commit $sha"
  git "${GH[@]}" fetch --no-tags --quiet "$remote" '+refs/heads/main:refs/remotes/origin/main' \
    || die "fetch of main from $remote failed"
  origin_main=$(git rev-parse --verify 'refs/remotes/origin/main^{commit}')
  [[ $origin_main == "$sha" ]] \
    || die "origin/main is $origin_main after a fresh fetch, not $sha (release the current main merge commit only)"
  note "ok: HEAD == origin/main == $sha"

  if git rev-parse -q --verify "refs/tags/$tag" >/dev/null; then
    die "tag $tag already exists locally ($(git rev-parse "refs/tags/$tag"))"
  fi
  remote_tags=$(git "${GH[@]}" ls-remote --tags "$remote" "refs/tags/$tag") \
    || die "ls-remote of $remote failed; cannot prove that $tag is new"
  [[ -z $remote_tags ]] || die "tag $tag already exists on $remote: $remote_tags"
  note "ok: tag $tag is new locally and on origin"

  channel=$(git cat-file blob "$sha:rust-toolchain.toml" 2>/dev/null \
    | sed -n 's/^[[:space:]]*channel[[:space:]]*=[[:space:]]*"\([^"]*\)".*/\1/p') || true
  [[ $channel == "$BUILDER_TOOLCHAIN" ]] \
    || die "rust-toolchain.toml channel '$channel' != builder toolchain $BUILDER_TOOLCHAIN; re-pin BUILDER_IMAGE in release.sh through a reviewed change"

  metadata_json=$("$cargo_bin" metadata --no-deps --format-version 1 --locked) \
    || die "cargo metadata --no-deps --format-version 1 --locked failed (is Cargo.lock current?)"
  # A pipe, not a here-string: the metadata is about 150 KB, and bash would put a here-string that
  # large in a temporary file.
  verdict=$(printf '%s\n' "$metadata_json" | jq -r --arg want "$version" "$VERSION_JQ") \
    || die "could not read the cargo metadata"
  [[ $verdict == OK ]] || die "workspace version does not match $tag:"$'\n'"$verdict"
  note "ok: jope-cli and the product crates are at $version"

  [[ $(git cat-file -t "$sha:$PUBKEY_PATH" 2>/dev/null) == blob ]] \
    || die "$PUBKEY_PATH is not committed at $sha"
  pub_sha=$(blob_sha256 "$sha" "$PUBKEY_PATH")
  [[ $pub_sha == "$PUBKEY_SHA256" ]] \
    || die "$PUBKEY_PATH has sha256 $pub_sha, not the custody fingerprint $PUBKEY_SHA256"
  [[ $(git cat-file -t "$sha:CHANGELOG.md" 2>/dev/null) == blob ]] \
    || die "CHANGELOG.md is not committed at $sha (generate it with git-cliff in the release PR)"
  notes_text=$(git cat-file blob "$sha:CHANGELOG.md" | extract_notes "$version") \
    || die "CHANGELOG.md has no '## [$version]' section with entries"
  note "ok: $PUBKEY_PATH fingerprint and CHANGELOG.md section [$version]"

  key_exists signing.key || die "signing key $key_dir/signing.key is missing (or sudo -n is unavailable)"
  key_exists signing.password || die "$key_dir/signing.password is missing (or sudo -n is unavailable)"
  cmp -s <(key_cat signing.pub) <(git cat-file blob "$sha:$PUBKEY_PATH") \
    || die "$PUBKEY_PATH does not match $key_dir/signing.pub"
  note "ok: signing key present in $key_dir and its public key matches $PUBKEY_PATH"

  if (( publish )); then check_publish_ready; fi
  check_run_dir_new
}

check_publish_ready() {
  local ident email out
  ident=$(git var GIT_COMMITTER_IDENT) || die "git has no committer identity for the tagger"
  email=${ident#*<}
  email=${email%%>*}
  [[ $email == "$TAGGER_EMAIL" ]] \
    || die "the tagger would be <$email>; the release tag must be made by the writer identity <$TAGGER_EMAIL> (set GIT_COMMITTER_NAME and GIT_COMMITTER_EMAIL, or user.name and user.email in the repository's own config; release.sh does not read ~/.gitconfig)"
  if out=$("$GH_ROLE" writer release view "$tag" -R "$REPO_SLUG" --json tagName 2>&1); then
    die "a GitHub release $tag already exists: $out"
  fi
  grep -qi 'release not found' <<<"$out" || die "cannot prove that no GitHub release $tag exists: $out"
  note "ok: tagger <$email>; no GitHub release $tag yet"
}

check_run_dir_new() {
  stamp=$(run_stamp)
  release_root=$out_root/release-$tag
  target_root=$release_root/target
  run_dir=$release_root/$mode-$stamp
  [[ ! -L $release_root ]] || die "$release_root is a symlink"
  [[ ! -e $release_root || -d $release_root ]] || die "$release_root is not a directory"
  [[ ! -L $target_root ]] || die "$target_root is a symlink"
  [[ ! -L $run_dir ]] || die "$run_dir is a symlink"
  if [[ -e $run_dir ]]; then
    [[ -d $run_dir && -z $(ls -A -- "$run_dir") ]] \
      || die "output directory $run_dir already exists and is not empty; evidence is never overwritten"
  fi
  note "ok: new output directory $run_dir"
}

# --- the run -------------------------------------------------------------------------------------

open_run_dir() {
  stage=setup
  mkdir -p -- "$run_dir" "$target_root/cargo-home"
  dist=$run_dir/dist
  mkdir -- "$dist" "$run_dir/home"
  # TMPDIR is private (0700) like release-tmp/, whatever the umask: it holds the containers' tmp/
  # and the tools' configuration directories.
  mkdir -m 0700 -- "$run_dir/tmp"
  [[ -d $run_dir/tmp && ! -L $run_dir/tmp && -O $run_dir/tmp && $(stat -c %a -- "$run_dir/tmp") == 700 ]] \
    || die "$run_dir/tmp is not a private (0700) directory owned by $(id -un)"
  export TMPDIR=$run_dir/tmp
  checks_file=$run_dir/readback-checks.jsonl
  exec 3>&1 4>&2
  exec > >(tee -a "$run_dir/release.log") 2>&1
  tee_pid=$!
  step "release.sh $tag $sha ($mode)"
  printf 'host=%s user=%s uid=%s repo=%s\n' "$(uname -n)" "$(id -un)" "$(id -u)" "$repo_root"
  printf 'builder=%s (%s) max_glibc=%s\n' "$BUILDER_IMAGE" "$BUILDER_IMAGE_NAME" "$MAX_GLIBC"
  printf 'preflight:\n'
  printf '  %s\n' "${early_log[@]}"
  printf '%s\n' "$metadata_json" > "$run_dir/metadata.json"
  tools_json > "$run_dir/tools.json"
  jq -r '.host[] | "tool \(.name): \(.resolved) \(.package // "?") \(.package_version // "?")"' "$run_dir/tools.json"
}

# Verifies that directory $2 holds exactly the tree of commit $1: the same paths, the same blob ids
# (hashed with no filters), the same modes (100644, 100755, 120000), and nothing else, not even an
# empty directory. $3 is a scratch directory. Submodules are refused, since git archive omits them.
# Run it from inside the repository.
verify_export() (
  commit=$1 dir=$2 work=$3
  [[ -d $dir && ! -L $dir ]] || die "export $dir is not a real directory"
  mkdir -p -- "$work"
  git ls-tree -r -z --full-tree "$commit" > "$work/tree.z" || die "git ls-tree $commit failed"
  : > "$work/expected.unsorted.z"
  while IFS= read -r -d '' rec; do
    meta=${rec%%$'\t'*} path=${rec#*$'\t'}
    read -r entry_mode kind oid <<<"$meta"
    [[ $kind == blob ]] || die "commit $commit has a $kind entry at '$path'; submodules are not exported, so it cannot be released"
    [[ $path != *$'\n'* ]] || die "commit $commit has a path containing a newline"
    printf '%s %s\t%s\0' "$entry_mode" "$oid" "$path" >> "$work/expected.unsorted.z"
  done < "$work/tree.z"
  LC_ALL=C sort -z "$work/expected.unsorted.z" > "$work/expected.z"

  odd=$(cd "$dir" && find . -mindepth 1 ! -type d ! -type f ! -type l -printf '%P\n' | head -n 5)
  [[ -z $odd ]] || die "the export holds special files: $odd"
  odd=$(cd "$dir" && find . -mindepth 1 -type d -empty -printf '%P\n' | head -n 5)
  [[ -z $odd ]] || die "the export holds empty directories that are not in $commit: $odd"
  paths=() modes=()
  while IFS= read -r -d '' rec; do
    perm=${rec%% *} path=${rec#* }
    [[ $path != *$'\n'* ]] || die "the export holds a path containing a newline"
    if (( 8#$perm & 8#100 )); then modes+=(100755); else modes+=(100644); fi
    paths+=("$path")
  done < <(cd "$dir" && find . -mindepth 1 -type f -printf '%m %P\0')
  : > "$work/actual.unsorted.z"
  if (( ${#paths[@]} )); then
    for path in "${paths[@]}"; do printf '%s/%s\n' "$dir" "$path"; done > "$work/paths.txt"
    git hash-object --no-filters --stdin-paths < "$work/paths.txt" > "$work/oids.txt" \
      || die "git hash-object failed on the export"
    mapfile -t oids < "$work/oids.txt"
    (( ${#oids[@]} == ${#paths[@]} )) || die "git hash-object returned ${#oids[@]} ids for ${#paths[@]} files"
    for i in "${!paths[@]}"; do
      printf '%s %s\t%s\0' "${modes[i]}" "${oids[i]}" "${paths[i]}"
    done >> "$work/actual.unsorted.z"
  fi
  while IFS= read -r -d '' path; do
    target=$(readlink -- "$dir/$path")
    oid=$(printf '%s' "$target" | git hash-object --no-filters --stdin)
    printf '120000 %s\t%s\0' "$oid" "$path" >> "$work/actual.unsorted.z"
  done < <(cd "$dir" && find . -mindepth 1 -type l -printf '%P\0')
  LC_ALL=C sort -z "$work/actual.unsorted.z" > "$work/actual.z"

  if ! cmp -s -- "$work/expected.z" "$work/actual.z"; then
    diff <(tr '\0' '\n' < "$work/expected.z") <(tr '\0' '\n' < "$work/actual.z") | head -n 20 >&2 || true
    die "the export in $dir is not exactly the tree of $commit (< commit, > export)"
  fi
  printf '%s\n' "$(tr -cd '\0' < "$work/expected.z" | wc -c)"
)

# Exports $sha into $run_dir/src and verifies it. The build, the receipt and the SBOM read only
# this export. tar -m gives every file the extraction time instead of the commit time; otherwise a
# commit dated before an earlier build of the shared target/ could look older than that build, and
# cargo would reuse stale workspace crates.
export_source() {
  stage='export'
  local src=$run_dir/src entries
  step "source: git archive $sha, verified against its tree"
  mkdir -- "$src"
  git -c core.autocrlf=false -c core.eol=lf archive --format=tar "$sha" | tar -x -m -f - -C "$src" \
    || die "git archive $sha | tar -x failed"
  entries=$(verify_export "$sha" "$src" "$run_dir/tmp/export-check") || exit 1
  src_dir=$src
  src_tree=$(git rev-parse --verify "$sha^{tree}")
  jq -n --arg commit "$sha" --arg tree "$src_tree" --argjson entries "$entries" --arg dir "$src" '
    {schema: "jope.release-source-export/v1", commit: $commit, tree: $tree, entries: $entries,
     directory: $dir, method: "git archive | tar -x -m, then paths, blob ids and modes compared with git ls-tree -r",
     verified: true}' > "$run_dir/source-export.json"
  printf 'export %s: %s entries, tree %s, verified\n' "$src" "$entries" "$src_tree"
}

# The export is compared with the commit again after the build and the SBOM, before anything is
# signed, so a change made to it during the run is caught.
recheck_export() {
  stage='export-recheck'
  local entries
  step "source: re-verify the export after the build and the SBOM"
  entries=$(verify_export "$sha" "$src_dir" "$run_dir/tmp/export-recheck") || exit 1
  printf 'export %s: still exactly %s (%s entries)\n' "$src_dir" "$sha" "$entries"
}

# The shared cargo home may hold only cargo's own caches. Cargo reads config.toml and credentials
# from there, so any other entry would change every later build of this tag and is refused.
check_cargo_home() (
  dir=$1
  shopt -s nullglob dotglob
  [[ -d $dir && ! -L $dir ]] || die "cargo home $dir is not a real directory"
  for path in "$dir"/*; do
    name=${path##*/}
    [[ ! -L $path ]] || die "refusing a symlink in the shared cargo home: $path"
    case $name in
      registry|git) [[ -d $path ]] || die "refusing $path: expected cargo's $name directory" ;;
      .package-cache|.package-cache-mutate|.global-cache|.global-cache-journal|.global-cache-wal|.global-cache-shm)
        [[ -f $path ]] || die "refusing $path: expected a regular file" ;;
      *) die "refusing an unexpected entry in the shared cargo home: $path (cargo reads config and credentials there; inspect it and move it out before re-running)" ;;
    esac
  done
)

# builder_container [docker args...] -- command...
# Every `docker run` of a release goes through here (test-release.sh checks that). It runs the
# pinned builder image as the operator, with the run's tmp/container/ (0700) mounted at
# CONTAINER_TMPDIR and at the container's /tmp, and TMPDIR set to CONTAINER_TMPDIR, so that nothing
# a tool writes as a temporary file lands in the image's /tmp: it stays in this run's directory.
# Bound every build/check/receipt container on shared training hosts. Fixed limits come after
# internal extra arguments so callers cannot accidentally override them. Labels identify only
# this release's containers for the operator's host-pressure monitoring.
builder_container() {
  local ctmp=$run_dir/tmp/container
  local -a extra=()
  while (( $# )) && [[ $1 != -- ]]; do extra+=("$1"); shift; done
  [[ ${1-} == -- ]] || die "builder_container: missing --"
  shift
  [[ -n $run_dir && -d $run_dir/tmp ]] || die "builder_container: there is no run directory"
  mkdir -p -- "$ctmp" || die "cannot create $ctmp"
  chmod 0700 -- "$ctmp" || die "cannot make $ctmp private (0700)"
  docker run --rm --user "$(id -u):$(id -g)" \
    -v "$ctmp:$CONTAINER_TMPDIR" -v "$ctmp:$CONTAINER_SYSTEM_TMP" -e "TMPDIR=$CONTAINER_TMPDIR" \
    "${extra[@]}" --cpus 4 --memory 20g --memory-swap 20g -e CARGO_BUILD_JOBS=4 \
    --label "ai.jopedime.release.tag=$tag" --label "ai.jopedime.release.run=$run_dir" \
    "$BUILDER_IMAGE" "$@"
}

# docker_run [extra docker args...] -- command...
# The pinned builder (builder_container), with the verified export read-only at /src, the shared
# target/ and cargo home, and a HOME that belongs to this run.
docker_run() {
  local -a extra=()
  while (( $# )) && [[ $1 != -- ]]; do extra+=("$1"); shift; done
  [[ ${1-} == -- ]] || die "docker_run: missing --"
  shift
  [[ -n $src_dir && -d $src_dir ]] || die "docker_run: there is no verified source export"
  check_cargo_home "$target_root/cargo-home" || exit 1
  builder_container \
    -v "$src_dir:/src:ro" -v "$target_root:/target" -v "$run_dir/home:/home/release" \
    -e CARGO_TARGET_DIR=/target -e CARGO_HOME=/target/cargo-home -e HOME=/home/release \
    -e RUSTUP_TOOLCHAIN="$BUILDER_TOOLCHAIN" -e CARGO_INCREMENTAL=0 \
    -e CARGO_PROFILE_DEV_DEBUG=line-tables-only -e CARGO_TERM_COLOR=never \
    "${extra[@]}" -w /src -- "$@"
}

build_jope() {
  stage=build
  local bdir=$run_dir/builder exe bin_host
  mkdir -p -- "$bdir"
  step "builder image $BUILDER_IMAGE"
  if ! docker image inspect "$BUILDER_IMAGE" >/dev/null 2>&1; then docker pull -q "$BUILDER_IMAGE"; fi
  docker image inspect --format '{{.Id}} {{json .RepoDigests}}' "$BUILDER_IMAGE" | tee "$bdir/image-inspect.txt"
  docker_run -- rustc -vV > "$bdir/rustc-vV.txt"
  docker_run -- ldd --version > "$bdir/ldd-version.txt"
  head -n1 "$bdir/ldd-version.txt" > "$bdir/builder-libc.txt"
  printf '%s\n' "$BUILDER_TOOLCHAIN" > "$bdir/rustup-toolchain.txt"
  cat "$bdir/rustc-vV.txt" "$bdir/builder-libc.txt"
  grep -qx "release: $BUILDER_TOOLCHAIN" "$bdir/rustc-vV.txt" || die "builder rustc is not $BUILDER_TOOLCHAIN"
  grep -qE "[[:space:]]${MAX_GLIBC//./\\.}\$" "$bdir/builder-libc.txt" || die "builder glibc is not $MAX_GLIBC"
  # The preflight read the versions from the checkout; this reads them from the export that is built.
  docker_run -- cargo metadata --no-deps --format-version 1 --locked > "$bdir/metadata-export.json" \
    || die "cargo metadata failed on the export"
  [[ $(jq -r --arg want "$version" "$VERSION_JQ" "$bdir/metadata-export.json") == OK ]] \
    || die "the export's workspace version does not match $tag (see $bdir/metadata-export.json)"

  step "$BUILD_COMMAND (in $BUILDER_IMAGE_NAME)"
  # shellcheck disable=SC2086  # BUILD_COMMAND is a fixed word list
  docker_run -- $BUILD_COMMAND > "$run_dir/build.jsonl"
  exe=$(select_executable "$run_dir/build.jsonl" /src/crates/jope-cli/Cargo.toml jope) \
    || die "could not select the jope executable from $run_dir/build.jsonl"
  bin_container=$exe
  bin_host=$(map_target_path "$exe")
  [[ -f $bin_host && -x $bin_host && ! -L $bin_host ]] || die "cargo reported $exe, but $bin_host is not an executable file"
  bin=$dist/jope-$tag-$TARGET_TRIPLE
  install -m 0755 -- "$bin_host" "$bin"
  printf 'binary %s <- %s (sha256 %s)\n' "$bin" "$bin_host" "$(sha256_of "$bin")"
}

check_binary() {
  stage=check
  local cdir=$run_dir/checks gpu out
  mkdir -p -- "$cdir/run"
  step "binary checks: glibc <= $MAX_GLIBC, no GPU libraries, prints 'jope $version'"
  objdump -T "$bin" > "$cdir/objdump-T.txt"
  objdump -p "$bin" > "$cdir/objdump-p.txt"
  observed_glibc=$(cat "$cdir/objdump-T.txt" "$cdir/objdump-p.txt" | max_glibc)
  [[ -n $observed_glibc ]] || die "no GLIBC_ symbol versions found in $bin"
  version_le "$observed_glibc" "$MAX_GLIBC" || die "binary needs GLIBC_$observed_glibc, above the $MAX_GLIBC ceiling"
  printf 'max GLIBC needed: %s (ceiling %s)\n' "$observed_glibc" "$MAX_GLIBC"
  needed_libs < "$cdir/objdump-p.txt" > "$cdir/needed.txt"
  [[ -s $cdir/needed.txt ]] || die "no NEEDED entries in $bin; cannot prove it links no GPU library"
  gpu=$(gpu_libs < "$cdir/needed.txt")
  [[ -z $gpu ]] || die "binary links GPU libraries: $gpu"
  printf 'NEEDED: %s\n' "$(tr '\n' ' ' < "$cdir/needed.txt")"
  binary_prints_version "$bin" "$version" "$cdir/run" > "$cdir/version-host.txt" \
    || die "the binary does not print 'jope $version' first (see $cdir/version-host.txt)"
  out=$(builder_container --network none -v "$dist:/dist:ro" -w / -- "/dist/${bin##*/}" </dev/null) \
    || die "the binary did not run inside $BUILDER_IMAGE_NAME"
  printf '%s\n' "$out" > "$cdir/version-bookworm.txt"
  [[ ${out%%$'\n'*} == "jope $version" ]] || die "inside $BUILDER_IMAGE_NAME the binary printed '${out%%$'\n'*}'"
  printf 'runs on the host and inside %s: jope %s\n' "$BUILDER_IMAGE_NAME" "$version"
}

make_receipt() {
  stage=receipt
  local tool name=jope-$tag.build-receipt.json
  step "jope-build-receipt (crates/jope-engine, package jope-parallel)"
  docker_run -- cargo build --locked -p jope-parallel --bin jope-build-receipt \
    --message-format=json-render-diagnostics > "$run_dir/receipt-tool.jsonl"
  tool=$(select_executable "$run_dir/receipt-tool.jsonl" /src/crates/jope-engine/Cargo.toml jope-build-receipt) \
    || die "could not select jope-build-receipt from $run_dir/receipt-tool.jsonl"
  mkdir -p -- "$run_dir/receipt"
  docker_run -v "$run_dir/builder:/builder:ro" -v "$run_dir/receipt:/out" -- \
    "$tool" --binary "$bin_container" --source-dir /src --output "/out/$name" \
    --lineage "jope-cli-$tag" --rustc-vv /builder/rustc-vV.txt \
    --rustup-toolchain /builder/rustup-toolchain.txt --profile release \
    --command "docker run $BUILDER_IMAGE ($(cat "$run_dir/builder/builder-libc.txt")): $BUILD_COMMAND" \
    --builder-image "$BUILDER_IMAGE" --max-glibc "$MAX_GLIBC" --forbid-gpu-libraries \
    || die "jope-build-receipt rejected the binary (see $run_dir/receipt/$name)"
  jq -e --arg bin "$(sha256_of "$bin")" --arg glibc "GLIBC_$observed_glibc" \
      --arg toml "$(blob_sha256 "$sha" Cargo.toml)" --arg lock "$(blob_sha256 "$sha" Cargo.lock)" '
      .bounds.passed == true and .binary_sha256 == $bin and .links_cuda_or_torch == false
      and .glibc_max_needed == $glibc
      and .cargo_toml_sha256 == $toml and .cargo_lock_sha256 == $lock' "$run_dir/receipt/$name" >/dev/null \
    || die "the build receipt does not bind this binary, its glibc/GPU bounds and the commit's Cargo.toml/Cargo.lock: $run_dir/receipt/$name"
  # /src is a git archive export, which has no .git, so the receipt's git identity is null. The
  # export was verified against $sha's tree (source-export.json), and the receipt's Cargo.toml and
  # Cargo.lock hashes are checked against the commit above. A git identity, if one is ever
  # recorded, must still be this commit and clean.
  if jq -e '.git == null' "$run_dir/receipt/$name" >/dev/null; then
    say "the receipt has no git identity, as expected for an export; source-export.json binds /src to $sha (tree $src_tree)"
  else
    jq -e --arg sha "$sha" '.git.commit == $sha and .git.clean == true' "$run_dir/receipt/$name" >/dev/null \
      || die "the build receipt records a different commit or a dirty tree: $run_dir/receipt/$name"
  fi
  install -m 0644 -- "$run_dir/receipt/$name" "$dist/$name"
}

# check_sboms <spdx> <cdx> <version>: refuses SBOMs that are not an SPDX document and a CycloneDX
# document that each name syft <version> as the tool that made them.
check_sboms() {
  jq -e --arg tool "Tool: syft-$3" \
    '(.spdxVersion | startswith("SPDX-")) and any(.creationInfo.creators[]?; . == $tool)' "$1" >/dev/null \
    || die "$1 is not an SPDX document created by syft $3"
  jq -e --arg v "$3" \
    '.bomFormat == "CycloneDX" and any(.metadata.tools.components[]?; .name == "syft" and .version == $v)' \
    "$2" >/dev/null || die "$2 is not a CycloneDX document made by syft $3"
}

make_sbom() {
  stage=sbom
  local work=$run_dir/tmp/syft sbom=$run_dir/sbom
  local spdx=$dist/jope-$tag.spdx.json cdx=$dist/jope-$tag.cdx.json
  step "SBOM: pinned syft $SYFT_VERSION, SPDX and CycloneDX, over the verified export of $sha"
  mkdir -p -- "$work" "$sbom"
  # Only this copy runs, and it is checked (digest, then reported version) right before the scan,
  # so a syft replaced after preflight never runs. on_exit removes the copy if the step fails.
  install -m 0500 -- "$syft_bin" "$work/syft"
  check_syft "$work/syft" "$SYFT_SHA256" "$SYFT_VERSION" "$work" > "$sbom/syft-version.json"
  cat "$sbom/syft-version.json"
  # With --config, syft reads this file and no other: not ~/.syft.yaml, an XDG config, or a
  # .syft.yaml in the export.
  printf 'check-for-app-update: false\n' > "$sbom/syft.yaml"
  # The export holds exactly the committed tree, with no .git and no ignored files, so nothing
  # needs excluding.
  (cd "$src_dir" && syft_env "$work" "$work/syft" scan dir:. --config "$sbom/syft.yaml" \
    --source-name JopeDime --source-version "$tag" -o "spdx-json=$spdx" -o "cyclonedx-json=$cdx") \
    || die "syft scan failed"
  rm -f -- "$work/syft"
  check_sboms "$spdx" "$cdx" "$SYFT_VERSION"
}

make_notes_and_manifest() {
  stage=manifest
  step "release notes (CHANGELOG.md section [$version]) and release manifest"
  printf '%s\n' "$notes_text" > "$dist/RELEASE-NOTES-$tag.md"
  write_manifest "$dist" "$tag" "$sha" "$version" "$observed_glibc"
  cat "$dist/release-manifest-$tag.json"
  stage=sums
  write_sums "$dist" "$tag"
  cat "$dist/SHA256SUMS-$tag.txt"
}

sign_sums() {
  stage=sign
  local sums=$dist/SHA256SUMS-$tag.txt sig=$dist/SHA256SUMS-$tag.txt.sig
  [[ -f $sums && ! -e $sig ]] || die "expected $sums and no $sig"
  step "cosign sign-blob as root with $KEY_DIR/signing.key (tlog upload off; pinned cosign $COSIGN_VERSION)"
  # See SIGN_SCRIPT: root copies cosign into a new directory under /root, checks the copy
  # against COSIGN_SHA256 and runs only the copy, with the password in its environment alone.
  sudo -n "${host_path[sh]:?require_tools has not checked sh}" -c "$SIGN_SCRIPT" sh "$sig" "$sums" "$COSIGN" "$COSIGN_SHA256" \
    || die "cosign sign-blob failed (or the cosign copy did not match the pinned digest)"
  # cosign running as root writes the signature 0600 root:root; hand it back so it can be verified,
  # uploaded and read by anyone.
  sudo -n "${host_path[chown]:?require_tools has not checked chown}" "$(id -u):$(id -g)" -- "$sig" \
    || die "could not chown $sig"
  sudo -n "${host_path[chmod]:?require_tools has not checked chmod}" 0644 -- "$sig" || die "could not chmod $sig"
  [[ -O $sig && -r $sig && -s $sig && ! -L $sig ]] || die "$sig is not a readable file owned by $(id -un)"
}

# Succeeds only when cosign is still the pinned build, exits 0 and says "Verified OK". It runs with
# an empty environment (see check_cosign): HOME, TMPDIR and TUF_ROOT are named, all inside the run
# directory.
cosign_verify() {
  local out rc=0
  [[ $(sha256_of "$COSIGN") == "$COSIGN_SHA256" ]] \
    || { say "$COSIGN is no longer the pinned cosign $COSIGN_VERSION ($COSIGN_SHA256)"; return 1; }
  # A subshell, so that a refusal fails this check instead of exiting the script (see record).
  ( private_dir "$run_dir/tmp/cosign-home" ) || return 1
  out=$(env -i PATH=/usr/bin:/bin HOME="$run_dir/tmp/cosign-home" TMPDIR="$run_dir/tmp" TUF_ROOT="$run_dir/tuf" \
    "$COSIGN" verify-blob --key "$1" --insecure-ignore-tlog=true --signature "$2" "$3" 2>&1) || rc=$?
  printf '%s\n' "$out"
  (( rc == 0 )) && grep -qx 'Verified OK' <<<"$out"
}

# tampered_rejected <pub> <sig> <sums> <copy path>: the tampered-copy negative. It passes only when
# the untouched sums verify, and a copy with one byte appended fails with an invalid signature. A
# missing signature or sums file therefore fails the check instead of passing it vacuously.
tampered_rejected() {
  local out rc=0
  [[ -f $2 && -f $3 ]] || { say "tampered-copy check: $2 or $3 is missing"; return 1; }
  cosign_verify "$1" "$2" "$3" >/dev/null \
    || { say "tampered-copy check: the untouched $3 does not verify, so a rejection would prove nothing"; return 1; }
  cp -- "$3" "$4" || return 1
  printf 'x' >> "$4" || return 1
  out=$(cosign_verify "$1" "$2" "$4") || rc=$?
  printf '%s\n' "$out"
  (( rc != 0 )) || { say "a tampered copy of $3 verified"; return 1; }
  grep -q 'invalid signature' <<<"$out" \
    || { say "the tampered copy failed, but not with an invalid signature"; return 1; }
}

# Succeeds when SHA256SUMS in $1 names exactly the release files there and every digest matches.
sums_check() {
  local dir=$1 file=$2 listed expected
  (cd "$dir" && sha256sum -c --strict -- "$file") || return 1
  listed=$(sed 's/^[0-9a-f]\{64\}  //' "$dir/$file")
  expected=$(dist_names "$dir" artifacts) || return 1
  [[ $listed == "$expected" ]] || { say "$file does not list exactly the release files in $dir"; return 1; }
}

verify_release() {
  stage=verify
  local sums=$dist/SHA256SUMS-$tag.txt sig=$dist/SHA256SUMS-$tag.txt.sig
  local pub=$src_dir/$PUBKEY_PATH tampered=$run_dir/verify/tampered-SHA256SUMS-$tag.txt
  mkdir -p -- "$run_dir/verify"
  step "verify: signature, tampered-copy negative, sha256sum -c, key custody"
  [[ $(sha256_of "$pub") == "$PUBKEY_SHA256" ]] || die "$PUBKEY_PATH fingerprint changed"
  cmp -s <(key_cat signing.pub) "$pub" || die "$PUBKEY_PATH differs from $KEY_DIR/signing.pub"
  cosign_verify "$pub" "$sig" "$sums" || die "cosign verify-blob did not say Verified OK"
  tampered_rejected "$pub" "$sig" "$sums" "$tampered" || die "the tampered-copy negative did not hold"
  say "tampered copy rejected with an invalid signature, as required"
  sums_check "$dist" "SHA256SUMS-$tag.txt" || die "sha256sum -c failed in $dist"
  jq -e --arg sha "$sha" --arg tag "$tag" --arg ver "$version" \
    '.git_sha == $sha and .tag == $tag and .crate_version == $ver' \
    "$dist/release-manifest-$tag.json" >/dev/null || die "the release manifest does not bind $tag, $sha and $version"
}

# record <name> <ok|fail> <command...>: runs the command and appends
# {name, command, rc, expected, ok} to $checks_file. It never exits the script; the verdict is
# left in last_check_ok (true or false).
record() {
  local name=$1 expect=$2 rc=0 ok=false
  shift 2
  "$@" || rc=$?
  if [[ $expect == ok ]]; then (( rc != 0 )) || ok=true; else (( rc == 0 )) || ok=true; fi
  jq -nc --arg name "$name" --arg cmd "$(printf '%q ' "$@")" --argjson rc "$rc" \
    --arg expected "$expect" --argjson ok "$ok" \
    '{name: $name, command: ($cmd | rtrimstr(" ")), rc: $rc, expected: $expected, ok: $ok}' >> "$checks_file"
  last_check_ok=$ok
  say "check $name: rc=$rc expected=$expect -> $([[ $ok == true ]] && echo PASS || echo FAIL)"
}

# skip_check <name> <reason>: records a check that was deliberately not run. It counts as failed.
skip_check() {
  jq -nc --arg name "$1" --arg reason "$2" \
    '{name: $name, command: null, rc: null, expected: "ok", ok: false, skipped: $reason}' >> "$checks_file"
  last_check_ok=false
  say "check $1: NOT RUN ($2) -> FAIL"
}

# The GitHub and remote reads of the readback. They are functions so that test-release.sh can
# stand in for GitHub and origin.
release_view() {
  "$GH_ROLE" writer release view "$tag" -R "$REPO_SLUG" \
    --json tagName,name,isDraft,isPrerelease,url,targetCommitish,createdAt,publishedAt,assets
}
release_download() { "$GH_ROLE" writer release download "$tag" -R "$REPO_SLUG" -D "$1"; }
remote_tag_lines() { git "${GH[@]}" ls-remote "$remote" "refs/tags/$tag" "refs/tags/$tag^{}"; }
# Pushes refs/tags/$tag and nothing else. With push.followTags, git would also push every local
# annotated tag that points into $sha's history and that origin lacks. --no-follow-tags overrides
# any such setting, and push.followTags is forced to false through GIT_CONFIG_COUNT as well.
push_release_tag() { git "${GH[@]}" push --no-follow-tags "$remote" "refs/tags/$tag:refs/tags/$tag"; }

# Succeeds when origin shows exactly refs/tags/$tag at $tag_object and its peel at $sha. It prints
# what origin showed.
remote_tag_matches() {
  local out want
  out=$(remote_tag_lines) || { say "ls-remote of $remote failed"; return 1; }
  printf '%s\n' "$out"
  want=$(printf '%s\t%s\n%s\t%s\n' "$tag_object" "refs/tags/$tag" "$sha" "refs/tags/$tag^{}" | LC_ALL=C sort)
  [[ -n $tag_object && $(LC_ALL=C sort <<<"$out") == "$want" ]] \
    || { say "origin does not show $tag as object $tag_object peeling to $sha"; return 1; }
}

publish_release() {
  local remote_tags out
  local -a assets
  step "publish: re-check freshness immediately before tagging"
  stage='publish-recheck'
  [[ -z $(git status --porcelain=v1 --untracked-files=all) ]] || die "the working tree changed during the run"
  [[ $(git rev-parse --verify 'HEAD^{commit}') == "$sha" ]] || die "HEAD moved during the run"
  git "${GH[@]}" fetch --no-tags --quiet "$remote" '+refs/heads/main:refs/remotes/origin/main' \
    || die "fetch of main from $remote failed"
  [[ $(git rev-parse --verify 'refs/remotes/origin/main^{commit}') == "$sha" ]] \
    || die "origin/main moved during the run; nothing was tagged"
  if git rev-parse -q --verify "refs/tags/$tag" >/dev/null; then die "tag $tag appeared locally during the run"; fi
  remote_tags=$(git "${GH[@]}" ls-remote --tags "$remote" "refs/tags/$tag") || die "ls-remote failed"
  [[ -z $remote_tags ]] || die "tag $tag appeared on origin during the run: $remote_tags"

  step "publish: annotated tag $tag -> $sha"
  stage='tag-create'
  git tag -a "$tag" "$sha" -m "JopeDime $tag"
  stage='tag-created'
  tag_object=$(git rev-parse --verify "refs/tags/$tag")
  [[ $(git cat-file -t "$tag_object") == tag ]] || die "$tag is not an annotated tag object"
  [[ $(git rev-parse --verify "refs/tags/$tag^{commit}") == "$sha" ]] || die "$tag does not peel to $sha"
  git cat-file -p "$tag_object"

  step "publish: push refs/tags/$tag"
  stage='tag-push'
  push_release_tag
  stage='tag-pushed'
  remote_tag_matches || die "after the push, origin does not show $tag as $tag_object -> $sha"

  step "publish: GitHub release $tag"
  stage='release-create'
  out=$(dist_names "$dist" all) || exit 1
  mapfile -t assets <<<"$out"
  assets=("${assets[@]/#/$dist/}")
  out=$("$GH_ROLE" writer release create "$tag" -R "$REPO_SLUG" --verify-tag \
    --title "JopeDime $tag" --notes-file "$dist/RELEASE-NOTES-$tag.md" "${assets[@]}")
  stage='release-created'
  printf '%s\n' "$out"
  release_url=$(tail -n1 <<<"$out")
}

same_file_set() {
  local a b name
  a=$(dist_names "$1" all) || return 1
  b=$(dist_names "$2" all) || return 1
  [[ $a == "$b" ]] || { say "asset names differ: [$a] vs [$b]"; return 1; }
  while IFS= read -r name; do
    cmp -s -- "$1/$name" "$2/$name" || { say "asset $name differs from dist"; return 1; }
  done <<<"$a"
}

readback() {
  stage=readback
  local rb=$run_dir/readback assets pub assets_json remote_object remote_commit oid ref integrity=1
  local sums=SHA256SUMS-$tag.txt bin_name=jope-$tag-$TARGET_TRIPLE
  mkdir -p -- "$rb/assets" "$rb/exec" "$rb/run"
  pub=$src_dir/$PUBKEY_PATH
  assets=$rb/assets
  : > "$checks_file"
  step "readback of the published release into $rb"
  # The tag is read from origin again now that the release exists; the local ref proves nothing
  # about what GitHub serves.
  record remote-tag ok remote_tag_matches > "$rb/ls-remote.txt"
  record release-view ok release_view > "$rb/release.json"
  # shellcheck disable=SC2016  # jq programs; $tag, $sha, $ver are jq variables
  record release-published ok jq -e --arg tag "$tag" '.tagName == $tag and .isDraft == false' "$rb/release.json"
  record release-download ok release_download "$assets"
  # The integrity checks. The downloaded binary is executed only when every one of them passed,
  # which makes it byte-identical to the dist binary that check_binary already ran.
  record asset-set-equals-dist ok same_file_set "$dist" "$assets"
  [[ $last_check_ok == true ]] || integrity=0
  record verify-blob ok cosign_verify "$pub" "$assets/$sums.sig" "$assets/$sums"
  [[ $last_check_ok == true ]] || integrity=0
  record verify-blob-tampered ok tampered_rejected "$pub" "$assets/$sums.sig" "$assets/$sums" "$rb/tampered-$sums"
  record sha256sum-c ok sums_check "$assets" "$sums"
  [[ $last_check_ok == true ]] || integrity=0
  # shellcheck disable=SC2016  # a jq program
  record manifest-binds-tag ok jq -e --arg sha "$sha" --arg tag "$tag" --arg ver "$version" \
    '.git_sha == $sha and .tag == $tag and .crate_version == $ver' "$assets/release-manifest-$tag.json"
  if (( integrity )) && [[ -f $assets/$bin_name && ! -L $assets/$bin_name ]]; then
    install -m 0755 -- "$assets/$bin_name" "$rb/exec/$bin_name"
    record binary-version ok binary_prints_version "$rb/exec/$bin_name" "$version" "$rb/run"
  else
    skip_check binary-version "the downloaded assets failed an integrity check, so the binary was not executed"
  fi

  remote_object='' remote_commit=''
  while IFS=$'\t' read -r oid ref; do
    case $ref in
      "refs/tags/$tag") remote_object=$oid ;;
      "refs/tags/$tag^{}") remote_commit=$oid ;;
    esac
  done < "$rb/ls-remote.txt"
  assets_json=$(digests_json "$assets" all) || assets_json=null
  jq -n --arg tag "$tag" --arg tag_object "$tag_object" --arg sha "$sha" \
    --arg remote_object "$remote_object" --arg remote_commit "$remote_commit" \
    --arg url "$release_url" --arg pub "$(sha256_of "$pub")" --arg repo "$REPO_SLUG" \
    --argjson assets "$assets_json" --slurpfile checks "$checks_file" '
    {schema: "jope.release-readback/v1", repository: $repo, tag: $tag, tag_object: $tag_object,
     tag_commit: $sha,
     origin_tag: {object: (if $remote_object == "" then null else $remote_object end),
                  commit: (if $remote_commit == "" then null else $remote_commit end)},
     release_url: $url, public_key_sha256: $pub, assets: $assets,
     checks: $checks, passed: ($checks | length > 0 and all(.[]; .ok))}' > "$run_dir/readback.json"
  cat "$run_dir/readback.json"
  jq -e '.passed' "$run_dir/readback.json" >/dev/null || die "the readback did not pass; see $run_dir/readback.json"
}

write_run_json() {
  local dist_json
  dist_json=$(digests_json "$dist" all) || dist_json=null
  jq -n --arg mode "$mode" --arg tag "$tag" --arg sha "$sha" --arg ver "$version" \
    --arg image "$BUILDER_IMAGE" --arg max "$MAX_GLIBC" --arg observed "$observed_glibc" \
    --arg url "$release_url" --arg pub "$PUBKEY_SHA256" --arg finished "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    --arg tree "$src_tree" --argjson dist "$dist_json" '
    {schema: "jope.release-run/v1", mode: $mode, tag: $tag, git_sha: $sha, crate_version: $ver,
     source_tree: $tree, builder_image: $image, max_glibc: $max, observed_max_glibc: $observed,
     public_key_sha256: $pub, dist: $dist, published: ($mode == "publish"),
     release_url: (if $url == "" then null else $url end), finished_at: $finished}' > "$run_dir/run.json"
}

# Seals the run directory: SHA256SUMS over every file except itself, the verified source export
# src/ (source-export.json records its commit and tree) and the tmp/ and tuf/ caches. Empty
# directories under tmp/ (the containers' and syft's) are removed first; tmp/ stays only when
# something was left in it.
seal_run_dir() {
  local -a files
  if [[ -d $run_dir/tmp && ! -L $run_dir/tmp ]]; then
    find "$run_dir/tmp" -mindepth 1 -depth -type d -empty -delete 2>/dev/null || true
  fi
  rmdir -- "$run_dir/tmp" 2>/dev/null || true
  mapfile -d '' -t files < <(cd "$run_dir" && find . \( -path ./src -o -path ./tmp -o -path ./tuf \) -prune \
    -o -type f ! -path ./SHA256SUMS -printf '%P\0' | LC_ALL=C sort -z)
  (( ${#files[@]} )) || return 0
  (cd "$run_dir" && sha256sum -- "${files[@]}" > SHA256SUMS)
}

recovery_help() {
  local next_steps
  next_steps=$(cat <<EOF
  $GH_ROLE writer release view $tag -R $REPO_SLUG     # must say 'release not found' before creating it
  $GH_ROLE writer release create $tag -R $REPO_SLUG --verify-tag --title "JopeDime $tag" \\
      --notes-file $dist/RELEASE-NOTES-$tag.md $dist/*
  Then read the release back as docs/release/README.md ("Verify a release") describes, into a new
  directory under $release_root/.
Never delete or move the pushed tag $tag automatically. Withdrawing it is an owner decision: record
it, and cut the next patch version rather than moving $tag.
EOF
)
  case $stage in
    init|preflight|setup) printf 'Nothing was tagged, pushed or published.\n' ;;
    export|build|check|receipt|sbom|export-recheck|manifest|sums|sign|verify|publish-recheck)
      printf 'Nothing was tagged, pushed or published. Fix the cause and re-run; each run gets a new\n'
      printf 'evidence directory, and the build cache in %s is reused.\n' "$target_root" ;;
    tag-create|tag-created)
      cat <<EOF
The annotated tag $tag exists LOCALLY ONLY and was NOT pushed. Nothing was published.
  Inspect:  git show --no-patch $tag
  Preserve the local-only tag; do not delete or move it. Use the next unused patch
  version for a fresh release run, then re-run with --publish.
EOF
      ;;
    tag-push)
      cat <<EOF
The push of $tag failed. It may or may not have reached origin. Check before doing anything else:
  git ls-remote $REPO_URL refs/tags/$tag 'refs/tags/$tag^{}'
If it is ABSENT on origin, preserve the local-only tag; do not delete or move it.
Use the next unused patch version for a fresh release run, then re-run with --publish.
If it is PRESENT (object $tag_object, commit $sha), create the release from this run's dist:
$next_steps
EOF
      ;;
    tag-pushed|release-create)
      cat <<EOF
Tag $tag is on origin (object $tag_object, commit $sha), but this run created no GitHub release.
The signed dist is complete in $dist. Create the release from it:
$next_steps
EOF
      ;;
    release-created|readback)
      cat <<EOF
The GitHub release $tag exists (${release_url:-see gh release view}), but the readback did not pass.
Evidence: $run_dir/readback/ and $run_dir/readback.json. Compare the published assets with dist:
  $GH_ROLE writer release download $tag -R $REPO_SLUG -D $release_root/manual-readback-\$(date -u +%Y%m%dT%H%M%SZ)
Replace a missing or wrong asset only from $dist:
  $GH_ROLE writer release upload $tag -R $REPO_SLUG --clobber $dist/<name>
Do not delete the release or the tag automatically. Withdrawing a published release is an owner decision.
EOF
      ;;
    *) printf 'Stage %s: inspect %s before retrying.\n' "$stage" "${run_dir:-the evidence directory}" ;;
  esac
}

on_exit() {
  local rc=$?
  set +e
  if (( rc != 0 )); then
    printf '\nrelease.sh: FAILED (exit %s) at stage %s\n' "$rc" "$stage" >&2
    recovery_help >&2
  fi
  if [[ -n $tee_pid ]]; then
    exec 1>&3 2>&4
    wait "$tee_pid" 2>/dev/null
    # make_sbom's copy of syft, left behind only when that step failed.
    rm -f -- "$run_dir/tmp/syft/syft"
    seal_run_dir || printf 'release.sh: could not write %s/SHA256SUMS\n' "$run_dir" >&2
    printf 'release.sh: evidence in %s (sealed by its SHA256SUMS)\n' "$run_dir" >&2
  fi
  exit "$rc"
}

main() {
  pin_tmpdir
  TEST_ROOT=$(realpath -m -- "${HOME:?HOME must be set}/.cache/jope-test-tmp")
  readonly TEST_ROOT
  parse_args "$@"
  trap on_exit EXIT
  umask 022
  init_test_mode
  preflight
  if (( test_mode )); then
    say "test mode: all guards passed for $tag at $sha (run dir would be $run_dir); nothing built"
    return 0
  fi
  open_run_dir
  export_source
  build_jope
  check_binary
  make_receipt
  make_sbom
  recheck_export
  make_notes_and_manifest
  sign_sums
  verify_release
  if (( publish )); then
    publish_release
    readback
  fi
  stage='done'
  write_run_json
  if (( publish )); then
    step "PUBLISHED $tag ($release_url); tag object $tag_object -> $sha; readback passed"
  else
    step "DRY RUN passed for $tag at $sha: nothing was tagged, pushed or published"
    say "to publish: deploy/release/release.sh $tag $sha --publish"
  fi
}

# The same test as the first lines, in keywords only (see "bash itself"): should an imported function
# have stood in for exit there, main() still does not run, and the script ends with status 1.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  [[ -o privileged && ${SHELLOPTS@a} != *x* ]] && main "$@"
fi
