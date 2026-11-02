from _runner_case_support import *


class TrimTests(unittest.TestCase):
    """deploy/runner/trim-deps.sh and common.sh jope_runner_trim_lane (bounded build-cache trim)."""

    SCRIPT = ROOT / 'deploy/runner/trim-deps.sh'

    def fixture(self, root):
        """A lane target tree. Returns (expected selected paths, every created path)."""
        created, selected = [], set()

        def make(rel, age):
            path = root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'x' * (len(rel) % 7 + 1))
            stamp = 1_700_000_000 + age
            os.utime(path, (stamp, stamp))
            created.append(path)
            return path

        def h(i):
            return '%016x' % (0xabc000 + i)
        for i in range(9):   # newest 6 rlibs stay; 0, 1, 2 are older duplicates
            p = make(f'ci/debug/deps/libfoo-{h(i)}.rlib', i)
            if i < 3:
                selected.add(p)
        for i in range(3):   # another kind of the same crate counts separately
            make(f'ci/debug/deps/libfoo-{h(i)}.rmeta', i)
        for i in range(8):   # test binaries (no extension)
            p = make(f'ci/debug/deps/foo-{h(i)}', i)
            if i < 2:
                selected.add(p)
        for i in range(7):
            p = make(f'ci/debug/deps/foo-{h(i)}.d', i)
            if i < 1:
                selected.add(p)
        for i in range(7):   # each lane directory counts separately
            p = make(f'ci2/debug/deps/libfoo-{h(i)}.rlib', i)
            if i < 1:
                selected.add(p)
        # Never candidates: other names, other directories, deeper paths, evidence.
        for i in range(9):
            make(f'ci/debug/build/foo-{h(i)}/out/libfoo-{h(i)}.rlib', i)
            make(f'ci/tmp/libfoo-{h(i)}.rlib', i)
            make(f'ci/dist-ci/deps/libfoo-{h(i)}.rlib', i)
            make(f'ci-evidence-retained/debug/deps/libfoo-{h(i)}.rlib', i)
            make(f'ci/debug/deps/README-{i}', i)
            make(f'ci/debug/deps/libbar-{i}23.rlib', i)
            make(f'ci/debug/deps/libbaz-{h(i).upper()}.rlib', i)
        return selected, created

    def run_trim(self, root, *args, awk=None):
        env = dict(os.environ)
        if awk:
            shim = root.parent / 'awk-shim'
            shim.mkdir(exist_ok=True)
            link = shim / 'awk'
            if not link.exists():
                link.symlink_to(awk)
            env['PATH'] = f'{shim}:{env["PATH"]}'
        return subprocess.run(['bash', str(self.SCRIPT), str(root), *args], capture_output=True, text=True, env=env)

    def scratch(self):
        base = ROOT / 'target' / 'tmp'
        base.mkdir(parents=True, exist_ok=True)
        return tempfile.TemporaryDirectory(prefix='runner-trim-fixture-', dir=base)

    def test_keeps_newest_builds_per_crate_and_kind_and_nothing_else(self):
        awks = [None] + [a for a in (shutil.which('mawk'), shutil.which('gawk')) if a]
        for awk in awks:
            with self.subTest(awk=awk), self.scratch() as name:
                root = Path(name) / 'v'
                selected, created = self.fixture(root)
                dry = self.run_trim(root, '6', awk=awk)
                self.assertEqual(0, dry.returncode, dry.stderr)
                rows = [line.split('\t') for line in dry.stdout.splitlines()]
                self.assertEqual({str(p) for p in selected}, {row[0] for row in rows})
                for row in rows:
                    self.assertEqual(3, len(row))
                    self.assertEqual(Path(row[0]).stat().st_size, int(row[1]))
                self.assertTrue(all(p.exists() for p in created), 'a dry run deleted files')
                applied = self.run_trim(root, '6', '--apply', awk=awk)
                self.assertEqual(0, applied.returncode, applied.stderr)
                self.assertEqual(dry.stdout, applied.stdout)
                for path in created:
                    self.assertEqual(path not in selected, path.exists(), path)
                again = self.run_trim(root, '6', '--apply', awk=awk)
                self.assertEqual((0, ''), (again.returncode, again.stdout))

    def test_refuses_bad_arguments(self):
        with self.scratch() as name:
            root = Path(name) / 'v'
            root.mkdir()
            for args in [('0',), ('-1',), ('six',), ('6', '--delete'), ()]:
                with self.subTest(args=args):
                    self.assertNotEqual(0, self.run_trim(root, *args).returncode)
            missing = self.run_trim(root / 'absent', '6', '--apply')
            self.assertEqual(2, missing.returncode)

    @unittest.skipIf(os.geteuid() == 0, 'root reads unreadable directories')
    def test_listing_error_deletes_nothing(self):
        with self.scratch() as name:
            root = Path(name) / 'v'
            selected, created = self.fixture(root)
            locked = root / 'ci3'
            (locked / 'debug').mkdir(parents=True)
            locked.chmod(0)
            try:
                result = self.run_trim(root, '6', '--apply')
            finally:
                locked.chmod(0o755)
            self.assertNotEqual(0, result.returncode)
            self.assertEqual('', result.stdout)
            self.assertTrue(all(p.exists() for p in created))

    def trim_lane(self, state, docker_body, lane='ci'):
        script = ('source deploy/runner/common.sh; JOPE_RUNNER_STATE=$1; JOPE_RUNNER_DOMAIN=pr; '
                  'calls=$1/calls; docker() { printf "%s\\n" "$*" >> "$calls"; ' + docker_body + ' }; '
                  'jope_runner_trim_lane "$2"')
        return subprocess.run(['bash', '-c', script, 'test', str(state), lane], cwd=ROOT, capture_output=True, text=True)

    def test_lane_trim_is_isolated_rate_limited_and_never_fatal(self):
        row = '/v/ci/debug/deps/libfoo-0123456789abcdef.rlib\\t10\\t1.5\\n'
        ok = f'if [[ $1 == run ]]; then printf "{row}"; fi; return 0;'
        with self.scratch() as name:
            state = Path(name)
            first = self.trim_lane(state, ok)
            self.assertEqual(0, first.returncode, first.stderr)
            calls = (state / 'calls').read_text().splitlines()
            run = [c for c in calls if c.startswith('run ')]
            self.assertEqual(1, len(run), calls)
            args = run[0].split()
            for flag in ('--network none', '--cap-drop ALL', '--read-only', '--user 1001'):
                self.assertIn(flag, run[0])
            mounts = [args[i + 1] for i, a in enumerate(args) if a == '-v']
            self.assertEqual(['jope-pr-ci-target:/v', '/opt/jope-runner/trim-deps.sh:/opt/jope-runner/trim-deps.sh:ro'], mounts)
            self.assertEqual(['/v', '3', '--apply'], args[-3:])
            manifests = sorted((state / 'trim').glob('*-ci.tsv'))
            self.assertEqual(1, len(manifests))
            self.assertIn('libfoo-0123456789abcdef.rlib\t10\t1.5', manifests[0].read_text())
            self.assertIn('removed 1 files, 10 bytes from jope-pr-ci-target', first.stdout)
            self.assertTrue((state / 'trim' / 'ci.last').exists())
            # Within the interval nothing runs again.
            (state / 'calls').unlink()
            second = self.trim_lane(state, ok)
            self.assertEqual(0, second.returncode, second.stderr)
            self.assertFalse((state / 'calls').exists())
            # After the interval it runs; a failing trim is reported, keeps the old stamp, never fails.
            stamp = state / 'trim' / 'ci.last'
            os.utime(stamp, (1, 1))
            failed = self.trim_lane(state, 'if [[ $1 == run ]]; then return 9; fi; return 0;')
            self.assertEqual(0, failed.returncode)
            self.assertIn('trim ci: failed', failed.stderr)
            self.assertEqual(1, int(stamp.stat().st_mtime))
            # A lane without a target volume is skipped.
            missing = self.trim_lane(state, 'if [[ $1 == volume ]]; then return 1; fi; return 0;', lane='ci2')
            self.assertEqual(0, missing.returncode)
            self.assertIn('no volume jope-pr-ci2-target', missing.stdout)

    def test_launcher_trims_only_while_holding_the_idle_lane(self):
        text = (ROOT / 'deploy/runner/run.sh').read_text()
        order = [text.index(marker) for marker in (
            'flock -n 9', 'jope_runner_check_existing "$lane"', 'jope_runner_trim_lane "$lane" || true',
            '/disk-guard.sh"', 'generate-jitconfig"', 'exec docker')]
        self.assertEqual(sorted(order), order)
        common = (ROOT / 'deploy/runner/common.sh').read_text()
        self.assertIn('/opt/jope-runner/trim-deps.sh\n', common.split('JOPE_RUNNER_DEPLOYED=(')[1].split(')')[0])
