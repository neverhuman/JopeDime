#!/usr/bin/env python3
"""Offline tests of deploy/ci/neural-env.sh against fixture SDK and CUDA trees.

Discovery only: nothing here compiles runtime.cpp, links, loads CUDA or needs a GPU. The fixture
libraries are text files whose SONAME a fake `readelf` (first on PATH) reports, the pattern
test-full-main.py uses for `tar`. The toolkit-layout cases preprocess the fixture headers with the
real C++ compiler the runner image pins (clang++ 18); distro-layout and shadowing cases use a fake
compiler that emulates `-E -H` with an explicit system include directory.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / 'deploy/ci/neural-env.sh'
RECEIPT = json.loads((ROOT / 'deploy/receipts/libtorch.json').read_text())
EXPECTED_LIBTORCH = RECEIPT['libtorch']['version']
EXPECTED_CUDART = RECEIPT['cuda']['cudart_version_macro']
EXPECTED_NCCL = RECEIPT['nccl']['version']
NAMES = ['LIBTORCH', 'LIBTORCH_CXX11_ABI', 'CUDA_HOME', 'JOPE_NCCL_INCLUDE', 'JOPE_CUDA_LIBRARY_PATH',
         'JOPE_NCCL_SONAME', 'JOPE_CUDART_SONAME', 'CXX']
COMPANIONS = ['nccl/lib/libnccl.so.2', 'cudnn/lib/libcudnn.so.9', 'cusparselt/lib/libcusparseLt.so.0',
              'cufile/lib/libcufile.so.0', 'nvshmem/lib/libnvshmem_host.so.3']

FAKE_READELF = '''#!/usr/bin/env python3
import sys
text = open(sys.argv[-1]).read()
for line in text.splitlines():
    if line.startswith('SONAME='):
        print(' 0x000000000000000e (SONAME)             Library soname: [%s]' % line[7:])
'''

# Emulates `CXX -x c++ -std=c++20 -E -H -o /dev/null -I... -`: -I directories first, then
# FAKE_CXX_SYSTEM_INCLUDE, printing '. <path>' to stderr for each top-level include it resolves.
FAKE_CXX = '''#!/usr/bin/env python3
import os, sys
args = sys.argv[1:]
if args == ['--version']:
    print('fake clang version 18.1.3')
    sys.exit(0)
dirs = [a[2:] for a in args if a.startswith('-I')] + [os.environ.get('FAKE_CXX_SYSTEM_INCLUDE', '/nonexistent')]
for line in sys.stdin.read().splitlines():
    name = line.split('<')[1].split('>')[0]
    forced = os.environ.get('FAKE_CXX_FORCE_' + name.split('.')[0].upper())
    found = forced or next((os.path.join(d, name) for d in dirs if os.path.isfile(os.path.join(d, name))), None)
    if not found:
        sys.stderr.write('<stdin>:1:10: fatal error: %s: No such file or directory\\n' % name)
        sys.exit(1)
    sys.stderr.write('. %s\\n' % found)
'''


def write(path, text=''):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def library(path):
    return write(path, 'SONAME=%s\n' % path.name)


class Fixture:
    def __init__(self, root):
        self.root = root
        self.sdk = root / 'sdk'
        self.bin = root / 'bin'
        self.receipt = root / 'evidence' / 'neural-discovery.json'
        self.receipt.parent.mkdir(parents=True)
        write(self.sdk / 'libtorch/build-version', EXPECTED_LIBTORCH + '\n')
        for name in ['libtorch.so', 'libtorch_cpu.so', 'libtorch_cuda.so', 'libc10.so', 'libc10_cuda.so']:
            write(self.sdk / 'libtorch/lib' / name)
        write(self.sdk / 'libtorch/include/c10/cuda/CUDAStream.h')
        write(self.sdk / 'libtorch/include/torch/csrc/api/include/torch/torch.h')
        for name in COMPANIONS:
            library(self.sdk / 'nvidia' / name)
        self.nccl_header(EXPECTED_NCCL)
        fake = write(self.bin / 'readelf', FAKE_READELF)
        fake.chmod(0o755)
        self.fake_cxx = write(self.bin / 'fake-cxx', FAKE_CXX)
        self.fake_cxx.chmod(0o755)

    def nccl_header(self, version):
        major, minor, patch = version.split('.')
        write(self.sdk / 'nvidia/nccl/include/nccl.h',
              '#define NCCL_MAJOR %s\n#define NCCL_MINOR %s\n#define NCCL_PATCH %s\n' % (major, minor, patch))

    def runtime_headers(self, include, version=EXPECTED_CUDART):
        write(include / 'cuda_runtime.h', '#define JOPE_FIXTURE_CUDA_RUNTIME 1\n')
        write(include / 'cuda_runtime_api.h', '#define CUDART_VERSION  %s\n' % version)

    def toolkit(self, version=EXPECTED_CUDART):
        home = self.root / 'cuda-12.8'
        self.runtime_headers(home / 'include', version)
        library(home / 'targets/x86_64-linux/lib/libcudart.so.12')
        (home / 'lib64').symlink_to('targets/x86_64-linux/lib')
        return home

    def distro(self):
        prefix = self.root / 'usr'
        home = prefix / 'lib/cuda'
        write(prefix / 'include/cuda.h')
        self.runtime_headers(prefix / 'include')
        (home / 'include').mkdir(parents=True)
        (home / 'include/cuda.h').symlink_to('../../../include/cuda.h')
        library(prefix / 'lib/x86_64-linux-gnu/libcudart.so.12')
        return home, prefix

    def run(self, cuda_home, cxx, mode='execute', **extra):
        env = {k: v for k, v in os.environ.items() if k not in NAMES and not k.startswith('JOPE_')}
        env.update(PATH=str(self.bin) + os.pathsep + os.environ['PATH'], JOPE_NEURAL_SDK_ROOT=str(self.sdk),
                   CUDA_HOME=str(cuda_home), CXX=str(cxx), JOPE_NEURAL_ENV_RECEIPT=str(self.receipt), **extra)
        if mode == 'execute':
            command = ['bash', str(SCRIPT)]
        else:  # full-lane.sh's pattern: sourced under errexit, a failure maps to exit 78
            command = ['bash', '-c', 'set -euo pipefail; source "$1" || exit 78; for n in %s; do printf "%%s=%%s\\n" "$n" "${!n}"; done'
                       % ' '.join(NAMES), 'full-lane.sh', str(SCRIPT)]
        return subprocess.run(command, env=env, capture_output=True, text=True, cwd=ROOT)

    def document(self):
        return json.loads(self.receipt.read_text())


def real_compiler():
    found = shutil.which('clang++') or shutil.which('g++')
    if not found:
        raise AssertionError('the runner image pins clang++ 18; no C++ compiler is on PATH')
    return found


class NeuralEnvTests(unittest.TestCase):
    def setUp(self):
        scratch = ROOT / 'target' / 'tmp'
        scratch.mkdir(parents=True, exist_ok=True)
        self.scratch = tempfile.TemporaryDirectory(prefix='neural-env ', dir=scratch)
        self.fixture = Fixture(Path(self.scratch.name))

    def tearDown(self):
        self.scratch.cleanup()

    def assertUnavailable(self, result, check):
        self.assertEqual(1, result.returncode, result.stderr)
        self.assertEqual('', result.stdout)
        self.assertIn('neural-env: ', result.stderr)
        document = self.fixture.document()
        self.assertEqual('unavailable', document['status'])
        self.assertEqual(check, document['failed_check'])
        self.assertEqual('failed', document['qualification']['discovery'])
        self.assertIsNone(document['environment'])

    def test_toolkit_layout_is_discovered_with_the_real_compiler(self):
        home = self.fixture.toolkit()
        result = self.fixture.run(home, real_compiler())
        self.assertEqual(0, result.returncode, result.stderr)
        exported = dict(line.split('=', 1) for line in result.stdout.splitlines())
        self.assertEqual(NAMES, list(exported))
        sdk = self.fixture.sdk
        self.assertEqual(str(home), exported['CUDA_HOME'])
        self.assertEqual(':'.join([str(sdk / 'nvidia' / p / 'lib') for p in ('nccl', 'cudnn', 'cusparselt', 'cufile', 'nvshmem')]
                                  + [str(home / 'lib64')]), exported['JOPE_CUDA_LIBRARY_PATH'])
        document = self.fixture.document()
        self.assertEqual('discovered', document['status'])
        self.assertEqual('toolkit', document['observed']['cuda']['layout'])
        self.assertEqual(str(EXPECTED_CUDART), document['observed']['cuda']['cudart_version'])
        self.assertEqual(str((home / 'include/cuda_runtime.h').resolve()), document['observed']['compiler']['resolved_cuda_runtime_h'])
        self.assertEqual(str(sdk / 'nvidia/nccl/include/nccl.h'), document['observed']['compiler']['resolved_nccl_h'])
        self.assertEqual(exported, document['environment'])
        for claim in ('native_build', 'native_link', 'native_runtime', 'gpu'):
            self.assertEqual('not-attempted', document['qualification'][claim])

    def test_sourced_mode_exports_the_same_environment(self):
        home = self.fixture.toolkit()
        result = self.fixture.run(home, real_compiler(), mode='source')
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertEqual(self.fixture.document()['environment'],
                         dict(line.split('=', 1) for line in result.stdout.splitlines()))

    def test_sourced_failure_writes_receipt_and_maps_to_unavailable(self):
        home = self.fixture.toolkit()
        (home / 'include/cuda_runtime.h').unlink()
        result = self.fixture.run(home, real_compiler(), mode='source')
        self.assertEqual(78, result.returncode, result.stderr)
        self.assertEqual('cuda-runtime-header', self.fixture.document()['failed_check'])

    def test_distro_split_layout_resolves_through_the_cuda_h_symlink(self):
        home, prefix = self.fixture.distro()
        result = self.fixture.run(home, self.fixture.fake_cxx, FAKE_CXX_SYSTEM_INCLUDE=str(prefix / 'include'))
        self.assertEqual(0, result.returncode, result.stderr)
        exported = dict(line.split('=', 1) for line in result.stdout.splitlines())
        self.assertTrue(exported['JOPE_CUDA_LIBRARY_PATH'].endswith(':' + str(prefix / 'lib/x86_64-linux-gnu')))
        observed = self.fixture.document()['observed']['cuda']
        self.assertEqual('distro-split', observed['layout'])
        self.assertEqual(str(prefix / 'include'), observed['include_dir'])

    def test_distro_layout_the_compiler_cannot_see_is_rejected(self):
        home, _ = self.fixture.distro()
        result = self.fixture.run(home, self.fixture.fake_cxx)  # no system include directory
        self.assertUnavailable(result, 'compiler-header-probe')

    def test_missing_runtime_header_fails_closed(self):
        home = self.fixture.toolkit()
        (home / 'include/cuda_runtime.h').unlink()
        self.assertUnavailable(self.fixture.run(home, real_compiler()), 'cuda-runtime-header')

    def test_cuda_runtime_version_must_match_the_receipt(self):
        home = self.fixture.toolkit(version=12090)
        self.assertUnavailable(self.fixture.run(home, real_compiler()), 'cudart-version')

    def test_cudart_must_exist_with_its_soname(self):
        home = self.fixture.toolkit()
        library_path = home / 'lib64/libcudart.so.12'
        write(library_path, 'SONAME=libcudart.so.13\n')
        self.assertUnavailable(self.fixture.run(home, real_compiler()), 'cudart-library')
        self.fixture.receipt.unlink()
        library_path.unlink()
        self.assertUnavailable(self.fixture.run(home, real_compiler()), 'cudart-library')

    def test_host_nccl_header_version_is_rejected(self):
        self.fixture.nccl_header('2.26.2')
        self.assertUnavailable(self.fixture.run(self.fixture.toolkit(), real_compiler()), 'nccl-header')

    def test_libtorch_build_version_must_match_the_receipt(self):
        write(self.fixture.sdk / 'libtorch/build-version', '2.13.0+cu130\n')
        self.assertUnavailable(self.fixture.run(self.fixture.toolkit(), real_compiler()), 'libtorch-build-version')

    def test_every_companion_library_is_required(self):
        home = self.fixture.toolkit()
        for name in COMPANIONS:
            with self.subTest(name=name):
                path = self.fixture.sdk / 'nvidia' / name
                path.unlink()
                self.assertUnavailable(self.fixture.run(home, real_compiler()), 'companion-libraries')
                library(path)
                self.fixture.receipt.unlink()

    def test_missing_compiler_fails_closed(self):
        home = self.fixture.toolkit()
        self.assertUnavailable(self.fixture.run(home, self.fixture.root / 'no-such-compiler'), 'compiler')

    def test_compiler_resolving_a_shadowing_header_is_rejected(self):
        home = self.fixture.toolkit()
        shadow = write(self.fixture.root / 'usr/include/nccl.h', '#define NCCL_MAJOR 2\n')
        result = self.fixture.run(home, self.fixture.fake_cxx, FAKE_CXX_FORCE_NCCL=str(shadow))
        self.assertUnavailable(result, 'compiler-header-probe')

    def test_existing_receipt_is_never_overwritten(self):
        write(self.fixture.receipt, 'retained\n')
        result = self.fixture.run(self.fixture.toolkit(), real_compiler())
        self.assertNotEqual(0, result.returncode)
        self.assertEqual('retained\n', self.fixture.receipt.read_text())


if __name__ == '__main__':
    unittest.main()
