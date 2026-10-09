"""Run ONLY this package's exact new offline fixture roster, once per receipt."""
import sys
sys.dont_write_bytecode = True
import argparse
import base64
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import platform
import time
import unittest

EXPECTED = ('tests_custody.CustodyTests.test_event_bound_failure_retains_prior_hash_chain', 'tests_custody.CustodyTests.test_existing_output_cannot_be_overwritten_or_resumed', 'tests_custody.CustodyTests.test_get_attempt_is_fsynced_before_transport_failure', 'tests_custody.CustodyTests.test_hard_exit_retains_attempt_without_invented_completion', 'tests_custody.CustodyTests.test_native_attempt_and_timing_are_durable_on_nonzero_exit', 'tests_custody.CustodyTests.test_partial_read_counts_survive_known_failure', 'tests_custody.CustodyTests.test_real_configure_failure_has_durable_attempt_and_final_receipt', 'tests_semantics.SemanticTests.test_configuration_exact_source_normalizer_and_clock_are_fixed', 'tests_semantics.SemanticTests.test_production_config_wrong_canonicalizer_refused_without_execution')

def sha(raw):return hashlib.sha256(raw).hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--receipt',required=True);args=ap.parse_args()
    out=Path(args.receipt);assert not out.exists(),'create-only qualification receipt'
    root=Path(__file__).resolve().parent;pins=json.loads((root/'SOURCE_PINS.json').read_bytes())
    def closure():
        for rel,want in pins.items():
            path=root/rel;assert path.is_file() and not path.is_symlink() and sha(path.read_bytes())==want,'pinned source drift: '+rel
        assert {str(p.relative_to(root))for p in root.rglob('*.py')}=={x for x in pins if x.endswith('.py')},'exact Python closure'
    closure();assert tuple(json.loads((root/'FIXTURES.json').read_bytes()))==EXPECTED,'frozen fixture roster changed'
    name='_generated_target_callbacks_qualification'
    assert not any(k==name or k.startswith(name+'.')for k in sys.modules),'isolated namespace already loaded'
    spec=importlib.util.spec_from_file_location(name,root/'__init__.py',submodule_search_locations=[str(root)])
    package=importlib.util.module_from_spec(spec);sys.modules[name]=package;spec.loader.exec_module(package)
    suite=unittest.TestSuite()
    for identity in EXPECTED:
        suite.addTests(unittest.defaultTestLoader.loadTestsFromName(name+'.'+identity))
    def flatten(item):
        if isinstance(item,unittest.TestSuite):
            for child in item:yield from flatten(child)
        else:yield item
    ids=tuple(sorted(t.id()[len(name)+1:]for t in flatten(suite)))
    assert ids==EXPECTED and len(ids)==len(set(ids)),'exact unique collected roster/count'
    started=time.time_ns();mono=time.monotonic_ns();stream=io.StringIO()
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    closure();origins={}
    for key,module in list(sys.modules.items()):
        if key==name or key.startswith(name+'.'):
            path=Path(module.__file__).resolve();assert path.is_relative_to(root),'foreign package import'
            rel=str(path.relative_to(root));assert rel in pins and sha(path.read_bytes())==pins[rel],'import source pin'
            origins[key]=rel
    passed=(result.testsRun==len(EXPECTED) and not any((result.failures,result.errors,result.skipped,result.expectedFailures,result.unexpectedSuccesses)))
    text=stream.getvalue().encode()
    receipt={'schema':'jope.generated-target-callbacks-offline-qualification/v1','passed':passed,'tests_run':result.testsRun,'new_custody_fixtures':7,'changed_configuration_fixtures':2,'prior_unaffected_fixtures_rerun':0,
       'fixture_ids':list(EXPECTED),'fixture_roster_sha256':sha((root/'FIXTURES.json').read_bytes()),'source_pins':pins,
       'source_manifest_sha256':sha((root/'SOURCE_PINS.json').read_bytes()),'loaded_package_origins':origins,
       'started_utc_ns':started,'finished_utc_ns':time.time_ns(),'elapsed_monotonic_ns':time.monotonic_ns()-mono,
       'python':platform.python_version(),'test_output':{'bytes':len(text),'sha256':sha(text),'base64':base64.b64encode(text).decode()},
       'live_runtime_qualified':False,'native_serde_json_qualified':False,'full128_replayed':False,
       's3_requests':0,'gp_executions':0,'database_writes':0,'optimizer_admission':False}
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('xb')as f:f.write((json.dumps(receipt,sort_keys=True,indent=2)+'\n').encode())
    print(json.dumps({'passed':passed,'tests':result.testsRun,'receipt':str(out),'receipt_sha256':sha(out.read_bytes())}))
    return 0 if passed else 1

if __name__=='__main__':raise SystemExit(main())
