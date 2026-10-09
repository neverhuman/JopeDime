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

EXPECTED = ('tests_contracts.AuthorityContractTests.test_actual_annotated_ref_projection_preserves_original', 'tests_contracts.AuthorityContractTests.test_copied_authority_root_is_not_a_live_role_proof', 'tests_contracts.AuthorityContractTests.test_current_physical_full_role_closure', 'tests_contracts.AuthorityContractTests.test_newer_authority_pointer_refuses_frozen_copy', 'tests_contracts.AuthorityContractTests.test_removed_or_changed_layer_fails', 'tests_contracts.AuthorityContractTests.test_target_role_join_cannot_echo_wrong_sha', 'tests_contracts.AuthorityContractTests.test_unknown_custody_annotation_and_version_are_refused', 'tests_io.ReaderIOTests.test_all_three_allowed_prefixes_use_get_only', 'tests_io.ReaderIOTests.test_consumed_byte_budget_is_checked_before_second_request', 'tests_io.ReaderIOTests.test_deadline_crossed_by_final_eof_cannot_return_success', 'tests_io.ReaderIOTests.test_declared_object_and_total_byte_caps_refuse_before_get', 'tests_io.ReaderIOTests.test_exact_streaming_read_counts_actual_bytes_and_closes', 'tests_io.ReaderIOTests.test_expired_deadline_refuses_before_get', 'tests_io.ReaderIOTests.test_get_attempt_limit_is_checked_before_second_request', 'tests_io.ReaderIOTests.test_hash_truncation_and_stream_overflow_fail_with_actual_byte_counts', 'tests_io.ReaderIOTests.test_identical_version_reads_are_not_cached', 'tests_io.ReaderIOTests.test_invalid_closed_references_and_namespaces_refuse_before_get', 'tests_io.ReaderIOTests.test_stream_error_closes_and_retains_already_read_byte_count', 'tests_io.ReaderIOTests.test_transport_error_charges_one_attempt_and_never_retries', 'tests_io.ReaderIOTests.test_version_and_length_header_mismatch_close_without_retry', 'tests_semantics.CanonicalPipeTests.test_changed_native_bytes_refused_before_subprocess', 'tests_semantics.CanonicalPipeTests.test_exact_canonical_only_argv_and_json_values', 'tests_semantics.CanonicalPipeTests.test_stdout_stderr_caps_and_nonzero_exit_fail', 'tests_semantics.SemanticTests.test_authority_never_accepts_an_unconfigured_echo', 'tests_semantics.SemanticTests.test_changed_context_or_program_cannot_be_verified', 'tests_semantics.SemanticTests.test_configuration_exact_source_normalizer_and_clock_are_fixed', 'tests_semantics.SemanticTests.test_encoder_weight_is_exact_not_any_hex', 'tests_semantics.SemanticTests.test_exact_local_file_refuses_symlink_and_changed_hash', 'tests_semantics.SemanticTests.test_missing_wrong_version_and_poisoned_bodies_refused', 'tests_semantics.SemanticTests.test_only_top_level_no_closure_or_default_callbacks', 'tests_semantics.SemanticTests.test_production_config_wrong_canonicalizer_refused_without_execution', 'tests_semantics.SemanticTests.test_real_gate_recomputes_two_folds_through_seven_body_store', 'tests_semantics.SemanticTests.test_replay_descriptor_rejects_parent_role_context_and_receipt_drift', 'tests_semantics.SemanticTests.test_score_row_value_forgery_refused', 'tests_semantics.SemanticTests.test_supplied_extra_artifact_ref_cannot_be_read', 'tests_semantics.SemanticTests.test_unknown_or_unpassed_parity_is_nonnumeric', 'tests_semantics.SemanticTests.test_valid_hex_wrong_weight_license_is_refused_before_file_read')

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
    for module in sorted({x.split('.')[0]for x in EXPECTED}):
        suite.addTests(unittest.defaultTestLoader.loadTestsFromName(name+'.'+module))
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
    receipt={'schema':'jope.generated-target-callbacks-offline-qualification/v1','passed':passed,'tests_run':result.testsRun,
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
