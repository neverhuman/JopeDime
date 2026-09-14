import copy
import unittest

import verify_manifest_v10 as v10


def manifest():
    return dict(schema='parallel.continuous_learning.v1', parent_manifest=dict(path='x', sha256='4' * 64), automatic_training=False, preregistration_sha256='p' * 64,
                preregistration_amendments=[dict(path='PROPOSAL-RNG.md', sha256='a' * 64), dict(path='ACTIVATION-AMENDMENT.md', sha256='b' * 64)],
                proposal_generation_protocol='generation_seeded_v1', neural_executable='bin/propose', gp_service_executable='bin/gp', bindings={'bin/propose': 'n' * 64, 'bin/gp': 'g' * 64},
                gp_dispatch_policy=dict(ordinary_order='oldest_generation_first_v1'), training_growth_gate=dict(minimum_new_complete_generations=1, minimum_new_verified_searches=300),
                training_diagnostic_guard=dict(consecutive_rejections=3), population_arms=dict(geometry='sequential_world8', k=1), release_revision='v10')


PRE = dict(sha256='p' * 64, block_bytes_sha256='q' * 64, digest_rule='canonical json')
AMEND = {'PROPOSAL-RNG.md': 'a' * 64, 'ACTIVATION-AMENDMENT.md': 'b' * 64}


class V10(unittest.TestCase):
    def test_complete_manifest_verifies_and_pending_is_never_verified(self):
        r = v10.verify(manifest(), PRE, AMEND, '4' * 64, 'n' * 64, 'g' * 64)
        self.assertEqual((r['verdict'], r['failing'], r['pending']), ('verified', [], []))
        r = v10.verify(manifest(), PRE, AMEND, None, None, None)   # missing expectations are pending, not passes
        self.assertEqual((r['verdict'], sorted(r['pending'])), ('not_verified', ['gp_binding', 'neural_binding', 'parent_chain']))

    def test_each_rule_fails_by_name(self):
        base = manifest()
        for mutate, name in ((lambda m: m.update(automatic_training=True), 'training_disabled'), (lambda m: m.update(preregistration_sha256='z' * 64), 'preregistration_digest'),
                             (lambda m: m.update(preregistration_amendments=[dict(sha256='a' * 64)]), 'amendments_bound'), (lambda m: m.update(proposal_generation_protocol='constant'), 'proposal_protocol'),
                             (lambda m: m['bindings'].update({'bin/propose': 'x' * 64}), 'neural_binding'), (lambda m: m['gp_dispatch_policy'].update(ordinary_order='first_pending'), 'dispatch_order'),
                             (lambda m: m['training_growth_gate'].update(minimum_new_verified_searches=400), 'growth_gate_v2'), (lambda m: m['training_diagnostic_guard'].update(consecutive_rejections=2), 'diagnostic_hold_3'),
                             (lambda m: m['population_arms'].update(geometry='parallel_world4'), 'population_geometry'), (lambda m: m.update(parent_manifest='0' * 64), 'parent_chain')):
            m = copy.deepcopy(base)
            mutate(m)
            r = v10.verify(m, PRE, AMEND, '4' * 64, 'n' * 64, 'g' * 64)
            self.assertEqual(r['failing'], [name], name)
        # the block-text digest is also accepted for the preregistration
        m = manifest(); m['preregistration_sha256'] = 'q' * 64
        self.assertEqual(v10.verify(m, PRE, AMEND, '4' * 64, 'n' * 64, 'g' * 64)['failing'], [])


if __name__ == '__main__':
    unittest.main()
